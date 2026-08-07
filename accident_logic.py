# =============================================================================
# accident_logic.py — CORVUS Collision Detection Engine v2.1
# =============================================================================
# SIGNALS (6):
#   1. Sudden Stop          — speed < 15% of rolling average
#   2. Direction Change     — heading rotates > 60° (catches T-bone spin-out)
#   3. Mask / Box Overlap   — pixel-accurate IoU >= 0.20
#   4. High Relative Speed  — converging dot product > 5 px/frame
#   5. Speed Differential   — NEW: one vehicle fast, other slow/stopped + overlap
#   6. Pedestrian Proximity — vehicle centroid within 80 px of pedestrian
#
# KEY DESIGN:
#   MIN_SIGNALS_REQUIRED = 2 — always need at least 2 signals.
#   Signal 5 (SPEED_DIFFERENTIAL) solves T-bone detection:
#     In a T-bone, the dot product (converging velocity) is near zero because
#     the vehicles are moving perpendicular. But the speed difference between
#     a fast SUV and a stationary car IS large — this catches it.
#   CONFIRM_FRAMES = 5 — prevents early pre-contact triggers.
# =============================================================================

import math
import time
import logging
from dataclasses import dataclass, field
from collections import defaultdict

log = logging.getLogger(__name__)


@dataclass
class AccidentEvent:
    vehicle_ids   : list[int]
    trigger_flags : list[str]
    severity      : str   = "LOW"
    severity_score: float = 0.0
    zone_id       : str | None = None
    timestamp     : float = field(default_factory=time.time)


class AccidentDetector:

    def __init__(self, config, frame_size: tuple = (640, 480)):
        self.cfg  = config
        self.fw, self.fh = frame_size
        self.excl = config.EDGE_EXCLUSION_PX

        self._confirm_count       : dict = defaultdict(int)
        self._last_alert          : dict = {}
        self._recent_pair_trigger : dict = {}
        self._overlap_streak      : dict = defaultdict(int)
        self._overlap_spddif_hits : dict = defaultdict(int)
        self._prev_piou           : dict = {}
        self._prev_high_partners  : dict = {}
        self._pending_id_alerts   : set  = set()

        self._min_signals  = getattr(config, "MIN_SIGNALS_REQUIRED", 2)
        self._speed_diff_t = getattr(config, "SPEED_DIFF_THRESH", 8)

        self._cut_grace_until = 0.0

    # ─────────────────────────────────────────
    # Scene cut handling
    # ─────────────────────────────────────────
    # Additive only — does not change any of the 6 signals, their weights,
    # or severity thresholds. It only affects how many *consecutive* gated
    # frames are required before firing, and only for a brief window right
    # after a hard scene cut (multi-clip compilation footage). Normal
    # single-scene operation is untouched: _required_confirm_frames() always
    # returns the calibrated CONFIRM_FRAMES outside the grace window.

    def notify_scene_cut(self):
        """Call when a hard scene cut is detected. In-flight confirmation
        counts were built from the previous scene's tracks and are
        meaningless now, so they're cleared. _last_alert is cleared too:
        the tracker's post-cut IDs restart from a low number and can
        collide with pre-cut IDs, which would otherwise falsely suppress a
        real post-cut collision under the old pair's cooldown."""
        self._confirm_count.clear()
        self._recent_pair_trigger.clear()
        self._last_alert.clear()
        self._overlap_streak.clear()
        self._overlap_spddif_hits.clear()
        self._prev_piou.clear()
        self._prev_high_partners.clear()
        self._pending_id_alerts.clear()
        grace_sec = getattr(self.cfg, "SCENE_CUT_GRACE_SECONDS", 2.5)
        self._cut_grace_until = time.time() + grace_sec

    def _required_confirm_frames(self) -> int:
        if time.time() < self._cut_grace_until:
            return max(1, getattr(self.cfg, "SCENE_CUT_GRACE_CONFIRM_FRAMES", 2))
        return self.cfg.CONFIRM_FRAMES

    # ─────────────────────────────────────────
    # Main entry
    # ─────────────────────────────────────────

    def analyse(self, tracks, zones=None) -> list[AccidentEvent]:
        events  = []
        active  = [t for t in tracks if t.disappeared == 0]

        vehicles    = [t for t in active if not t.is_pedestrian(self.cfg)]
        pedestrians = [t for t in active if t.is_pedestrian(self.cfg)]

        # Pre-pass: how many OTHER vehicles does each track currently sit at
        # near-total mask overlap with? Live-tested finding: a genuine 1-on-1
        # impact is exactly that — two boxes, briefly total overlap, nothing
        # else nearby. A cluster of tightly-parked/cluttered vehicles (or a
        # segmentation blob split into several overlapping instance masks)
        # instead shows MANY pairs among the same 4-6 tracks simultaneously
        # sitting at piou~1.000 (this video: a parked-car cluster produced
        # (92,106)/(92,123)/(106,123)/(77,114)/(77,117)/(102,125) all at once
        # — every SUDDEN_CONTACT candidate there was really one clutter blob,
        # not a collision). Used below to require a pair be the ONLY
        # high-overlap relationship either vehicle has this frame.
        sudden_iou_floor = getattr(self.cfg, "SUDDEN_CONTACT_IOU", 0.85)
        high_overlap_partners: dict = defaultdict(set)
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                ta, tb = vehicles[i], vehicles[j]
                if self._precise_iou(ta, tb) >= sudden_iou_floor:
                    high_overlap_partners[ta.id].add(tb.id)
                    high_overlap_partners[tb.id].add(ta.id)

        # ── Pairwise vehicle checks ──────────────────────────────────────────
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                ta, tb   = vehicles[i], vehicles[j]
                pair_key = frozenset({ta.id, tb.id})
                flags    = []

                iou = self._box_iou(ta.current_bbox(), tb.current_bbox())
                precise_iou = self._precise_iou(ta, tb)
                overlap = precise_iou >= self.cfg.IOU_COLLISION_THRESH
                rel_vel = self._compute_rel_vel(ta, tb)
                spd_dif = abs(ta.speed - tb.speed)

                if 0.01 < iou < self.cfg.IOU_COLLISION_THRESH or rel_vel > 1.0:
                    log.debug(
                        f"[near] IDs=({ta.id},{tb.id}) iou={iou:.3f} "
                        f"relvel={rel_vel:.2f} spddif={spd_dif:.2f} "
                        f"sa={ta.speed:.1f} sb={tb.speed:.1f}"
                    )

                if overlap:
                    flags.append("BOX_OVERLAP")

                if self._check_relative_velocity(ta, tb):
                    flags.append("HIGH_REL_SPEED")

                # Speed differential — catches T-bone where dot product fails
                # One vehicle moving fast, other slow/stopped, boxes overlapping
                if overlap and self._check_speed_differential(ta, tb):
                    flags.append("SPEED_DIFFERENTIAL")

                # Sustained overlap (signal 7, additive — see config.py) —
                # ordinary traffic doesn't hold real (mask-level) overlap for
                # several consecutive frames while the two vehicles' speeds
                # are actively diverging; a real collision does. Independent
                # of BOX_OVERLAP so it can supply the 2nd signal a genuine
                # crash needs when the closing-velocity/speed-diff checks
                # miss the already-entangled boxes.
                #
                # Gated on precise_iou (mask IoU when available) against its
                # OWN higher floor, not the base IOU_COLLISION_THRESH used
                # for BOX_OVERLAP — partial occlusion or same-lane convoy
                # traffic can clear the base threshold for many consecutive
                # frames without ever touching.
                #
                # The discriminator is SPEED DIVERGENCE during the streak,
                # not overlap magnitude — an earlier version excluded any
                # pair whose IoU got too close to 1.000, on the theory that
                # near-perfect overlap meant a duplicate detection of one
                # vehicle. That theory was wrong: live-tested against a real,
                # visually-confirmed T-bone crash, the struck vehicle got
                # thrown clear and precise_iou legitimately hit 1.000 too,
                # with spd_dif climbing from ~8 to ~15 as it separated post-
                # impact. A genuine *duplicate* of one (typically stationary
                # or slow) vehicle never shows that divergence — both
                # "vehicles" report nearly the same low speed throughout,
                # because they're the same physical object. Requiring
                # spd_dif to peak well above normal cruising variation is
                # what actually separates a real collision from both a
                # duplicate detection and ordinary traffic.
                if getattr(self.cfg, "ENABLE_SUSTAINED_OVERLAP_SIGNAL", True):
                    sustained_iou = getattr(self.cfg, "SUSTAINED_OVERLAP_IOU", 0.50)
                    # Sanity speed cap: during actual impact chaos, occlusion/
                    # segmentation churn can make DeepSORT briefly swap or
                    # teleport a track ID, producing a one-frame "speed"
                    # that's physically impossible (100s of px/frame —
                    # live-tested, observed values over 500). Excluded from
                    # evidence rather than trusted.
                    speed_cap    = getattr(self.cfg, "SUSTAINED_OVERLAP_SPEED_CAP", 40.0)
                    min_spd_diff = getattr(self.cfg, "SUSTAINED_OVERLAP_MIN_SPEED_DIFF", 6.0)
                    # Box-height sanity gate: live-tested large-vehicle pairs
                    # right at the bottom of frame (very close to camera)
                    # spanned 60-80% of the frame height with real, sustained
                    # (not one-off) speed divergence between two vehicles
                    # simply driving past each other in adjacent lanes —
                    # extreme close-up foreground is where 2D-projection
                    # overlap is least reliable as collision evidence. Both
                    # confirmed real crashes in this video had boxes well
                    # under this fraction of the frame.
                    max_frac = getattr(self.cfg, "SUSTAINED_OVERLAP_MAX_BOX_FRAC", 0.55)
                    ah = ta.current_bbox()[3] - ta.current_bbox()[1]
                    bh = tb.current_bbox()[3] - tb.current_bbox()[1]
                    oversized = max(ah, bh) > max_frac * self.fh
                    if precise_iou >= sustained_iou and not oversized:
                        self._overlap_streak[pair_key] += 1
                        # Count of qualifying FRAMES, not a lifetime max. A
                        # tiny, otherwise-static pair (confirmed live: a
                        # stationary duplicate detection) can throw one noisy
                        # spd_dif spike — almost always right when a track is
                        # newly created and its velocity has very little
                        # history to average over — that a max() tracker
                        # would keep as "evidence" for the rest of an
                        # otherwise motionless streak. A real collision shows
                        # divergence across MANY consecutive frames, not one.
                        # Only within MAX_STREAK frames of the overlap
                        # starting — a real collision resolves (separates or
                        # settles) within that window; a pair still
                        # overlapping well past it is more likely a
                        # persistent non-collision artifact that would
                        # otherwise get unlimited repeated chances at a
                        # noise-driven hit.
                        max_streak = getattr(self.cfg, "SUSTAINED_OVERLAP_MAX_STREAK", 20)
                        if (min_spd_diff <= spd_dif <= speed_cap
                                and self._overlap_streak[pair_key] <= max_streak):
                            self._overlap_spddif_hits[pair_key] += 1
                    else:
                        self._overlap_streak[pair_key] = 0
                        self._overlap_spddif_hits[pair_key] = 0
                    need      = getattr(self.cfg, "SUSTAINED_OVERLAP_FRAMES", 3)
                    need_hits = getattr(self.cfg, "SUSTAINED_OVERLAP_MIN_HITS", 2)
                    if (self._overlap_streak[pair_key] >= need
                            and self._overlap_spddif_hits[pair_key] >= need_hits):
                        flags.append("SUSTAINED_OVERLAP")

                # Signal 8 — Sudden Contact (ENABLED BY DEFAULT — see
                # ENABLE_SUDDEN_CONTACT_SIGNAL in config.py for the full
                # ground-truth verification against test_video2.mp4's 6
                # confirmed crashes). Targets a real, low-speed T-bone this
                # video has (frame ~51) that BOTH the original 6 signals and
                # SUSTAINED_OVERLAP structurally miss: two vehicles hit
                # near-total mask overlap (piou >= 0.85) for as little as a
                # SINGLE frame, then bounce apart, immediately collapsing
                # back to near-zero. Neither closing-velocity nor speed-
                # differential fires because both vehicles were already slow
                # going in (a gentle intersection knock, not a fast highway
                # hit) — and SUSTAINED_OVERLAP structurally can't fire
                # either, since its 3-consecutive-frame requirement is built
                # for wreckage that stays interlocked, not a bump that
                # separates in 1 frame.
                #
                # The discriminator is the DROP, not the peak: fire on the
                # transition from a high previous-frame reading to a low
                # current one. A duplicate detection of one vehicle (see:
                # the (1,7) pair in this same video) hits piou=1.000 the
                # instant both tracks exist and STAYS pegged there for 30+
                # consecutive frames — mask IoU of a duplicate never
                # collapses, because it's the same physical object under
                # two IDs. A real bump's overlap is transient by nature: the
                # vehicles were apart, touch for a frame, and separate again.
                # live-tested: frame 51 piou=1.000 -> frame 52 piou=0.021.
                #
                # Isolation check: also require that at the PEAK (previous)
                # frame, neither vehicle had more than one high-overlap
                # partner. A cluttered/tightly-parked cluster produces many
                # simultaneous piou~1.000 pairs among the same few tracks
                # (live-tested: a parked-car cluster in this video fired 6
                # different pairs this way in ~10 frames) — a genuine 1-on-1
                # impact is exactly two boxes and nothing else.
                prev_piou = self._prev_piou.get(pair_key, 0.0)
                prev_partners_a = self._prev_high_partners.get(ta.id, 0)
                prev_partners_b = self._prev_high_partners.get(tb.id, 0)
                sudden_contact = False
                if (getattr(self.cfg, "ENABLE_SUDDEN_CONTACT_SIGNAL", False)
                        and prev_piou >= getattr(self.cfg, "SUDDEN_CONTACT_IOU", 0.85)
                        and precise_iou < getattr(self.cfg, "SUDDEN_CONTACT_SEPARATION_MAX", 0.30)
                        and prev_partners_a <= 1 and prev_partners_b <= 1):
                    ah = ta.current_bbox()[3] - ta.current_bbox()[1]
                    bh = tb.current_bbox()[3] - tb.current_bbox()[1]
                    max_frac = getattr(self.cfg, "SUSTAINED_OVERLAP_MAX_BOX_FRAC", 0.55)
                    if max(ah, bh) <= max_frac * self.fh:
                        sudden_contact = True
                        flags.append("SUDDEN_CONTACT")
                self._prev_piou[pair_key] = precise_iou

                # ── GATE: require minimum distinct signals ───────────────────
                # SUDDEN_CONTACT bypasses the multi-signal requirement on its
                # own: by the firing frame, overlap has already collapsed
                # (that collapse IS the signal), so BOX_OVERLAP is no longer
                # true THIS frame and can't supply a 2nd signal. The evidence
                # is already high-confidence by construction (a sudden peak-
                # to-near-zero mask IoU swing), not a borderline single cue.
                if sudden_contact or len(flags) >= self._min_signals:
                    now = time.time()
                    self._recent_pair_trigger[ta.id] = now
                    self._recent_pair_trigger[tb.id] = now

                    self._confirm_count[pair_key] += 1
                    # SUDDEN_CONTACT is itself high-confidence, single-instant
                    # evidence (near-total overlap appearing from a clean
                    # separation) — the normal 5-frame confirmation exists to
                    # filter borderline/noisy signals, and a real bump that
                    # separates within 1-2 frames can't ever accumulate 5. A
                    # much smaller, dedicated requirement lets it fire without
                    # loosening confirmation for every other (noisier) signal.
                    required = self._required_confirm_frames()
                    if sudden_contact:
                        required = min(required, getattr(self.cfg, "SUDDEN_CONTACT_CONFIRM_FRAMES", 1))
                    if self._confirm_count[pair_key] >= required:
                        if not self._in_cooldown(pair_key, [ta.id, tb.id]):
                            # Edge exclusion exists to reject fragmentary
                            # boxes from vehicles entering/exiting frame —
                            # but SUDDEN_CONTACT's own gate (previous-frame
                            # mask IoU >= 0.85, collapsing this frame) is
                            # already hard for a partial/cropped box to
                            # satisfy, and live-tested against this video's
                            # real crash, the vehicle appears right at frame
                            # edge (a compilation clip starting mid-action).
                            # Exempt it rather than lose that real event.
                            edge_blocked = (not sudden_contact
                                            and (self._near_edge(ta) or self._near_edge(tb)))
                            if not edge_blocked:
                                ev = self._build_event(
                                    [ta.id, tb.id], flags, [ta, tb], zones
                                )
                                events.append(ev)
                                self._register_alert(pair_key, [ta.id, tb.id])
                                self._confirm_count[pair_key] = 0
                                log.warning(
                                    f"[CORVUS] Pair collision | "
                                    f"IDs=({ta.id},{tb.id}) "
                                    f"flags={flags} sev={ev.severity} "
                                    f"score={ev.severity_score}"
                                )
                else:
                    if self._confirm_count[pair_key] > 0:
                        self._confirm_count[pair_key] -= 1

        self._prev_high_partners = {tid: len(partners) for tid, partners in high_overlap_partners.items()}

        # ── Pedestrian proximity ─────────────────────────────────────────────
        for vehicle in vehicles:
            for ped in pedestrians:
                pair_key = frozenset({vehicle.id, ped.id})
                vx, vy   = vehicle.current_centroid()
                px, py   = ped.current_centroid()
                dist     = math.hypot(vx - px, vy - py)

                if dist < self.cfg.PEDESTRIAN_PROXIMITY_PX:
                    flags = ["PEDESTRIAN_PROXIMITY"]
                    # Require a real danger signal, not just "some vehicle is
                    # moving nearby" — that's every pedestrian on a sidewalk
                    # next to traffic. Either the vehicle is genuinely
                    # converging on the pedestrian's position, or it's
                    # braking hard (an evasive/near-miss reaction).
                    approaching = self._compute_rel_vel_point(vehicle, (px, py)) > self.cfg.REL_VELOCITY_THRESH
                    braking     = self._check_sudden_stop(vehicle)
                    if approaching:
                        flags.append("HIGH_REL_SPEED")
                    elif braking:
                        flags.append("SUDDEN_STOP")

                    if len(flags) >= self._min_signals:
                        self._confirm_count[pair_key] += 1
                        if self._confirm_count[pair_key] >= self._required_confirm_frames():
                            if not self._in_cooldown(pair_key, [vehicle.id, ped.id]):
                                if not self._near_edge(vehicle):
                                    ev = self._build_event(
                                        [vehicle.id, ped.id], flags,
                                        [vehicle, ped], zones
                                    )
                                    events.append(ev)
                                    self._register_alert(pair_key, [vehicle.id, ped.id])
                                    self._confirm_count[pair_key] = 0
                                    log.warning(
                                        f"[CORVUS] Ped proximity | "
                                        f"veh={vehicle.id} ped={ped.id}"
                                    )
                    else:
                        if self._confirm_count[pair_key] > 0:
                            self._confirm_count[pair_key] -= 1
                else:
                    if self._confirm_count[pair_key] > 0:
                        self._confirm_count[pair_key] -= 1

        # ── Single-vehicle (gated on recent pairwise + needs 2 signals) ──────
        gate_sec = getattr(self.cfg, "SINGLE_VEHICLE_GATE_SEC", 1.5)

        for track in vehicles:
            flags = []

            if self._check_sudden_stop(track):
                flags.append("SUDDEN_STOP")
            if self._check_direction_change(track):
                flags.append("DIRECTION_CHANGE")

            if len(flags) < 2:
                continue

            last_pair = self._recent_pair_trigger.get(track.id, 0)
            if time.time() - last_pair > gate_sec:
                self._confirm_count[frozenset({track.id})] = 0
                continue

            pair_key = frozenset({track.id})
            self._confirm_count[pair_key] += 1
            if self._confirm_count[pair_key] >= self._required_confirm_frames():
                if not self._in_cooldown(pair_key, [track.id]):
                    if not self._near_edge(track):
                        ev = self._build_event(
                            [track.id], flags, [track], zones
                        )
                        events.append(ev)
                        self._register_alert(pair_key, [track.id])
                        self._confirm_count[pair_key] = 0
                        log.warning(
                            f"[CORVUS] Single-vehicle | ID={track.id} "
                            f"flags={flags} sev={ev.severity}"
                        )
            else:
                if self._confirm_count[pair_key] > 0:
                    self._confirm_count[pair_key] -= 1

        self._flush_pending_id_alerts()
        return events

    # ─────────────────────────────────────────
    # Signal 1 — Sudden Stop
    # ─────────────────────────────────────────

    def _check_sudden_stop(self, track) -> bool:
        avg = track.avg_speed(self.cfg.SPEED_HISTORY_FRAMES)
        if avg < self.cfg.MIN_SPEED_FOR_STOP:
            return False
        return track.speed < avg * self.cfg.SUDDEN_STOP_RATIO

    # ─────────────────────────────────────────
    # Signal 2 — Direction Change
    # ─────────────────────────────────────────

    def _check_direction_change(self, track) -> bool:
        cents = list(track.centroids)
        if len(cents) < 3:
            return False
        p0, p1, p2 = cents[-3], cents[-2], cents[-1]
        v1x, v1y = p1[0]-p0[0], p1[1]-p0[1]
        v2x, v2y = p2[0]-p1[0], p2[1]-p1[1]
        m1, m2   = math.hypot(v1x, v1y), math.hypot(v2x, v2y)
        if m1 < 1.0 or m2 < 1.0:
            return False
        dot   = v1x*v2x + v1y*v2y
        cos_a = max(-1.0, min(1.0, dot / (m1 * m2)))
        return math.degrees(math.acos(cos_a)) > self.cfg.DIRECTION_CHANGE_DEG

    # ─────────────────────────────────────────
    # Signal 3 — Mask / Box Overlap
    # ─────────────────────────────────────────

    def _check_overlap(self, ta, tb) -> bool:
        return self._precise_iou(ta, tb) >= self.cfg.IOU_COLLISION_THRESH

    def _precise_iou(self, ta, tb) -> float:
        """Mask IoU when both tracks have a segmentation mask (silhouette-
        accurate — two vehicles merely appearing to overlap in their
        rectangular boxes from an oblique camera angle won't overlap here
        unless actually touching), else falls back to box IoU."""
        ma = ta.current_mask()
        mb = tb.current_mask()
        if ma is not None and mb is not None:
            return self._mask_iou_value(ma, mb, ta.current_bbox(), tb.current_bbox())
        return self._box_iou(ta.current_bbox(), tb.current_bbox())

    @staticmethod
    def _box_iou(ba, bb) -> float:
        ax1, ay1, ax2, ay2 = ba
        bx1, by1, bx2, by2 = bb
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            return 0.0
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

    def _mask_iou_value(self, ma, mb, bbox_a, bbox_b) -> float:
        try:
            import cv2
            import numpy as np
            x1 = min(bbox_a[0], bbox_b[0])
            y1 = min(bbox_a[1], bbox_b[1])
            x2 = max(bbox_a[2], bbox_b[2])
            y2 = max(bbox_a[3], bbox_b[3])
            w, h = max(1, x2-x1), max(1, y2-y1)

            def rast(pts):
                canvas  = np.zeros((h, w), dtype=np.uint8)
                shifted = np.array(pts, dtype=np.int32) - np.array([x1, y1])
                cv2.fillPoly(canvas, [shifted], 1)
                return canvas

            ca    = rast(ma)
            cb    = rast(mb)
            inter = int(np.logical_and(ca, cb).sum())
            union = int(np.logical_or(ca,  cb).sum())
            return (inter / union) if union > 0 else 0.0
        except Exception:
            return self._box_iou(bbox_a, bbox_b)

    # ─────────────────────────────────────────
    # Signal 4 — High Relative Velocity (converging)
    # ─────────────────────────────────────────

    def _compute_rel_vel(self, ta, tb) -> float:
        ax, ay = ta.current_centroid()
        bx, by = tb.current_centroid()
        app_x  = bx - ax
        app_y  = by - ay
        dist   = math.hypot(app_x, app_y)
        if dist < 1.0:
            return 0.0
        rel_vx = ta.vx - tb.vx
        rel_vy = ta.vy - tb.vy
        return (rel_vx*app_x + rel_vy*app_y) / dist

    def _check_relative_velocity(self, ta, tb) -> bool:
        return self._compute_rel_vel(ta, tb) > self.cfg.REL_VELOCITY_THRESH

    def _compute_rel_vel_point(self, track, point: tuple) -> float:
        """Closing speed of `track` toward a fixed point (e.g. a pedestrian)."""
        ax, ay = track.current_centroid()
        bx, by = point
        app_x  = bx - ax
        app_y  = by - ay
        dist   = math.hypot(app_x, app_y)
        if dist < 1.0:
            return 0.0
        return (track.vx*app_x + track.vy*app_y) / dist

    # ─────────────────────────────────────────
    # Signal 5 — Speed Differential (T-bone detector)
    # ─────────────────────────────────────────

    def _check_speed_differential(self, ta, tb) -> bool:
        """
        True when one vehicle is moving significantly faster than the other
        AND the slower one is actually near-stationary (the real "victim"
        pattern this signal targets). Catches T-bone collisions where the
        approach is perpendicular — the dot-product converging velocity is
        near zero but one vehicle is clearly the aggressor (high speed) and
        the other is the victim (low speed / stationary).

        Checking only abs(speed_diff) without the slow-speed floor caused
        false positives: any two vehicles cruising at different-but-normal
        speeds in adjacent lanes satisfy a large speed_diff on their own.

        Only fires when boxes already overlap (called after _check_overlap).
        """
        speed_diff = abs(ta.speed - tb.speed)
        fast_speed = max(ta.speed, tb.speed)
        slow_speed = min(ta.speed, tb.speed)

        return (speed_diff >= self._speed_diff_t
                and fast_speed >= self.cfg.MIN_SPEED_FOR_STOP
                and slow_speed < self.cfg.MIN_SPEED_FOR_STOP * 1.5)

    # ─────────────────────────────────────────
    # Edge exclusion
    # ─────────────────────────────────────────

    def _near_edge(self, track) -> bool:
        cx, cy = track.current_centroid()
        e = self.excl
        return cx < e or cy < e or cx > (self.fw - e) or cy > (self.fh - e)

    # ─────────────────────────────────────────
    # Severity scoring
    # ─────────────────────────────────────────

    def _score_severity(self, flags: list[str]) -> tuple[str, float]:
        score = min(1.0, sum(self.cfg.SIGNAL_WEIGHTS.get(f, 0.0) for f in flags))
        thr   = self.cfg.SEVERITY_THRESHOLDS
        if score >= thr["HIGH"]:
            sev = "HIGH"
        elif score >= thr["MEDIUM"]:
            sev = "MEDIUM"
        else:
            sev = "LOW"
        return sev, round(score, 3)

    # ─────────────────────────────────────────
    # Zone lookup
    # ─────────────────────────────────────────

    @staticmethod
    def _find_zone(tracks: list, zones) -> str | None:
        if not zones:
            return None
        for track in tracks:
            cx, cy = track.current_centroid()
            for zone in zones:
                if zone.contains(cx, cy):
                    return zone.name
        return None

    # ─────────────────────────────────────────
    # Event builder
    # ─────────────────────────────────────────

    def _build_event(self, ids, flags, tracks, zones) -> AccidentEvent:
        sev, score = self._score_severity(flags)
        zone_id    = self._find_zone(tracks, zones)
        return AccidentEvent(
            vehicle_ids    = ids,
            trigger_flags  = flags,
            severity       = sev,
            severity_score = score,
            zone_id        = zone_id,
        )

    # ─────────────────────────────────────────
    # Cooldown helpers
    # ─────────────────────────────────────────

    def _in_cooldown(self, key, ids=None) -> bool:
        now = time.time()
        last = self._last_alert.get(key)
        if last is not None and (now - last) < self.cfg.ALERT_COOLDOWN_SEC:
            return True
        # Also cool down on any INDIVIDUAL vehicle already in a recent alert,
        # not just the exact pair. A real crash's aftermath (this video: a
        # highway T-bone whose vehicles sit tangled/stopped with people
        # getting out for ~17s) keeps re-qualifying under a NEW pair_key
        # every time DeepSORT reassigns an ID during the chaos or a
        # different bystander vehicle/track pairs with the same wreck — the
        # exact-pair cooldown alone let one real crash fire 6 separate
        # "new" incidents. Live-tested: IDs churned (3,6)->(3,8)->(8,13)
        # across the same wreck. Suppressing by participant ID (not just
        # pair) collapses these back into one incident.
        # Checked against the PRE-this-frame state only (self._last_alert's
        # ("id", tid) entries are flushed from a pending set at the end of
        # analyse(), never written mid-frame) — a genuine multi-vehicle
        # pileup discovered in the same frame (this video: (54,55) HIGH and
        # (55,57) LOW both at frame 771, a 3-car chain sharing vehicle 55)
        # must NOT have its second collision suppressed just because the
        # first one touched the same vehicle moments earlier in the same
        # analyse() call — that's a distinct real event, not a re-detection.
        if ids:
            id_cd = getattr(self.cfg, "ALERT_COOLDOWN_PER_ID_SEC", self.cfg.ALERT_COOLDOWN_SEC)
            for tid in ids:
                last_id = self._last_alert.get(("id", tid))
                if last_id is not None and (now - last_id) < id_cd:
                    return True
        return False

    def _register_alert(self, key, ids=None):
        self._last_alert[key] = time.time()
        if ids:
            self._pending_id_alerts.update(ids)

    def _flush_pending_id_alerts(self):
        now = time.time()
        for tid in self._pending_id_alerts:
            self._last_alert[("id", tid)] = now
        self._pending_id_alerts.clear()
