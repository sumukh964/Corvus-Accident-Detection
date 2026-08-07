# =============================================================================
# zones.py — Configurable Polygon Zone Engine
# =============================================================================
import json
import time
import logging
import numpy as np

log = logging.getLogger(__name__)


class ZoneEvent:
    __slots__ = ("event_type", "zone_name", "track_id", "timestamp")

    def __init__(self, event_type: str, zone_name: str, track_id: int):
        self.event_type = event_type
        self.zone_name  = zone_name
        self.track_id   = track_id
        self.timestamp  = time.time()


class Zone:
    """One named polygon zone with associated rules."""

    def __init__(self, name: str, polygon: list,
                 speed_limit_px: float = 0,
                 loiter_sec: float     = 0,
                 intrusion: bool       = False,
                 color: list           = None):
        self.name        = name
        self.polygon     = np.array(polygon, dtype=np.int32)
        self.speed_limit = speed_limit_px
        self.loiter_sec  = loiter_sec
        self.intrusion   = intrusion
        self.color       = tuple(color) if color else (0, 165, 255)

        self._entry_times: dict[int, float] = {}

    def contains(self, x: int, y: int) -> bool:
        import cv2
        result = cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False)
        return result >= 0

    def update(self, tracks) -> list:
        events      = []
        current_ids = set()

        for track in tracks:
            if track.disappeared > 0:
                continue
            cx, cy = track.current_centroid()

            if self.contains(cx, cy):
                current_ids.add(track.id)

                if self.intrusion and track.id not in self._entry_times:
                    events.append(ZoneEvent("ZONE_INTRUSION", self.name, track.id))
                    log.info(f"[Zone] INTRUSION | zone={self.name} id={track.id}")

                if track.id not in self._entry_times:
                    self._entry_times[track.id] = time.time()

                if self.loiter_sec > 0:
                    dwell = time.time() - self._entry_times[track.id]
                    if dwell > self.loiter_sec:
                        events.append(ZoneEvent("ZONE_LOITERING", self.name, track.id))
                        log.info(f"[Zone] LOITERING | zone={self.name} id={track.id} dwell={dwell:.1f}s")
                        self._entry_times[track.id] = time.time()

                if self.speed_limit > 0 and track.speed > self.speed_limit:
                    events.append(ZoneEvent("ZONE_SPEEDING", self.name, track.id))

            else:
                self._entry_times.pop(track.id, None)

        gone = set(self._entry_times) - current_ids
        for tid in gone:
            self._entry_times.pop(tid, None)

        return events


class ZoneManager:
    """Loads zones from JSON and manages per-frame updates."""

    def __init__(self, zones_file: str):
        self.zones: list[Zone] = []
        self._load(zones_file)

    def _load(self, path: str):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            for z in data.get("zones", []):
                self.zones.append(Zone(
                    name           = z["name"],
                    polygon        = z["polygon"],
                    speed_limit_px = z.get("speed_limit_px", 0),
                    loiter_sec     = z.get("loiter_sec", 0),
                    intrusion      = z.get("intrusion", False),
                    color          = z.get("color", [0, 165, 255]),
                ))
            log.info(f"[Zones] Loaded {len(self.zones)} zone(s) from {path}")
            print(f"[Zones] ✅ {len(self.zones)} zone(s) loaded.")
        except FileNotFoundError:
            log.warning(f"[Zones] {path} not found — running without zones.")
            print(f"[Zones] ⚠️  No zones file at {path} — zones disabled.")
        except Exception as e:
            log.error(f"[Zones] Failed to load {path}: {e}")

    def update(self, tracks) -> list:
        events = []
        for zone in self.zones:
            events.extend(zone.update(tracks))
        return events

    def reset(self) -> None:
        """Clear per-zone dwell/entry state — call on a detected scene cut.
        Zone polygons are calibrated for one fixed camera; after a hard cut
        to a different scene, stale entry timestamps from the old view would
        otherwise keep counting toward loitering thresholds against tracks
        that no longer exist."""
        for zone in self.zones:
            zone._entry_times.clear()

    def draw(self, frame) -> None:
        import cv2
        for zone in self.zones:
            pts     = zone.polygon.reshape((-1, 1, 2))
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone.polygon], zone.color)
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
            cv2.polylines(frame, [pts], isClosed=True,
                          color=zone.color, thickness=2, lineType=cv2.LINE_AA)
            cx = int(zone.polygon[:, 0].mean())
            cy = int(zone.polygon[:, 1].mean())
            cv2.putText(frame, zone.name, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, zone.color, 1, cv2.LINE_AA)
