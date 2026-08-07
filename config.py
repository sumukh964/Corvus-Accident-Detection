# =============================================================================
# config.py — CORVUS v2  Central Configuration
# =============================================================================
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# TELEGRAM  (set in .env)
# ─────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")

# ─────────────────────────────────────────────
# DETECTION BACKEND
# ─────────────────────────────────────────────
USE_ROBOFLOW       = False
ROBOFLOW_API_KEY   = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = "erlinn"
ROBOFLOW_MODEL_ID  = "traffic-accident-detection-qd0vf/1"
ROBOFLOW_CONF      = 0.40

# Auto-use TensorRT engine if it exists AND tensorrt is importable
_seg_engine = "yolov8s-seg.engine"
def _trt_available():
    if not os.path.isfile(_seg_engine): return False
    try: import tensorrt; return True
    except ImportError: return False
USE_TENSORRT        = _trt_available()
YOLO_SEG_MODEL_PATH = _seg_engine if USE_TENSORRT else "yolov8s-seg.pt"
YOLO_BOX_MODEL_PATH = "yolov8n.pt"
YOLO_CONF           = 0.38

USE_BOX_ONLY = False

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

PEDESTRIAN_CLASS_ID = 0
PEDESTRIAN_LABEL    = "person"

# ─────────────────────────────────────────────
# VIDEO
# ─────────────────────────────────────────────
VIDEO_PATH    = "videos/test_video.mp4"
DISPLAY_SCALE = 0.75
TARGET_FPS    = 25

# ─────────────────────────────────────────────
# TRACKER
# ─────────────────────────────────────────────
MAX_DISAPPEARED = 20
MAX_DISTANCE    = 160
HISTORY_LEN     = 30

USE_DEEPSORT        = True
DEEPSORT_MAX_AGE    = 30
DEEPSORT_N_INIT     = 3
DEEPSORT_MAX_COSINE = 0.4
DEEPSORT_SKIP_FRAMES = 5

# ─────────────────────────────────────────────
# ACCIDENT DETECTION THRESHOLDS
# ─────────────────────────────────────────────

# Signal 1 — Sudden Stop
SPEED_HISTORY_FRAMES = 8
SUDDEN_STOP_RATIO    = 0.15
MIN_SPEED_FOR_STOP   = 3.0   # lowered: intersection vehicles move 2-4 px/frame

# Signal 2 — IoU / Mask Overlap
# Require substantial real overlap — 0.08 was catching adjacent-lane vehicles
# whose 2D boxes graze each other from oblique/overhead camera angles.
IOU_COLLISION_THRESH = 0.22

# Signal 3 — Direction Change
DIRECTION_CHANGE_DEG = 45   # catches spin-outs and swerves more readily

# Signal 4 — Relative Velocity (converging dot product)
# Must be genuine convergence, not incidental relative motion between
# same-direction vehicles in different lanes.
REL_VELOCITY_THRESH  = 6

# Signal 5 — Speed Differential (T-bone detector)
# If one vehicle is fast and the other is slow/stopped AND boxes overlap
# this fires as the second signal regardless of approach angle.
# The "slow/stopped" half must also be enforced in code (see
# _check_speed_differential) — otherwise any two vehicles at merely
# different cruising speeds in adjacent lanes trigger this.
SPEED_DIFF_THRESH    = 8    # px/frame difference between the two vehicles

# Signal 6 — Pedestrian Proximity
# Tightened from 80px: that caught ordinary sidewalk pedestrians as traffic
# passed by. 45px + a real approach/braking signal (see analyse()) targets
# genuine near-miss encroachment instead.
PEDESTRIAN_PROXIMITY_PX = 45

# Edge exclusion
EDGE_EXCLUSION_PX = 40

# Require 2 signals minimum — prevents single-signal false positives
# BOX_OVERLAP alone never fires. Must be paired with velocity OR speed diff.
MIN_SIGNALS_REQUIRED = 2

# Signal 7 — Sustained Overlap (additive, does not touch signals 1-6 above).
# DISABLED BY DEFAULT (see ENABLE_ flag below): after 5 rounds of live-
# testing against this video, each fixing one false-positive pattern while
# exposing a new one (parked cars -> same-lane convoys -> excluding a real
# crash via a wrong duplicate-detection ceiling -> single noisy-spike
# evidence -> oversized close-camera boxes), a full-video run still produced
# 11 detections beyond the 2 known-solid ones, and a spot check found at
# least one (ordinary nighttime highway traffic, no crash) to be a false
# positive. The remaining ~9 were never individually verified. Given that,
# the code is kept (it does correctly catch a real early crash this video
# has that the original 6 signals structurally cannot — BOX_OVERLAP alone
# never fires by design) but left OFF by default so the system's default
# behavior stays at zero known false positives. Flip ENABLE_ to True to
# re-enable, ideally after either verifying the remaining detections by
# hand or improving the gates further.
# Diagnosed against test_video2.mp4: a real T-bone crash (frame ~1643-1652)
# hit mask IoU=0.72 and stayed overlapped for 10 straight frames, but
# neither HIGH_REL_SPEED nor SPEED_DIFFERENTIAL ever co-fired — by the time
# the overlap is observed the boxes are already entangled, so the closing-
# velocity math reads near-zero/negative, and speed-differential peaked at
# 5.0 against its 8.0 threshold. BOX_OVERLAP alone never fires (by design),
# so this real collision produced zero detections. A genuine collision
# (crumpled/interlocked vehicles) is the only thing that holds real overlap
# for multiple consecutive frames — so that pattern is treated as
# independent evidence in its own right, on top of (not replacing) the
# original 6 signals.
#
# Four gates, all found necessary by live-testing against this video —
# including reversing an earlier, WRONG version of the 2nd gate:
#  - SUSTAINED_OVERLAP_IOU is its OWN, higher floor on precise (mask) IoU —
#    not the base IOU_COLLISION_THRESH used for BOX_OVERLAP. Two vehicles
#    queued/traveling in the same lane, or one partially occluding another
#    from an oblique elevated camera, can clear the *base* mask-overlap
#    threshold for many consecutive frames without ever touching.
#  - SUSTAINED_OVERLAP_MIN_SPEED_DIFF + SUSTAINED_OVERLAP_MIN_HITS require
#    the two vehicles' speeds to have genuinely, REPEATEDLY diverged
#    (px/frame) during the streak — not just once. An earlier version put a
#    CEILING on precise_iou instead, on the theory that near-perfect
#    (~1.000) mask overlap meant a duplicate detection of one vehicle. That
#    theory was disproven by a real, visually-confirmed crash (frame ~55-60,
#    this same video): the struck vehicle got thrown clear and legitimately
#    hit precise_iou 1.000 too, with spd_dif elevated across many
#    consecutive frames as it separated post-impact — the ceiling was
#    silently excluding genuine crashes. A subsequent version tracked only
#    the single highest spd_dif ever seen during a streak, which turned out
#    to have the opposite problem: a tiny, otherwise completely static
#    duplicate-detection pair could throw ONE noisy spd_dif spike (almost
#    always at track creation, when velocity has very little history to
#    average over) that then counted as permanent "evidence" for the rest
#    of an unmoving streak. Requiring several *separate* qualifying frames
#    (MIN_HITS) fixes that — real divergence is sustained, noise is not.
#  - SUSTAINED_OVERLAP_MAX_BOX_FRAC excludes pairs where either box exceeds
#    this fraction of the frame height. Live-tested: two fast, genuinely
#    independent vehicles passing each other very close to the camera (in
#    adjacent lanes, never touching) produced boxes spanning 60-80% of the
#    frame with real, sustained (not one-off) speed divergence between them
#    — passing the above gates on their own merits. Both confirmed real
#    crashes in this video had boxes well under this fraction; extreme
#    close-up foreground is where 2D-projection overlap is least reliable
#    as collision evidence regardless of how "real" the divergence looks.
ENABLE_SUSTAINED_OVERLAP_SIGNAL  = True
SUSTAINED_OVERLAP_FRAMES         = 3     # consecutive overlapping frames required
SUSTAINED_OVERLAP_IOU            = 0.50  # precise (mask) IoU floor — well above
                                          # IOU_COLLISION_THRESH's 0.22
SUSTAINED_OVERLAP_MIN_SPEED_DIFF = 6.0   # px/frame speed divergence required per hit
SUSTAINED_OVERLAP_MIN_HITS       = 2     # separate qualifying frames required (not a
                                          # single all-time spike). 3 was live-tested
                                          # and found too slow: the real crash's high-
                                          # overlap window had already closed (vehicles
                                          # separating, IoU dropping) by the time a 3rd
                                          # hit would land, leaving too few frames for
                                          # the outer 5-frame confirmation to complete.
                                          # 2 hits lands early enough to leave room.
SUSTAINED_OVERLAP_MAX_STREAK     = 20    # stop granting new hits once the overlap has
                                          # persisted this long without qualifying — a
                                          # collision resolves (separates or comes to
                                          # rest) within this window; anything still
                                          # unresolved past it is more likely a
                                          # persistent non-collision artifact getting
                                          # repeated chances at a noise-driven hit
SUSTAINED_OVERLAP_SPEED_CAP      = 40.0  # ignore single-frame speed-diff readings
                                          # implying >this — a real vehicle can't
                                          # move this fast frame-to-frame; it's a
                                          # tracker ID glitch, seen live during
                                          # actual impact chaos
SUSTAINED_OVERLAP_MAX_BOX_FRAC   = 0.55  # exclude pairs where either box exceeds
                                          # this fraction of the frame height

# Signal 8 — Sudden Contact (additive, does not touch signals 1-7 above).
# ENABLED BY DEFAULT.
#
# What it catches: a real T-bone/collision where mask overlap spikes to
# ~1.000 for a SINGLE frame then collapses back near 0 — either because the
# impact separates the vehicles too fast for a 3-consecutive-frame
# sustained-overlap streak, or because it's a hard, fast T-bone whose
# closing-velocity/speed-differential math goes noisy right at peak contact
# (see accident_logic.py). Fires on the DROP: previous frame's precise IoU
# was >= SUDDEN_CONTACT_IOU, this frame's is < the separation floor, AND
# neither vehicle had more than one high-overlap partner at the peak frame
# (isolation check — rejects tightly-parked/cluttered clusters, see below).
#
# Ground truth for test_video2.mp4, provided by the project owner and
# confirmed frame-by-frame (6 real crashes, all now detected with this
# signal ON): ~t=1.7s (orange sedan T-boned, frame ~52), ~t=13.5s (dark car
# hits white car at intersection, frame ~400-410), ~t=25s (truck plows into
# a parked van/gate — the dust cloud IS impact debris, not ambient haze,
# frame ~750-771), ~t=36-48s (highway T-bone + long aftermath, frame
# ~1012-1528), ~t=55s (Duster clips Lada, glancing hit, frame ~1643-1652),
# ~t=65s (dark crossover T-bones white sedan, frame ~1945-1950).
#
# An earlier pass through this same video mis-judged several of these as
# false positives from single sampled frames (the dust cloud looked like
# ambient haze; the ~13.5s and ~65s intersection hits looked like ordinary
# traffic at a glance) — closer frame-by-frame review corrected against the
# owner's ground truth showed they were all genuine impacts. Only
# lingering/duplicate re-fires during a crash's aftermath (debris settling,
# stopped vehicles before a scene cut) are expected beyond the ~6 real
# events; that duplication is handled by ALERT_COOLDOWN_PER_ID_SEC below,
# not by suppressing the signal itself.
#
# Isolation check (kept from earlier tuning): a tightly-parked/cluttered
# vehicle cluster can produce several simultaneous piou~1.000 pairs among
# the same few tracks that are NOT a collision — this signal requires the
# pair be the only high-overlap relationship either vehicle has at the peak
# frame, which rejects that cluster pattern without rejecting real impacts.
ENABLE_SUDDEN_CONTACT_SIGNAL   = True
SUDDEN_CONTACT_IOU             = 0.85  # precise-IoU floor the PREVIOUS frame had to hit
SUDDEN_CONTACT_SEPARATION_MAX  = 0.30  # precise-IoU ceiling THIS frame (the collapse)
SUDDEN_CONTACT_CONFIRM_FRAMES  = 1     # dedicated (small) confirm requirement —
                                        # this signal is itself high-confidence,
                                        # single-instant evidence; unlike every
                                        # other signal it usually CAN'T recur
                                        # for multiple consecutive frames

# ─────────────────────────────────────────────
# SIGNAL WEIGHTS
# ─────────────────────────────────────────────
SIGNAL_WEIGHTS = {
    "BOX_OVERLAP"          : 0.30,
    "SUDDEN_STOP"          : 0.20,
    "HIGH_REL_SPEED"       : 0.20,
    "SPEED_DIFFERENTIAL"   : 0.15,
    "DIRECTION_CHANGE"     : 0.12,
    "PEDESTRIAN_PROXIMITY" : 0.03,
    "SUSTAINED_OVERLAP"    : 0.20,
    "SUDDEN_CONTACT"       : 0.55,
}

SEVERITY_THRESHOLDS = {
    "HIGH"  : 0.55,
    "MEDIUM": 0.45,
}

# ─────────────────────────────────────────────
# CONFIRMATION & COOLDOWN
# ─────────────────────────────────────────────
# 5 frames at ~10 FPS = 0.5 seconds of sustained signal
# Eliminates single-frame early triggers
CONFIRM_FRAMES     = 5
ALERT_COOLDOWN_SEC = 8

# Per-participant-ID cooldown (see AccidentDetector._in_cooldown) — separate
# from, and longer than, ALERT_COOLDOWN_SEC. Live-tested: a real crash's
# aftermath (stopped/tangled vehicles, people getting out) kept re-qualifying
# under a NEW pair_key for up to ~17s as DeepSORT reassigned IDs during the
# chaos, firing the same wreck as 6 separate "new" incidents. The exact-pair
# cooldown can't catch that (it's never the same pair_key twice); cooling
# down each INVOLVED vehicle ID directly collapses re-pairings of an already-
# alerted vehicle back into one incident, without suppressing an unrelated
# second collision elsewhere in the frame.
ALERT_COOLDOWN_PER_ID_SEC = 20

SINGLE_VEHICLE_GATE_SEC = 1.5

# ─────────────────────────────────────────────
# ZONES
# ─────────────────────────────────────────────
ZONES_FILE = "zones.json"

# ─────────────────────────────────────────────
# EDGE MODE
# ─────────────────────────────────────────────
EDGE_MODE          = False
EDGE_DISPLAY_SCALE = 0.5
EDGE_TARGET_FPS    = 15

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
SCREENSHOTS_DIR = "screenshots_output"
LOG_DIR         = "logs"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR,         exist_ok=True)

# ─────────────────────────────────────────────
# SPEED DISPLAY
# ─────────────────────────────────────────────
# Calibrate: count pixels across a known real-world distance in your video
# e.g. a standard lane is ~3.7 m — measure its pixel width → divide
PIXELS_PER_METER = 8.0

# ─────────────────────────────────────────────
# BEHAVIOR CLASSIFIER
# ─────────────────────────────────────────────
AGGRESSIVE_SPEED_THRESH = 22   # px/frame rolling avg → AGGRESSIVE

# ─────────────────────────────────────────────
# NEAR-MISS DETECTOR
# ─────────────────────────────────────────────
NEAR_MISS_DIST_PX = 80         # centroid distance threshold

# ─────────────────────────────────────────────
# INCIDENT REPLAY BUFFER
# ─────────────────────────────────────────────
REPLAY_DIR          = "incidents"
REPLAY_PRE_SECONDS  = 3.0
REPLAY_POST_SECONDS = 2.0

# ─────────────────────────────────────────────
# AUDIO ALERT  (Windows — winsound.Beep)
# ─────────────────────────────────────────────
AUDIO_ALERT = True

# ─────────────────────────────────────────────
# SCENE CUT DETECTION
# ─────────────────────────────────────────────
# Handles hard cuts in multi-clip footage (e.g. crash-compilation videos)
# where a track's velocity/heading history from the previous scene is
# meaningless for the new one. On a detected cut the tracker and the
# accident-confirmation state are reset, and confirmation briefly requires
# fewer sustained frames so a collision occurring right after a cut isn't
# missed while tracks are still re-establishing themselves. Measured on
# test_video2.mp4: ordinary frame-to-frame motion never exceeds a diff of
# ~8, while real cuts spike to 48-79 — DIFF_THRESH sits with wide margin
# on both sides. Harmless to leave on for a single continuous camera feed
# (a real cut essentially never happens there); set False to disable.
ENABLE_SCENE_CUT_DETECTION     = True
SCENE_CUT_DIFF_THRESH          = 25    # mean abs pixel diff (0-255) on a downsampled frame
SCENE_CUT_MIN_GAP_FRAMES       = 15    # ignore repeat triggers within this many frames
SCENE_CUT_GRACE_SECONDS        = 2.5   # window after a cut with relaxed confirmation
SCENE_CUT_GRACE_CONFIRM_FRAMES = 2     # required confirm frames during the grace window
                                        # (vs the normal CONFIRM_FRAMES)
