# =============================================================================
# behavior.py — Vehicle Behavior Classifier
# =============================================================================
import math
import logging

log = logging.getLogger(__name__)

_STATIONARY_SPEED = 1.5   # px/frame rolling avg → STATIONARY
_ERRATIC_CHANGES  = 3     # heading changes in recent window → ERRATIC
_ERRATIC_ANGLE    = 35    # degrees per step counted as a change

BEHAVIOR_COLOR = {
    "NORMAL"    : None,
    "AGGRESSIVE": (  0, 184, 255),
    "ERRATIC"   : ( 26,  26, 255),
    "STATIONARY": (100, 110, 120),
}
BEHAVIOR_TAG = {
    "AGGRESSIVE": "AGG",
    "ERRATIC"   : "ERT",
    "STATIONARY": "STA",
}


class BehaviorClassifier:
    """
    Classifies each Track into one of four categories:
        NORMAL | AGGRESSIVE | ERRATIC | STATIONARY

    Uses rolling speed average and recent heading-change count.
    """

    def classify(self, track, config=None) -> str:
        avg = track.avg_speed(10)

        if avg < _STATIONARY_SPEED:
            return "STATIONARY"

        cents = list(track.centroids)
        if len(cents) < 4:
            return "NORMAL"

        changes = 0
        window  = cents[-10:]
        for i in range(2, len(window)):
            v1 = (window[i-1][0] - window[i-2][0], window[i-1][1] - window[i-2][1])
            v2 = (window[i][0]   - window[i-1][0], window[i][1]   - window[i-1][1])
            m1, m2 = math.hypot(*v1), math.hypot(*v2)
            if m1 > 0.5 and m2 > 0.5:
                dot   = v1[0]*v2[0] + v1[1]*v2[1]
                cos_a = max(-1.0, min(1.0, dot / (m1 * m2)))
                if math.degrees(math.acos(cos_a)) > _ERRATIC_ANGLE:
                    changes += 1

        if changes >= _ERRATIC_CHANGES:
            return "ERRATIC"

        thresh = getattr(config, "AGGRESSIVE_SPEED_THRESH", 22)
        if avg > thresh:
            return "AGGRESSIVE"

        return "NORMAL"
