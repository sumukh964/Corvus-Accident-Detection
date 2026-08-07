# =============================================================================
# scene_cut.py — Hard Scene-Cut Detector
# =============================================================================
# Flags abrupt frame-to-frame visual jumps (a hard edit cut between clips) as
# distinct from ordinary motion within one continuous shot. Downsamples each
# frame to a tiny grayscale thumbnail and compares mean absolute difference
# against the previous thumbnail's — cheap enough to run every frame even
# alongside the segmentation pipeline.
#
# Calibration note: measured on test_video2.mp4, ordinary motion (fast cars,
# headlights, camera shake) never pushes the diff above ~8; real cuts spike
# to 48-79. The default threshold sits with wide margin on both sides.
# =============================================================================

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

_THUMB_SIZE = (48, 27)   # tiny — just enough to catch a scene change cheaply


class SceneCutDetector:

    def __init__(self, config):
        self.enabled   = getattr(config, "ENABLE_SCENE_CUT_DETECTION", True)
        self.threshold = getattr(config, "SCENE_CUT_DIFF_THRESH", 25)
        self.min_gap   = getattr(config, "SCENE_CUT_MIN_GAP_FRAMES", 15)

        self._prev_thumb     = None
        self._frame_ctr       = 0
        self._last_cut_frame  = -10_000

    def check(self, frame: np.ndarray) -> bool:
        """Call once per video frame, in order. Returns True on a detected cut."""
        self._frame_ctr += 1
        if not self.enabled:
            return False

        thumb = cv2.cvtColor(cv2.resize(frame, _THUMB_SIZE), cv2.COLOR_BGR2GRAY)

        is_cut = False
        if self._prev_thumb is not None:
            diff = float(np.mean(cv2.absdiff(thumb, self._prev_thumb)))
            if (diff > self.threshold
                    and self._frame_ctr - self._last_cut_frame >= self.min_gap):
                is_cut = True
                self._last_cut_frame = self._frame_ctr
                log.info(f"[SceneCut] Detected at frame {self._frame_ctr} (diff={diff:.1f})")

        self._prev_thumb = thumb
        return is_cut
