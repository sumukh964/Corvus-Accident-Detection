# =============================================================================
# replay.py — Incident Replay Buffer
# =============================================================================
# Keeps a rolling deque of the last PRE_SECONDS of rendered frames.
# When trigger() is called on a confirmed collision event it starts collecting
# POST_SECONDS more frames, then saves the complete clip as incidents/*.mp4.
#
# Usage in main loop (order matters):
#   replay_buf.add_frame(display_frame)   # before accident handling
#   for event in acc_events:
#       replay_buf.trigger(event, config.REPLAY_DIR)
# =============================================================================

import cv2
import os
import time
import logging
from collections import deque

log = logging.getLogger(__name__)


class ReplayBuffer:

    def __init__(self, fps: float = 25.0,
                 pre_seconds: float  = 3.0,
                 post_seconds: float = 2.0):
        self._fps       = max(1.0, fps)
        self._pre_cap   = int(self._fps * pre_seconds)
        self._post_cap  = int(self._fps * post_seconds)
        self._pre       = deque(maxlen=self._pre_cap)
        self._post: list = []
        self._recording = False
        self._pending   = None      # (event, output_dir)
        print(
            f"[Replay] Ready — pre={pre_seconds:.1f}s  post={post_seconds:.1f}s  "
            f"max_clip={pre_seconds+post_seconds:.0f}s"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def add_frame(self, frame) -> None:
        """Call every frame, BEFORE accident handling."""
        if not self._recording:
            self._pre.append(frame.copy())
        else:
            self._post.append(frame.copy())
            if len(self._post) >= self._post_cap:
                self._save()

    def trigger(self, event, output_dir: str) -> None:
        """Start a post-collision recording. No-op if one is already active."""
        if self._recording:
            return
        self._recording = True
        self._post      = []
        self._pending   = (event, output_dir)
        log.info(f"[Replay] Recording triggered — IDs {event.vehicle_ids}")

    def close(self) -> None:
        """Flush any incomplete clip on shutdown."""
        if self._recording and self._post:
            self._save()

    def reset(self) -> None:
        """Drop the buffered pre-roll — call on a detected scene cut so a
        post-cut incident doesn't get a nonsensical pre-roll spliced in
        from the previous (unrelated) scene. Flushes any in-flight
        recording first so it isn't silently lost."""
        if self._recording and self._post:
            self._save()
        self._pre.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _save(self) -> str:
        if not self._pending:
            self._reset()
            return ""

        event, output_dir = self._pending
        frames = list(self._pre) + self._post
        if not frames:
            self._reset()
            return ""

        os.makedirs(output_dir, exist_ok=True)
        ts    = time.strftime("%Y%m%d_%H%M%S")
        ids   = "_".join(str(i) for i in event.vehicle_ids)
        fname = f"replay_{ts}_{ids}_{event.severity}.mp4"
        path  = os.path.join(output_dir, fname)

        h, w  = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self._fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()

        n = len(frames)
        log.info(f"[Replay] Saved {n} frames ({n/self._fps:.1f}s) → {path}")
        print(f"[Replay] Clip → {path}")
        self._reset()
        return path

    def _reset(self):
        self._recording = False
        self._post      = []
        self._pending   = None
