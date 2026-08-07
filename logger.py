# =============================================================================
# logger.py — Structured JSON Event Logger
# =============================================================================
# Writes two files per session:
#   logs/events_<timestamp>.json   — one JSON object per line (JSONL)
#   logs/summary_<timestamp>.json  — end-of-session statistics
# =============================================================================

import json
import time
import os
import logging

log = logging.getLogger(__name__)


class EventLogger:
    """
    Writes structured JSONL logs for every accident and zone event.
    Call  log_accident(event, frame_no)  and  log_zone(event, frame_no).
    Call  close(fps, total_frames)  at shutdown to write the summary.
    """

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")

        self._event_path   = os.path.join(log_dir, f"events_{ts}.jsonl")
        self._summary_path = os.path.join(log_dir, f"summary_{ts}.json")
        self._fh           = open(self._event_path, "a", encoding="utf-8")

        # Counters
        self._accident_count        = 0
        self._zone_event_count      = 0
        self._suppressed_cooldown   = 0
        self._suppressed_edge       = 0
        self._severity_counts       = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        self._signal_hits: dict[str, int] = {}
        self._session_start         = time.time()

        print(f"[Logger] ✅ Event log → {self._event_path}")

    # ─────────────────────────────────────────
    # Public logging methods
    # ─────────────────────────────────────────

    def log_accident(self, event, frame_no: int, screenshot_path: str = "") -> None:
        record = {
            "type"           : "ACCIDENT",
            "timestamp"      : time.strftime("%Y-%m-%d %H:%M:%S",
                                             time.localtime(event.timestamp)),
            "epoch"          : round(event.timestamp, 3),
            "frame"          : frame_no,
            "vehicle_ids"    : event.vehicle_ids,
            "trigger_flags"  : event.trigger_flags,
            "severity"       : event.severity,
            "severity_score" : event.severity_score,
            "zone"           : event.zone_id,
            "screenshot"     : screenshot_path,
        }
        self._write(record)
        self._accident_count += 1
        self._severity_counts[event.severity] = \
            self._severity_counts.get(event.severity, 0) + 1
        for flag in event.trigger_flags:
            self._signal_hits[flag] = self._signal_hits.get(flag, 0) + 1

    def log_zone_event(self, zone_event, frame_no: int) -> None:
        record = {
            "type"      : zone_event.event_type,
            "timestamp" : time.strftime("%Y-%m-%d %H:%M:%S",
                                        time.localtime(zone_event.timestamp)),
            "epoch"     : round(zone_event.timestamp, 3),
            "frame"     : frame_no,
            "zone"      : zone_event.zone_name,
            "track_id"  : zone_event.track_id,
        }
        self._write(record)
        self._zone_event_count += 1

    def count_suppressed_cooldown(self): self._suppressed_cooldown += 1
    def count_suppressed_edge(self):     self._suppressed_edge     += 1

    # ─────────────────────────────────────────
    # Session close + summary
    # ─────────────────────────────────────────

    def close(self, avg_fps: float = 0, total_frames: int = 0) -> None:
        duration = round(time.time() - self._session_start, 1)
        summary  = {
            "session_start"       : time.strftime("%Y-%m-%d %H:%M:%S",
                                                  time.localtime(self._session_start)),
            "duration_sec"        : duration,
            "total_frames"        : total_frames,
            "avg_fps"             : round(avg_fps, 2),
            "accident_events"     : self._accident_count,
            "zone_events"         : self._zone_event_count,
            "suppressed_cooldown" : self._suppressed_cooldown,
            "suppressed_edge"     : self._suppressed_edge,
            "severity_breakdown"  : self._severity_counts,
            "signal_contributions": self._signal_hits,
        }
        try:
            with open(self._summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"[Logger] 📊 Summary → {self._summary_path}")
            log.info(f"[Logger] Summary written: {summary}")
        except Exception as e:
            log.error(f"[Logger] Failed to write summary: {e}")
        finally:
            self._fh.close()

    # ─────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────

    def _write(self, record: dict) -> None:
        try:
            self._fh.write(json.dumps(record) + "\n")
            self._fh.flush()
        except Exception as e:
            log.error(f"[Logger] Write error: {e}")
