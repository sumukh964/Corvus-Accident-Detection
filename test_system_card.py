"""
Renders a static test frame for eyeballing the sidebar (5-card layout:
SYSTEM / ACTIVE DETECTIONS / THREAT LEVEL / ZONES / INCIDENT LOG).
Run from the CORVUS project root: python test_system_card.py

Note: this intentionally does NOT slice out fixed-pixel per-card crops
anymore — a previous version hardcoded each card's height for cropping,
and it silently went stale (wrong crop boundaries) the moment card layout
changed. Just look at the full sidebar render below instead.
"""
import time
import cv2
import numpy as np

import utils
utils.update_state(
    fps            = 23.4,
    frame_no       = 1042,
    n_vehicles     = 8,
    n_pedestrians  = 1,
    n_tracks       = 9,
    mode           = "SEG",
    use_tensorrt   = True,
    session_start  = time.time() - 312,
    ttc_sec        = 3.1,
    ttc_pair       = (2, 5),
    ttc_delta_kmh  = 42.0,
    risk_history   = [0.1, 0.15, 0.2, 0.25, 0.3, 0.28, 0.22],
    heatmap_on     = False,
    active_zones   = [],
    zone_status    = [
        {"name": "Intersection", "occupied": 2},
        {"name": "Crosswalk",    "occupied": 0},
        {"name": "No-Entry",     "occupied": 0},
    ],
    pixels_per_meter = 10,
    video_fps      = 30.0,
)

canvas = np.zeros((utils.CANVAS_H, utils.CANVAS_W, 3), dtype=np.uint8)
utils.draw_side_panel(canvas)
cv2.imwrite("test_system_card.png", canvas)

sidebar = canvas[:, utils.PANEL_X:utils.CANVAS_W]
cv2.imwrite("test_cards_all.png", sidebar)
print("Written: test_system_card.png (full canvas)")
print("Written: test_cards_all.png (sidebar only)")
