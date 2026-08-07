# =============================================================================
# calibrate_zones.py — Interactive Zone Calibration Tool
# =============================================================================
# Run ONCE to tune zone coordinates to your video:
#   python calibrate_zones.py --video videos/test_video.mp4
#
# Controls:
#   Left-click       → add point to current polygon
#   Right-click      → finish current zone polygon
#   N                → name the zone (prompted in console)
#   S                → save all zones to zones.json and exit
#   R                → reset all zones
#   ESC / Q          → quit without saving
# =============================================================================

import cv2
import json
import argparse
import config

zones_data  = []
current_pts = []
drawing     = False


def mouse_cb(event, x, y, flags, param):
    global current_pts, drawing
    frame_ref = param["frame"]

    if event == cv2.EVENT_LBUTTONDOWN:
        current_pts.append([x, y])
        drawing = True
        cv2.circle(frame_ref, (x, y), 4, (0, 255, 255), -1)
        if len(current_pts) > 1:
            cv2.line(frame_ref, tuple(current_pts[-2]),
                     tuple(current_pts[-1]), (0, 255, 255), 2)
        cv2.imshow("Zone Calibrator", frame_ref)

    elif event == cv2.EVENT_RBUTTONDOWN and current_pts:
        # Close polygon
        cv2.line(frame_ref, tuple(current_pts[-1]),
                 tuple(current_pts[0]), (0, 255, 255), 2)
        cv2.polylines(frame_ref,
                      [__import__("numpy").array(current_pts)],
                      True, (0, 255, 0), 2)
        cv2.imshow("Zone Calibrator", frame_ref)
        drawing = False


def main():
    global current_pts, zones_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=config.VIDEO_PATH)
    parser.add_argument("--out",   default=config.ZONES_FILE)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Cannot open {args.video}")
        return

    import numpy as np
    h, w = frame.shape[:2]
    sw   = int(w * config.DISPLAY_SCALE)
    sh   = int(h * config.DISPLAY_SCALE)
    frame = cv2.resize(frame, (sw, sh))
    canvas = frame.copy()

    param = {"frame": canvas}
    cv2.namedWindow("Zone Calibrator")
    cv2.setMouseCallback("Zone Calibrator", mouse_cb, param)

    print("\n[Calibrator] Left-click = add point | Right-click = close polygon")
    print("[Calibrator] N = name zone | S = save | R = reset | Q = quit\n")
    cv2.putText(canvas, "Left-click: add pt | Right-click: close | N: name | S: save",
                (8, sh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
    cv2.imshow("Zone Calibrator", canvas)

    colors = [(0, 165, 255), (0, 255, 180), (0, 50, 220),
              (255, 140, 0), (190, 0, 255)]
    color_idx = 0

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):
            print("[Calibrator] Quit without saving.")
            break

        elif key == ord('n') and not drawing and current_pts:
            name = input("  Zone name: ").strip() or f"Zone{len(zones_data)+1}"
            spd  = input("  Speed limit (px/frame, 0=none): ").strip()
            loit = input("  Loiter seconds (0=none): ").strip()
            intr = input("  Intrusion zone? (y/n): ").strip().lower() == "y"
            col  = list(colors[color_idx % len(colors)])
            color_idx += 1

            zones_data.append({
                "name"          : name,
                "polygon"       : current_pts[:],
                "speed_limit_px": float(spd)  if spd  else 0,
                "loiter_sec"    : float(loit) if loit else 0,
                "intrusion"     : intr,
                "color"         : col,
            })
            print(f"  ✅ Zone '{name}' saved ({len(current_pts)} points)")
            current_pts = []
            param["frame"] = frame.copy()
            for zd in zones_data:
                pts = np.array(zd["polygon"], dtype=np.int32)
                cv2.polylines(param["frame"], [pts], True,
                              tuple(zd["color"]), 2)
                cx = int(pts[:, 0].mean())
                cy = int(pts[:, 1].mean())
                cv2.putText(param["frame"], zd["name"], (cx-20, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            tuple(zd["color"]), 1)
            cv2.imshow("Zone Calibrator", param["frame"])

        elif key == ord('s'):
            out = {"_note": "Calibrated zones", "zones": zones_data}
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)
            print(f"\n[Calibrator] ✅ {len(zones_data)} zone(s) saved → {args.out}")
            break

        elif key == ord('r'):
            zones_data  = []
            current_pts = []
            param["frame"] = frame.copy()
            cv2.imshow("Zone Calibrator", param["frame"])
            print("[Calibrator] Reset.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
