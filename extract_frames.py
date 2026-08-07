"""One-off helper: dump specific frame numbers from a video as PNGs for
ground-truth review. Usage: python extract_frames.py <video> <out_dir> <frame1> [frame2 ...]
"""
import sys
import os
import cv2

video_path = sys.argv[1]
out_dir    = sys.argv[2]
frames     = [int(f) for f in sys.argv[3:]]

os.makedirs(out_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

for f in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f - 1))
    ret, frame = cap.read()
    if not ret:
        print(f"frame {f}: read failed")
        continue
    t = f / fps
    mm, ss = int(t // 60), t % 60
    out_path = os.path.join(out_dir, f"frame_{f:05d}_t{mm:02d}m{ss:05.2f}s.png")
    cv2.imwrite(out_path, frame)
    print(f"frame {f} -> {out_path} (t={mm:02d}:{ss:05.2f})")

cap.release()
