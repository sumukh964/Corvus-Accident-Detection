import sys, time
import cv2
import config
from detector import VehicleDetector
from tracker import CentroidTracker
from accident_logic import AccidentDetector

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "videos/test_video2.mp4"

detector = VehicleDetector(config)
tracker  = CentroidTracker(config)

cap = cv2.VideoCapture(VIDEO)
src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
accident = AccidentDetector(config, frame_size=(src_w, src_h))

events = []
frame_no = 0
t0 = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1
    dets = detector.detect(frame)
    tracks = tracker.update(dets, frame)
    evs = accident.analyse(tracks, None)
    for e in evs:
        events.append((frame_no, e.vehicle_ids, e.trigger_flags, e.severity, e.severity_score))
cap.release()

print(f"\nProcessed {frame_no} frames in {time.time()-t0:.1f}s")
print(f"Total accident events fired: {len(events)}")
for f, ids, flags, sev, score in events:
    print(f"  frame {f:5d} | ids={ids} | {flags} | {sev} ({score})")
