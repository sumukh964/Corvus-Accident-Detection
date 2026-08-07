#!/usr/bin/env python
# =============================================================================
# tensorrt_export.py — Export YOLOv8s-seg to TensorRT FP16
# =============================================================================
# Run ONCE after first setup:
#
#     python tensorrt_export.py
#
# This generates  yolov8s-seg.engine  in the project root.
# On next startup CORVUS auto-detects the .engine file and uses it.
# Typical FPS gain on RTX 4060: ~10 FPS → 30-35 FPS.
#
# Requirements:
#     pip install tensorrt   (or NVIDIA wheel from https://pypi.ngc.nvidia.com)
#     CUDA 12+ toolkit + cuDNN must be installed
# =============================================================================

import sys
import os

MODEL_PT  = "yolov8s-seg.pt"
MODEL_ENG = "yolov8s-seg.engine"
IMG_SIZE  = 640


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[TRT Export] ultralytics not found — run: pip install ultralytics")
        sys.exit(1)

    if os.path.isfile(MODEL_ENG):
        print(f"[TRT Export] {MODEL_ENG} already exists — delete it to re-export.")
        print("[TRT Export] CORVUS will auto-use TensorRT on next start.")
        return

    print(f"[TRT Export] Loading {MODEL_PT} ...")
    try:
        model = YOLO(MODEL_PT)
    except Exception as e:
        print(f"[TRT Export] Failed to load model: {e}")
        sys.exit(1)

    print("[TRT Export] Exporting to TensorRT FP16  (3-10 min on first run) ...")
    try:
        path = model.export(
            format   = "engine",
            imgsz    = IMG_SIZE,
            half     = True,       # FP16 — RTX 4060 handles natively
            device   = 0,          # GPU 0
            dynamic  = False,      # fixed batch=1
            simplify = True,
        )
        print(f"\n[TRT Export] Done → {path}")
        print("\n  CORVUS will auto-detect yolov8s-seg.engine on next launch.")
        print("  No config changes needed — just run:  python main.py\n")
    except Exception as e:
        print(f"\n[TRT Export] Export failed: {e}")
        print("  Possible fix:  pip install tensorrt --extra-index-url "
              "https://pypi.ngc.nvidia.com")
        sys.exit(1)


if __name__ == "__main__":
    main()
