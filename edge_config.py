# =============================================================================
# edge_config.py — Edge Deployment Configuration & Model Export
# =============================================================================
# Activated by:  python main.py --edge
#
# What it does:
#   1. Auto-detects available hardware (CUDA → OpenVINO → CPU)
#   2. Switches to lighter model settings (lower res, box-only, lower FPS cap)
#   3. Provides  export_openvino()  to quantise YOLOv8 to INT8 for Jetson/Pi
# =============================================================================

import logging
import os

log = logging.getLogger(__name__)


def detect_hardware() -> str:
    """
    Returns the best available compute backend string.
    Priority: CUDA → OpenVINO → CPU
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            log.info(f"[Edge] CUDA detected: {gpu}")
            print(f"[Edge] ✅ CUDA: {gpu}")
            return "cuda"
    except ImportError:
        pass

    try:
        import openvino  # noqa: F401
        log.info("[Edge] OpenVINO detected.")
        print("[Edge] ✅ OpenVINO runtime available.")
        return "openvino"
    except ImportError:
        pass

    log.info("[Edge] Falling back to CPU.")
    print("[Edge] ℹ️  Running on CPU.")
    return "cpu"


def apply_edge_settings(config) -> None:
    """
    Mutate config in-place to use lighter settings for edge deployment.
    Called from main.py when --edge flag is set.
    """
    config.EDGE_MODE      = True
    config.USE_BOX_ONLY   = True          # seg masks too slow on CPU/edge
    config.DISPLAY_SCALE  = config.EDGE_DISPLAY_SCALE
    config.TARGET_FPS     = config.EDGE_TARGET_FPS
    config.USE_DEEPSORT   = False         # DeepSORT embedder requires too much RAM
    config.YOLO_CONF      = 0.45          # higher threshold = fewer detections = faster

    hw = detect_hardware()
    config._hardware = hw

    print(
        f"[Edge] ⚡ Edge mode active | "
        f"box-only | scale={config.DISPLAY_SCALE} | "
        f"FPS cap={config.TARGET_FPS} | hw={hw}"
    )
    log.info(f"[Edge] Edge settings applied. Hardware: {hw}")


def export_openvino(model_path: str = "yolov8n.pt",
                    output_dir: str  = "openvino_model",
                    imgsz: int       = 416) -> str:
    """
    Export a YOLOv8 model to OpenVINO INT8 format.

    Usage (run once before deployment):
        python -c "from edge_config import export_openvino; export_openvino()"

    Parameters
    ----------
    model_path : source .pt file
    output_dir : where to save the OpenVINO IR files
    imgsz      : inference image size (smaller = faster on edge)

    Returns
    -------
    Path to exported model directory
    """
    try:
        from ultralytics import YOLO
        model   = YOLO(model_path)
        out     = model.export(
            format  = "openvino",
            imgsz   = imgsz,
            int8    = True,
            dynamic = False,
        )
        log.info(f"[Edge] OpenVINO export complete → {out}")
        print(f"[Edge] ✅ OpenVINO INT8 model exported → {out}")
        return str(out)
    except Exception as e:
        log.error(f"[Edge] Export failed: {e}")
        print(f"[Edge] ❌ Export failed: {e}")
        return ""


def load_openvino_model(model_dir: str):
    """
    Load an exported OpenVINO model for inference.
    Returns a YOLO model object using the OpenVINO backend.
    """
    try:
        from ultralytics import YOLO
        xml_files = [f for f in os.listdir(model_dir) if f.endswith(".xml")]
        if not xml_files:
            raise FileNotFoundError(f"No .xml found in {model_dir}")
        xml_path = os.path.join(model_dir, xml_files[0])
        model    = YOLO(xml_path)
        log.info(f"[Edge] OpenVINO model loaded from {xml_path}")
        print(f"[Edge] ✅ OpenVINO model: {xml_path}")
        return model
    except Exception as e:
        log.error(f"[Edge] Failed to load OpenVINO model: {e}")
        raise
