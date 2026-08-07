# =============================================================================
# detector.py — Vehicle + Pedestrian Detection
# =============================================================================
# Default:   YOLOv8n-seg  →  pixel-accurate segmentation masks
# Fallback:  YOLOv8n      →  bounding boxes  (--box-only / --edge flags)
# Both backends return a unified Detection namedtuple.
# =============================================================================

from collections import namedtuple
import numpy as np
import logging

log = logging.getLogger(__name__)

# ── Unified detection result ─────────────────────────────────────────────────
# mask_contour: list of (x,y) points defining the object outline, or None
Detection = namedtuple("Detection", ["bbox", "label", "confidence",
                                     "class_id", "mask_contour"])


class VehicleDetector:
    """
    Detects vehicles and pedestrians using YOLOv8 (seg or box).

    Call  detect(frame)  every frame → list[Detection]
    """

    def __init__(self, config):
        self.cfg      = config
        self.seg_mode = not config.USE_BOX_ONLY and not config.EDGE_MODE

        if config.USE_ROBOFLOW:
            self._init_roboflow()
        elif self.seg_mode:
            self._init_yolo_seg()
        else:
            self._init_yolo_box()

    # ─────────────────────────────────────────
    # Initialisers
    # ─────────────────────────────────────────

    def _init_yolo_seg(self):
        from ultralytics import YOLO
        self.backend    = "yolo_seg"
        self.yolo_model = YOLO(self.cfg.YOLO_SEG_MODEL_PATH)
        log.info(f"[Detector] YOLOv8-seg loaded: {self.cfg.YOLO_SEG_MODEL_PATH}")
        print(f"[Detector] ✅ Segmentation model: {self.cfg.YOLO_SEG_MODEL_PATH}")

    def _init_yolo_box(self):
        from ultralytics import YOLO
        self.backend    = "yolo_box"
        self.yolo_model = YOLO(self.cfg.YOLO_BOX_MODEL_PATH)
        log.info(f"[Detector] YOLOv8-box loaded: {self.cfg.YOLO_BOX_MODEL_PATH}")
        print(f"[Detector] ✅ Box-only model: {self.cfg.YOLO_BOX_MODEL_PATH}")

    def _init_roboflow(self):
        try:
            from inference_sdk import InferenceHTTPClient
            self.backend   = "roboflow"
            self.rf_client = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=self.cfg.ROBOFLOW_API_KEY,
            )
            self.rf_model_id = self.cfg.ROBOFLOW_MODEL_ID
            # Keep local seg model as fallback
            self._init_yolo_seg()
            print("[Detector] ✅ Roboflow InferenceSDK ready.")
        except ImportError:
            log.warning("[Detector] inference_sdk not found — using YOLOv8.")
            self.cfg.USE_ROBOFLOW = False
            self._init_yolo_seg()

    # ─────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list:
        if self.backend == "roboflow":
            try:
                return self._detect_roboflow(frame)
            except Exception as e:
                log.warning(f"[Detector] Roboflow error ({e}) — fallback to YOLO.")
                return self._detect_yolo(frame)
        else:
            return self._detect_yolo(frame)

    # ─────────────────────────────────────────
    # YOLOv8 backend (seg + box)
    # ─────────────────────────────────────────

    def _detect_yolo(self, frame: np.ndarray) -> list:
        results = self.yolo_model(frame, conf=self.cfg.YOLO_CONF, iou=0.45, verbose=False)[0]
        detections = []
        h, w = frame.shape[:2]

        # Accepted class IDs: vehicles + pedestrian
        accepted = set(self.cfg.VEHICLE_CLASSES.keys()) | {self.cfg.PEDESTRIAN_CLASS_ID}

        for idx, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            if cls_id not in accepted:
                continue

            conf  = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if cls_id == self.cfg.PEDESTRIAN_CLASS_ID:
                label = self.cfg.PEDESTRIAN_LABEL
            else:
                label = self.cfg.VEHICLE_CLASSES[cls_id]

            # Segmentation mask contour (only in seg mode)
            mask_contour = None
            if self.seg_mode and results.masks is not None:
                try:
                    xy = results.masks.xy[idx]
                    if xy is not None and len(xy) > 2:
                        mask_contour = xy.astype(int).tolist()
                except (IndexError, AttributeError):
                    pass

            detections.append(Detection(
                bbox         = (x1, y1, x2, y2),
                label        = label,
                confidence   = conf,
                class_id     = cls_id,
                mask_contour = mask_contour,
            ))

        return detections

    # ─────────────────────────────────────────
    # Roboflow backend
    # ─────────────────────────────────────────

    def _detect_roboflow(self, frame: np.ndarray) -> list:
        import cv2, tempfile, os as _os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        cv2.imwrite(tmp_path, frame)
        try:
            result = self.rf_client.infer(tmp_path, model_id=self.rf_model_id)
        finally:
            _os.unlink(tmp_path)

        detections = []
        h, w = frame.shape[:2]
        for pred in result.get("predictions", []):
            conf = float(pred.get("confidence", 0))
            if conf < self.cfg.ROBOFLOW_CONF:
                continue
            cx, cy = pred["x"], pred["y"]
            bw, bh = pred["width"], pred["height"]
            x1 = max(0, int(cx - bw / 2))
            y1 = max(0, int(cy - bh / 2))
            x2 = min(w, int(cx + bw / 2))
            y2 = min(h, int(cy + bh / 2))
            label = pred.get("class", "vehicle").lower()
            detections.append(Detection(
                bbox=(x1, y1, x2, y2), label=label,
                confidence=conf, class_id=-1, mask_contour=None,
            ))
        return detections

    @property
    def uses_segmentation(self) -> bool:
        return self.seg_mode
