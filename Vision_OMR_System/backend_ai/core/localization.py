"""
localization.py
===============
YOLOv8 inference wrapper for OMR bubble detection.

Uses the Ultralytics library to run inference against the custom-trained
OMR_best_model.pt weights.  A custom post-processing step applies
class-aware Non-Maximum Suppression (NMS) with tunable IoU / confidence
thresholds so overlapping detections on tightly packed bubble grids are
handled correctly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np

# Lazy import – ultralytics is only required at inference time
try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ultralytics is not installed.  Run: pip install ultralytics"
    ) from exc

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_WEIGHTS_PATH = Path(__file__).parent.parent / "weights" / "OMR_best_model.pt"

# ---------------------------------------------------------------------------
# Detection result type
# ---------------------------------------------------------------------------


class BubbleDetection:
    """Represents a single detected bubble region."""

    __slots__ = ("x1", "y1", "x2", "y2", "confidence", "class_id", "class_name")

    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        confidence: float,
        class_id: int,
        class_name: str,
    ) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


# ---------------------------------------------------------------------------
# Model singleton (loaded once per worker process)
# ---------------------------------------------------------------------------

_model: YOLO | None = None


def _get_model(weights_path: str | Path = _WEIGHTS_PATH) -> YOLO:
    global _model
    if _model is None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"YOLOv8 weights not found at: {weights_path}\n"
                "Place OMR_best_model.pt inside backend_ai/weights/"
            )
        _model = YOLO(str(weights_path))
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_yolo_inference(
    image: np.ndarray,
    conf_threshold: float = 0.20,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
    weights_path: str | Path = _WEIGHTS_PATH,
) -> List[BubbleDetection]:
    """
    Run YOLOv8 inference on a preprocessed OMR image.

    Parameters
    ----------
    image          : BGR numpy array (output of preprocess.preprocess_image)
    conf_threshold : Minimum confidence to keep a detection (default 0.35)
    iou_threshold  : IoU threshold for NMS (default 0.45)
    imgsz          : Inference image size fed to YOLO (default 640)
    weights_path   : Override path to .pt weights file

    Returns
    -------
    List[BubbleDetection]
        Filtered, NMS-deduplicated list of detected bubbles.
    """
    model = _get_model(weights_path)

    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False,
    )

    detections: List[BubbleDetection] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, str(cls_id))
            detections.append(
                BubbleDetection(x1, y1, x2, y2, conf, cls_id, cls_name)
            )

    return _custom_nms(detections, iou_threshold)


# ---------------------------------------------------------------------------
# Custom NMS
# ---------------------------------------------------------------------------


def _iou(a: BubbleDetection, b: BubbleDetection) -> float:
    """Intersection-over-Union of two bounding boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _custom_nms(
    detections: List[BubbleDetection], iou_threshold: float
) -> List[BubbleDetection]:
    """
    Class-aware NMS: suppress lower-confidence overlapping boxes
    within the same class only.
    """
    by_class: dict[int, List[BubbleDetection]] = {}
    for det in detections:
        by_class.setdefault(det.class_id, []).append(det)

    kept: List[BubbleDetection] = []
    for cls_dets in by_class.values():
        cls_dets.sort(key=lambda d: d.confidence, reverse=True)
        suppressed = [False] * len(cls_dets)
        for i, det_i in enumerate(cls_dets):
            if suppressed[i]:
                continue
            kept.append(det_i)
            for j in range(i + 1, len(cls_dets)):
                if not suppressed[j] and _iou(det_i, cls_dets[j]) > iou_threshold:
                    suppressed[j] = True
    return kept
