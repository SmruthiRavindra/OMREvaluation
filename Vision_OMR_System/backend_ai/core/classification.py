"""
classification.py
=================
Micro-classifier / spot-checker for individual OMR bubbles.

After YOLO localizes the bubble bounding boxes, this module crops each
bubble ROI and performs a lightweight secondary classification to determine
whether the bubble is:

  - "filled"   (student marked it)
  - "empty"    (no mark)
  - "ambiguous" (partial / smudged mark — triggers manual review flag)

The classifier uses a simple OpenCV + threshold-based approach by default
(no extra ML model needed).  If you have a dedicated Keras/ONNX
micro-classifier, swap it in via `set_micro_model()`.
"""

from __future__ import annotations

from enum import Enum
from typing import List

import cv2
import numpy as np

from .localization import BubbleDetection


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BubbleState(str, Enum):
    FILLED = "filled"
    EMPTY = "empty"
    AMBIGUOUS = "ambiguous"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class ClassificationResult:
    """Augments a BubbleDetection with its fill state."""

    __slots__ = ("detection", "state", "fill_ratio", "needs_review")

    def __init__(
        self,
        detection: BubbleDetection,
        state: BubbleState,
        fill_ratio: float,
    ) -> None:
        self.detection = detection
        self.state = state
        self.fill_ratio = fill_ratio
        self.needs_review = state == BubbleState.AMBIGUOUS

    def to_dict(self) -> dict:
        return {
            **self.detection.to_dict(),
            "state": self.state.value,
            "fill_ratio": round(self.fill_ratio, 4),
            "needs_review": self.needs_review,
        }


# ---------------------------------------------------------------------------
# Thresholds (tunable)
# ---------------------------------------------------------------------------

# NOTE: OMR bubbles have a printed circle border that contributes ~40%
# dark pixels even when the bubble is completely empty.  We crop to the
# inner region and apply a circular mask so only the interior is measured.
#
# If dark pixel ratio inside the masked inner ROI exceeds FILL_THRESHOLD → FILLED
FILL_THRESHOLD: float = 0.55
# Below EMPTY_THRESHOLD → EMPTY  |  between → AMBIGUOUS
EMPTY_THRESHOLD: float = 0.45


# ---------------------------------------------------------------------------
# Optional micro-model slot
# ---------------------------------------------------------------------------

_micro_model = None  # Keras / ONNX model, if provided


def set_micro_model(model) -> None:
    """
    Register an external micro-classifier (e.g., a Keras or ONNX model).

    The model must accept a (1, 64, 64, 3) float32 array and return a
    probability score in [0, 1] where 1 = filled.
    """
    global _micro_model
    _micro_model = model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_bubble(
    image: np.ndarray,
    detection: BubbleDetection,
) -> ClassificationResult:
    """
    Classify a single detected bubble as filled / empty / ambiguous.

    Parameters
    ----------
    image     : Full BGR image (perspective-corrected)
    detection : A BubbleDetection from localization.run_yolo_inference

    Returns
    -------
    ClassificationResult
    """
    roi = _crop_roi(image, detection)

    if _micro_model is not None:
        return _classify_with_model(roi, detection)
    return _classify_with_threshold(roi, detection)


def classify_all(
    image: np.ndarray,
    detections: List[BubbleDetection],
) -> List[ClassificationResult]:
    """Batch-classify all detected bubbles."""
    return [classify_bubble(image, det) for det in detections]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _crop_roi(image: np.ndarray, det: BubbleDetection) -> np.ndarray:
    """Crop and resize the bubble region to 64×64 for classification."""
    h, w = image.shape[:2]
    x1 = max(0, int(det.x1))
    y1 = max(0, int(det.y1))
    x2 = min(w, int(det.x2))
    y2 = min(h, int(det.y2))
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    return cv2.resize(roi, (64, 64))


def _extract_inner_region(roi: np.ndarray, shrink: float = 0.35) -> tuple:
    """
    Extract the inner circular region of a bubble, excluding the
    printed border ring.

    Parameters
    ----------
    roi     : 64×64 BGR image of the bubble
    shrink  : fraction to shrink inward from each edge (0.35 = keep
              the central 30% diameter circle).  The printed border
              ring typically occupies the outer ~30-35% on each side.

    Returns
    -------
    (gray_inner, mask) where gray_inner is the masked grayscale image
    and mask is the circular binary mask (255 = inside, 0 = outside).
    """
    size = roi.shape[0]  # 64
    center = size // 2
    # Inner radius: keep central portion, skip printed border
    inner_radius = int(center * (1.0 - shrink))

    # Create circular mask
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (center, center), inner_radius, 255, -1)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return gray, mask


def _classify_with_threshold(
    roi: np.ndarray, detection: BubbleDetection
) -> ClassificationResult:
    """
    Threshold-based classifier with inner-region masking.

    Strategy:
      1. Extract the inner circular region (excludes printed border)
      2. Apply Otsu binarization on the inner region
      3. Compute fill ratio = dark pixels / total pixels inside mask
      4. Compare against tuned thresholds

    This avoids the printed circle border inflating the fill ratio
    and causing empty bubbles to be classified as filled.
    """
    gray, mask = _extract_inner_region(roi)

    # Otsu binarization (inverted: dark ink → white)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Only count pixels inside the circular mask
    masked_binary = cv2.bitwise_and(binary, mask)
    mask_pixel_count = np.count_nonzero(mask)

    if mask_pixel_count == 0:
        return ClassificationResult(detection, BubbleState.EMPTY, 0.0)

    fill_ratio = float(np.count_nonzero(masked_binary)) / mask_pixel_count

    if fill_ratio >= FILL_THRESHOLD:
        state = BubbleState.FILLED
    elif fill_ratio <= EMPTY_THRESHOLD:
        state = BubbleState.EMPTY
    else:
        state = BubbleState.AMBIGUOUS

    return ClassificationResult(detection, state, fill_ratio)


def _classify_with_model(
    roi: np.ndarray, detection: BubbleDetection
) -> ClassificationResult:
    """Run inference through the registered micro-model."""
    tensor = roi.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)  # (1, 64, 64, 3)
    score = float(_micro_model.predict(tensor)[0][0])

    if score >= FILL_THRESHOLD:
        state = BubbleState.FILLED
    elif score <= EMPTY_THRESHOLD:
        state = BubbleState.EMPTY
    else:
        state = BubbleState.AMBIGUOUS

    return ClassificationResult(detection, state, score)
