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

# NOTE: OMR bubbles have a printed circle border that we dynamically locate
# and exclude. We crop to the detected bubble's inner region and apply a
# circular mask so only the interior is measured.
#
# If dark pixel ratio inside the masked inner ROI exceeds FILL_THRESHOLD → FILLED
FILL_THRESHOLD: float = 0.30
# Below EMPTY_THRESHOLD → EMPTY  |  between → AMBIGUOUS
EMPTY_THRESHOLD: float = 0.15

# Fixed grayscale threshold to separate dark ink/pencil (letters/marks) from
# the white paper inside the bubble. Values below this are considered ink.
FIXED_GRAY_THRESHOLD: int = 135


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


def _crop_roi(image: np.ndarray, det: BubbleDetection, pad_factor: float = 0.25) -> np.ndarray:
    """
    Crop and resize the bubble region to 64×64 for classification.
    Applies padding on all sides to ensure the printed border of the bubble
    is fully enclosed in the crop and does not touch the edges.
    """
    h, w = image.shape[:2]
    
    # Bounding box dimensions
    bw = det.x2 - det.x1
    bh = det.y2 - det.y1
    
    # Apply padding
    pad_w = int(bw * pad_factor)
    pad_h = int(bh * pad_factor)
    
    x1 = max(0, int(det.x1) - pad_w)
    y1 = max(0, int(det.y1) - pad_h)
    x2 = min(w, int(det.x2) + pad_w)
    y2 = min(h, int(det.y2) + pad_h)
    
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    return cv2.resize(roi, (64, 64))


def _extract_inner_region(roi: np.ndarray, shrink: float = 0.35) -> tuple:
    """
    Extract the inner circular region of a bubble, excluding the
    printed border ring. Uses contour-based dynamic centering and sizing
    to ensure the border is perfectly ignored even if the crop is shifted or resized.

    Parameters
    ----------
    roi     : 64×64 BGR image of the bubble
    shrink  : fraction to shrink inward from the detected bubble radius
              (0.35 = keep the central 65% radius circle).

    Returns
    -------
    (gray, mask) where gray is the grayscale image
    and mask is the circular binary mask (255 = inside, 0 = outside).
    """
    size = roi.shape[0]  # 64
    center = size // 2

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Otsu binarization (inverted: dark ink/border → white) to locate bubble structures
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of the binarized bubble
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Default fallback: center of the image and typical bubble radius
    cx, cy = float(center), float(center)
    r = float(center * 0.7)  # e.g., 22.4 pixels for a 64x64 ROI

    # Find the contour that represents the bubble border (closest to the center and of reasonable size)
    best_contour = None
    min_dist = float('inf')
    for c in contours:
        (ccx, ccy), cr = cv2.minEnclosingCircle(c)
        # OMR bubbles in a 64x64 crop typically have a radius between 12 and 30 pixels
        if 12 <= cr <= 30:
            dist = np.sqrt((ccx - center) ** 2 + (ccy - center) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_contour = c

    if best_contour is not None:
        (cx, cy), r = cv2.minEnclosingCircle(best_contour)

    # Create circular mask centered exactly on the detected bubble, shrunk to exclude the border
    mask = np.zeros((size, size), dtype=np.uint8)
    inner_radius = max(5, int(r * (1.0 - shrink)))
    cv2.circle(mask, (int(cx), int(cy)), inner_radius, 255, -1)

    return gray, mask


def _classify_with_threshold(
    roi: np.ndarray, detection: BubbleDetection
) -> ClassificationResult:
    """
    Threshold-based classifier with inner-region masking.

    Strategy:
      1. Extract the inner circular region (excludes printed border)
      2. Apply a robust fixed threshold of 135 to classify dark ink/marks
      3. Compute fill ratio = dark pixels / total pixels inside mask
      4. Compare against tuned thresholds
    """
    gray, mask = _extract_inner_region(roi)

    # Fixed thresholding (inverted: dark ink/pencil → white)
    _, binary = cv2.threshold(gray, FIXED_GRAY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

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
