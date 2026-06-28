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

The classifier uses a simple OpenCV + threshold-based approach by default.
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
# Optional micro-model slot
# ---------------------------------------------------------------------------

_micro_model = None  # Keras / ONNX model, if provided


def set_micro_model(model) -> None:
    """Register an external micro-classifier."""
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
    Classify a single detected bubble (fallback/individual call).
    """
    roi = _crop_roi(image, detection)
    gray, mask = _extract_inner_region(roi)
    
    mask_indices = mask > 0
    mean_inner = float(np.mean(gray[mask_indices])) if np.any(mask_indices) else 255.0
    
    # Static fallback thresholds if batch info is missing
    filled_threshold = 100
    empty_threshold = 135
    
    if mean_inner <= filled_threshold:
        state = BubbleState.FILLED
    elif mean_inner >= empty_threshold:
        state = BubbleState.EMPTY
    else:
        state = BubbleState.AMBIGUOUS
        
    # Otsu fill ratio
    mask_pixels = np.count_nonzero(mask)
    _, otsu_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_masked = cv2.bitwise_and(otsu_bin, mask)
    fill_ratio = np.count_nonzero(otsu_masked) / mask_pixels if mask_pixels > 0 else 0.0
    
    return ClassificationResult(detection, state, fill_ratio)


def classify_all(
    image: np.ndarray,
    detections: List[BubbleDetection],
) -> List[ClassificationResult]:
    """
    Batch-classify all detected bubbles using dynamic thresholding based on the 
    overall sheet background paper intensity. This is highly robust to lighting 
    gradients, low-resolution cameras, and heavy shadows.
    """
    if not detections:
        return []

    # First pass: crop and calculate inner mask mean/std/Otsu for each bubble
    bubble_details = []
    means = []
    
    for det in detections:
        roi = _crop_roi(image, det)
        gray, mask = _extract_inner_region(roi)
        
        mask_indices = mask > 0
        mean_inner = float(np.mean(gray[mask_indices])) if np.any(mask_indices) else 255.0
        std_inner = float(np.std(gray[mask_indices])) if np.any(mask_indices) else 0.0
        
        # Otsu fill ratio for fallback/meta compatibility
        mask_pixels = np.count_nonzero(mask)
        _, otsu_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu_masked = cv2.bitwise_and(otsu_bin, mask)
        fill_ratio = np.count_nonzero(otsu_masked) / mask_pixels if mask_pixels > 0 else 0.0
        
        means.append(mean_inner)
        bubble_details.append((det, gray, mask, mean_inner, std_inner, fill_ratio))
        
    # Find the paper reference intensity using the 95th percentile of all bubbles.
    # At least 75% of bubbles on a sheet are always unfilled (empty paper background).
    # Using the 95th percentile guarantees finding a true paper white baseline even under shadows.
    paper_ref = float(np.percentile(means, 95))
    
    # Define relative thresholds from the paper white baseline
    filled_threshold = paper_ref - 40.0
    empty_threshold = paper_ref - 20.0
    
    results = []
    for det, gray, mask, mean_inner, std_inner, fill_ratio in bubble_details:
        if mean_inner <= filled_threshold:
            # Filled bubbles must be solid/uniform (low std deviation). 
            # High standard deviation indicates a mixture of pen and paper (partial fill/ambiguous).
            if std_inner < 18.0:
                state = BubbleState.FILLED
            else:
                state = BubbleState.AMBIGUOUS
        elif mean_inner >= empty_threshold:
            state = BubbleState.EMPTY
        else:
            # Ambiguous zone: verify with YOLO model's confidence
            if det.class_name == "unfilled" and det.confidence > 0.75:
                state = BubbleState.EMPTY
            elif det.class_name == "filled" and det.confidence > 0.80:
                state = BubbleState.FILLED
            else:
                state = BubbleState.AMBIGUOUS
                
        results.append(ClassificationResult(det, state, fill_ratio))
        
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _crop_roi(image: np.ndarray, det: BubbleDetection, pad_factor: float = 0.15) -> np.ndarray:
    """
    Crop and resize the bubble region to 64×64 for classification.
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


def _extract_inner_region(roi: np.ndarray, shrink: float = 0.40) -> tuple:
    """
    Extract the inner circular region of a bubble, excluding the printed border ring.
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

    # Find the contour representing the bubble border (closest to center and of reasonable size)
    best_contour = None
    min_dist = float('inf')
    for c in contours:
        (ccx, ccy), cr = cv2.minEnclosingCircle(c)
        if 12 <= cr <= 30:
            dist = np.sqrt((ccx - center) ** 2 + (ccy - center) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_contour = c

    if best_contour is not None:
        (cx, cy), r = cv2.minEnclosingCircle(best_contour)

    # Create circular mask centered exactly on the detected bubble
    mask = np.zeros((size, size), dtype=np.uint8)
    inner_radius = max(5, int(r * (1.0 - shrink)))
    cv2.circle(mask, (int(cx), int(cy)), inner_radius, 255, -1)

    return gray, mask
