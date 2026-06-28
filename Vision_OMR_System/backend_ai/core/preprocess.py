"""
preprocess.py
=============
OpenCV-based image pre-processing pipeline for OMR sheet normalization.

Steps:
  1. Upscale small mobile images to a minimum working resolution
  2. CLAHE contrast enhancement for uneven mobile lighting
  3. Bilateral filter – noise reduction preserving edges
  4. Adaptive threshold / Canny edge detection
  5. Homography (perspective warp) – corrects camera tilt / skew
"""

import cv2
import numpy as np
from typing import Tuple


# Minimum width we want to work with for reliable bubble detection
MIN_WIDTH = 800

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Full pre-processing pipeline: decode → upscale → enhance → denoise → warp.

    Parameters
    ----------
    image_bytes : bytes
        Raw image bytes (JPEG / PNG) from the mobile client.

    Returns
    -------
    np.ndarray
        Perspective-corrected, denoised BGR image ready for YOLO inference.
    """
    img = _decode(image_bytes)
    img = _ensure_min_resolution(img)
    img = _enhance_contrast(img)
    img = _bilateral_filter(img)
    img = _perspective_warp(img)
    return img


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes to an OpenCV BGR image."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes. Ensure JPEG/PNG input.")
    return img


def _ensure_min_resolution(img: np.ndarray) -> np.ndarray:
    """
    Upscale very small mobile images so that bubbles are large enough
    for reliable classification. Low-res cameras (e.g., 640×480) produce
    bubbles that are only ~8 pixels wide, making fill-ratio unreliable.
    """
    h, w = img.shape[:2]
    if w < MIN_WIDTH:
        scale = MIN_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return img


def _enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to
    normalize uneven lighting from mobile camera flashes and shadows.
    This is applied per-channel in LAB color space so colors are preserved.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _bilateral_filter(img: np.ndarray) -> np.ndarray:
    """
    Apply bilateral filter to reduce noise while keeping bubble edges sharp.

    d=9          : diameter of pixel neighbourhood
    sigmaColor=75: filter sigma in the color space
    sigmaSpace=75: filter sigma in coordinate space
    """
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)


def _canny_edges(gray: np.ndarray) -> np.ndarray:
    """Canny edge map used for corner detection."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, threshold1=30, threshold2=100)


def _find_sheet_corners(edges: np.ndarray) -> np.ndarray | None:
    """
    Detect the four corners of the OMR sheet via contour analysis.

    Returns a (4, 2) float32 array [TL, TR, BR, BL] or None if not found.
    """
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) != 4:
        return None

    pts = approx.reshape(4, 2).astype(np.float32)
    return _order_points(pts)


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order corners as [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]    # TL
    rect[2] = pts[np.argmax(s)]    # BR
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect


def _perspective_warp(
    img: np.ndarray,
    output_size: Tuple[int, int] = (800, 1100),
) -> np.ndarray:
    """
    Attempt homography-based perspective correction.

    If the sheet boundary cannot be detected, returns the original image
    (graceful degradation — YOLO is still invoked on the raw frame).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = _canny_edges(gray)
    corners = _find_sheet_corners(edges)

    if corners is None:
        return img  # graceful fall-through

    w, h = output_size
    dst = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    
    # Discard warp if it results in a low-contrast flat/solid color (e.g. warping desk mat/background noise)
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    if gray_warped.std() < 18.0:
        return img
    return warped
