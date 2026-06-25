"""
preprocess.py
=============
OpenCV-based image pre-processing pipeline for OMR sheet normalization.

Steps:
  1. Bilateral filter  – noise reduction preserving edges
  2. Adaptive threshold / Canny edge detection
  3. Homography (perspective warp) – corrects camera tilt / skew
"""

import cv2
import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Full pre-processing pipeline: decode → denoise → warp.

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
    return warped
