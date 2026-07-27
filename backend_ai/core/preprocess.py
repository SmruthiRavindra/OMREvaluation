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

# Cache CLAHE instance at module level — creating it per-request wastes ~1-2ms
# and allocates a new histogram table each time.
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Full pre-processing pipeline: decode → upscale → denoise → enhance → warp.

    Order matters for performance:
      1. Decode raw bytes
      2. Downscale to MIN_WIDTH — bilateral filter cost scales with pixel count,
         so we shrink first and pay the filter cost only on the working resolution.
      3. Bilateral filter — noise reduction at the smaller resolution (~4-9× faster)
      4. CLAHE contrast enhancement — cheap at working resolution
      5. Perspective warp — homography transform

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
    img = _ensure_min_resolution(img)   # downscale first!
    img = _bilateral_filter(img)        # now runs on 800px, not 4000px
    img = _enhance_contrast(img)
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
    Downscale very large images to MIN_WIDTH before the expensive bilateral
    filter step.  Also upscales very small mobile images (e.g., 640×480) so
    that bubbles are large enough for reliable fill-ratio classification.

    Running this BEFORE bilateral filter means the filter operates on
    ~640k pixels instead of potentially 4-8M pixels on a 4K phone image
    — a 4-9× speedup for that step alone.
    """
    h, w = img.shape[:2]
    if w == MIN_WIDTH:
        return img
    scale = MIN_WIDTH / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    # Use INTER_AREA for downscaling (better quality), INTER_CUBIC for upscaling
    interp = cv2.INTER_AREA if w > MIN_WIDTH else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def _enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to
    normalize uneven lighting from mobile camera flashes and shadows.
    Applied per-channel in LAB color space so colors are preserved.

    Uses the module-level cached _CLAHE instance (no per-call allocation).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)   # use cached instance
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
    Optimised for cluttered backgrounds in live webcam feeds.
    """
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    # Sort contours by area descending and check top candidates
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    h_f, w_f = edges.shape[:2]
    total_area = h_f * w_f
    min_area = total_area * 0.25  # Reduced threshold to support perspective tilts
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
            
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        
        # Check if the contour is a convex 4-corner polygon
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2).astype(np.float32)
            return _order_points(pts)
            
    return None


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


def _find_sheet_corners_otsu(gray: np.ndarray) -> np.ndarray | None:
    """
    Detect OMR sheet corners using Otsu binarization.
    Highly robust for bright sheets on darker/cluttered backgrounds.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological closing to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    h_f, w_f = gray.shape[:2]
    total_area = h_f * w_f
    min_area = total_area * 0.15 # Reduced slightly to support perspective tilts
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
            
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2).astype(np.float32)
            return _order_points(pts)
            
    return None


# ---------------------------------------------------------------------------
# TC#19 — Multiple-sheet probe
# ---------------------------------------------------------------------------

# Minimum fraction of total image area for a quad to count as a candidate sheet.
# A value of 0.20 means any rectangle covering ≥20% of the frame is considered
# a sheet outline.  Two such quads → ambiguous capture.
_SHEET_AREA_FRACTION: float = 0.20


def _count_large_quads(gray: np.ndarray) -> int:
    """
    Count how many distinct large quad-shaped contours exist in the frame.

    Returns an integer count.  Normally 1 for a good single-sheet capture;
    >1 means there are likely multiple sheets in the same photo.

    This is a *read-only* probe — it does not modify any image data and
    shares the same Otsu-based thresholding already used by
    _find_sheet_corners_otsu, so it adds negligible extra work.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    h, w = gray.shape[:2]
    min_area = h * w * _SHEET_AREA_FRACTION
    quad_count = 0

    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < min_area:
            break  # remaining contours are too small to be sheets
        hull   = cv2.convexHull(c)
        peri   = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad_count += 1

    return quad_count


def _perspective_warp(
    img: np.ndarray,
    output_size: Tuple[int, int] = (800, 1100),
) -> np.ndarray:
    """
    Attempt homography-based perspective correction.
    Tries robust Otsu-based contours first, falling back to Canny edges.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try Otsu contour detection first
    corners = _find_sheet_corners_otsu(gray)
    
    # Fallback to Canny edges
    if corners is None:
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
    
    # Discard warp if it results in a low-contrast flat/solid color
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    if gray_warped.std() < 18.0:
        return img
    return warped


def preprocess_image_detect(image_bytes: bytes) -> Tuple[np.ndarray, bool, bool]:
    """
    Full pre-processing pipeline returning the cleaned image plus two flags:
      - warped      : True if perspective correction was successfully applied.
      - multi_sheet : True if >1 large sheet-like quad was found in the frame
                      (TC#19 multiple-sheets guard).  Callers should reject the
                      image and ask the user to retake with a single sheet.

    Pipeline order (same as preprocess_image for consistency):
      decode → downscale → bilateral filter → CLAHE → [multi-sheet probe] → warp

    The multi_sheet probe is a read-only parallel pass on the same grayscale
    data — it does NOT alter the warp path.
    """
    img = _decode(image_bytes)
    img = _ensure_min_resolution(img)   # downscale first
    img = _bilateral_filter(img)        # filter at working resolution
    img = _enhance_contrast(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # TC#19 — probe for multiple sheets BEFORE warping (read-only, no side-effects)
    multi_sheet: bool = _count_large_quads(gray) > 1

    # Try Otsu binarization contour detection first
    corners = _find_sheet_corners_otsu(gray)

    # Fallback to Canny edges
    if corners is None:
        edges = _canny_edges(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        corners = _find_sheet_corners(edges_closed)

    if corners is None:
        return img, False, multi_sheet

    w, h = (800, 1100)
    dst = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (w, h))

    # Discard warp if it results in a low-contrast flat/solid color (desk mat, background)
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    if gray_warped.std() < 18.0:
        return img, False, multi_sheet

    return warped, True, multi_sheet
