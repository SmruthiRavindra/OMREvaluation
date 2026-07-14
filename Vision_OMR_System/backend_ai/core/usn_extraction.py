"""
usn_extraction.py
=================
Robust handwritten USN (University Seat Number) recognition engine.

Uses EasyOCR with strict A-Z / 0-9 character whitelisting to read
alphanumeric handwriting that defeats dictionary-based OCR models.

The pipeline:
  1. Crop the USN bounding-box region from the clean OMR image.
  2. Preprocess the crop (grayscale → CLAHE → adaptive threshold).
  3. Run EasyOCR with `allowlist` restricted to uppercase + digits.
  4. Apply structural VTU USN correction on the raw OCR output.
  5. Return the best candidate via multi-scale consensus ranking.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Lazy-loaded EasyOCR reader (one-time init, ~2-3 s on CPU)
# ---------------------------------------------------------------------------
_reader = None

_ALPHANUMERIC_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _get_reader():
    """Lazily initialise the EasyOCR reader on first call."""
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(
            ["en"],
            gpu=False,
            verbose=False,
        )
    return _reader


# ---------------------------------------------------------------------------
# VTU branch table
# ---------------------------------------------------------------------------
VALID_BRANCHES = frozenset([
    "CS", "IS", "EC", "EE", "ME", "CV", "AI", "AD", "CI", "C1", "LI",
    "AS", "BT", "CH", "AE", "ML", "CB", "CD", "CY", "IM", "TX",
])

# ---------------------------------------------------------------------------
# Character confusion maps (handwriting OCR ↔ structural position)
# ---------------------------------------------------------------------------
_LETTER_TO_DIGIT = {
    "O": "0", "D": "0", "U": "0",
    "I": "1", "L": "1", "T": "1", "J": "1",
    "Z": "2", "Q": "2",
    "E": "3", "C": "3",
    "A": "4", "H": "4", "Y": "4", "X": "4",
    "S": "5",
    "G": "6",
    "F": "7",
    "B": "8", "R": "8",
    "P": "9",
}

_DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "F",
    "8": "B",
    "9": "P",
}


def _char_to_digit(c: str) -> str:
    """Convert a character to its digit equivalent if it's a letter."""
    if c.isdigit():
        return c
    return _LETTER_TO_DIGIT.get(c, c)


def _char_to_letter(c: str) -> str:
    """Convert a character to its letter equivalent if it's a digit."""
    if c.isalpha():
        return c
    return _DIGIT_TO_LETTER.get(c, c)


# ---------------------------------------------------------------------------
# Branch resolution
# ---------------------------------------------------------------------------
def resolve_branch(candidate: str) -> str:
    """
    Resolve a 1-or-2 character OCR candidate to a valid VTU branch code.
    Tries all plausible character substitutions before falling back to CS.
    """
    candidate = candidate.upper()

    # Direct match
    if candidate in VALID_BRANCHES:
        return candidate

    # Single-character shortcuts
    _SINGLE = {"I": "IS", "E": "EC", "A": "AI", "M": "ME"}
    if len(candidate) == 1 and candidate in _SINGLE:
        return _SINGLE[candidate]

    # Try all substitution combinations for 2-char candidates
    if len(candidate) >= 2:
        c0, c1 = candidate[0], candidate[1]
        opts0 = [c0] + ([_DIGIT_TO_LETTER[c0]] if c0 in _DIGIT_TO_LETTER else [])
        opts1 = [c1] + ([_DIGIT_TO_LETTER[c1]] if c1 in _DIGIT_TO_LETTER else [])
        for a in opts0:
            for b in opts1:
                combo = a + b
                if combo in VALID_BRANCHES:
                    return combo

    return "CS"


# ---------------------------------------------------------------------------
# Structural USN correction
# ---------------------------------------------------------------------------
def correct_usn_format(text: str) -> str:
    """
    Clean raw OCR text → plausible VTU USN.

    VTU format: <Region:1digit><College:2letters><Year:2digits><Branch:2letters><Roll:3-4digits>
    Example:    4VV23CS001

    The prefix 4VV is always forced (all sheets are from the same region/college).
    Year, Branch and Roll are extracted dynamically from the OCR text.
    """
    # ── 1. Strip everything except A-Z, 0-9, and a few known OCR glyphs ──
    text = re.sub(r"[^A-Za-z0-9()\[\]{}]", "", text).upper()
    if not text:
        return "UNKNOWN"

    # ── 2. Remove common printed header noise that Tesseract/EasyOCR picks up ──
    text = re.sub(r"^(DATE|PATE|OATE|LITA|UM|TIME|MAX|REG|NO|USN)+", "", text)

    # ── 3. Common single-glyph OCR substitutions ──
    text = text.replace("W", "VV")
    text = text.replace("(", "4").replace("[", "4").replace("{", "4")

    # ── 4. Normalise the 4VV prefix ──
    # GVV / LVV / 1VV / IVV / TVV → 4VV
    text = re.sub(r"^[GLIT1][VUW]{2}", "4VV", text)
    text = re.sub(r"^4[VUW]{2}", "4VV", text)

    # If it starts with VV<digit>, prepend 4
    if re.match(r"^VV\d", text):
        text = "4" + text
    # If it starts with V<digit> (missing second V), insert V
    elif re.match(r"^V\d", text):
        text = "4V" + text
    # If it starts with two letter-like chars followed by digits, prepend 4
    elif re.match(r"^[VUWGLTY]{2}\d", text):
        text = "4" + text

    # ── 5. Match the structural pattern ──
    # Region(1) College(2) Year(2) BranchRoll(3-6)
    pattern = (
        r"([1-9IYLTAHG6V4CDUO]"       # Region: digit or misread letter
        r"[V1LWCUYGC]{2}"              # College: 2 letters (V-like)
        r"[A-Z0-9]{2}"                 # Year: 2 alphanumeric
        r"[A-Z0-9]{1,2}"              # Branch: 1 or 2 chars
        r"[0-9OIZSBLGT]{3,4})"        # Roll: 3-4 digit-like chars
    )
    match = re.search(pattern, text)

    if not match:
        # Fallback: if text is approximately USN length, return cleaned version
        if 8 <= len(text) <= 12:
            return text
        return "UNKNOWN"

    raw = match.group(1)

    # ── 6. Decode each structural segment ──
    region = "4"
    college = "VV"
    year = "".join(_char_to_digit(c) for c in raw[3:5])

    # Split branch vs roll in the remainder
    tail = raw[5:]
    if not tail:
        return "UNKNOWN"

    # Heuristic: if second char looks like a digit, branch is 1 char
    c0 = tail[0]
    c1 = tail[1] if len(tail) > 1 else ""

    digit_like = set("0123456789OQZ")

    if c0 == "C" and c1 in ("1", "I", "L", "T"):
        # C1 is a valid branch
        branch_raw = c0 + c1
        roll_raw = tail[2:]
    elif c1 in digit_like:
        branch_raw = c0
        roll_raw = tail[1:]
    else:
        branch_raw = c0 + c1
        roll_raw = tail[2:]

    branch = resolve_branch(branch_raw)
    roll = "".join(_char_to_digit(c) for c in roll_raw)

    usn = region + college + year + branch + roll
    return usn


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------
def _preprocess_for_ocr(
    gray: np.ndarray,
    scale: float = 2.0,
    method: str = "clahe",
) -> np.ndarray:
    """
    Resize and enhance a grayscale USN crop for OCR.

    Methods:
      - 'clahe'   : CLAHE histogram equalisation → adaptive threshold
      - 'otsu'    : Otsu global threshold
      - 'adaptive': Adaptive Gaussian threshold
    """
    # Up-scale for better character segmentation
    large = cv2.resize(
        gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )

    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(large)
        # Adaptive threshold produces cleaner strokes than Otsu on uneven lighting
        processed = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=15,
        )
    elif method == "otsu":
        _, processed = cv2.threshold(
            large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif method == "adaptive":
        processed = cv2.adaptiveThreshold(
            large, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=21,
            C=10,
        )
    else:
        processed = large

    return processed


# ---------------------------------------------------------------------------
# Multi-scale EasyOCR runner
# ---------------------------------------------------------------------------
def _run_easyocr_on_image(img: np.ndarray) -> str:
    """
    Run EasyOCR on a preprocessed image with character whitelisting.
    Returns the concatenated raw text (uppercase, alphanumeric only).
    """
    reader = _get_reader()
    results = reader.readtext(
        img,
        allowlist=_ALPHANUMERIC_WHITELIST,
        detail=0,               # return strings only
        paragraph=True,         # merge nearby text boxes
        min_size=5,
        text_threshold=0.3,
        low_text=0.3,
        mag_ratio=1.0,          # Disable internal CRAFT magnification to save 40%+ CPU latency
    )
    raw = " ".join(results).strip().upper()
    # Final cleanup: keep only A-Z 0-9
    raw = re.sub(r"[^A-Z0-9]", "", raw)
    return raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_usn_from_roi(
    image: np.ndarray,
    x1: float, y1: float,
    x2: float, y2: float,
) -> str:
    """
    Crop the USN bounding box detected by YOLO, run multi-scale EasyOCR
    with alphanumeric whitelisting, and return the best VTU USN candidate.

    Parameters
    ----------
    image : np.ndarray
        The full preprocessed OMR sheet image (BGR).
    x1, y1, x2, y2 : float
        YOLO-detected bounding box coordinates for the USN region.

    Returns
    -------
    str
        Corrected USN string, or ``"UNKNOWN"`` if extraction fails.
    """
    h, w = image.shape[:2]

    # ── 1. Crop the USN area (left 42 % of the header-row box) ───────────
    box_w = x2 - x1
    x2_cropped = x1 + (box_w * 0.42)

    pad = 8
    rx1 = max(0, int(x1) - pad)
    ry1 = max(0, int(y1) - pad)
    rx2 = min(w, int(x2_cropped) + pad)
    ry2 = min(h, int(y2) + pad)

    if rx2 <= rx1 or ry2 <= ry1:
        return "UNKNOWN"

    roi = image[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return "UNKNOWN"

    # ── Debug: save crop for inspection ───────────────────────────────────
    debug_dir = Path(__file__).parent.parent / "debug_output"
    debug_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(debug_dir / "usn_crop_test.png"), roi)

    # ── 2. Convert to grayscale ───────────────────────────────────────────
    if len(roi.shape) == 3 and roi.shape[2] == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    # ── 3. Optimized Single-Stage OCR (scale 2.0 with CLAHE & adaptive threshold) ──
    try:
        processed = _preprocess_for_ocr(gray, scale=2.0, method="clahe")
        raw_text = _run_easyocr_on_image(processed)
        if raw_text:
            corrected = correct_usn_format(raw_text)
            if corrected != "UNKNOWN":
                print(f"[USN EasyOCR] '{corrected}' (raw='{raw_text}', scale=2.0)")
                return corrected
    except Exception as exc:
        print(f"[USN EasyOCR Error] {exc}")

    print("[USN EasyOCR] No valid USN extracted")
    return "UNKNOWN"
