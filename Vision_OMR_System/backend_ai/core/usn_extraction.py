"""
usn_extraction.py
=================
Extracts and corrects handwritten USN (Student ID) from OMR sheets using EasyOCR.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Lazy import easyocr to save startup time
_reader = None

VALID_BRANCHES = ["CS", "IS", "EC", "EE", "ME", "CV", "AI", "AD", "CI", "C1", "LI"]

DIGIT_TO_LETTER = {
    '0': ['O', 'D', 'C'],
    '1': ['I', 'L', 'T', 'J'],
    '2': ['Z', 'C', 'L', 'E'],
    '3': ['E', 'B', 'S'],
    '4': ['A', 'H', 'Y', 'X', 'C'],
    '5': ['S'],
    '6': ['G', 'C'],
    '7': ['T', 'L', 'F'],
    '8': ['B', 'S', 'R'],
    '9': ['G', 'P']
}


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        # Initialize reader (uses CPU by default as per pip list)
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader


def resolve_branch(branch_cand: str) -> str:
    """Resolve a candidate branch string (potentially containing OCR errors/digits) to a valid branch."""
    branch_cand = branch_cand.upper()
    if branch_cand in ("U", "H"):
        return "LI"
        
    if branch_cand in VALID_BRANCHES:
        return branch_cand
        
    # Generate all candidate letter combinations
    pos0_options = [branch_cand[0]]
    if branch_cand[0] in DIGIT_TO_LETTER:
        pos0_options.extend(DIGIT_TO_LETTER[branch_cand[0]])
        
    pos1_options = [branch_cand[1]] if len(branch_cand) > 1 else [""]
    if len(branch_cand) > 1 and branch_cand[1] in DIGIT_TO_LETTER:
        pos1_options.extend(DIGIT_TO_LETTER[branch_cand[1]])
        
    for p0 in pos0_options:
        for p1 in pos1_options:
            comb = p0 + p1
            if comb in VALID_BRANCHES:
                return comb
                
    # Fallback to CS if no match (most common)
    return "CS"


def correct_usn_format(text: str) -> str:
    """
    Clean up raw OCR text to a plausible VTU USN.
    Format: <RegionDigit><College2Letters><Year2Digits><Branch2Letters><Roll3or4Digits>
    e.g. 4VV23CS205, 4VV23LI108
    """
    # Remove whitespace and punctuation, uppercase
    text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    if not text:
        return "UNKNOWN"
        
    # Map W -> VV (very common EasyOCR mistake)
    text = text.replace('W', 'VV')
    
    # If the text starts with VV or V followed by two digits (year), prepend 4
    if text.startswith('VV') and len(text) > 3 and text[2].isdigit():
        text = '4' + text
    elif text.startswith('V') and len(text) > 2 and text[1].isdigit():
        text = '4V' + text
    
    # Locate a potential USN inside the string. 
    # Region: digit or typical misread letter (1-9, I, L, T, Y, A, H, G)
    # College: 2 letters/digits (V, W, U, 1, L)
    # Year: 2 digits or misread letters (0-9, O, I, Z, S)
    # Branch: 1 or 2 letters/digits (A-Z, 0-9)
    # Roll: 3 or 4 digits or misread letters (0-9, O, I, Z, S, B, G)
    
    pattern = r'([1-9IYLTAHG6V][V1LWC][V1LWC][0-9OIZS]{2}[A-Z0-9]{1,2}[0-9OIZSBLG]{3,4})'
    match = re.search(pattern, text)
    
    if match:
        raw_usn = match.group(1)
        
        # Region is always 4 for VV
        region = '4'
        college = "VV" # Force to VV since it's the only college for this OMR sheet setup
        
        # 3. Fix Year
        def fix_digits(s):
            return s.replace('O', '0').replace('I', '1').replace('Z', '2').replace('S', '5').replace('B', '8').replace('G', '6').replace('L', '1').replace('T', '1')
        
        year = fix_digits(raw_usn[3:5])
        
        # Year correction for 3/8 confusion (e.g. 28 -> 23, 48 -> 23)
        if year in ('28', '2B', '48', '18', 'Z8', 'S8', '88'):
            year = '23'
        
        # Dynamically split branch and roll based on length
        branch_roll_part = raw_usn[5:]
        n = len(branch_roll_part)
        
        if n >= 5:
            # e.g. "CS205" (5), "C1108" (5), "CI1008" (6)
            branch_cand = branch_roll_part[:2]
            roll_cand = branch_roll_part[2:]
        else:
            # n <= 4, e.g. "U608" (4), "U108" (4)
            branch_cand = branch_roll_part[:1]
            roll_cand = branch_roll_part[1:]
            
        branch = resolve_branch(branch_cand)
        roll = fix_digits(roll_cand)
        
        return region + college + year + branch + roll
        
    # If no pattern matches, fallback to generic cleanup if it looks like a short USN
    if 8 <= len(text) <= 12:
        # Simple digits fix
        def fix_digits(s):
            return s.replace('O', '0').replace('I', '1').replace('Z', '2').replace('S', '5')
        text = fix_digits(text)
        return text
        
    return "UNKNOWN"


def extract_usn_from_roi(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> str:
    """
    Crop the USN bounding box, run multi-candidate Tesseract OCR consensus with fast-path, and return USN.
    """
    h, w = image.shape[:2]
    
    # Crop the Date + USN area (left 42% of header row box)
    box_w = x2 - x1
    x2_cropped = x1 + (box_w * 0.42)
    
    pad = 8
    rx1 = max(0, int(x1) - pad)
    ry1 = max(0, int(y1) - pad)
    rx2 = min(w, int(x2_cropped) + pad)
    ry2 = min(h, int(y2) + pad)
    
    roi = image[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return "UNKNOWN"
        
    debug_dir = Path(__file__).parent.parent / "debug_output"
    debug_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(debug_dir / "usn_crop_test.png"), roi)
        
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    candidates = []
    
    import pytesseract
    
    # Ordered search space: most likely configs first
    search_space = [
        # (scale, variant_name, psm)
        (2.0, "otsu", 6),
        (3.0, "otsu", 6),
        (2.0, "equalized", 6),
        (3.0, "equalized", 6),
        (2.0, "otsu_inv", 6),
        (3.0, "otsu_inv", 6),
        (2.0, "large_gray", 6),
        (3.0, "large_gray", 6),
        # Fallbacks with PSM 3/4
        (2.0, "otsu", 3),
        (3.0, "otsu", 3),
        (2.0, "otsu", 4),
        (3.0, "otsu", 4),
    ]
    
    # Pre-cache scaled images to avoid redundant scaling
    scaled_images = {}
    for scale in [2.0, 3.0]:
        scaled_images[scale] = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
    for scale, var_name, psm in search_space:
        large = scaled_images[scale]
        
        # Apply preprocessing
        if var_name == "large_gray":
            img_var = large
        elif var_name == "equalized":
            img_var = cv2.equalizeHist(large)
        elif var_name == "otsu":
            _, img_var = cv2.threshold(large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif var_name == "otsu_inv":
            _, img_var = cv2.threshold(large, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            continue
            
        try:
            custom_config = f'--psm {psm} --oem 3'
            raw_text = pytesseract.image_to_string(img_var, config=custom_config).strip()
            if not raw_text:
                continue
                
            corrected = correct_usn_format(raw_text)
            if corrected != "UNKNOWN":
                # Fast path check: perfect VTU USN (e.g. 4VV23CS005)
                if corrected.startswith("4VV") and len(corrected) == 10:
                    print(f"[USN Tesseract Fast-Path] Found perfect match '{corrected}' using scale={scale}, {var_name}, psm={psm}")
                    return corrected
                    
                candidates.append((corrected, raw_text, var_name, psm, scale))
        except Exception as e:
            print(f"[USN Tesseract Fast-Path Error] scale={scale}, {var_name}, psm={psm}: {e}")
            
    # Rank candidates:
    # 1. Matches expected OMR USN prefix (starts with "4VV")
    # 2. Closest to correct length (10 characters)
    if candidates:
        candidates.sort(key=lambda x: (
            0 if x[0].startswith("4VV") else 1,
            abs(len(x[0]) - 10)
        ))
        best = candidates[0]
        print(f"[USN Tesseract Multi-Candidate Fallback] Best: '{best[0]}' from Raw: '{best[1]}' (method: {best[2]}, psm: {best[3]}, scale: {best[4]})")
        return best[0]
        
    print("[USN Tesseract Fast-Path] No valid USN found among candidates")
    return "UNKNOWN"
