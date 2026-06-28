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
    Crop the USN bounding box, preprocess it, and run OCR to extract the student USN.
    """
    h, w = image.shape[:2]
    
    # Slice only the first 42% of the bounding box width.
    # The USN (Date field) always sits on the far left of the row.
    box_w = x2 - x1
    x2_cropped = x1 + (box_w * 0.42)
    
    # Add padding to ensure no edge clipping
    pad = 8
    rx1 = max(0, int(x1) - pad)
    ry1 = max(0, int(y1) - pad)
    rx2 = min(w, int(x2_cropped) + pad)
    ry2 = min(h, int(y2) + pad)
    
    roi = image[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return "UNKNOWN"
    
    # Preprocessing options:
    # 1. Resized BGR for modern CNN-based EasyOCR detection
    roi_large = cv2.resize(roi, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # 2. Convert to grayscale & enhanced CLAHE as fallbacks
    gray = cv2.cvtColor(roi_large, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    
    # 3. Thresholded image
    blurred = cv2.GaussianBlur(enhanced_gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    try:
        reader = _get_reader()
        
        candidates = []
        
        # Test on raw BGR, resized BGR, enhanced gray and thresholded images
        for img_input in [roi, roi_large, enhanced_gray, thresh]:
            results = reader.readtext(img_input, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            
            if not results:
                continue
            
            boxes = []
            for bbox, text, conf in results:
                text = text.strip()
                if len(text) < 2:
                    continue
                if text.upper() == 'USN':
                    continue
                y_coords = [p[1] for p in bbox]
                height = max(y_coords) - min(y_coords)
                boxes.append((height, bbox, text, conf))
                
            if not boxes:
                continue
                
            boxes.sort(key=lambda x: x[0], reverse=True)
            max_height = boxes[0][0]
            
            # Filter out tiny print/instructions
            usn_boxes = [b for b in boxes if b[0] >= max_height * 0.40]
            usn_boxes.sort(key=lambda x: min([p[0] for p in x[1]]))
            
            extracted_pieces = [b[2] for b in usn_boxes]
            
            if extracted_pieces:
                raw_text = "".join(extracted_pieces)
                raw_text = raw_text.replace('USN', '').replace('usn', '')
                
                corrected = correct_usn_format(raw_text)
                
                # Dynamic fallback for highly distorted images
                if corrected == "UNKNOWN" and len(raw_text) >= 8:
                    clean_fallback = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
                    if len(clean_fallback) > 3:
                        if clean_fallback[0] in ('Y', 'H', 'A', 'U', '6', 'V'):
                            clean_fallback = '4' + clean_fallback[1:]
                        if clean_fallback[1] in ('1', 'W', 'U'):
                            clean_fallback = clean_fallback[0] + 'VV' + clean_fallback[2:]
                    corrected = clean_fallback
                    
                if corrected != "UNKNOWN":
                    avg_conf = sum(b[3] for b in usn_boxes) / len(usn_boxes)
                    candidates.append((corrected, avg_conf, raw_text))
        
        if candidates:
            # Sort by:
            # 1. Matches expected OMR USN prefix (4VV)
            # 2. Diff to target length (10 chars), then confidence
            candidates.sort(key=lambda x: (
                0 if x[0].startswith("4VV") else 1,
                abs(len(x[0]) - 10) if len(x[0]) < 10 else abs(len(x[0]) - 10.5),
                -x[1]
            ))
            best = candidates[0]
            print(f"[USN Extraction] Raw: '{best[2]}' -> Corrected: '{best[0]}' (conf: {best[1]:.2f})")
            return best[0]
            
    except Exception as e:
        print(f"[USN Extraction] Error running OCR: {e}")
        
    return "UNKNOWN"
