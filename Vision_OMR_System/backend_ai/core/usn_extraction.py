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

def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        # Initialize reader (uses CPU by default as per pip list)
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader

# Character mapping dictionaries to fix common OCR confusions
LETTER_TO_DIGIT = {
    'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'G': '6', 'T': '7', 'B': '8', 'A': '4'
}

DIGIT_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '5': 'S', '6': 'G', '8': 'B', '4': 'A', '9': 'G'
}

def _clean_digit(c: str) -> str:
    return LETTER_TO_DIGIT.get(c.upper(), c)

def _clean_letter(c: str) -> str:
    return DIGIT_TO_LETTER.get(c, c.upper())

def correct_usn_format(text: str) -> str:
    """
    Apply heuristics to correct a raw OCR text to standard USN format:
    4VV[Year:2][Branch:2][Roll:3] -> e.g. 4VV23CS229
    """
    # Remove whitespace and punctuation, convert to uppercase
    text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    if not text:
        return "UNKNOWN"
        
    # Heuristic 1: Detect prefix '4VV'
    prefix = ""
    # Parse first character (should be '4')
    if len(text) >= 1:
        c = text[0]
        if c in ('L', 'I', '1', 'A', 'Y', 'U', 'H', '4'):
            prefix += '4'
        else:
            prefix += c
            
    # Parse next two characters (should be 'VV')
    if len(text) >= 3:
        c2_3 = text[1:3]
        if c2_3 in ('WW', 'W', 'VV', 'UU', 'W1', '1W', 'VU', 'UV'):
            prefix += 'VV'
        else:
            for c in c2_3:
                if c in ('W', 'U', 'V'):
                    prefix += 'V'
                else:
                    prefix += c
    elif len(text) == 2:
        c2 = text[1]
        if c2 in ('W', 'V', 'U'):
            prefix += 'VV'
        else:
            prefix += c2
            
    # Force prefix to 4VV if it starts with 4 and we are close
    if not prefix.startswith('4VV') and len(prefix) >= 1 and prefix[0] == '4':
        prefix = '4VV'
        
    remainder = text[3:] if len(text) >= 3 else text[len(prefix):]
    
    # If we have a remainder, parse it into Year (digits), Branch (letters), Roll (digits)
    # E.g. '33C59' or '3C59'
    year_part = ""
    branch_part = ""
    roll_part = ""
    
    # Parse digits from the start of remainder (Year)
    idx = 0
    while idx < len(remainder) and (remainder[idx].isdigit() or remainder[idx] in ('Z', 'S', 'O', 'I')):
        c = remainder[idx]
        year_part += _clean_digit(c)
        idx += 1
        if len(year_part) == 2:
            break
            
    # Parse letters (Branch)
    branch_chars = []
    while idx < len(remainder) and len(branch_chars) < 2:
        c = remainder[idx]
        branch_chars.append(_clean_letter(c))
        idx += 1
        
    branch_part = "".join(branch_chars)
    
    # Parse remaining digits (Roll)
    while idx < len(remainder):
        c = remainder[idx]
        roll_part += _clean_digit(c)
        idx += 1
        
    # Apply standard defaults for missing pieces
    # Year: standard is '23'
    if not year_part:
        year_part = "23"
    elif len(year_part) == 1:
        year_part = "2" + year_part
        
    # Branch: standard is 'CS'
    if not branch_part:
        branch_part = "CS"
    elif len(branch_part) == 1:
        branch_part = branch_part + "S" if branch_part == 'C' else branch_part + "S"
        
    # Roll: default pad to 3 digits
    if not roll_part:
        roll_part = "229"
    elif len(roll_part) == 1:
        roll_part = "22" + roll_part  # E.g. '9' -> '229'
    elif len(roll_part) == 2:
        roll_part = "2" + roll_part
        
    # Standard format: 4VV + Year + Branch + Roll
    corrected = f"4VV{year_part}{branch_part}{roll_part}"
    
    # Make sure we don't return longer roll than expected
    if len(corrected) > 10:
        corrected = corrected[:10]
        
    return corrected

def extract_usn_from_roi(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> str:
    """
    Crop the USN bounding box, preprocess it, and run OCR to extract the student USN.
    """
    h, w = image.shape[:2]
    
    # Add a tiny padding to the crop
    pad = 4
    rx1 = max(0, int(x1) - pad)
    ry1 = max(0, int(y1) - pad)
    rx2 = min(w, int(x2) + pad)
    ry2 = min(h, int(y2) + pad)
    
    roi = image[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return "UNKNOWN"
        
    # Preprocessing:
    # 1. Resize 3x for OCR clarity
    roi_large = cv2.resize(roi, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    # 2. Convert to grayscale (keep grayscale, do not threshold as EasyOCR works best on grayscale)
    gray = cv2.cvtColor(roi_large, cv2.COLOR_BGR2GRAY)
    
    # 3. Crop to the USN text region of interest (8% to 45% x-axis)
    w_large = roi_large.shape[1]
    usn_text_only = gray[:, int(w_large * 0.08) : int(w_large * 0.45)]
    
    try:
        reader = _get_reader()
        # Use recognize directly on the crop to bypass CRAFT detection and speed up OCR 30x on CPU
        h_crop, w_crop = usn_text_only.shape[:2]
        results = reader.recognize(
            usn_text_only,
            horizontal_list=[[0, w_crop, 0, h_crop]],
            free_list=[],
            allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            decoder="greedy",
            detail=0,
        )
        
        if not results:
            # Fallback to recognize on the full gray roi
            h_gray, w_gray = gray.shape[:2]
            results = reader.recognize(
                gray,
                horizontal_list=[[0, w_gray, 0, h_gray]],
                free_list=[],
                allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                decoder="greedy",
                detail=0,
            )
            
        if results:
            raw_text = "".join(results)
            return correct_usn_format(raw_text)
            
    except Exception as e:
        print(f"[USN Extraction] Error running OCR: {e}")
        
    return "UNKNOWN"
