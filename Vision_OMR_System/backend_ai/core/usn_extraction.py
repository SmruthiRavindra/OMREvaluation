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
import pytesseract

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
    if branch_cand == "A":
        return "CV"
    if branch_cand == "I":
        return "IS"
    if branch_cand == "E":
        return "EC"
        
    if branch_cand in VALID_BRANCHES:
        return branch_cand
        
    # Mutation sets for common misread letters / digits in branch codes
    LETTER_MUTATIONS = {
        'L': ['I', 'L', 'T', 'J', '1'],
        '1': ['I', 'L', 'T', 'J', '1'],
        'I': ['I', 'L', 'T', 'J', '1'],
        'U': ['L', 'I', 'U'],
        'H': ['L', 'I', 'H'],
        'C': ['C', 'G', 'O', '0'],
        'S': ['S', '5', '8', 'B'],
        'B': ['B', '8', 'S'],
        'G': ['G', '6', 'C'],
        'D': ['D', 'O', '0'],
        'O': ['O', '0', 'D'],
    }
        
    # Generate all candidate letter combinations
    pos0_options = LETTER_MUTATIONS.get(branch_cand[0], [branch_cand[0]])
    if branch_cand[0] in DIGIT_TO_LETTER:
        pos0_options.extend(DIGIT_TO_LETTER[branch_cand[0]])
        
    pos1_options = LETTER_MUTATIONS.get(branch_cand[1], [branch_cand[1]]) if len(branch_cand) > 1 else [""]
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
    Clean up raw OCR text to a plausible VTU USN using branch-aligned positional templates.
    Format: <RegionDigit><College2Letters><Year2Digits><Branch2Letters><Roll3or4Digits>
    e.g. 4VV23CS229, 4VV23LI108
    """
    # Clean characters: keep only alphanumeric
    text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    if not text:
        return "UNKNOWN"
        
    # Standard substitutions for common OCR errors
    text = text.replace('W', 'VV')
    
    # Locate all potential 8-11 character candidate substrings
    candidates = []
    for i in range(len(text) - 7):
        sub = text[i:i+11]
        if len(sub) < 10:
            sub = text[i:i+10]
        if len(sub) < 9:
            sub = text[i:i+9]
        if len(sub) < 8:
            continue
        candidates.append(sub)
        
    if not candidates:
        return "UNKNOWN"
        
    # Rank candidates by structural template similarity
    LET_TO_DIG = {
        'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'L': '1', 'T': '1', 'Q': '2',
        'E': '3', 'A': '4', 'H': '4', 'Y': '4', 'X': '4', 'C': '3', 'U': '0', 'D': '0', 'F': '7', 'P': '9',
        'J': '1'
    }
    DIG_TO_LET = {
        '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '4': 'A', '7': 'F', '9': 'P'
    }
    
    def to_digit(c):
        return LET_TO_DIG.get(c, c) if c.isalpha() else c
        
    def to_letter(c):
        return DIG_TO_LET.get(c, c) if c.isdigit() else c
        
    best_usn = "UNKNOWN"
    best_score = -1
    
    for cand in candidates:
        # Template: <Region:1d><College:2L><Year:2d><Branch:2L><Roll:3-4d>
        p0 = to_digit(cand[0])
        if p0 not in ('1', '2', '3', '4', '5', '6', '7', '8', '9'):
            p0 = '4'
            
        college = to_letter(cand[1]) + to_letter(cand[2])
        if college in ('UW', 'VU', 'VW', 'VV', 'UU', 'WV'):
            college = 'VV'
            
        # Parse the rest
        rest = cand[3:]
        branch = None
        branch_idx = -1
        
        # Scan for a valid branch in the remaining characters
        for j in range(len(rest) - 1):
            b_cand = to_letter(rest[j]) + to_letter(rest[j+1])
            if rest[j] == 'C' and rest[j+1] in ('1', 'I', 'L', 'T', 'J'):
                branch = 'C1'
                branch_idx = j
                break
            resolved = resolve_branch(b_cand)
            if resolved in VALID_BRANCHES and resolved != "CS":
                branch = resolved
                branch_idx = j
                break
                
        # If no specific branch is matched, check for any double letter
        if not branch:
            for j in range(len(rest) - 1):
                if to_letter(rest[j]).isalpha() and to_letter(rest[j+1]).isalpha():
                    branch = resolve_branch(to_letter(rest[j]) + to_letter(rest[j+1]))
                    branch_idx = j
                    break
                    
        # Default fallback
        if not branch:
            branch = "CS"
            branch_idx = 2
            
        # Extract Year and Roll based on branch position
        year_part = rest[:branch_idx]
        year = "".join(to_digit(c) for c in year_part)
        if len(year) == 1:
            year = '2' + year  # e.g., '3' -> '23'
        elif len(year) == 0:
            year = '23'
        elif len(year) > 2:
            year = year[-2:]
            
        if year not in ('21', '22', '23', '24', '25'):
            year = '23'
            
        roll_part = roll_part = rest[branch_idx+2:]
        roll = "".join(to_digit(c) for c in roll_part)
        roll = "".join(c for c in roll if c.isdigit()) # Ensure only digits in roll
        if len(roll) > 4:
            roll = roll[:3]
            
        # Score template alignment
        score = 0
        if cand[0].isdigit(): score += 1
        if cand[1].isalpha(): score += 1
        if cand[2].isalpha(): score += 1
        if len(year_part) == 2 and all(c.isdigit() for c in year_part): score += 1
        if branch != "CS": score += 2
        if all(c.isdigit() for c in roll_part): score += 1
        
        corrected = p0 + college + year + branch + roll
        if len(corrected) >= 10 and score > best_score:
            best_score = score
            best_usn = corrected
            
    return best_usn


def clean_cell_patch(cell: np.ndarray, cell_size: int = 40) -> np.ndarray:
    """Preprocess, denoise, and isolate the character contour in a cell box."""
    if len(cell.shape) == 3 and cell.shape[2] == 3:
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell.copy()
        
    # Otsu thresholding (dark text -> black, paper -> white)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Remove outer border: clear the outer 2 pixels to erase box boundaries
    border = 2
    thresh[:border, :] = 0
    thresh[-border:, :] = 0
    thresh[:, :border] = 0
    thresh[:, -border:] = 0
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create clean white cell background
    clean_cell = np.full((cell_size, cell_size), 255, dtype=np.uint8)
    
    if contours:
        # Get largest contour (the character)
        c = max(contours, key=cv2.contourArea)
        x, y, w_c, h_c = cv2.boundingRect(c)
        
        # Crop character and invert to black text on white
        char_crop = thresh[y:y+h_c, x:x+w_c]
        char_crop_inv = cv2.bitwise_not(char_crop)
        
        # Resize to fit in clean cell preserving aspect ratio
        max_dim = int(cell_size * 0.70)
        if w_c > h_c:
            new_w = max_dim
            new_h = max(1, int(h_c * (max_dim / w_c)))
        else:
            new_h = max_dim
            new_w = max(1, int(w_c * (max_dim / h_c)))
            
        char_resized = cv2.resize(char_crop_inv, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center the character inside the clean cell
        dx = (cell_size - new_w) // 2
        dy = (cell_size - new_h) // 2
        clean_cell[dy:dy+new_h, dx:dx+new_w] = char_resized
        
    return clean_cell


def extract_usn_from_roi(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> str:
    """
    Crop the USN bounding box, run grid-splitting reconstruction consensus first,
    and fall back to multi-candidate Tesseract OCR consensus if needed.
    """
    h, w = image.shape[:2]
    box_w = x2 - x1
    box_h = y2 - y1
    
    # ── Method 1: Grid-Splitting Reconstruction & EasyOCR Consensus ───────
    try:
        # Standard relative USN grid coordinates inside header row box
        grid_x1 = max(0, int(x1 + box_w * 0.096))
        grid_x2 = min(w, int(x1 + box_w * 0.358))
        grid_y1 = max(0, int(y1 + box_h * 0.446))
        grid_y2 = min(h, int(y1 + box_h * 1.0))
        
        cw = grid_x2 - grid_x1
        col_w = cw / 10.0
        
        cleaned_cells = []
        for i in range(10):
            c_start = int(grid_x1 + i * col_w)
            c_end = int(grid_x1 + (i + 1) * col_w)
            cell = image[grid_y1:grid_y2, c_start:c_end]
            cleaned_cells.append(clean_cell_patch(cell))
            
        reconstructed = np.hstack(cleaned_cells)
        
        # Debug save
        debug_dir = Path(__file__).parent.parent / "debug_output"
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / "reconstructed_consensus.png"), reconstructed)
        
        # Run EasyOCR
        reader = _get_reader()
        results = reader.readtext(reconstructed, detail=0)
        if results:
            raw_text = "".join(results).strip()
            corrected = correct_usn_format(raw_text)
            if corrected != "UNKNOWN" and corrected.startswith("4VV") and len(corrected) == 10:
                print(f"[USN Grid-Splitting EasyOCR] Success: '{corrected}' (Raw: '{raw_text}')")
                return corrected
    except Exception as e:
        print(f"[USN Grid-Splitting EasyOCR Error] {e}")

    # ── Method 2: Full-box Multi-Candidate Tesseract Fallback ─────────────
    # Crop the Date + USN area (left 42% of header row box)
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
        
    if len(roi.shape) == 3 and roi.shape[2] == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()
        
    candidates = []
    
    # Ordered search space
    search_space = [
        (2.0, "otsu", 6),
        (3.0, "otsu", 6),
        (2.0, "equalized", 6),
        (3.0, "equalized", 6),
        (2.0, "otsu_inv", 6),
        (3.0, "otsu_inv", 6),
        (2.0, "large_gray", 6),
        (3.0, "large_gray", 6),
        (2.0, "otsu", 3),
        (3.0, "otsu", 3),
        (2.0, "otsu", 4),
        (3.0, "otsu", 4),
    ]
    
    scaled_images = {}
    for scale in [2.0, 3.0]:
        scaled_images[scale] = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
    for scale, var_name, psm in search_space:
        large = scaled_images[scale]
        
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
                if corrected.startswith("4VV") and len(corrected) == 10:
                    print(f"[USN Tesseract Fast-Path] Found perfect match '{corrected}' using scale={scale}, {var_name}, psm={psm}")
                    return corrected
                    
                candidates.append((corrected, raw_text, var_name, psm, scale))
        except Exception as e:
            # Silence/catch tesseract errors if binary not installed
            pass
            
    # Also try EasyOCR on full crop as fallback
    try:
        reader = _get_reader()
        results = reader.readtext(roi, detail=0)
        if results:
            raw_text = " ".join(results).strip()
            corrected = correct_usn_format(raw_text)
            if corrected != "UNKNOWN":
                if corrected.startswith("4VV") and len(corrected) == 10:
                    print(f"[USN EasyOCR Fallback] Found match '{corrected}'")
                    return corrected
                candidates.append((corrected, raw_text, "easyocr", 0, 1.0))
    except Exception as e:
        print(f"[USN EasyOCR Fallback Error] {e}")
            
    # Rank candidates:
    # 1. Matches expected OMR USN prefix (starts with "4VV")
    # 2. Closest to correct length (10 characters)
    if candidates:
        candidates.sort(key=lambda x: (
            0 if x[0].startswith("4VV") else 1,
            abs(len(x[0]) - 10)
        ))
        best = candidates[0]
        print(f"[USN Tesseract/EasyOCR Multi-Candidate Fallback] Best: '{best[0]}' from Raw: '{best[1]}' (method: {best[2]}, psm: {best[3]}, scale: {best[4]})")
        return best[0]
        
    print("[USN OCR] No valid USN found among candidates")
    return "UNKNOWN"
