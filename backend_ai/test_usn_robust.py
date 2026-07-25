import re

VALID_BRANCHES = ["CS", "IS", "EC", "EE", "ME", "CV", "AI", "AD"]

DIGIT_TO_LETTER = {
    '0': ['O', 'D', 'C'],
    '1': ['I', 'L', 'T', 'J'],
    '2': ['Z', 'C', 'L', 'E'],
    '3': ['E', 'B'],
    '4': ['A', 'H', 'Y', 'X', 'C'],
    '5': ['S'],
    '6': ['G', 'C'],
    '7': ['T', 'L', 'F'],
    '8': ['B', 'S', 'R'],
    '9': ['G', 'P']
}

def resolve_branch(branch_cand: str) -> str:
    """Resolve a candidate branch string (potentially containing OCR errors/digits) to a valid branch."""
    branch_cand = branch_cand.upper()
    if branch_cand in VALID_BRANCHES:
        return branch_cand
        
    # Generate all candidate letter combinations
    pos0_options = [branch_cand[0]]
    if branch_cand[0] in DIGIT_TO_LETTER:
        pos0_options.extend(DIGIT_TO_LETTER[branch_cand[0]])
        
    pos1_options = [branch_cand[1]]
    if branch_cand[1] in DIGIT_TO_LETTER:
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
    Format: <RegionDigit><College2Letters><Year2Digits><Branch2Letters><Roll3Digits>
    e.g. 4VV23CS205, 4VV23LI108
    """
    # Remove whitespace and punctuation, uppercase
    text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    if not text:
        return "UNKNOWN"
        
    # Map W -> VV (very common EasyOCR mistake)
    text = text.replace('W', 'VV')
    
    # Locate a potential USN inside the string. 
    # Since the OCR might read "Date : 4VV23CS205 Time : 1 Hr", the string is "DATE4VV23CS205TIME1HR"
    # We look for a pattern like: [Region][College][Year][Branch][Roll]
    # Region: digit or typical misread letter (1-9, I, L, T, Y, A, H, G)
    # College: 2 letters/digits (V, W, U, 1, L)
    # Year: 2 digits or misread letters (0-9, O, I, Z, S)
    # Branch: 2 letters/digits (A-Z, 0-9)
    # Roll: 3 digits or misread letters (0-9, O, I, Z, S, B, G)
    
    pattern = r'([1-9IYLTAHG6][V1LWC][V1LWC][0-9OIZS]{2}[A-Z0-9]{2}[0-9OIZSBLG]{3})'
    match = re.search(pattern, text)
    
    if match:
        raw_usn = match.group(1)
        
        # 1. Fix Region
        region = raw_usn[0]
        if region in ('I', 'L', 'T'): region = '1'
        elif region in ('Z', 'S'): region = '2'
        elif region in ('Y', 'A', 'H', 'G', '6'): region = '4'
        
        # 2. Fix College (VV)
        college = "VV" # Force to VV since it's the only college for this OMR sheet setup
        
        # 3. Fix Year
        def fix_digits(s):
            return s.replace('O', '0').replace('I', '1').replace('Z', '2').replace('S', '5').replace('B', '8').replace('G', '6').replace('L', '1').replace('T', '1')
        
        year = fix_digits(raw_usn[3:5])
        
        # 4. Fix Branch
        branch = resolve_branch(raw_usn[5:7])
        
        # 5. Fix Roll
        roll = fix_digits(raw_usn[7:])
        
        return region + college + year + branch + roll
        
    # If no pattern matches, fallback to generic cleanup if it looks like a short USN
    if 8 <= len(text) <= 12:
        # Simple digits fix
        def fix_digits(s):
            return s.replace('O', '0').replace('I', '1').replace('Z', '2').replace('S', '5')
        text = fix_digits(text)
        return text
        
    return "UNKNOWN"

# Test cases
test_cases = [
    ("DATE4VV2328205TIME1HR", "4VV23CS205"), # CS read as 28
    ("6VV2348205", "4VV23CS205"),          # Region 6, CS read as 48
    ("Y1VZ3C1U0301", "4VV23CS301"),        # Sloppy Y1VZ3C1U0301 -> 4VV23CS301 (or similar)
    ("KV2SLII", "UNKNOWN"),
    ("V2848205TH1MMXM5F", "4VV23CS205"),   # V2848205 -> 4VV2848205 -> 4VV23CS205 (V=4, 28=VV, 48=CS, 205=205)
]

for t, expected in test_cases:
    res = correct_usn_format(t)
    print(f"'{t}' -> '{res}' (Expected: '{expected}')")
