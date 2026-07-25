import re

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

def resolve_branch(branch_cand: str) -> str:
    branch_cand = branch_cand.upper()
    if branch_cand in ("U", "H"):
        return "LI"
        
    if branch_cand in VALID_BRANCHES:
        return branch_cand
        
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
                
    return "CS"

def correct_usn_format(text: str) -> str:
    text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    if not text:
        return "UNKNOWN"
        
    text = text.replace('W', 'VV')
    
    # If the text starts with VV or V followed by two digits (year), prepend 4
    if text.startswith('VV') and len(text) > 3 and text[2].isdigit():
        text = '4' + text
    elif text.startswith('V') and len(text) > 2 and text[1].isdigit():
        text = '4V' + text
    
    # Allow 1 or 2 characters for branch ([A-Z0-9]{1,2})
    pattern = r'([1-9IYLTAHG6V][V1LWC][V1LWC][0-9OIZS]{2}[A-Z0-9]{1,2}[0-9OIZSBLG]{3,4})'
    match = re.search(pattern, text)
    
    if match:
        raw_usn = match.group(1)
        
        # Region is always 4 for VV
        region = '4'
        college = "VV"
        
        def fix_digits(s):
            return s.replace('O', '0').replace('I', '1').replace('Z', '2').replace('S', '5').replace('B', '8').replace('G', '6').replace('L', '1').replace('T', '1')
        
        year = fix_digits(raw_usn[3:5])
        if year in ('28', '2B', '48', '18', 'Z8', 'S8', '88'):
            year = '23'
            
        # Dynamically split branch and roll based on length
        branch_roll_part = raw_usn[5:]
        n = len(branch_roll_part)
        
        if n >= 5:
            # e.g. "CS205" (5), "C1108" (5), "CI1008" (6)
            # The branch is always 2 characters, roll is the rest
            branch_cand = branch_roll_part[:2]
            roll_cand = branch_roll_part[2:]
        else:
            # n <= 4, e.g. "U608" (4), "U108" (4)
            # The branch is 1 character, roll is the rest
            branch_cand = branch_roll_part[:1]
            roll_cand = branch_roll_part[1:]
            
        branch = resolve_branch(branch_cand)
        roll = fix_digits(roll_cand)
        
        return region + college + year + branch + roll
        
    return "UNKNOWN"

# Test cases
test_cases = [
    ("4VV23CS205", "4VV23CS205"),
    ("W2328205", "4VV23CS205"),          # CS read as 28, W -> VV
    ("4VV23LI108", "4VV23LI108"),        # LI branch, 3 digits
    ("4VV23C1108", "4VV23C1108"),        # C1 branch, 3 digits
    ("4VV23CI1008", "4VV23CI1008"),      # CI branch, 4 digits
    ("Y1VV23L1108", "4VV23LI108"),       # L1 -> LI
    ("YVV23U608", "4VV23LI608"),         # U -> LI (merged LI handwriting)
    ("4VV23U108", "4VV23LI108"),         # U -> LI
]

for t, expected in test_cases:
    res = correct_usn_format(t)
    print(f"'{t}' -> '{res}' (Expected: '{expected}')")
