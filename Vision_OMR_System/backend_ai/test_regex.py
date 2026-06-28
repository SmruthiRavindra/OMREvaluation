import re

def correct_usn_format(text: str) -> str:
    # Remove whitespace and punctuation, uppercase
    text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    if not text or len(text) < 3:
        return "UNKNOWN"
    
    # 1. Broadly fix W -> VV before regex search
    text = text.replace('W', 'VV')
    
    match = re.search(r'([1-9I][A-Z]{2}[0-9OIZS]{2}[A-Z]{2}[0-9OIZS]{3})', text)
    
    if match:
        cleaned = match.group(1)
        def fix_digits(s):
            return s.replace('O', '0').replace('I', '1').replace('Z', '2').replace('S', '5')
        
        region = fix_digits(cleaned[0])
        college = cleaned[1:3]
        year = fix_digits(cleaned[3:5])
        branch = cleaned[5:7]
        roll = fix_digits(cleaned[7:])
        return region + college + year + branch + roll

    if len(text) <= 12:
        return text

    return "UNKNOWN"

test_cases = [
    ("A1DKATYO78CIGII1U162ISDDOMRA5CESHIGOAPOG963", "UNKNOWN"), # No VTU USN inside
    ("garbagetext4VV21IS042moregarbage", "4VV21IS042"),
    ("garbagetext4W21IS042moregarbage", "4VV21IS042"),
    ("garbagetext4VVZ1IS04Zmoregarbage", "4VV21IS042"), # Z -> 2
    ("garbagetext4VVO1ISO4Omoregarbage", "4VV01IS040"), # O -> 0
    ("garbagetext4W21CS001moregarbage", "4VV21CS001"),
    ("1RV20EC123", "1RV20EC123"),
    ("IRV20EC123", "1RV20EC123"), # I -> 1 in region
    ("4vv21is005", "4VV21IS005"),
    ("123", "123"), # short fallback
    ("4VV", "4VV"), # short fallback
]

for text, expected in test_cases:
    result = correct_usn_format(text)
    print(f"'{text}' -> '{result}' (Expected: '{expected}')")
    assert result == expected, f"Failed on {text}"
    
print("All tests passed!")
