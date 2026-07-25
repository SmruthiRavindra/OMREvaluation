import cv2
from core.usn_extraction import extract_usn_from_roi

def test_extract():
    # We will load the original preprocessed sheet and the USN box coordinates
    img = cv2.imread("backend_ai/debug_output/preprocessed_sheet.png")
    if img is None:
        print("preprocessed_sheet.png not found")
        return
        
    # The USN box coordinates from inspect_latest_usn.py:
    # x1=44.6, y1=386.5, x2=869.9, y2=437.2
    x1, y1, x2, y2 = 44.6, 386.5, 869.9, 437.2
    
    usn = extract_usn_from_roi(img, x1, y1, x2, y2)
    print(f"\nFinal Extracted USN: '{usn}'")

if __name__ == "__main__":
    test_extract()
