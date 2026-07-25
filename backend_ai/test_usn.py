import cv2
import sys
from core.usn_extraction import extract_usn_from_roi

def test_usn():
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print("Could not read image")
        return
    
    h, w = img.shape[:2]
    print(f"Image shape: {w}x{h}")
    
    # Let's say the USN box is roughly the top part of the image, we can just pass the whole image as ROI
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    print("EasyOCR on raw grayscale:")
    for b, t, c in results:
        print(f"  {t} (conf: {c})")
        
    usn = extract_usn_from_roi(img, 0, 0, w, h)
    print(f"Extracted USN: {usn}")

if __name__ == "__main__":
    test_usn()
