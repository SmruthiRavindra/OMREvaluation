import cv2
import easyocr

def test_raw_vs_enhanced():
    img = cv2.imread("backend_ai/debug_output/preprocessed_sheet.png")
    if img is None:
        print("preprocessed_sheet.png not found")
        return
        
    x1, y1, x2, y2 = 44.6, 386.5, 869.9, 437.2
    box_w = x2 - x1
    x2_cropped = x1 + (box_w * 0.42)
    pad = 8
    rx1 = max(0, int(x1) - pad)
    ry1 = max(0, int(y1) - pad)
    rx2 = min(img.shape[1], int(x2_cropped) + pad)
    ry2 = min(img.shape[0], int(y2) + pad)
    
    roi = img[ry1:ry2, rx1:rx2]
    
    reader = easyocr.Reader(['en'], gpu=False)
    
    # 1. Raw ROI
    print("\n--- 1. Raw ROI (BGR) ---")
    res = reader.readtext(roi, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for b, t, c in res:
        print(f"  OCR: '{t}' (conf: {c:.2f})")
        
    # 2. Resized ROI (BGR) 2x
    print("\n--- 2. Resized 2x (BGR) ---")
    roi_large = cv2.resize(roi, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    res = reader.readtext(roi_large, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for b, t, c in res:
        print(f"  OCR: '{t}' (conf: {c:.2f})")
        
    # 3. Resized 2x Gray (No CLAHE, no thresh)
    print("\n--- 3. Resized 2x Gray ---")
    gray = cv2.cvtColor(roi_large, cv2.COLOR_BGR2GRAY)
    res = reader.readtext(gray, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for b, t, c in res:
        print(f"  OCR: '{t}' (conf: {c:.2f})")

if __name__ == "__main__":
    test_raw_vs_enhanced()
