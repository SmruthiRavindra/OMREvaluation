import cv2
import easyocr
import re

def test_ocr_processing():
    img = cv2.imread("backend_ai/debug_output/usn_crop_test.png")
    if img is None:
        print("usn_crop_test.png not found")
        return
        
    h, w = img.shape[:2]
    # Preprocess
    roi_large = cv2.resize(img, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi_large, cv2.COLOR_BGR2GRAY)
    
    # Try different enhancements
    # 1. Standard CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    cv2.imwrite("backend_ai/debug_output/usn_enhanced.png", enhanced_gray)
    
    # 2. Otsu thresholding on enhanced
    blurred = cv2.GaussianBlur(enhanced_gray, (3, 3), 0)
    _, thresh_enhanced = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("backend_ai/debug_output/usn_thresh_enhanced.png", thresh_enhanced)
    
    # 3. Adaptive Thresholding directly on gray
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    cv2.imwrite("backend_ai/debug_output/usn_adaptive.png", adaptive)

    # 4. Global thresholding on original gray
    _, thresh_global = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("backend_ai/debug_output/usn_thresh_global.png", thresh_global)
    
    # Run EasyOCR
    reader = easyocr.Reader(['en'], gpu=False)
    
    methods = {
        "enhanced_gray": enhanced_gray,
        "thresh_enhanced": thresh_enhanced,
        "adaptive": adaptive,
        "thresh_global": thresh_global,
        "raw_gray": gray
    }
    
    for name, processed_img in methods.items():
        print(f"\n--- Method: {name} ---")
        results = reader.readtext(processed_img, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
        for bbox, text, conf in results:
            print(f"  Text: '{text}' (conf: {conf:.2f})")

if __name__ == "__main__":
    test_ocr_processing()
