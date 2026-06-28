import cv2
import easyocr

img = cv2.imread("backend_ai/debug_output/usn_crop_test.png")
if img is None:
    print("usn_crop_test.png not found")
    exit(1)
    
h, w = img.shape[:2]
# Crop just the first 35% of the width
usn_only_width = int(w * 0.35)
usn_only_crop = img[:, :usn_only_width]

cv2.imwrite("backend_ai/debug_output/usn_only_crop.png", usn_only_crop)
print(f"Saved usn_only_crop.png: {usn_only_crop.shape}")

# Preprocess
roi_large = cv2.resize(usn_only_crop, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(roi_large, cv2.COLOR_BGR2GRAY)

# Standard CLAHE
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
enhanced_gray = clahe.apply(gray)
cv2.imwrite("backend_ai/debug_output/usn_only_enhanced.png", enhanced_gray)

# Run EasyOCR
reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext(enhanced_gray, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

print("\n--- OCR Results on 35% Crop ---")
for bbox, text, conf in results:
    print(f"Text: '{text}' (conf: {conf:.2f})")
