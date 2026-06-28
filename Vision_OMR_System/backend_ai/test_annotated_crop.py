import cv2
import easyocr

image_path = r'c:\Users\smrut\.gemini\antigravity-ide\brain\e93f1af1-7dfd-4d5e-b18c-41a7b55142a4\media__1782556200462.png'
img = cv2.imread(image_path)
h, w = img.shape[:2]

# The USN box in the annotated image is approximately:
# y1 = 110, y2 = 180 (header area)
# x1 = 70, x2 = 950
crop = img[110:180, 70:950]

cv2.imwrite("temp_crop.png", crop)

reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext(crop, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

print("OCR Results on Crop:")
for bbox, text, conf in results:
    print(f"Text: '{text}' (Conf: {conf})")
