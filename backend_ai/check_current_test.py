import cv2
from main import run_yolo_inference
import os

img_path = "debug_output/preprocessed_sheet.png"
if not os.path.exists(img_path):
    print("Error: debug_output/preprocessed_sheet.png does not exist")
    exit()

img = cv2.imread(img_path)
print(f"Loaded image size: {img.shape}")

dets = run_yolo_inference(img)
print(f"YOLO detected {len(dets)} items.")
for idx, d in enumerate(dets[:10]):
    print(f"  Det {idx}: class={d.class_name}, conf={d.confidence:.3f}, bbox={d.x1},{d.y1},{d.x2},{d.y2}")
