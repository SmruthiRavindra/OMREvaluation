import time
import cv2
from pathlib import Path

# Set up path and imports
import sys
sys.path.insert(0, str(Path('c:/Users/smrut/OneDrive/Desktop/OMREvaluation/Vision_OMR_System/backend_ai')))

from core.preprocess import preprocess_image
from core.localization import run_yolo_inference
from core.classification import classify_all
from core import extract_usn_from_roi
from core.scoring import score_sheet, SheetLayout

image_path = Path('c:/Users/smrut/OneDrive/Desktop/OMREvaluation/Vision_OMR_System/backend_ai/debug_output/preprocessed_sheet.png')
with open(image_path, "rb") as f:
    contents = f.read()

# Warm up models first
print("Warming up models...")
clean_img = preprocess_image(contents)
detections = run_yolo_inference(clean_img)
usn_dets = [d for d in detections if d.class_name == "usn"]
bubble_dets = [d for d in detections if d.class_name != "usn"]
if usn_dets:
    extract_usn_from_roi(clean_img, usn_dets[0].x1, usn_dets[0].y1, usn_dets[0].x2, usn_dets[0].y2)
classify_all(clean_img, bubble_dets)

print("\n--- Benchmarking Pipeline ---")

t0 = time.perf_counter()
clean_img = preprocess_image(contents)
t1 = time.perf_counter()
print(f"1. Preprocessing: {t1-t0:.4f}s")

t_yolo_start = time.perf_counter()
detections = run_yolo_inference(clean_img)
t_yolo_end = time.perf_counter()
print(f"2. YOLO Inference + NMS: {t_yolo_end-t_yolo_start:.4f}s")

usn_dets = [d for d in detections if d.class_name == "usn"]
bubble_dets = [d for d in detections if d.class_name != "usn"]

print(f"   Detections count: total={len(detections)}, bubbles={len(bubble_dets)}, usn={len(usn_dets)}")

t_usn_start = time.perf_counter()
usn_value = None
if usn_dets:
    usn_value = extract_usn_from_roi(clean_img, usn_dets[0].x1, usn_dets[0].y1, usn_dets[0].x2, usn_dets[0].y2)
t_usn_end = time.perf_counter()
print(f"3. USN OCR Extraction: {t_usn_end-t_usn_start:.4f}s (Result: {usn_value})")

t_classify_start = time.perf_counter()
classifications = classify_all(clean_img, bubble_dets)
t_classify_end = time.perf_counter()
print(f"4. Bubble Classification (Secondary Spot Checker): {t_classify_end-t_classify_start:.4f}s")

t_scoring_start = time.perf_counter()
layout = SheetLayout(questions_per_column=15, num_columns=2, options="ABCD")
report = score_sheet(classifications, None, layout)
t_scoring_end = time.perf_counter()
print(f"5. Scoring / Mapping: {t_scoring_end-t_scoring_start:.4f}s")

total_time = time.perf_counter() - t0
print(f"\nTotal Pipeline Time (Warm): {total_time:.4f}s")
