import cv2
import numpy as np
import os
from main import run_yolo_inference, classify_all, map_bubbles_to_grid, _annotate_image
from core.preprocess import preprocess_image
from core.scoring import SheetLayout

# Decode raw bytes
with open("test_input_new.png", "rb") as f:
    contents = f.read()

# Preprocess
clean_img = preprocess_image(contents)
print(f"Clean image shape: {clean_img.shape}")

# YOLO
detections = run_yolo_inference(clean_img)
usn_dets = [d for d in detections if d.class_name == "usn"]
bubble_dets = [d for d in detections if d.class_name != "usn"]
print(f"USN detections: {len(usn_dets)}, Bubble detections: {len(bubble_dets)}")

# Classify
classifications = classify_all(clean_img, bubble_dets)
print(f"Classifications: {len(classifications)}")

layout = SheetLayout(
    questions_per_column=15,
    num_columns=2,
    options="ABCD",
)

usn_y2 = usn_dets[0].y2 if usn_dets else None
grid = map_bubbles_to_grid(classifications, layout, usn_y2)
valid_classifications = []
for q_num, opts in grid.items():
    for opt_letter, cr in opts.items():
        valid_classifications.append(cr)
print(f"Valid mapped bubbles: {len(valid_classifications)}")
