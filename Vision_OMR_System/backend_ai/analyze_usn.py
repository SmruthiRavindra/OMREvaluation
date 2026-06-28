import cv2
import numpy as np
from core.localization import run_yolo_inference
from core.preprocess import preprocess_image

def analyze_sheet():
    # Read the preprocessed sheet
    img_path = "backend_ai/debug_output/preprocessed_sheet.png"
    img = cv2.imread(img_path)
    if img is None:
        print("Preprocessed sheet not found")
        return
        
    print(f"Loaded preprocessed sheet: {img.shape}")
    
    # Run YOLO inference
    detections = run_yolo_inference(img)
    print(f"Total detections: {len(detections)}")
    
    usn_detections = [d for d in detections if d.class_name == "usn"]
    print(f"USN detections: {len(usn_detections)}")
    
    if usn_detections:
        det = usn_detections[0]
        print(f"USN Box: x1={det.x1}, y1={det.y1}, x2={det.x2}, y2={det.y2}, conf={det.confidence}")
        # Crop USN box
        pad = 8
        h, w = img.shape[:2]
        rx1 = max(0, int(det.x1) - pad)
        ry1 = max(0, int(det.y1) - pad)
        rx2 = min(w, int(det.x2) + pad)
        ry2 = min(h, int(det.y2) + pad)
        
        usn_crop = img[ry1:ry2, rx1:rx2]
        cv2.imwrite("backend_ai/debug_output/usn_crop_test.png", usn_crop)
        print("Saved usn_crop_test.png")

if __name__ == "__main__":
    analyze_sheet()
