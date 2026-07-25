import cv2
from core.localization import run_yolo_inference

def test_low_conf():
    img = cv2.imread("backend_ai/debug_output/preprocessed_sheet.png")
    if img is None:
        print("preprocessed_sheet.png not found")
        return
        
    # Run YOLO with conf_threshold = 0.20
    detections = run_yolo_inference(img, conf_threshold=0.20)
    
    q5_dets = [d for d in detections if 590 <= d.y1 <= 650]
    q5_dets.sort(key=lambda d: d.x1)
    
    print(f"Detections in Q5 range with conf=0.20: {len(q5_dets)}")
    for d in q5_dets:
        print(f"class={d.class_name}, conf={d.confidence:.2f}, x1={d.x1:.1f}, y1={d.y1:.1f}, x2={d.x2:.1f}, y2={d.y2:.1f}")

if __name__ == "__main__":
    test_low_conf()
