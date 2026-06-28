import cv2
from core.localization import run_yolo_inference

def find_bubble_coords():
    img = cv2.imread("backend_ai/debug_output/preprocessed_sheet.png")
    if img is None:
        print("preprocessed_sheet.png not found")
        return
        
    detections = run_yolo_inference(img)
    # Filter only bubble detections
    bubble_dets = [d for d in detections if d.class_name != "usn"]
    
    # Sort by y1, then x1
    bubble_dets.sort(key=lambda d: (d.y1, d.x1))
    
    print(f"Total bubble detections: {len(bubble_dets)}")
    for i, d in enumerate(bubble_dets[:20]):
        print(f"Det {i}: class={d.class_name}, conf={d.confidence:.2f}, x1={d.x1:.1f}, y1={d.y1:.1f}, x2={d.x2:.1f}, y2={d.y2:.1f}")

if __name__ == "__main__":
    find_bubble_coords()
