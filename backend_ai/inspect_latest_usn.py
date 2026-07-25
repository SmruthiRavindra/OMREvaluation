import cv2
import easyocr
from core.localization import run_yolo_inference

def inspect_latest_usn():
    img = cv2.imread("backend_ai/debug_output/preprocessed_sheet.png")
    if img is None:
        print("preprocessed_sheet.png not found")
        return
        
    detections = run_yolo_inference(img)
    usn_dets = [d for d in detections if d.class_name == "usn"]
    
    if not usn_dets:
        print("No USN box detected")
        return
        
    det = usn_dets[0]
    print(f"USN BBox: x1={det.x1:.1f}, y1={det.y1:.1f}, x2={det.x2:.1f}, y2={det.y2:.1f}, conf={det.confidence:.2f}")
    
    h, w = img.shape[:2]
    # Let's save both the 100% crop and the 42% crop
    box_w = det.x2 - det.x1
    pad = 8
    
    # 100% Crop
    rx1_100 = max(0, int(det.x1) - pad)
    rx2_100 = min(w, int(det.x2) + pad)
    ry1 = max(0, int(det.y1) - pad)
    ry2 = min(h, int(det.y2) + pad)
    crop_100 = img[ry1:ry2, rx1_100:rx2_100]
    cv2.imwrite("backend_ai/debug_output/latest_usn_crop_100.png", crop_100)
    print("Saved latest_usn_crop_100.png")
    
    # 42% Crop
    rx2_42 = min(w, int(det.x1 + box_w * 0.42) + pad)
    crop_42 = img[ry1:ry2, rx1_100:rx2_42]
    cv2.imwrite("backend_ai/debug_output/latest_usn_crop_42.png", crop_42)
    print("Saved latest_usn_crop_42.png")
    
    # Run EasyOCR on both
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Let's try raw 42% crop, resized 42% crop, and different grayscale versions
    for name, cr in [("42% Crop Raw", crop_42), ("100% Crop Raw", crop_100)]:
        print(f"\n--- {name} ---")
        res = reader.readtext(cr, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        for bbox, text, conf in res:
            print(f"  OCR: '{text}' (conf: {conf:.2f})")
            
if __name__ == "__main__":
    inspect_latest_usn()
