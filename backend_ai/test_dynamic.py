import cv2
import numpy as np
import glob
import os

def test_dynamic_thresholding():
    # Load raw preprocessed sheet
    img = cv2.imread("backend_ai/debug_output/preprocessed_sheet.png")
    if img is None:
        print("preprocessed_sheet.png not found")
        return
        
    # Let's mock a set of bubble detections from actual coordinates
    # We will include the 4 Q1 bubbles and the 15 debug bubbles
    # First, let's extract the ROIs for the 15 debug bubbles
    debug_rois = []
    roi_paths = glob.glob("backend_ai/debug_output/bubble_*_roi.png")
    
    bubble_details = []
    
    # Process Q1 real bubbles
    q1_crops = {
        "Q1_A_unfilled": img[423:448, 177:205],
        "Q1_B_filled": img[424:450, 226:253],
        "Q1_C_unfilled": img[426:450, 274:300],
        "Q1_D_unfilled": img[428:452, 320:348],
    }
    
    for name, crop in q1_crops.items():
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cx, cy = w // 2, h // 2
        mask = np.zeros_like(gray)
        r = int(min(w, h) * 0.35)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        mean_inner = float(np.mean(gray[mask > 0]))
        std_inner = float(np.std(gray[mask > 0]))
        bubble_details.append((name, gray, mask, mean_inner, std_inner))
        
    # Process the 15 debug ROIs (all are filled)
    for path in sorted(roi_paths, key=lambda x: int(os.path.basename(x).split('_')[1])):
        crop = cv2.imread(path)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        size = gray.shape[0]
        center = size // 2
        mask = np.zeros((size, size), dtype=np.uint8)
        # Use fallback mask centered at 32 with radius 13
        cv2.circle(mask, (center, center), 13, 255, -1)
        mean_inner = float(np.mean(gray[mask > 0]))
        std_inner = float(np.std(gray[mask > 0]))
        bubble_details.append((os.path.basename(path), gray, mask, mean_inner, std_inner))
        
    # Get all means
    means = [b[3] for b in bubble_details]
    # The 95th percentile represents a typical unfilled bubble under the current lighting
    paper_ref = float(np.percentile(means, 95))
    filled_threshold = paper_ref - 45
    empty_threshold = paper_ref - 18
    
    print(f"Paper Reference: {paper_ref:.2f}")
    print(f"Filled Threshold: {filled_threshold:.2f}")
    print(f"Empty Threshold: {empty_threshold:.2f}\n")
    
    for name, gray, mask, mean_inner, std_inner in bubble_details:
        if mean_inner <= filled_threshold:
            if std_inner < 18:
                state = "FILLED"
            else:
                state = "AMBIGUOUS"
        elif mean_inner >= empty_threshold:
            state = "EMPTY"
        else:
            state = "AMBIGUOUS"
            
        print(f"{name:20s}: Mean={mean_inner:6.2f}, Std={std_inner:5.2f} -> Classified: {state}")

if __name__ == "__main__":
    test_dynamic_thresholding()
