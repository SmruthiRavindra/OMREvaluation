import cv2
import numpy as np

def analyze_q1():
    img = cv2.imread("backend_ai/debug_output/preprocessed_sheet.png")
    if img is None:
        print("preprocessed_sheet.png not found")
        return
        
    # We will crop Q1 A (unfilled), Q1 B (filled), Q1 C (unfilled), Q1 D (unfilled)
    # Based on coordinates:
    # Q1 y is around 260 to 290.
    # A x is around 200 to 225.
    # B x is around 250 to 275.
    # C x is around 300 to 325.
    # D x is around 350 to 375.
    
    crops = {
        "Q1_A_unfilled": img[260:295, 195:230],
        "Q1_B_filled": img[260:295, 245:280],
        "Q1_C_unfilled": img[260:295, 295:330],
        "Q1_D_unfilled": img[260:295, 345:380],
    }
    
    for name, crop in crops.items():
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        min_val, max_val, _, _ = cv2.minMaxLoc(gray)
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        cv2.imwrite(f"backend_ai/debug_output/{name}.png", crop)
        print(f"{name}:")
        print(f"  Min={min_val:.1f}, Max={max_val:.1f}, Mean={mean_val:.1f}, Std={std_val:.1f}, Contrast={max_val-min_val:.1f}")

if __name__ == "__main__":
    analyze_q1()
