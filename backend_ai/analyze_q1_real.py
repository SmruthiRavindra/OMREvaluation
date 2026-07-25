import cv2
import numpy as np

def analyze_q1_real():
    img = cv2.imread("backend_ai/debug_output/preprocessed_sheet.png")
    if img is None:
        print("preprocessed_sheet.png not found")
        return
        
    crops = {
        "Q1_A_unfilled": img[423:448, 177:205],
        "Q1_B_filled": img[424:450, 226:253],
        "Q1_C_unfilled": img[426:450, 274:300],
        "Q1_D_unfilled": img[428:452, 320:348],
    }
    
    for name, crop in crops.items():
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        min_val, max_val, _, _ = cv2.minMaxLoc(gray)
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Inner mask
        h, w = gray.shape
        cx, cy = w // 2, h // 2
        mask = np.zeros_like(gray)
        # Use 35% of minimum dimension as radius (excludes borders)
        r = int(min(w, h) * 0.35)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        
        # Mean inside mask
        mean_inner = np.mean(gray[mask > 0])
        std_inner = np.std(gray[mask > 0])
        min_inner = np.min(gray[mask > 0])
        max_inner = np.max(gray[mask > 0])
        
        # Otsu thresholding
        _, otsu_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu_masked = cv2.bitwise_and(otsu_bin, mask)
        otsu_fill_ratio = np.count_nonzero(otsu_masked) / np.count_nonzero(mask)
        
        # Global threshold (e.g. 120)
        _, abs_bin = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        abs_masked = cv2.bitwise_and(abs_bin, mask)
        abs_fill_ratio = np.count_nonzero(abs_masked) / np.count_nonzero(mask)
        
        print(f"{name}:")
        print(f"  Whole Crop: Min={min_val:.1f}, Max={max_val:.1f}, Mean={mean_val:.1f}, Std={std_val:.1f}")
        print(f"  Inner Mask: Min={min_inner:.1f}, Max={max_inner:.1f}, Mean={mean_inner:.1f}, Std={std_inner:.1f}")
        print(f"  Otsu Fill={otsu_fill_ratio:.3f}, Abs Fill (120)={abs_fill_ratio:.3f}")

if __name__ == "__main__":
    analyze_q1_real()
