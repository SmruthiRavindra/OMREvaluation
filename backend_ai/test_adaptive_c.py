import cv2
import numpy as np

def test_adaptive_c():
    img = cv2.imread("backend_ai/debug_output/preprocessed_sheet.png")
    if img is None:
        print("preprocessed_sheet.png not found")
        return
        
    crops = {
        "Q1_A_unfilled": img[423:448, 177:205],
        "Q1_B_filled": img[424:450, 226:253],
        "Q1_C_unfilled": img[426:450, 274:300],
        "Q1_D_unfilled": img[428:452, 320:348],
        # Let's crop Q5 A (filled) and Q5 B (unfilled) from the image if we can find them
        # Q5 should be around y = 424 + 4*48 = 616
        "Q5_A_filled": img[610:635, 175:203],
        "Q5_B_unfilled": img[610:635, 224:252]
    }
    
    for name, crop in crops.items():
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cx, cy = w // 2, h // 2
        mask = np.zeros_like(gray)
        r = int(min(w, h) * 0.35)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        mask_pixels = np.count_nonzero(mask)
        
        print(f"\n{name}:")
        for c_val in [8, 12, 16, 20]:
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, c_val)
            masked = cv2.bitwise_and(adaptive, mask)
            fill_ratio = np.count_nonzero(masked) / mask_pixels if mask_pixels > 0 else 0
            print(f"  C={c_val:2d} -> Fill Ratio: {fill_ratio:.3f}")

if __name__ == "__main__":
    test_adaptive_c()
