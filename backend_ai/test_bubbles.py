import cv2
import numpy as np
import glob
import os

def test_bubble_classification():
    roi_paths = glob.glob("backend_ai/debug_output/bubble_*_roi.png")
    print(f"Found {len(roi_paths)} bubble ROIs to test")
    
    for path in sorted(roi_paths, key=lambda x: int(os.path.basename(x).split('_')[1])):
        roi = cv2.imread(path)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate basic stats
        min_val, max_val, _, _ = cv2.minMaxLoc(gray)
        contrast = max_val - min_val
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # 1. Inner circular region extraction (shrunk to 60%)
        size = gray.shape[0]
        center = size // 2
        
        # Dynamic contour centering as in classification.py
        _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cx, cy, r = float(center), float(center), float(center * 0.7)
        best_contour = None
        min_dist = float('inf')
        for c in contours:
            (ccx, ccy), cr = cv2.minEnclosingCircle(c)
            if 12 <= cr <= 30:
                dist = np.sqrt((ccx - center) ** 2 + (ccy - center) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_contour = c
        if best_contour is not None:
            (cx, cy), r = cv2.minEnclosingCircle(best_contour)
            
        mask = np.zeros((size, size), dtype=np.uint8)
        inner_radius = max(5, int(r * 0.60)) # 60% of detected radius
        cv2.circle(mask, (int(cx), int(cy)), inner_radius, 255, -1)
        
        # Count non-zero mask pixels
        mask_pixels = np.count_nonzero(mask)
        
        # Otsu thresholding on gray
        _, otsu_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu_masked = cv2.bitwise_and(otsu_bin, mask)
        otsu_fill_ratio = np.count_nonzero(otsu_masked) / mask_pixels if mask_pixels > 0 else 0
        
        # Absolute thresholding (e.g. threshold = 127)
        _, abs_bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        abs_masked = cv2.bitwise_and(abs_bin, mask)
        abs_fill_ratio = np.count_nonzero(abs_masked) / mask_pixels if mask_pixels > 0 else 0
        
        # Print stats
        # Let's read the meta file if it exists
        meta_path = path.replace("_roi.png", "_meta.txt")
        meta_info = ""
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta_info = f.read().strip()
                
        print(f"Bubble {os.path.basename(path)}:")
        print(f"  Meta: {meta_info}")
        print(f"  Min={min_val:.1f}, Max={max_val:.1f}, Contrast={contrast:.1f}, Mean={mean_val:.1f}, Std={std_val:.1f}")
        print(f"  Otsu Fill={otsu_fill_ratio:.3f}, Abs Fill (127)={abs_fill_ratio:.3f}")

if __name__ == "__main__":
    test_bubble_classification()
