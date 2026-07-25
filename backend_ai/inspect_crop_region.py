import cv2
import numpy as np

img = cv2.imread("test_input_new.png")
if img is None:
    print("Error: test_input_new.png not found")
    exit()

# Crop region
crop = img[73:803, 722:1606]
cv2.imwrite("debug_output/region_crop.png", crop)
print(f"Saved region_crop.png with shape {crop.shape}")

# Print std dev of gray version
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
print(f"Crop gray std: {gray.std()}")
