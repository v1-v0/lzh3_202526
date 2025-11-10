import cv2
import numpy as np

img = cv2.imread('./source/1/1 N NO 1_ch00.tif', cv2.IMREAD_GRAYSCALE)
if img is None:
	raise FileNotFoundError("Image not found or path is incorrect.")

# Preprocess: CLAHE for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(img)

# Edge detection with Canny
edges = cv2.Canny(enhanced, 50, 150)  # Adjust thresholds based on image

# Optional: Dilate to connect edges
kernel = np.ones((3,3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

# Find and draw contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
outlined = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(outlined, contours, -1, (0, 255, 0), 1)  # Green outlines

cv2.imwrite('edges_outlined.png', outlined)