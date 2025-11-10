import cv2
import numpy as np

# Load image (replace 'your_image.png' with your file)
img = cv2.imread('./source/1/1 N NO 1_ch00.tif', cv2.IMREAD_GRAYSCALE)

# Preprocess: Blur to reduce noise
if img is None:
	raise FileNotFoundError("Image not found. Please check the file path.")
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Thresholding (Otsu's for automatic threshold)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw outlines on original image
outlined = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to color for red outlines
cv2.drawContours(outlined, contours, -1, (0, 0, 255), 2)  # Red outlines, thickness 2

# Save or display
cv2.imwrite('outlined_bacteria.png', outlined)
cv2.imshow('Outlined Bacteria', outlined)
cv2.waitKey(0)
cv2.destroyAllWindows()