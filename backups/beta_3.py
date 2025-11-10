import cv2
import numpy as np

# Load image
img = cv2.imread('./source/1/1 N NO 1_ch00.tif', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image could not be loaded. Check the file path.")

# GaussianBlur
blurred = cv2.GaussianBlur(img, (5, 5), 0)  # type: ignore

# Threshold
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background and foreground
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)  # type: ignore

# Markers
_, markers = cv2.connectedComponents(sure_fg)  # type: ignore
markers = markers + 1
markers[unknown == 255] = 0

# Watershed
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # type: ignore
markers = cv2.watershed(img_color, markers)
img_color[markers == -1] = [255, 0, 0]  # Blue outlines at boundaries

cv2.imwrite('watershed_outlined.png', img_color)