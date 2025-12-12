import cv2
import numpy as np
import os
import glob

DEBUG_DIR = "debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

for f in glob.glob(os.path.join(DEBUG_DIR, "*")):
    try:
        os.remove(f)
    except OSError:
        pass

def save_debug(name, img):
    cv2.imwrite(os.path.join(DEBUG_DIR, name), img)

# Load image
path = r"source\10\10 P 1_ch00.tif"
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(path)

if img.ndim == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"Loaded: {img.dtype}, {img.shape}, [{img.min()}-{img.max()}]")

# Convert to 8-bit
if img.dtype == np.uint16:
    img_8bit = np.zeros_like(img, dtype=np.uint8)
    cv2.normalize(img, img_8bit, 0, 255, cv2.NORM_MINMAX)
    img = img_8bit
    save_debug("01_converted_8bit.png", img)
    print(f"Converted: {img.dtype}, [{img.min()}-{img.max()}]")

# Background subtraction
bg = cv2.GaussianBlur(img, (0, 0), sigmaX=15, sigmaY=15)
enhanced = cv2.subtract(bg, img)
enhanced_blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

save_debug("02_enhanced.png", enhanced)
save_debug("03_enhanced_blur.png", enhanced_blur)

# Threshold
_, thresh = cv2.threshold(enhanced_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
save_debug("04_thresh_raw.png", thresh)

# Morphology
kernel = np.ones((3, 3), np.uint8)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
closed = cv2.dilate(closed, kernel, iterations=1)
closed = cv2.erode(closed, kernel, iterations=1)
save_debug("05_closed.png", closed)

# Fill holes
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
solid = np.where(labels > 0, 255, 0).astype(np.uint8)
save_debug("06_solid.png", solid)

# Find contours
contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Clumps found: {len(contours)}")

# Filter by area
filtered = [c for c in contours if 100 <= cv2.contourArea(c) <= 5000]
print(f"Filtered objects: {len(filtered)}")

# Visualize
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(vis, filtered, -1, (0, 0, 255), 1)
'''
cv2.putText(vis, f"Objects: {len(filtered)}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
'''
save_debug("07_contours_final.png", vis)
print("✓ Processing complete")

# --------------------------------------------------
# Export Statistics to CSV
# --------------------------------------------------
import csv

stats_file = os.path.join(DEBUG_DIR, "object_stats.csv")

with open(stats_file, 'w', newline='') as f:
    writer = csv.writer(f)

    writer.writerow(['Object_ID', 'Area', 'Perimeter'])


    for i, c in enumerate(filtered, 1):
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        
        # Calculate circularity (1.0 = perfect circle)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Calculate bounding box and aspect ratio
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Calculate centroid
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        writer.writerow([i, f"{area:.1f}", f"{perimeter:.1f}"])

print(f"CSV saved: {stats_file} ({len(filtered)} objects)")