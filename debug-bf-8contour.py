import cv2
import numpy as np
import os
import glob

# --------------------------------------------------
# DEBUG FOLDER SETUP
# --------------------------------------------------
DEBUG_DIR = "debug"

# Create debug folder if not exist
os.makedirs(DEBUG_DIR, exist_ok=True)

# Clear existing files in debug folder at start
for f in glob.glob(os.path.join(DEBUG_DIR, "*")):
    try:
        os.remove(f)
    except OSError:
        pass

# Helper to save debug images into debug folder
def save_debug(name, img):
    path = os.path.join(DEBUG_DIR, name)
    cv2.imwrite(path, img)


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
path = r"source\12\12 N NO 1_ch00.tif"
# path = r"source\12\12 N NO 2_ch00.tif"
# path = r"source\12\12 N NO 3_ch00.tif"

# 1) Load & grayscale
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(path)

if img.ndim == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --------------------------------------------------
# STEP 1: Smooth background (top-hat style)
# --------------------------------------------------
bg = cv2.GaussianBlur(img, (0, 0), sigmaX=15, sigmaY=15)

# Emphasize darker blobs: background - image (bright where objects are dark)
enhanced = cv2.subtract(bg, img)

# Optional small blur to reduce noise
enhanced_blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

save_debug("enhanced.png", enhanced)
save_debug("enhanced_blur.png", enhanced_blur)

# --------------------------------------------------
# STEP 2: Threshold on enhanced image
# --------------------------------------------------
_, thresh = cv2.threshold(
    enhanced_blur, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Or use adaptive threshold instead of Otsu:
# thresh = cv2.adaptiveThreshold(
#     enhanced_blur, 255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
#     31,    # block size (odd) – tune
#     -5     # C – tune; more negative → more objects
# )

save_debug("thresh_raw.png", thresh)

# --------------------------------------------------
# STEP 3: Make each clump a single solid blob
# --------------------------------------------------
kernel = np.ones((3, 3), np.uint8)

# 3A) Close small gaps and bridge thin breaks
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

# Optional: slight smoothing of edges
closed = cv2.dilate(closed, kernel, iterations=1)
closed = cv2.erode(closed, kernel, iterations=1)

save_debug("closed.png", closed)

# 3B) Fill internal holes so clumps are solid inside
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    closed, connectivity=8
)

solid = np.where(labels > 0, 255, 0).astype(np.uint8)
save_debug("solid.png", solid)

# --------------------------------------------------
# STEP 4: Find & filter contours (ONE per clump)
# --------------------------------------------------
contours, hierarchy = cv2.findContours(
    solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
print("Number of clumps (solid blobs):", len(contours))

# Filter by area to ignore tiny blobs / noise
min_area = 100    # adjust
max_area = 5000   # adjust
filtered = [
    c for c in contours
    if min_area <= cv2.contourArea(c) <= max_area
]
print("Filtered blobs:", len(filtered))

# --------------------------------------------------
# STEP 5: Visualize
# --------------------------------------------------
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(vis, filtered, -1, (0, 0, 255), 1)

save_debug("contours_clumps_outer_only.png", vis)


# In[ ]:




