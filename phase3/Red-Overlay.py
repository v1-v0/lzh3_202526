import cv2
import numpy as np
import os
import sys
import shutil

# ==========================================
# 1. CONFIGURATION
# ==========================================
PATH_GREY_IMAGE = "source\\12\\12 N NO 1_ch00.tif"
PATH_RED_IMAGE  = "source\\12\\12 N NO 1_ch01.tif"
PATH_METADATA_XML = "source\\12\\MetaData\\12 N NO 1_Properties.xml"
DEBUG_DIR = "debug"

# ==========================================
# 2. CLEAR DEBUG FOLDER
# ==========================================
print(f"[INFO] Clearing '{DEBUG_DIR}' folder...")

if os.path.exists(DEBUG_DIR):
    try:
        shutil.rmtree(DEBUG_DIR)  # Deletes the folder and all contents
    except OSError as e:
        print(f"[WARNING] Could not delete debug folder: {e}")

# Recreate the empty folder
os.makedirs(DEBUG_DIR, exist_ok=True)

# ==========================================
# 3. LOAD IMAGES
# ==========================================
print("[INFO] Loading images...")

img_grey = cv2.imread(PATH_GREY_IMAGE, cv2.IMREAD_GRAYSCALE)
img_red_raw = cv2.imread(PATH_RED_IMAGE, cv2.IMREAD_UNCHANGED)

if img_grey is None or img_red_raw is None:
    print("[ERROR] One or more images could not be loaded.")
    sys.exit(1)

# Handle Red Channel (if image is already RGB, take the Red channel)
if len(img_red_raw.shape) == 3:
    img_red = img_red_raw[:, :, 2]
else:
    img_red = img_red_raw

# ==========================================
# 4. PREPARE LAYERS
# ==========================================

# --- FIX FOR PYLANCE ERROR ---
# Instead of passing None, we create an empty array of the same size/type first.
# This satisfies the type checker.
img_red_norm = np.zeros_like(img_red)
cv2.normalize(img_red, img_red_norm, 0, 255, cv2.NORM_MINMAX)
img_red_norm = img_red_norm.astype(np.uint8)

# Convert Brightfield (Grey) to 3-Channel BGR
img_grey_bgr = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)

# Create a Color Layer for Red (Blue=0, Green=0, Red=Value)
zeros = np.zeros_like(img_grey)
img_red_layer = cv2.merge([zeros, zeros, img_red_norm])

# ==========================================
# 5. GENERATE COMPOSITES
# ==========================================

# --- OPTION A: Standard Merge (Grey + Red) ---
# Natural microscope look.
overlay_standard = cv2.addWeighted(img_grey_bgr, 1.0, img_red_layer, 0.8, 0)

# --- OPTION B: Darkened Background (High Contrast) ---
# Darken Brightfield by 50% to make Red pop.
img_grey_dark = (img_grey_bgr * 0.5).astype(np.uint8)
overlay_contrast = cv2.addWeighted(img_grey_dark, 1.0, img_red_layer, 1.0, 0)

# ==========================================
# 6. SAVE RESULTS
# ==========================================
print("[INFO] Saving visualizations...")

cv2.imwrite(os.path.join(DEBUG_DIR, "vis_standard_merge.png"), overlay_standard)
cv2.imwrite(os.path.join(DEBUG_DIR, "vis_high_contrast.png"), overlay_contrast)

print(f"[SUCCESS] Process complete. Check '{DEBUG_DIR}/' for output.")