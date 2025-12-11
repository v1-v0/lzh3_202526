#!/usr/bin/env python
# coding: utf-8

# In[41]:


import cv2
import numpy as np
import os
import sys
import shutil
import xml.etree.ElementTree as ET

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "IMAGE_PATH": r"source\12\12 N NO 1_ch00.tif", 
    "METADATA_XML": r"source\12\MetaData\12 N NO 1_Properties.xml",
    "DEBUG_DIR": "debug_segmentation",

    "DEFAULT_PIXEL_SIZE_UM": 0.11,
    "MIN_AREA_PX": 50,
    "MAX_AREA_PX": 10000,
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def setup_debug_dir(directory):
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
        except OSError:
            pass
    os.makedirs(directory, exist_ok=True)
    print(f"[INFO] Debug directory ready: {directory}")

def get_pixel_scale(xml_path):
    if not os.path.exists(xml_path):
        return None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        dimensions = root.find(".//ImageDescription/Dimensions")
        if dimensions is not None:
            for dim in dimensions.findall("DimensionDescription"):
                if dim.get("DimID") in ["X", "1"]:
                    voxel_val = dim.get("Voxel")
                    if voxel_val:
                        scale_x = float(voxel_val)
                        unit = dim.get("Unit", "um")
                        if unit == "m": scale_x *= 1e6
                        return scale_x
    except Exception as e:
        print(f"[ERROR] XML parsing error: {e}")
    return None

def normalize_to_8bit(img):
    """Safely converts an image to 8-bit."""
    if img is None:
        return np.zeros((100, 100), dtype=np.uint8) # Fallback for type safety

    if img.dtype == np.uint8: 
        return img

    img_float = img.astype(np.float32)
    img_norm = np.zeros(img.shape, dtype=np.uint8)
    cv2.normalize(img_float, img_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img_norm

# ==========================================
# 3. SEGMENTATION PIPELINE
# ==========================================

def preprocess_for_dark_objects(img):
    # Background estimation
    bg = cv2.GaussianBlur(img, (0, 0), sigmaX=15, sigmaY=15)
    # Subtract image from background (Dark objects become bright signal)
    enhanced = cv2.subtract(bg, img)
    # Denoise
    enhanced_blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return enhanced, enhanced_blur

def segment_and_measure(enhanced_img, pixel_size_um):
    # Otsu Thresholding
    _, thresh = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find Contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    valid_contours = []
    pixel_area_um2 = pixel_size_um ** 2
    count_id = 1

    # Create a clean mask of only valid objects
    final_mask = np.zeros_like(closed)

    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if not (CONFIG["MIN_AREA_PX"] <= area_px <= CONFIG["MAX_AREA_PX"]):
            continue

        valid_contours.append(cnt)
        cv2.drawContours(final_mask, [cnt], -1, 255, -1)

        perimeter_px = cv2.arcLength(cnt, closed=True)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x,y,w,h = cv2.boundingRect(cnt)
            cx, cy = x+w//2, y+h//2

        results.append({
            "id": count_id,
            "area_um2": area_px * pixel_area_um2,
            "perimeter_um": perimeter_px * pixel_size_um,
            "centroid": (cx, cy)
        })
        count_id += 1

    return valid_contours, results, final_mask

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    setup_debug_dir(CONFIG["DEBUG_DIR"])

    # 1. Load Calibration
    px_size = get_pixel_scale(CONFIG["METADATA_XML"]) or CONFIG["DEFAULT_PIXEL_SIZE_UM"]
    print(f"[INFO] Pixel size: {px_size:.4f} µm/px")

    # 2. Load Image
    if not os.path.exists(CONFIG["IMAGE_PATH"]):
        print(f"[ERROR] Image path does not exist: {CONFIG['IMAGE_PATH']}")
        return

    img_raw = cv2.imread(CONFIG["IMAGE_PATH"], cv2.IMREAD_UNCHANGED)

    # --- FIX 1: Explicit Check for None ---
    if img_raw is None:
        print(f"[ERROR] Failed to load image data from {CONFIG['IMAGE_PATH']}")
        return
    # --------------------------------------

    if len(img_raw.shape) == 3:
        img_grey = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    else:
        img_grey = img_raw

    # 3. Process
    enhanced, enhanced_blur = preprocess_for_dark_objects(img_grey)

    # 4. Analyze
    valid_contours, data, mask = segment_and_measure(enhanced_blur, px_size)
    print(f"[INFO] Found {len(data)} valid objects.")

    # ==========================================
    # 5. VISUALIZATION (Gradient Red Overlay)
    # ==========================================

    # Base image converted to BGR
    # normalize_to_8bit is now guaranteed to return a valid array
    base_img = cv2.cvtColor(normalize_to_8bit(img_grey), cv2.COLOR_GRAY2BGR)

    # A. Create the Gradient Signal
    signal_masked = cv2.bitwise_and(enhanced, enhanced, mask=mask)

    # --- FIX 2: Type-Safe Normalize ---
    # Initialize the destination array first
    signal_norm = np.zeros_like(signal_masked)
    cv2.normalize(signal_masked, signal_norm, 0, 255, cv2.NORM_MINMAX)
    # ----------------------------------

    # Apply a Red Colormap manually (Black -> Red)
    red_gradient = np.zeros_like(base_img)
    red_gradient[:, :, 2] = signal_norm 

    # B. Blend
    vis_img = base_img.copy()
    base_roi = cv2.bitwise_and(base_img, base_img, mask=mask)

    # Blend the ROI with the Red Gradient
    blended_roi = cv2.addWeighted(base_roi, 1.0, red_gradient, 0.8, 0)

    # Put the blended ROI back into the main image
    mask_inv = cv2.bitwise_not(mask)
    bg_part = cv2.bitwise_and(vis_img, vis_img, mask=mask_inv)
    vis_img = cv2.add(bg_part, blended_roi)

    # C. Draw Solid Outlines and Labels
    for i, cnt in enumerate(valid_contours):
        item = data[i]
        cv2.drawContours(vis_img, [cnt], -1, (0, 255, 0), 1)
        cx, cy = item['centroid']
        cv2.putText(vis_img, str(item['id']), (cx - 10, cy - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # 6. Save Outputs
    cv2.imwrite(os.path.join(CONFIG["DEBUG_DIR"], "01_enhanced_signal.png"), normalize_to_8bit(enhanced))
    cv2.imwrite(os.path.join(CONFIG["DEBUG_DIR"], "02_binary_mask.png"), mask)
    cv2.imwrite(os.path.join(CONFIG["DEBUG_DIR"], "03_result_overlay.png"), vis_img)

    # 7. Print Table
    print("\n" + "="*50)
    print(f"{'ID':<5} {'Area (µm²)':<15} {'Perimeter (µm)':<15}")
    print("="*50)
    for d in data:
        print(f"{d['id']:<5} {d['area_um2']:<15.2f} {d['perimeter_um']:<15.2f}")
    print("="*50)

if __name__ == "__main__":
    main()


# In[45]:


import cv2
import numpy as np
import os
import sys
import shutil
import xml.etree.ElementTree as ET

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    # Structural Image (for finding shapes)
    "IMAGE_PATH": r"source\12\12 N NO 1_ch00.tif", 

    # Fluorescence Image (for measuring intensity)
    "PATH_RED_IMAGE": r"source\12\12 N NO 1_ch01.tif",

    "METADATA_XML": r"source\12\MetaData\12 N NO 1_Properties.xml",
    "DEBUG_DIR": "debug_segmentation",

    "DEFAULT_PIXEL_SIZE_UM": 0.11,
    "MIN_AREA_PX": 50,
    "MAX_AREA_PX": 10000,
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def setup_debug_dir(directory):
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
        except OSError:
            pass
    os.makedirs(directory, exist_ok=True)
    print(f"[INFO] Debug directory ready: {directory}")

def get_pixel_scale(xml_path):
    if not os.path.exists(xml_path):
        return None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        dimensions = root.find(".//ImageDescription/Dimensions")
        if dimensions is not None:
            for dim in dimensions.findall("DimensionDescription"):
                if dim.get("DimID") in ["X", "1"]:
                    voxel_val = dim.get("Voxel")
                    if voxel_val:
                        scale_x = float(voxel_val)
                        unit = dim.get("Unit", "um")
                        if unit == "m": scale_x *= 1e6
                        return scale_x
    except Exception as e:
        print(f"[ERROR] XML parsing error: {e}")
    return None

def normalize_to_8bit(img):
    if img is None:
        return np.zeros((100, 100), dtype=np.uint8)
    if img.dtype == np.uint8: 
        return img
    img_float = img.astype(np.float32)
    img_norm = np.zeros(img.shape, dtype=np.uint8)
    cv2.normalize(img_float, img_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img_norm

def load_fluorescence_channel(path):
    """Loads the red channel or grayscale intensity image."""
    if not os.path.exists(path):
        print(f"[WARNING] Fluorescence image not found: {path}")
        return None

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # If RGB, pick Red channel (index 2 in BGR)
    if len(img.shape) == 3:
        b, g, r = cv2.split(img)
        # Simple heuristic: if R is dominant or it's just a container for data
        return r

    return img

# ==========================================
# 3. SEGMENTATION & MEASUREMENT
# ==========================================

def preprocess_for_dark_objects(img):
    bg = cv2.GaussianBlur(img, (0, 0), sigmaX=15, sigmaY=15)
    enhanced = cv2.subtract(bg, img)
    enhanced_blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return enhanced, enhanced_blur

def segment_and_measure(structural_img, fluo_img, pixel_size_um):
    """
    Finds contours in 'structural_img', then measures intensity 
    inside those contours using 'fluo_img'.

    Filters out objects with 0 total intensity and ensures IDs are sequential.
    """
    # 1. Thresholding on Structural Image
    _, thresh = cv2.threshold(structural_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    valid_contours = []
    pixel_area_um2 = pixel_size_um ** 2

    # Initialize ID counter
    count_id = 1
    final_mask = np.zeros_like(closed)

    for cnt in contours:
        # --- 1. Area Filter ---
        area_px = cv2.contourArea(cnt)
        if not (CONFIG["MIN_AREA_PX"] <= area_px <= CONFIG["MAX_AREA_PX"]):
            continue

        # --- 2. Measure Intensity ---
        # Draw filled contour on a temporary mask to extract pixel values
        single_obj_mask = np.zeros_like(closed)
        cv2.drawContours(single_obj_mask, [cnt], -1, 255, -1)

        # Use meanStdDev for type safety (returns numpy array)
        mean_val, _ = cv2.meanStdDev(fluo_img, mask=single_obj_mask)
        mean_intensity = mean_val[0][0]

        # Calculate Total Intensity (Integrated Density)
        total_intensity = mean_intensity * area_px

        # --- 3. Intensity Filter ---
        # Only proceed if the object actually has fluorescence signal
        if total_intensity <= 0:
            continue

        # --- 4. Store Valid Object ---
        # If we reached here, the object is valid.
        valid_contours.append(cnt)

        # Add to the visual mask
        cv2.drawContours(final_mask, [cnt], -1, 255, -1)

        # Calculate Centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x,y,w,h = cv2.boundingRect(cnt)
            cx, cy = x+w//2, y+h//2

        results.append({
            "id": count_id,  # This will now be 1, 2, 3... sequentially
            "area_um2": area_px * pixel_area_um2,
            "mean_intensity": mean_intensity,
            "total_intensity": total_intensity,
            "centroid": (cx, cy)
        })

        # Only increment ID if we actually added a result
        count_id += 1

    return valid_contours, results, final_mask






def main():
    setup_debug_dir(CONFIG["DEBUG_DIR"])

    # 1. Load Calibration
    px_size = get_pixel_scale(CONFIG["METADATA_XML"]) or CONFIG["DEFAULT_PIXEL_SIZE_UM"]
    print(f"[INFO] Pixel size: {px_size:.4f} µm/px")

    # 2. Load Structural Image (ch00)
    img_raw = cv2.imread(CONFIG["IMAGE_PATH"], cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        print(f"[ERROR] Failed to load structural image: {CONFIG['IMAGE_PATH']}")
        return
    img_grey = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY) if len(img_raw.shape) == 3 else img_raw

    # 3. Load Fluorescence Image (ch01)
    fluo_raw = load_fluorescence_channel(CONFIG["PATH_RED_IMAGE"])
    if fluo_raw is None:
        print("[ERROR] Could not load fluorescence image.")
        return

    # Ensure dimensions match
    if fluo_raw.shape != img_grey.shape:
        fluo_raw = cv2.resize(fluo_raw, (img_grey.shape[1], img_grey.shape[0]))

    # 4. Process Structural Image
    enhanced, enhanced_blur = preprocess_for_dark_objects(img_grey)

    # 5. Analyze (Pass both structural and fluo images)
    # We pass 'fluo_raw' (not normalized) to get accurate scientific intensity values
    valid_contours, data, mask = segment_and_measure(enhanced_blur, fluo_raw, px_size)
    print(f"[INFO] Found {len(data)} valid objects.")

    # ==========================================
    # 6. VISUALIZATION
    # ==========================================

    # Create visual overlay
    img_struct_8u = normalize_to_8bit(img_grey)
    fluo_8u = normalize_to_8bit(fluo_raw)

    base_img = cv2.cvtColor(img_struct_8u, cv2.COLOR_GRAY2BGR)
    red_layer = np.zeros_like(base_img)

    # Mask the red layer so it only shows inside objects
    fluo_masked = cv2.bitwise_and(fluo_8u, fluo_8u, mask=mask)
    red_layer[:, :, 2] = fluo_masked 

    vis_img = cv2.addWeighted(base_img, 0.6, red_layer, 1.5, 0)

    for i, cnt in enumerate(valid_contours):
        item = data[i]
        cv2.drawContours(vis_img, [cnt], -1, (0, 255, 0), 1)
        cx, cy = item['centroid']
        # Display Total Intensity on the image
        label = f"{int(item['total_intensity'])}"
        cv2.putText(vis_img, label, (cx - 15, cy - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # 7. Save Outputs
    cv2.imwrite(os.path.join(CONFIG["DEBUG_DIR"], "03_result_overlay.png"), vis_img)

    # 8. Print Table
    print("\n" + "="*65)
    print(f"{'ID':<5} {'Area (µm²)':<15} {'Mean Int.':<15} {'Total Int.':<15}")
    print("="*65)
    for d in data:
        print(f"{d['id']:<5} {d['area_um2']:<15.2f} {d['mean_intensity']:<15.2f} {d['total_intensity']:<15.0f}")
    print("="*65)

if __name__ == "__main__":
    main()


# In[47]:


import cv2
import numpy as np
import pandas as pd
import os
import shutil
import xml.etree.ElementTree as ET

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    # File Paths
    "IMAGE_PATH": "phase_contrast_image.tif",   # Channel 1: Structural
    "PATH_RED_IMAGE": "red_fluorescence.tif",   # Channel 2: Fluorescence
    "METADATA_XML": "metadata.xml",             

    # Segmentation Parameters
    "MIN_AREA_PX": 100,             
    "MAX_AREA_PX": 10000,           

    # Output
    "DEBUG_DIR": "debug_output",
    "CSV_FILENAME": "results_red_fluorescence.csv"
}

# ==========================================
# 2. DUMMY DATA GENERATOR
# ==========================================
def generate_dummy_data_if_missing():
    if not os.path.exists(CONFIG["IMAGE_PATH"]):
        print("Generating dummy Phase Contrast image...")
        img = np.full((512, 512), 200, dtype=np.uint8)
        cv2.circle(img, (150, 150), 40, 50, -1) 
        cv2.circle(img, (350, 350), 50, 60, -1) 
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        cv2.imwrite(CONFIG["IMAGE_PATH"], img)

    if not os.path.exists(CONFIG["PATH_RED_IMAGE"]):
        print("Generating dummy Fluorescence image...")
        img_red = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(img_red, (150, 150), 40, (0, 0, 200), -1) 
        cv2.circle(img_red, (350, 350), 50, (0, 0, 50), -1) 
        cv2.imwrite(CONFIG["PATH_RED_IMAGE"], img_red)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def setup_debug_dir():
    if os.path.exists(CONFIG["DEBUG_DIR"]):
        shutil.rmtree(CONFIG["DEBUG_DIR"])
    os.makedirs(CONFIG["DEBUG_DIR"])

def get_pixel_scale(xml_path):
    if not os.path.exists(xml_path):
        return 1.0
    return 1.0 

def load_fluorescence_channel(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load {path}")
    if len(img.shape) == 3:
        return img[:, :, 2] 
    return img

# ==========================================
# 4. IMAGE PROCESSING PIPELINE
# ==========================================

def preprocess_for_dark_objects(img):
    """
    Enhances dark objects on light background.
    """
    # 1. Background Subtraction
    bg = cv2.GaussianBlur(img, (101, 101), 0)

    # Cast to int16 to avoid overflow during subtraction
    diff_raw = bg.astype(np.int16) - img.astype(np.int16)
    diff_clipped = np.clip(diff_raw, 0, 255).astype(np.uint8)

    # 2. Enhance contrast
    # FIX: Explicitly create destination array to satisfy type checker
    norm_dst = np.zeros(diff_clipped.shape, dtype=np.uint8)
    diff = cv2.normalize(diff_clipped, norm_dst, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite(f"{CONFIG['DEBUG_DIR']}/01_preprocessed_diff.png", diff)
    return diff

def segment_and_measure(structural_img, fluorescence_img, pixel_scale):
    # --- A. Segmentation ---
    enhanced = preprocess_for_dark_objects(structural_img)

    # Thresholding
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological Operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    cv2.imwrite(f"{CONFIG['DEBUG_DIR']}/02_binary_mask.png", binary)

    # Find Contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    final_mask = np.zeros_like(binary)

    # --- B. Measurement ---
    cell_id = 1

    for cnt in contours:
        area_px = cv2.contourArea(cnt)

        if CONFIG["MIN_AREA_PX"] < area_px < CONFIG["MAX_AREA_PX"]:

            # Mask for single cell
            single_cell_mask = np.zeros_like(binary)
            cv2.drawContours(single_cell_mask, [cnt], -1, 255, -1)
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)

            # Measure Fluorescence
            # cv2.mean returns a tuple (val0, val1, val2, val3)
            mean_tuple = cv2.mean(fluorescence_img, mask=single_cell_mask)
            mean_val = float(mean_tuple[0])

            total_intensity = mean_val * area_px

            # Calculate Centroid using Moments (Fixes np.mean type error)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            results.append({
                "id": cell_id,
                "area_px": area_px,
                "area_um2": area_px * (pixel_scale ** 2),
                "mean_fluorescence": mean_val,
                "total_intensity": total_intensity,
                "contour": cnt,
                "centroid": (cx, cy)
            })
            cell_id += 1

    return results, final_mask

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    generate_dummy_data_if_missing()
    setup_debug_dir()

    print("Loading images...")
    img_struct = cv2.imread(CONFIG["IMAGE_PATH"], cv2.IMREAD_GRAYSCALE)
    img_fluor = load_fluorescence_channel(CONFIG["PATH_RED_IMAGE"])

    if img_struct is None or img_fluor is None:
        print("Error: Could not read input images.")
        return

    pixel_scale = get_pixel_scale(CONFIG["METADATA_XML"])

    print("Segmenting and measuring...")
    results, mask = segment_and_measure(img_struct, img_fluor, pixel_scale)

    # --- Visualization ---
    output_img = cv2.cvtColor(img_struct, cv2.COLOR_GRAY2BGR)

    # Red overlay
    red_overlay = np.zeros_like(output_img)
    red_overlay[:, :, 2] = mask 
    output_img = cv2.addWeighted(output_img, 0.7, red_overlay, 0.3, 0)

    # Draw Contours and Labels
    for res in results:
        cnt = res['contour']
        cx, cy = res['centroid']

        # Draw Green Outline
        cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)

        # --- LABEL: Sequence Number (ID) ---
        label_text = f"#{res['id']}" 

        # Draw Text (Yellow)
        cv2.putText(output_img, label_text, (cx - 10, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imwrite("result_overlay.png", output_img)
    print(f"Analysis complete. Found {len(results)} cells.")
    print("Saved visualization to 'result_overlay.png'")

    # --- Export Data ---
    # Exclude non-serializable objects for CSV
    csv_data = []
    for r in results:
        row = r.copy()
        del row['contour']
        del row['centroid']
        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    df.to_csv(CONFIG["CSV_FILENAME"], index=False)
    print(f"Saved data to '{CONFIG['CSV_FILENAME']}'")

    if not df.empty:
        print("\nResults Preview:")
        print(df[["id", "area_px", "mean_fluorescence"]].head())

if __name__ == "__main__":
    main()


# In[ ]:


import cv2
import numpy as np
import pandas as pd
import os
import shutil
import xml.etree.ElementTree as ET
from typing import Tuple, cast

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    # File Paths
    "IMAGE_PATH": "phase_contrast_image.tif",   # Channel 1: Structural
    "PATH_RED_IMAGE": "red_fluorescence.tif",   # Channel 2: Fluorescence
    "METADATA_XML": "metadata.xml",             

    # Segmentation Parameters
    "MIN_AREA_PX": 100,             
    "MAX_AREA_PX": 10000,           

    # Output
    "DEBUG_DIR": "debug_output",
    "CSV_FILENAME": "results_red_fluorescence.csv"
}

# ==========================================
# 2. DUMMY DATA GENERATOR
# ==========================================
def generate_dummy_data_if_missing():
    if not os.path.exists(CONFIG["IMAGE_PATH"]):
        print("Generating dummy Phase Contrast image...")
        img = np.full((512, 512), 200, dtype=np.uint8)
        cv2.circle(img, (150, 150), 40, 50, -1) 
        cv2.circle(img, (350, 350), 50, 60, -1) 
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        cv2.imwrite(CONFIG["IMAGE_PATH"], img)

    if not os.path.exists(CONFIG["PATH_RED_IMAGE"]):
        print("Generating dummy Fluorescence image...")
        img_red = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(img_red, (150, 150), 40, (0, 0, 200), -1) 
        cv2.circle(img_red, (350, 350), 50, (0, 0, 50), -1) 
        cv2.imwrite(CONFIG["PATH_RED_IMAGE"], img_red)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def setup_debug_dir():
    if os.path.exists(CONFIG["DEBUG_DIR"]):
        shutil.rmtree(CONFIG["DEBUG_DIR"])
    os.makedirs(CONFIG["DEBUG_DIR"])

def get_pixel_scale(xml_path):
    if not os.path.exists(xml_path):
        return 1.0
    return 1.0 

def load_fluorescence_channel(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load {path}")
    if len(img.shape) == 3:
        return img[:, :, 2] 
    return img

# ==========================================
# 4. IMAGE PROCESSING PIPELINE
# ==========================================

def preprocess_for_dark_objects(img):
    """
    Enhances dark objects on light background.
    """
    # 1. Background Subtraction
    bg = cv2.GaussianBlur(img, (101, 101), 0)

    # Cast to int16 to avoid overflow during subtraction
    diff_raw = bg.astype(np.int16) - img.astype(np.int16)
    diff_clipped = np.clip(diff_raw, 0, 255).astype(np.uint8)

    # 2. Enhance contrast
    # Explicitly create destination array to satisfy type checker
    norm_dst = np.zeros(diff_clipped.shape, dtype=np.uint8)
    diff = cv2.normalize(diff_clipped, norm_dst, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite(f"{CONFIG['DEBUG_DIR']}/01_preprocessed_diff.png", diff)
    return diff

def segment_and_measure(structural_img, fluorescence_img, pixel_scale):
    # --- A. Segmentation ---
    enhanced = preprocess_for_dark_objects(structural_img)

    # Thresholding
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological Operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    cv2.imwrite(f"{CONFIG['DEBUG_DIR']}/02_binary_mask.png", binary)

    # Find Contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    final_mask = np.zeros_like(binary)

    # --- B. Measurement ---
    cell_id = 1

    for cnt in contours:
        area_px = cv2.contourArea(cnt)

        if CONFIG["MIN_AREA_PX"] < area_px < CONFIG["MAX_AREA_PX"]:

            # Mask for single cell
            single_cell_mask = np.zeros_like(binary)
            cv2.drawContours(single_cell_mask, [cnt], -1, 255, -1)
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)

            # Measure Fluorescence
            # FIX: Explicitly cast the result to a Tuple to satisfy Pylance
            # cv2.mean returns a Scalar (tuple of 4 floats)
            mean_scalar = cv2.mean(fluorescence_img, mask=single_cell_mask)
            mean_tuple = cast(Tuple[float, float, float, float], mean_scalar)
            mean_val = mean_tuple[0]

            total_intensity = mean_val * area_px

            # Calculate Centroid using Moments
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            results.append({
                "id": cell_id,
                "area_px": area_px,
                "area_um2": area_px * (pixel_scale ** 2),
                "mean_fluorescence": mean_val,
                "total_intensity": total_intensity,
                "contour": cnt,
                "centroid": (cx, cy)
            })
            cell_id += 1

    return results, final_mask

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    generate_dummy_data_if_missing()
    setup_debug_dir()

    print("Loading images...")
    img_struct = cv2.imread(CONFIG["IMAGE_PATH"], cv2.IMREAD_GRAYSCALE)
    img_fluor = load_fluorescence_channel(CONFIG["PATH_RED_IMAGE"])

    if img_struct is None or img_fluor is None:
        print("Error: Could not read input images.")
        return

    pixel_scale = get_pixel_scale(CONFIG["METADATA_XML"])

    print("Segmenting and measuring...")
    results, mask = segment_and_measure(img_struct, img_fluor, pixel_scale)

    # --- Visualization ---
    output_img = cv2.cvtColor(img_struct, cv2.COLOR_GRAY2BGR)

    # Red overlay
    red_overlay = np.zeros_like(output_img)
    red_overlay[:, :, 2] = mask 
    output_img = cv2.addWeighted(output_img, 0.7, red_overlay, 0.3, 0)

    # Draw Contours and Labels
    for res in results:
        cnt = res['contour']
        cx, cy = res['centroid']

        # Draw Green Outline
        cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)

        # --- LABEL: Sequence Number (ID) ---
        label_text = f"#{res['id']}" 

        # Draw Text (Yellow)
        cv2.putText(output_img, label_text, (cx - 10, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imwrite("result_overlay.png", output_img)
    print(f"Analysis complete. Found {len(results)} cells.")
    print("Saved visualization to 'result_overlay.png'")

    # --- Export Data ---
    # Exclude non-serializable objects for CSV
    csv_data = []
    for r in results:
        row = r.copy()
        del row['contour']
        del row['centroid']
        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    df.to_csv(CONFIG["CSV_FILENAME"], index=False)
    print(f"Saved data to '{CONFIG['CSV_FILENAME']}'")

    if not df.empty:
        print("\nResults Preview:")
        print(df[["id", "area_px", "mean_fluorescence"]].head())

if __name__ == "__main__":
    main()

