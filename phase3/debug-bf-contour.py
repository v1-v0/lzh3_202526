#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
CONFIG = {
    "IMAGE_PATH": r"source\12\12 N NO 1_ch00.tif",
    "METADATA_XML": r"source\12\12 N NO 1.xml",
    "DEBUG_DIR": "debug",
    "DEFAULT_PIXEL_SIZE_UM": 0.11,  # Fallback if XML fails
    "MIN_AREA_PX": 100,
    "MAX_AREA_PX": 5000,
}

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def setup_debug_dir(directory):
    """Creates the debug directory and clears existing files."""
    os.makedirs(directory, exist_ok=True)
    for f in glob.glob(os.path.join(directory, "*")):
        try:
            os.remove(f)
        except OSError:
            pass
    print(f"[INFO] Debug directory ready: {directory}")


def normalize_to_8bit(img):
    """
    Normalizes a 16-bit (or any bit depth) image to 8-bit (0-255) for visualization.
    """
    if img.dtype == np.uint8:
        return img
    
    imin, imax = img.min(), img.max()
    if imax > imin:
        # Float conversion for precision, then cast to uint8
        norm = (img - imin) / (imax - imin)
        return (norm * 255).astype(np.uint8)
    return np.zeros_like(img, dtype=np.uint8)


def save_debug_image(filename, img, directory):
    """Saves an image to the debug folder, auto-converting to 8-bit."""
    path = os.path.join(directory, filename)
    img_8bit = normalize_to_8bit(img)
    cv2.imwrite(path, img_8bit)


def load_pixel_size_from_metadata(xml_path):
    """
    Parses Leica XML metadata to find physical pixel size in microns.
    Returns: float (µm/px) or None.
    """
    if not os.path.isfile(xml_path):
        print(f"[WARN] Metadata XML not found: {xml_path}")
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Find DimensionDescription nodes
        dim_nodes = root.findall(".//DimensionDescription")
        if not dim_nodes:
            return None

        # Prioritize 'X' dimension (DimID="1" or "X")
        x_dim = next((d for d in dim_nodes if d.attrib.get("DimID") in ("X", "1")), dim_nodes[0])

        num_elems = float(x_dim.attrib["NumberOfElements"])
        length = float(x_dim.attrib["Length"])
        unit = x_dim.attrib.get("Unit", "").strip()

        if num_elems <= 0: return None

        # Unit conversion
        if unit in ("µm", "um"):
            length_um = length
        elif unit == "m":
            length_um = length * 1e6
        else:
            print(f"[WARN] Unknown unit '{unit}'. Cannot convert.")
            return None

        return length_um / num_elems

    except Exception as e:
        print(f"[WARN] Metadata parsing error: {e}")
        return None


# --------------------------------------------------
# IMAGE PROCESSING PIPELINE
# --------------------------------------------------

def preprocess_image(img):
    """
    Step 1: Background subtraction (Top-hat style) to highlight dark objects.
    """
    # Smooth background
    bg = cv2.GaussianBlur(img, (0, 0), sigmaX=15, sigmaY=15)
    
    # Subtract image from background (Brightfield logic: objects are darker)
    enhanced = cv2.subtract(bg, img)
    
    # Denoise
    enhanced_blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return enhanced, enhanced_blur


def segment_image(enhanced_img):
    """
    Step 2 & 3: Thresholding and Morphological closing.
    """
    # Otsu Thresholding
    _, thresh = cv2.threshold(
        enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological Closing (Bridge gaps)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Clean edges
    closed = cv2.dilate(closed, kernel, iterations=1)
    closed = cv2.erode(closed, kernel, iterations=1)

    # Fill holes (Solidify)
    # Using connected components to ensure we have solid masks
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    solid_mask = np.where(labels > 0, 255, 0).astype(np.uint8)
    
    return thresh, closed, solid_mask


def analyze_contours(mask, original_img, pixel_size_um):
    """
    Step 4: Find contours and calculate physical properties.
    IDs are assigned sequentially (1, 2, 3...) only for valid contours.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pixel_area_um2 = pixel_size_um ** 2
    results = []
    valid_contours = []
    
    # Initialize a sequential counter for valid blobs
    current_id = 1

    for cnt in contours:
        area_px = cv2.contourArea(cnt)

        # Filter by area
        if not (CONFIG["MIN_AREA_PX"] <= area_px <= CONFIG["MAX_AREA_PX"]):
            continue

        valid_contours.append(cnt)
        
        # Geometry
        perimeter_px = cv2.arcLength(cnt, closed=True)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx_px, cy_px = M["m10"] / M["m00"], M["m01"] / M["m00"]
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cx_px, cy_px = x + w / 2.0, y + h / 2.0

        # Intensity (Total intensity under the mask)
        c_mask = np.zeros(original_img.shape, dtype=np.uint8)
        cv2.drawContours(c_mask, [cnt], -1, 255, thickness=-1)
        total_intensity = float(original_img[c_mask == 255].sum())

        results.append({
            "id": current_id,  # Use the sequential counter
            "contour": cnt,
            "area_px": area_px,
            "area_um2": area_px * pixel_area_um2,
            "perimeter_px": perimeter_px,
            "perimeter_um": perimeter_px * pixel_size_um,
            "centroid_px": (cx_px, cy_px),
            "centroid_um": (cx_px * pixel_size_um, cy_px * pixel_size_um),
            "total_intensity": total_intensity
        })
        
        # Increment counter only after a valid addition
        current_id += 1

    return valid_contours, results


def visualize_results(original_img, results, valid_contours):
    """
    Step 5: Draw overlays on the original image.
    """
    # Convert to 8-bit BGR for drawing
    vis_img = cv2.cvtColor(normalize_to_8bit(original_img), cv2.COLOR_GRAY2BGR)
    
    # Draw all contours
    cv2.drawContours(vis_img, valid_contours, -1, (0, 0, 255), 1)

    # Draw labels
    for res in results:
        cx, cy = res["centroid_px"]
        cx_i, cy_i = int(round(cx)), int(round(cy))
        
        # Centroid dot
        cv2.circle(vis_img, (cx_i, cy_i), 3, (0, 255, 0), -1)
        # ID Text
        cv2.putText(vis_img, str(res["id"]), (cx_i + 5, cy_i - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        
    return vis_img


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():
    setup_debug_dir(CONFIG["DEBUG_DIR"])

    # 1. Calibration
    px_size = load_pixel_size_from_metadata(CONFIG["METADATA_XML"])
    if px_size is None:
        px_size = CONFIG["DEFAULT_PIXEL_SIZE_UM"]
        print(f"[INFO] Using DEFAULT pixel size: {px_size:.6f} µm/px")
    else:
        print(f"[INFO] Using METADATA pixel size: {px_size:.6f} µm/px")

    # 2. Load Image
    if not os.path.exists(CONFIG["IMAGE_PATH"]):
        print(f"[ERROR] Image not found: {CONFIG['IMAGE_PATH']}")
        return

    img = cv2.imread(CONFIG["IMAGE_PATH"], cv2.IMREAD_UNCHANGED)
    if img is None:
        print("[ERROR] Failed to load image data.")
        return

    # Ensure Grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"[INFO] Image loaded. Shape: {img.shape}, Dtype: {img.dtype}")

    # 3. Processing
    enhanced, enhanced_blur = preprocess_image(img)
    thresh, closed, solid_mask = segment_image(enhanced_blur)
    
    # 4. Analysis
    valid_contours, data = analyze_contours(solid_mask, img, px_size)
    print(f"[INFO] Analysis complete. Found {len(data)} valid clumps.")

    # 5. Visualization
    final_vis = visualize_results(img, data, valid_contours)

    # 6. Save Debug Outputs
    save_debug_image("01_enhanced.png", enhanced, CONFIG["DEBUG_DIR"])
    save_debug_image("02_enhanced_blur.png", enhanced_blur, CONFIG["DEBUG_DIR"])
    save_debug_image("03_thresh.png", thresh, CONFIG["DEBUG_DIR"])
    save_debug_image("04_closed.png", closed, CONFIG["DEBUG_DIR"])
    save_debug_image("05_solid_mask.png", solid_mask, CONFIG["DEBUG_DIR"])
    save_debug_image("06_final_result.png", final_vis, CONFIG["DEBUG_DIR"])

    # 7. Print Report
    print("-" * 60)
    print(f"{'ID':<5} {'Area (µm²)':<15} {'Perim (µm)':<15} {'Intensity':<15}")
    print("-" * 60)
    for d in data:
        print(f"{d['id']:<5} {d['area_um2']:<15.2f} {d['perimeter_um']:<15.2f} {d['total_intensity']:<15.0f}")

if __name__ == "__main__":
    main()