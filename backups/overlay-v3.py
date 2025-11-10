"""
Enhanced Python script for bacteria contour detection with DUAL-CHANNEL MATCHING.
Features:
- Dual contouring: BOTH fluorescence (_ch01.tif) AND grayscale (_ch00.tif)
- Match fluorescent bacteria to grayscale bacteria using spatial overlap
- Use LARGER contour between gray and fluorescence for matched bacteria
- BRIGHT, SHARP fluorescence overlay matching source intensity
- Fluorescence overlay FULLY COVERS the selected contour region
- Mark grayscale-only bacteria with "g" suffix
- Identify non-matching bacteria across channels
- Robust error handling and validation
- Smart label positioning with collision avoidance
- Sequential numbering across dataset
- Comprehensive morphological measurements
- Progress tracking and detailed reporting
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
from pathlib import Path
from skimage import io, filters, morphology, measure
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from shapely.geometry import Polygon
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    "source_folder": ["source//1"],
    "output_folder": ["outputs//v3"],
    "gray_threshold": 52,              # Threshold for grayscale (dark objects)
    "fluor_threshold": 5,            # Threshold for fluorescence (bright objects)
    "min_area": 400,                   # Minimum area for grayscale bacteria
    "min_fluor_area": 400,             # Minimum area for fluorescent regions
    "overlap_threshold": 0.80,         # 80% overlap required for match
    "contour_thickness": 1,
    "min_contour_perimeter": 10,
    "outline_color": (0, 255, 0),      # BGR: Green for grayscale bacteria
    "fluor_outline_color": (0, 0, 255), # BGR: Red for fluorescent bacteria
    "grayscale_only_color": (255, 0, 255), # BGR: Magenta for grayscale-only
    "fluorescence_color": (0, 0, 255),  # BGR: Red for fluorescence overlay
    "fluorescence_alpha": 0.7,         # Increased for brighter overlay
    "fluorescence_intensity_boost": 1.5, # Boost fluorescence brightness
    "fluorescence_gamma": 0.8,         # Gamma correction for sharper appearance
    "arrow_length": 40,
    "arrow_thickness": 1,
    "arrow_tip_length": 0.3,
    "font_scale": 2,
    "font_thickness": 2,
    "label_offset_candidates": [
        (40, -40), (40, 40), (-40, -40), (-40, 40),
        (0, -40), (0, 40), (40, 0), (-40, 0),
        (40, -40), (40, 40), (-40, -40), (-40, 40)
    ]
}


def validate_config():
    """Validate configuration parameters"""
    errors = []
    
    if not CONFIG["source_folder"]:
        errors.append("source_folder cannot be empty")
    
    if not isinstance(CONFIG["source_folder"], list):
        errors.append("source_folder must be a list")
    
    if CONFIG["gray_threshold"] < 0 or CONFIG["gray_threshold"] > 255:
        errors.append("gray_threshold must be between 0 and 255")
    
    if CONFIG["fluor_threshold"] < 0 or CONFIG["fluor_threshold"] > 255:
        errors.append("fluor_threshold must be between 0 and 255")
    
    if CONFIG["min_area"] <= 0:
        errors.append("min_area must be positive")
    
    if CONFIG["min_fluor_area"] <= 0:
        errors.append("min_fluor_area must be positive")
    
    if CONFIG["overlap_threshold"] < 0 or CONFIG["overlap_threshold"] > 1:
        errors.append("overlap_threshold must be between 0 and 1")
    
    if CONFIG["fluorescence_alpha"] < 0 or CONFIG["fluorescence_alpha"] > 1:
        errors.append("fluorescence_alpha must be between 0 and 1")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    print("✓ Configuration validated successfully")


def load_image_pairs(source_folders=CONFIG["source_folder"]):
    """
    Load paired '_ch00.tif' (grayscale) and '_ch01.tif' (fluorescence) files.
    Returns lists of grayscale images, fluorescence images, and their paths.
    """
    gray_images = []
    fluor_images = []
    image_paths = []
    
    print(f"\n{'='*70}")
    print("LOADING IMAGE PAIRS")
    print(f"{'='*70}")
    
    for folder in source_folders:
        folder_path = Path(folder)
        
        if not folder_path.exists():
            print(f"⚠ Warning: Folder '{folder}' does not exist, skipping...")
            continue
        
        if not folder_path.is_dir():
            print(f"⚠ Warning: '{folder}' is not a directory, skipping...")
            continue
        
        # Find all '_ch00.tif' files
        gray_files = sorted([
            f for f in os.listdir(folder) 
            if f.endswith('_ch00.tif') and os.path.isfile(os.path.join(folder, f))
        ])
        
        if not gray_files:
            print(f"⚠ No '_ch00.tif' files found in '{folder}'")
            continue
        
        print(f"\nFolder: {folder}")
        print(f"Found {len(gray_files)} grayscale image(s)")
        
        for gray_file in gray_files:
            # Construct fluorescence filename
            fluor_file = gray_file.replace('_ch00.tif', '_ch01.tif')
            
            gray_path = os.path.join(folder, gray_file)
            fluor_path = os.path.join(folder, fluor_file)
            
            try:
                # Check if fluorescence file exists
                if not os.path.exists(fluor_path):
                    print(f"  ⚠ Warning: No matching fluorescence file for {gray_file}")
                    print(f"    Expected: {fluor_file}")
                    continue
                
                print(f"  Loading pair: {gray_file} + {fluor_file}...", end=" ")
                
                # Load grayscale image
                gray_img = io.imread(gray_path)
                if len(gray_img.shape) == 3:
                    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)
                elif len(gray_img.shape) != 2:
                    print(f"✗ Unexpected gray shape {gray_img.shape}, skipping")
                    continue
                
                # Load fluorescence image
                fluor_img = io.imread(fluor_path)
                if len(fluor_img.shape) == 3:
                    fluor_img = cv2.cvtColor(fluor_img, cv2.COLOR_RGB2GRAY)
                elif len(fluor_img.shape) != 2:
                    print(f"✗ Unexpected fluor shape {fluor_img.shape}, skipping")
                    continue
                
                # Check if dimensions match
                if gray_img.shape != fluor_img.shape:
                    print(f"✗ Dimension mismatch: gray{gray_img.shape} vs fluor{fluor_img.shape}")
                    continue
                
                gray_images.append(gray_img)
                fluor_images.append(fluor_img)
                image_paths.append(gray_path)
                
                print(f"✓ Shape: {gray_img.shape}")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
    
    if not gray_images:
        raise ValueError("No image pairs loaded from any source folder.")
    
    print(f"\n{'='*70}")
    print(f"✓ Successfully loaded {len(gray_images)} image pair(s)")
    print(f"{'='*70}\n")
    
    return gray_images, fluor_images, image_paths


def preprocess_grayscale(image):
    """Preprocess grayscale image for dark object detection"""
    # Apply median filter for noise reduction
    median_filtered = filters.median(image, morphology.disk(3))
    
    # Normalize to uint8
    if median_filtered.max() > 0:
        img_normalized = (median_filtered / median_filtered.max() * 255).astype(np.uint8)
    else:
        img_normalized = median_filtered.astype(np.uint8)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_normalized)
    
    return enhanced


def preprocess_fluorescence(image):
    """Preprocess fluorescence image for bright object detection"""
    # Apply median filter for noise reduction
    median_filtered = filters.median(image, morphology.disk(3))
    
    # Normalize to uint8
    if median_filtered.max() > 0:
        img_normalized = (median_filtered / median_filtered.max() * 255).astype(np.uint8)
    else:
        img_normalized = median_filtered.astype(np.uint8)
    
    # Apply CLAHE for fluorescence
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_normalized)
    
    return enhanced


def segment_grayscale(preprocessed, threshold=CONFIG["gray_threshold"]):
    """Segment grayscale bacteria (dark objects)"""
    # Inverted binary thresholding (dark objects = foreground)
    binary = np.asarray(preprocessed) <= threshold
    
    # Distance transform
    distance_result = ndi.distance_transform_edt(binary)
    if isinstance(distance_result, tuple):
        distance = distance_result[0]
    else:
        distance = distance_result
    if distance is None:
        raise RuntimeError("distance_transform_edt returned None")
    distance = np.asarray(distance, dtype=np.float32)
    
    # Find local maxima for watershed markers
    local_max = morphology.local_maxima(distance)
    markers = measure.label(local_max)
    
    # Apply watershed segmentation
    labels = watershed(-distance, markers, mask=binary)
    
    # Create binary mask
    separated = labels > 0
    
    return separated, labels


def segment_fluorescence(preprocessed, threshold=CONFIG["fluor_threshold"]):
    """Segment fluorescence bacteria (bright objects)"""
    # Normal binary thresholding (bright objects = foreground)
    binary = np.asarray(preprocessed) >= threshold
    
    # Distance transform
    distance_result = ndi.distance_transform_edt(binary)
    if isinstance(distance_result, tuple):
        distance = distance_result[0]
    else:
        distance = distance_result
    if distance is None:
        raise RuntimeError("distance_transform_edt returned None")
    distance = np.asarray(distance, dtype=np.float32)
    
    # Find local maxima for watershed markers
    local_max = morphology.local_maxima(distance)
    markers = measure.label(local_max)
    
    # Apply watershed segmentation
    labels = watershed(-distance, markers, mask=binary)
    
    # Create binary mask
    separated = labels > 0
    
    return separated, labels


def postprocess(binary_mask):
    """Post-process binary mask"""
    min_size = 50
    
    # Remove small objects
    cleaned = morphology.remove_small_objects(binary_mask, min_size=min_size)
    
    # Fill holes
    filled = ndi.binary_fill_holes(cleaned)
    
    # Morphological opening
    opened = morphology.binary_opening(filled, morphology.disk(2))
    
    # Morphological closing
    final = morphology.binary_closing(opened, morphology.disk(2))
    
    return final


def analyze_and_get_large(final_mask, min_area, channel_name=""):
    """Label connected components and filter by minimum area"""
    labeled = measure.label(final_mask)
    props = measure.regionprops(labeled)
    
    # Filter by area
    large_props = [prop for prop in props if prop.area > min_area]
    
    print(f"\n{channel_name} - Detected {len(props)} total region(s)")
    print(f"{channel_name} - Bacteria with area > {min_area} pixels: {len(large_props)}")
    
    return labeled, props, large_props


def contour_to_polygon(contour):
    """Convert OpenCV contour to Shapely polygon"""
    try:
        if len(contour) < 3:
            return None
        points = contour.reshape(-1, 2)
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = make_valid(polygon)
        return polygon
    except:
        return None


def calculate_overlap(polygon1, polygon2):
    """Calculate overlap percentage between two polygons"""
    try:
        if polygon1 is None or polygon2 is None:
            return 0.0
        
        if not polygon1.is_valid or not polygon2.is_valid:
            return 0.0
        
        intersection = polygon1.intersection(polygon2)
        if intersection.is_empty:
            return 0.0
        
        # Calculate overlap as intersection / smaller area
        area1 = polygon1.area
        area2 = polygon2.area
        smaller_area = min(area1, area2)
        
        if smaller_area == 0:
            return 0.0
        
        overlap_percent = intersection.area / smaller_area
        return overlap_percent
    except:
        return 0.0


def match_fluorescence_to_grayscale(gray_props, fluor_props, gray_labeled, fluor_labeled):
    """
    Match fluorescence bacteria to grayscale bacteria based on spatial overlap.
    Returns: dict mapping gray_index -> fluor_index (or None if no match)
    Also returns contours for both channels
    """
    matches = {}  # gray_index -> fluor_index
    fluor_matched = set()  # Track which fluor bacteria have been matched
    
    print(f"\n{'='*70}")
    print("MATCHING FLUORESCENCE TO GRAYSCALE BACTERIA")
    print(f"{'='*70}")
    print(f"Grayscale bacteria: {len(gray_props)}")
    print(f"Fluorescent bacteria: {len(fluor_props)}")
    print(f"Overlap threshold: {CONFIG['overlap_threshold']*100:.0f}%\n")
    
    # Extract contours for all bacteria
    gray_contours = []
    for i, prop in enumerate(gray_props):
        region_mask = (gray_labeled == prop.label).astype(np.uint8)
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            gray_contours.append(max(contours, key=cv2.contourArea))
        else:
            gray_contours.append(None)
    
    fluor_contours = []
    for i, prop in enumerate(fluor_props):
        region_mask = (fluor_labeled == prop.label).astype(np.uint8)
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            fluor_contours.append(max(contours, key=cv2.contourArea))
        else:
            fluor_contours.append(None)
    
    # Convert to polygons
    gray_polygons = [contour_to_polygon(c) if c is not None else None for c in gray_contours]
    fluor_polygons = [contour_to_polygon(c) if c is not None else None for c in fluor_contours]
    
    # Match each grayscale bacterium to fluorescent bacteria
    for gray_idx, gray_polygon in enumerate(gray_polygons):
        if gray_polygon is None:
            matches[gray_idx] = None
            continue
        
        best_match = None
        best_overlap = 0.0
        
        for fluor_idx, fluor_polygon in enumerate(fluor_polygons):
            if fluor_idx in fluor_matched:
                continue
            
            if fluor_polygon is None:
                continue
            
            overlap = calculate_overlap(gray_polygon, fluor_polygon)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = fluor_idx
        
        # Check if best match meets threshold
        if best_match is not None and best_overlap >= CONFIG['overlap_threshold']:
            matches[gray_idx] = best_match
            fluor_matched.add(best_match)
            print(f"  Grayscale #{gray_idx+1} ↔ Fluorescence #{best_match+1} (overlap: {best_overlap*100:.1f}%)")
        else:
            matches[gray_idx] = None
            if best_overlap > 0:
                print(f"  Grayscale #{gray_idx+1} - No match (best: {best_overlap*100:.1f}% < threshold)")
            else:
                print(f"  Grayscale #{gray_idx+1} - No match (no overlap)")
    
    # Summary
    matched_count = sum(1 for v in matches.values() if v is not None)
    unmatched_count = len(matches) - matched_count
    
    print(f"\n{'='*70}")
    print(f"MATCHING SUMMARY:")
    print(f"  Matched: {matched_count}")
    print(f"  Grayscale-only (marked 'g'): {unmatched_count}")
    print(f"  Fluorescence-only: {len(fluor_props) - len(fluor_matched)}")
    print(f"{'='*70}\n")
    
    return matches, gray_contours, fluor_contours


def find_label_position(labeled, cX, cY, text_w, text_h, baseline, width, height):
    """Find suitable position for label text"""
    candidates = CONFIG["label_offset_candidates"]
    
    for dx, dy in candidates:
        tx = cX + dx
        ty = cY + dy
        
        # Clamp to image bounds
        tx = max(0, min(tx, width - text_w))
        ty = max(text_h + baseline, min(ty, height))
        
        # Define rectangle for overlap check
        rect_x1 = max(0, tx)
        rect_y1 = max(0, ty - text_h - baseline)
        rect_x2 = min(width, tx + text_w)
        rect_y2 = min(height, ty)
        
        if rect_x1 >= rect_x2 or rect_y1 >= rect_y2:
            continue
        
        try:
            if np.all(labeled[rect_y1:rect_y2, rect_x1:rect_x2] == 0):
                return (tx, ty), True
        except:
            continue
    
    tx = max(0, min(cX + 20, width - text_w))
    ty = max(text_h + baseline, min(cY - 20, height))
    return (tx, ty), False


def draw_arrow_to_bacterium(overlay, text_pos, text_w, text_h, baseline, cX, cY, color):
    """Draw arrow from text label to bacterium centroid"""
    text_center_x = text_pos[0] + text_w // 2
    text_center_y = text_pos[1] - (text_h + baseline) // 2
    arrow_start = (text_center_x, text_center_y)
    
    vec = np.array([cX - arrow_start[0], cY - arrow_start[1]], dtype=float)
    dist = np.linalg.norm(vec)
    
    if dist > CONFIG["arrow_length"]:
        unit_vec = vec / dist
        arrow_end = (
            int(arrow_start[0] + unit_vec[0] * CONFIG["arrow_length"]),
            int(arrow_start[1] + unit_vec[1] * CONFIG["arrow_length"])
        )
        cv2.arrowedLine(
            overlay, arrow_start, arrow_end, 
            color, 
            CONFIG["arrow_thickness"], 
            tipLength=CONFIG["arrow_tip_length"]
        )


def create_mask_from_contour(contour, shape):
    """Create a binary mask from a contour"""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, -1)  # -1 fills the contour
    return mask


def apply_fluorescence_overlay(overlay, fluorescence, mask_to_use):
    """
    Apply BRIGHT, SHARP fluorescence channel as colored overlay.
    Uses the provided mask (can be from grayscale or fluorescence contour).
    Fully covers the entire masked region matching source intensity.
    """
    # Create enhanced fluorescence for the entire image
    fluor_enhanced = fluorescence.copy().astype(np.float32)
    
    # Apply gamma correction for sharper appearance
    gamma = CONFIG["fluorescence_gamma"]
    if fluorescence.max() > 0:
        fluor_enhanced = fluor_enhanced / fluorescence.max()
        fluor_enhanced = np.power(fluor_enhanced, gamma)
        fluor_enhanced = fluor_enhanced * 255
    
    # Boost intensity
    fluor_enhanced = fluor_enhanced * CONFIG["fluorescence_intensity_boost"]
    fluor_enhanced = np.clip(fluor_enhanced, 0, 255).astype(np.uint8)
    
    # Create bright red fluorescence layer
    fluor_colored = np.zeros_like(overlay)
    fluor_colored[:, :, 2] = fluor_enhanced  # Red channel in BGR
    
    # Apply to the masked region with high alpha for brightness
    alpha = CONFIG["fluorescence_alpha"]
    mask_3d = np.stack([mask_to_use] * 3, axis=2).astype(bool)
    
    # Use additive blending for brightness
    overlay[mask_3d] = np.clip(
        overlay[mask_3d].astype(np.float32) * (1 - alpha) + 
        fluor_colored[mask_3d].astype(np.float32) * alpha,
        0, 255
    ).astype(np.uint8)
    
    return overlay


def visualize_dual_channel(original, fluorescence, gray_labeled, fluor_labeled, 
                          gray_props, fluor_props, matches, gray_contours, fluor_contours,
                          image_path, image_index, bacterium_counter, output_path="outlined_bacteria.png"):
    """
    Create visualization with dual-channel matching.
    Uses LARGER contour between gray and fluorescence for matched bacteria.
    Fluorescence overlay FULLY COVERS the selected contour region.
    Matched bacteria: green outline + BRIGHT, SHARP red fluorescence
    Grayscale-only bacteria: magenta outline + label with 'g' suffix
    """
    data_collected = []
    non_matching = []  # List of bacteria that don't match
    
    try:
        # Create RGB overlay from grayscale
        if original.max() > 0:
            overlay = np.dstack([original, original, original])
            overlay = (overlay / overlay.max() * 255).astype(np.uint8)
        else:
            overlay = np.zeros((*original.shape, 3), dtype=np.uint8)
        
        height, width = original.shape
        
        # Process each grayscale bacterium
        for gray_idx, gray_prop in enumerate(gray_props):
            try:
                # Get binary mask for this region
                region_mask = (gray_labeled == gray_prop.label).astype(np.uint8)
                
                # Check if this bacterium has a fluorescence match
                fluor_idx = matches.get(gray_idx)
                has_match = fluor_idx is not None
                
                # Get contours
                gray_contour = gray_contours[gray_idx]
                
                # Determine which contour to use
                selected_contour = None
                contour_source = ""
                
                if has_match and fluor_idx < len(fluor_contours):
                    fluor_contour = fluor_contours[fluor_idx]
                    
                    # Compare areas and use larger contour
                    if gray_contour is not None and fluor_contour is not None:
                        gray_area = cv2.contourArea(gray_contour)
                        fluor_area = cv2.contourArea(fluor_contour)
                        
                        if fluor_area > gray_area:
                            selected_contour = fluor_contour
                            contour_source = "fluorescence"
                        else:
                            selected_contour = gray_contour
                            contour_source = "grayscale"
                        
                        print(f"  Bacterium {gray_idx+1}: Using {contour_source} contour "
                              f"(gray={gray_area:.1f}, fluor={fluor_area:.1f} pixels²)")
                    else:
                        selected_contour = gray_contour if gray_contour is not None else fluor_contour
                        contour_source = "fallback"
                else:
                    selected_contour = gray_contour
                    contour_source = "grayscale-only"
                
                if selected_contour is None:
                    print(f"  ⚠ Warning: No contour available for bacterium {gray_idx+1}")
                    continue
                
                # Validate contour
                perimeter = cv2.arcLength(selected_contour, True)
                if perimeter < CONFIG["min_contour_perimeter"]:
                    print(f"  ⚠ Warning: Contour too small for bacterium {gray_idx+1}")
                    continue
                
                # Apply BRIGHT fluorescence overlay if matched
                # KEY FIX: Use mask from SELECTED contour, not just fluor_labeled region
                if has_match:
                    # Create mask from the selected contour (covers entire selected region)
                    contour_mask = create_mask_from_contour(selected_contour, original.shape)
                    overlay = apply_fluorescence_overlay(overlay, fluorescence, contour_mask)
                    outline_color = CONFIG["outline_color"]  # Green
                    label_suffix = ""
                else:
                    outline_color = CONFIG["grayscale_only_color"]  # Magenta
                    label_suffix = "g"
                
                # Calculate centroid from the selected contour
                M = cv2.moments(selected_contour)
                if M["m00"] == 0:
                    print(f"  ⚠ Warning: Zero area contour for bacterium {gray_idx+1}")
                    continue
                
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Draw contour AFTER fluorescence overlay
                cv2.drawContours(
                    overlay, [selected_contour], -1, 
                    outline_color, 
                    CONFIG["contour_thickness"]
                )
                
                # Calculate mean fluorescence intensity
                fluor_intensity = np.mean(fluorescence[region_mask > 0])
                
                # Calculate area from the selected contour
                contour_area = cv2.contourArea(selected_contour)
                
                # Prepare label text
                annotation = f"{bacterium_counter}{label_suffix}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size, baseline = cv2.getTextSize(
                    annotation, font, CONFIG["font_scale"], CONFIG["font_thickness"]
                )
                text_w, text_h = text_size
                
                # Find suitable label position
                text_pos, found_good_pos = find_label_position(
                    gray_labeled, cX, cY, text_w, text_h, baseline, width, height
                )
                
                # Draw label text
                cv2.putText(
                    overlay, annotation, text_pos, font, 
                    CONFIG["font_scale"], outline_color, 
                    CONFIG["font_thickness"]
                )
                
                # Draw arrow from label to bacterium
                draw_arrow_to_bacterium(
                    overlay, text_pos, text_w, text_h, baseline, cX, cY, outline_color
                )
                
                # Collect morphological data
                filename = os.path.basename(image_path)
                contour_data = {
                    "Bacterium_ID": f"{bacterium_counter}{label_suffix}",
                    "Image": filename,
                    "Image_Index": image_index,
                    "Bacterium_In_Image": gray_idx + 1,
                    "Has_Fluorescence_Match": has_match,
                    "Match_Type": "Matched" if has_match else "Grayscale-only",
                    "Contour_Source": contour_source,
                    "Area": round(contour_area, 2),
                    "Perimeter": round(perimeter, 2),
                    "Centroid_X": cX,
                    "Centroid_Y": cY,
                    "Major_Axis_Length": round(gray_prop.major_axis_length, 2),
                    "Minor_Axis_Length": round(gray_prop.minor_axis_length, 2),
                    "Aspect_Ratio": round(gray_prop.major_axis_length / gray_prop.minor_axis_length if gray_prop.minor_axis_length > 0 else 0, 2),
                    "Eccentricity": round(gray_prop.eccentricity, 3),
                    "Solidity": round(gray_prop.solidity, 3),
                    "Extent": round(gray_prop.extent, 3),
                    "Orientation_Degrees": round(np.degrees(gray_prop.orientation), 2),
                    "Fluorescence_Mean_Intensity": round(fluor_intensity, 2),
                    "Contour_Points": len(selected_contour)
                }
                data_collected.append(contour_data)
                
                # Track non-matching bacteria
                if not has_match:
                    non_matching.append(contour_data)
                
                # Increment counter
                bacterium_counter += 1
                
            except Exception as e:
                print(f"  ✗ Error processing grayscale bacterium {gray_idx+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save visualization
        output_dir = Path(CONFIG["output_folder"][0])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / output_path
        
        cv2.imwrite(str(output_file), overlay)
        print(f"  ✓ Saved: {output_file}")
        
        return data_collected, non_matching, bacterium_counter
        
    except Exception as e:
        print(f"  ✗ Error in visualization: {e}")
        import traceback
        traceback.print_exc()
        return [], [], bacterium_counter


def save_master_csv(all_data, master_csv_filename):
    """Save all morphological data to master CSV file"""
    try:
        with open(master_csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            fieldnames = [
                "Bacterium_ID", "Image", "Image_Index", "Bacterium_In_Image",
                "Has_Fluorescence_Match", "Match_Type", "Contour_Source",
                "Area", "Perimeter", "Centroid_X", "Centroid_Y", 
                "Major_Axis_Length", "Minor_Axis_Length", "Aspect_Ratio",
                "Eccentricity", "Solidity", "Extent", "Orientation_Degrees",
                "Fluorescence_Mean_Intensity", "Contour_Points"
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_data)
        
        print(f"✓ Master CSV saved: {master_csv_filename}")
        return True
    except Exception as e:
        print(f"✗ Error saving CSV: {e}")
        return False


def save_non_matching_csv(non_matching_data, csv_filename):
    """Save non-matching (grayscale-only) bacteria to separate CSV"""
    try:
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            fieldnames = [
                "Bacterium_ID", "Image", "Image_Index", "Bacterium_In_Image",
                "Area", "Perimeter", "Centroid_X", "Centroid_Y",
                "Fluorescence_Mean_Intensity"
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(non_matching_data)
        
        print(f"✓ Non-matching bacteria CSV saved: {csv_filename}")
        return True
    except Exception as e:
        print(f"✗ Error saving non-matching CSV: {e}")
        return False


def print_summary_statistics(all_data, non_matching_data, dataset_name):
    """Print comprehensive summary statistics"""
    if not all_data:
        print("\n⚠ No bacteria detected in any images!")
        return
    
    # Extract statistics
    num_images = len(set(d['Image'] for d in all_data))
    total_bacteria = len(all_data)
    matched_bacteria = sum(1 for d in all_data if d['Has_Fluorescence_Match'])
    unmatched_bacteria = len(non_matching_data)
    avg_per_image = total_bacteria / num_images if num_images > 0 else 0
    
    # Contour source statistics
    gray_contours = sum(1 for d in all_data if d.get('Contour_Source') == 'grayscale')
    fluor_contours = sum(1 for d in all_data if d.get('Contour_Source') == 'fluorescence')
    
    areas = [d['Area'] for d in all_data]
    fluorescence = [d['Fluorescence_Mean_Intensity'] for d in all_data]
    matched_fluor = [d['Fluorescence_Mean_Intensity'] for d in all_data if d['Has_Fluorescence_Match']]
    unmatched_fluor = [d['Fluorescence_Mean_Intensity'] for d in all_data if not d['Has_Fluorescence_Match']]
    
    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"Images processed: {num_images}")
    print(f"Total bacteria detected: {total_bacteria}")
    print(f"Average bacteria per image: {avg_per_image:.2f}")
    print(f"\nMatching Results:")
    print(f"  Matched (with fluorescence): {matched_bacteria}")
    print(f"  Grayscale-only (marked 'g'): {unmatched_bacteria}")
    print(f"  Match rate: {matched_bacteria/total_bacteria*100:.1f}%")
    print(f"\nContour Selection (for matched bacteria):")
    print(f"  Used grayscale contour: {gray_contours}")
    print(f"  Used fluorescence contour: {fluor_contours}")
    
    print(f"\nArea Statistics (pixels²):")
    print(f"  Mean:   {np.mean(areas):>8.2f}")
    print(f"  Median: {np.median(areas):>8.2f}")
    
    print(f"\nFluorescence Intensity - All Bacteria:")
    print(f"  Mean:   {np.mean(fluorescence):>8.2f}")
    print(f"  Median: {np.median(fluorescence):>8.2f}")
    
    if matched_fluor:
        print(f"\nFluorescence Intensity - Matched Bacteria:")
        print(f"  Mean:   {np.mean(matched_fluor):>8.2f}")
        print(f"  Median: {np.median(matched_fluor):>8.2f}")
    
    if unmatched_fluor:
        print(f"\nFluorescence Intensity - Grayscale-only Bacteria:")
        print(f"  Mean:   {np.mean(unmatched_fluor):>8.2f}")
        print(f"  Median: {np.median(unmatched_fluor):>8.2f}")
    
    print(f"\nOutputs saved to: {CONFIG['output_folder'][0]}/")
    print(f"{'='*70}\n")


def create_summary_plots(all_data, non_matching_data, dataset_name):
    """Create summary visualization plots"""
    if not all_data:
        return
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Dual-Channel Bacterial Analysis - Dataset: {dataset_name}', 
                     fontsize=14, fontweight='bold')
        
        # Separate matched and unmatched
        matched = [d for d in all_data if d['Has_Fluorescence_Match']]
        unmatched = [d for d in all_data if not d['Has_Fluorescence_Match']]
        
        # Fluorescence distribution comparison
        if matched and unmatched:
            matched_fluor = [d['Fluorescence_Mean_Intensity'] for d in matched]
            unmatched_fluor = [d['Fluorescence_Mean_Intensity'] for d in unmatched]
            
            axes[0, 0].hist(matched_fluor, bins=20, alpha=0.7, label='Matched', color='green', edgecolor='black')
            axes[0, 0].hist(unmatched_fluor, bins=20, alpha=0.7, label='Grayscale-only', color='magenta', edgecolor='black')
            axes[0, 0].set_xlabel('Fluorescence Intensity', fontweight='bold')
            axes[0, 0].set_ylabel('Frequency', fontweight='bold')
            axes[0, 0].set_title('Fluorescence Distribution by Match Type')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
        
        # Match statistics pie chart
        match_counts = [len(matched), len(unmatched)]
        labels = [f'Matched\n({len(matched)})', f'Grayscale-only\n({len(unmatched)})']
        colors = ['#2ecc71', '#e74c3c']
        axes[0, 1].pie(match_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Matching Statistics')
        
        # Area distribution
        areas = [d['Area'] for d in all_data]
        axes[1, 0].hist(areas, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(np.mean(areas), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(areas):.1f}')
        axes[1, 0].set_xlabel('Area (pixels²)', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontweight='bold')
        axes[1, 0].set_title('Area Distribution (Using Larger Contour)')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Area vs Fluorescence scatter (color-coded by match type)
        matched_areas = [d['Area'] for d in matched]
        matched_fluor = [d['Fluorescence_Mean_Intensity'] for d in matched]
        unmatched_areas = [d['Area'] for d in unmatched]
        unmatched_fluor = [d['Fluorescence_Mean_Intensity'] for d in unmatched]
        
        if matched:
            axes[1, 1].scatter(matched_areas, matched_fluor, c='green', 
                             s=50, alpha=0.6, edgecolors='black', linewidth=0.5, label='Matched')
        if unmatched:
            axes[1, 1].scatter(unmatched_areas, unmatched_fluor, c='magenta', 
                             s=50, alpha=0.6, edgecolors='black', linewidth=0.5, label='Grayscale-only')
        axes[1, 1].set_xlabel('Area (pixels²)', fontweight='bold')
        axes[1, 1].set_ylabel('Fluorescence Intensity', fontweight='bold')
        axes[1, 1].set_title('Area vs Fluorescence')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        output_dir = Path(CONFIG["output_folder"][0])
        plot_path = output_dir / f'{dataset_name}_summary_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"✓ Summary plots saved: {plot_path}")
        
    except Exception as e:
        print(f"✗ Error creating summary plots: {e}")


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("DUAL-CHANNEL BACTERIA DETECTION WITH FLUORESCENCE MATCHING")
    print("Using LARGER contour + BRIGHT, SHARP fluorescence overlay")
    print("Fluorescence overlay FULLY COVERS selected contour region")
    print("="*70)
    
    try:
        # Validate configuration
        validate_config()
        
        # Extract dataset name from first source folder
        dataset_name = Path(CONFIG["source_folder"][0]).name
        if not dataset_name:
            dataset_name = "default"
        
        # Setup output folder
        output_dir = Path(CONFIG["output_folder"][0])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV filenames
        master_csv_filename = output_dir / f"{dataset_name}_bacterial_morphology_master.csv"
        non_matching_csv_filename = output_dir / f"{dataset_name}_grayscale_only_bacteria.csv"
        
        # Load image pairs
        gray_images, fluor_images, image_paths = load_image_pairs()
        total_images = len(gray_images)
        
        # Initialize counters and data storage
        bacterium_counter = 1
        all_data = []
        all_non_matching = []
        processed_count = 0
        failed_count = 0
        
        # Process each image pair
        print(f"\n{'='*70}")
        print(f"PROCESSING {total_images} IMAGE PAIR(S)")
        print(f"{'='*70}\n")
        
        for idx, (gray_img, fluor_img, image_path) in enumerate(zip(gray_images, fluor_images, image_paths), 1):
            try:
                filename = os.path.basename(image_path)
                print(f"[{idx}/{total_images}] Processing: {filename}")
                
                # Preprocess grayscale image
                gray_preprocessed = preprocess_grayscale(gray_img)
                
                # Segment grayscale
                gray_binary, gray_labels = segment_grayscale(gray_preprocessed)
                
                # Postprocess grayscale
                gray_final = postprocess(gray_binary)
                
                # Analyze grayscale
                gray_labeled, gray_props_all, gray_props = analyze_and_get_large(
                    gray_final, CONFIG["min_area"], "GRAYSCALE"
                )
                
                # Preprocess fluorescence image
                fluor_preprocessed = preprocess_fluorescence(fluor_img)
                
                # Segment fluorescence
                fluor_binary, fluor_labels = segment_fluorescence(fluor_preprocessed)
                
                # Postprocess fluorescence
                fluor_final = postprocess(fluor_binary)
                
                # Analyze fluorescence
                fluor_labeled, fluor_props_all, fluor_props = analyze_and_get_large(
                    fluor_final, CONFIG["min_fluor_area"], "FLUORESCENCE"
                )
                
                if not gray_props:
                    print(f"  ⚠ No grayscale bacteria meeting criteria found")
                    processed_count += 1
                    continue
                
                # Match fluorescence to grayscale bacteria and get contours
                matches, gray_contours, fluor_contours = match_fluorescence_to_grayscale(
                    gray_props, fluor_props, gray_labeled, fluor_labeled
                )
                
                # Generate output filename
                prefix = Path(filename).stem.split("_")[0]
                output_filename = f"{prefix}_dual_channel_{idx}.png"
                
                # Visualize with dual-channel matching (using larger contour)
                data_this, non_matching_this, bacterium_counter = visualize_dual_channel(
                    gray_img, fluor_img, gray_labeled, fluor_labeled,
                    gray_props, fluor_props, matches, gray_contours, fluor_contours,
                    image_path, idx, bacterium_counter, output_path=output_filename
                )
                
                all_data.extend(data_this)
                all_non_matching.extend(non_matching_this)
                processed_count += 1
                print(f"  ✓ [{idx}/{total_images}] Completed: {len(data_this)} bacteria detected "
                      f"({len(data_this) - len(non_matching_this)} matched, {len(non_matching_this)} grayscale-only)\n")
                
            except Exception as e:
                print(f"  ✗ [{idx}/{total_images}] Failed: {e}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                continue
        
        # Save results
        print(f"\n{'='*70}")
        print("SAVING RESULTS")
        print(f"{'='*70}\n")
        
        if all_data:
            # Save master CSV
            save_master_csv(all_data, master_csv_filename)
            
            # Save non-matching bacteria CSV
            if all_non_matching:
                save_non_matching_csv(all_non_matching, non_matching_csv_filename)
            
            # Create summary plots
            create_summary_plots(all_data, all_non_matching, dataset_name)
            
            # Print summary statistics
            print_summary_statistics(all_data, all_non_matching, dataset_name)
        else:
            print("⚠ No bacteria detected in any images. No CSV or plots generated.\n")
        
        # Final summary
        print(f"{'='*70}")
        print("PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Successfully processed: {processed_count}/{total_images}")
        print(f"Failed: {failed_count}/{total_images}")
        print(f"Total bacteria detected: {len(all_data)}")
        print(f"Matched bacteria: {len(all_data) - len(all_non_matching)}")
        print(f"Grayscale-only bacteria: {len(all_non_matching)}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())