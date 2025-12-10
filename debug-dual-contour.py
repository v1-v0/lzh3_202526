import cv2
import numpy as np
import os
import glob
import csv

# --------------------------------------------------
# TUNABLE PARAMETERS
# --------------------------------------------------
GAUSSIAN_SIGMA = 15
MORPH_ITERATIONS = 1
DILATE_ITERATIONS = 1
ERODE_ITERATIONS = 1
MIN_OBJECT_AREA = 100
MAX_OBJECT_AREA = 5000

# Red channel visualization parameters
RED_NORMALIZE = True      # Use normalization for contrast enhancement
RED_BRIGHTNESS = 1.2      # Multiply intensity after normalization (1.0 = original)
RED_GAMMA = 0.7           # Gamma correction (1.0 = original, <1 = brighter, >1 = darker)

# --------------------------------------------------
# DEBUG FOLDER SETUP
# --------------------------------------------------
DEBUG_DIR = "debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

for f in glob.glob(os.path.join(DEBUG_DIR, "*")):
    try:
        os.remove(f)
    except OSError:
        pass

def save_debug(name, img):
    path = os.path.join(DEBUG_DIR, name)
    cv2.imwrite(path, img)

def adjust_red_channel(img, normalize=True, brightness=1.0, gamma=1.0):
    """
    Adjust red channel with normalization, brightness, and gamma
    """
    # Convert to float for processing
    img_float = img.astype(np.float32)
    
    # Apply normalization for contrast enhancement
    if normalize:
        # Stretch histogram to use full 0-255 range
        min_val = np.min(img_float)
        max_val = np.max(img_float)
        if max_val > min_val:
            img_float = (img_float - min_val) * 255.0 / (max_val - min_val)
        print(f"Red channel normalized: [{min_val:.1f}, {max_val:.1f}] -> [0, 255]")
    
    # Apply brightness
    if brightness != 1.0:
        img_float = img_float * brightness
        img_float = np.clip(img_float, 0, 255)
    
    # Apply gamma correction
    if gamma != 1.0:
        # Normalize to 0-1 range
        img_normalized = img_float / 255.0
        # Apply gamma
        img_normalized = np.power(img_normalized, gamma)
        # Back to 0-255 range
        img_float = img_normalized * 255.0
    
    return img_float.astype(np.uint8)

# --------------------------------------------------
# LOAD IMAGES
# --------------------------------------------------
grey_path = "source/12/12 N NO 1_ch00.tif"
red_path = "source/12/12 N NO 1_ch01.tif"

# Load brightfield (for contour detection)
img_bf = cv2.imread(grey_path, cv2.IMREAD_UNCHANGED)
if img_bf is None:
    raise FileNotFoundError(grey_path)

if img_bf.ndim == 3:
    img_bf = cv2.cvtColor(img_bf, cv2.COLOR_BGR2GRAY)

print(f"Brightfield loaded: dtype={img_bf.dtype}, shape={img_bf.shape}, range=[{img_bf.min()}, {img_bf.max()}]")

# Load red fluorescence channel
img_red = cv2.imread(red_path, cv2.IMREAD_UNCHANGED)
if img_red is None:
    raise FileNotFoundError(red_path)

if img_red.ndim == 3:
    img_red = cv2.cvtColor(img_red, cv2.COLOR_BGR2GRAY)

print(f"Red channel loaded: dtype={img_red.dtype}, shape={img_red.shape}, range=[{img_red.min()}, {img_red.max()}]")

# --------------------------------------------------
# CONVERT BOTH CHANNELS TO 8-BIT
# --------------------------------------------------
if img_bf.dtype == np.uint16:
    img_bf_8bit = np.zeros_like(img_bf, dtype=np.uint8)
    cv2.normalize(img_bf, img_bf_8bit, 0, 255, cv2.NORM_MINMAX)
    img_bf = img_bf_8bit
    save_debug("01a_bf_converted_8bit.png", img_bf)
    print(f"Brightfield converted: dtype={img_bf.dtype}, range=[{img_bf.min()}, {img_bf.max()}]")
else:
    print("Brightfield already 8-bit")

if img_red.dtype == np.uint16:
    img_red_8bit = np.zeros_like(img_red, dtype=np.uint8)
    cv2.normalize(img_red, img_red_8bit, 0, 255, cv2.NORM_MINMAX)
    img_red = img_red_8bit
    save_debug("01b_red_converted_8bit.png", img_red)
    print(f"Red channel converted: dtype={img_red.dtype}, range=[{img_red.min()}, {img_red.max()}]")
else:
    print("Red channel already 8-bit")

# --------------------------------------------------
# APPLY ENHANCEMENT TO RED CHANNEL
# --------------------------------------------------
img_red_enhanced = adjust_red_channel(
    img_red, 
    normalize=RED_NORMALIZE,
    brightness=RED_BRIGHTNESS, 
    gamma=RED_GAMMA
)
save_debug("01c_red_enhanced.png", img_red_enhanced)

# --------------------------------------------------
# BRIGHTFIELD PROCESSING (unchanged)
# --------------------------------------------------
bg = cv2.GaussianBlur(img_bf, (0, 0), sigmaX=GAUSSIAN_SIGMA, sigmaY=GAUSSIAN_SIGMA)
enhanced = cv2.subtract(bg, img_bf)
enhanced_blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

save_debug("02_enhanced.png", enhanced)
save_debug("03_enhanced_blur.png", enhanced_blur)

_, thresh = cv2.threshold(
    enhanced_blur, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

save_debug("04_thresh_raw.png", thresh)

kernel = np.ones((3, 3), np.uint8)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)
closed = cv2.dilate(closed, kernel, iterations=DILATE_ITERATIONS)
closed = cv2.erode(closed, kernel, iterations=ERODE_ITERATIONS)

save_debug("05_closed.png", closed)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    closed, connectivity=8
)

solid = np.where(labels > 0, 255, 0).astype(np.uint8)
save_debug("06_solid.png", solid)

contours, hierarchy = cv2.findContours(
    solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
print(f"Clumps found: {len(contours)}")

min_area = MIN_OBJECT_AREA
max_area = MAX_OBJECT_AREA
filtered = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
print(f"Filtered objects (by area): {len(filtered)}")

# --------------------------------------------------
# MEASURE RED FLUORESCENCE INTENSITY (BOTH CHANNELS)
# --------------------------------------------------
print(f"\nMeasuring red fluorescence intensity from both channels...")

object_data = []
objects_without_red = 0

for c in filtered:
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    
    # Calculate centroid for labeling
    M = cv2.moments(c)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    
    # Create mask for this contour
    mask = np.zeros_like(img_red, dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, -1)
    
    # ===== ORIGINAL CHANNEL MEASUREMENTS =====
    red_pixels_orig = img_red[mask == 255]
    if len(red_pixels_orig) > 0:
        red_pixels_float = red_pixels_orig.astype(np.float32)
        red_total_orig = float(np.sum(red_pixels_float))
    else:
        red_total_orig = 0.0
    
    # ===== FILTER OUT OBJECTS WITH NO RED INTENSITY =====
    if red_total_orig == 0.0:
        objects_without_red += 1
        continue  # Skip this object
    
    intensity_per_area_orig = red_total_orig / area if area > 0 else 0.0
    
    # ===== ENHANCED CHANNEL MEASUREMENTS =====
    red_pixels_enh = img_red_enhanced[mask == 255]
    if len(red_pixels_enh) > 0:
        red_pixels_float = red_pixels_enh.astype(np.float32)
        red_total_enh = float(np.sum(red_pixels_float))
    else:
        red_total_enh = 0.0
    
    intensity_per_area_enh = red_total_enh / area if area > 0 else 0.0
    
    object_data.append({
        'contour': c,
        'area': area,
        'perimeter': perimeter,
        'centroid_x': cx,
        'centroid_y': cy,
        # Original channel measurements
        'red_total_orig': red_total_orig,
        'intensity_per_area_orig': intensity_per_area_orig,
        # Enhanced channel measurements
        'red_total_enh': red_total_enh,
        'intensity_per_area_enh': intensity_per_area_enh,
    })

print(f"Objects with red intensity: {len(object_data)}")
print(f"Objects without red intensity (removed): {objects_without_red}")

# --------------------------------------------------
# SORT BY ORIGINAL INTENSITY PER AREA (DESCENDING)
# --------------------------------------------------
object_data.sort(key=lambda x: x['intensity_per_area_orig'], reverse=True)

# Assign Object_IDs based on sorted order (original channel)
for i, obj in enumerate(object_data, 1):
    obj['object_id'] = i

print(f"Objects sorted by Fluorescence Intensity per Area (original channel, descending)")

# --------------------------------------------------
# EXPORT STATISTICS - ORIGINAL CHANNEL
# --------------------------------------------------
stats_file_orig = os.path.join(DEBUG_DIR, "fluorescence_stats_original.csv")

with open(stats_file_orig, 'w', newline='') as f:
    writer = csv.writer(f)
    # Header row
    writer.writerow(['Particle ID', 'Area (px²)', 'Perimeter (px)', 'Total Fluorescence', 'Fluorescence Intensity (per px²)'])
    # Separator row for better readability
    writer.writerow(['----------', '----------', '-------------', '------------------', '------------------------------'])
    
    for obj in object_data:
        writer.writerow([
            obj['object_id'],
            f"{obj['area']:.1f}",
            f"{obj['perimeter']:.1f}",
            f"{obj['red_total_orig']:.0f}",
            f"{obj['intensity_per_area_orig']:.2f}"
        ])
    
    # Summary row
    writer.writerow([])
    writer.writerow(['SUMMARY', '', '', '', ''])
    writer.writerow(['Total Particles:', len(object_data), '', '', ''])
    writer.writerow(['Avg Area:', f"{np.mean([obj['area'] for obj in object_data]):.1f}", 'px²', '', ''])
    writer.writerow(['Avg Total Fluorescence:', f"{np.mean([obj['red_total_orig'] for obj in object_data]):.1f}", '', '', ''])
    writer.writerow(['Avg Intensity:', f"{np.mean([obj['intensity_per_area_orig'] for obj in object_data]):.2f}", 'per px²', '', ''])

print(f"✓ CSV (original) saved: {stats_file_orig}")

# --------------------------------------------------
# EXPORT STATISTICS - ENHANCED CHANNEL
# --------------------------------------------------
stats_file_enh = os.path.join(DEBUG_DIR, "fluorescence_stats_enhanced.csv")

with open(stats_file_enh, 'w', newline='') as f:
    writer = csv.writer(f)
    # Header row
    writer.writerow(['Particle ID', 'Area (px²)', 'Perimeter (px)', 'Total Fluorescence', 'Fluorescence Intensity (per px²)'])
    # Separator row
    writer.writerow(['----------', '----------', '-------------', '------------------', '------------------------------'])
    
    for obj in object_data:
        writer.writerow([
            obj['object_id'],
            f"{obj['area']:.1f}",
            f"{obj['perimeter']:.1f}",
            f"{obj['red_total_enh']:.0f}",
            f"{obj['intensity_per_area_enh']:.2f}"
        ])
    
    # Summary row
    writer.writerow([])
    writer.writerow(['SUMMARY', '', '', '', ''])
    writer.writerow(['Total Particles:', len(object_data), '', '', ''])
    writer.writerow(['Avg Area:', f"{np.mean([obj['area'] for obj in object_data]):.1f}", 'px²', '', ''])
    writer.writerow(['Avg Total Fluorescence:', f"{np.mean([obj['red_total_enh'] for obj in object_data]):.1f}", '', '', ''])
    writer.writerow(['Avg Intensity:', f"{np.mean([obj['intensity_per_area_enh'] for obj in object_data]):.2f}", 'per px²', '', ''])

print(f"✓ CSV (enhanced) saved: {stats_file_enh}")

# --------------------------------------------------
# EXPORT COMBINED STATISTICS
# --------------------------------------------------
stats_file_combined = os.path.join(DEBUG_DIR, "fluorescence_stats_comparison.csv")

with open(stats_file_combined, 'w', newline='') as f:
    writer = csv.writer(f)
    # Header with sections
    writer.writerow(['', '', '', '--- ORIGINAL CHANNEL ---', '', '--- ENHANCED CHANNEL ---', '', ''])
    writer.writerow(['Particle ID', 'Area (px²)', 'Perimeter (px)', 
                     'Total Fluor.', 'Intensity (per px²)',
                     'Total Fluor.', 'Intensity (per px²)',
                     'Enhancement Ratio'])
    # Separator row
    writer.writerow(['----------', '----------', '-------------', 
                     '------------', '-----------------',
                     '------------', '-----------------',
                     '------------------'])
    
    for obj in object_data:
        ratio = obj['red_total_enh'] / obj['red_total_orig'] if obj['red_total_orig'] > 0 else 0.0
        writer.writerow([
            obj['object_id'],
            f"{obj['area']:.1f}",
            f"{obj['perimeter']:.1f}",
            f"{obj['red_total_orig']:.0f}",
            f"{obj['intensity_per_area_orig']:.2f}",
            f"{obj['red_total_enh']:.0f}",
            f"{obj['intensity_per_area_enh']:.2f}",
            f"{ratio:.2f}x"
        ])
    
    # Summary section
    writer.writerow([])
    writer.writerow(['SUMMARY', '', '', '', '', '', '', ''])
    writer.writerow(['----------', '', '', '', '', '', '', ''])
    writer.writerow(['Total Particles:', len(object_data), '', '', '', '', '', ''])
    writer.writerow(['Particles Excluded (no fluorescence):', objects_without_red, '', '', '', '', '', ''])
    writer.writerow([])
    writer.writerow(['', '', '', '--- ORIGINAL ---', '', '--- ENHANCED ---', '', ''])
    writer.writerow(['Avg Area:', f"{np.mean([obj['area'] for obj in object_data]):.1f}", 'px²', '', '', '', '', ''])
    writer.writerow(['Avg Total Fluorescence:', '', '', 
                     f"{np.mean([obj['red_total_orig'] for obj in object_data]):.1f}", '',
                     f"{np.mean([obj['red_total_enh'] for obj in object_data]):.1f}", '', ''])
    writer.writerow(['Avg Intensity:', '', '', 
                     f"{np.mean([obj['intensity_per_area_orig'] for obj in object_data]):.2f}", 'per px²',
                     f"{np.mean([obj['intensity_per_area_enh'] for obj in object_data]):.2f}", 'per px²', ''])
    writer.writerow([])
    writer.writerow(['Enhancement Settings:', '', '', '', '', '', '', ''])
    writer.writerow(['  Normalize:', RED_NORMALIZE, '', '', '', '', '', ''])
    writer.writerow(['  Brightness:', RED_BRIGHTNESS, '', '', '', '', '', ''])
    writer.writerow(['  Gamma:', RED_GAMMA, '', '', '', '', '', ''])

print(f"✓ CSV (comparison) saved: {stats_file_combined}")
print(f"Total particles analyzed: {len(object_data)}")

# --------------------------------------------------
# VISUALIZATIONS WITH LABELS
# --------------------------------------------------
# 07a: Brightfield with contours and labels
vis_bf = cv2.cvtColor(img_bf, cv2.COLOR_GRAY2BGR)
for obj in object_data:
    cv2.drawContours(vis_bf, [obj['contour']], -1, (0, 0, 255), 1)
    # Add label at centroid
    label = str(obj['object_id'])
    cv2.putText(vis_bf, label, (obj['centroid_x'], obj['centroid_y']), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
save_debug("07a_bf_contours.png", vis_bf)

# --------------------------------------------------
# 07b: Red channel with enhanced contrast (normalization + adjustments)
# --------------------------------------------------
# Create RGB visualization with enhanced red channel
vis_red_display = np.zeros((img_red.shape[0], img_red.shape[1], 3), dtype=np.uint8)
vis_red_display[:, :, 2] = img_red_enhanced  # Red channel (enhanced)
vis_red_display[:, :, 0] = img_red_enhanced // 8  # Slight blue for depth
vis_red_display[:, :, 1] = img_red_enhanced // 8  # Slight green for depth

# Draw contours and labels
for obj in object_data:
    cv2.drawContours(vis_red_display, [obj['contour']], -1, (0, 255, 0), 1)
    # Add label at centroid
    label = str(obj['object_id'])
    cv2.putText(vis_red_display, label, (obj['centroid_x'], obj['centroid_y']), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

save_debug("07b_red_contours.png", vis_red_display)

# Overlay: Red on top of brightfield
vis_overlay = cv2.cvtColor(img_bf, cv2.COLOR_GRAY2BGR)
# Add red channel as red overlay (alpha blending)
red_overlay = np.zeros_like(vis_overlay)
red_overlay[:, :, 2] = img_red  # Red channel
vis_overlay = cv2.addWeighted(vis_overlay, 0.7, red_overlay, 0.3, 0)

all_contours = [obj['contour'] for obj in object_data]
cv2.drawContours(vis_overlay, all_contours, -1, (0, 255, 0), 1)
save_debug("07c_overlay_contours.png", vis_overlay)

# Color-coded by intensity per area (original channel)
vis_intensity = cv2.cvtColor(img_bf, cv2.COLOR_GRAY2BGR)

intensities_per_area = [obj['intensity_per_area_orig'] for obj in object_data]
if intensities_per_area:
    min_int, max_int = min(intensities_per_area), max(intensities_per_area)
    
    for obj in object_data:
        intensity = obj['intensity_per_area_orig']
        # Normalize intensity to 0-1 range
        if max_int > min_int:
            normalized = (intensity - min_int) / (max_int - min_int)
        else:
            normalized = 0.5
        
        # Color from blue (low) to red (high)
        color = (int(255 * normalized), 0, int(255 * (1 - normalized)))
        cv2.drawContours(vis_intensity, [obj['contour']], -1, color, 2)

save_debug("07d_intensity_coded.png", vis_intensity)

print("✓ Processing complete")
print(f"Enhancement settings: Normalize={RED_NORMALIZE}, Brightness={RED_BRIGHTNESS}, Gamma={RED_GAMMA}")