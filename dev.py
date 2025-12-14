"""
Particle Scout - Advanced Particle Analysis Tool
Analyzes fluorescent particles from microscopy images with QuPath ROI support
"""

import cv2
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from datetime import datetime
import exporter as exp


# ============================================================================
# TUNABLE PARAMETERS
# ============================================================================

# Particle detection
MIN_PARTICLE_AREA_UM2 = 0.5
MAX_PARTICLE_AREA_UM2 = 500
MIN_CIRCULARITY = 0.3

# Red channel visualization
RED_NORMALIZE = True
RED_BRIGHTNESS = 1.2
RED_GAMMA = 0.7

# Scale bar parameters
SCALE_BAR_LENGTH_UM = 10
SCALE_BAR_HEIGHT = 4
SCALE_BAR_MARGIN = 15
SCALE_BAR_COLOR = (255, 255, 255)
SCALE_BAR_BG_COLOR = (0, 0, 0)
SCALE_BAR_TEXT_COLOR = (255, 255, 255)
SCALE_BAR_FONT_SCALE = 0.5
SCALE_BAR_FONT_THICKNESS = 1

# Outlier trimming
ENABLE_TRIMMING = True
TRIM_PERCENTAGE = 0.20  # Remove top and bottom 20%

# Visualization
OVERLAY_ALPHA = 0.4
CONTOUR_THICKNESS = 2
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.5
LABEL_THICKNESS = 1

# Mandatory control group name
CONTROL_GROUP_NAME = 'Control group'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def add_scale_bar(img, pixel_size, unit='um', length_um=SCALE_BAR_LENGTH_UM):
    """Add a scale bar to the image"""
    if pixel_size is None or pixel_size <= 0:
        return img
    
    bar_length_px = int(round(length_um / pixel_size))
    
    if bar_length_px < 10:
        return img
    
    h, w = img.shape[:2]
    bar_x = w - bar_length_px - SCALE_BAR_MARGIN
    bar_y = h - SCALE_BAR_HEIGHT - SCALE_BAR_MARGIN
    
    if unit == 'µm' or unit == 'um':
        label = f"{length_um} um"
    else:
        label = f"{length_um} {unit}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(
        label, font, SCALE_BAR_FONT_SCALE, SCALE_BAR_FONT_THICKNESS
    )
    
    text_x = bar_x + (bar_length_px - text_w) // 2
    text_y = bar_y - 8
    
    bg_padding = 5
    bg_x1 = min(bar_x, text_x) - bg_padding
    bg_y1 = text_y - text_h - bg_padding
    bg_x2 = max(bar_x + bar_length_px, text_x + text_w) + bg_padding
    bg_y2 = bar_y + SCALE_BAR_HEIGHT + bg_padding
    
    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), SCALE_BAR_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    cv2.rectangle(img, (bar_x, bar_y), 
                  (bar_x + bar_length_px, bar_y + SCALE_BAR_HEIGHT),
                  SCALE_BAR_COLOR, -1)
    
    cv2.putText(img, label, (text_x, text_y), 
                font, SCALE_BAR_FONT_SCALE, SCALE_BAR_TEXT_COLOR, 
                SCALE_BAR_FONT_THICKNESS, cv2.LINE_AA)
    
    return img


def parse_qupath_roi(roi_path):
    """Parse QuPath ROI XML file to extract polygon coordinates"""
    try:
        tree = ET.parse(roi_path)
        root = tree.getroot()
        
        coords = []
        for vertex in root.findall('.//Vertex'):
            x_str = vertex.get('X')
            y_str = vertex.get('Y')
            
            if x_str is None or y_str is None:
                print(f"⚠ Warning: Vertex missing X or Y coordinate in {roi_path}")
                continue
            
            try:
                x = float(x_str)
                y = float(y_str)
                coords.append((int(round(x)), int(round(y))))
            except (ValueError, TypeError) as e:
                print(f"⚠ Warning: Invalid coordinate value in {roi_path}: {e}")
                continue
        
        if len(coords) < 3:
            print(f"⚠ Warning: ROI has fewer than 3 vertices: {roi_path}")
            return None
        
        return coords
    
    except Exception as e:
        print(f"✗ Error parsing ROI file {roi_path}: {e}")
        return None


def create_roi_mask(coords, image_shape):
    """Create binary mask from ROI coordinates"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(coords, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def adjust_red_channel(img, normalize=True, brightness=1.0, gamma=1.0):
    """Adjust red channel with normalization, brightness, and gamma"""
    img_float = img.astype(np.float32)
    
    if normalize:
        min_val = np.min(img_float)
        max_val = np.max(img_float)
        if max_val > min_val:
            img_float = (img_float - min_val) * 255.0 / (max_val - min_val)
    
    if brightness != 1.0:
        img_float = img_float * brightness
        img_float = np.clip(img_float, 0, 255)
    
    if gamma != 1.0:
        img_normalized = img_float / 255.0
        img_normalized = np.power(img_normalized, gamma)
        img_float = img_normalized * 255.0
    
    return img_float.astype(np.uint8)


def find_image_groups(source_dir):
    """Find all available group directories in source directory"""
    source_path = Path(source_dir)
    if not source_path.exists():
        return []
    
    groups = [d.name for d in source_path.iterdir() 
              if d.is_dir() and not d.name.startswith('.') and d.name != CONTROL_GROUP_NAME]
    return sorted(groups)


def prompt_for_sample_group(source_dir):
    """Interactively prompt user to select sample group"""
    available_groups = find_image_groups(source_dir)
    
    if not available_groups:
        print(f"\n❌ No sample group directories found in {source_dir}")
        print(f"   (Only found '{CONTROL_GROUP_NAME}' or no directories)")
        return None
    
    print(f"\n{'='*60}")
    print("AVAILABLE SAMPLE GROUPS")
    print(f"{'='*60}")
    for group in available_groups:
        print(f"  • {group}")
    print(f"{'='*60}\n")
    
    while True:
        sample_group = input("Enter SAMPLE group folder name: ").strip()
        
        if not sample_group:
            print("❌ Sample group name cannot be empty. Please try again.")
            continue
        
        if sample_group in available_groups:
            print(f"✓ Selected sample group: {sample_group}")
            return sample_group
        else:
            print(f"❌ Group '{sample_group}' not found.")
            print(f"   Available groups: {', '.join(available_groups)}")
            retry = input("   Try again? [Y/n]: ").strip().lower()
            if retry in ['n', 'no']:
                return None


def trim_outliers(data, trim_percentage=TRIM_PERCENTAGE):
    """
    Remove top and bottom percentages of data based on intensity_per_area
    
    Args:
        data: List of particle dictionaries
        trim_percentage: Percentage to trim from each end (default 0.20 = 20%)
    
    Returns:
        Trimmed list of particles (middle 60% by default)
    """
    if not data:
        return []
    
    sorted_data = sorted(data, key=lambda x: x['intensity_per_area'])
    
    n = len(sorted_data)
    lower_cutoff = int(n * trim_percentage)
    upper_cutoff = int(n * (1 - trim_percentage))
    
    trimmed = sorted_data[lower_cutoff:upper_cutoff]
    
    print(f"  Original particles: {n}")
    print(f"  Removed bottom {trim_percentage*100:.0f}%: {lower_cutoff} particles")
    print(f"  Removed top {trim_percentage*100:.0f}%: {n - upper_cutoff} particles")
    print(f"  Retained: {len(trimmed)} particles ({len(trimmed)/n*100:.1f}%)")
    
    return trimmed


# ============================================================================
# PARTICLE DETECTION AND ANALYSIS
# ============================================================================

def detect_particles(image, roi_mask, pixels_per_um, 
                     min_area_um2=MIN_PARTICLE_AREA_UM2,
                     max_area_um2=MAX_PARTICLE_AREA_UM2,
                     min_circularity=MIN_CIRCULARITY):
    """Detect and analyze particles in fluorescence image within ROI"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Store original image for intensity measurements
    original_img = gray.copy()
    
    # Enhance red channel for better detection
    enhanced = adjust_red_channel(gray, RED_NORMALIZE, RED_BRIGHTNESS, RED_GAMMA)
    
    # Apply ROI mask
    masked = cv2.bitwise_and(enhanced, enhanced, mask=roi_mask)
    
    # Thresholding - Otsu's method
    _, binary = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    particles = []
    particle_id = 1
    
    # Convert area thresholds to pixels
    min_area_px = min_area_um2 * (pixels_per_um ** 2)
    max_area_px = max_area_um2 * (pixels_per_um ** 2)
    
    for contour in contours:
        area_px = cv2.contourArea(contour)
        
        if area_px < min_area_px or area_px > max_area_px:
            continue
        
        perimeter_px = cv2.arcLength(contour, True)
        if perimeter_px == 0:
            continue
        
        circularity = 4 * np.pi * area_px / (perimeter_px ** 2)
        
        if circularity < min_circularity:
            continue
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Create mask for this particle
        particle_mask = np.zeros_like(gray)
        cv2.drawContours(particle_mask, [contour], -1, 255, -1)
        
        # Calculate intensity measurements on ORIGINAL image
        particle_pixels = original_img[particle_mask == 255]
        
        if len(particle_pixels) == 0:
            continue
        
        mean_intensity = float(np.mean(particle_pixels))
        total_intensity = float(np.sum(particle_pixels.astype(np.float64)))
        std_intensity = float(np.std(particle_pixels))
        
        # Convert measurements to micrometers
        area_um2 = area_px / (pixels_per_um ** 2)
        perimeter_um = perimeter_px / pixels_per_um
        
        # Intensity per area
        intensity_per_area = total_intensity / area_um2 if area_um2 > 0 else 0.0
        
        # Store particle data
        particle = {
            'id': particle_id,
            'contour': contour,
            'centroid': (cx, cy),
            'area_px': area_px,
            'area_um2': area_um2,
            'perimeter_px': perimeter_px,
            'perimeter_um': perimeter_um,
            'circularity': circularity,
            'mean_intensity': mean_intensity,
            'total_intensity': total_intensity,
            'std_intensity': std_intensity,
            'intensity_per_area': intensity_per_area
        }
        
        particles.append(particle)
        particle_id += 1
    
    return particles


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_overlay_image(image, particles, roi_coords, save_path, pixels_per_um=None):
    """Create visualization with detected particles and ROI overlay"""
    # Convert to BGR if grayscale
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()
    
    # Draw ROI boundary in cyan
    roi_pts = np.array(roi_coords, dtype=np.int32)
    cv2.polylines(overlay, [roi_pts], True, (255, 255, 0), 2)
    
    # Draw each particle
    for particle in particles:
        # Draw contour in green
        cv2.drawContours(overlay, [particle['contour']], -1, (0, 255, 0), CONTOUR_THICKNESS)
        
        # Draw centroid
        cv2.circle(overlay, particle['centroid'], 3, (0, 0, 255), -1)
        
        # Add label with particle ID
        label = f"{particle['id']}"
        label_pos = (particle['centroid'][0] + 5, particle['centroid'][1] - 5)
        cv2.putText(overlay, label, label_pos, LABEL_FONT, LABEL_FONT_SCALE, 
                   (255, 255, 255), LABEL_THICKNESS, cv2.LINE_AA)
    
    # Add scale bar if pixel size is available
    if pixels_per_um is not None and pixels_per_um > 0:
        overlay = add_scale_bar(overlay, 1.0 / pixels_per_um, 'um', SCALE_BAR_LENGTH_UM)
    
    # Save overlay
    cv2.imwrite(save_path, overlay)
    print(f"  ✓ Overlay saved: {save_path}")
    
    return overlay


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_image_pair(green_path, red_path, roi_path, pixels_per_um, output_dir, 
                       group_name, image_index):
    """Process a pair of green/red channel images with ROI"""
    print(f"\n{'='*60}")
    print(f"Processing {group_name.upper()} - Image Pair {image_index}")
    print(f"{'='*60}")
    
    # Load images
    green_img = cv2.imread(green_path, cv2.IMREAD_GRAYSCALE)
    red_img = cv2.imread(red_path, cv2.IMREAD_GRAYSCALE)
    
    if green_img is None or red_img is None:
        print(f"✗ Error: Could not load images")
        return []
    
    print(f"  ✓ Loaded images: {green_img.shape}")
    
    # Parse ROI
    roi_coords = parse_qupath_roi(roi_path)
    if roi_coords is None:
        print(f"✗ Error: Could not parse ROI")
        return []
    
    print(f"  ✓ Loaded ROI with {len(roi_coords)} vertices")
    
    # Create ROI mask
    roi_mask = create_roi_mask(roi_coords, green_img.shape)
    
    # Detect particles in red channel
    particles = detect_particles(red_img, roi_mask, pixels_per_um)
    
    print(f"  ✓ Detected {len(particles)} particles")
    
    # Create output subdirectory for this image pair
    pair_dir = os.path.join(output_dir, f"{group_name}_pair_{image_index}")
    os.makedirs(pair_dir, exist_ok=True)
    
    # Create overlay visualization with scale bar
    overlay_path = os.path.join(pair_dir, f"overlay_{group_name}_{image_index}.png")
    overlay_img = create_overlay_image(red_img, particles, roi_coords, overlay_path, pixels_per_um)
    
    # Save particle data summary
    summary_path = os.path.join(pair_dir, f"particles_{group_name}_{image_index}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Particle Analysis Summary - {group_name.upper()} Pair {image_index}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Total particles detected: {len(particles)}\n\n")
        f.write(f"{'ID':<5} {'Area(um²)':<12} {'Circularity':<12} {'Intensity/Area':<15}\n")
        f.write(f"{'-'*60}\n")
        for p in particles:
            f.write(f"{p['id']:<5} {p['area_um2']:<12.2f} {p['circularity']:<12.3f} "
                   f"{p['intensity_per_area']:<15.2f}\n")
    
    print(f"  ✓ Summary saved: {summary_path}")
    
    # Store overlay image in particle data for Excel export
    for p in particles:
        p['overlay_image'] = overlay_img
        p['image_name'] = f"{group_name}_pair_{image_index}"
    
    return particles


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Particle Scout - Advanced Particle Analysis')
    parser.add_argument('--input', '-i', default='source', help='Input source directory')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--pixels-per-um', '-p', type=float, default=1.0, 
                       help='Pixels per micrometer (default: 1.0)')
    parser.add_argument('--sample-group', '-s', help='Sample group directory name')
    parser.add_argument('--no-trim', action='store_true', 
                       help='Disable outlier trimming (enabled by default)')
    parser.add_argument('--trim-percentage', type=float, default=TRIM_PERCENTAGE,
                       help=f'Percentage to trim from each end (default: {TRIM_PERCENTAGE})')
    
    args = parser.parse_args()
    
    source_dir = args.input
    pixels_per_um = args.pixels_per_um
    control_group = CONTROL_GROUP_NAME
    enable_trimming = not args.no_trim
    trim_percentage = args.trim_percentage
    
    # Interactive sample group selection if not specified
    if not args.sample_group:
        sample_group = prompt_for_sample_group(source_dir)
        if sample_group is None:
            print("\n❌ Sample group selection cancelled or failed")
            return
    else:
        sample_group = args.sample_group
    
    # Verify control group exists
    control_path = Path(source_dir) / control_group
    if not control_path.exists():
        print(f"\n❌ Control group directory not found: {control_path}")
        print(f"   Expected: {source_dir}/{control_group}")
        return
    
    # Verify sample group exists
    sample_path = Path(source_dir) / sample_group
    if not sample_path.exists():
        print(f"\n❌ Sample group directory not found: {sample_path}")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("PARTICLE SCOUT - ANALYSIS STARTED")
    print("="*60)
    print(f"Source directory: {source_dir}")
    print(f"Control group: {control_group}")
    print(f"Sample group: {sample_group}")
    print(f"Output directory: {output_dir}")
    print(f"Pixels per μm: {pixels_per_um}")
    print(f"Outlier trimming: {'Enabled' if enable_trimming else 'Disabled'}")
    if enable_trimming:
        print(f"Trim percentage: {trim_percentage*100:.0f}% each end")
    
    # Find all green/red image pairs and ROIs
    control_dir = str(control_path)
    sample_dir = str(sample_path)
    
    control_green = sorted(glob.glob(os.path.join(control_dir, '*_green.tif')))
    control_red = sorted(glob.glob(os.path.join(control_dir, '*_red.tif')))
    control_rois = sorted(glob.glob(os.path.join(control_dir, '*.xml')))
    
    sample_green = sorted(glob.glob(os.path.join(sample_dir, '*_green.tif')))
    sample_red = sorted(glob.glob(os.path.join(sample_dir, '*_red.tif')))
    sample_rois = sorted(glob.glob(os.path.join(sample_dir, '*.xml')))
    
    print(f"\nFound {len(control_green)} control image pairs")
    print(f"Found {len(sample_green)} sample image pairs")
    
    # Process control group
    control_particles = []
    for i, (green, red, roi) in enumerate(zip(control_green, control_red, control_rois), 1):
        particles = process_image_pair(green, red, roi, pixels_per_um, output_dir, 'control', i)
        control_particles.extend(particles)
    
    # Process sample group
    sample_particles = []
    for i, (green, red, roi) in enumerate(zip(sample_green, sample_red, sample_rois), 1):
        particles = process_image_pair(green, red, roi, pixels_per_um, output_dir, sample_group, i)
        sample_particles.extend(particles)
    
    # Apply trimming if enabled
    control_trimmed = control_particles
    sample_trimmed = sample_particles
    
    if enable_trimming:
        print(f"\n{'='*60}")
        print("APPLYING OUTLIER TRIMMING")
        print(f"{'='*60}")
        
        if control_particles:
            print(f"\nControl group:")
            control_trimmed = trim_outliers(control_particles, trim_percentage)
        
        if sample_particles:
            print(f"\n{sample_group}:")
            sample_trimmed = trim_outliers(sample_particles, trim_percentage)
    
    # Generate summary statistics
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE - SUMMARY")
    print(f"{'='*60}")
    print(f"Control particles: {len(control_particles)}")
    print(f"Sample particles: {len(sample_particles)}")
    
    if enable_trimming:
        print(f"\nAfter trimming:")
        print(f"Control particles: {len(control_trimmed)}")
        print(f"Sample particles: {len(sample_trimmed)}")
    
    if control_trimmed:
        control_intensities = [p['intensity_per_area'] for p in control_trimmed]
        print(f"\nControl Intensity/Area Statistics:")
        print(f"  Mean: {np.mean(control_intensities):.2f}")
        print(f"  SD: {np.std(control_intensities, ddof=1):.2f}")
        print(f"  SEM: {np.std(control_intensities, ddof=1) / np.sqrt(len(control_intensities)):.2f}")
    
    if sample_trimmed:
        sample_intensities = [p['intensity_per_area'] for p in sample_trimmed]
        print(f"\n{sample_group} Intensity/Area Statistics:")
        print(f"  Mean: {np.mean(sample_intensities):.2f}")
        print(f"  SD: {np.std(sample_intensities, ddof=1):.2f}")
        print(f"  SEM: {np.std(sample_intensities, ddof=1) / np.sqrt(len(sample_intensities)):.2f}")
    
    # Generate reports using exporter module
    print(f"\n{'='*60}")
    print("GENERATING REPORTS")
    print(f"{'='*60}")
    
    # Create charts directory
    charts_dir = os.path.join(output_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    unit_str = 'um'
    
    # Generate comparison chart
    comparison_chart_path = os.path.join(charts_dir, 'comparison_chart.png')
    comparison_img = exp.create_comparison_chart(
        control_trimmed, sample_trimmed, sample_group,
        comparison_chart_path, unit_str
    )
    
    # Generate histograms
    control_hist_path = os.path.join(charts_dir, 'control_histogram.png')
    control_hist_img = exp.create_intensity_histogram(
        control_trimmed, 'Control', control_hist_path, unit_str
    )
    
    sample_hist_path = os.path.join(charts_dir, f'{sample_group}_histogram.png')
    sample_hist_img = exp.create_intensity_histogram(
        sample_trimmed, sample_group, sample_hist_path, unit_str
    )
    
    # Create Excel report
    excel_path = os.path.join(output_dir, f'particle_analysis_report_{timestamp}.xlsx')
    exp.create_excel_report(
        control_trimmed, sample_trimmed, sample_group, excel_path,
        comparison_chart_img=comparison_img,
        control_hist_img=control_hist_img,
        sample_hist_img=sample_hist_img,
        unit_str=unit_str,
        pixels_per_um=pixels_per_um
    )
    
    print(f"\n{'='*60}")
    print("ALL PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()