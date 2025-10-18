"""
Interactive Dual-Channel Bacteria Detection Review Tool
Features:
- Side-by-side visualization of grayscale and fluorescence channels
- Real-time threshold adjustment with sliders
- Interactive parameter tuning (min area, overlap threshold, etc.)
- Live contour detection preview
- Individual channel analysis
- Configuration export for main script
- Histogram visualization for threshold guidance
- Detailed contour size and intensity listing
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons
import cv2
from pathlib import Path
from skimage import io, filters, morphology, measure
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import warnings
warnings.filterwarnings('ignore')

class BacteriaReviewTool:
    def __init__(self, source_folder="source//5"):
        """Initialize the review tool"""
        self.source_folder = source_folder
        self.current_index = 0
        
        # Default configuration (matching your main script)
        self.config = {
            "gray_threshold": 60,
            "fluor_threshold": 5,
            "min_area": 400,
            "min_fluor_area": 400,
            "overlap_threshold": 0.80,
            "show_labels": True,
            "show_contours": True,
            "show_filled": False,
        }
        
        # Load image pairs
        self.load_image_pairs()
        
        if not self.image_pairs:
            raise ValueError(f"No image pairs found in {source_folder}")
        
        # Create the interactive interface
        self.create_interface()
        
    def load_image_pairs(self):
        """Load all grayscale and fluorescence image pairs"""
        self.image_pairs = []
        folder_path = Path(self.source_folder)
        
        if not folder_path.exists():
            print(f"Error: Folder '{self.source_folder}' does not exist")
            return
        
        # Find all grayscale files
        gray_files = sorted([
            f for f in os.listdir(self.source_folder)
            if f.endswith('_ch00.tif') and os.path.isfile(os.path.join(self.source_folder, f))
        ])
        
        for gray_file in gray_files:
            fluor_file = gray_file.replace('_ch00.tif', '_ch01.tif')
            gray_path = os.path.join(self.source_folder, gray_file)
            fluor_path = os.path.join(self.source_folder, fluor_file)
            
            if os.path.exists(fluor_path):
                try:
                    gray_img = io.imread(gray_path)
                    fluor_img = io.imread(fluor_path)
                    
                    # Convert to grayscale if needed
                    if len(gray_img.shape) == 3:
                        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)
                    if len(fluor_img.shape) == 3:
                        fluor_img = cv2.cvtColor(fluor_img, cv2.COLOR_RGB2GRAY)
                    
                    self.image_pairs.append({
                        'gray': gray_img,
                        'fluor': fluor_img,
                        'gray_file': gray_file,
                        'fluor_file': fluor_file
                    })
                except Exception as e:
                    print(f"Error loading {gray_file}: {e}")
        
        print(f"Loaded {len(self.image_pairs)} image pair(s)")
    
    def preprocess_grayscale(self, image):
        """Preprocess grayscale image"""
        median_filtered = filters.median(image, morphology.disk(3))
        
        if median_filtered.max() > 0:
            img_normalized = (median_filtered / median_filtered.max() * 255).astype(np.uint8)
        else:
            img_normalized = median_filtered.astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_normalized)
        
        return enhanced
    
    def preprocess_fluorescence(self, image):
        """Preprocess fluorescence image"""
        median_filtered = filters.median(image, morphology.disk(3))
        
        if median_filtered.max() > 0:
            img_normalized = (median_filtered / median_filtered.max() * 255).astype(np.uint8)
        else:
            img_normalized = median_filtered.astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_normalized)
        
        return enhanced
    
    def segment_grayscale(self, preprocessed, threshold):
        """Segment grayscale bacteria (dark objects)"""
        binary = np.asarray(preprocessed) <= threshold
        
        distance_result = ndi.distance_transform_edt(binary)
        if isinstance(distance_result, tuple):
            distance = distance_result[0]
        else:
            distance = distance_result
        distance = np.asarray(distance, dtype=np.float32)
        
        local_max = morphology.local_maxima(distance)
        markers = measure.label(local_max)
        
        labels = watershed(-distance, markers, mask=binary)
        separated = labels > 0
        
        return separated, labels
    
    def segment_fluorescence(self, preprocessed, threshold):
        """Segment fluorescence bacteria (bright objects)"""
        binary = np.asarray(preprocessed) >= threshold
        
        distance_result = ndi.distance_transform_edt(binary)
        if isinstance(distance_result, tuple):
            distance = distance_result[0]
        else:
            distance = distance_result
        distance = np.asarray(distance, dtype=np.float32)
        
        local_max = morphology.local_maxima(distance)
        markers = measure.label(local_max)
        
        labels = watershed(-distance, markers, mask=binary)
        separated = labels > 0
        
        return separated, labels
    
    def postprocess(self, binary_mask):
        """Post-process binary mask"""
        min_size = 50
        cleaned = morphology.remove_small_objects(binary_mask, min_size=min_size)
        filled = ndi.binary_fill_holes(cleaned)
        opened = morphology.binary_opening(filled, morphology.disk(2))
        final = morphology.binary_closing(opened, morphology.disk(2))
        return final
    
    def detect_bacteria(self, image, is_grayscale=True):
        """Detect bacteria in image with current settings"""
        # Preprocess
        if is_grayscale:
            preprocessed = self.preprocess_grayscale(image)
            threshold = self.config['gray_threshold']
            min_area = self.config['min_area']
            binary, labels = self.segment_grayscale(preprocessed, threshold)
        else:
            preprocessed = self.preprocess_fluorescence(image)
            threshold = self.config['fluor_threshold']
            min_area = self.config['min_fluor_area']
            binary, labels = self.segment_fluorescence(preprocessed, threshold)
        
        # Postprocess
        final = self.postprocess(binary)
        
        # Label and get properties
        labeled = measure.label(final)
        props = measure.regionprops(labeled)
        
        # Filter by area
        large_props = [prop for prop in props if prop.area > min_area]
        
        return preprocessed, labeled, props, large_props
    
    def get_contour_intensities(self, image, labeled, props):
        """Get intensity statistics for each contour"""
        contour_data = []
        
        for i, prop in enumerate(props):
            # Get the region mask
            region_mask = (labeled == prop.label)
            
            # Get pixel intensities within the region
            intensities = image[region_mask]
            
            # Calculate statistics
            mean_intensity = np.mean(intensities)
            median_intensity = np.median(intensities)
            min_intensity = np.min(intensities)
            max_intensity = np.max(intensities)
            std_intensity = np.std(intensities)
            
            contour_data.append({
                'id': i + 1,
                'area': prop.area,
                'mean_intensity': mean_intensity,
                'median_intensity': median_intensity,
                'min_intensity': min_intensity,
                'max_intensity': max_intensity,
                'std_intensity': std_intensity,
                'centroid': prop.centroid
            })
        
        return contour_data
    
    def create_interface(self):
        """Create the interactive matplotlib interface"""
        self.fig = plt.figure(figsize=(20, 12))

        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title('Bacteria Detection Review Tool')
        
        # Create grid layout
        gs = self.fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3, 
                                   left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        # Image display axes
        self.ax_gray_orig = self.fig.add_subplot(gs[0:2, 0])
        self.ax_gray_proc = self.fig.add_subplot(gs[0:2, 1])
        self.ax_fluor_orig = self.fig.add_subplot(gs[0:2, 2])
        self.ax_fluor_proc = self.fig.add_subplot(gs[0:2, 3])
        
        # Histogram axes
        self.ax_gray_hist = self.fig.add_subplot(gs[2, 0:2])
        self.ax_fluor_hist = self.fig.add_subplot(gs[2, 2:4])
        
        # Stats text area (expanded for contour details)
        self.ax_stats = self.fig.add_subplot(gs[3:5, :])
        self.ax_stats.axis('off')
        
        # Create sliders and controls
        self.create_controls()
        
        # Initial update
        self.update_display()
        
        plt.show()
    
    def create_controls(self):
        """Create interactive control widgets"""
        # Slider axes (positioned below the main plots)
        slider_left = 0.15
        slider_width = 0.3
        slider_height = 0.02
        slider_spacing = 0.025
        
        base_y = 0.02
        
        # Grayscale threshold slider
        ax_gray_thresh = plt.axes((slider_left, base_y + 5*slider_spacing, slider_width, slider_height))
        self.slider_gray_thresh = Slider(
            ax_gray_thresh, 'Gray Threshold', 0, 255, 
            valinit=self.config['gray_threshold'], valstep=1, color='gray'
        )
        self.slider_gray_thresh.on_changed(self.update_gray_threshold)
        
        # Fluorescence threshold slider
        ax_fluor_thresh = plt.axes((slider_left, base_y + 4*slider_spacing, slider_width, slider_height))
        self.slider_fluor_thresh = Slider(
            ax_fluor_thresh, 'Fluor Threshold', 0, 255, 
            valinit=self.config['fluor_threshold'], valstep=1, color='red'
        )
        self.slider_fluor_thresh.on_changed(self.update_fluor_threshold)
        
        # Min area slider
        ax_min_area = plt.axes((slider_left, base_y + 3*slider_spacing, slider_width, slider_height))
        self.slider_min_area = Slider(
            ax_min_area, 'Min Gray Area', 10, 200, 
            valinit=self.config['min_area'], valstep=5, color='green'
        )
        self.slider_min_area.on_changed(self.update_min_area)
        
        # Min fluor area slider
        ax_min_fluor = plt.axes((slider_left, base_y + 2*slider_spacing, slider_width, slider_height))
        self.slider_min_fluor = Slider(
            ax_min_fluor, 'Min Fluor Area', 5, 100, 
            valinit=self.config['min_fluor_area'], valstep=5, color='orange'
        )
        self.slider_min_fluor.on_changed(self.update_min_fluor_area)
        
        # Overlap threshold slider
        ax_overlap = plt.axes((slider_left, base_y + 1*slider_spacing, slider_width, slider_height))
        self.slider_overlap = Slider(
            ax_overlap, 'Overlap %', 0, 100, 
            valinit=self.config['overlap_threshold']*100, valstep=5, color='blue'
        )
        self.slider_overlap.on_changed(self.update_overlap)
        
        # Navigation buttons
        button_width = 0.08
        button_height = 0.03
        button_y = base_y + 6*slider_spacing
        
        ax_prev = plt.axes((slider_left + slider_width + 0.05, button_y, button_width, button_height))
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_prev.on_clicked(self.prev_image)
        
        ax_next = plt.axes((slider_left + slider_width + 0.14, button_y, button_width, button_height))
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.next_image)
        
        ax_export = plt.axes((slider_left + slider_width + 0.23, button_y, button_width, button_height))
        self.btn_export = Button(ax_export, 'Export Config')
        self.btn_export.on_clicked(self.export_config)
        
        # Checkboxes for display options
        checkbox_x = slider_left + slider_width + 0.33
        ax_checks = plt.axes((checkbox_x, base_y, 0.15, 0.10))
        ax_checks.set_frame_on(False)
        self.check_display = CheckButtons(
            ax_checks, 
            ['Show Labels', 'Show Contours', 'Show Filled'],
            [self.config['show_labels'], self.config['show_contours'], self.config['show_filled']]
        )
        self.check_display.on_clicked(self.update_display_options)
    
    def update_gray_threshold(self, val):
        """Update grayscale threshold"""
        self.config['gray_threshold'] = int(val)
        self.update_display()
    
    def update_fluor_threshold(self, val):
        """Update fluorescence threshold"""
        self.config['fluor_threshold'] = int(val)
        self.update_display()
    
    def update_min_area(self, val):
        """Update minimum grayscale area"""
        self.config['min_area'] = int(val)
        self.update_display()
    
    def update_min_fluor_area(self, val):
        """Update minimum fluorescence area"""
        self.config['min_fluor_area'] = int(val)
        self.update_display()
    
    def update_overlap(self, val):
        """Update overlap threshold"""
        self.config['overlap_threshold'] = val / 100
        self.update_display()
    
    def update_display_options(self, label):
        """Update display options"""
        if label == 'Show Labels':
            self.config['show_labels'] = not self.config['show_labels']
        elif label == 'Show Contours':
            self.config['show_contours'] = not self.config['show_contours']
        elif label == 'Show Filled':
            self.config['show_filled'] = not self.config['show_filled']
        self.update_display()
    
    def prev_image(self, event):
        """Go to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
    
    def next_image(self, event):
        """Go to next image"""
        if self.current_index < len(self.image_pairs) - 1:
            self.current_index += 1
            self.update_display()
    
    def export_config(self, event):
        """Export current configuration"""
        output_file = "bacteria_detection_config.txt"
        with open(output_file, 'w') as f:
            f.write("# Bacteria Detection Configuration\n")
            f.write("# Copy these values to your main script's CONFIG dictionary\n\n")
            f.write("CONFIG = {\n")
            f.write(f'    "gray_threshold": {self.config["gray_threshold"]},\n')
            f.write(f'    "fluor_threshold": {self.config["fluor_threshold"]},\n')
            f.write(f'    "min_area": {self.config["min_area"]},\n')
            f.write(f'    "min_fluor_area": {self.config["min_fluor_area"]},\n')
            f.write(f'    "overlap_threshold": {self.config["overlap_threshold"]:.2f},\n')
            f.write("}\n")
        print(f"✓ Configuration exported to {output_file}")
    
    def update_display(self):
        """Update all display elements"""
        if not self.image_pairs:
            return
        
        pair = self.image_pairs[self.current_index]
        gray_img = pair['gray']
        fluor_img = pair['fluor']
        
        # Detect bacteria
        gray_proc, gray_labeled, gray_props, gray_large = self.detect_bacteria(gray_img, is_grayscale=True)
        fluor_proc, fluor_labeled, fluor_props, fluor_large = self.detect_bacteria(fluor_img, is_grayscale=False)
        
        # Get contour intensity data
        gray_contour_data = self.get_contour_intensities(gray_img, gray_labeled, gray_large)
        fluor_contour_data = self.get_contour_intensities(fluor_img, fluor_labeled, fluor_large)
        
        # Clear all axes
        self.ax_gray_orig.clear()
        self.ax_gray_proc.clear()
        self.ax_fluor_orig.clear()
        self.ax_fluor_proc.clear()
        self.ax_gray_hist.clear()
        self.ax_fluor_hist.clear()
        
        # Display original images
        self.ax_gray_orig.imshow(gray_img, cmap='gray')
        self.ax_gray_orig.set_title(f'Grayscale Original\n{pair["gray_file"]}', fontsize=10)
        self.ax_gray_orig.axis('off')
        
        self.ax_fluor_orig.imshow(fluor_img, cmap='hot')
        self.ax_fluor_orig.set_title(f'Fluorescence Original\n{pair["fluor_file"]}', fontsize=10)
        self.ax_fluor_orig.axis('off')
        
        # Create visualizations with detected bacteria
        gray_vis = self.create_visualization(gray_img, gray_labeled, gray_large, is_grayscale=True)
        fluor_vis = self.create_visualization(fluor_img, fluor_labeled, fluor_large, is_grayscale=False)
        
        self.ax_gray_proc.imshow(gray_vis)
        self.ax_gray_proc.set_title(f'Grayscale Detected: {len(gray_large)} bacteria\nThreshold: {self.config["gray_threshold"]}, Min Area: {self.config["min_area"]}', fontsize=10)
        self.ax_gray_proc.axis('off')
        
        self.ax_fluor_proc.imshow(fluor_vis)
        self.ax_fluor_proc.set_title(f'Fluorescence Detected: {len(fluor_large)} bacteria\nThreshold: {self.config["fluor_threshold"]}, Min Area: {self.config["min_fluor_area"]}', fontsize=10)
        self.ax_fluor_proc.axis('off')
        
        # Plot histograms
        self.ax_gray_hist.hist(gray_img.ravel(), bins=100, color='gray', alpha=0.7, edgecolor='black')
        self.ax_gray_hist.axvline(self.config['gray_threshold'], color='red', linestyle='--', linewidth=2, label=f'Threshold: {self.config["gray_threshold"]}')
        self.ax_gray_hist.set_title('Grayscale Histogram', fontsize=10)
        self.ax_gray_hist.set_xlabel('Intensity')
        self.ax_gray_hist.set_ylabel('Frequency')
        self.ax_gray_hist.legend()
        self.ax_gray_hist.grid(alpha=0.3)
        
        self.ax_fluor_hist.hist(fluor_img.ravel(), bins=100, color='red', alpha=0.7, edgecolor='black')
        self.ax_fluor_hist.axvline(self.config['fluor_threshold'], color='blue', linestyle='--', linewidth=2, label=f'Threshold: {self.config["fluor_threshold"]}')
        self.ax_fluor_hist.set_title('Fluorescence Histogram', fontsize=10)
        self.ax_fluor_hist.set_xlabel('Intensity')
        self.ax_fluor_hist.set_ylabel('Frequency')
        self.ax_fluor_hist.legend()
        self.ax_fluor_hist.grid(alpha=0.3)
        
        # Update statistics with contour details
        self.update_statistics(gray_large, fluor_large, gray_contour_data, fluor_contour_data)
        
        # Update figure
        self.fig.canvas.draw_idle()
    
    def create_visualization(self, image, labeled, props, is_grayscale=True):
        """Create visualization with detected bacteria"""
        # Create RGB image
        if image.max() > 0:
            vis = np.dstack([image, image, image])
            vis = (vis / vis.max() * 255).astype(np.uint8)
        else:
            vis = np.zeros((*image.shape, 3), dtype=np.uint8)
        
        # Draw bacteria
        for i, prop in enumerate(props):
            # Get contour
            region_mask = (labeled == prop.label).astype(np.uint8)
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            
            # Choose color
            if is_grayscale:
                color = (0, 255, 0)  # Green
            else:
                color = (0, 0, 255)  # Red
            
            # Draw filled region
            if self.config['show_filled']:
                cv2.drawContours(vis, [contour], -1, color, -1)
                vis = cv2.addWeighted(vis, 0.7, np.dstack([image, image, image]).astype(np.uint8), 0.3, 0)
            
            # Draw contour
            if self.config['show_contours']:
                cv2.drawContours(vis, [contour], -1, color, 2)
            
            # Draw label
            if self.config['show_labels']:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    label_text = f"{i+1}"
                    cv2.putText(vis, label_text, (cX-10, cY-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return vis
    
    def update_statistics(self, gray_props, fluor_props, gray_contour_data, fluor_contour_data):
        """Update statistics display with detailed contour information"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # Build comprehensive statistics text
        stats_lines = []
        
        # Header
        stats_lines.append(f"Image {self.current_index + 1} of {len(self.image_pairs)} | Config: Gray Thr={self.config['gray_threshold']}, Fluor Thr={self.config['fluor_threshold']}, Gray MinArea={self.config['min_area']}, Fluor MinArea={self.config['min_fluor_area']}, Overlap={self.config['overlap_threshold']*100:.0f}%")
        stats_lines.append("=" * 180)
        
        # Grayscale Channel Details
        stats_lines.append(f"GRAYSCALE CHANNEL - {len(gray_props)} bacteria detected")
        stats_lines.append("-" * 180)
        
        if gray_contour_data:
            stats_lines.append(f"{'ID':<4} {'Area':<8} {'Mean Int':<10} {'Median Int':<11} {'Min Int':<9} {'Max Int':<9} {'Std Dev':<10} {'Centroid (Y, X)':<20}")
            stats_lines.append("-" * 180)
            
            for data in gray_contour_data:
                stats_lines.append(
                    f"{data['id']:<4} "
                    f"{data['area']:<8.0f} "
                    f"{data['mean_intensity']:<10.2f} "
                    f"{data['median_intensity']:<11.2f} "
                    f"{data['min_intensity']:<9.0f} "
                    f"{data['max_intensity']:<9.0f} "
                    f"{data['std_intensity']:<10.2f} "
                    f"({data['centroid'][0]:.1f}, {data['centroid'][1]:.1f})"
                )
            
            # Summary statistics
            gray_areas = [d['area'] for d in gray_contour_data]
            gray_mean_ints = [d['mean_intensity'] for d in gray_contour_data]
            stats_lines.append("-" * 180)
            stats_lines.append(f"Summary: Area (mean={np.mean(gray_areas):.1f}, median={np.median(gray_areas):.1f}, std={np.std(gray_areas):.1f}) | "
                             f"Intensity (mean={np.mean(gray_mean_ints):.2f}, median={np.median(gray_mean_ints):.2f}, std={np.std(gray_mean_ints):.2f})")
        else:
            stats_lines.append("No bacteria detected in grayscale channel")
        
        stats_lines.append("")
        
        # Fluorescence Channel Details
        stats_lines.append(f"FLUORESCENCE CHANNEL - {len(fluor_props)} bacteria detected")
        stats_lines.append("-" * 180)
        
        if fluor_contour_data:
            stats_lines.append(f"{'ID':<4} {'Area':<8} {'Mean Int':<10} {'Median Int':<11} {'Min Int':<9} {'Max Int':<9} {'Std Dev':<10} {'Centroid (Y, X)':<20}")
            stats_lines.append("-" * 180)
            
            for data in fluor_contour_data:
                stats_lines.append(
                    f"{data['id']:<4} "
                    f"{data['area']:<8.0f} "
                    f"{data['mean_intensity']:<10.2f} "
                    f"{data['median_intensity']:<11.2f} "
                    f"{data['min_intensity']:<9.0f} "
                    f"{data['max_intensity']:<9.0f} "
                    f"{data['std_intensity']:<10.2f} "
                    f"({data['centroid'][0]:.1f}, {data['centroid'][1]:.1f})"
                )
            
            # Summary statistics
            fluor_areas = [d['area'] for d in fluor_contour_data]
            fluor_mean_ints = [d['mean_intensity'] for d in fluor_contour_data]
            stats_lines.append("-" * 180)
            stats_lines.append(f"Summary: Area (mean={np.mean(fluor_areas):.1f}, median={np.median(fluor_areas):.1f}, std={np.std(fluor_areas):.1f}) | "
                             f"Intensity (mean={np.mean(fluor_mean_ints):.2f}, median={np.median(fluor_mean_ints):.2f}, std={np.std(fluor_mean_ints):.2f})")
        else:
            stats_lines.append("No bacteria detected in fluorescence channel")
        
        # Combine all lines
        stats_text = "\n".join(stats_lines)
        
        # Display with smaller font to fit more data
        self.ax_stats.text(0.02, 0.98, stats_text, 
                          ha='left', va='top', 
                          fontsize=7, fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                          transform=self.ax_stats.transAxes)


def main():
    """Main function to run the review tool"""
    print("="*70)
    print("BACTERIA DETECTION REVIEW TOOL")
    print("="*70)
    print("\nInstructions:")
    print("  - Use sliders to adjust thresholds and parameters")
    print("  - Use 'Previous' and 'Next' buttons to navigate images")
    print("  - Toggle checkboxes to change visualization options")
    print("  - Click 'Export Config' to save settings for main script")
    print("  - Close window to exit")
    print("\n" + "="*70 + "\n")
    
    # You can change this to your source folder
    source_folder = "source//1"
    
    try:
        tool = BacteriaReviewTool(source_folder=source_folder)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()