"""
Interactive Dual-Channel Bacteria Detection Review Tool
Features:
- Side-by-side visualization of grayscale and fluorescence channels
- Real-time threshold adjustment with sliders
- Interactive parameter tuning (min area, overlap threshold, etc.)
- Live contour detection preview
- Individual channel analysis
- Configuration export for main script
- Detailed contour size and intensity listing with scrollable view
- Co-localization matching between channels
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

class ScrollableText:
    """Helper class for scrollable text display"""
    def __init__(self, ax, text_lines, fontsize=8):
        self.ax = ax
        self.text_lines = text_lines
        self.fontsize = fontsize
        self.scroll_position = 0
        self.lines_per_page = 40  # Approximate visible lines
        
        self.text_obj = None
        self.render()
        
    def render(self):
        """Render the visible portion of text"""
        self.ax.clear()
        self.ax.axis('off')
        
        # Calculate visible lines
        start_idx = self.scroll_position
        end_idx = min(start_idx + self.lines_per_page, len(self.text_lines))
        visible_text = "\n".join(self.text_lines[start_idx:end_idx])
        
        # Show scroll indicator
        scroll_info = ""
        if len(self.text_lines) > self.lines_per_page:
            scroll_info = f"\n[Showing {start_idx+1}-{end_idx} of {len(self.text_lines)} lines | Scroll: ↑↓ arrows or mouse wheel]"
        
        # Display text
        self.text_obj = self.ax.text(0.05, 0.95, visible_text + scroll_info, 
                                     ha='left', va='top', 
                                     fontsize=self.fontsize, fontfamily='monospace',
                                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2),
                                     transform=self.ax.transAxes)
    
    def scroll_up(self):
        """Scroll up"""
        if self.scroll_position > 0:
            self.scroll_position = max(0, self.scroll_position - 5)
            self.render()
            return True
        return False
    
    def scroll_down(self):
        """Scroll down"""
        max_scroll = max(0, len(self.text_lines) - self.lines_per_page)
        if self.scroll_position < max_scroll:
            self.scroll_position = min(max_scroll, self.scroll_position + 5)
            self.render()
            return True
        return False

class BacteriaReviewTool:
    def __init__(self, source_folder="source//5"):
        """Initialize the review tool"""
        self.source_folder = source_folder
        self.current_index = 0
        
        # Default configuration (matching your main script)
        self.config = {
            "gray_threshold": 55,
            "fluor_threshold": 5,
            "min_area": 200,
            "min_fluor_area": 200,
            "overlap_threshold": 0.80,
            "show_labels": True,
            "show_contours": True,
            "show_filled": False,
        }
        
        # Load image pairs
        self.load_image_pairs()
        
        if not self.image_pairs:
            raise ValueError(f"No image pairs found in {source_folder}")
        
        # Scrollable text holder
        self.scrollable_stats = None
        
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
            min_intensity = np.min(intensities)
            max_intensity = np.max(intensities)
            
            contour_data.append({
                'id': i + 1,
                'area': prop.area,
                'mean_intensity': mean_intensity,
                'min_intensity': min_intensity,
                'max_intensity': max_intensity,
                'centroid': prop.centroid,
                'label': prop.label,
                'bbox': prop.bbox
            })
        
        return contour_data
    
    def calculate_colocalization(self, gray_labeled, fluor_labeled, gray_props, fluor_props):
        """Calculate which grayscale and fluorescence bacteria are co-localized"""
        colocalized = []
        overlap_threshold = self.config['overlap_threshold']
        
        for gray_prop in gray_props:
            gray_mask = (gray_labeled == gray_prop['label'])
            
            for fluor_prop in fluor_props:
                fluor_mask = (fluor_labeled == fluor_prop['label'])
                
                # Calculate overlap
                intersection = np.logical_and(gray_mask, fluor_mask)
                intersection_area = np.sum(intersection)
                
                # Calculate overlap ratio relative to grayscale bacteria
                overlap_ratio = intersection_area / gray_prop['area']
                
                if overlap_ratio >= overlap_threshold:
                    colocalized.append({
                        'gray_id': gray_prop['id'],
                        'fluor_id': fluor_prop['id'],
                        'overlap_ratio': overlap_ratio,
                        'gray_area': gray_prop['area'],
                        'fluor_area': fluor_prop['area'],
                        'intersection_area': intersection_area
                    })
        
        return colocalized
    
    def create_interface(self):
        """Create the interactive matplotlib interface"""
        self.fig = plt.figure(figsize=(20, 10))

        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title('Bacteria Detection Review Tool')
        
        # Create grid layout - removed histogram rows
        gs = self.fig.add_gridspec(3, 4, hspace=0.3, wspace=0.35, 
                                   left=0.05, right=0.95, top=0.96, bottom=0.04)
        
        # Image display axes (top row - larger now)
        self.ax_gray_orig = self.fig.add_subplot(gs[0, 0])
        self.ax_gray_proc = self.fig.add_subplot(gs[0, 1])
        self.ax_fluor_orig = self.fig.add_subplot(gs[0, 2])
        self.ax_fluor_proc = self.fig.add_subplot(gs[0, 3])
        
        # Bottom section: Controls (left) and Stats (right) side-by-side
        self.ax_controls = self.fig.add_subplot(gs[1:3, 0:2])
        self.ax_controls.axis('off')
        
        self.ax_stats = self.fig.add_subplot(gs[1:3, 2:4])
        self.ax_stats.axis('off')
        
        # Create sliders and controls
        self.create_controls()
        
        # Connect scroll events
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Initial update
        self.update_display()
        
        plt.show()
    
    def create_controls(self):
        """Create interactive control widgets in the left column"""
        # Get the position of ax_controls in figure coordinates
        bbox = self.ax_controls.get_position()

        # Horizontal sizing (keep or tweak)
        slider_left = bbox.x0 + 0.02
        slider_width = (bbox.x1 - bbox.x0) * 0.85
        slider_height = 0.02

        # Vertical layout: compute spacing and center the block inside ax_controls
        slider_count = 5  # number of sliders you have (Gray, Fluor, Min Gray Area, Min Fluor Area, Overlap)
        slider_spacing = 0.035  # gap between sliders
        controls_height = bbox.y1 - bbox.y0
        total_sliders_height = slider_count * slider_spacing

        # center the block vertically in ax_controls; add a small upward nudge if you want it closer to center-top
        base_y = bbox.y0 + (controls_height - total_sliders_height) / 2 + 0.02

        # When creating each slider, use index-based positions (top to bottom)
        # example for slider i (0..slider_count-1) with i=0 being top:
        def slider_y(i):
            # top-most slider index 0 uses highest y
            return base_y + (slider_count - 1 - i) * slider_spacing

        # Grayscale threshold slider
        ax_gray_thresh = plt.axes((slider_left, slider_y(0), slider_width, slider_height))
        self.slider_gray_thresh = Slider(
            ax_gray_thresh, 'Gray Threshold', 0, 255, 
            valinit=self.config['gray_threshold'], valstep=1, color='gray'
        )
        self.slider_gray_thresh.on_changed(self.update_gray_threshold)
        
        # Fluorescence threshold slider
        ax_fluor_thresh = plt.axes((slider_left, slider_y(1), slider_width, slider_height))
        self.slider_fluor_thresh = Slider(
            ax_fluor_thresh, 'Fluor Threshold', 0, 255, 
            valinit=self.config['fluor_threshold'], valstep=1, color='red'
        )
        self.slider_fluor_thresh.on_changed(self.update_fluor_threshold)
        
        # Min area slider
        ax_min_area = plt.axes((slider_left, slider_y(2), slider_width, slider_height))
        self.slider_min_area = Slider(
            ax_min_area, 'Min Gray Area', 50, 1000, 
            valinit=self.config['min_area'], valstep=10, color='green'
        )
        self.slider_min_area.on_changed(self.update_min_area)
        
        # Min fluor area slider
        ax_min_fluor = plt.axes((slider_left, slider_y(3), slider_width, slider_height))
        self.slider_min_fluor = Slider(
            ax_min_fluor, 'Min Fluor Area', 50, 1000, 
            valinit=self.config['min_fluor_area'], valstep=10, color='orange'
        )
        self.slider_min_fluor.on_changed(self.update_min_fluor_area)
        
        # Overlap threshold slider
        ax_overlap = plt.axes((slider_left, slider_y(4), slider_width, slider_height))
        self.slider_overlap = Slider(
            ax_overlap, 'Overlap %', 0, 100, 
            valinit=self.config['overlap_threshold']*100, valstep=5, color='blue'
        )
        self.slider_overlap.on_changed(self.update_overlap)
        
        # Checkboxes for display options - MOVED UP to avoid overlap
        checkbox_y = base_y + 3.5*slider_spacing
        ax_checks = plt.axes((slider_left, checkbox_y, slider_width, 0.06))
        ax_checks.set_frame_on(False)
        self.check_display = CheckButtons(
            ax_checks, 
            ['Show Labels', 'Show Contours', 'Show Filled'],
            [self.config['show_labels'], self.config['show_contours'], self.config['show_filled']]
        )
        self.check_display.on_clicked(self.update_display_options)
        
        # Navigation buttons
        button_width = slider_width * 0.28
        button_height = 0.03
        button_y = base_y + 1.5*slider_spacing
        button_spacing = 0.01
        
        ax_prev = plt.axes((slider_left, button_y, button_width, button_height))
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_prev.on_clicked(self.prev_image)
        
        ax_next = plt.axes((slider_left + button_width + button_spacing, button_y, button_width, button_height))
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.next_image)
        
        ax_export = plt.axes((slider_left + 2*(button_width + button_spacing), button_y, button_width, button_height))
        self.btn_export = Button(ax_export, 'Export Config')
        self.btn_export.on_clicked(self.export_config)
    
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
    
    def on_scroll(self, event):
        """Handle mouse scroll events"""
        if self.scrollable_stats is None:
            return
        
        if event.button == 'up':
            if self.scrollable_stats.scroll_up():
                self.fig.canvas.draw_idle()
        elif event.button == 'down':
            if self.scrollable_stats.scroll_down():
                self.fig.canvas.draw_idle()
    
    def on_key(self, event):
        """Handle keyboard events"""
        if self.scrollable_stats is None:
            return
        
        if event.key == 'up':
            if self.scrollable_stats.scroll_up():
                self.fig.canvas.draw_idle()
        elif event.key == 'down':
            if self.scrollable_stats.scroll_down():
                self.fig.canvas.draw_idle()
    
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
        
        # Calculate co-localization
        colocalized = self.calculate_colocalization(gray_labeled, fluor_labeled, gray_contour_data, fluor_contour_data)
        
        # Clear all axes
        self.ax_gray_orig.clear()
        self.ax_gray_proc.clear()
        self.ax_fluor_orig.clear()
        self.ax_fluor_proc.clear()
        
        # Display original images
        self.ax_gray_orig.imshow(gray_img, cmap='gray')
        self.ax_gray_orig.set_title(f'Grayscale Original\n{pair["gray_file"]}', fontsize=10)
        self.ax_gray_orig.axis('off')
        
        self.ax_fluor_orig.imshow(fluor_img, cmap='hot')
        self.ax_fluor_orig.set_title(f'Fluorescence Original\n{pair["fluor_file"]}', fontsize=10)
        self.ax_fluor_orig.axis('off')
        
        # Create visualizations with detected bacteria
        gray_vis = self.create_visualization(gray_img, gray_labeled, gray_large, is_grayscale=True, colocalized=colocalized)
        fluor_vis = self.create_visualization(fluor_img, fluor_labeled, fluor_large, is_grayscale=False, colocalized=colocalized)
        
        self.ax_gray_proc.imshow(gray_vis)
        self.ax_gray_proc.set_title(f'Grayscale: {len(gray_large)} total, {len(colocalized)} co-localized\nThreshold: {self.config["gray_threshold"]}, Min Area: {self.config["min_area"]}', fontsize=10)
        self.ax_gray_proc.axis('off')
        
        self.ax_fluor_proc.imshow(fluor_vis)
        self.ax_fluor_proc.set_title(f'Fluorescence: {len(fluor_large)} total\nThreshold: {self.config["fluor_threshold"]}, Min Area: {self.config["min_fluor_area"]}', fontsize=10)
        self.ax_fluor_proc.axis('off')
        
        # Update statistics with scrollable contour details
        self.update_statistics(gray_large, fluor_large, gray_contour_data, fluor_contour_data, colocalized)
        
        # Update figure
        self.fig.canvas.draw_idle()
    
    def create_visualization(self, image, labeled, props, is_grayscale=True, colocalized=None):
        """Create visualization with detected bacteria, highlighting co-localized ones"""
        # Create RGB image
        if image.max() > 0:
            vis = np.dstack([image, image, image])
            vis = (vis / vis.max() * 255).astype(np.uint8)
        else:
            vis = np.zeros((*image.shape, 3), dtype=np.uint8)
        
        # Get co-localized IDs
        colocalized_gray_ids = set()
        colocalized_fluor_ids = set()
        if colocalized:
            colocalized_gray_ids = {c['gray_id'] for c in colocalized}
            colocalized_fluor_ids = {c['fluor_id'] for c in colocalized}
        
        # Draw bacteria
        for i, prop in enumerate(props):
            # Get contour
            region_mask = (labeled == prop.label).astype(np.uint8)
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            
            # Determine if this bacterium is co-localized
            is_colocalized = False
            if is_grayscale:
                is_colocalized = (i + 1) in colocalized_gray_ids
            else:
                is_colocalized = (i + 1) in colocalized_fluor_ids
            
            # Choose color based on co-localization
            if is_colocalized:
                # Co-localized: yellow/bright color
                color = (255, 255, 0)  # Yellow
            else:
                # Not co-localized: original colors
                if is_grayscale:
                    color = (0, 255, 0)  # Green
                else:
                    color = (255, 0, 0)  # Red
            
            # Draw filled region
            if self.config['show_filled']:
                cv2.drawContours(vis, [contour], -1, color, -1)
                vis = cv2.addWeighted(vis, 0.7, np.dstack([image, image, image]).astype(np.uint8), 0.3, 0)
            
            # Draw contour (thicker for co-localized)
            if self.config['show_contours']:
                thickness = 3 if is_colocalized else 2
                cv2.drawContours(vis, [contour], -1, color, thickness)
            
            # Draw label
            if self.config['show_labels']:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    label_text = f"{i+1}"
                    if is_colocalized:
                        label_text += "*"  # Add asterisk for co-localized
                    
                    cv2.putText(vis, label_text, (cX-10, cY-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis
    
    def update_statistics(self, gray_props, fluor_props, gray_contour_data, fluor_contour_data, colocalized):
        """Update statistics display with detailed contour information and scrolling"""
        # Build comprehensive statistics text as list of lines
        stats_lines = []
        
        # Header with image info
        stats_lines.append(f"═══════════════ DETECTED CONTOUR DETAILS ═══════════════")
        stats_lines.append(f"Image {self.current_index + 1}/{len(self.image_pairs)} | {self.image_pairs[self.current_index]['gray_file']}")
        stats_lines.append(f"Overlap Threshold: {self.config['overlap_threshold']*100:.0f}%")
        stats_lines.append("")
        
        # Co-localization Summary
        stats_lines.append(f"┌─ CO-LOCALIZATION SUMMARY ─┐")
        stats_lines.append(f"  Total grayscale bacteria: {len(gray_props)}")
        stats_lines.append(f"  Total fluorescence bacteria: {len(fluor_props)}")
        stats_lines.append(f"  Co-localized pairs: {len(colocalized)}")
        
        if len(gray_props) > 0:
            coloc_pct = (len(colocalized) / len(gray_props)) * 100
            stats_lines.append(f"  Co-localization rate: {coloc_pct:.1f}%")
        
        stats_lines.append("")
        
        # Co-localization Details
        if colocalized:
            stats_lines.append(f"┌─ CO-LOCALIZED PAIRS ({len(colocalized)}) ─┐")
            stats_lines.append(f"{'Gray':<5} {'Fluor':<6} {'Overlap':<8} {'G.Area':<8} {'F.Area':<8} {'Intersect':<10}")
            stats_lines.append("─" * 60)
            
            for coloc in colocalized:
                stats_lines.append(
                    f"{coloc['gray_id']:<5} "
                    f"{coloc['fluor_id']:<6} "
                    f"{coloc['overlap_ratio']*100:>6.1f}% "
                    f"{coloc['gray_area']:<8.0f} "
                    f"{coloc['fluor_area']:<8.0f} "
                    f"{coloc['intersection_area']:<10.0f}"
                )
        else:
            stats_lines.append(f"┌─ CO-LOCALIZED PAIRS ─┐")
            stats_lines.append("  No co-localized bacteria detected")
        
        stats_lines.append("")
        stats_lines.append("")
        
        # Grayscale Channel Details
        stats_lines.append(f"┌─ GRAYSCALE CHANNEL ({len(gray_props)} bacteria) ─┐")
        stats_lines.append("")
        
        # Get co-localized IDs for marking
        colocalized_gray_ids = {c['gray_id'] for c in colocalized}
        
        if gray_contour_data:
            stats_lines.append(f"{'ID':<4} {'*':<2} {'Area':<8} {'Mean':<8} {'Min':<7} {'Max':<7} {'Centroid (Y,X)':<18}")
            stats_lines.append("─" * 65)
            
            for data in gray_contour_data:
                coloc_mark = "*" if data['id'] in colocalized_gray_ids else " "
                stats_lines.append(
                    f"{data['id']:<4} "
                    f"{coloc_mark:<2} "
                    f"{data['area']:<8.0f} "
                    f"{data['mean_intensity']:<8.1f} "
                    f"{data['min_intensity']:<7.0f} "
                    f"{data['max_intensity']:<7.0f} "
                    f"({data['centroid'][0]:.1f}, {data['centroid'][1]:.1f})"
                )
            
            # Summary statistics
            gray_areas = [d['area'] for d in gray_contour_data]
            gray_mean_ints = [d['mean_intensity'] for d in gray_contour_data]
            stats_lines.append("─" * 65)
            stats_lines.append(f"Area: μ={np.mean(gray_areas):.1f}, σ={np.std(gray_areas):.1f} | "
                             f"Intensity: μ={np.mean(gray_mean_ints):.1f}, σ={np.std(gray_mean_ints):.1f}")
        else:
            stats_lines.append("  No bacteria detected")
        
        stats_lines.append("")
        stats_lines.append(f"* = Co-localized with fluorescence")
        stats_lines.append("")
        
        # Fluorescence Channel Details
        stats_lines.append(f"┌─ FLUORESCENCE CHANNEL ({len(fluor_props)} bacteria) ─┐")
        stats_lines.append("")
        
        # Get co-localized IDs for marking
        colocalized_fluor_ids = {c['fluor_id'] for c in colocalized}
        
        if fluor_contour_data:
            stats_lines.append(f"{'ID':<4} {'*':<2} {'Area':<8} {'Mean':<8} {'Min':<7} {'Max':<7} {'Centroid (Y,X)':<18}")
            stats_lines.append("─" * 65)
            
            for data in fluor_contour_data:
                coloc_mark = "*" if data['id'] in colocalized_fluor_ids else " "
                stats_lines.append(
                    f"{data['id']:<4} "
                    f"{coloc_mark:<2} "
                    f"{data['area']:<8.0f} "
                    f"{data['mean_intensity']:<8.1f} "
                    f"{data['min_intensity']:<7.0f} "
                    f"{data['max_intensity']:<7.0f} "
                    f"({data['centroid'][0]:.1f}, {data['centroid'][1]:.1f})"
                )
            
            # Summary statistics
            fluor_areas = [d['area'] for d in fluor_contour_data]
            fluor_mean_ints = [d['mean_intensity'] for d in fluor_contour_data]
            stats_lines.append("─" * 65)
            stats_lines.append(f"Area: μ={np.mean(fluor_areas):.1f}, σ={np.std(fluor_areas):.1f} | "
                             f"Intensity: μ={np.mean(fluor_mean_ints):.1f}, σ={np.std(fluor_mean_ints):.1f}")
        else:
            stats_lines.append("  No bacteria detected")
        
        stats_lines.append("")
        stats_lines.append(f"* = Co-localized with grayscale")
        
        # Create scrollable text display
        self.scrollable_stats = ScrollableText(self.ax_stats, stats_lines, fontsize=8)


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
    print("  - Scroll with mouse wheel or ↑↓ arrow keys in stats panel")
    print("  - Co-localized bacteria are marked with * and shown in YELLOW")
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