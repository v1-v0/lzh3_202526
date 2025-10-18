"""
Interactive Dual-Channel Bacteria Detection Review Tool with Excel Export
Features:
- Generates review sheet for each image pair
- Shows ORIGINAL images with detection overlays (using raw source images)
- Uses MODE intensity instead of mean
- Detailed contour statistics table (without min intensity)
- Exports all data to Excel file
- Improved layout to avoid overlapping
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from skimage import io, filters, morphology, measure
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from scipy import stats
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class BacteriaReviewTool:
    def __init__(self, source_folder="source//5"):
        """Initialize the review tool"""
        self.source_folder = source_folder
        
        # Fixed configuration
        self.config = {
            "gray_threshold": 53,
            "fluor_threshold": 5,
            "min_area": 400,
            "min_fluor_area": 400,
            "overlap_threshold": 0.80,
        }
        
        self.load_image_pairs()
        
        if not self.image_pairs:
            raise ValueError(f"No image pairs found in {source_folder}")
        
        # Storage for all results
        self.all_results = []
        
        # Generate review sheets for all images
        self.generate_all_review_sheets()
        
        # Export to Excel
        self.export_to_excel()
        
    def load_image_pairs(self):
        """Load all grayscale and fluorescence image pairs"""
        self.image_pairs = []
        folder_path = Path(self.source_folder)
        
        if not folder_path.exists():
            print(f"Error: Folder '{self.source_folder}' does not exist")
            return
        
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
        
        final = self.postprocess(binary)
        labeled = measure.label(final)
        props = measure.regionprops(labeled)
        large_props = [prop for prop in props if prop.area > min_area]
        
        return preprocessed, labeled, props, large_props
    
    def calculate_mode(self, intensities):
        """Calculate mode of intensity values"""
        if len(intensities) == 0:
            return 0
        
        mode_result = stats.mode(intensities, keepdims=True)
        return float(mode_result.mode[0])
    
    def get_contour_intensities(self, image, labeled, props):
        """Get intensity statistics for each contour (using MODE instead of mean)"""
        contour_data = []
        
        for i, prop in enumerate(props):
            region_mask = (labeled == prop.label)
            intensities = image[region_mask]
            
            mode_intensity = self.calculate_mode(intensities)
            max_intensity = np.max(intensities)
            std_intensity = np.std(intensities)
            
            contour_data.append({
                'id': i + 1,
                'area': prop.area,
                'perimeter': prop.perimeter,
                'eccentricity': prop.eccentricity,
                'mode_intensity': mode_intensity,
                'max_intensity': max_intensity,
                'std_intensity': std_intensity,
                'centroid_y': prop.centroid[0],
                'centroid_x': prop.centroid[1],
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
                
                intersection = np.logical_and(gray_mask, fluor_mask)
                intersection_area = np.sum(intersection)
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
    
    def create_overlay_visualization(self, image, labeled, props, is_grayscale=True, colocalized=None):
        """Create overlay visualization on ORIGINAL image without preprocessing"""
        # Convert original image to RGB for visualization
        vis = np.dstack([image, image, image]).astype(np.uint8)
        
        colocalized_gray_ids = set()
        colocalized_fluor_ids = set()
        if colocalized:
            colocalized_gray_ids = {c['gray_id'] for c in colocalized}
            colocalized_fluor_ids = {c['fluor_id'] for c in colocalized}
        
        for i, prop in enumerate(props):
            region_mask = (labeled == prop.label).astype(np.uint8)
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            
            is_colocalized = False
            if is_grayscale:
                is_colocalized = (i + 1) in colocalized_gray_ids
            else:
                is_colocalized = (i + 1) in colocalized_fluor_ids
            
            if is_colocalized:
                color = (255, 255, 0)  # Yellow
                thickness = 3
            else:
                color = (0, 255, 0) if is_grayscale else (255, 0, 0)
                thickness = 2
            
            cv2.drawContours(vis, [contour], -1, color, thickness)
            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                label_text = f"{i+1}"
                if is_colocalized:
                    label_text += "*"
                
                cv2.putText(vis, label_text, (cX-10, cY-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis
    
    def create_final_visualization(self, gray_img, fluor_img, gray_labeled, fluor_labeled, 
                                   gray_props, fluor_large, colocalized):
        """Create final merged visualization showing co-localization using ORIGINAL images"""
        # Create RGB composite using ORIGINAL images
        gray_norm = gray_img.astype(np.uint8)
        fluor_norm = fluor_img.astype(np.uint8)
        
        # Create composite: gray in blue, fluor in red
        vis = np.zeros((*gray_img.shape, 3), dtype=np.uint8)
        vis[:, :, 0] = fluor_norm  # Red channel
        vis[:, :, 2] = gray_norm   # Blue channel
        
        # Draw only co-localized bacteria
        colocalized_gray_ids = {c['gray_id'] for c in colocalized}
        
        for i, prop in enumerate(gray_props):
            if (i + 1) not in colocalized_gray_ids:
                continue
                
            region_mask = (gray_labeled == prop.label).astype(np.uint8)
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            
            # Draw in bright yellow for co-localized
            cv2.drawContours(vis, [contour], -1, (255, 255, 0), 3)
            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                cv2.putText(vis, f"{i+1}", (cX-10, cY-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis
    
    def format_statistics_text(self, gray_props, fluor_props, gray_contour_data, 
                               fluor_contour_data, colocalized, filename):
        """Format statistics as text for display (using MODE, no MIN)"""
        lines = []
        
        lines.append(f"═══════════════ DETECTION REPORT ═══════════════")
        lines.append(f"Source: {filename}")
        lines.append(f"Parameters: Gray Thresh={self.config['gray_threshold']}, "
                    f"Fluor Thresh={self.config['fluor_threshold']}, "
                    f"Overlap={self.config['overlap_threshold']*100:.0f}%")
        lines.append("")
        
        # Summary
        lines.append(f"┌─── SUMMARY ───┐")
        lines.append(f"  Grayscale bacteria:     {len(gray_props):>3}")
        lines.append(f"  Fluorescence bacteria:  {len(fluor_props):>3}")
        lines.append(f"  Co-localized pairs:     {len(colocalized):>3}")
        
        if len(gray_props) > 0:
            coloc_pct = (len(colocalized) / len(gray_props)) * 100
            lines.append(f"  Co-localization rate:   {coloc_pct:>5.1f}%")
        
        lines.append("")
        
        # Co-localization pairs
        if colocalized:
            lines.append(f"┌─── CO-LOCALIZED PAIRS ({len(colocalized)}) ───┐")
            lines.append(f"{'Gray':<6} {'Fluor':<7} {'Overlap':<9} {'G.Area':<9} {'F.Area':<9} {'Intersect'}")
            lines.append("─" * 60)
            
            for coloc in colocalized:
                lines.append(
                    f"{coloc['gray_id']:<6} "
                    f"{coloc['fluor_id']:<7} "
                    f"{coloc['overlap_ratio']*100:>7.1f}% "
                    f"{coloc['gray_area']:<9.0f} "
                    f"{coloc['fluor_area']:<9.0f} "
                    f"{coloc['intersection_area']:.0f}"
                )
        else:
            lines.append(f"┌─── CO-LOCALIZED PAIRS ───┐")
            lines.append("  No co-localized bacteria detected")
        
        lines.append("")
        
        # Grayscale details
        colocalized_gray_ids = {c['gray_id'] for c in colocalized}
        
        lines.append(f"┌─── GRAYSCALE CHANNEL ({len(gray_props)} bacteria) ───┐")
        if gray_contour_data:
            lines.append(f"{'ID':<5} {'*':<2} {'Area':<9} {'Mode':<9} {'Max':<8} {'Centroid (Y, X)'}")
            lines.append("─" * 55)
            
            for data in gray_contour_data:
                coloc_mark = "*" if data['id'] in colocalized_gray_ids else " "
                lines.append(
                    f"{data['id']:<5} "
                    f"{coloc_mark:<2} "
                    f"{data['area']:<9.0f} "
                    f"{data['mode_intensity']:<9.1f} "
                    f"{data['max_intensity']:<8.0f} "
                    f"({data['centroid_y']:.1f}, {data['centroid_x']:.1f})"
                )
            
            # Summary statistics
            gray_areas = [d['area'] for d in gray_contour_data]
            gray_mode_ints = [d['mode_intensity'] for d in gray_contour_data]
            lines.append("─" * 55)
            lines.append(f"Stats: Area μ={np.mean(gray_areas):.1f} σ={np.std(gray_areas):.1f} | "
                        f"Intensity(mode) μ={np.mean(gray_mode_ints):.1f}")
        else:
            lines.append("  No bacteria detected")
        
        lines.append("")
        
        # Fluorescence details
        colocalized_fluor_ids = {c['fluor_id'] for c in colocalized}
        
        lines.append(f"┌─── FLUORESCENCE CHANNEL ({len(fluor_props)} bacteria) ───┐")
        if fluor_contour_data:
            lines.append(f"{'ID':<5} {'*':<2} {'Area':<9} {'Mode':<9} {'Max':<8} {'Centroid (Y, X)'}")
            lines.append("─" * 55)
            
            for data in fluor_contour_data:
                coloc_mark = "*" if data['id'] in colocalized_fluor_ids else " "
                lines.append(
                    f"{data['id']:<5} "
                    f"{coloc_mark:<2} "
                    f"{data['area']:<9.0f} "
                    f"{data['mode_intensity']:<9.1f} "
                    f"{data['max_intensity']:<8.0f} "
                    f"({data['centroid_y']:.1f}, {data['centroid_x']:.1f})"
                )
            
            # Summary statistics
            fluor_areas = [d['area'] for d in fluor_contour_data]
            fluor_mode_ints = [d['mode_intensity'] for d in fluor_contour_data]
            lines.append("─" * 55)
            lines.append(f"Stats: Area μ={np.mean(fluor_areas):.1f} σ={np.std(fluor_areas):.1f} | "
                        f"Intensity(mode) μ={np.mean(fluor_mode_ints):.1f}")
        else:
            lines.append("  No bacteria detected")
        
        lines.append("")
        lines.append("Legend: * = Co-localized | Yellow contours = Co-localized")
        
        return "\n".join(lines)
    
    def generate_review_sheet(self, pair_index):
        """Generate a review sheet for a single image pair using original source images"""
        pair = self.image_pairs[pair_index]
        gray_img = pair['gray']
        fluor_img = pair['fluor']
        filename = pair['gray_file']
        
        # Detect bacteria (preprocessing only for segmentation)
        gray_proc, gray_labeled, gray_props, gray_large = self.detect_bacteria(gray_img, is_grayscale=True)
        fluor_proc, fluor_labeled, fluor_props, fluor_large = self.detect_bacteria(fluor_img, is_grayscale=False)
        
        # Get contour intensity data (MODE instead of MEAN, no MIN)
        gray_contour_data = self.get_contour_intensities(gray_img, gray_labeled, gray_large)
        fluor_contour_data = self.get_contour_intensities(fluor_img, fluor_labeled, fluor_large)
        
        # Calculate co-localization
        colocalized = self.calculate_colocalization(gray_labeled, fluor_labeled, 
                                                    gray_contour_data, fluor_contour_data)
        
        # Store results for Excel export
        result = {
            'filename': filename,
            'gray_contour_data': gray_contour_data,
            'fluor_contour_data': fluor_contour_data,
            'colocalized': colocalized,
            'gray_count': len(gray_large),
            'fluor_count': len(fluor_large),
            'coloc_count': len(colocalized)
        }
        self.all_results.append(result)
        
        # Create figure with better spacing
        fig = plt.figure(figsize=(16, 10))
        
        # Set window title
        manager = getattr(fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(f'Review: {filename}')
        
        # Create grid layout with adjusted spacing
        gs = fig.add_gridspec(2, 3, 
                             height_ratios=[1, 1],
                             width_ratios=[1, 1, 1.2],
                             hspace=0.3, wspace=0.25,
                             left=0.05, right=0.98, top=0.93, bottom=0.05)
        
        # Add main title
        fig.suptitle(f'Bacteria Detection Review Sheet\nSource: {self.source_folder} | File: {filename}',
                    fontsize=14, fontweight='bold', y=0.97)
        
        # Top row: Original images with overlays
        # Top left: Grayscale with overlay
        ax_gray = fig.add_subplot(gs[0, 0])
        gray_vis = self.create_overlay_visualization(gray_img, gray_labeled, gray_large, 
                                                     is_grayscale=True, colocalized=colocalized)
        ax_gray.imshow(gray_vis)
        ax_gray.set_title(f'GRAYSCALE CHANNEL (Original)\n{len(gray_large)} detected | {len(colocalized)} co-localized', 
                         fontsize=11, fontweight='bold')
        ax_gray.axis('off')
        
        # Top middle: Fluorescence with overlay
        ax_fluor = fig.add_subplot(gs[0, 1])
        fluor_vis = self.create_overlay_visualization(fluor_img, fluor_labeled, fluor_large, 
                                                      is_grayscale=False, colocalized=colocalized)
        ax_fluor.imshow(fluor_vis)
        ax_fluor.set_title(f'FLUORESCENCE CHANNEL (Original)\n{len(fluor_large)} detected', 
                          fontsize=11, fontweight='bold')
        ax_fluor.axis('off')
        
        # Top right: Statistics
        ax_stats = fig.add_subplot(gs[0, 2])
        ax_stats.axis('off')
        
        stats_text = self.format_statistics_text(gray_large, fluor_large, 
                                                gray_contour_data, fluor_contour_data, 
                                                colocalized, filename)
        
        ax_stats.text(0.02, 0.98, stats_text, 
                     ha='left', va='top', 
                     fontsize=7.5, fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3),
                     transform=ax_stats.transAxes)
        
        # Bottom row
        # Bottom left: Final output
        ax_final = fig.add_subplot(gs[1, 0])
        final_vis = self.create_final_visualization(gray_img, fluor_img, gray_labeled, fluor_labeled, 
                                                    gray_large, fluor_large, colocalized)
        ax_final.imshow(final_vis)
        coloc_pct = (len(colocalized) / len(gray_large) * 100) if len(gray_large) > 0 else 0
        ax_final.set_title(f'FINAL OUTPUT (Co-localized Only)\nRate: {coloc_pct:.1f}%', 
                          fontsize=11, fontweight='bold')
        ax_final.axis('off')
        
        # Bottom middle: Parameters info
        ax_params = fig.add_subplot(gs[1, 1])
        ax_params.axis('off')
        info_text = (
            "DETECTION PARAMETERS:\n"
            f"• Grayscale Threshold: {self.config['gray_threshold']}\n"
            f"• Fluorescence Threshold: {self.config['fluor_threshold']}\n"
            f"• Min Grayscale Area: {self.config['min_area']} px\n"
            f"• Min Fluorescence Area: {self.config['min_fluor_area']} px\n"
            f"• Overlap Threshold: {self.config['overlap_threshold']*100:.0f}%\n\n"
            "INTENSITY STATISTICS:\n"
            f"• Using MODE instead of mean\n"
            f"• Min intensity removed\n\n"
            "DISPLAY:\n"
            f"• Using ORIGINAL source images\n\n"
            "COLOR LEGEND:\n"
            "• Green: Grayscale bacteria\n"
            "• Red: Fluorescence bacteria\n"
            "• Yellow: Co-localized bacteria\n"
            "• * marker: Co-localized ID"
        )
        ax_params.text(0.1, 0.5, info_text,
                     ha='left', va='center',
                     fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2),
                     transform=ax_params.transAxes)
        
        # Bottom right: Empty (reserved for future use or cleaner layout)
        ax_empty = fig.add_subplot(gs[1, 2])
        ax_empty.axis('off')
        
        return fig
    
    def generate_all_review_sheets(self):
        """Generate review sheets for all image pairs"""
        print(f"\nGenerating review sheets for {len(self.image_pairs)} image pair(s)...\n")
        
        for i, pair in enumerate(self.image_pairs):
            print(f"Processing {i+1}/{len(self.image_pairs)}: {pair['gray_file']}")
            fig = self.generate_review_sheet(i)
            
            # Save the figure
            output_filename = f"review_{pair['gray_file'].replace('.tif', '.png')}"
            fig.savefig(output_filename, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved: {output_filename}")
            
            plt.close(fig)
        
        print(f"\n{'='*70}")
        print(f"✓ All review sheets generated successfully!")
        print(f"{'='*70}\n")
    
    def export_to_excel(self):
        """Export all results to Excel file (MODE intensity, no MIN)"""
        print("\nExporting results to Excel...")
        
        # Create Excel writer
        output_file = f"bacteria_detection_results_{self.source_folder.replace('/', '_').replace('//', '_')}.xlsx"
        writer = pd.ExcelWriter(output_file, engine='openpyxl')
        
        # Sheet 1: Summary
        summary_data = []
        for result in self.all_results:
            coloc_rate = (result['coloc_count'] / result['gray_count'] * 100) if result['gray_count'] > 0 else 0
            
            # Calculate statistics
            gray_areas = [d['area'] for d in result['gray_contour_data']]
            fluor_areas = [d['area'] for d in result['fluor_contour_data']]
            
            gray_intensities = [d['mode_intensity'] for d in result['gray_contour_data']]
            fluor_intensities = [d['mode_intensity'] for d in result['fluor_contour_data']]
            
            summary_data.append({
                'Filename': result['filename'],
                'Grayscale_Count': result['gray_count'],
                'Fluorescence_Count': result['fluor_count'],
                'Colocalized_Count': result['coloc_count'],
                'Colocalization_Rate_%': round(coloc_rate, 2),
                'Gray_Avg_Area': round(np.mean(gray_areas), 2) if gray_areas else 0,
                'Gray_Std_Area': round(np.std(gray_areas), 2) if gray_areas else 0,
                'Gray_Avg_Mode_Intensity': round(np.mean(gray_intensities), 2) if gray_intensities else 0,
                'Fluor_Avg_Area': round(np.mean(fluor_areas), 2) if fluor_areas else 0,
                'Fluor_Std_Area': round(np.std(fluor_areas), 2) if fluor_areas else 0,
                'Fluor_Avg_Mode_Intensity': round(np.mean(fluor_intensities), 2) if fluor_intensities else 0,
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Grayscale Details (MODE, no MIN)
        gray_data = []
        for result in self.all_results:
            colocalized_ids = {c['gray_id'] for c in result['colocalized']}
            
            for data in result['gray_contour_data']:
                gray_data.append({
                    'Filename': result['filename'],
                    'Bacteria_ID': data['id'],
                    'Colocalized': 'Yes' if data['id'] in colocalized_ids else 'No',
                    'Area_px': round(data['area'], 2),
                    'Perimeter_px': round(data['perimeter'], 2),
                    'Eccentricity': round(data['eccentricity'], 3),
                    'Mode_Intensity': round(data['mode_intensity'], 2),
                    'Max_Intensity': round(data['max_intensity'], 2),
                    'Std_Intensity': round(data['std_intensity'], 2),
                    'Centroid_Y': round(data['centroid_y'], 2),
                    'Centroid_X': round(data['centroid_x'], 2),
                })
        
        gray_df = pd.DataFrame(gray_data)
        gray_df.to_excel(writer, sheet_name='Grayscale_Bacteria', index=False)
        
        # Sheet 3: Fluorescence Details (MODE, no MIN)
        fluor_data = []
        for result in self.all_results:
            colocalized_ids = {c['fluor_id'] for c in result['colocalized']}
            
            for data in result['fluor_contour_data']:
                fluor_data.append({
                    'Filename': result['filename'],
                    'Bacteria_ID': data['id'],
                    'Colocalized': 'Yes' if data['id'] in colocalized_ids else 'No',
                    'Area_px': round(data['area'], 2),
                    'Perimeter_px': round(data['perimeter'], 2),
                    'Eccentricity': round(data['eccentricity'], 3),
                    'Mode_Intensity': round(data['mode_intensity'], 2),
                    'Max_Intensity': round(data['max_intensity'], 2),
                    'Std_Intensity': round(data['std_intensity'], 2),
                    'Centroid_Y': round(data['centroid_y'], 2),
                    'Centroid_X': round(data['centroid_x'], 2),
                })
        
        fluor_df = pd.DataFrame(fluor_data)
        fluor_df.to_excel(writer, sheet_name='Fluorescence_Bacteria', index=False)
        
        # Sheet 4: Colocalization Pairs
        coloc_data = []
        for result in self.all_results:
            for coloc in result['colocalized']:
                coloc_data.append({
                    'Filename': result['filename'],
                    'Gray_Bacteria_ID': coloc['gray_id'],
                    'Fluor_Bacteria_ID': coloc['fluor_id'],
                    'Overlap_Ratio_%': round(coloc['overlap_ratio'] * 100, 2),
                    'Gray_Area_px': round(coloc['gray_area'], 2),
                    'Fluor_Area_px': round(coloc['fluor_area'], 2),
                    'Intersection_Area_px': round(coloc['intersection_area'], 2),
                })
        
        coloc_df = pd.DataFrame(coloc_data)
        coloc_df.to_excel(writer, sheet_name='Colocalization_Pairs', index=False)
        
        # Sheet 5: Parameters
        params_data = [{
            'Parameter': key,
            'Value': value
        } for key, value in self.config.items()]
        
        params_df = pd.DataFrame(params_data)
        params_df.to_excel(writer, sheet_name='Detection_Parameters', index=False)
        
        # Save and close
        writer.close()
        
        print(f"✓ Excel file saved: {output_file}")
        print(f"  - Summary sheet: {len(summary_data)} images")
        print(f"  - Grayscale bacteria: {len(gray_data)} detections")
        print(f"  - Fluorescence bacteria: {len(fluor_data)} detections")
        print(f"  - Colocalization pairs: {len(coloc_data)} pairs")
        print(f"  - Using MODE intensity (no MIN)")
        print(f"{'='*70}\n")


def main():
    """Main function to run the review tool"""
    print("="*70)
    print("BACTERIA DETECTION REVIEW SHEET GENERATOR WITH EXCEL EXPORT")
    print("="*70)
    print("\nUpdates:")
    print("  • Using MODE intensity instead of mean")
    print("  • Removed MIN intensity")
    print("  • Using ORIGINAL source images for display")
    print("  • Improved layout to avoid overlapping")
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