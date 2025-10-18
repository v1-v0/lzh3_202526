"""
Interactive Dual-Channel Bacteria Detection Review Tool with Excel Export

This tool analyzes paired grayscale and fluorescence microscopy images to detect
bacteria and identify co-localization patterns. It generates detailed review sheets
and exports comprehensive statistics to Excel.

Features:
- Detects bacteria in grayscale (dark objects) and fluorescence (bright objects) channels
- Calculates co-localization between channels
- Generates visual review sheets with detection overlays
- Exports detailed statistics to Excel workbook
- Uses MODE intensity for robust statistics
"""

import os
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import ndimage as ndi
from scipy import stats
from skimage import io, filters, morphology, measure
from skimage.segmentation import watershed

warnings.filterwarnings('ignore')


class BacteriaReviewTool:
    """
    A tool for analyzing dual-channel bacteria microscopy images.
    
    This class handles loading image pairs, detecting bacteria in both channels,
    calculating co-localization, and generating review sheets with Excel export.
    """
    
    # Default detection parameters
    DEFAULT_CONFIG = {
        "gray_threshold": 53,
        "fluor_threshold": 5,
        "min_area": 400,
        "min_fluor_area": 400,
        "overlap_threshold": 0.80,
    }
    
    def __init__(self, source_folder: str = "source/5"):
        """
        Initialize the review tool.
        
        Args:
            source_folder: Path to folder containing image pairs
        """
        self.source_folder = source_folder
        self.config = self.DEFAULT_CONFIG.copy()
        self.image_pairs: List[Dict] = []
        self.all_results: List[Dict] = []
        
        # Load and process images
        self.load_image_pairs()
        
        if not self.image_pairs:
            raise ValueError(f"No image pairs found in {source_folder}")
        
        # Generate review sheets for all images
        self.generate_all_review_sheets()
        
        # Export results to Excel
        self.export_to_excel()
    
    def load_image_pairs(self) -> None:
        """Load all grayscale and fluorescence image pairs from source folder."""
        folder_path = Path(self.source_folder)
        
        if not folder_path.exists():
            print(f"Error: Folder '{self.source_folder}' does not exist")
            return
        
        # Find all grayscale channel files
        gray_files = sorted([
            f for f in os.listdir(self.source_folder)
            if f.endswith('_ch00.tif') and os.path.isfile(os.path.join(self.source_folder, f))
        ])
        
        # Load paired images
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
    
    def preprocess_image(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Preprocess image with median filtering and CLAHE enhancement.
        
        Args:
            image: Input grayscale image
            clip_limit: CLAHE clip limit for contrast enhancement
            
        Returns:
            Enhanced image
        """
        # Apply median filter to reduce noise
        median_filtered = filters.median(image, morphology.disk(3))
        
        # Normalize to 8-bit range
        if median_filtered.max() > 0:
            img_normalized = (median_filtered / median_filtered.max() * 255).astype(np.uint8)
        else:
            img_normalized = median_filtered.astype(np.uint8)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_normalized)
        
        return enhanced
    
    def preprocess_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Preprocess grayscale channel image."""
        return self.preprocess_image(image, clip_limit=2.0)
    
    def preprocess_fluorescence(self, image: np.ndarray) -> np.ndarray:
        """Preprocess fluorescence channel image."""
        return self.preprocess_image(image, clip_limit=3.0)
    
    def segment_image(self, preprocessed: np.ndarray, threshold: int, 
                     is_grayscale: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment bacteria using thresholding and watershed algorithm.
        
        Args:
            preprocessed: Preprocessed image
            threshold: Intensity threshold for segmentation
            is_grayscale: True for dark objects (grayscale), False for bright objects (fluorescence)
            
        Returns:
            Tuple of (binary mask, labeled regions)
        """
        # Create binary mask
        if is_grayscale:
            binary = np.asarray(preprocessed) <= threshold  # Dark objects
        else:
            binary = np.asarray(preprocessed) >= threshold  # Bright objects
        
        # Distance transform for watershed
        distance_result = ndi.distance_transform_edt(binary)
        if isinstance(distance_result, tuple):
            distance = distance_result[0]
        else:
            distance = distance_result
        distance = np.asarray(distance, dtype=np.float32)
        
        # Find local maxima as markers
        local_max = morphology.local_maxima(distance)
        markers = measure.label(local_max)
        
        # Apply watershed to separate touching objects
        labels = watershed(-distance, markers, mask=binary)
        separated = labels > 0
        
        return separated, labels
    
    def segment_grayscale(self, preprocessed: np.ndarray, threshold: int) -> Tuple[np.ndarray, np.ndarray]:
        """Segment grayscale bacteria (dark objects)."""
        return self.segment_image(preprocessed, threshold, is_grayscale=True)
    
    def segment_fluorescence(self, preprocessed: np.ndarray, threshold: int) -> Tuple[np.ndarray, np.ndarray]:
        """Segment fluorescence bacteria (bright objects)."""
        return self.segment_image(preprocessed, threshold, is_grayscale=False)
    
    def postprocess(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Post-process binary mask to clean up segmentation.
        
        Args:
            binary_mask: Binary segmentation mask
            
        Returns:
            Cleaned binary mask
        """
        min_size = 50
        cleaned = morphology.remove_small_objects(binary_mask, min_size=min_size)
        filled = ndi.binary_fill_holes(cleaned)
        opened = morphology.binary_opening(filled, morphology.disk(2))
        final = morphology.binary_closing(opened, morphology.disk(2))
        return final
    
    def detect_bacteria(self, image: np.ndarray, is_grayscale: bool = True) -> Tuple:
        """
        Detect bacteria in image with current settings.
        
        Args:
            image: Input image
            is_grayscale: True for grayscale channel, False for fluorescence
            
        Returns:
            Tuple of (preprocessed image, labeled regions, all properties, large properties)
        """
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
        
        # Post-process and extract properties
        final = self.postprocess(binary)
        labeled = measure.label(final)
        props = measure.regionprops(labeled)
        
        # Filter by minimum area
        large_props = [prop for prop in props if prop.area > min_area]
        
        return preprocessed, labeled, props, large_props
    
    @staticmethod
    def calculate_mode(intensities: np.ndarray) -> float:
        """
        Calculate mode of intensity values.
        
        Args:
            intensities: Array of intensity values
            
        Returns:
            Mode intensity value
        """
        if len(intensities) == 0:
            return 0.0
        
        mode_result = stats.mode(intensities, keepdims=True)
        return float(mode_result.mode[0])
    
    def get_contour_intensities(self, image: np.ndarray, labeled: np.ndarray, 
                               props: List) -> List[Dict]:
        """
        Get intensity statistics for each detected contour.
        
        Args:
            image: Original image
            labeled: Labeled regions
            props: Region properties
            
        Returns:
            List of dictionaries containing contour statistics
        """
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
    
    def calculate_colocalization(self, gray_labeled: np.ndarray, fluor_labeled: np.ndarray,
                                gray_props: List[Dict], fluor_props: List[Dict]) -> List[Dict]:
        """
        Calculate which grayscale and fluorescence bacteria are co-localized.
        
        Args:
            gray_labeled: Labeled grayscale regions
            fluor_labeled: Labeled fluorescence regions
            gray_props: Grayscale contour properties
            fluor_props: Fluorescence contour properties
            
        Returns:
            List of co-localization pairs with overlap statistics
        """
        colocalized = []
        overlap_threshold = self.config['overlap_threshold']
        
        for gray_prop in gray_props:
            gray_mask = (gray_labeled == gray_prop['label'])
            
            for fluor_prop in fluor_props:
                fluor_mask = (fluor_labeled == fluor_prop['label'])
                
                # Calculate overlap
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
    
    def create_overlay_visualization(self, image: np.ndarray, labeled: np.ndarray, 
                                    props: List[Dict], is_grayscale: bool = True,
                                    colocalized: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Create overlay visualization on original image.
        
        Args:
            image: Original image
            labeled: Labeled regions
            props: Region properties
            is_grayscale: True for grayscale channel
            colocalized: List of co-localization pairs
            
        Returns:
            RGB visualization with contour overlays
        """
        # Convert to RGB for visualization
        vis = np.dstack([image, image, image]).astype(np.uint8)
        
        # Identify co-localized bacteria
        colocalized_gray_ids = set()
        colocalized_fluor_ids = set()
        if colocalized:
            colocalized_gray_ids = {c['gray_id'] for c in colocalized}
            colocalized_fluor_ids = {c['fluor_id'] for c in colocalized}
        
        # Draw contours
        for i, prop in enumerate(props):
            region_mask = (labeled == prop['label']).astype(np.uint8)
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            
            # Determine if co-localized
            is_colocalized = False
            if is_grayscale:
                is_colocalized = (i + 1) in colocalized_gray_ids
            else:
                is_colocalized = (i + 1) in colocalized_fluor_ids
            
            # Set color and thickness
            if is_colocalized:
                color = (255, 255, 0)  # Yellow for co-localized
                thickness = 3
            else:
                color = (0, 255, 0) if is_grayscale else (255, 0, 0)  # Green or Red
                thickness = 2
            
            # Draw contour
            cv2.drawContours(vis, [contour], -1, color, thickness)
            
            # Add label
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
    
    def create_final_visualization(self, gray_img: np.ndarray, fluor_img: np.ndarray,
                                  gray_labeled: np.ndarray, fluor_labeled: np.ndarray,
                                  gray_props: List[Dict], fluor_large: List,
                                  colocalized: List[Dict]) -> np.ndarray:
        """
        Create final merged visualization showing co-localization.
        
        Args:
            gray_img: Original grayscale image
            fluor_img: Original fluorescence image
            gray_labeled: Labeled grayscale regions
            fluor_labeled: Labeled fluorescence regions
            gray_props: Grayscale properties
            fluor_large: Fluorescence properties
            colocalized: Co-localization pairs
            
        Returns:
            RGB composite visualization
        """
        # Create RGB composite
        gray_norm = gray_img.astype(np.uint8)
        fluor_norm = fluor_img.astype(np.uint8)
        
        vis = np.zeros((*gray_img.shape, 3), dtype=np.uint8)
        vis[:, :, 0] = fluor_norm  # Red channel
        vis[:, :, 2] = gray_norm   # Blue channel
        
        # Draw only co-localized bacteria
        colocalized_gray_ids = {c['gray_id'] for c in colocalized}
        
        for i, prop in enumerate(gray_props):
            if (i + 1) not in colocalized_gray_ids:
                continue
            
            region_mask = (gray_labeled == prop['label']).astype(np.uint8)
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            
            # Draw in bright yellow for co-localized
            cv2.drawContours(vis, [contour], -1, (255, 255, 0), 3)
            
            # Add label
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                cv2.putText(vis, f"{i+1}", (cX-10, cY-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis
    
    def format_statistics_text(self, gray_props: List, fluor_props: List,
                              gray_contour_data: List[Dict], fluor_contour_data: List[Dict],
                              colocalized: List[Dict], filename: str) -> str:
        """
        Format statistics as text for display.
        
        Args:
            gray_props: Grayscale properties
            fluor_props: Fluorescence properties
            gray_contour_data: Grayscale contour statistics
            fluor_contour_data: Fluorescence contour statistics
            colocalized: Co-localization pairs
            filename: Source filename
            
        Returns:
            Formatted statistics text
        """
        lines = []
        
        # Header
        lines.append("═══════════════ DETECTION REPORT ═══════════════")
        lines.append(f"Source: {filename}")
        lines.append(f"Parameters: Gray Thresh={self.config['gray_threshold']}, "
                    f"Fluor Thresh={self.config['fluor_threshold']}, "
                    f"Overlap={self.config['overlap_threshold']*100:.0f}%")
        lines.append("")
        
        # Summary
        lines.append("┌─── SUMMARY ───┐")
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
            lines.append("┌─── CO-LOCALIZED PAIRS ───┐")
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
    
    def generate_review_sheet(self, pair_index: int) -> Figure:
        """
        Generate a review sheet for a single image pair.
        
        Args:
            pair_index: Index of the image pair to process
            
        Returns:
            Matplotlib figure containing the review sheet
        """
        pair = self.image_pairs[pair_index]
        gray_img = pair['gray']
        fluor_img = pair['fluor']
        filename = pair['gray_file']
        fluor_path = os.path.join(self.source_folder, pair['fluor_file'])
        
        # Load original fluorescence image directly from source
        try:
            orig_fluor_img = io.imread(fluor_path)
        except Exception as e:
            print(f"Error loading original fluorescence image {pair['fluor_file']}: {e}")
            orig_fluor_img = fluor_img
        
        # Detect bacteria
        gray_proc, gray_labeled, gray_props, gray_large = self.detect_bacteria(gray_img, is_grayscale=True)
        fluor_proc, fluor_labeled, fluor_props, fluor_large = self.detect_bacteria(fluor_img, is_grayscale=False)
        
        # Get contour intensity data
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
        
        # Create figure with portrait orientation
        fig = plt.figure(figsize=(12, 16))
        
        # Set window title
        manager = getattr(fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(f'Review: {filename}')
        
        # Create grid layout
        gs = fig.add_gridspec(5, 2, 
                             height_ratios=[1.5, 1.5, 1.2, 1.0, 0.3],
                             width_ratios=[1, 1],
                             hspace=0.35, wspace=0.2,
                             left=0.05, right=0.95, top=0.90, bottom=0.10)
        
        # Add main title
        fig.suptitle(f'Bacteria Detection Review Sheet\nSource: {self.source_folder} | File: {filename}',
                    fontsize=12, fontweight='bold', y=0.95)
        
        # Row 1: Grayscale and Fluorescence with overlays
        ax_gray = fig.add_subplot(gs[0, 0])
        gray_vis = self.create_overlay_visualization(gray_img, gray_labeled, gray_large, 
                                                     is_grayscale=True, colocalized=colocalized)
        ax_gray.imshow(gray_vis)
        ax_gray.set_title(f'GRAYSCALE CHANNEL\n{len(gray_large)} detected | {len(colocalized)} co-localized', 
                         fontsize=10, fontweight='bold')
        ax_gray.axis('off')
        
        ax_fluor = fig.add_subplot(gs[0, 1])
        fluor_vis = self.create_overlay_visualization(fluor_img, fluor_labeled, fluor_large, 
                                                      is_grayscale=False, colocalized=colocalized)
        ax_fluor.imshow(fluor_vis)
        ax_fluor.set_title(f'FLUORESCENCE CHANNEL\n{len(fluor_large)} detected', 
                          fontsize=10, fontweight='bold')
        ax_fluor.axis('off')
        
        # Row 2: Original fluorescence and Final output
        ax_orig_fluor = fig.add_subplot(gs[1, 0])
        ax_orig_fluor.imshow(orig_fluor_img)
        ax_orig_fluor.set_title('ORIGINAL FLUORESCENCE\n(Direct from _ch01.tif)', 
                               fontsize=10, fontweight='bold')
        ax_orig_fluor.axis('off')
        
        ax_final = fig.add_subplot(gs[1, 1])
        final_vis = self.create_final_visualization(gray_img, fluor_img, gray_labeled, fluor_labeled, 
                                                    gray_large, fluor_large, colocalized)
        ax_final.imshow(final_vis)
        coloc_pct = (len(colocalized) / len(gray_large) * 100) if len(gray_large) > 0 else 0
        ax_final.set_title(f'FINAL OUTPUT\nCo-localization: {coloc_pct:.1f}%', 
                          fontsize=10, fontweight='bold')
        ax_final.axis('off')
        
        # Row 3: Statistics
        ax_stats = fig.add_subplot(gs[2, :])
        ax_stats.axis('off')
        stats_text = self.format_statistics_text(gray_large, fluor_large, 
                                                gray_contour_data, fluor_contour_data, 
                                                colocalized, filename)
        ax_stats.text(0.02, 0.98, stats_text, 
                     ha='left', va='top', 
                     fontsize=6, fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3, pad=0.5),
                     transform=ax_stats.transAxes)
        
        # Row 4: Parameters
        ax_params = fig.add_subplot(gs[3, :])
        ax_params.axis('off')
        info_text = (
            "DETECTION PARAMETERS:\n"
            f"• Grayscale Threshold: {self.config['gray_threshold']}\n"
            f"• Fluorescence Threshold: {self.config['fluor_threshold']}\n"
            f"• Min Grayscale Area: {self.config['min_area']} px\n"
            f"• Min Fluorescence Area: {self.config['min_fluor_area']} px\n"
            f"• Overlap Threshold: {self.config['overlap_threshold']*100:.0f}%\n\n"
            "INTENSITY STATISTICS:\n"
            "• Using MODE instead of mean\n"
            "• Min intensity removed\n\n"
            "DISPLAY:\n"
            "• Portrait orientation\n"
            "• Original fluorescence from _ch01.tif (unprocessed)\n"
            "• Reserved space for header and footer\n"
            "• Maximized image sizes\n"
            "• Fixed text overlap\n\n"
            "COLOR LEGEND:\n"
            "• Green: Grayscale bacteria\n"
            "• Red: Fluorescence bacteria\n"
            "• Yellow: Co-localized bacteria\n"
            "• * marker: Co-localized ID"
        )
        ax_params.text(0.02, 0.98, info_text,
                     ha='left', va='top',
                     fontsize=7.5,
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2, pad=0.5),
                     transform=ax_params.transAxes)
        
        # Row 5: Empty space
        ax_empty = fig.add_subplot(gs[4, :])
        ax_empty.axis('off')
        
        return fig
    
    def generate_all_review_sheets(self) -> None:
        """Generate review sheets for all image pairs."""
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
        print("✓ All review sheets generated successfully!")
        print(f"{'='*70}\n")
    
    def export_to_excel(self) -> None:
        """Export all results to Excel file."""
        print("\nExporting results to Excel...")
        
        # Create Excel writer
        output_file = f"bacteria_detection_results_{self.source_folder.replace('/', '_')}.xlsx"
        writer = pd.ExcelWriter(output_file, engine='openpyxl')
        
        # Sheet 1: Summary
        summary_data = []
        for result in self.all_results:
            coloc_rate = (result['coloc_count'] / result['gray_count'] * 100) if result['gray_count'] > 0 else 0
            
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
        
        # Sheet 2: Grayscale Details
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
        
        # Sheet 3: Fluorescence Details
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
    """Main function to run the review tool."""
    print("="*70)
    print("BACTERIA DETECTION REVIEW SHEET GENERATOR WITH EXCEL EXPORT")
    print("="*70)
    print("\nFeatures:")
    print("  • Dual-channel bacteria detection (grayscale + fluorescence)")
    print("  • Co-localization analysis with overlap threshold")
    print("  • MODE intensity statistics for robust analysis")
    print("  • Visual review sheets with detection overlays")
    print("  • Comprehensive Excel export with multiple sheets")
    print("  • Portrait orientation optimized for review")
    print("\n" + "="*70 + "\n")
    
    # Configure source folder
    source_folder = "source/1"
    
    try:
        tool = BacteriaReviewTool(source_folder=source_folder)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


