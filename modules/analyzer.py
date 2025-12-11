# modules/analyzer.py
"""
Particle analysis module for contouring, counting, sizing, and intensity measurement
"""

import cv2
import numpy as np
import os

class ParticleAnalyzer:
    """Handles particle segmentation and measurement"""
    
    def __init__(self, config):
        self.config = config
    
    def analyze(self, img_bf, img_red_original, img_red_enhanced, pixel_size, unit, bit_depth):
        """
        Perform complete particle analysis
        
        Returns:
            tuple: (object_data, objects_excluded)
        """
        # Segment particles
        print("Segmenting particles...")
        contours = self._segment_brightfield(img_bf, pixel_size, unit)
        
        # Filter by area
        print(f"Filtering by area ({self.config.MIN_OBJECT_AREA}-{self.config.MAX_OBJECT_AREA} pixels)...")
        filtered = [c for c in contours 
                   if self.config.MIN_OBJECT_AREA <= cv2.contourArea(c) <= self.config.MAX_OBJECT_AREA]
        print(f"  ✓ {len(filtered)} particles after filtering")
        
        # Calculate conversion factors
        area_factor = (pixel_size ** 2) if pixel_size else 1.0
        max_possible_value = self._get_max_value(bit_depth)
        bit_conversion_factor = max_possible_value / 255.0 if bit_depth > 8 else 1.0
        
        # Measure fluorescence
        print("Measuring fluorescence intensity...")
        object_data, objects_excluded = self._measure_fluorescence(
            filtered, img_red_original, img_red_enhanced,
            pixel_size, area_factor, unit, bit_depth, bit_conversion_factor
        )
        
        # Create visualizations
        print("Creating visualizations...")
        self._create_visualizations(
            img_bf, img_red_enhanced, object_data, pixel_size, unit
        )
        
        return object_data, objects_excluded
    
    def _segment_brightfield(self, img_bf, pixel_size, unit):
        """Segment particles from brightfield image"""
        # Background subtraction
        bg = cv2.GaussianBlur(img_bf, (0, 0), sigmaX=self.config.GAUSSIAN_SIGMA, 
                              sigmaY=self.config.GAUSSIAN_SIGMA)
        enhanced = cv2.subtract(bg, img_bf)
        enhanced_blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        self._save_debug("02_enhanced.png", enhanced, pixel_size, unit)
        self._save_debug("03_enhanced_blur.png", enhanced_blur, pixel_size, unit)
        
        # Threshold
        _, thresh = cv2.threshold(enhanced_blur, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._save_debug("04_thresh_raw.png", thresh, pixel_size, unit)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, 
                                  iterations=self.config.MORPH_ITERATIONS)
        closed = cv2.dilate(closed, kernel, iterations=self.config.DILATE_ITERATIONS)
        closed = cv2.erode(closed, kernel, iterations=self.config.ERODE_ITERATIONS)
        
        self._save_debug("05_closed.png", closed, pixel_size, unit)
        
        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            closed, connectivity=8
        )
        
        solid = np.where(labels > 0, 255, 0).astype(np.uint8)
        self._save_debug("06_solid.png", solid, pixel_size, unit)
        
        # Find contours
        contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  ✓ {len(contours)} initial contours found")
        
        return contours
    
    def _measure_fluorescence(self, contours, img_red_orig, img_red_enh, 
                             pixel_size, area_factor, unit, bit_depth, bit_conversion_factor):
        """Measure fluorescence intensity for each particle"""
        object_data = []
        objects_excluded = 0
        
        # Convert to 8-bit for scaled measurements
        img_red_8bit = np.zeros_like(img_red_orig, dtype=np.uint8)
        cv2.normalize(img_red_orig, img_red_8bit, 0, 255, cv2.NORM_MINMAX)
        
        for c in contours:
            # Geometric measurements
            area_px = cv2.contourArea(c)
            perimeter_px = cv2.arcLength(c, True)
            area_physical = area_px * area_factor
            perimeter_physical = perimeter_px * pixel_size if pixel_size else perimeter_px
            
            # Centroid
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            
            # Create mask
            mask = np.zeros_like(img_red_orig, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            
            # Measurements from original bit depth
            red_pixels_orig = img_red_orig[mask == 255]
            if len(red_pixels_orig) > 0:
                red_total_orig = float(np.sum(red_pixels_orig.astype(np.float64)))
                red_mean_orig = float(np.mean(red_pixels_orig.astype(np.float64)))
                red_std_orig = float(np.std(red_pixels_orig.astype(np.float64)))
            else:
                red_total_orig = 0.0
                red_mean_orig = 0.0
                red_std_orig = 0.0
            
            # Skip particles with no fluorescence
            if red_total_orig == 0.0:
                objects_excluded += 1
                continue
            
            # Intensity per area
            intensity_per_area_orig = red_total_orig / area_physical if area_physical > 0 else 0.0
            mean_intensity_per_area_orig = red_mean_orig / area_factor if area_factor > 0 else red_mean_orig
            
            # Measurements from 8-bit scaled
            red_pixels_8bit = img_red_8bit[mask == 255]
            if len(red_pixels_8bit) > 0:
                red_total_8bit = float(np.sum(red_pixels_8bit.astype(np.float64)))
                red_mean_8bit = float(np.mean(red_pixels_8bit.astype(np.float64)))
                red_total_8bit_scaled = red_total_8bit * bit_conversion_factor
                red_mean_8bit_scaled = red_mean_8bit * bit_conversion_factor
            else:
                red_total_8bit = 0.0
                red_mean_8bit = 0.0
                red_total_8bit_scaled = 0.0
                red_mean_8bit_scaled = 0.0
            
            intensity_per_area_8bit_scaled = red_total_8bit_scaled / area_physical if area_physical > 0 else 0.0
            
            # Measurements from enhanced
            red_pixels_enh = img_red_enh[mask == 255]
            if len(red_pixels_enh) > 0:
                red_total_enh = float(np.sum(red_pixels_enh.astype(np.float64)))
                red_mean_enh = float(np.mean(red_pixels_enh.astype(np.float64)))
            else:
                red_total_enh = 0.0
                red_mean_enh = 0.0
            
            intensity_per_area_enh = red_total_enh / area_physical if area_physical > 0 else 0.0
            
            object_data.append({
                'contour': c,
                'area_px': area_px,
                'area_physical': area_physical,
                'perimeter_px': perimeter_px,
                'perimeter_physical': perimeter_physical,
                'centroid_x': cx,
                'centroid_y': cy,
                'red_total_orig': red_total_orig,
                'red_mean_orig': red_mean_orig,
                'red_std_orig': red_std_orig,
                'intensity_per_area_orig': intensity_per_area_orig,
                'mean_intensity_per_area_orig': mean_intensity_per_area_orig,
                'red_total_8bit': red_total_8bit,
                'red_mean_8bit': red_mean_8bit,
                'red_total_8bit_scaled': red_total_8bit_scaled,
                'red_mean_8bit_scaled': red_mean_8bit_scaled,
                'intensity_per_area_8bit_scaled': intensity_per_area_8bit_scaled,
                'red_total_enh': red_total_enh,
                'red_mean_enh': red_mean_enh,
                'intensity_per_area_enh': intensity_per_area_enh,
            })
        
        # Sort by intensity
        object_data.sort(key=lambda x: x['intensity_per_area_orig'], reverse=True)
        
        # Assign IDs
        for i, obj in enumerate(object_data, 1):
            obj['object_id'] = i
        
        print(f"  ✓ {len(object_data)} particles measured")
        print(f"  ✓ {objects_excluded} particles excluded (no fluorescence)")
        
        return object_data, objects_excluded
    
    def _create_visualizations(self, img_bf, img_red_enh, object_data, pixel_size, unit):
        """Create visualization images"""
        # Brightfield with contours
        vis_bf = cv2.cvtColor(img_bf, cv2.COLOR_GRAY2BGR)
        for obj in object_data:
            cv2.drawContours(vis_bf, [obj['contour']], -1, (0, 0, 255), 1)
            label = str(obj['object_id'])
            cv2.putText(vis_bf, label, (obj['centroid_x'], obj['centroid_y']), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        self._save_debug("07a_bf_contours.png", vis_bf, pixel_size, unit)
        
        # Red channel with contours
        vis_red = np.zeros((img_red_enh.shape[0], img_red_enh.shape[1], 3), dtype=np.uint8)
        vis_red[:, :, 2] = img_red_enh
        vis_red[:, :, 0] = img_red_enh // 8
        vis_red[:, :, 1] = img_red_enh // 8
        
        for obj in object_data:
            cv2.drawContours(vis_red, [obj['contour']], -1, (0, 255, 0), 1)
            label = str(obj['object_id'])
            cv2.putText(vis_red, label, (obj['centroid_x'], obj['centroid_y']), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        self._save_debug("07b_red_contours.png", vis_red, pixel_size, unit)
        
        # Intensity-coded visualization
        vis_intensity = cv2.cvtColor(img_bf, cv2.COLOR_GRAY2BGR)
        intensities = [obj['intensity_per_area_orig'] for obj in object_data]
        
        if intensities:
            min_int, max_int = min(intensities), max(intensities)
            for obj in object_data:
                intensity = obj['intensity_per_area_orig']
                if max_int > min_int:
                    normalized = (intensity - min_int) / (max_int - min_int)
                else:
                    normalized = 0.5
                color = (int(255 * normalized), 0, int(255 * (1 - normalized)))
                cv2.drawContours(vis_intensity, [obj['contour']], -1, color, 2)
        
        self._save_debug("07d_intensity_coded.png", vis_intensity, pixel_size, unit)
    
    def _get_max_value(self, bit_depth):
        """Get maximum possible value for bit depth"""
        if bit_depth == 12:
            return 4095
        elif bit_depth == 14:
            return 16383
        elif bit_depth == 16:
            return 65535
        else:
            return 255
    
    def _save_debug(self, name, img, pixel_size, unit):
        """Save debug image"""
        from modules.image_processor import ImageProcessor
        processor = ImageProcessor(self.config)
        processor._save_debug(name, img, pixel_size, unit)