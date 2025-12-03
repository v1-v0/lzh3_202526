#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bacteria segmentation algorithms.
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import List, Sequence, Tuple, cast
from cv2.typing import MatLike


class BacteriaSegmenter:
    """Handles bacteria segmentation from bright-field microscopy images."""
    
    def watershed_separation(self, cleaned: np.ndarray, 
                             gray_bf: np.ndarray) -> List[np.ndarray]:
        """Apply watershed algorithm to separate touching bacteria."""
        distance = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
        
        threshold_val = (
            self.params['watershed_dilate'].get() * distance.max() / 100.0
        )
        _, sure_fg = cv2.threshold(distance, threshold_val, 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        
        # Additional opening to separate markers
        open_k_size = self._ensure_odd(self.params['open_kernel'].get())
        open_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_k_size, open_k_size)
        )
        sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, open_k, iterations=1)
        
        markers, _ = cast(Tuple[np.ndarray, int], ndimage.label(sure_fg))
        markers = markers.astype(np.int32)
        markers += 1
        markers[cleaned == 0] = 0
        
        watershed_input = cv2.cvtColor(gray_bf, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(watershed_input, markers)
        
        contour_mask = (markers > 1).astype(np.uint8) * 255
        res = cv2.findContours(
            contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours: Sequence[MatLike] = res[-2]
        
        # Filter by minimum area
        min_area = self.params['min_area'].get()
        bacteria = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        return bacteria
    
    def __init__(self, params: dict):
        """Initialize segmenter with parameters.
        
        Args:
            params: Dictionary of segmentation parameters
        """
        self.params = params
    
    def segment(self, gray_bf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Segment bacteria from bright-field image.
        
        Args:
            gray_bf: Grayscale bright-field image
            
        Returns:
            Tuple of (enhanced, threshold, cleaned, contours)
        """
        enhanced = self._enhance_image(gray_bf)
        thresh = self._threshold_image(enhanced)
        cleaned = self._morphological_cleanup(thresh)
        contours = self.watershed_separation(cleaned, gray_bf)
        
        return enhanced, thresh, cleaned, contours
    
    def _enhance_image(self, gray_bf: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement if enabled."""
        if self.params['enable_clahe'].get():
            bf8 = cv2.convertScaleAbs(gray_bf)
            clahe = cv2.createCLAHE(
                clipLimit=self.params['clahe_clip'].get(),
                tileGridSize=(self.params['clahe_tile'].get(),) * 2
            )
            return clahe.apply(bf8)
        return cv2.convertScaleAbs(gray_bf)
    
    def _threshold_image(self, enhanced: np.ndarray) -> np.ndarray:
        """Apply thresholding (Otsu or manual)."""
        if self.params['use_otsu'].get():
            _, thresh = cv2.threshold(
                enhanced, 0, 255, 
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        else:
            _, thresh = cv2.threshold(
                enhanced, 
                self.params['manual_threshold'].get(), 
                255, 
                cv2.THRESH_BINARY_INV
            )
        return thresh
    
    def _morphological_cleanup(self, thresh: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up binary image."""
        open_k_size = self._ensure_odd(self.params['open_kernel'].get())
        close_k_size = self._ensure_odd(self.params['close_kernel'].get())
        
        open_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_k_size, open_k_size)
        )
        close_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_k_size, close_k_size)
        )
        
        opened = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, open_k, 
            iterations=self.params['open_iter'].get()
        )
        cleaned = cv2.morphologyEx(
            opened, cv2.MORPH_CLOSE, close_k, 
            iterations=self.params['close_iter'].get()
        )
        return cleaned
    
    @staticmethod
    def _ensure_odd(value: int) -> int:
        """Ensure kernel size is odd."""
        if value % 2 == 0:
            return max(1, value - 1)
        return value
    
    @staticmethod
    def filter_bacteria(contours, min_area):
        """Filter bacteria contours by minimum area.
        
        Args:
            contours: List of contours
            min_area: Minimum area threshold in pixels
            
        Returns:
            Filtered list of contours
        """
        filtered = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                filtered.append(contour)
        
        return filtered
    
    @staticmethod
    def watershed_segmentation(binary, watershed_dilate):
        """Perform watershed segmentation on binary image.
        
        Args:
            binary: Input binary image
            watershed_dilate: Number of dilation iterations before watershed
            
        Returns:
            Tuple of (labeled_image, contours)
        """
        # Distance transform
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Find sure foreground area
        if watershed_dilate > 0:
            kernel = np.ones((3, 3), np.uint8)
            sure_fg = cv2.dilate(dist_transform, kernel, iterations=watershed_dilate)
        else:
            sure_fg = dist_transform
        
        # Threshold to get markers
        _, sure_fg = cv2.threshold(sure_fg, 0.3 * sure_fg.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        
        # Find unknown region
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(binary, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg.astype(np.uint8), sure_fg.astype(np.uint8))
        
        # Label markers
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        binary_3ch = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(binary_3ch, markers)
        
        # Create labeled image
        labeled_image = markers.copy()
        labeled_image[labeled_image == -1] = 0
        
        # Extract contours
        contours = []
        for label in range(2, int(markers.max()) + 1):
            mask = ((markers == label) * 255).astype(np.uint8)
            res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = res[-2] if len(res) >= 2 else res[0]
            if len(cnts) > 0:
                contours.extend(cnts)
        
        return labeled_image, contours