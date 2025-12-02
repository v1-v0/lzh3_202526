#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image processing utilities for microscopy images.
"""

import cv2
import numpy as np
from typing import Optional, List


class ImageProcessor:
    """Handles image processing operations."""
    
    @staticmethod
    def load_image(path: str, grayscale: bool = True) -> Optional[np.ndarray]:
        """Load image from file.
        
        Args:
            path: Path to image file
            grayscale: Convert to grayscale if True
            
        Returns:
            Loaded image or None if failed
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        
        if grayscale and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return img
    
    @staticmethod
    def apply_fluorescence_adjustments(img: np.ndarray, 
                                      brightness: float, 
                                      gamma: float) -> np.ndarray:
        """Apply brightness and gamma adjustments to fluorescence image.
        
        Args:
            img: Input fluorescence image
            brightness: Brightness multiplier
            gamma: Gamma correction value
            
        Returns:
            Adjusted image (8-bit)
        """
        f = img.astype(np.float32)
        f = f / (f.max() if f.max() else 1)
        f = np.power(f, gamma) * brightness
        f = np.clip(f, 0, 1)
        return (f * 255).astype(np.uint8)
    
    @staticmethod
    def create_overlay(bf_img: np.ndarray, 
                      fluor_img: Optional[np.ndarray],
                      contours: List[np.ndarray],
                      brightness: float = 2.0,
                      gamma: float = 0.5) -> np.ndarray:
        """Create overlay combining bright-field, fluorescence, and contours.
        
        Args:
            bf_img: Bright-field grayscale image
            fluor_img: Fluorescence grayscale image (optional)
            contours: List of bacteria contours
            brightness: Fluorescence brightness multiplier
            gamma: Fluorescence gamma correction
            
        Returns:
            RGB overlay image
        """
        overlay = cv2.cvtColor(bf_img, cv2.COLOR_GRAY2RGB)
        
        if fluor_img is not None:
            f8 = ImageProcessor.apply_fluorescence_adjustments(
                fluor_img, brightness, gamma
            )
            red_channel = overlay[:, :, 2].astype(np.float32)
            red_channel = np.clip(red_channel + f8.astype(np.float32), 0, 255)
            overlay[:, :, 2] = red_channel.astype(np.uint8)
        
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
        return overlay
    
    @staticmethod
    def create_fluorescence_display(img: np.ndarray,
                                   brightness: float,
                                   gamma: float) -> np.ndarray:
        """Create RGB display of fluorescence channel.
        
        Args:
            img: Fluorescence grayscale image
            brightness: Brightness multiplier
            gamma: Gamma correction value
            
        Returns:
            RGB image with red channel
        """
        f8 = ImageProcessor.apply_fluorescence_adjustments(img, brightness, gamma)
        red = np.zeros((f8.shape[0], f8.shape[1], 3), dtype=np.uint8)
        red[:, :, 2] = f8
        return cv2.cvtColor(red, cv2.COLOR_BGR2RGB)