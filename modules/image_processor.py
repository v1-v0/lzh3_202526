# modules/image_processor.py
"""
Image pre-processing module for enhancement and preparation
"""

import cv2
import numpy as np
import os

class ImageProcessor:
    """Handles all image pre-processing and enhancement"""
    
    def __init__(self, config):
        self.config = config
    
    def preprocess(self, img_bf, img_red, pixel_size, unit):
        """
        Pre-process both brightfield and fluorescence images
        
        Returns:
            tuple: (img_bf_processed, img_red_enhanced, img_red_original)
        """
        print("\nProcessing brightfield...")
        
        # Keep original red channel
        img_red_original = img_red.copy()
        
        # Convert to 8-bit if needed
        img_bf_8bit = self._convert_to_8bit(img_bf, "Brightfield")
        img_red_8bit = self._convert_to_8bit(img_red, "Red channel")
        
        # Save before enhancement
        self._save_debug("01c_bf_before_normalize.png", img_bf_8bit, pixel_size, unit)
        self._save_debug("01e_red_before_enhance.png", img_red_8bit, pixel_size, unit)
        
        # Apply enhancement
        print("\nApplying normalization and enhancement...")
        
        img_bf_normalized = self._adjust_channel(
            img_bf_8bit,
            normalize=self.config.BF_NORMALIZE,
            brightness=self.config.BF_BRIGHTNESS,
            gamma=self.config.BF_GAMMA,
            channel_name="Brightfield"
        )
        
        img_red_enhanced = self._adjust_channel(
            img_red_8bit,
            normalize=self.config.RED_NORMALIZE,
            brightness=self.config.RED_BRIGHTNESS,
            gamma=self.config.RED_GAMMA,
            channel_name="Red channel"
        )
        
        # Save after enhancement
        self._save_debug("01d_bf_after_normalize.png", img_bf_normalized, pixel_size, unit)
        self._save_debug("01f_red_after_enhance.png", img_red_enhanced, pixel_size, unit)
        
        return img_bf_normalized, img_red_enhanced, img_red_original
    
    def _convert_to_8bit(self, img, name):
        """Convert image to 8-bit if needed"""
        if img.dtype == np.uint16:
            img_8bit = np.zeros_like(img, dtype=np.uint8)
            cv2.normalize(img, img_8bit, 0, 255, cv2.NORM_MINMAX)
            print(f"  ✓ {name}: Converted to 8-bit")
            return img_8bit
        return img.copy()
    
    def _adjust_channel(self, img, normalize=True, brightness=1.0, gamma=1.0, channel_name="Channel"):
        """Apply normalization, brightness, and gamma adjustments"""
        img_float = img.astype(np.float32)
        
        if normalize:
            min_val = np.min(img_float)
            max_val = np.max(img_float)
            if max_val > min_val:
                img_float = (img_float - min_val) * 255.0 / (max_val - min_val)
            print(f"  ✓ {channel_name} normalized: [{min_val:.1f}, {max_val:.1f}] → [0, 255]")
        
        if brightness != 1.0:
            img_float = img_float * brightness
            img_float = np.clip(img_float, 0, 255)
            print(f"  ✓ {channel_name} brightness: {brightness}x")
        
        if gamma != 1.0:
            img_normalized = img_float / 255.0
            img_normalized = np.power(img_normalized, gamma)
            img_float = img_normalized * 255.0
            print(f"  ✓ {channel_name} gamma: {gamma}")
        
        return img_float.astype(np.uint8)
    
    def _add_scale_bar(self, img, pixel_size, unit='um', length_um=None):
        """Add scale bar to image"""
        if length_um is None:
            length_um = self.config.SCALE_BAR_LENGTH_UM
        
        if pixel_size is None or pixel_size <= 0:
            return img
        
        bar_length_px = int(round(length_um / pixel_size))
        
        if bar_length_px < 10:
            return img
        
        h, w = img.shape[:2]
        bar_x = w - bar_length_px - self.config.SCALE_BAR_MARGIN
        bar_y = h - self.config.SCALE_BAR_HEIGHT - self.config.SCALE_BAR_MARGIN
        
        label = f"{length_um} um" if unit in ['µm', 'um'] else f"{length_um} {unit}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        (text_w, text_h), _ = cv2.getTextSize(
            label, font, self.config.SCALE_BAR_FONT_SCALE, self.config.SCALE_BAR_FONT_THICKNESS
        )
        
        text_x = bar_x + (bar_length_px - text_w) // 2
        text_y = bar_y - 8
        
        # Background
        bg_padding = 5
        bg_x1 = min(bar_x, text_x) - bg_padding
        bg_y1 = text_y - text_h - bg_padding
        bg_x2 = max(bar_x + bar_length_px, text_x + text_w) + bg_padding
        bg_y2 = bar_y + self.config.SCALE_BAR_HEIGHT + bg_padding
        
        overlay = img.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), 
                      self.config.SCALE_BAR_BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Scale bar
        cv2.rectangle(img, (bar_x, bar_y), 
                      (bar_x + bar_length_px, bar_y + self.config.SCALE_BAR_HEIGHT),
                      self.config.SCALE_BAR_COLOR, -1)
        
        # Text
        cv2.putText(img, label, (text_x, text_y), font, 
                    self.config.SCALE_BAR_FONT_SCALE, self.config.SCALE_BAR_TEXT_COLOR, 
                    self.config.SCALE_BAR_FONT_THICKNESS, cv2.LINE_AA)
        
        return img
    
    def _save_debug(self, name, img, pixel_size=None, unit='um'):
        """Save debug image with optional scale bar"""
        path = os.path.join(self.config.DEBUG_DIR, name)
        img_with_scale = img.copy()
        
        if pixel_size is not None and pixel_size > 0:
            img_with_scale = self._add_scale_bar(img_with_scale, pixel_size, unit)
        
        cv2.imwrite(path, img_with_scale)