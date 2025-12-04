#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Drawing utilities for labels, scale bars, and overlays.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional, Dict, Union
from ..config.parameters import FONT_PATHS


class LabelDrawer:
    """Handles smart label positioning and drawing."""
    
    def __init__(self, 
                 arrow_length: int = 60,
                 label_offset: int = 15,
                 font_size: int = 20):
        """Initialize label drawer.
        
        Args:
            arrow_length: Length of arrow in pixels
            label_offset: Offset from arrow tip to label
            font_size: Font size for labels
        """
        self.arrow_length = arrow_length
        self.label_offset = label_offset
        self.font_size = font_size
        self.font = self._load_font()
    
    def _load_font(self) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
        """Load font from system, trying multiple paths."""
        for fp in FONT_PATHS:
            try:
                return ImageFont.truetype(fp, self.font_size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()
    
    def draw_labels(self,
                   img_bgr: np.ndarray,
                   stats: List[Dict],
                   selected_index: int = -1) -> np.ndarray:
        """Draw numbered labels on bacteria with smart positioning.
        
        Args:
            img_bgr: BGR image to draw on
            stats: List of bacteria statistics
            selected_index: Index of selected bacterium (-1 for none)
            
        Returns:
            Image with drawn labels
        """
        if not stats:
            return img_bgr
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        h, w = img_bgr.shape[:2]
        contours = [s['contour'] for s in stats]
        occupancy_map = self._create_occupancy_map((h, w), contours)
        
        for idx, stat in enumerate(stats):
            self._draw_single_label(
                draw, stat, idx, occupancy_map, 
                (h, w), selected_index == idx
            )
        
        img_rgb_array = np.array(pil_img)
        return cv2.cvtColor(img_rgb_array, cv2.COLOR_RGB2BGR)
    
    def _create_occupancy_map(self, 
                             img_shape: Tuple[int, int],
                             contours: List[np.ndarray],
                             margin: int = 20) -> np.ndarray:
        """Create map showing occupied regions."""
        h, w = img_shape
        occupancy = np.zeros((h, w), dtype=np.uint8)
        
        for contour in contours:
            cv2.drawContours(occupancy, [contour], -1, 255, -1)
            if margin > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (margin*2, margin*2)
                )
                occupancy = cv2.dilate(occupancy, kernel, iterations=1)
        
        # Mark borders as occupied
        occupancy[:margin, :] = 255
        occupancy[-margin:, :] = 255
        occupancy[:, :margin] = 255
        occupancy[:, -margin:] = 255
        
        return occupancy
    
    def _draw_single_label(self,
                          draw: ImageDraw.ImageDraw,
                          stat: Dict,
                          idx: int,
                          occupancy_map: np.ndarray,
                          img_shape: Tuple[int, int],
                          is_selected: bool) -> None:
        """Draw single label with arrow."""
        contour = stat['contour']
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        label_text = str(idx + 1)
        bbox = draw.textbbox((0, 0), label_text, font=self.font)
        text_w = int(bbox[2] - bbox[0])
        text_h = int(bbox[3] - bbox[1])
        
        result = self._find_best_position(
            (cx, cy), img_shape, (text_w, text_h), occupancy_map
        )
        
        if result is None:
            return
        
        arrow_x, arrow_y, label_x, label_y, angle = result
        
        # Colors
        arrow_color = (255, 128, 0) if is_selected else (255, 255, 0)
        arrow_width = 3 if is_selected else 2
        
        # Draw arrow
        draw.line([(cx, cy), (arrow_x, arrow_y)], 
                 fill=arrow_color, width=arrow_width)
        
        # Draw arrowhead
        self._draw_arrowhead(draw, (arrow_x, arrow_y), angle, 
                           arrow_color, arrow_width, is_selected)
        
        # Draw label background
        padding = 4
        bg_rect = [
            label_x - padding, label_y - padding,
            label_x + text_w + padding, label_y + text_h + padding
        ]
        draw.rectangle(bg_rect, fill=(0, 0, 0, 200))
        
        # Draw text
        draw.text((label_x, label_y), label_text, 
                 font=self.font, fill=arrow_color)
        
        # Update occupancy
        h, w = occupancy_map.shape
        y1, y2 = max(0, label_y), min(h, label_y + text_h)
        x1, x2 = max(0, label_x), min(w, label_x + text_w)
        occupancy_map[y1:y2, x1:x2] = 255
    
    def _find_best_position(self,
                           centroid: Tuple[int, int],
                           img_shape: Tuple[int, int],
                           label_size: Tuple[int, int],
                           occupancy_map: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
        """Find best label position by testing angles."""
        cx, cy = centroid
        h, w = img_shape
        label_w, label_h = label_size
        
        angles = [i * 22.5 for i in range(16)]
        best_score = float('inf')
        best_pos = None
        
        for angle in angles:
            score = self._calculate_position_score(
                centroid, angle, label_size, occupancy_map
            )
            
            if score < best_score:
                best_score = score
                rad = np.deg2rad(angle)
                arrow_x = int(cx + self.arrow_length * np.cos(rad))
                arrow_y = int(cy - self.arrow_length * np.sin(rad))
                label_x = int(arrow_x + self.label_offset * np.cos(rad) - label_w / 2)
                label_y = int(arrow_y - self.label_offset * np.sin(rad) - label_h / 2)
                label_x = max(0, min(label_x, w - label_w))
                label_y = max(0, min(label_y, h - label_h))
                best_pos = (arrow_x, arrow_y, label_x, label_y, angle)
        
        return best_pos
    
    def _calculate_position_score(self,
                                  centroid: Tuple[int, int],
                                  angle: float,
                                  label_size: Tuple[int, int],
                                  occupancy_map: np.ndarray) -> float:
        """Calculate quality score for label position."""
        cx, cy = centroid
        h, w = occupancy_map.shape
        label_w, label_h = label_size
        
        rad = np.deg2rad(angle)
        arrow_x = int(cx + self.arrow_length * np.cos(rad))
        arrow_y = int(cy - self.arrow_length * np.sin(rad))
        label_x = int(arrow_x + self.label_offset * np.cos(rad) - label_w / 2)
        label_y = int(arrow_y - self.label_offset * np.sin(rad) - label_h / 2)
        
        # Check bounds
        if (label_x < 0 or label_x + label_w >= w or
            label_y < 0 or label_y + label_h >= h):
            return float('inf')
        
        # Check label overlap
        label_region = occupancy_map[label_y:label_y+label_h, 
                                    label_x:label_x+label_w]
        occupied_pixels = np.sum(label_region > 0)
        total_pixels = label_region.size
        
        if total_pixels == 0:
            return float('inf')
        
        # Check arrow overlap
        arrow_score = 0
        num_samples = 10
        for i in range(num_samples):
            t = i / num_samples
            sx = int(cx + t * self.arrow_length * np.cos(rad))
            sy = int(cy - t * self.arrow_length * np.sin(rad))
            if 0 <= sx < w and 0 <= sy < h and occupancy_map[sy, sx] > 0:
                arrow_score += 10
        
        return (occupied_pixels / total_pixels) * 100 + arrow_score
    
    def _draw_arrowhead(self,
                       draw: ImageDraw.ImageDraw,
                       tip: Tuple[int, int],
                       angle: float,
                       color: Tuple[int, int, int],
                       width: int,
                       is_selected: bool) -> None:
        """Draw arrowhead at arrow tip."""
        arrow_x, arrow_y = tip
        head_len = 10 if is_selected else 8
        head_angle = 25
        
        angle_rad = np.deg2rad(angle)
        
        # Left wing
        left_angle = angle_rad + np.deg2rad(180 - head_angle)
        left_x = int(arrow_x + head_len * np.cos(left_angle))
        left_y = int(arrow_y - head_len * np.sin(left_angle))
        draw.line([(arrow_x, arrow_y), (left_x, left_y)], 
                 fill=color, width=width)
        
        # Right wing
        right_angle = angle_rad + np.deg2rad(180 + head_angle)
        right_x = int(arrow_x + head_len * np.cos(right_angle))
        right_y = int(arrow_y - head_len * np.sin(right_angle))
        draw.line([(arrow_x, arrow_y), (right_x, right_y)], 
                 fill=color, width=width)


class ScaleBarDrawer:
    """Draws scale bars with physical units."""
    
    @staticmethod
    def draw(img_pil: Image.Image,
            pixel_size_um: float,
            target_length_um: float = 5.0) -> Image.Image:
        """Draw scale bar on PIL image.
        
        Args:
            img_pil: PIL image to draw on
            pixel_size_um: Pixel size in micrometers
            target_length_um: Target scale bar length in micrometers
            
        Returns:
            Image with scale bar drawn
        """
        draw = ImageDraw.Draw(img_pil)
        img_width, img_height = img_pil.size
        
        # Calculate bar length
        bar_length_px = int(target_length_um / pixel_size_um)
        
        # Adjust if too long
        max_bar_width = img_width * 0.25
        if bar_length_px > max_bar_width:
            bar_length_px = int(max_bar_width)
            target_length_um = bar_length_px * pixel_size_um
        
        bar_thickness = 3
        margin = 15
        
        # Position at bottom-right
        x1 = img_width - margin - bar_length_px
        y1 = img_height - margin - bar_thickness - 20
        x2 = img_width - margin
        y2 = y1 + bar_thickness
        
        # Draw bar with outline
        draw.rectangle([x1-1, y1-1, x2+1, y2+1], fill='black')
        draw.rectangle([x1, y1, x2, y2], fill='white')
        
        # Draw text label
        font_size = 12
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        text = f"{target_length_um:.1f} µm"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        text_x = x1 + (bar_length_px - text_w) // 2
        text_y = y2 + 3
        
        # Text with outline
        for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            draw.text((text_x+dx, text_y+dy), text, font=font, fill='black')
        draw.text((text_x, text_y), text, font=font, fill='white')
        
        return img_pil