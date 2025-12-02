#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Canvas management for image display.
"""

import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional, Tuple, List


class CanvasManager:
    """Manages image display on canvas widgets."""
    
    def __init__(self):
        """Initialize canvas manager."""
        self.canvas_images: dict = {}  # Store PhotoImage references
    
    def display_image(self, 
                     img: np.ndarray,
                     canvas: tk.Canvas,
                     scale_factor: float = 0.95,
                     convert_to_rgb: bool = True) -> None:
        """Display image on canvas with automatic scaling.
        
        Args:
            img: Image array (grayscale or BGR)
            canvas: Canvas widget
            scale_factor: Maximum scale relative to canvas size
            convert_to_rgb: Convert to RGB if True
        """
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        
        # Convert to RGB
        if convert_to_rgb:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Scale image
        h, w = img.shape[:2]
        scale = min(cw / w, ch / h) * scale_factor
        if scale < 1:
            nw, nh = int(w * scale), int(h * scale)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        
        # Create PhotoImage
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        
        # Display
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor=tk.CENTER)
        
        # Store reference
        canvas_id = str(id(canvas))
        self.canvas_images[canvas_id] = photo
    
    def display_fluorescence(self,
                           img: np.ndarray,
                           canvas: tk.Canvas,
                           brightness: float,
                           gamma: float) -> None:
        """Display fluorescence image with adjustments.
        
        Args:
            img: Fluorescence grayscale image
            canvas: Canvas widget
            brightness: Brightness multiplier
            gamma: Gamma correction value
        """
        from ..core.image_processing import ImageProcessor
        
        # Apply adjustments and create red channel display
        rgb = ImageProcessor.create_fluorescence_display(img, brightness, gamma)
        self.display_image(rgb, canvas, convert_to_rgb=False)
    
    def get_image_coordinates(self,
                             canvas: tk.Canvas,
                             event_x: int,
                             event_y: int,
                             img_shape: Tuple[int, int],
                             scale_factor: float = 0.95) -> Optional[Tuple[int, int]]:
        """Convert canvas coordinates to image coordinates.
        
        Args:
            canvas: Canvas widget
            event_x: X coordinate on canvas
            event_y: Y coordinate on canvas
            img_shape: Original image shape (height, width)
            scale_factor: Scale factor used for display
            
        Returns:
            Tuple of (x, y) image coordinates or None if out of bounds
        """
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return None
        
        h, w = img_shape
        scale = min(cw / w, ch / h) * scale_factor
        scale = min(scale, 1.0)
        
        # Calculate offset
        ox = (cw - int(w * scale)) // 2
        oy = (ch - int(h * scale)) // 2
        
        # Convert to image coordinates
        ix = int((event_x - ox) / scale)
        iy = int((event_y - oy) / scale)
        
        # Check bounds
        if 0 <= ix < w and 0 <= iy < h:
            return (ix, iy)
        
        return None
    
    def draw_crosshair(self,
                      canvas: tk.Canvas,
                      x: int,
                      y: int,
                      size: int = 12,
                      color: str = "red",
                      width: int = 3) -> List[int]:
        """Draw crosshair marker on canvas.
        
        Args:
            canvas: Canvas widget
            x: X coordinate
            y: Y coordinate
            size: Crosshair size
            color: Line color
            width: Line width
            
        Returns:
            List of canvas item IDs
        """
        ids = []
        ids.append(canvas.create_line(
            x - size, y, x + size, y, fill=color, width=width
        ))
        ids.append(canvas.create_line(
            x, y - size, x, y + size, fill=color, width=width
        ))
        return ids
    
    def clear_markers(self, canvas: tk.Canvas, item_ids: List[int]) -> None:
        """Clear markers from canvas.
        
        Args:
            canvas: Canvas widget
            item_ids: List of canvas item IDs to delete
        """
        for item_id in item_ids:
            canvas.delete(item_id)