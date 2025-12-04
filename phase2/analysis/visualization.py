"""Visualization utilities for analysis results."""
import cv2
import numpy as np

class ResultVisualizer:
    """Handles visualization of analysis results."""
    
    def __init__(self):
        pass
    
    def visualize_results(self, image, results):
        """Visualize analysis results on image."""
        return image
    
    def draw_labels(self, image, labels):
        """Draw labels on image."""
        return image
    
    def draw_overlays(self, image, data):
        """Draw overlays on image."""
        return image
    
    @staticmethod
    def draw_segmentation(bf_image, contours):
        """Draw segmentation contours on brightfield image.
        
        Args:
            bf_image: Brightfield grayscale image
            contours: List of contours
            
        Returns:
            RGB image with contours drawn
        """
        import cv2
        result = cv2.cvtColor(bf_image, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        return result

    @staticmethod
    def create_overlay(bf_image, fluor_image, contours, stats, 
                    fluor_brightness=2.0, fluor_gamma=0.5,
                    show_labels=True, label_font_size=0.5,
                    arrow_length=30, label_offset=10,
                    highlight_index=None, show_scale_bar=True,
                    pixel_size_um=0.065):
        """Create overlay visualization.
        
        Args:
            bf_image: Brightfield image
            fluor_image: Fluorescence image (optional)
            contours: List of contours
            stats: List of statistics dictionaries
            fluor_brightness: Fluorescence brightness multiplier
            fluor_gamma: Fluorescence gamma correction
            show_labels: Whether to show labels
            label_font_size: Font size for labels
            arrow_length: Arrow length in pixels
            label_offset: Label offset from centroid
            highlight_index: Index of bacterium to highlight
            show_scale_bar: Whether to show scale bar
            pixel_size_um: Pixel size in micrometers
            
        Returns:
            RGB overlay image
        """
        from core.image_processing import ImageProcessor
        
        # Create base overlay
        overlay = cv2.cvtColor(bf_image, cv2.COLOR_GRAY2RGB)
        
        # Add fluorescence
        if fluor_image is not None:
            f8 = ImageProcessor.apply_fluorescence_adjustments(
                fluor_image, fluor_brightness, fluor_gamma
            )
            red_channel = overlay[:, :, 2].astype(np.float32)
            red_channel = np.clip(red_channel + f8.astype(np.float32), 0, 255)
            overlay[:, :, 2] = red_channel.astype(np.uint8)
        
        # Draw contours
        for i, contour in enumerate(contours):
            color = (255, 255, 0) if i != highlight_index else (0, 255, 255)
            thickness = 2 if i != highlight_index else 3
            cv2.drawContours(overlay, [contour], -1, color, thickness)
        
        # Draw labels
        if show_labels and stats:
            for stat in stats:
                i = stat['index'] - 1
                cx, cy = stat['centroid_x'], stat['centroid_y']
                
                # Draw arrow
                label_x = cx + label_offset + arrow_length
                label_y = cy - label_offset
                cv2.arrowedLine(overlay, (cx, cy), (label_x, label_y),
                            (255, 255, 255), 1, tipLength=0.3)
                
                # Draw label
                label = f"{stat['index']}"
                if stat['has_fluorescence']:
                    label += " +"
                
                cv2.putText(overlay, label, (label_x + 5, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, label_font_size,
                        (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw scale bar
        if show_scale_bar:
            bar_um = 10  # 10 micrometer bar
            bar_px = int(bar_um / pixel_size_um)
            h, w = overlay.shape[:2]
            x1, y1 = w - bar_px - 20, h - 30
            x2, y2 = w - 20, h - 30
            
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.putText(overlay, f"{bar_um} μm", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return overlay

