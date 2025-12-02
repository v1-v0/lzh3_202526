#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Statistical analysis of segmented bacteria.
"""

import cv2
import numpy as np
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class BacteriaStatistics:
    """Calculates and manages bacteria statistics."""
    
    def __init__(self, pixel_size_um: float = 0.1289):
        """Initialize statistics manager.
        
        Args:
            pixel_size_um: Pixel size in micrometers
        """
        self.pixel_size_um = pixel_size_um
        self.pixel_area_um2 = pixel_size_um ** 2
    
    def calculate_all(self, 
                     contours: List[np.ndarray],
                     bf_img: np.ndarray,
                     fluor_img: Optional[np.ndarray] = None) -> List[Dict]:
        """Calculate statistics for all bacteria contours.
        
        Args:
            contours: List of bacteria contours
            bf_img: Bright-field image
            fluor_img: Fluorescence image (optional)
            
        Returns:
            List of statistics dictionaries
        """
        stats = []
        for idx, cnt in enumerate(contours):
            area_px = cv2.contourArea(cnt)
            area_um2 = area_px * self.pixel_area_um2
            
            mask = np.zeros(bf_img.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            
            f_mean = f_total = f_per = 0.0
            if fluor_img is not None:
                vals = fluor_img[mask == 255]
                if len(vals):
                    f_mean = float(np.mean(vals))
                    f_total = float(np.sum(vals))
                    f_per = f_total / area_px if area_px else 0.0
            
            stats.append({
                'orig_idx': idx,
                'bf_area_px': area_px,
                'bf_area_um2': area_um2,
                'fluor_mean': f_mean,
                'fluor_total': f_total,
                'fluor_per_area': f_per,
                'contour': cnt
            })
        
        return stats
    
    @staticmethod
    def sort_stats(stats: List[Dict], 
                   sort_key: str = "fluor_per_area",
                   descending: bool = True) -> List[Dict]:
        """Sort statistics by given key.
        
        Args:
            stats: List of statistics dictionaries
            sort_key: Key to sort by
            descending: Sort in descending order if True
            
        Returns:
            Sorted statistics list
        """
        if sort_key == 'index':
            return sorted(stats, key=lambda s: s.get('orig_idx', 0), 
                        reverse=descending)
        return sorted(stats, key=lambda s: s.get(sort_key, 0.0), 
                     reverse=descending)
    
    @staticmethod
    def filter_by_fluorescence(stats: List[Dict], 
                              min_fluor_per_area: float) -> List[Dict]:
        """Filter bacteria by minimum fluorescence/area ratio.
        
        Args:
            stats: List of statistics dictionaries
            min_fluor_per_area: Minimum fluorescence per area threshold
            
        Returns:
            Filtered statistics list
        """
        return [s for s in stats if s['fluor_per_area'] >= min_fluor_per_area]
    
    def export_to_csv(self,
                     stats: List[Dict],
                     filepath: Path,
                     image_name: str,
                     metadata: Optional[Dict] = None,
                     parameters: Optional[Dict] = None) -> None:
        """Export statistics to CSV file with metadata.
        
        Args:
            stats: List of statistics dictionaries
            filepath: Output CSV file path
            image_name: Name of analyzed image
            metadata: Image metadata dictionary
            parameters: Analysis parameters dictionary
        """
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(["# Bacteria Segmentation Analysis"])
            writer.writerow([f"# Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
            writer.writerow([f"# Image: {image_name}"])
            
            # Metadata
            if metadata:
                writer.writerow([f"# Sample: {metadata.get('sample_name', 'N/A')}"])
                writer.writerow([f"# Pixel Size: {self.pixel_size_um:.4f} µm"])
                writer.writerow([f"# Objective: {metadata.get('objective', 'N/A')}"])
                if 'acquired' in metadata:
                    writer.writerow([f"# Acquired: {metadata['acquired']}"])
                if 'exposure_times' in metadata:
                    exp = metadata['exposure_times']
                    writer.writerow([f"# BF Exposure: {exp.get('brightfield_ms', 'N/A')} ms"])
                    writer.writerow([f"# Fluor Exposure: {exp.get('fluorescence_ms', 'N/A')} ms"])
            else:
                writer.writerow([f"# ⚠️  No metadata available"])
                writer.writerow([f"# Pixel Size: {self.pixel_size_um:.4f} µm (default)"])
            
            # Parameters
            if parameters:
                writer.writerow(["# Analysis Parameters:"])
                for key, value in parameters.items():
                    writer.writerow([f"# {key}: {value}"])
            
            writer.writerow([])
            
            # Data
            writer.writerow([
                "Index", "BF Area (px²)", "BF Area (µm²)", 
                "Fluor Mean", "Fluor Total", "Fluor/Area"
            ])
            
            for i, s in enumerate(stats, 1):
                writer.writerow([
                    i,
                    f"{s['bf_area_px']:.1f}",
                    f"{s['bf_area_um2']:.3f}",
                    f"{s['fluor_mean']:.2f}",
                    f"{s['fluor_total']:.1f}",
                    f"{s['fluor_per_area']:.3f}"
                ])