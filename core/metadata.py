#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metadata loading and handling for microscopy images.
"""

import json
from pathlib import Path
from typing import Optional, Dict


class MetadataManager:
    """Manages loading and accessing image metadata."""
    
    def __init__(self, metadata_dir: Optional[Path] = None):
        """Initialize metadata manager.
        
        Args:
            metadata_dir: Path to metadata JSON directory. Defaults to cwd/metadata_json.
        """
        self.metadata_dir = metadata_dir or (Path.cwd() / "metadata_json")
    
    def load_for_image(self, image_path: Path) -> Optional[Dict]:
        """Load metadata JSON for given image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Metadata dictionary or None if not found
        """
        if not self.metadata_dir.exists():
            print(f"  ℹ️  No metadata_json folder found")
            return None
        
        sample_name = image_path.stem.replace('_ch00', '').replace('_ch01', '')
        metadata_file = self.metadata_dir / f"{sample_name}.json"
        
        if not metadata_file.exists():
            print(f"  ℹ️  No metadata file: {metadata_file.name}")
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"  ✅ Metadata loaded: {metadata_file.name}")
            return metadata
        except Exception as e:
            print(f"  ⚠️  Failed to load metadata: {e}")
            return None
    
    @staticmethod
    def get_pixel_size(metadata: Optional[Dict], default: float = 0.1289) -> float:
        """Extract pixel size from metadata with fallback.
        
        Args:
            metadata: Metadata dictionary
            default: Default pixel size in µm
            
        Returns:
            Pixel size in µm
        """
        if metadata:
            return metadata.get('pixel_size_um', default)
        return default
    
    @staticmethod
    def get_display_settings(metadata: Optional[Dict]) -> Optional[Dict[str, float]]:
        """Extract fluorescence display settings from metadata.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Dictionary with 'brightness' and 'gamma' keys, or None
        """
        if not metadata or 'channels' not in metadata:
            return None
        
        channels = metadata['channels']
        if 'Red' not in channels:
            return None
        
        red_channel = channels['Red']
        black_val = red_channel['normalized']['BlackValue']
        white_val = red_channel['normalized']['WhiteValue']
        gamma_val = red_channel.get('GammaValue', 1.0)
        
        brightness = 1.0 / white_val if white_val > 0 else 2.0
        brightness = max(0.5, min(5.0, brightness))
        gamma = max(0.2, min(2.0, gamma_val))
        
        return {'brightness': brightness, 'gamma': gamma}