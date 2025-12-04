#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File and folder management utilities.
"""

from pathlib import Path
from typing import List, Optional


class ImageFileManager:
    """Manages image file scanning and validation."""
    
    @staticmethod
    def is_valid_brightfield_file(filepath: Path) -> bool:
        """Check if file is a valid bright-field image.
        
        Args:
            filepath: Path to file
            
        Returns:
            True if valid bright-field file
        """
        name = filepath.name
        return (
            not name.startswith('._') and
            not name.startswith('.') and
            name.endswith('_ch00.tif')
        )
    
    @staticmethod
    def get_fluorescence_path(bf_path: Path) -> Optional[Path]:
        """Get matching fluorescence image path for bright-field image.
        
        Args:
            bf_path: Path to bright-field image (_ch00.tif)
            
        Returns:
            Path to fluorescence image (_ch01.tif) or None
        """
        if not bf_path.name.endswith('_ch00.tif'):
            return None
        
        fluor_path = bf_path.parent / bf_path.name.replace('_ch00.tif', '_ch01.tif')
        
        if fluor_path.exists() and not fluor_path.name.startswith('._'):
            return fluor_path
        
        return None
    
    @staticmethod
    def scan_folder(folder_path: Path, 
                   recursive: bool = False) -> List[Path]:
        """Scan folder for valid bright-field images.
        
        Args:
            folder_path: Path to folder to scan
            recursive: Scan subfolders if True
            
        Returns:
            Sorted list of valid image paths
        """
        if not folder_path.exists() or not folder_path.is_dir():
            return []
        
        valid_files = []
        
        try:
            if recursive:
                files = folder_path.rglob('*_ch00.tif')
            else:
                files = folder_path.glob('*_ch00.tif')
            
            for f in files:
                if ImageFileManager.is_valid_brightfield_file(f):
                    valid_files.append(f)
        
        except PermissionError as e:
            print(f"❌ Permission denied: {e}")
            return []
        except Exception as e:
            print(f"❌ Error scanning folder: {e}")
            return []
        
        return sorted(valid_files, key=lambda p: p.name)
    
    @staticmethod
    def get_subfolders(folder_path: Path) -> List[Path]:
        """Get list of subfolders in given folder.
        
        Args:
            folder_path: Path to parent folder
            
        Returns:
            Sorted list of subfolder paths
        """
        if not folder_path.exists() or not folder_path.is_dir():
            return []
        
        try:
            subfolders = [
                item for item in folder_path.iterdir()
                if item.is_dir() and not item.name.startswith('.')
            ]
            return sorted(subfolders, key=lambda p: p.name.lower())
        except Exception as e:
            print(f"❌ Error getting subfolders: {e}")
            return []