# modules/file_handler.py
"""
File handling module for loading microscopy images and metadata
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
import glob

class FileHandler:
    """Handles loading of microscopy images and metadata"""
    
    def __init__(self, config):
        self.config = config
        self._clear_debug_dir()
    
    def _clear_debug_dir(self):
        """Clear debug directory before starting"""
        for f in glob.glob(os.path.join(self.config.DEBUG_DIR, "*")):
            try:
                os.remove(f)
            except OSError:
                pass
    
    def load_all(self, grey_path, red_path, meta_path):
        """
        Load all required files
        
        Returns:
            tuple: (img_bf, img_red, pixel_size, unit, bit_depth)
        """
        print("Loading metadata...")
        pixel_size_x, pixel_size_y, unit, bit_depth = self.parse_metadata(meta_path)
        
        # Calculate average pixel size
        if pixel_size_x and pixel_size_y:
            pixel_size = (pixel_size_x + pixel_size_y) / 2.0
            print(f"✓ Physical calibration: {pixel_size:.6f} {unit}/pixel")
        else:
            pixel_size = None
            unit = 'pixels'
            print("⚠ No physical calibration found")
        
        print("\nLoading brightfield image...")
        img_bf = self.load_image(grey_path)
        
        print("Loading fluorescence image...")
        img_red = self.load_image(red_path)
        
        # Detect bit depth from data if not in metadata
        if img_red.dtype == np.uint16:
            max_val = img_red.max()
            if max_val <= 4095:
                detected_bit_depth = 12
            elif max_val <= 16383:
                detected_bit_depth = 14
            else:
                detected_bit_depth = 16
            
            if bit_depth and bit_depth != detected_bit_depth:
                print(f"⚠ Metadata says {bit_depth}-bit, data suggests {detected_bit_depth}-bit")
                print(f"  Using detected: {detected_bit_depth}-bit")
            
            bit_depth = detected_bit_depth
        else:
            bit_depth = 8
        
        print(f"✓ Bit depth: {bit_depth}-bit")
        
        return img_bf, img_red, pixel_size, unit, bit_depth
    
    def load_image(self, path):
        """Load a single image file"""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        
        # Convert to grayscale if needed
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        print(f"  ✓ Loaded: {os.path.basename(path)}")
        print(f"    dtype={img.dtype}, shape={img.shape}, range=[{img.min()}, {img.max()}]")
        
        return img
    
    def parse_metadata(self, xml_path):
        """
        Parse metadata XML to extract physical pixel size and bit depth
        
        Returns:
            tuple: (pixel_size_x, pixel_size_y, unit, bit_depth)
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            pixel_size_x = None
            pixel_size_y = None
            unit = None
            bit_depth = None
            
            # Method 1: DimensionDescription tags (Leica LAS format)
            for dim in root.iter('DimensionDescription'):
                dim_id = dim.get('DimID')
                length = dim.get('Length')
                num_elements = dim.get('NumberOfElements')
                dim_unit = dim.get('Unit')
                
                if length and num_elements:
                    pixel_size = float(length) / float(num_elements)
                    
                    if dim_id == 'X':
                        pixel_size_x = pixel_size
                        if dim_unit:
                            unit = dim_unit
                    elif dim_id == 'Y':
                        pixel_size_y = pixel_size
                        if dim_unit and not unit:
                            unit = dim_unit
            
            # Method 2: ChannelDescription with OpticalResolutionXY
            if not pixel_size_x or not pixel_size_y:
                for channel in root.iter('ChannelDescription'):
                    optical_res = channel.get('OpticalResolutionXY')
                    if optical_res:
                        parts = optical_res.split()
                        if len(parts) >= 2:
                            pixel_size_x = pixel_size_y = float(parts[0])
                            unit = parts[1]
                    
                    # Get bit depth
                    if not bit_depth:
                        resolution = channel.get('Resolution')
                        if resolution:
                            bit_depth = int(resolution)
            
            # Normalize unit
            if unit in ['µm', 'μm']:
                unit = 'um'
            
            if pixel_size_x and pixel_size_y and not unit:
                unit = 'um'
            
            if pixel_size_x and pixel_size_y:
                print(f"  ✓ X: {pixel_size_x:.6f} {unit}/pixel")
                print(f"  ✓ Y: {pixel_size_y:.6f} {unit}/pixel")
                if bit_depth:
                    print(f"  ✓ Bit depth from metadata: {bit_depth}-bit")
            
            return pixel_size_x, pixel_size_y, unit, bit_depth
            
        except Exception as e:
            print(f"⚠ Error parsing metadata: {e}")
            return None, None, None, None