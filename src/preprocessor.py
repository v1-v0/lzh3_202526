# 📦 **preprocessor.py v2.6 - WITH CLAHE FL ENHANCEMENT**

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import tifffile
import sys
import re
from datetime import datetime
import json
import shutil

# Import the validator class
try:
    from loader import DataLoader
except ImportError:
    print("CRITICAL ERROR: Could not import 'loader.py'.")
    sys.exit(1)

class PreprocessingModule:
    def __init__(self, debug_dir="debug"):
        # Parameters derived from "Metadata Microscopy Specifications"
        self.target_pixel_size = 0.1095  # μm/pixel (specification fallback)
        
        # DYNAMIC DIMENSION: Initialized to None, learned from first image
        self.target_dim = None   
        
        # DYNAMIC BIT DEPTH: Auto-detected from first image
        self.detected_bit_depth = None
        self.bit_depth_name = None
        
        # ============================================================
        # NORMALIZATION SETTINGS
        # ============================================================
        # Choose normalization method: "adaptive" or "percentile"
        self.normalization_method = "adaptive"  # Default: adaptive (uses actual min/max)
        
        # Percentile normalization parameters (only used if method = "percentile")
        self.percentile_low = 1.0    # Lower percentile for clipping
        self.percentile_high = 99.0  # Upper percentile for clipping
        
        # Noise reduction parameters
        self.gaussian_kernel = (5, 5)    # Kernel size for Gaussian blur
        self.sigma = 1.0                 # Standard deviation for Gaussian
        
        # ============================================================
        # CLAHE ENHANCEMENT SETTINGS (NEW IN v2.6)
        # ============================================================
        self.apply_clahe_to_fl = True         # Enable CLAHE for FL channel
        self.clahe_clip_limit = 2.0           # CLAHE clip limit (higher = more contrast)
        self.clahe_tile_size = (8, 8)         # CLAHE tile grid size
        
        # ============================================================
        # BACKGROUND CORRECTION - OPTIMIZED FOR MICROGELS
        # ============================================================
        self.bf_apply_background = False      # DISABLED for microgel analysis
        self.fl_apply_background = False      # DISABLED (no sensor black level needed)
        
        # Background correction parameters (kept for optional use)
        self.bf_gaussian_sigma = 15           # Sigma for background estimation
        self.bf_post_blur_kernel = (3, 3)     # Post-enhancement smoothing kernel
        self.fl_black_level_8bit = 7          # Sensor black level (if enabled)
        self.fl_black_level_12bit = 28        # Sensor black level for 12-bit (if enabled)
        
        # Morphological operations parameters
        self.morph_kernel_size = (3, 3)       # Kernel for morphological ops
        self.morph_close_iterations = 1       # Closing iterations
        self.morph_dilate_iterations = 1      # Dilation iterations
        self.morph_erode_iterations = 1       # Erosion iterations
        
        # DEBUG LOGGING SETUP
        self.debug_dir = debug_dir
        self.debug_enabled = True
        self._setup_debug_logging()

    def _setup_debug_logging(self):
        """Creates debug directory and initializes logging. Clears old debug sessions."""
        if not self.debug_enabled:
            return
        
        if os.path.exists(self.debug_dir):
            try:
                shutil.rmtree(self.debug_dir)
                print(f"[i] Cleared existing debug folder: {self.debug_dir}")
            except Exception as e:
                print(f"[!] Warning: Could not clear debug folder: {e}")
                print(f"[i] Continuing with existing debug folder...")
        
        os.makedirs(self.debug_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.debug_dir, f"session_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        self.log_dir = os.path.join(self.session_dir, "logs")
        self.img_dir = os.path.join(self.session_dir, "images")
        self.stats_dir = os.path.join(self.session_dir, "statistics")
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.log_dir, "preprocessing.log")
        
        self.session_stats = {
            "session_start": timestamp,
            "pipeline_version": "2.6",
            "optimization": "microgel_segmentation_clahe_enhancement",
            "normalization_method": self.normalization_method,
            "clahe_enabled": self.apply_clahe_to_fl,
            "images_processed": [],
            "errors": [],
            "warnings": []
        }
        
        self._log(f"{'='*60}")
        self._log(f"DEBUG SESSION STARTED: {timestamp}")
        self._log(f"Session directory: {self.session_dir}")
        self._log(f"Pipeline optimization: MICROGEL SEGMENTATION + CLAHE")
        self._log(f"Normalization method: {self.normalization_method.upper()}")
        self._log(f"CLAHE Enhancement: {'ENABLED' if self.apply_clahe_to_fl else 'DISABLED'}")
        if self.apply_clahe_to_fl:
            self._log(f"  - Clip Limit: {self.clahe_clip_limit}")
            self._log(f"  - Tile Size: {self.clahe_tile_size}")
        self._log(f"{'='*60}\n")

    def _log(self, message):
        """Writes message to both console and log file."""
        print(message)
        if self.debug_enabled:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')

    def _log_image_stats(self, image, name, sample_id, stage):
        """Logs detailed image statistics."""
        if not self.debug_enabled:
            return
        
        # Handle RGB images
        if len(image.shape) == 3:
            image_for_stats = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_for_stats = image
        
        stats = {
            "sample_id": sample_id,
            "image_name": name,
            "stage": stage,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "min": float(np.min(image_for_stats)),
            "max": float(np.max(image_for_stats)),
            "mean": float(np.mean(image_for_stats)),
            "median": float(np.median(image_for_stats)),
            "std": float(np.std(image_for_stats)),
            "non_zero_pixels": int(np.count_nonzero(image_for_stats)),
            "total_pixels": int(image_for_stats.size)
        }
        
        stats["percentile_1"] = float(np.percentile(image_for_stats, 1))
        stats["percentile_25"] = float(np.percentile(image_for_stats, 25))
        stats["percentile_75"] = float(np.percentile(image_for_stats, 75))
        stats["percentile_99"] = float(np.percentile(image_for_stats, 99))
        
        stats_file = os.path.join(
            self.stats_dir, 
            f"{sample_id}_{name}_{stage}_stats.json"
        )
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats

    def _save_debug_image(self, image, name, sample_id, stage):
        """Saves image to debug directory for inspection."""
        if not self.debug_enabled:
            return
        
        # Handle RGB images (don't normalize)
        if len(image.shape) == 3:
            img_to_save = image.copy()
        else:
            # Normalize grayscale to 8-bit for visualization
            if image.dtype != np.uint8:
                img_to_save = self.normalize_to_8bit(image.copy())
            else:
                img_to_save = image.copy()
        
        filename = f"{sample_id}_{name}_{stage}.png"
        filepath = os.path.join(self.img_dir, filename)
        cv2.imwrite(filepath, img_to_save)
        
        return filepath

    def _log_conversion_details(self, sample_id, image_type, original_shape, 
                                original_dtype, final_shape, final_dtype, 
                                conversion_steps):
        """Logs detailed conversion pipeline for each image."""
        if not self.debug_enabled:
            return
        
        details = {
            "sample_id": sample_id,
            "image_type": image_type,
            "original": {
                "shape": list(original_shape),
                "dtype": str(original_dtype)
            },
            "final": {
                "shape": list(final_shape),
                "dtype": str(final_dtype)
            },
            "conversion_steps": conversion_steps
        }
        
        conv_file = os.path.join(
            self.log_dir,
            f"{sample_id}_{image_type}_conversion.json"
        )
        with open(conv_file, 'w') as f:
            json.dump(details, f, indent=2)

    def parse_pixel_size(self, xml_path):
        """Parses XML metadata to extract calibrated pixel size."""
        try:
            if not os.path.exists(xml_path):
                self._log(f"   [!] XML file not found: {os.path.basename(xml_path)}")
                return self.target_pixel_size
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Strategy 1: DimensionDescription
            for dim_desc in root.findall(".//DimensionDescription[@DimID='X']"):
                voxel = dim_desc.get('Voxel')
                if voxel:
                    pixel_size = float(voxel)
                    self._log(f"   [i] Extracted pixel size from XML (DimensionDescription): {pixel_size} μm/pixel")
                    return pixel_size
            
            # Strategy 2: PhysicalSizeX
            for pixels in root.findall(".//Pixels"):
                phys_size = pixels.get('PhysicalSizeX')
                if phys_size:
                    pixel_size = float(phys_size)
                    self._log(f"   [i] Extracted pixel size from XML (PhysicalSizeX): {pixel_size} μm/pixel")
                    return pixel_size
            
            # Strategy 3: Distance/Value
            for distance in root.findall(".//Distance[@Id='X']"):
                value_elem = distance.find('Value')
                if value_elem is not None and value_elem.text:
                    pixel_size = float(value_elem.text)
                    self._log(f"   [i] Extracted pixel size from XML (Distance/Value): {pixel_size} μm/pixel")
                    return pixel_size
            
            # Strategy 4: Element iteration
            for elem in root.iter():
                tag_lower = elem.tag.lower()
                if any(keyword in tag_lower for keyword in ['resolution', 'scalex', 'calibration', 'pixelsize']):
                    if elem.text:
                        try:
                            pixel_size = float(elem.text)
                            if 0.01 <= pixel_size <= 1.0:
                                self._log(f"   [i] Extracted pixel size from XML ({elem.tag}): {pixel_size} μm/pixel")
                                return pixel_size
                        except ValueError:
                            continue
                    
                    for attr_name, attr_value in elem.attrib.items():
                        attr_lower = attr_name.lower()
                        if any(keyword in attr_lower for keyword in ['voxel', 'physical', 'size', 'resolution', 'scale']):
                            try:
                                pixel_size = float(attr_value)
                                if 0.01 <= pixel_size <= 1.0:
                                    self._log(f"   [i] Extracted pixel size from XML ({elem.tag}.{attr_name}): {pixel_size} μm/pixel")
                                    return pixel_size
                            except ValueError:
                                continue
            
            # Strategy 5: Regex search
            xml_string = ET.tostring(root, encoding='unicode')
            patterns = [
                r'Voxel\s*=\s*["\']?([\d.]+)["\']?',
                r'PhysicalSize[XY]?\s*=\s*["\']?([\d.]+)["\']?',
                r'Resolution\s*=\s*["\']?([\d.]+)["\']?',
                r'ScaleX\s*=\s*["\']?([\d.]+)["\']?',
                r'CalibrationX?\s*=\s*["\']?([\d.]+)["\']?',
                r'PixelSize[XY]?\s*=\s*["\']?([\d.]+)["\']?',
                r'<Value>([\d.]+)</Value>',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, xml_string, re.IGNORECASE)
                if match:
                    try:
                        pixel_size = float(match.group(1))
                        if 0.01 <= pixel_size <= 1.0:
                            self._log(f"   [i] Extracted pixel size from XML (regex): {pixel_size} μm/pixel")
                            return pixel_size
                    except (ValueError, IndexError):
                        continue
            
            self._log(f"   [i] XML missing calibration, using default: {self.target_pixel_size} μm/pixel")
            return self.target_pixel_size
            
        except ET.ParseError as e:
            self._log(f"   [!] XML Parse Error: {e}")
            return self.target_pixel_size
        except Exception as e:
            self._log(f"   [!] Metadata Error: {e}")
            return self.target_pixel_size

    def convert_to_grayscale(self, image, sample_id, image_type):
        """Converts RGB/multi-channel images to grayscale."""
        conversion_steps = []
        original_shape = image.shape
        original_dtype = image.dtype
        
        if len(image.shape) == 2:
            conversion_steps.append("Already 2D grayscale - no conversion needed")
            self._log(f"   [i] {image_type}: Already grayscale {image.shape}")
            result = image
        
        elif len(image.shape) == 3:
            channels = image.shape[2]
            
            if channels == 3:
                conversion_steps.append(f"Detected RGB image (3 channels)")
                
                if image.dtype == np.uint8:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    conversion_steps.append("Applied cv2.cvtColor(RGB2GRAY) for uint8")
                else:
                    gray = (0.299 * image[:, :, 0] + 
                           0.587 * image[:, :, 1] + 
                           0.114 * image[:, :, 2])
                    gray = gray.astype(image.dtype)
                    conversion_steps.append(f"Applied weighted RGB→Gray formula, preserved {image.dtype}")
                
                self._log(f"   [i] {image_type}: Converted RGB (3-channel) to grayscale")
                result = gray
            
            elif channels == 4:
                conversion_steps.append(f"Detected RGBA image (4 channels)")
                rgb_only = image[:, :, :3]
                
                if image.dtype == np.uint8:
                    gray = cv2.cvtColor(rgb_only, cv2.COLOR_RGB2GRAY)
                    conversion_steps.append("Ignored alpha, applied cv2.cvtColor(RGB2GRAY)")
                else:
                    gray = (0.299 * rgb_only[:, :, 0] + 
                           0.587 * rgb_only[:, :, 1] + 
                           0.114 * rgb_only[:, :, 2])
                    gray = gray.astype(image.dtype)
                    conversion_steps.append(f"Ignored alpha, applied weighted formula")
                
                self._log(f"   [i] {image_type}: Converted RGBA (4-channel) to grayscale")
                result = gray
            
            else:
                conversion_steps.append(f"Warning: Unexpected {channels}-channel image, extracted first channel")
                self._log(f"   [!] {image_type}: Warning: Unexpected {channels}-channel image, using first channel")
                result = image[:, :, 0]
        
        else:
            error_msg = f"Unexpected image shape: {image.shape}"
            conversion_steps.append(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        self._log_conversion_details(
            sample_id, image_type,
            original_shape, original_dtype,
            result.shape, result.dtype,
            conversion_steps
        )
        
        return result

    def detect_bit_depth(self, image):
        """Auto-detects bit depth of input image."""
        dtype = image.dtype
        max_val = np.max(image)
        
        if dtype == np.uint8:
            return 8, "8-bit", 255
        elif dtype == np.uint16:
            if max_val <= 4095:
                return 12, "12-bit (stored as uint16)", 4095
            else:
                return 16, "16-bit", 65535
        elif dtype == np.uint32:
            return 32, "32-bit", 4294967295
        elif dtype == np.float32 or dtype == np.float64:
            if max_val <= 1.0:
                return None, "float (normalized)", 1.0
            else:
                return None, f"float (raw, max={max_val:.2f})", max_val
        else:
            return None, f"unknown ({dtype})", max_val

    def normalize_to_8bit(self, image):
        """
        Converts any bit-depth image to 8-bit using ADAPTIVE scaling.
        
        Uses actual image range instead of theoretical max.
        This prevents over-darkening of underexposed microscopy images.
        """
        # Detect bit depth (for logging)
        bit_depth, bit_name, theoretical_max = self.detect_bit_depth(image)
        
        # Get ACTUAL image intensity range
        img_min = float(np.min(image))
        img_max = float(np.max(image))
        
        # Case 1: Image has no contrast (all pixels same value)
        if img_max == img_min:
            self._log(f"   [!] Warning: Image has no contrast (uniform intensity = {img_min})")
            return np.zeros_like(image, dtype=np.uint8)
        
        # Case 2: ADAPTIVE NORMALIZATION
        img_float = image.astype(np.float64)
        normalized = (img_float - img_min) / (img_max - img_min)
        scaled = normalized * 255.0
        result = np.clip(scaled, 0, 255).astype(np.uint8)
        
        # Verification
        result_min = int(np.min(result))
        result_max = int(np.max(result))
        result_mean = float(np.mean(result))
        
        # Log transformation
        self._log(f"   [DEBUG] Adaptive Normalization: {bit_name}")
        self._log(f"           Input range:  [{img_min:.2f}, {img_max:.2f}] (actual)")
        self._log(f"           Theoretical max: {theoretical_max}")
        self._log(f"           Output range: [{result_min}, {result_max}]")
        self._log(f"           Output mean:  {result_mean:.2f}")
        
        return result

    def normalize_to_8bit_percentile(self, image, lower_percentile=None, upper_percentile=None):
        """
        Converts to 8-bit using percentile-based normalization.
        """
        if lower_percentile is None:
            lower_percentile = self.percentile_low
        if upper_percentile is None:
            upper_percentile = self.percentile_high
        
        bit_depth, bit_name, theoretical_max = self.detect_bit_depth(image)
        
        p_low = np.percentile(image, lower_percentile)
        p_high = np.percentile(image, upper_percentile)
        
        self._log(f"   [DEBUG] Percentile Normalization: {bit_name}")
        self._log(f"           P{lower_percentile}={p_low:.2f}, P{upper_percentile}={p_high:.2f}")
        
        if p_high <= p_low:
            self._log(f"   [!] Warning: Invalid percentile range, using full range")
            return self.normalize_to_8bit(image)
        
        img_clipped = np.clip(image, p_low, p_high)
        img_float = img_clipped.astype(np.float64)
        normalized = (img_float - p_low) / (p_high - p_low)
        scaled = normalized * 255.0
        result = np.clip(scaled, 0, 255).astype(np.uint8)
        
        result_min = int(np.min(result))
        result_max = int(np.max(result))
        result_mean = float(np.mean(result))
        
        self._log(f"           Output range: [{result_min}, {result_max}]")
        self._log(f"           Output mean:  {result_mean:.2f}")
        
        return result

    def apply_noise_reduction(self, image_8bit):
        """Applies Gaussian blur for noise reduction."""
        return cv2.GaussianBlur(image_8bit, self.gaussian_kernel, self.sigma)

    def enhance_fl_clahe(self, fl_8bit):
        """
        Enhances fluorescence contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        NEW IN v2.6 - Optimized for sparse fluorescence imaging.
        
        CLAHE works by:
        1. Dividing image into tiles (e.g., 8×8 grid)
        2. Applying histogram equalization to each tile
        3. Limiting contrast amplification to avoid noise amplification
        
        Args:
            fl_8bit (np.ndarray): 8-bit normalized FL image
            
        Returns:
            np.ndarray: CLAHE-enhanced FL image (uint8)
        """
        if not self.apply_clahe_to_fl:
            return fl_8bit
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, 
            tileGridSize=self.clahe_tile_size
        )
        
        # Apply CLAHE
        enhanced = clahe.apply(fl_8bit)
        
        # Log enhancement statistics
        original_mean = float(np.mean(fl_8bit))
        original_std = float(np.std(fl_8bit))
        enhanced_mean = float(np.mean(enhanced))
        enhanced_std = float(np.std(enhanced))
        
        self._log(f"   [i] CLAHE Enhancement Applied:")
        self._log(f"       Original: mean={original_mean:.2f}, std={original_std:.2f}")
        self._log(f"       Enhanced: mean={enhanced_mean:.2f}, std={enhanced_std:.2f}")
        self._log(f"       Contrast boost: {enhanced_std/original_std:.2f}x")
        
        return enhanced

    def correct_bf_background(self, bf_image_8bit):
        """Corrects uneven illumination in bright-field images."""
        bg_estimate = cv2.GaussianBlur(
            bf_image_8bit, 
            (0, 0),
            sigmaX=self.bf_gaussian_sigma, 
            sigmaY=self.bf_gaussian_sigma
        )
        
        corrected = cv2.subtract(bg_estimate, bf_image_8bit)
        corrected = cv2.GaussianBlur(corrected, self.bf_post_blur_kernel, 0)
        
        return corrected

    def correct_fl_background(self, fl_image_8bit, original_bit_depth):
        """Corrects sensor black level in fluorescence images."""
        if not self.fl_apply_background:
            return fl_image_8bit
        
        if original_bit_depth == 12:
            black_level_8bit = int(self.fl_black_level_12bit * 255 / 4095)
        else:
            black_level_8bit = self.fl_black_level_8bit
        
        corrected = np.clip(
            fl_image_8bit.astype(np.int16) - black_level_8bit, 
            0, 
            255
        ).astype(np.uint8)
        
        return corrected

    def apply_morphological_cleanup(self, binary_mask):
        """Applies morphological operations to clean up binary mask."""
        kernel = np.ones(self.morph_kernel_size, np.uint8)
        
        cleaned = cv2.morphologyEx(
            binary_mask, 
            cv2.MORPH_CLOSE, 
            kernel, 
            iterations=self.morph_close_iterations
        )
        
        cleaned = cv2.dilate(
            cleaned, 
            kernel, 
            iterations=self.morph_dilate_iterations
        )
        
        cleaned = cv2.erode(
            cleaned, 
            kernel, 
            iterations=self.morph_erode_iterations
        )
        
        return cleaned

    def create_red_fluorescence_visualization(self, fl_gray_8bit):
        """
        Creates a red-colored visualization of fluorescence channel.
        
        Args:
            fl_gray_8bit (np.ndarray): 8-bit grayscale FL image
            
        Returns:
            np.ndarray: RGB image with FL in red channel (H, W, 3) - BGR format
        """
        rgb_visual = np.zeros((fl_gray_8bit.shape[0], fl_gray_8bit.shape[1], 3), dtype=np.uint8)
        rgb_visual[:, :, 2] = fl_gray_8bit  # Red channel (BGR format)
        
        return rgb_visual

    def validate_image_quality(self, image, image_type, sample_id):
        """Performs basic quality checks on loaded images."""
        warnings = []
        
        if image.size == 0:
            return False, ["Image is empty (0 pixels)"]
        
        if len(image.shape) != 2:
            warnings.append(f"Image is not 2D after conversion (shape: {image.shape})")
        
        bit_depth, bit_name, max_value = self.detect_bit_depth(image)
        saturated_pixels = np.sum(image >= max_value * 0.99)
        saturation_percent = (saturated_pixels / image.size) * 100
        
        if saturation_percent > 1.0:
            warnings.append(f"High saturation: {saturation_percent:.2f}% of pixels at max value")
        
        mean_intensity = np.mean(image)
        if mean_intensity < max_value * 0.01:
            warnings.append(f"Very dark image: mean intensity = {mean_intensity:.2f} (max = {max_value})")
        
        std_intensity = np.std(image)
        if std_intensity < 1.0:
            warnings.append(f"Very low contrast: std = {std_intensity:.2f}")
        
        if warnings:
            for warning in warnings:
                self._log(f"   [⚠] QC Warning ({image_type}): {warning}")
        
        return True, warnings

    def load_and_preprocess(self, file_registry):
        """
        Main preprocessing pipeline for microscopy images.
        
        OPTIMIZED FOR MICROGEL SEGMENTATION + CLAHE ENHANCEMENT (v2.6)
        
        Pipeline:
        1. Parse XML metadata (pixel size)
        2. Load raw TIFF images
        3. Convert to grayscale (if RGB)
        4. Quality Control: Validate dimensions, bit depth, image quality
        5. Normalize to 8-bit (adaptive or percentile method)
        6. Enhancement: Light noise reduction + CLAHE for FL
        7. Create red FL visualization
        8. Output: Both processed 8-bit and raw original
        """
        processed_batch = []
        total_warnings = 0

        self._log(f"\n{'='*60}")
        self._log(f"  PREPROCESSING PIPELINE v2.6 - Processing {len(file_registry)} image sets")
        self._log(f"{'='*60}")
        self._log(f"  OPTIMIZATION: Microgel Segmentation + CLAHE")
        self._log(f"  Normalization: {self.normalization_method.upper()}")
        if self.normalization_method == "percentile":
            self._log(f"  Percentiles: P{self.percentile_low}-P{self.percentile_high}")
        self._log(f"  BF Background Correction: {'ENABLED' if self.bf_apply_background else 'DISABLED (preserves microgel structures)'}")
        self._log(f"  FL Background Correction: {'ENABLED' if self.fl_apply_background else 'DISABLED'}")
        self._log(f"  FL CLAHE Enhancement: {'ENABLED' if self.apply_clahe_to_fl else 'DISABLED'}")
        if self.apply_clahe_to_fl:
            self._log(f"    - Clip Limit: {self.clahe_clip_limit}")
            self._log(f"    - Tile Size: {self.clahe_tile_size}")
        self._log(f"  FL Visualization: RED channel (RGB output)")
        self._log(f"{'='*60}\n")

        for idx, entry in enumerate(file_registry, 1):
            sample_id = entry['id']
            self._log(f"[{idx}/{len(file_registry)}] Processing: {sample_id}")
            
            qc_warnings = []
            
            # ============================================================
            # STEP 1: Parse XML Metadata
            # ============================================================
            pixel_size = self.parse_pixel_size(entry['xml'])

            # ============================================================
            # STEP 2: Load Raw Images
            # ============================================================
            try:
                img_bf_raw = tifffile.imread(entry['bf'])
                img_fl_raw = tifffile.imread(entry['fl'])
                
                self._log(f"   [DEBUG] Loaded BF raw: shape={img_bf_raw.shape}, dtype={img_bf_raw.dtype}")
                self._log(f"   [DEBUG] Loaded FL raw: shape={img_fl_raw.shape}, dtype={img_fl_raw.dtype}")
                
                if self.debug_enabled:
                    self._save_debug_image(img_bf_raw[:,:,0] if len(img_bf_raw.shape)==3 else img_bf_raw, 
                                          "BF", sample_id, "01_raw")
                    self._save_debug_image(img_fl_raw[:,:,0] if len(img_fl_raw.shape)==3 else img_fl_raw, 
                                          "FL", sample_id, "01_raw")
                    self._log_image_stats(img_bf_raw[:,:,0] if len(img_bf_raw.shape)==3 else img_bf_raw, 
                                         "BF", sample_id, "01_raw")
                    self._log_image_stats(img_fl_raw[:,:,0] if len(img_fl_raw.shape)==3 else img_fl_raw, 
                                         "FL", sample_id, "01_raw")
                
            except Exception as e:
                error_msg = f"Read Error: {e}"
                self._log(f"   [!] {error_msg}")
                self._log(f"   [!] Skipping {sample_id}\n")
                self.session_stats["errors"].append({
                    "sample_id": sample_id,
                    "error": error_msg,
                    "stage": "load_raw"
                })
                continue

            # ============================================================
            # STEP 3: Convert to Grayscale
            # ============================================================
            try:
                img_bf_gray = self.convert_to_grayscale(img_bf_raw, sample_id, "BF")
                img_fl_gray = self.convert_to_grayscale(img_fl_raw, sample_id, "FL")
                
                self._log(f"   [DEBUG] BF grayscale: shape={img_bf_gray.shape}, dtype={img_bf_gray.dtype}")
                self._log(f"   [DEBUG] FL grayscale: shape={img_fl_gray.shape}, dtype={img_fl_gray.dtype}")
                
                if self.debug_enabled:
                    self._save_debug_image(img_bf_gray, "BF", sample_id, "02_grayscale")
                    self._save_debug_image(img_fl_gray, "FL", sample_id, "02_grayscale")
                    self._log_image_stats(img_bf_gray, "BF", sample_id, "02_grayscale")
                    self._log_image_stats(img_fl_gray, "FL", sample_id, "02_grayscale")
                
            except Exception as e:
                error_msg = f"Grayscale conversion error: {e}"
                self._log(f"   [!] {error_msg}")
                self._log(f"   [!] Skipping {sample_id}\n")
                self.session_stats["errors"].append({
                    "sample_id": sample_id,
                    "error": error_msg,
                    "stage": "grayscale_conversion"
                })
                continue

            # ============================================================
            # STEP 4: Quality Control
            # ============================================================
            bf_depth, bf_name, bf_max = self.detect_bit_depth(img_bf_gray)
            fl_depth, fl_name, fl_max = self.detect_bit_depth(img_fl_gray)
            
            if self.detected_bit_depth is None:
                self.detected_bit_depth = bf_depth
                self.bit_depth_name = bf_name
                self._log(f"   [i] Auto-detected bit depth: {bf_name} (max value: {bf_max})")
            
            if bf_depth != fl_depth:
                error_msg = f"Bit depth mismatch: BF: {bf_name}, FL: {fl_name}"
                self._log(f"   [!] {error_msg}")
                self._log(f"   [!] Skipping {sample_id}\n")
                self.session_stats["errors"].append({
                    "sample_id": sample_id,
                    "error": error_msg,
                    "stage": "bit_depth_validation"
                })
                continue
            
            if bf_depth != self.detected_bit_depth:
                warning_msg = f"Bit depth changed from {self.bit_depth_name} to {bf_name}"
                self._log(f"   [!] Warning: {warning_msg}")
                qc_warnings.append(warning_msg)

            if self.target_dim is None:
                self.target_dim = img_bf_gray.shape
                self._log(f"   [i] Auto-detected target dimensions: {self.target_dim}")
            
            if img_bf_gray.shape != self.target_dim:
                error_msg = f"Dimension mismatch: Expected {self.target_dim}, Got {img_bf_gray.shape}"
                self._log(f"   [!] {error_msg}")
                self._log(f"   [!] Skipping {sample_id}\n")
                self.session_stats["errors"].append({
                    "sample_id": sample_id,
                    "error": error_msg,
                    "stage": "dimension_validation"
                })
                continue
            
            if img_bf_gray.shape != img_fl_gray.shape:
                error_msg = f"Channel dimension mismatch: BF {img_bf_gray.shape}, FL {img_fl_gray.shape}"
                self._log(f"   [!] {error_msg}")
                self._log(f"   [!] Skipping {sample_id}\n")
                self.session_stats["errors"].append({
                    "sample_id": sample_id,
                    "error": error_msg,
                    "stage": "dimension_validation"
                })
                continue
            
            bf_valid, bf_warnings = self.validate_image_quality(img_bf_gray, "BF", sample_id)
            fl_valid, fl_warnings = self.validate_image_quality(img_fl_gray, "FL", sample_id)
            
            qc_warnings.extend(bf_warnings)
            qc_warnings.extend(fl_warnings)
            
            if qc_warnings:
                total_warnings += len(qc_warnings)

            # ============================================================
            # STEP 5: Normalize to 8-bit (ADAPTIVE or PERCENTILE)
            # ============================================================
            if self.normalization_method == "percentile":
                self._log(f"   [i] Using PERCENTILE normalization (P{self.percentile_low}-P{self.percentile_high})")
                bf_8bit = self.normalize_to_8bit_percentile(img_bf_gray)
                fl_8bit = self.normalize_to_8bit_percentile(img_fl_gray)
            else:
                self._log(f"   [i] Using ADAPTIVE normalization (actual min/max)")
                bf_8bit = self.normalize_to_8bit(img_bf_gray)
                fl_8bit = self.normalize_to_8bit(img_fl_gray)
            
            self._log(f"   [DEBUG] BF 8-bit: shape={bf_8bit.shape}, dtype={bf_8bit.dtype}, range=[{bf_8bit.min()},{bf_8bit.max()}]")
            self._log(f"   [DEBUG] FL 8-bit: shape={fl_8bit.shape}, dtype={fl_8bit.dtype}, range=[{fl_8bit.min()},{fl_8bit.max()}]")
            
            if self.debug_enabled:
                self._save_debug_image(bf_8bit, "BF", sample_id, "03_8bit")
                self._save_debug_image(fl_8bit, "FL", sample_id, "03_8bit")
                self._log_image_stats(bf_8bit, "BF", sample_id, "03_8bit")
                self._log_image_stats(fl_8bit, "FL", sample_id, "03_8bit")

            # ============================================================
            # STEP 6: Enhancement - OPTIMIZED FOR MICROGELS
            # ============================================================
            bf_denoised = self.apply_noise_reduction(bf_8bit)
            fl_denoised = self.apply_noise_reduction(fl_8bit)
            
            if self.debug_enabled:
                self._save_debug_image(bf_denoised, "BF", sample_id, "04_denoised")
                self._save_debug_image(fl_denoised, "FL", sample_id, "04_denoised")
                self._log_image_stats(bf_denoised, "BF", sample_id, "04_denoised")
                self._log_image_stats(fl_denoised, "FL", sample_id, "04_denoised")

            if self.bf_apply_background:
                bf_final = self.correct_bf_background(bf_denoised)
                self._log(f"   [i] Applied BF background correction")
            else:
                bf_final = bf_denoised
                self._log(f"   [i] Skipped BF background correction (preserves microgel structures)")

            if self.fl_apply_background:
                fl_corrected = self.correct_fl_background(fl_denoised, bf_depth)
                self._log(f"   [i] Applied FL background correction")
            else:
                fl_corrected = fl_denoised
                self._log(f"   [i] Skipped FL background correction (disabled)")
            
            # ============================================================
            # STEP 6B: CLAHE ENHANCEMENT FOR FL (NEW IN v2.6)
            # ============================================================
            fl_final = self.enhance_fl_clahe(fl_corrected)
            
            if self.debug_enabled:
                self._save_debug_image(bf_final, "BF", sample_id, "05_final")
                self._save_debug_image(fl_final, "FL", sample_id, "05_final_clahe")
                self._log_image_stats(bf_final, "BF", sample_id, "05_final")
                self._log_image_stats(fl_final, "FL", sample_id, "05_final_clahe")

            # ============================================================
            # STEP 7: Create Red FL Visualization + Register Output
            # ============================================================
            fl_red_visual = self.create_red_fluorescence_visualization(fl_final)
            self._log(f"   [i] Created red FL visualization (RGB)")
            
            if self.debug_enabled:
                self._save_debug_image(fl_red_visual, "FL", sample_id, "06_red_visual")
                self._log_image_stats(fl_red_visual, "FL", sample_id, "06_red_visual")

            processed_data = {
                "id": sample_id,
                "group": entry.get('group', 'unknown'),
                "pixel_size_um": pixel_size,
                "image_dimensions": self.target_dim,
                "bit_depth": bf_name,
                
                # Processed data (8-bit GRAYSCALE) - USE FOR SEGMENTATION
                "bf_img_8bit": bf_final,
                "fl_img_8bit": fl_final,  # CLAHE-enhanced
                
                # Visualization (RGB) - USE FOR DISPLAY/FIGURES
                "fl_red_visual": fl_red_visual,
                
                # Original raw data (grayscale) - USE FOR INTENSITY MEASUREMENTS
                "raw_bf": img_bf_gray,
                "raw_fl": img_fl_gray,
                
                # Quality Control
                "qc_warnings": qc_warnings
            }
            
            processed_batch.append(processed_data)
            
            self.session_stats["images_processed"].append({
                "sample_id": sample_id,
                "group": entry.get('group', 'unknown'),
                "pixel_size": pixel_size,
                "dimensions": list(self.target_dim),
                "bit_depth": bf_name,
                "warnings_count": len(qc_warnings)
            })
            
            self._log(f"   [✓] Successfully processed {sample_id}\n")

        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        self._log(f"{'='*60}")
        self._log(f"  PREPROCESSING COMPLETE")
        self._log(f"{'='*60}")
        self._log(f"  Detected bit depth: {self.bit_depth_name}")
        self._log(f"  Target dimensions: {self.target_dim}")
        self._log(f"  Total processed: {len(processed_batch)}/{len(file_registry)} image sets")
        self._log(f"  Failed: {len(file_registry) - len(processed_batch)} image sets")
        self._log(f"  QC warnings: {total_warnings}")
        self._log(f"  Normalization: {self.normalization_method.upper()}")
        if self.normalization_method == "percentile":
            self._log(f"  Percentiles: P{self.percentile_low}-P{self.percentile_high}")
        self._log(f"  BF Background: {'ENABLED' if self.bf_apply_background else 'DISABLED (microgel optimization)'}")
        self._log(f"  FL Background: {'ENABLED' if self.fl_apply_background else 'DISABLED'}")
        self._log(f"  FL CLAHE: {'ENABLED' if self.apply_clahe_to_fl else 'DISABLED'}")
        self._log(f"  FL Visualization: RED channel (RGB)")
        self._log(f"{'='*60}\n")
        
        self.session_stats["session_end"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_stats["total_processed"] = len(processed_batch)
        self.session_stats["total_failed"] = len(file_registry) - len(processed_batch)
        self.session_stats["total_warnings"] = total_warnings
        
        stats_summary_file = os.path.join(self.session_dir, "session_summary.json")
        with open(stats_summary_file, 'w') as f:
            json.dump(self.session_stats, f, indent=2)
        
        self._log(f"DEBUG SESSION SAVED TO: {self.session_dir}\n")

        return processed_batch


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================

if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"#  MICROSCOPY IMAGE PREPROCESSING PIPELINE")
    print(f"#  Component 1: Image Pre-processing Module")
    print(f"#  Version: 2.6 (CLAHE Enhancement)")
    print(f"#  Debug Logging: ENABLED")
    print(f"{'#'*60}\n")
    
    ROOT_DIR = os.path.join(os.getcwd(), "source")
    
    if not os.path.exists(ROOT_DIR):
        print(f"ERROR: Source directory not found at: {ROOT_DIR}")
        sys.exit(1)

    print(f"Root Directory: {ROOT_DIR}")
    print("\nAvailable folders:")
    
    try:
        subfolders = [f.name for f in os.scandir(ROOT_DIR) if f.is_dir()]
        if not subfolders:
            print("  (No folders found)")
            sys.exit(1)
        else:
            for i, folder in enumerate(subfolders, 1):
                print(f"  {i}. {folder}")
    except Exception as e:
        print(f"Error reading directory: {e}")
        sys.exit(1)

    SAMPLE_FOLDER = input("\nEnter the Sample Folder name (from 10 to 19): ").strip()
    
    full_sample_path = os.path.join(ROOT_DIR, SAMPLE_FOLDER)
    if not os.path.exists(full_sample_path):
        print(f"\nERROR: Folder '{SAMPLE_FOLDER}' does not exist inside 'source'.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  INITIALIZING DATA VALIDATOR")
    print(f"{'='*60}\n")
    
    loader = DataLoader(ROOT_DIR, SAMPLE_FOLDER)
    
    print("Validating Control Group...")
    control_files, control_errs = loader.get_file_pairs(loader.control_folder)
    
    print(f"Validating Sample Group ({SAMPLE_FOLDER})...")
    sample_files, sample_errs = loader.get_file_pairs(loader.sample_folder)

    print(f"\n{'='*60}")
    print(f"  VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  Control Group: {len(control_files)} valid image sets")
    print(f"  Sample Group:  {len(sample_files)} valid image sets")
    
    if control_errs or sample_errs:
        print(f"\n  ⚠️  WARNING: Found {len(control_errs) + len(sample_errs)} validation errors")
        for e in control_errs:
            print(f"    [Control] {e}")
        for e in sample_errs:
            print(f"    [Sample]  {e}")
        
        if len(control_files) == 0 and len(sample_files) == 0:
            print("\nERROR: No valid image pairs found.")
            sys.exit(1)
    print(f"{'='*60}\n")

    for c in control_files:
        c['group'] = 'Control'
    for s in sample_files:
        s['group'] = 'Sample'

    full_registry = control_files + sample_files

    if not full_registry:
        print("ERROR: No valid image pairs found.")
        sys.exit(1)

    print(f"Total registered: {len(full_registry)} image sets")

    # ============================================================
    # INITIALIZE PREPROCESSOR WITH CLAHE ENABLED
    # ============================================================
    preprocessor = PreprocessingModule(debug_dir="debug")
    
    # CLAHE is ENABLED by default in v2.6
    # To adjust parameters:
    # preprocessor.clahe_clip_limit = 3.0  # Higher = more contrast (but more noise)
    # preprocessor.clahe_tile_size = (16, 16)  # Larger tiles = smoother enhancement
    
    # To disable CLAHE:
    # preprocessor.apply_clahe_to_fl = False
    
    clean_data = preprocessor.load_and_preprocess(full_registry)

    if len(clean_data) == 0:
        print("\n❌ PIPELINE FAILED: No images successfully preprocessed.")
        sys.exit(1)
    
    print(f"\n{'#'*60}")
    print(f"#  PIPELINE COMPONENT 1 COMPLETE")
    print(f"{'#'*60}")
    print(f"\n✓ Successfully preprocessed {len(clean_data)}/{len(full_registry)} image sets")
    print(f"\nData keys:")
    print(f"  - 'bf_img_8bit' (for segmentation - grayscale)")
    print(f"  - 'fl_img_8bit' (CLAHE-enhanced for bacterial detection)")
    print(f"  - 'fl_red_visual' (for display - RGB with red FL)")
    print(f"  - 'raw_bf', 'raw_fl' (for intensity measurements)")
    print(f"\n✓ Optimization: Microgel Segmentation + CLAHE Enhancement")
    print(f"  - Normalization: {preprocessor.normalization_method.upper()}")
    if preprocessor.normalization_method == "percentile":
        print(f"  - Percentiles: P{preprocessor.percentile_low}-P{preprocessor.percentile_high}")
    print(f"  - BF Background: DISABLED (preserves microgel structures)")
    print(f"  - FL Background: DISABLED")
    print(f"  - FL CLAHE: {'ENABLED' if preprocessor.apply_clahe_to_fl else 'DISABLED'}")
    if preprocessor.apply_clahe_to_fl:
        print(f"    • Clip Limit: {preprocessor.clahe_clip_limit}")
        print(f"    • Tile Size: {preprocessor.clahe_tile_size}")
    print(f"  - FL Visualization: RED channel (RGB)")
    print(f"{'#'*60}\n")
    
    if clean_data:
        print("Sample Statistics (first processed image):")
        print("-" * 60)
        sample = clean_data[0]
        print(f"  ID: {sample['id']}")
        print(f"  Group: {sample['group']}")
        print(f"  Pixel size: {sample['pixel_size_um']:.4f} μm/pixel")
        print(f"  Image size: {sample['image_dimensions']}")
        print(f"  Bit depth: {sample['bit_depth']}")
        print(f"\n  8-bit Processed Images (for segmentation - GRAYSCALE):")
        print(f"    BF range: [{sample['bf_img_8bit'].min()}, {sample['bf_img_8bit'].max()}]")
        print(f"    BF mean: {sample['bf_img_8bit'].mean():.2f}")
        print(f"    FL range (CLAHE-enhanced): [{sample['fl_img_8bit'].min()}, {sample['fl_img_8bit'].max()}]")
        print(f"    FL mean (CLAHE-enhanced): {sample['fl_img_8bit'].mean():.2f}")
        
        print(f"\n  Red FL Visualization (for display - RGB):")
        print(f"    FL Red Visual shape: {sample['fl_red_visual'].shape}")
        print(f"    FL Red Visual dtype: {sample['fl_red_visual'].dtype}")
        print(f"    Red channel range: [{sample['fl_red_visual'][:,:,2].min()}, {sample['fl_red_visual'][:,:,2].max()}]")
        
        print(f"\n  Original Raw Images (for measurements):")
        print(f"    BF raw range: [{sample['raw_bf'].min()}, {sample['raw_bf'].max()}]")
        print(f"    FL raw range: [{sample['raw_fl'].min()}, {sample['raw_fl'].max()}]")
        
        if sample['qc_warnings']:
            print(f"\n  QC Warnings: {len(sample['qc_warnings'])}")
            for warning in sample['qc_warnings']:
                print(f"    - {warning}")
        else:
            print(f"\n  QC Warnings: None")
        
        print("-" * 60)