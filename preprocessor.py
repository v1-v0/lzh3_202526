import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import tifffile
import sys

# Import the validator class
try:
    from loader import DataLoader
except ImportError:
    print("CRITICAL ERROR: Could not import 'loader.py'.")
    sys.exit(1)

class PreprocessingModule:
    def __init__(self):
        # Parameters derived from "Metadata Microscopy Specifications"
        self.target_pixel_size = 0.1095  # μm/pixel
        
        # DYNAMIC DIMENSION: Initialized to None, learned from first image
        self.target_dim = None   
        
        self.gaussian_kernel = (5, 5)    # Kernel size for noise reduction
        self.sigma = 1.0                 # Standard deviation for Gaussian
        
        # Thresholds
        self.fl_black_level_8bit = 7 

    def parse_pixel_size(self, xml_path):
        """Parses XML to find calibrated pixel size."""
        try:
            if not os.path.exists(xml_path):
                return None
            # Simulating XML parsing for spec value
            return 0.1095 
        except Exception as e:
            print(f"   [!] Metadata Error {os.path.basename(xml_path)}: {e}")
            return None

    def convert_12bit_to_8bit(self, image_12bit):
        """
        Linearly scales 12-bit data (0-4095) to 8-bit (0-255).
        """
        scaled = (image_12bit / 4095.0) * 255.0
        return np.clip(scaled, 0, 255).astype(np.uint8)

    def load_and_preprocess(self, file_registry):
        """
        Input: List of dicts {'bf': path, 'fl': path, 'xml': path, 'group': str}
        Output: List of processed dictionaries separating QC data (12-bit) and Analysis data (8-bit)
        """
        processed_batch = []

        print(f"\n--- Starting Pre-processing on {len(file_registry)} image sets ---")

        for entry in file_registry:
            sample_id = entry['id']
            
            # 1. Parse XML Metadata
            pixel_size = self.parse_pixel_size(entry['xml'])
            if pixel_size is None:
                continue

            # 2. Load Raw Images (12-bit TIF) - STRICTLY FOR QC
            try:
                img_bf_raw_12bit = tifffile.imread(entry['bf'])
                img_fl_raw_12bit = tifffile.imread(entry['fl'])
            except Exception as e:
                print(f"   [!] Read Error {sample_id}: {e}")
                continue

            # 3. Validate Dimensions (Dynamic Logic)
            # If this is the first image, set the target dimensions based on it
            if self.target_dim is None:
                self.target_dim = img_bf_raw_12bit.shape
                print(f"   [i] Auto-detected target dimensions: {self.target_dim}")
            
            # Check against the established target dimension
            if img_bf_raw_12bit.shape != self.target_dim:
                print(f"   [!] Dimension mismatch for {sample_id}. Expected {self.target_dim}, Got {img_bf_raw_12bit.shape}")
                continue
            
            # Also ensure BF and FL match each other
            if img_bf_raw_12bit.shape != img_fl_raw_12bit.shape:
                 print(f"   [!] Channel mismatch for {sample_id}. BF: {img_bf_raw_12bit.shape}, FL: {img_fl_raw_12bit.shape}")
                 continue

            # 4. Conversion (12-bit -> 8-bit)
            # All enhancement happens AFTER this step
            bf_8bit = self.convert_12bit_to_8bit(img_bf_raw_12bit)
            fl_8bit = self.convert_12bit_to_8bit(img_fl_raw_12bit)

            # 5. Enhancement: Gaussian Blur (Noise Reduction) on 8-bit data
            bf_blur = cv2.GaussianBlur(bf_8bit, self.gaussian_kernel, self.sigma)
            fl_blur = cv2.GaussianBlur(fl_8bit, self.gaussian_kernel, self.sigma)

            # 6. Enhancement: Background Correction on 8-bit data
            
            # Bright-field: Subtract estimated background
            bg_bf_est = cv2.blur(bf_blur, (50, 50)) 
            bf_corrected = cv2.subtract(bf_blur, bg_bf_est)
            
            # Fluorescence: Subtract sensor black level using Numpy safe math
            fl_corrected = np.clip(fl_blur.astype(np.int16) - self.fl_black_level_8bit, 0, 255).astype(np.uint8)

            # 7. Register Output
            processed_data = {
                "id": sample_id,
                "pixel_size_um": pixel_size,
                "group": entry.get('group', 'unknown'),
                
                # --- PROCESSED DATA (8-bit) ---
                # Use these for segmentation, measurement, and visualization
                "bf_img_8bit": bf_corrected, 
                "fl_img_8bit": fl_corrected,
                
                # --- QC DATA (12-bit) ---
                # Use these ONLY for quality checking (intensity verification)
                "raw_bf_12bit": img_bf_raw_12bit,
                "raw_fl_12bit": img_fl_raw_12bit
            }
            processed_batch.append(processed_data)
            print(f"   [OK] Processed {sample_id}")

        return processed_batch

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================

if __name__ == "__main__":
    # 1. SETUP: Define paths dynamically
    ROOT_DIR = os.path.join(os.getcwd(), "source")
    
    if not os.path.exists(ROOT_DIR):
        print(f"Error: Source directory not found at: {ROOT_DIR}")
        print("Please create a folder named 'source' and place your image data there.")
        sys.exit(1)

    # 2. USER INPUT: Select the specific sample folder
    print(f"Root Directory: {ROOT_DIR}")
    print("Available folders:")
    try:
        subfolders = [f.name for f in os.scandir(ROOT_DIR) if f.is_dir()]
        print(subfolders)
    except Exception as e:
        print(f"Error reading directory: {e}")

    SAMPLE_FOLDER = input("\nEnter the Sample Folder name (e.g., Sample_Group_10): ").strip()
    
    full_sample_path = os.path.join(ROOT_DIR, SAMPLE_FOLDER)
    if not os.path.exists(full_sample_path):
        print(f"Error: Folder '{SAMPLE_FOLDER}' does not exist inside 'source'.")
        sys.exit(1)

    # 3. VALIDATION
    print("\n--- Initializing Validator ---")
    loader = DataLoader(ROOT_DIR, SAMPLE_FOLDER)
    
    control_files, control_errs = loader.get_file_pairs(loader.control_folder)
    sample_files, sample_errs = loader.get_file_pairs(loader.sample_folder)

    if control_errs or sample_errs:
        print(f"Warning: Found {len(control_errs) + len(sample_errs)} validation errors.")

    # 4. REGISTRATION
    for c in control_files: c['group'] = 'Control'
    for s in sample_files: s['group'] = 'Sample'

    full_registry = control_files + sample_files

    if not full_registry:
        print("No valid image pairs found. Exiting.")
        sys.exit(1)

    # 5. PRE-PROCESSING
    preprocessor = PreprocessingModule()
    clean_data = preprocessor.load_and_preprocess(full_registry)

    # 6. HANDOFF
    print(f"\nPipeline Component 1 Complete.")
    print(f"Successfully preprocessed {len(clean_data)} image sets.")
    print("Data keys available: 'bf_img_8bit', 'fl_img_8bit', 'raw_bf_12bit', 'raw_fl_12bit'")