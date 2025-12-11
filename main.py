# main.py
"""
Main script for fluorescence microscopy analysis
Orchestrates the entire analysis pipeline
"""

import os
import numpy as np
from modules.file_handler import FileHandler
from modules.image_processor import ImageProcessor
from modules.analyzer import ParticleAnalyzer
from modules.reporter import ExcelReporter
from config import Config

def main():
    print("="*60)
    print("FLUORESCENCE MICROSCOPY ANALYSIS PIPELINE")
    print("="*60)
    
    # Initialize configuration
    config = Config()
    
    # Step 1: File Handling
    print("\n[1/4] Loading files...")
    file_handler = FileHandler(config)
    img_bf, img_red, pixel_size, unit, bit_depth = file_handler.load_all(
        grey_path=config.GREY_PATH,
        red_path=config.RED_PATH,
        meta_path=config.META_PATH
    )
    
    # Step 2: Image Pre-processing
    print("\n[2/4] Pre-processing images...")
    processor = ImageProcessor(config)
    img_bf_processed, img_red_enhanced, img_red_original = processor.preprocess(
        img_bf, img_red, pixel_size, unit
    )
    
    # Step 3: Contouring, Counting, Sizing, and Intensity Measurement
    print("\n[3/4] Analyzing particles...")
    analyzer = ParticleAnalyzer(config)
    object_data, objects_excluded = analyzer.analyze(
        img_bf_processed,
        img_red_original,
        img_red_enhanced,
        pixel_size,
        unit,
        bit_depth
    )
    
    # Step 4: Report Output to Excel
    print("\n[4/4] Generating Excel report...")
    reporter = ExcelReporter(config)
    reporter.generate_report(
        object_data,
        objects_excluded,
        pixel_size,
        unit,
        bit_depth
    )
    
    print("\n" + "="*60)
    print("✓ ANALYSIS COMPLETE")
    print("="*60)
    print(f"Total particles analyzed: {len(object_data)}")
    print(f"Particles excluded: {objects_excluded}")
    if pixel_size:
        print(f"Physical calibration: {pixel_size:.6f} {unit}/pixel")
    print(f"Results saved to: {config.DEBUG_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()