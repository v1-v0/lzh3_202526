# config.py
"""
Configuration file for all tunable parameters
"""

import os

class Config:
    """Central configuration for the analysis pipeline"""
    
    # File paths
    GREY_PATH = "source/12/12 N NO 1_ch00.tif"
    RED_PATH = "source/12/12 N NO 1_ch01.tif"
    META_PATH = "source/12/MetaData/12 N NO 1_Properties.xml"
    
    # Output directories
    DEBUG_DIR = "debug"
    
    # Image processing parameters
    GAUSSIAN_SIGMA = 15
    MORPH_ITERATIONS = 1
    DILATE_ITERATIONS = 1
    ERODE_ITERATIONS = 1
    MIN_OBJECT_AREA = 100
    MAX_OBJECT_AREA = 5000
    
    # Red channel enhancement parameters
    RED_NORMALIZE = True
    RED_BRIGHTNESS = 1.2
    RED_GAMMA = 0.7
    
    # Brightfield enhancement parameters
    BF_NORMALIZE = True
    BF_BRIGHTNESS = 1.0
    BF_GAMMA = 1.0
    
    # Scale bar parameters
    SCALE_BAR_LENGTH_UM = 10
    SCALE_BAR_HEIGHT = 5
    SCALE_BAR_MARGIN = 15
    SCALE_BAR_COLOR = (255, 255, 255)
    SCALE_BAR_BG_COLOR = (0, 0, 0)
    SCALE_BAR_TEXT_COLOR = (255, 255, 255)
    SCALE_BAR_FONT_SCALE = 0.6
    SCALE_BAR_FONT_THICKNESS = 2
    
    # Error bar plotting parameters
    ERROR_PERCENTAGE = 0.1
    PLOT_DPI = 150
    
    def __init__(self):
        """Initialize and create necessary directories"""
        os.makedirs(self.DEBUG_DIR, exist_ok=True)