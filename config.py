"""
config.py - Configuration and Constants
Centralized settings for the Bacteria Segmentation application
"""

import platform
from pathlib import Path

# ============================================================================
# APPLICATION METADATA
# ============================================================================
APP_NAME = "Bacteria Segmentation Tuner"
APP_VERSION = "2.0"
APP_DESCRIPTION = "Interactive brightfield/fluorescence microscopy analysis"

# ============================================================================
# DEFAULT PROCESSING PARAMETERS
# ============================================================================
DEFAULT_PARAMS = {
    # Thresholding
    "use_otsu": False,
    "manual_threshold": 110,

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    "enable_clahe": True,
    "clahe_clip": 5.0,
    "clahe_tile": 32,

    # Morphological operations
    "open_kernel": 3,
    "close_kernel": 5,
    "open_iter": 3,
    "close_iter": 2,

    # Segmentation
    "min_area": 67,  # pixels
    "watershed_dilate": 15,

    # Fluorescence display
    "fluor_brightness": 2.0,
    "fluor_gamma": 0.5,

    # Visualization
    "show_labels": True,
    "label_fontsize": 20,
    "arrow_length": 60,
    "label_offset": 15,

    # Filtering
    "min_fluor_per_area": 10,
    "show_scale_bar": True,
}

# ============================================================================
# UI STYLING - COLORS
# ============================================================================
DARK_MODE_COLORS = {
    "bg": "#2b2b2b",
    "fg": "#e0e0e0",
    "frame": "#3c3c3c",
    "button_bg": "#4a4a4a",
    "button_fg": "#ffffff",
    "canvas_bg": "#1e1e1e",
}

LIGHT_MODE_COLORS = {
    "bg": "#ffffff",
    "fg": "#000000",
    "frame": "#f0f0f0",
    "button_bg": "#e1e1e1",
    "button_fg": "#000000",
    "canvas_bg": "#f8f9fa",
}

# ============================================================================
# FONTS & SIZES
# =========