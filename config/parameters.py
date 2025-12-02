#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Default parameters and constants for bacteria segmentation.
"""

DEFAULT_PARAMS = {
    'use_otsu': False,
    'manual_threshold': 110,
    'enable_clahe': True,
    'clahe_clip': 5.0,
    'clahe_tile': 32,
    'open_kernel': 3,
    'close_kernel': 5,
    'open_iter': 3,
    'close_iter': 2,
    'min_area': 67,
    'watershed_dilate': 15,
    'fluor_brightness': 2.0,
    'fluor_gamma': 0.5,
    'show_labels': True,
    'label_font_size': 20,
    'arrow_length': 60,
    'label_offset': 15,
    'min_fluor_per_area': 10,
    'show_scale_bar': True,
}

PARAMETER_RANGES = {
    'manual_threshold': (0, 255),
    'clahe_clip': (1, 10),
    'clahe_tile': (4, 32),
    'open_kernel': (1, 15),
    'close_kernel': (1, 15),
    'open_iter': (1, 5),
    'close_iter': (1, 5),
    'min_area': (10, 500),
    'watershed_dilate': (1, 20),
    'fluor_brightness': (0.5, 5),
    'fluor_gamma': (0.2, 2),
    'label_font_size': (10, 60),
    'arrow_length': (20, 100),
    'label_offset': (5, 50),
    'min_fluor_per_area': (0, 255),
}

DEFAULT_PIXEL_SIZE_UM = 0.1289

FONT_PATHS = [
    "arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
]
