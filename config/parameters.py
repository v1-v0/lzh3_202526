#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Default parameters and constants for bacteria segmentation.
"""

import tkinter as tk
from typing import Dict, Any

DEFAULT_PARAMS = {
    'use_otsu': True,  # Otsu generally more robust
    'manual_threshold': 130,
    'enable_clahe': True,
    'clahe_clip': 3.0,
    'clahe_tile': 16,
    'open_kernel': 3,
    'close_kernel': 5,
    'open_iter': 2,
    'close_iter': 3,
    'min_area': 50,
    'watershed_dilate': 25,  # Key parameter for separation
    'fluor_brightness': 2.0,
    'fluor_gamma': 0.5,
    'show_labels': True,
    'label_font_size': 20,
    'arrow_length': 60,
    'label_offset': 15,
    'min_fluor_per_area': 10,
    'show_scale_bar': True,
    'pixel_size_um': 0.1289,
}

PARAMETER_RANGES = {
    'manual_threshold': (50, 200),
    'clahe_clip': (1, 10),
    'clahe_tile': (4, 64),
    'open_kernel': (1, 15),
    'close_kernel': (1, 15),
    'open_iter': (1, 5),
    'close_iter': (1, 5),
    'min_area': (10, 1000),
    'watershed_dilate': (5, 60),  # Expanded for better bacteria separation
    'fluor_brightness': (0.5, 5),
    'fluor_gamma': (0.2, 2),
    'label_font_size': (10, 60),
    'arrow_length': (20, 100),
    'label_offset': (5, 50),
    'min_fluor_per_area': (0, 255),
    'pixel_size_um': (0.01, 1.0),
}

DEFAULT_PIXEL_SIZE_UM = 0.1289

FONT_PATHS = [
    "arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
]


class ParameterManager:
    """Manages application parameters and settings."""
    
    def __init__(self):
        self._tk_vars = {}
        self._values = self._get_default_values()
    
    def _get_default_values(self) -> Dict[str, Any]:
        """Return default parameter values."""
        return DEFAULT_PARAMS.copy()
    
    def get_tk_variables(self) -> Dict[str, tk.Variable]:
        """Get Tkinter variables for UI binding."""
        if not self._tk_vars:
            for key, value in self._values.items():
                if isinstance(value, bool):
                    self._tk_vars[key] = tk.BooleanVar(value=value)
                elif isinstance(value, int):
                    self._tk_vars[key] = tk.IntVar(value=value)
                elif isinstance(value, float):
                    self._tk_vars[key] = tk.DoubleVar(value=value)
                else:
                    self._tk_vars[key] = tk.StringVar(value=str(value))
        return self._tk_vars
    
    def get_values(self) -> Dict[str, Any]:
        """Get current parameter values from Tkinter variables."""
        if self._tk_vars:
            # Update values from Tkinter variables
            for key, var in self._tk_vars.items():
                self._values[key] = var.get()
        return self._values.copy()
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default parameter values."""
        return self._get_default_values()
    
    def reset_to_defaults(self):
        """Reset parameters to defaults."""
        defaults = self._get_default_values()
        self._values = defaults
        # Update Tkinter variables if they exist
        if self._tk_vars:
            for key, value in defaults.items():
                if key in self._tk_vars:
                    self._tk_vars[key].set(value)
    
    def load_from_file(self, filepath: str):
        """Load parameters from file."""
        import json
        with open(filepath, 'r') as f:
            loaded = json.load(f)
            self._values.update(loaded)
            # Update Tkinter variables if they exist
            if self._tk_vars:
                for key, value in loaded.items():
                    if key in self._tk_vars:
                        self._tk_vars[key].set(value)
    
    def save_to_file(self, filepath: str):
        """Save parameters to file."""
        import json
        # Get current values from Tkinter variables
        current_values = self.get_values()
        with open(filepath, 'w') as f:
            json.dump(current_values, f, indent=2)