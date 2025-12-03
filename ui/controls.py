#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parameter control panels.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Callable
from .widgets import ToolTip, ProgressEntry
from ..config.parameters import PARAMETER_RANGES


class ParameterControls:
    """Manages parameter control panels."""
    
    def __init__(self, parent: ttk.Frame, params: Dict[str, tk.Variable], 
                 on_change: Callable):
        """Initialize parameter controls.
        
        Args:
            parent: Parent frame
            params: Dictionary of parameter variables
            on_change: Callback when parameter changes
        """
        self.parent = parent
        self.params = params
        self.on_change = on_change
        self.entries: Dict[str, ProgressEntry] = {}
    
    def create_threshold_controls(self):
        """Create threshold parameter controls."""
        lf = ttk.LabelFrame(self.parent, text=" Threshold ", padding=8)
        lf.pack(fill=tk.X, pady=(0, 5))
        
        cb = ttk.Checkbutton(
            lf, text="Use Otsu", 
            variable=self.params['use_otsu'],
            command=self.on_change
        )
        cb.pack(anchor=tk.W, pady=2)
        ToolTip(cb, "Automatically calculate threshold using Otsu's method")
        
        entry = ProgressEntry(
            lf, "Manual Threshold:", 0, 255, 
            self.params['manual_threshold'].get(),
            lambda v: self._update_param('manual_threshold', v),
            tooltip="Threshold value (0-255)"
        )
        entry.pack(fill=tk.X, pady=1)
        self.entries['manual_threshold'] = entry
        
        return lf
    
    def create_clahe_controls(self):
        """Create CLAHE parameter controls."""
        lf = ttk.LabelFrame(self.parent, text=" CLAHE Enhancement ", padding=8)
        lf.pack(fill=tk.X, pady=(0, 5))
        
        cb = ttk.Checkbutton(
            lf, text="Enable CLAHE",
            variable=self.params['enable_clahe'],
            command=self.on_change
        )
        cb.pack(anchor=tk.W, pady=2)
        ToolTip(cb, "Apply Contrast Limited Adaptive Histogram Equalization")
        
        entry = ProgressEntry(
            lf, "Clip Limit:", 1, 10,
            self.params['clahe_clip'].get(),
            lambda v: self._update_param('clahe_clip', v),
            resolution=0.1, is_float=True,
            tooltip="CLAHE clip limit (1-10)"
        )
        entry.pack(fill=tk.X, pady=1)
        self.entries['clahe_clip'] = entry
        
        entry = ProgressEntry(
            lf, "Tile Size:", 4, 32,
            self.params['clahe_tile'].get(),
            lambda v: self._update_param('clahe_tile', v),
            tooltip="CLAHE tile grid size (4-32)"
        )
        entry.pack(fill=tk.X, pady=1)
        self.entries['clahe_tile'] = entry
        
        return lf
    
    def create_morphology_controls(self):
        """Create morphology parameter controls."""
        lf = ttk.LabelFrame(self.parent, text=" Morphology ", padding=8)
        lf.pack(fill=tk.X, pady=(0, 5))
        
        params = [
            ("open_kernel", "Open Kernel:", "Opening kernel size (odd, 1-15)"),
            ("open_iter", "Open Iterations:", "Opening iterations (1-5)"),
            ("close_kernel", "Close Kernel:", "Closing kernel size (odd, 1-15)"),
            ("close_iter", "Close Iterations:", "Closing iterations (1-5)"),
        ]
        
        for param_name, label, tooltip in params:
            mn, mx = PARAMETER_RANGES[param_name]
            entry = ProgressEntry(
                lf, label, mn, mx,
                self.params[param_name].get(),
                lambda v, p=param_name: self._update_param(p, v),
                tooltip=tooltip
            )
            entry.pack(fill=tk.X, pady=1)
            self.entries[param_name] = entry
        
        return lf
    
    def create_watershed_controls(self):
        """Create watershed and filtering parameter controls."""
        lf = ttk.LabelFrame(self.parent, text=" Watershed & Filtering ", padding=8)
        lf.pack(fill=tk.X, pady=(0, 5))
        
        entry = ProgressEntry(
            lf, "Watershed Dilate:", 1, 20,
            self.params['watershed_dilate'].get(),
            lambda v: self._update_param('watershed_dilate', v),
            tooltip="Watershed marker dilation (1-20)"
        )
        entry.pack(fill=tk.X, pady=1)
        self.entries['watershed_dilate'] = entry
        
        entry = ProgressEntry(
            lf, "Min Area (px²):", 10, 500,
            self.params['min_area'].get(),
            lambda v: self._update_param('min_area', v),
            tooltip="Minimum bacteria area in pixels (10-500)"
        )
        entry.pack(fill=tk.X, pady=1)
        self.entries['min_area'] = entry
        
        return lf
    
    def create_fluorescence_controls(self):
        """Create fluorescence parameter controls."""
        lf = ttk.LabelFrame(self.parent, text=" Fluorescence ", padding=8)
        lf.pack(fill=tk.X, pady=(0, 5))
        
        entry = ProgressEntry(
            lf, "Brightness:", 0.5, 5,
            self.params['fluor_brightness'].get(),
            lambda v: self._update_param('fluor_brightness', v),
            resolution=0.1, is_float=True,
            tooltip="Fluorescence brightness multiplier (0.5-5)"
        )
        entry.pack(fill=tk.X, pady=1)
        self.entries['fluor_brightness'] = entry
        
        entry = ProgressEntry(
            lf, "Gamma:", 0.2, 2,
            self.params['fluor_gamma'].get(),
            lambda v: self._update_param('fluor_gamma', v),
            resolution=0.1, is_float=True,
            tooltip="Fluorescence gamma correction (0.2-2)"
        )
        entry.pack(fill=tk.X, pady=1)
        self.entries['fluor_gamma'] = entry
        
        entry = ProgressEntry(
            lf, "Min Fluor/Area:", 0, 255,
            self.params['min_fluor_per_area'].get(),
            lambda v: self._update_param('min_fluor_per_area', v),
            resolution=0.1, is_float=True,
            tooltip="Minimum fluorescence per area ratio (0-255)"
        )
        entry.pack(fill=tk.X, pady=1)
        self.entries['min_fluor_per_area'] = entry
        
        return lf
    
    def create_label_controls(self):
        """Create label display parameter controls."""
        lf = ttk.LabelFrame(self.parent, text=" Labels & Scale ", padding=8)
        lf.pack(fill=tk.X, pady=(0, 5))
        
        cb = ttk.Checkbutton(
            lf, text="Show Labels",
            variable=self.params['show_labels'],
            command=self.on_change
        )
        cb.pack(anchor=tk.W, pady=2)
        ToolTip(cb, "Display numbered labels for bacteria")
        
        cb_scale = ttk.Checkbutton(
            lf, text="Show Scale Bar",
            variable=self.params['show_scale_bar'],
            command=self.on_change
        )
        cb_scale.pack(anchor=tk.W, pady=2)
        ToolTip(cb_scale, "Display scale bar with physical units (µm)")
        
        params = [
            ("label_font_size", "Font Size:", "Label font size (10-60)"),
            ("arrow_length", "Arrow Length:", "Arrow length in pixels (20-100)"),
            ("label_offset", "Label Offset:", "Label offset from arrow (5-50)"),
        ]
        
        for param_name, label, tooltip in params:
            mn, mx = PARAMETER_RANGES[param_name]
            entry = ProgressEntry(
                lf, label, mn, mx,
                self.params[param_name].get(),
                lambda v, p=param_name: self._update_param(p, v),
                tooltip=tooltip
            )
            entry.pack(fill=tk.X, pady=1)
            self.entries[param_name] = entry
        
        return lf
    
    def _update_param(self, param_name: str, value: float):
        """Update parameter and trigger change callback."""
        self.params[param_name].set(value)
        self.on_change()
    
    def reset_to_defaults(self, defaults: Dict):
        """Reset all parameters to default values.
        
        Args:
            defaults: Dictionary of default values
        """
        for key, value in defaults.items():
            if key in self.params:
                self.params[key].set(value)
                if key in self.entries:
                    self.entries[key].set_value(value)