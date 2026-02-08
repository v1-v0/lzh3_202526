"""
Integrated Pathogen Configuration Manager
Combines main menu, config management, and segmentation tuner
"""

from logging import config
import os
import sys
import json
from arrow import get
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider, Button
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, fields
import xml.etree.ElementTree as ET
import ast
import astor

from bacteria_configs import SegmentationConfig, _manager

# ==================================================
# SECTION 0.5: Responsive UI Configuration
# ==================================================

class UIScaler:
    """Manages responsive UI scaling"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.base_width = 1920
        self.base_height = 1080
        self.min_width = 1280
        self.min_height = 720
        
        # Get current screen size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Calculate scale factors
        self.width_scale = screen_width / self.base_width
        self.height_scale = screen_height / self.base_height
        self.scale = min(self.width_scale, self.height_scale)
        
        print(f" Screen: {screen_width}x{screen_height}")
        print(f" Scale factor: {self.scale:.2f}")
    
    def scale_font(self, base_size: int) -> int:
        """Scale font size"""
        return max(8, int(base_size * self.scale))
    
    def scale_dimension(self, base_dim: int) -> int:
        """Scale a dimension"""
        return int(base_dim * self.scale)
    
    def get_window_size(self) -> tuple[int, int]:
        """Get scaled window size"""
        # For 1920x1080, use 95% of screen to leave room for taskbar
        width = int(self.base_width * 0.95)
        height = int(self.base_height * 0.90)
        return width, height


# ==================================================
# SECTION 0.6: Modern Parameter Panel (Tkinter-based) - FIXED RESET
# ==================================================

class ParameterPanel(ttk.Frame):
    """Modern collapsible parameter panel with sliders and input boxes"""
    
    def __init__(self, parent, tuner_instance):
        super().__init__(parent)
        self.tuner = tuner_instance
        self.sliders = {}
        self.value_labels = {}
        self.input_boxes = {}
        self.section_frames = {}
        
        # ✅ Store the INITIAL pathogen-specific values as defaults
        self.default_params = {
            'gaussian_sigma': self.tuner.params['gaussian_sigma'],
            'manual_threshold': self.tuner.manual_threshold,
            'morph_kernel_size': self.tuner.morph_kernel_size,
            'morph_iterations': self.tuner.morph_iterations,
            'min_area': self.tuner.params['min_area'],
            'max_area': self.tuner.params['max_area'],
            'dilate_iterations': self.tuner.params['dilate_iterations'],
            'erode_iterations': self.tuner.params['erode_iterations'],
            'min_circularity': self.tuner.min_circularity,
            'max_circularity': self.tuner.max_circularity,
            'min_aspect_ratio': self.tuner.min_aspect_ratio,
            'max_aspect_ratio': self.tuner.max_aspect_ratio,
            'min_solidity': self.tuner.min_solidity,
            'threshold_mode': self.tuner.threshold_mode,
            'invert_image': self.tuner.invert_image
        }
        
        self.configure(relief=tk.RIDGE, borderwidth=1, width=320)
        self.pack_propagate(False)
        
        # Header
        header = tk.Frame(self, bg=SegmentationTuner.COLORS['secondary'], height=40)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="🎚️ PARAMETERS",
            font=("Segoe UI", 12, "bold"),
            bg=SegmentationTuner.COLORS['secondary'],
            fg="white"
        ).pack(side=tk.LEFT, padx=15, pady=8)
        
        # Scrollable canvas with VISIBLE scrollbar
        canvas_container = tk.Frame(self, bg='white')
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas
        self.canvas = tk.Canvas(canvas_container, bg='#f8f9fa', highlightthickness=0)
        
        # Create CUSTOM styled scrollbar (more visible)
        scrollbar_frame = tk.Frame(canvas_container, bg='#cccccc', width=16)
        scrollbar_frame.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_frame.pack_propagate(False)
        
        # Custom scrollbar style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            "Custom.Vertical.TScrollbar",
            background='#888888',
            troughcolor='#e0e0e0',
            bordercolor='#cccccc',
            arrowcolor='white',
            darkcolor='#666666',
            lightcolor='#aaaaaa',
            gripcount=0
        )
        style.map(
            "Custom.Vertical.TScrollbar",
            background=[('active', '#555555'), ('!active', '#888888')]
        )
        
        scrollbar = ttk.Scrollbar(
            scrollbar_frame,
            orient="vertical",
            command=self.canvas.yview,
            style="Custom.Vertical.TScrollbar"
        )
        scrollbar.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Create scrollable frame
        self.scrollable_frame = tk.Frame(self.canvas, bg='#f8f9fa')
        
        # Configure scroll region
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window inside canvas
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configure canvas to expand window width
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Link scrollbar to canvas
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Enable mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        
        # Build sections
        self._create_sections()
    
    def _on_canvas_configure(self, event):
        """Update canvas window width when canvas is resized"""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _create_sections(self):
        """Create collapsible parameter sections"""
        
        # SEGMENTATION Section
        seg_content = self._create_collapsible_section(
            "SEGMENTATION",
            SegmentationTuner.COLORS['primary'],
            expanded=True
        )
        
        self._add_slider_with_input(seg_content, "Gaussian σ", "gaussian_sigma",
                                   self.tuner.params['gaussian_sigma'], 0.5, 20.0, 0.1)
        
        self._add_threshold_controls(seg_content)
        
        self._add_slider_with_input(seg_content, "Morph Kernel", "morph_kernel_size",
                                   self.tuner.morph_kernel_size, 3, 15, 2)
        
        self._add_slider_with_input(seg_content, "Morph Iterations", "morph_iterations",
                                   self.tuner.morph_iterations, 0, 5, 1)
        
        # FILTERING Section
        filt_content = self._create_collapsible_section(
            "FILTERING",
            SegmentationTuner.COLORS['info'],
            expanded=True
        )
        
        self._add_slider_with_input(filt_content, "Min Area (px)", "min_area",
                                   self.tuner.params['min_area'], 10, 5000, 10)
        
        self._add_slider_with_input(filt_content, "Max Area (px)", "max_area",
                                   self.tuner.params['max_area'], 100, 20000, 100)
        
        self._add_slider_with_input(filt_content, "Dilate Iterations", "dilate_iterations",
                                   self.tuner.params['dilate_iterations'], 0, 5, 1)
        
        self._add_slider_with_input(filt_content, "Erode Iterations", "erode_iterations",
                                   self.tuner.params['erode_iterations'], 0, 5, 1)
        
        # SHAPE FILTERS Section
        shape_content = self._create_collapsible_section(
            "SHAPE FILTERS",
            SegmentationTuner.COLORS['purple'],
            expanded=True
        )
        
        self._add_slider_with_input(shape_content, "Min Circularity", "min_circularity",
                                   self.tuner.min_circularity, 0.0, 1.0, 0.01)
        
        self._add_slider_with_input(shape_content, "Max Circularity", "max_circularity",
                                   self.tuner.max_circularity, 0.0, 1.0, 0.01)
        
        self._add_slider_with_input(shape_content, "Min Aspect Ratio", "min_aspect_ratio",
                                   self.tuner.min_aspect_ratio, 0.0, 10.0, 0.1)
        
        self._add_slider_with_input(shape_content, "Max Aspect Ratio", "max_aspect_ratio",
                                   self.tuner.max_aspect_ratio, 0.0, 20.0, 0.1)
        
        self._add_slider_with_input(shape_content, "Min Solidity", "min_solidity",
                                   self.tuner.min_solidity, 0.0, 1.0, 0.01)
        
        # CONTROL BUTTONS Section
        self._create_control_buttons()
        
        # Add some bottom padding
        tk.Frame(self.scrollable_frame, bg='#f8f9fa', height=20).pack()
    
    def _create_collapsible_section(self, title: str, color: str, expanded: bool = True):
        """Create a collapsible section with header"""
        container = tk.Frame(self.scrollable_frame, bg='#f8f9fa')
        container.pack(fill=tk.X, padx=8, pady=(8, 2))
        
        # Header button
        header_frame = tk.Frame(container, bg=color, cursor="hand2", height=32)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Arrow label
        arrow_var = tk.StringVar(value="▼" if expanded else "▶")
        arrow_label = tk.Label(
            header_frame,
            textvariable=arrow_var,
            font=("Segoe UI", 10, "bold"),
            bg=color,
            fg="white",
            width=2
        )
        arrow_label.pack(side=tk.LEFT, padx=(8, 2))
        
        # Title label
        title_label = tk.Label(
            header_frame,
            text=title,
            font=("Segoe UI", 10, "bold"),
            bg=color,
            fg="white",
            anchor="w"
        )
        title_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 8))
        
        # Content frame
        content_frame = tk.Frame(container, bg='white', relief=tk.SUNKEN, borderwidth=1)
        if expanded:
            content_frame.pack(fill=tk.X, padx=2, pady=(0, 2))
        
        # Toggle function
        def toggle(event=None):
            if content_frame.winfo_ismapped():
                content_frame.pack_forget()
                arrow_var.set("▶")
            else:
                content_frame.pack(fill=tk.X, padx=2, pady=(0, 2))
                arrow_var.set("▼")
        
        # Bind click events
        header_frame.bind("<Button-1>", toggle)
        arrow_label.bind("<Button-1>", toggle)
        title_label.bind("<Button-1>", toggle)
        
        return content_frame
    
    def _add_slider_with_input(self, parent, label: str, param_name: str, 
                               initial: float, min_val: float, max_val: float, step: float):
        """Add a parameter slider with input box"""
        container = tk.Frame(parent, bg='white')
        container.pack(fill=tk.X, padx=8, pady=6)
        
        # Label and input box row
        label_row = tk.Frame(container, bg='white')
        label_row.pack(fill=tk.X)
        
        tk.Label(
            label_row,
            text=label + ":",
            font=("Segoe UI", 9),
            bg='white',
            fg='#555',
            anchor='w'
        ).pack(side=tk.LEFT)
        
        # Input box (Entry widget)
        input_frame = tk.Frame(label_row, bg='white')
        input_frame.pack(side=tk.RIGHT)
        
        if step < 1:
            value_text = f"{initial:.2f}"
        else:
            value_text = f"{int(initial)}"
        
        input_var = tk.StringVar(value=value_text)
        input_box = tk.Entry(
            input_frame,
            textvariable=input_var,
            font=("Segoe UI", 9, "bold"),
            width=8,
            bg='#f0f0f0',
            fg=SegmentationTuner.COLORS['header'],
            relief=tk.SUNKEN,
            borderwidth=1,
            justify='center'
        )
        input_box.pack(side=tk.LEFT)
        self.input_boxes[param_name] = (input_box, input_var, step)
        
        # Bind Enter key to update from input
        def on_input_change(event=None):
            try:
                val = float(input_var.get())
                val = max(min_val, min(max_val, val))

                if param_name == 'morph_kernel_size':
                    if val % 2 == 0:
                        val += 1  # Ensure kernel size is odd
                    val = max(min_val, min(max_val, val))

                if step >= 1:
                    val = int(round(val / step) * step)
                else:
                    val = round(val / step) * step
                self.sliders[param_name].set(val)
                if step < 1:
                    input_var.set(f"{val:.2f}")
                else:
                    input_var.set(f"{int(val)}")
            except ValueError:
                val = self.sliders[param_name].get()
                if step < 1:
                    input_var.set(f"{val:.2f}")
                else:
                    input_var.set(f"{int(val)}")
        
        input_box.bind('<Return>', on_input_change)
        input_box.bind('<FocusOut>', on_input_change)
        
        # Slider
        slider = tk.Scale(
            container,
            from_=min_val,
            to=max_val,
            resolution=step,
            orient=tk.HORIZONTAL,
            showvalue=False,
            bg='white',
            highlightthickness=0,
            troughcolor='#ddd',
            activebackground=SegmentationTuner.COLORS['primary'],
            command=lambda val: self._on_slider_change(param_name, val, step, input_var)
        )
        slider.set(initial)
        slider.pack(fill=tk.X, pady=(2, 0))
        self.sliders[param_name] = slider
        
        # Range label
        tk.Label(
            container,
            text=f"Range: {min_val} - {max_val}",
            font=("Segoe UI", 7),
            bg='white',
            fg='#999'
        ).pack(anchor='w')
    
    def _add_threshold_controls(self, parent):
        """Add threshold mode controls"""
        container = tk.Frame(parent, bg='white')
        container.pack(fill=tk.X, padx=8, pady=6)
        
        tk.Label(
            container,
            text="Threshold Method:",
            font=("Segoe UI", 9),
            bg='white',
            fg='#555',
            anchor='w'
        ).pack(anchor="w")
        
        # Threshold mode button
        self.thresh_mode_btn = tk.Button(
            container,
            text=f"{self.tuner.threshold_mode.upper()}",
            font=("Segoe UI", 9, "bold"),
            bg=SegmentationTuner.COLORS['primary'],
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            pady=6,
            command=self._cycle_threshold_mode
        )
        self.thresh_mode_btn.pack(fill=tk.X, pady=2)
        
        # Manual threshold slider
        self.manual_threshold_container = tk.Frame(container, bg='white')
        if self.tuner.threshold_mode == "manual":
            self.manual_threshold_container.pack(fill=tk.X, pady=4)
        
        self._add_slider_with_input(self.manual_threshold_container, "Threshold Value", 
                                   "manual_threshold",
                                   self.tuner.manual_threshold, 0, 255, 1)
    
    def _create_control_buttons(self):
        """Create control buttons section"""
        control_section = tk.Frame(self.scrollable_frame, bg='#f8f9fa')
        control_section.pack(fill=tk.X, padx=8, pady=10)
        
        # Invert button
        invert_color = SegmentationTuner.COLORS['success'] if self.tuner.invert_image else SegmentationTuner.COLORS['gray']
        self.invert_btn = tk.Button(
            control_section,
            text=f"INVERT: {'ON' if self.tuner.invert_image else 'OFF'}",
            font=("Segoe UI", 9, "bold"),
            bg=invert_color,
            fg="white",
            relief=tk.RAISED,
            cursor="hand2",
            pady=8,
            command=self._toggle_invert
        )
        self.invert_btn.pack(fill=tk.X, pady=3)
        
        # Apply suggestions button
        apply_btn = tk.Button(
            control_section,
            text="✨ APPLY SUGGESTIONS",
            font=("Segoe UI", 9, "bold"),
            bg=SegmentationTuner.COLORS['warning'],
            fg="white",
            relief=tk.RAISED,
            cursor="hand2",
            pady=8,
            command=self._apply_suggestions
        )
        apply_btn.pack(fill=tk.X, pady=3)
        
        # Reset button
        reset_btn = tk.Button(
            control_section,
            text="🔄 RESET TO PRESET",
            font=("Segoe UI", 9, "bold"),
            bg=SegmentationTuner.COLORS['danger'],
            fg="white",
            relief=tk.RAISED,
            cursor="hand2",
            pady=8,
            command=self._reset_to_default
        )
        reset_btn.pack(fill=tk.X, pady=3)
    
    def _on_slider_change(self, param_name: str, value: str, step: float, input_var: tk.StringVar):
        """Handle slider value change"""
        val = float(value)
        
        if step >= 1:
            val = int(round(val / step) * step)
        else:
            val = round(val / step) * step
        
        if step < 1:
            input_var.set(f"{val:.2f}")
        else:
            input_var.set(f"{int(val)}")
        
        if param_name in self.tuner.params:
            self.tuner.update_parameter(param_name, val)
        elif param_name in ['morph_kernel_size', 'morph_iterations']:
            self.tuner.update_morph(param_name, val)
        elif param_name == 'manual_threshold':
            self.tuner.update_threshold(param_name, val)
        else:
            self.tuner.update_shape_filter(param_name, val)
    
    def _cycle_threshold_mode(self):
        """Cycle through threshold modes"""
        modes = ["otsu", "manual", "adaptive"]
        current_idx = modes.index(self.tuner.threshold_mode)
        next_mode = modes[(current_idx + 1) % len(modes)]
        
        self.tuner.threshold_mode = next_mode
        self.thresh_mode_btn.config(text=next_mode.upper())
        
        if next_mode == "manual":
            self.manual_threshold_container.pack(fill=tk.X, pady=4)
        else:
            self.manual_threshold_container.pack_forget()
        
        self.tuner.update_visualization()
    
    def _toggle_invert(self):
        """Toggle image inversion"""
        self.tuner.invert_image = not self.tuner.invert_image
        
        invert_color = SegmentationTuner.COLORS['success'] if self.tuner.invert_image else SegmentationTuner.COLORS['gray']
        self.invert_btn.config(
            text=f"INVERT: {'ON' if self.tuner.invert_image else 'OFF'}",
            bg=invert_color
        )
        
        self.tuner.update_visualization()
    
    def _apply_suggestions(self):
        """Apply parameter suggestions"""
        if not self.tuner.current_suggestions:
            messagebox.showinfo("Info", "Click on a particle first to get suggestions",
                              parent=self.winfo_toplevel())
            return
        
        for param, value in self.tuner.current_suggestions.items():
            if param in self.sliders:
                self.sliders[param].set(value)
        
        messagebox.showinfo("Applied", "Suggestions have been applied!",
                          parent=self.winfo_toplevel())


    def _reset_to_default(self):
        """✅ Reset all parameters to PATHOGEN-SPECIFIC preset values"""
        # Get pathogen name for message
        pathogen_name = getattr(self.tuner, 'current_pathogen', "preset")

        # Confirm reset
        confirm = messagebox.askyesno(
            "Reset Parameters",
            f"Reset all parameters to {pathogen_name.upper()} preset?\n\n"
            f"This will discard ALL current changes.",  # ✅ Fixed typo
            parent=self.winfo_toplevel()
        )
        
        if not confirm:
            return
        
        # ✅ Try to load from bacteria_configs JSON first
        config_key = pathogen_name.lower().replace(' ', '_').replace('.', '')
        config_file = Path("bacteria_configs") / f"{config_key}.json"

        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # ✅ Handle nested "config" structure
                if "config" in json_data:
                    config_data = json_data["config"]
                else:
                    config_data = json_data
                
                # ✅ Convert µm² to pixels for area parameters
                um2_per_px2 = self.tuner.pixel_size_um ** 2
                
                # Reset sliders with proper parameter mapping
                slider_mappings = {
                    'gaussian_sigma': config_data.get('gaussian_sigma', 2.0),
                    'manual_threshold': config_data.get('manual_threshold', 127),
                    'morph_kernel_size': config_data.get('morph_kernel_size', 3),
                    'morph_iterations': config_data.get('morph_iterations', 1),
                    'min_area': config_data.get('min_area_um2', 3.0) / um2_per_px2,  # ✅ Convert
                    'max_area': config_data.get('max_area_um2', 100.0) / um2_per_px2,  # ✅ Convert
                    'dilate_iterations': config_data.get('dilate_iterations', 0),
                    'erode_iterations': config_data.get('erode_iterations', 0),
                    'min_circularity': config_data.get('min_circularity', 0.0),
                    'max_circularity': config_data.get('max_circularity', 1.0),
                    'min_aspect_ratio': config_data.get('min_aspect_ratio', 0.2),
                    'max_aspect_ratio': config_data.get('max_aspect_ratio', 10.0),
                    'min_solidity': config_data.get('min_solidity', 0.3),
                }
                
                for param_name, value in slider_mappings.items():
                    if param_name in self.sliders:
                        self.sliders[param_name].set(value)
                
                # Reset threshold mode
                threshold_mode = config_data.get('threshold_mode', 'otsu')
                self.tuner.threshold_mode = threshold_mode
                self.thresh_mode_btn.config(text=threshold_mode.upper())
                
                if threshold_mode == "manual":
                    self.manual_threshold_container.pack(fill=tk.X, pady=4)
                else:
                    self.manual_threshold_container.pack_forget()
                
                # Reset invert
                invert_value = config_data.get('invert_image', False)
                self.tuner.invert_image = invert_value
                invert_color = SegmentationTuner.COLORS['success'] if invert_value else SegmentationTuner.COLORS['gray']
                self.invert_btn.config(
                    text=f"INVERT: {'ON' if invert_value else 'OFF'}",
                    bg=invert_color
                )
                
                # Trigger visualization update
                self.tuner.update_visualization()
                
                messagebox.showinfo("Reset Complete", 
                                f"Parameters reset to {pathogen_name.upper()} preset from config file!",
                                parent=self.winfo_toplevel())
                return
                
            except Exception as e:
                print(f"❌ Failed to load config file for reset: {e}")
                import traceback
                traceback.print_exc()
                # Fall through to default_params fallback
        
        # ✅ FALLBACK: Use stored default_params (loaded at initialization)
        print(f"ℹ️ No config file found, using default_params stored at initialization")
        
        for param_name, default_value in self.default_params.items():
            if param_name in self.sliders:
                self.sliders[param_name].set(default_value)
            elif param_name == 'threshold_mode':
                self.tuner.threshold_mode = default_value
                self.thresh_mode_btn.config(text=default_value.upper())
                if default_value == "manual":
                    self.manual_threshold_container.pack(fill=tk.X, pady=4)
                else:
                    self.manual_threshold_container.pack_forget()
            elif param_name == 'invert_image':
                self.tuner.invert_image = default_value
                invert_color = SegmentationTuner.COLORS['success'] if default_value else SegmentationTuner.COLORS['gray']
                self.invert_btn.config(
                    text=f"INVERT: {'ON' if default_value else 'OFF'}",
                    bg=invert_color
                )
        
        # Trigger visualization update
        self.tuner.update_visualization()
        
        messagebox.showinfo("Reset Complete", 
                        f"Parameters reset to {pathogen_name.upper()} initial values!",
                        parent=self.winfo_toplevel())



# ==================================================
# SECTION 1: Configuration Data Classes
# ==================================================

@property
def min_area_px(self) -> float:
    """Convert min_area_um2 to pixels²"""
    um2_per_px2 = 0.012
    return self.min_area_um2 / um2_per_px2

@property
def max_area_px(self) -> float:
    """Convert max_area_um2 to pixels²"""
    um2_per_px2 = 0.012
    return self.max_area_um2 / um2_per_px2


# Built-in configurations
PROTEUS_MIRABILIS = SegmentationConfig(
    name='Proteus mirabilis',
    description='bacteria segmentation - Tuned 2026-02-02',
    gaussian_sigma=2.17,
    min_area_um2=3.60,
    max_area_um2=72.11
)

KLEBSIELLA_PNEUMONIAE = SegmentationConfig(
    name='Klebsiella pneumoniae',
    description='bacteria segmentation - Tuned 2026-02-03',
    gaussian_sigma=4.05,
    min_area_um2=4.19,
    max_area_um2=154.00
)

STREPTOCOCCUS_MITIS = SegmentationConfig(
    name='Streptococcus mitis',
    description='Alpha-hemolytic, gram-positive cocci in chains',
    gaussian_sigma=12.0,
    min_area_um2=0.2,
    max_area_um2=50.0,
    min_aspect_ratio=0.8,
    max_aspect_ratio=6.0,
    min_circularity=0.5,
    max_circularity=1.0,
    min_solidity=0.7
)

DEFAULT_CONFIG = SegmentationConfig(
    name='Default (General Purpose)',
    description='Generic bacteria detection profile',
    gaussian_sigma=15.0,
    min_area_um2=0.3,
    max_area_um2=2000.0
)

_CONFIGS: Dict[str, SegmentationConfig] = {
    'proteus_mirabilis': PROTEUS_MIRABILIS,
    'klebsiella_pneumoniae': KLEBSIELLA_PNEUMONIAE,
    'streptococcus_mitis': STREPTOCOCCUS_MITIS,
    'default': DEFAULT_CONFIG
}


def get_config(bacteria_type: str) -> SegmentationConfig:
    """Get configuration for specified bacteria type"""
    if bacteria_type not in _CONFIGS:
        print(f"[WARN] Unknown bacteria type '{bacteria_type}', using default")
        return DEFAULT_CONFIG
    return _CONFIGS[bacteria_type]


# ==================================================
# SECTION 2: Config File Manager (AST-based)
# ==================================================

class ConfigFileManager:
    """Manages bacteria_configs.py using AST parsing"""
    
    def __init__(self, config_file: Path):
        self.config_file = config_file
        self.tree: Optional[ast.Module] = None
        self.source: Optional[str] = None
        
    def load(self) -> bool:
        """Load and parse the config file"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.source = f.read()
            
            self.tree = ast.parse(self.source)
            return True
        except Exception as e:
            print(f"❌ Failed to load config file: {e}")
            return False
    
    def find_config_assignment(self, var_name: str) -> Optional[Tuple[int, ast.Assign]]:
        """Find the assignment node for a specific config variable"""
        if self.tree is None:
            return None
        
        for idx, node in enumerate(self.tree.body):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        return idx, node
        
        return None
    
    def create_config_assignment(self, var_name: str, config_data: dict) -> ast.Assign:
        """Create an AST assignment node for a SegmentationConfig"""
        keywords = []
        
        for key, value in config_data.items():
            if isinstance(value, str):
                value_node = ast.Constant(value=value)
            elif isinstance(value, (int, float)):
                value_node = ast.Constant(value=value)
            elif isinstance(value, bool):
                value_node = ast.Constant(value=value)
            else:
                value_node = ast.Constant(value=value)
            
            keywords.append(ast.keyword(arg=key, value=value_node))
        
        config_call = ast.Call(
            func=ast.Name(id='SegmentationConfig', ctx=ast.Load()),
            args=[],
            keywords=keywords
        )
        
        assignment = ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=config_call
        )
        
        return assignment
    
    def update_config(self, var_name: str, config_data: dict) -> bool:
        """Update or add a configuration"""
        if self.tree is None:
            if not self.load():
                return False
        
        assert self.tree is not None
        
        new_node = self.create_config_assignment(var_name, config_data)
        result = self.find_config_assignment(var_name)
        
        if result:
            idx, _ = result
            self.tree.body[idx] = new_node
            print(f"  ✓ Updated existing {var_name} configuration")
        else:
            default_idx = self._find_default_config_index()
            
            if default_idx is not None:
                self.tree.body.insert(default_idx, new_node)
                print(f"  ✓ Inserted new {var_name} configuration before DEFAULT")
            else:
                self.tree.body.append(new_node)
                print(f"  ✓ Appended new {var_name} configuration")
        
        return True
    
    def _find_default_config_index(self) -> Optional[int]:
        """Find the index of DEFAULT config assignment"""
        result = self.find_config_assignment('DEFAULT')
        return result[0] if result else None
    
    def save(self, backup: bool = True) -> bool:
        """Save the modified AST back to file"""
        if self.tree is None:
            print("❌ No AST tree to save")
            return False
        
        if astor is None:
            print("❌ 'astor' module not available. Cannot save.")
            return False
        
        try:
            if backup:
                backup_path = self.config_file.with_suffix('.py.bak')
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    with open(backup_path, 'w', encoding='utf-8') as bf:
                        bf.write(f.read())
                print(f"  ✓ Created backup: {backup_path.name}")
            
            new_source = astor.to_source(self.tree)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                f.write(new_source)
            
            print(f"  ✓ Saved to {self.config_file.name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save: {e}")
            return False
    
    def validate_syntax(self) -> bool:
        """Validate that the generated file has valid Python syntax"""
        if self.tree is None or astor is None:
            return False
        
        try:
            compile(astor.to_source(self.tree), str(self.config_file), 'exec')
            return True
        except SyntaxError as e:
            print(f"❌ Syntax error in generated code: {e}")
            return False


def config_to_dict(config: SegmentationConfig) -> dict:
    """Convert SegmentationConfig to dictionary"""
    return {
        field.name: getattr(config, field.name)
        for field in fields(config)
    }


def update_bacteria_config(bacterium: str, config: SegmentationConfig, backup: bool = True) -> bool:
    """Update bacteria configuration in bacteria_configs.py"""
    config_file = Path(__file__).parent / "bacteria_configs.py"
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return False
    
    manager = ConfigFileManager(config_file)
    
    if not manager.load():
        return False
    
    var_name = bacterium.upper().replace(' ', '_').replace('.', '')
    config_dict = config_to_dict(config)
    
    if not manager.update_config(var_name, config_dict):
        return False
    
    if not manager.validate_syntax():
        print("❌ Generated code has syntax errors - not saving")
        return False
    
    return manager.save(backup=backup)


# ==================================================
# SECTION 3: Unicode-Safe File I/O
# ==================================================

def safe_imread(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    """Read image with Unicode path support on Windows"""
    try:
        with open(path, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, flags)
        
        if img is None:
            print(f"[WARN] cv2.imdecode returned None for {path.name}")
            return None
            
        return img
    except Exception as e:
        print(f"[ERROR] Failed to read image {path.name}: {e}")
        return None


def safe_imwrite(path: Path, img: np.ndarray, params: Optional[list] = None) -> bool:
    """Write image with Unicode path support on Windows"""
    try:
        ext = path.suffix.lower()
        if not ext:
            ext = '.png'
        
        if params is None:
            is_success, buffer = cv2.imencode(ext, img)
        else:
            is_success, buffer = cv2.imencode(ext, img, params)
        
        if not is_success:
            print(f"[WARN] cv2.imencode failed for {path.name}")
            return False
        
        with open(path, 'wb') as f:
            f.write(buffer.tobytes())
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write image {path.name}: {e}")
        return False


def safe_xml_parse(xml_path: Path) -> Optional[ET.ElementTree]:
    """Parse XML file with Unicode path support"""
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        root = ET.fromstring(content)
        tree = ET.ElementTree(root)
        
        return tree
    except FileNotFoundError:
        print(f"[WARN] XML file not found: {xml_path.name}")
        return None
    except ET.ParseError as e:
        print(f"[ERROR] XML parse error in {xml_path.name}: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to read XML {xml_path.name}: {e}")
        return None


def validate_path_encoding(path: Path) -> bool:
    """Check if path can be properly encoded for filesystem"""
    try:
        path_str = str(path.resolve())
        path_str.encode(sys.getfilesystemencoding())
        return True
    except UnicodeEncodeError as e:
        print(f"[WARN] Path encoding issue: {path}")
        print(f"       {e}")
        return False
    except Exception as e:
        print(f"[WARN] Path validation error: {e}")
        return False


# ==================================================
# SECTION 4: Metadata Extraction
# ==================================================

def find_metadata_paths(img_path: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Find associated metadata XML files"""
    base = img_path.stem
    if base.endswith("_ch00"):
        base = base[:-5]
    md_dir = img_path.parent / "MetaData"
    xml_main = md_dir / f"{base}.xml"
    xml_props = md_dir / f"{base}_Properties.xml"

    return (
        xml_props if xml_props.exists() else None,
        xml_main if xml_main.exists() else None,
    )


def _require_attr(elem: ET.Element, attr: str, context: str) -> str:
    """Helper to extract required XML attribute"""
    v = elem.get(attr)
    if v is None:
        raise ValueError(f"Missing attribute '{attr}' in {context}")
    return v


def _parse_float(s: str) -> float:
    """Parse float with comma/period handling"""
    return float(s.strip().replace(",", "."))


def get_pixel_size_um(
    xml_props_path: Optional[Path],
    xml_main_path: Optional[Path],
) -> Tuple[float, float]:
    """Extract pixel size with detailed error reporting"""
    errors = []
    
    if xml_props_path is not None:
        try:
            tree = safe_xml_parse(xml_props_path)
            if tree is None:
                raise ValueError(f"Could not parse {xml_props_path.name}")
            
            root = tree.getroot()
            if root is None:
                raise ValueError(f"Empty XML document: {xml_props_path.name}")

            dims = root.findall(".//ImageDescription/Dimensions/DimensionDescription")
            by_id = {d.get("DimID"): d for d in dims}

            def read_dim(dim_id: str) -> Tuple[float, int, str]:
                d = by_id.get(dim_id)
                if d is None:
                    raise ValueError(f"Missing DimensionDescription with DimID='{dim_id}'")

                length_s = _require_attr(d, "Length", f"{xml_props_path.name} DimID={dim_id}")
                n_s = _require_attr(d, "NumberOfElements", f"{xml_props_path.name} DimID={dim_id}")
                unit = _require_attr(d, "Unit", f"{xml_props_path.name} DimID={dim_id}")

                length = _parse_float(length_s)
                n = int(n_s)
                return length, n, unit

            x_len, x_n, x_unit = read_dim("X")
            y_len, y_n, y_unit = read_dim("Y")

            if x_unit != "µm" or y_unit != "µm":
                raise ValueError(f"Unexpected units: X={x_unit}, Y={y_unit}")

            return float(x_len / x_n), float(y_len / y_n)
        except Exception as e:
            errors.append(f"Properties XML: {e}")

    if xml_main_path is not None:
        try:
            tree = safe_xml_parse(xml_main_path)
            if tree is None:
                raise ValueError(f"Could not parse {xml_main_path.name}")
            
            root = tree.getroot()
            if root is None:
                raise ValueError(f"Empty XML document: {xml_main_path.name}")

            dims = root.findall(".//ImageDescription/Dimensions/DimensionDescription")
            by_id = {d.get("DimID"): d for d in dims}

            def read_dim(dim_id: str) -> Tuple[float, int, str]:
                d = by_id.get(dim_id)
                if d is None:
                    raise ValueError(f"Missing DimensionDescription with DimID='{dim_id}'")

                length_s = _require_attr(d, "Length", f"{xml_main_path.name} DimID={dim_id}")
                n_s = _require_attr(d, "NumberOfElements", f"{xml_main_path.name} DimID={dim_id}")
                unit = _require_attr(d, "Unit", f"{xml_main_path.name} DimID={dim_id}")

                length = _parse_float(length_s)
                n = int(n_s)
                return length, n, unit

            x_len_m, x_n, x_unit = read_dim("1")
            y_len_m, y_n, y_unit = read_dim("2")

            if x_unit != "m" or y_unit != "m":
                raise ValueError(f"Unexpected units: X={x_unit}, Y={y_unit}")

            return float((x_len_m * 1e6) / x_n), float((y_len_m * 1e6) / y_n)
        except Exception as e:
            errors.append(f"Main XML: {e}")
    
    error_summary = "\n  - ".join(errors) if errors else "No XML files provided"
    raise ValueError(f"Could not determine pixel size (µm/px).\nAttempted sources:\n  - {error_summary}")


def normalize_to_8bit(img: np.ndarray) -> np.ndarray:
    """Normalize image to 8-bit"""
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        out = np.zeros_like(img, dtype=np.uint8)
        cv2.normalize(img, out, 0, 255, cv2.NORM_MINMAX)
        return out
    img_f = img.astype(np.float32)
    mn, mx = float(np.min(img_f)), float(np.max(img_f))
    if mx <= mn:
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img_f - mn) * (255.0 / (mx - mn))).clip(0, 255).astype(np.uint8)


# ==================================================
# SECTION 5: Segmentation Tuner
# ==================================================

class SegmentationTuner:
    """Interactive segmentation parameter tuner"""
    
    DEFAULT_PARAMS = {
        "gaussian_sigma": 2.0,
        "min_area": 20,
        "max_area": 5000,
        "dilate_iterations": 0,
        "erode_iterations": 0,
    }

    DEFAULT_THRESHOLD_PARAMS = {
    "threshold_mode": "otsu",
    "manual_threshold": 127,
    "morph_kernel_size": 3,
    "morph_iterations": 1,
    }   



    DEFAULT_SHAPE_FILTERS = {
        "min_circularity": 0.0,
        "max_circularity": 1.0,
        "min_aspect_ratio": 0.2,
        "max_aspect_ratio": 10.0,
        "min_solidity": 0.3,
    }
    
    FALLBACK_UM_PER_PX = 0.109492
    
    COLORS = {
        'bg': '#f0f0f0',
        'header': '#2c3e50',
        'primary': '#3498db',
        'success': '#27ae60',
        'warning': '#e67e22',
        'info': '#16a085',
        'secondary': '#34495e',
        'danger': '#e74c3c',
        'purple': '#9b59b6',
        'gray': '#95a5a6',
    }

    def __init__(self, root: tk.Tk, image_path: str, bacterium: str,       structure: str, mode: str, return_callback=None):
        """Initialize the segmentation tuner"""
        # ✅ Store root reference FIRST
        self.current_pathogen = bacterium
        self.master = root
        self.root = root
        
        # ✅ Initialize UI scaler
        self.ui_scaler = UIScaler(root)
        
        self.image_path = Path(image_path)
        self.bacterium = bacterium
        self.structure = structure
        self.mode = mode
        self.return_callback = return_callback
        
        # Initialize image navigation attributes
        self.image_list = []
        self.image_index = 0
        
        if not validate_path_encoding(self.image_path):
            raise ValueError(f"Path contains problematic characters: {image_path}")
        
        self.original_image = self._load_image(self.image_path)
        self.pixel_size_um, self.has_metadata = self._load_pixel_size()
        
        self._initialize_parameters()
        
        self.processed_image: np.ndarray = np.zeros_like(self.original_image)
        self.binary_mask: np.ndarray = np.zeros_like(self.original_image)
        self.contours: List[np.ndarray] = []
        self.contour_areas: List[float] = []
        self.current_suggestions: Dict[str, Any] = {}
        
        self.sliders: Dict[str, Slider] = {}
        self.param_labels: Dict[str, tk.Label] = {}
        
        self.setup_gui()

    def quit(self, event=None):
        """Quit the application"""
        if messagebox.askyesno("Quit", "Are you sure you want to quit?\n\nUnsaved changes will be lost."):
            self.master.quit()
            self.master.destroy()
    
    def back(self, event=None):
        """Go back to previous image or state"""
        if hasattr(self, 'image_index') and self.image_index > 0:
            self.image_index -= 1
            self.load_image_at_index(self.image_index)
        else:
            messagebox.showinfo("Info", "Already at the first image")
    
    def load_image_at_index(self, index: int):
        """Load image at specified index from image list"""
        if not hasattr(self, 'image_list') or not self.image_list:
            messagebox.showwarning("Warning", "No image list available")
            return
        
        if 0 <= index < len(self.image_list):
            image_path = self.image_list[index]
            try:
                self.original_image = self._load_image(image_path)
                self.image_path = image_path
                self.image_index = index
                
                # Update window title
                self.master.title(f"Tuner - [{index + 1}/{len(self.image_list)}] {image_path.name}")
                
                # Refresh visualization
                self.update_visualization()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{e}")
        else:
            messagebox.showwarning("Warning", f"Invalid image index: {index}")
    
    def save_and_apply(self, event=None):
        """Save parameters to JSON and update bacteria configuration"""
        # First save to temporary JSON (session file)
        if not self.save():
            messagebox.showerror("Error", "Failed to save parameters")
            return
        
        try:
            um2_per_px2 = self.pixel_size_um ** 2
            
            # Create config object (using the imported SegmentationConfig)
            config = SegmentationConfig(
                name=f"{self.bacterium}",
                description=f"{self.structure} segmentation - Tuned {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                gaussian_sigma=float(self.params['gaussian_sigma']),
                min_area_um2=float(self.params['min_area']) * um2_per_px2,
                max_area_um2=float(self.params['max_area']) * um2_per_px2,
                dilate_iterations=int(self.params['dilate_iterations']),
                erode_iterations=int(self.params['erode_iterations']),
                min_circularity=float(self.min_circularity),
                max_circularity=float(self.max_circularity),
                min_aspect_ratio=float(self.min_aspect_ratio),
                max_aspect_ratio=float(self.max_aspect_ratio),
                min_mean_intensity=0,
                max_mean_intensity=255,
                max_edge_gradient=200,
                min_solidity=float(self.min_solidity),
                max_fraction_of_image=0.25,
                fluor_min_area_um2=3.0,
                fluor_max_area_um2=2000.0,
                fluor_match_min_intersection_px=5.0,
                invert_image=bool(self.invert_image),
                threshold_mode=str(self.threshold_mode),
                manual_threshold=int(self.manual_threshold),
                
                morph_kernel_size=int(self.morph_kernel_size),
                morph_iterations=int(self.morph_iterations),
    

                pixel_size_um=float(self.pixel_size_um),
                last_modified=datetime.now().isoformat(),
                tuned_by="Interactive Tuner"
            )
            
            # Convert bacterium name to key
            bacteria_key = self.bacterium.lower().replace(' ', '_').replace('.', '').replace('-', '_')
            
            # Update configuration using imported manager
            success = _manager.update_config(bacteria_key, config)
            
            if success:
                config_file = _manager._get_config_path(bacteria_key)
                
                messagebox.showinfo(
                    "Success",
                    f"✓ Configuration saved!\n\n"
                    f"Bacterium: {self.bacterium}\n"
                    f"Saved to: {config_file.name}\n\n"
                )
                
                print(f"\n{'='*80}")
                print(f"CONFIGURATION SAVED")
                print(f"{'='*80}")
                print(f"Bacterium: {self.bacterium}")
                print(f"Key: {bacteria_key}")
                print(f"File: {config_file}")
                print(f"\nParameters:")
                print(f"  Gaussian σ: {config.gaussian_sigma:.2f}")
                print(f"  Min Area: {config.min_area_um2:.2f} µm²")
                print(f"  Max Area: {config.max_area_um2:.2f} µm²")
                print(f"  Circularity: {config.min_circularity:.2f} - {config.max_circularity:.2f}")
                print(f"  Aspect Ratio: {config.min_aspect_ratio:.2f} - {config.max_aspect_ratio:.2f}")
                print(f"  Solidity: ≥ {config.min_solidity:.2f}")
                print(f"  Pixel size: {config.pixel_size_um:.6f} µm")
                print(f"{'='*80}\n")
            else:
                messagebox.showerror(
                    "Error",
                    "Failed to update configuration\n"
                    "Check console for details"
                )
                
        except Exception as e:
            print(f"✗ Error saving configuration: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to save configuration:\n{e}")
    
    def save(self) -> bool:
        """Save current parameters to temporary JSON file
        
        Returns:
            bool: True if successful
        """
        try:
            session_file = Path("tuner_session.json")
            
            session_data = {
                'bacterium': self.bacterium,
                'structure': self.structure,
                'pixel_size_um': self.pixel_size_um,
                'parameters': {
                    'gaussian_sigma': self.params['gaussian_sigma'],
                    'min_area': self.params['min_area'],
                    'max_area': self.params['max_area'],
                    'dilate_iterations': self.params['dilate_iterations'],
                    'erode_iterations': self.params['erode_iterations'],
                    'min_circularity': self.min_circularity,
                    'max_circularity': self.max_circularity,
                    'min_aspect_ratio': self.min_aspect_ratio,
                    'max_aspect_ratio': self.max_aspect_ratio,
                    'min_solidity': self.min_solidity,
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2)
            
            print(f"✓ Session saved to: {session_file}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to save session: {e}")
            return False
    
    
         
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load and validate image"""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = safe_imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = normalize_to_8bit(image)
        
        print(f"✅ Loaded image: {image_path.name}")
        print(f"   Shape: {image.shape}, Dtype: {image.dtype}")
        return image
    
    def _load_pixel_size(self) -> Tuple[float, bool]:
        """Load pixel size from metadata with fallback"""
        try:
            xml_props, xml_main = find_metadata_paths(self.image_path)
            
            if xml_props or xml_main:
                um_per_px_x, um_per_px_y = get_pixel_size_um(xml_props, xml_main)
                um_per_px_avg = (um_per_px_x + um_per_px_y) / 2.0
                
                print(f"✅ Loaded pixel size from metadata: {um_per_px_avg:.6f} µm/px")
                return um_per_px_avg, True
            else:
                print(f"⚠ No metadata found, using fallback: {self.FALLBACK_UM_PER_PX} µm/px")
                return self.FALLBACK_UM_PER_PX, False
                
        except Exception as e:
            print(f"⚠ Error loading metadata, using fallback: {e}")
            return self.FALLBACK_UM_PER_PX, False


    def _initialize_parameters(self):
        """Initialize parameters - loads from bacteria_configs first, then session JSON"""
        # ✅ ALWAYS initialize shape filters first (default values)
        for key, value in self.DEFAULT_SHAPE_FILTERS.items():
            setattr(self, key, value)
        
        # Convert bacterium name to config key
        config_key = self.bacterium.lower().replace(' ', '_').replace('.', '').replace('-', '_')
        
        # ===================================================================
        # Priority 1: Load from bacteria_configs/{pathogen}.json (PERMANENT)
        # ===================================================================
        config_file = Path("bacteria_configs") / f"{config_key}.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # ✅ FIX: Handle nested structure with "config" key
                if "config" in json_data:
                    config_data = json_data["config"]
                else:
                    config_data = json_data
                
                # Extract parameters
                um2_per_px2 = self.pixel_size_um ** 2
                
                self.threshold_mode = config_data.get('threshold_mode', 'otsu')
                self.manual_threshold = config_data.get('manual_threshold', 127)
                self.morph_kernel_size = config_data.get('morph_kernel_size', 3)
                self.morph_iterations = config_data.get('morph_iterations', 1)


                self.params = {
                    "gaussian_sigma": float(config_data.get('gaussian_sigma', 2.0)),
                    "min_area": float(config_data.get('min_area_um2', 3.0) / um2_per_px2),
                    "max_area": float(config_data.get('max_area_um2', 100.0) / um2_per_px2),
                    "dilate_iterations": int(config_data.get('dilate_iterations', 0)),
                    "erode_iterations": int(config_data.get('erode_iterations', 0)),
                }
                
                # Override shape filters from config
                self.min_circularity = float(config_data.get('min_circularity', 0.0))
                self.max_circularity = float(config_data.get('max_circularity', 1.0))
                self.min_aspect_ratio = float(config_data.get('min_aspect_ratio', 0.2))
                self.max_aspect_ratio = float(config_data.get('max_aspect_ratio', 10.0))
                self.min_solidity = float(config_data.get('min_solidity', 0.3))
                
                self.invert_image = False
                
                print(f"✅ Loaded PERMANENT config from: {config_file}")
                print(f"   • Gaussian σ: {self.params['gaussian_sigma']:.2f}")
                print(f"   • Min area: {self.params['min_area']:.1f} px "
                    f"({config_data.get('min_area_um2', 0):.2f} µm²)")
                print(f"   • Max area: {self.params['max_area']:.1f} px "
                    f"({config_data.get('max_area_um2', 0):.2f} µm²)")
                print(f"   • Circularity: {self.min_circularity:.2f} - {self.max_circularity:.2f}")
                print(f"   • Aspect ratio: {self.min_aspect_ratio:.2f} - {self.max_aspect_ratio:.2f}")
                print(f"   • Solidity: ≥ {self.min_solidity:.2f}")
                return
                
            except Exception as e:
                print(f"⚠️ Could not load bacteria_configs JSON: {e}")
                import traceback
                traceback.print_exc()
        
        # ===================================================================
        # Priority 2: Load from session JSON (temporary tuning)
        # ===================================================================
        json_filename = f"segmentation_params_{self.bacterium}_{self.structure}_{self.mode}.json"
        
        if Path(json_filename).exists():
            try:
                with open(json_filename, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                
                params_dict = saved_data['parameters'].copy()
                self.invert_image = params_dict.pop('invert_image', False)
                
                # Load main parameters
                self.params = {
                    "gaussian_sigma": params_dict.get('gaussian_sigma', 2.0),
                    "min_area": params_dict.get('min_area', 20),
                    "max_area": params_dict.get('max_area', 5000),
                    "dilate_iterations": params_dict.get('dilate_iterations', 0),
                    "erode_iterations": params_dict.get('erode_iterations', 0),
                }
                
                # Load shape filters if present
                self.min_circularity = params_dict.get('min_circularity', 0.0)
                self.max_circularity = params_dict.get('max_circularity', 1.0)
                self.min_aspect_ratio = params_dict.get('min_aspect_ratio', 0.2)
                self.max_aspect_ratio = params_dict.get('max_aspect_ratio', 10.0)
                self.min_solidity = params_dict.get('min_solidity', 0.3)

                self.threshold_mode = "otsu"  # "otsu", "manual", "adaptive"
                self.manual_threshold = 127   # For manual mode
                self.morph_kernel_size = 3    # From SegmentationConfig
                self.morph_iterations = 1     # CLOSE iterations


                
                print(f"✅ Restored TEMP session from: {json_filename}")
                return
                
            except Exception as e:
                print(f"⚠️ Could not load session JSON: {e}")
                # Continue to next priority
        
        # ===================================================================
        # Priority 3: Load from bacteria_configs.py (legacy Python config)
        # ===================================================================
        if config_key in _CONFIGS:
            try:
                saved_config = _CONFIGS[config_key]
                um2_per_px2 = self.pixel_size_um ** 2
                
                self.params = {
                    "gaussian_sigma": float(saved_config.gaussian_sigma),
                    "min_area": float(saved_config.min_area_um2 / um2_per_px2),
                    "max_area": float(saved_config.max_area_um2 / um2_per_px2),
                    "dilate_iterations": int(saved_config.dilate_iterations),
                    "erode_iterations": int(saved_config.erode_iterations),
                }
                
                # Override shape filters from config
                self.min_circularity = float(saved_config.min_circularity)
                self.max_circularity = float(saved_config.max_circularity)
                self.min_aspect_ratio = float(saved_config.min_aspect_ratio)
                self.max_aspect_ratio = float(saved_config.max_aspect_ratio)
                self.min_solidity = float(saved_config.min_solidity)
                
                self.invert_image = False
                
                print(f"✅ Loaded config for {self.bacterium} from bacteria_configs.py")
                print(f"   • Gaussian σ: {self.params['gaussian_sigma']:.2f}")
                print(f"   • Min area: {self.params['min_area']:.1f} px "
                    f"({saved_config.min_area_um2:.2f} µm²)")
                print(f"   • Max area: {self.params['max_area']:.1f} px "
                    f"({saved_config.max_area_um2:.2f} µm²)")
                return
                
            except Exception as e:
                print(f"⚠️ Error loading bacteria_configs.py: {e}")

        
        # ===================================================================
        # Priority 4: Use defaults (nothing found)
        # ===================================================================
        print(f"ℹ️ No saved config found for '{self.bacterium}' - using defaults")
        self.params = self.DEFAULT_PARAMS.copy()
        self.threshold_mode = self.DEFAULT_THRESHOLD_PARAMS['threshold_mode']
        self.manual_threshold = self.DEFAULT_THRESHOLD_PARAMS['manual_threshold']
        self.morph_kernel_size = self.DEFAULT_THRESHOLD_PARAMS['morph_kernel_size']
        self.morph_iterations = self.DEFAULT_THRESHOLD_PARAMS['morph_iterations']
        self.invert_image = False

    def update_threshold(self, param_name: str, value: float):
        """Update threshold parameter"""
        setattr(self, param_name, value)
        self.update_visualization()

    def update_morph(self, param_name: str, value: float):
        """Update morphology parameter"""
        setattr(self, param_name, value)
        self.update_visualization()

    def cycle_threshold_mode(self, event):
        """Cycle through threshold modes: Otsu → Manual → Adaptive"""
        modes = ["otsu", "manual", "adaptive"]
        current_idx = modes.index(self.threshold_mode)
        next_idx = (current_idx + 1) % len(modes)
        self.threshold_mode = modes[next_idx]
        
        mode_text = f"THRESHOLD\n{self.threshold_mode.upper()}"
        self.btn_thresh_mode.label.set_text(mode_text)
        
        # Enable/disable manual threshold slider
        if 'manual_threshold' in self.sliders:
            self.sliders['manual_threshold'].set_active(self.threshold_mode == "manual")
        
        self.update_visualization()

    def setup_gui(self):
        """Setup the GUI with vertical slider panel on left"""
        window_width, window_height = self.ui_scaler.get_window_size()
        
        self.root.title(f"Segmentation Tuner - {self.bacterium}")
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.minsize(1280, 720)
        self.root.configure(bg=self.COLORS['bg'])
        
        self.root.resizable(True, True)
        self.root.bind('<Configure>', self._on_window_resize)
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (window_width // 2)
        y = (self.root.winfo_screenheight() // 2) - (window_height // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self._create_header(main_container)
        
        # NEW: Content frame with 3 columns
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # NEW: Left panel - vertical sliders
        self._create_vertical_slider_panel(content_frame)
        
        # Middle panel - image display
        self._create_image_panel(content_frame)
        
        # Right panel - parameters & analysis
        self._create_right_panel(content_frame)
        
        print("✅ GUI Setup Complete")
        self.update_visualization()
        print("✅ Initial visualization complete")

    def _create_vertical_slider_panel(self, parent: ttk.Frame):
        """Create modern vertical slider panel on the left side"""
        # Use the new ParameterPanel class
        self.parameter_panel = ParameterPanel(parent, tuner_instance=self)
        self.parameter_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))

    def _on_window_resize(self, event):
        """Handle window resize events"""
        # Only process resize events from the main window
        if event.widget == self.root:
            # Debounce resize events
            if hasattr(self, '_resize_timer'):
                self.root.after_cancel(self._resize_timer)
            self._resize_timer = self.root.after(100, self._refresh_layout)
    
    def _refresh_layout(self):
        """Refresh layout after resize"""
        # Update canvas sizes if needed
        if hasattr(self, 'canvas_image'):
            self.canvas_image.draw()
        if hasattr(self, 'canvas_hist'):
            self.canvas_hist.draw()
        if hasattr(self, 'canvas_sliders'):
            self.canvas_sliders.draw()

    def _create_header(self, parent: ttk.Frame):
        """Create header section with scaled fonts"""
        header_height = self.ui_scaler.scale_dimension(45)
        header_frame = tk.Frame(parent, bg=self.COLORS['header'], height=header_height)
        header_frame.pack(fill=tk.X, pady=(0, 5))
        header_frame.pack_propagate(False)
        
        # Scaled fonts
        title_font = ("Segoe UI", self.ui_scaler.scale_font(14), "bold")
        badge_font = ("Segoe UI", self.ui_scaler.scale_font(9))
        button_font = ("Segoe UI", self.ui_scaler.scale_font(9), "bold")
        
        # Left side - Title
        title_text = f"🔬 {self.bacterium} - {self.structure}"
        tk.Label(
            header_frame,
            text=title_text,
            font=title_font,
            bg=self.COLORS['header'],
            fg="white"
        ).pack(side=tk.LEFT, padx=20, pady=6)
        
        # Mode badge
        mode_text = f"Mode: {self.mode} {'(Inverted)' if self.invert_image else ''}"
        tk.Label(
            header_frame,
            text=mode_text,
            font=badge_font,
            bg=self.COLORS['secondary'],
            fg="white",
            padx=12,
            pady=4,
            relief=tk.RAISED
        ).pack(side=tk.LEFT, pady=6)
        
        # Pixel size badge
        pixel_color = self.COLORS['success'] if self.has_metadata else self.COLORS['warning']
        pixel_text = f"Pixel: {self.pixel_size_um:.6f} µm"
        if not self.has_metadata:
            pixel_text += " (fallback)"
        
        tk.Label(
            header_frame,
            text=pixel_text,
            font=badge_font,
            bg=pixel_color,
            fg="white",
            padx=12,
            pady=4,
            relief=tk.RAISED
        ).pack(side=tk.LEFT, padx=10, pady=6)
        
        # Load image button
        tk.Button(
            header_frame,
            text="📁 LOAD IMAGE",
            font=button_font,
            bg=self.COLORS['primary'],
            fg="white",
            padx=12,
            pady=4,
            relief=tk.RAISED,
            command=self.load_new_image,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=10, pady=6)
        
        # Contour count (right side, before buttons)
        self.contour_count_label = tk.Label(
            header_frame,
            text="Contours: 0",
            font=badge_font,
            bg=self.COLORS['success'],
            fg="white",
            padx=10,
            pady=4,
            relief=tk.RAISED
        )
        self.contour_count_label.pack(side=tk.RIGHT, padx=(10, 20), pady=6)
        
        # Right side - Action buttons (scaled)
        button_specs = [
            ("❌ QUIT", self.COLORS['danger'], self.quit),
            ("✅ SAVE & APPLY", self.COLORS['success'], self.save_and_apply),
            ("💾 SAVE JSON", self.COLORS['secondary'], self.save),
            ("⬅ BACK", self.COLORS['secondary'], self.back),
        ]
        
        for text, color, command in button_specs:
            tk.Button(
                header_frame,
                text=text,
                font=button_font,
                bg=color,
                fg="white",
                activebackground=color,
                activeforeground="white",
                relief=tk.RAISED,
                command=command,
                cursor="hand2",
                padx=12,
                pady=4
            ).pack(side=tk.RIGHT, padx=5, pady=6)

    
    def _create_content_area(self, parent: ttk.Frame) -> ttk.Frame:
        """Create main content area"""
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        self._create_image_panel(content_frame)
        self._create_right_panel(content_frame)
        
        return content_frame
    
    def _create_image_panel(self, parent: ttk.Frame):
        """Create left image panel with responsive sizing"""
        left_panel = ttk.Frame(parent, relief=tk.RIDGE, borderwidth=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        header = tk.Frame(left_panel, bg=self.COLORS['secondary'], height=30)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="📷 IMAGE ANALYSIS - Original + Contours",
            font=("Segoe UI", 11, "bold"),
            bg=self.COLORS['secondary'],
            fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=5)
        
        canvas_frame = ttk.Frame(left_panel)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # For 1920x1080, make image area larger
        self.fig_image = Figure(figsize=(11, 8), facecolor='white', dpi=100)
        self.ax_image = self.fig_image.add_subplot(111)
        self.canvas_image = FigureCanvasTkAgg(self.fig_image, canvas_frame)
        self.canvas_image.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_image.mpl_connect("button_press_event", self.on_image_click)
        
        instruction = tk.Frame(left_panel, bg=self.COLORS['primary'], height=28)
        instruction.pack(fill=tk.X)
        instruction.pack_propagate(False)
        
        tk.Label(
            instruction,
            text="💡 Click on a particle to analyze and get parameter suggestions",
            font=("Segoe UI", 9),
            bg=self.COLORS['primary'],
            fg="white"
        ).pack(pady=4)



    def _create_right_panel(self, parent: ttk.Frame):
        """Create right control panel with fixed optimal width"""
        # Fixed width that works well on 1920x1080
        right_panel = ttk.Frame(parent, width=420)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        self._create_parameters_section(right_panel)
        self._create_target_analysis_section(right_panel)
        self._create_histogram_section(right_panel)



    def _create_parameters_section(self, parent: ttk.Frame):
        """Create parameters display section with scaled fonts"""
        header_height = self.ui_scaler.scale_dimension(28)
        header = tk.Frame(parent, bg=self.COLORS['secondary'], height=header_height)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        header_font = ("Segoe UI", self.ui_scaler.scale_font(10), "bold")
        
        tk.Label(
            header,
            text="⚙️ PARAMETERS",
            font=header_font,
            bg=self.COLORS['secondary'],
            fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=4)
        
        display = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        display.pack(fill=tk.X, padx=5, pady=(5, 0))
        
        inner = ttk.Frame(display)
        inner.pack(fill=tk.X, padx=8, pady=8)
        
        metadata_status = "✓ From metadata" if self.has_metadata else "⚠ Fallback"
        self._add_param_section(inner, "Basic Information", [
            ("Pathogen:", self.bacterium, self.COLORS['danger']),
            ("Structure:", self.structure, self.COLORS['purple']),
            ("Mode:", f"{self.mode} particles", self.COLORS['primary']),
            ("Pixel size:", f"{self.pixel_size_um:.6f} µm/px", 
             self.COLORS['success'] if self.has_metadata else self.COLORS['warning']),
            ("Metadata:", metadata_status, 
             self.COLORS['success'] if self.has_metadata else self.COLORS['warning']),
        ])
        
        ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        
        self._add_param_section(inner, "Segmentation", [
            ("Invert:", "ON" if self.invert_image else "OFF",
             self.COLORS['success'] if self.invert_image else self.COLORS['gray']),
            ("Gaussian σ:", f"{self.params['gaussian_sigma']:.1f}", None),
            ("Threshold:", self.threshold_mode.upper(), self.COLORS['primary']),
            ("Morph kernel:", f"{self.morph_kernel_size}x{self.morph_kernel_size}", None),
            ("Morph iter:", str(int(self.morph_iterations)), None),
        ])
        
        ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        
        um2_per_px2 = self.pixel_size_um ** 2
        min_area_um2 = self.params['min_area'] * um2_per_px2
        max_area_um2 = self.params['max_area'] * um2_per_px2
        
        self._add_param_section(inner, "Filtering & Morphology", [
            ("Min area:", f"{self.params['min_area']:.0f} px ({min_area_um2:.2f} µm²)", None),
            ("Max area:", f"{self.params['max_area']:.0f} px ({max_area_um2:.2f} µm²)", None),
            ("Dilate iter:", str(self.params["dilate_iterations"]), None),
            ("Erode iter:", str(self.params["erode_iterations"]), None),
        ])
        
        ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        
        self._add_param_section(inner, "Shape Filters (Auto-applied)", [
            ("Circularity:", f"{self.min_circularity:.2f} - {self.max_circularity:.2f}", None),
            ("Aspect ratio:", f"{self.min_aspect_ratio:.2f} - {self.max_aspect_ratio:.2f}", None),
            ("Solidity:", f"≥ {self.min_solidity:.2f}", None),
        ])
    
    def _add_param_section(self, parent: ttk.Frame, title: str,
                        params: List[Tuple[str, str, Optional[str]]]):
        """Add a parameter display section with scaled fonts"""
        title_font = ("Segoe UI", self.ui_scaler.scale_font(9), "bold")
        label_font = ("Segoe UI", self.ui_scaler.scale_font(8))
        value_font = ("Segoe UI", self.ui_scaler.scale_font(8), "bold")
        
        label_width = self.ui_scaler.scale_dimension(15)
        
        tk.Label(
            parent,
            text=title,
            font=title_font,
            foreground=self.COLORS['header']
        ).pack(anchor="w", pady=(0, 4))
        
        for label_text, value_text, color in params:
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=1)
            
            tk.Label(
                frame,
                text=label_text,
                font=label_font,
                foreground="gray",
                width=label_width,
                anchor="w"
            ).pack(side=tk.LEFT)
            
            padx = self.ui_scaler.scale_dimension(6)
            
            if color:
                value_label = tk.Label(
                    frame,
                    text=value_text,
                    font=value_font,
                    foreground="white",
                    bg=color,
                    padx=padx,
                    pady=1,
                    relief=tk.RAISED
                )
            else:
                value_label = tk.Label(
                    frame,
                    text=value_text,
                    font=value_font,
                    foreground=self.COLORS['header']
                )
            value_label.pack(side=tk.LEFT)
            
            self.param_labels[label_text] = value_label

    def _create_target_analysis_section(self, parent: ttk.Frame):
        """Create target analysis section with responsive sizing"""
        header_height = self.ui_scaler.scale_dimension(26)
        header = tk.Frame(parent, bg=self.COLORS['warning'], height=header_height)
        header.pack(fill=tk.X, pady=(8, 0))
        header.pack_propagate(False)
        
        header_font = ("Segoe UI", self.ui_scaler.scale_font(9), "bold")
        
        tk.Label(
            header,
            text="🎯 TARGET ANALYSIS",
            font=header_font,
            bg=self.COLORS['warning'],
            fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=3)
        
        display = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        display.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        label_font = ("Segoe UI", self.ui_scaler.scale_font(8))
        wraplength = self.ui_scaler.scale_dimension(360)
        
        self.target_analysis_label = tk.Label(
            display,
            text="Click on image to analyze a particle",
            font=label_font,
            foreground="gray",
            justify=tk.LEFT,
            wraplength=wraplength,
            bg="white",
            anchor="w",
            padx=8,
            pady=8
        )
        self.target_analysis_label.pack(fill=tk.BOTH, expand=True)
    
    def _create_histogram_section(self, parent: ttk.Frame):
        """Create histogram section with optimal sizing"""
        header = tk.Frame(parent, bg=self.COLORS['info'], height=28)
        header.pack(fill=tk.X, pady=(5, 0))
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="📊 AREA DISTRIBUTION",
            font=("Segoe UI", 10, "bold"),
            bg=self.COLORS['info'],
            fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=4)
        
        canvas_frame = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        
        # Optimal size for 1920x1080
        self.fig_hist = Figure(figsize=(5.2, 3.5), facecolor='white', dpi=80)
        self.ax_hist = self.fig_hist.add_subplot(111)
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, canvas_frame)
        self.canvas_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    def _create_control_panel(self, parent: ttk.Frame):
        """Create slider control panel with responsive sizing"""
        panel = ttk.Frame(parent)
        panel.pack(fill=tk.X, pady=(5, 0))
        
        header = tk.Frame(panel, bg=self.COLORS['secondary'], height=30)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="🎚️ ADJUST PARAMETERS",
            font=("Segoe UI", 11, "bold"),
            bg=self.COLORS['secondary'],
            fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=5)
        
        slider_frame = ttk.Frame(panel, relief=tk.SUNKEN, borderwidth=1)
        slider_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # ✅ FIX: Calculate size based on available space
        screen_width = self.root.winfo_screenwidth()
        available_width = screen_width * 0.92  # 92% of screen width
        fig_width_inches = available_width / 100  # Convert to inches at 100 DPI
        
        # ✅ FIX: Reduced height to ensure visibility
        self.fig_sliders = Figure(
            figsize=(fig_width_inches, 2.5),  # Reduced from 3.2 to 2.5
            facecolor='#f8f9fa',
            dpi=100
        )
        self.canvas_sliders = FigureCanvasTkAgg(self.fig_sliders, slider_frame)
        self.canvas_sliders.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self._create_sliders()

    def _create_sliders(self):
        """Create parameter sliders with organized layout"""
        
        self.fig_sliders.clear()
        
        # Section headers with better positioning
        self.fig_sliders.text(0.18, 0.97, 'SEGMENTATION', ha='center', va='top',
                            fontsize=9, fontweight='bold', color=self.COLORS['header'])
        self.fig_sliders.text(0.50, 0.97, 'FILTERING & MORPHOLOGY', ha='center', va='top',
                            fontsize=9, fontweight='bold', color=self.COLORS['header'])
        self.fig_sliders.text(0.82, 0.97, 'SHAPE FILTERS', ha='center', va='top',
                            fontsize=9, fontweight='bold', color=self.COLORS['header'])
        
        slider_height = 0.10
        slider_width = 0.25
        
        # Column positions (left edge of each slider)
        col1_x = 0.04   # Left column
        col2_x = 0.36   # Middle column
        col3_x = 0.68   # Right column
        
        # Row positions (bottom edge of each slider)
        row1_y = 0.78
        row2_y = 0.62
        row3_y = 0.46
        row4_y = 0.30
        row5_y = 0.14
        
        # ==================================================
        # COLUMN 1: SEGMENTATION & ASPECT RATIO
        # ==================================================
        
        # Row 1: Gaussian σ
        ax = self.fig_sliders.add_axes((col1_x, row1_y, slider_width, slider_height))
        slider = Slider(ax, 'Gaussian σ', 0.5, 20.0, 
                    valinit=self.params['gaussian_sigma'],
                    color=self.COLORS['primary'])
        slider.on_changed(lambda val: self.update_parameter('gaussian_sigma', val))
        self.sliders['gaussian_sigma'] = slider

        # Row 2: Manual Threshold
        ax = self.fig_sliders.add_axes((col1_x, row2_y, slider_width, slider_height))
        slider = Slider(ax, 'Threshold', 0, 255,
                    valinit=self.manual_threshold,
                    valstep=1, color=self.COLORS['primary'])
        slider.on_changed(lambda val: self.update_threshold('manual_threshold', val))
        self.sliders['manual_threshold'] = slider
        
        # Row 3: Morph Kernel Size
        ax = self.fig_sliders.add_axes((col1_x, row3_y, slider_width, slider_height))
        slider = Slider(ax, 'Morph Kernel', 3, 15,
                    valinit=self.morph_kernel_size,
                    valstep=2, color=self.COLORS['primary'])  # Odd numbers only
        slider.on_changed(lambda val: self.update_morph('morph_kernel_size', val))
        self.sliders['morph_kernel_size'] = slider
        
        # Row 4: Morph Iterations
        ax = self.fig_sliders.add_axes((col1_x, row4_y, slider_width, slider_height))
        slider = Slider(ax, 'Morph Iter', 0, 5,
                    valinit=self.morph_iterations,
                    valstep=1, color=self.COLORS['primary'])
        slider.on_changed(lambda val: self.update_morph('morph_iterations', val))
        self.sliders['morph_iterations'] = slider

        # Row 5: Threshold mode button (Row 5)
        ax_thresh_mode = self.fig_sliders.add_axes((col1_x, row5_y, 0.13, slider_height * 1.5))
        mode_text = f"THRESHOLD\n{self.threshold_mode.upper()}"
        self.btn_thresh_mode = Button(ax_thresh_mode, mode_text,
                                    color=self.COLORS['primary'],
                                    hovercolor=self.COLORS['header'])
        self.btn_thresh_mode.on_clicked(self.cycle_threshold_mode)

        '''
        # Row 3: Min Aspect Ratio
        ax = self.fig_sliders.add_axes((col1_x, row3_y, slider_width, slider_height))
        slider = Slider(ax, 'Min Aspect', 0.1, 5.0,
                    valinit=self.min_aspect_ratio,
                    valstep=0.1, color=self.COLORS['purple'])
        slider.on_changed(lambda val: self.update_shape_filter('min_aspect_ratio', val))
        self.sliders['min_aspect_ratio'] = slider
        
        # Row 4: Max Aspect Ratio
        ax = self.fig_sliders.add_axes((col1_x, row4_y, slider_width, slider_height))
        slider = Slider(ax, 'Max Aspect', 1.0, 20.0,
                    valinit=self.max_aspect_ratio,
                    valstep=0.1, color=self.COLORS['purple'])
        slider.on_changed(lambda val: self.update_shape_filter('max_aspect_ratio', val))
        self.sliders['max_aspect_ratio'] = slider
        '''
        # ==================================================
        # COLUMN 2: FILTERING & MORPHOLOGY
        # ==================================================
        
        # Row 1: Min Area
        ax = self.fig_sliders.add_axes((col2_x, row1_y, slider_width, slider_height))
        slider = Slider(ax, 'Min Area (px)', 10, 5000,
                    valinit=self.params['min_area'],
                    valstep=10, color=self.COLORS['info'])
        slider.on_changed(lambda val: self.update_parameter('min_area', val))
        self.sliders['min_area'] = slider
        
        # Row 2: Max Area
        ax = self.fig_sliders.add_axes((col2_x, row2_y, slider_width, slider_height))
        slider = Slider(ax, 'Max Area (px)', 100, 20000,
                    valinit=self.params['max_area'],
                    valstep=100, color=self.COLORS['info'])
        slider.on_changed(lambda val: self.update_parameter('max_area', val))
        self.sliders['max_area'] = slider
        
        # Row 3: Dilate
        ax = self.fig_sliders.add_axes((col2_x, row3_y, slider_width, slider_height))
        slider = Slider(ax, 'Dilate Iter', 0, 5,
                    valinit=self.params['dilate_iterations'],
                    valstep=1, color=self.COLORS['warning'])
        slider.on_changed(lambda val: self.update_parameter('dilate_iterations', val))
        self.sliders['dilate_iterations'] = slider
        
        # Row 4: Erode
        ax = self.fig_sliders.add_axes((col2_x, row4_y, slider_width, slider_height))
        slider = Slider(ax, 'Erode Iter', 0, 5,
                    valinit=self.params['erode_iterations'],
                    valstep=1, color=self.COLORS['warning'])
        slider.on_changed(lambda val: self.update_parameter('erode_iterations', val))
        self.sliders['erode_iterations'] = slider
        
        # ==================================================
        # COLUMN 3: SHAPE FILTERS
        # ==================================================
        
        # Row 1: Min Circularity
        ax = self.fig_sliders.add_axes((col3_x, row1_y, slider_width, slider_height))
        slider = Slider(ax, 'Min Circular', 0.0, 1.0,
                    valinit=self.min_circularity,
                    valstep=0.01, color=self.COLORS['purple'])
        slider.on_changed(lambda val: self.update_shape_filter('min_circularity', val))
        self.sliders['min_circularity'] = slider
        
        # Row 2: Max Circularity
        ax = self.fig_sliders.add_axes((col3_x, row2_y, slider_width, slider_height))
        slider = Slider(ax, 'Max Circular', 0.0, 1.0,
                    valinit=self.max_circularity,
                    valstep=0.01, color=self.COLORS['purple'])
        slider.on_changed(lambda val: self.update_shape_filter('max_circularity', val))
        self.sliders['max_circularity'] = slider
        
        # Row 3: Min Solidity
        ax = self.fig_sliders.add_axes((col3_x, row3_y, slider_width, slider_height))
        slider = Slider(ax, 'Min Solidity', 0.0, 1.0,
                    valinit=self.min_solidity,
                    valstep=0.01, color=self.COLORS['purple'])
        slider.on_changed(lambda val: self.update_shape_filter('min_solidity', val))
        self.sliders['min_solidity'] = slider
        
        # ==================================================
        # CONTROL BUTTONS (Bottom rows)
        # ==================================================
        
        # INVERT button
        invert_color = self.COLORS['success'] if self.invert_image else self.COLORS['gray']
        invert_text = f'INVERT\n{"ON" if self.invert_image else "OFF"}'
        
        ax_invert = self.fig_sliders.add_axes((col2_x, row5_y, 0.13, slider_height * 1.5))
        self.btn_invert = Button(ax_invert, invert_text, color=invert_color,
                                hovercolor=self.COLORS['success'])
        self.btn_invert.on_clicked(self.toggle_invert)
        
        # APPLY SUGGESTIONS button
        ax_apply = self.fig_sliders.add_axes((col2_x + 0.15, row5_y, 0.13, slider_height * 1.5))
        self.btn_apply = Button(ax_apply, "APPLY\nSUGGESTIONS",
                            color=self.COLORS['primary'],
                            hovercolor='#5dade2')
        self.btn_apply.on_clicked(self.apply_suggestions)



    def process_image(self):
        """Process image - MATCHES dev2a.py segmentation approach"""
        img = self.original_image.copy()
        
        # Apply inversion if needed
        if self.invert_image:
            img = cv2.bitwise_not(img)
        
        # ============================================================
        # BACKGROUND SUBTRACTION APPROACH (matches dev2a.py)
        # ============================================================
        
        # Step 1: Create background model with Gaussian blur
        bg = cv2.GaussianBlur(
            img, (0, 0),
            sigmaX=self.params["gaussian_sigma"],
            sigmaY=self.params["gaussian_sigma"]
        )
        
        # Step 2: Subtract original from background to enhance particles
        enhanced = cv2.subtract(bg, img)
        
        # Step 3: Additional smoothing
        enhanced_blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        self.processed_image = enhanced_blur
        
        # ============================================================
        # OTSU THRESHOLDING (matches dev2a.py)
        # ============================================================
        
        if self.threshold_mode == "otsu":
            _, binary = cv2.threshold(
                enhanced_blur, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif self.threshold_mode == "manual":
            _, binary = cv2.threshold(
                enhanced_blur,
                self.manual_threshold,
                255,
                cv2.THRESH_BINARY
            )
        elif self.threshold_mode == "adaptive":
            binary = cv2.adaptiveThreshold(
                enhanced_blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
        else:
            _, binary = cv2.threshold(
                enhanced_blur, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        # ============================================================
        # MORPHOLOGICAL OPERATIONS (matches dev2a.py)
        # ============================================================
        kernel_size = int(self.morph_kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd size
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Close operation (combines dilation + erosion)
        if int(self.morph_iterations) > 0:
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_CLOSE, kernel,
                iterations=int(self.morph_iterations)  # Hardcoded in dev2a
            )

        # Dilate
        if int(self.params["dilate_iterations"]) > 0:
            binary = cv2.dilate(
                binary, kernel,
                iterations=int(self.params["dilate_iterations"])
            )
        
        # Erode
        if int(self.params["erode_iterations"]) > 0:
            binary = cv2.erode(
                binary, kernel,
                iterations=int(self.params["erode_iterations"])
            )
        
        self.binary_mask = binary
        
        # ============================================================
        # CONTOUR DETECTION AND FILTERING (matches dev2a.py)
        # ============================================================
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        self.contours = []
        self.contour_areas = []
        
        # Calculate image-based thresholds
        um2_per_px2 = self.pixel_size_um ** 2
        H, W = img.shape[:2]
        img_area_px = float(H * W)
        max_big_area_px = 0.25 * img_area_px  # MAX_FRACTION_OF_IMAGE_AREA
        
        # Filter contours
        for cnt in contours:
            area_px = float(cv2.contourArea(cnt))
            
            # Skip zero-area contours
            if area_px <= 0:
                continue
            
            # Area range filter
            if not (self.params["min_area"] <= area_px <= self.params["max_area"]):
                continue
            
            # Oversized particle filter
            if area_px >= max_big_area_px:
                continue
            
            # ============================================================
            # SHAPE FILTERS (matches dev2a.py)
            # ============================================================
            
            # Circularity
            perimeter = float(cv2.arcLength(cnt, True))
            if perimeter > 0:
                circularity = (4 * np.pi * area_px) / (perimeter ** 2)
            else:
                circularity = 0.0
            
            if not (self.min_circularity <= circularity <= self.max_circularity):
                continue
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0.0
            
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                continue
            
            # Solidity
            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            solidity = area_px / hull_area if hull_area > 0 else 0.0
            
            if solidity < self.min_solidity:
                continue
            
            # All filters passed - accept this contour
            self.contours.append(cnt)
            self.contour_areas.append(area_px)



    def update_visualization(self):
        """Update all visualizations"""
        self.process_image()
        self._update_image_display()
        self._update_histogram()
        self._update_param_displays()
        self.contour_count_label.config(text=f"Contours: {len(self.contours)}")
    
    def _update_image_display(self):
        """Update main image display"""
        self.ax_image.clear()
        
        if len(self.original_image.shape) == 2:
            display = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB)
        else:
            display = self.original_image.copy()
        
        cv2.drawContours(display, self.contours, -1, (0, 255, 0), 2)
        
        self.ax_image.imshow(display)
        self.ax_image.set_title("Original + Contours", fontsize=12,
                               fontweight='bold', pad=10)
        self.ax_image.axis("off")
        
        self.canvas_image.draw()
    
    def _update_histogram(self):
        """Update area distribution histogram"""
        self.ax_hist.clear()
        
        if not self.contour_areas:
            self.ax_hist.text(0.5, 0.5, "No contours detected",
                            ha='center', va='center', fontsize=10, color='gray')
            self.ax_hist.set_xlim(0, 1)
            self.ax_hist.set_ylim(0, 1)
        else:
            areas_px = np.array(self.contour_areas)
            um2_per_px2 = self.pixel_size_um ** 2
            areas_um2 = areas_px * um2_per_px2
            
            self.ax_hist.hist(areas_um2, bins=30, color=self.COLORS['primary'],
                            alpha=0.7, edgecolor='black')
            
            median = float(np.median(areas_um2))
            mean = float(np.mean(areas_um2))
            min_area = float(np.min(areas_um2))
            max_area = float(np.max(areas_um2))
            
            self.ax_hist.axvline(median, color='orange', linestyle='--',
                               linewidth=2, label=f'Median: {median:.2f} µm²')
            self.ax_hist.axvline(mean, color='red', linestyle='--',
                               linewidth=2, label=f'Mean: {mean:.2f} µm²')
            self.ax_hist.axvline(min_area, color='green', linestyle=':',
                               linewidth=1.5, label=f'Min: {min_area:.2f} µm²')
            self.ax_hist.axvline(max_area, color='purple', linestyle=':',
                               linewidth=1.5, label=f'Max: {max_area:.2f} µm²')
            
            self.ax_hist.set_xlabel("Area (µm²)", fontsize=9)
            self.ax_hist.set_ylabel("Count", fontsize=9)
            self.ax_hist.set_title(f"Distribution (n={len(areas_um2)})",
                                  fontsize=10, fontweight='bold')
            self.ax_hist.legend(fontsize=7, loc='upper right')
            self.ax_hist.grid(True, alpha=0.3)
        
        self.canvas_hist.draw()
    
    def _update_param_displays(self):
        """Update parameter display values"""
        invert_label = self.param_labels["Invert:"]
        invert_text = "ON" if self.invert_image else "OFF"
        invert_color = self.COLORS['success'] if self.invert_image else self.COLORS['gray']
        invert_label.config(text=invert_text, bg=invert_color)
        
        um2_per_px2 = self.pixel_size_um ** 2
        min_area_um2 = self.params['min_area'] * um2_per_px2
        max_area_um2 = self.params['max_area'] * um2_per_px2
        
        updates = {
            "Gaussian σ:": f"{self.params['gaussian_sigma']:.1f}",
            "Min area:": f"{self.params['min_area']:.0f} px ({min_area_um2:.2f} µm²)",
            "Max area:": f"{self.params['max_area']:.0f} px ({max_area_um2:.2f} µm²)",
            "Dilate iter:": str(int(self.params["dilate_iterations"])),
            "Erode iter:": str(int(self.params["erode_iterations"])),
        }
        
        for key, value in updates.items():
            if key in self.param_labels:
                self.param_labels[key].config(text=value)




    def update_parameter(self, param_name: str, value: float):
        """Update a parameter and refresh visualization"""
        self.params[param_name] = value
        self.update_visualization()
    
    def update_shape_filter(self, filter_name: str, value: float):
        """Update shape filter attribute and refresh"""
        setattr(self, filter_name, value)
        self.update_visualization()
        
        # Update display
        if filter_name in ["min_circularity", "max_circularity"]:
            key = "Circularity:"
            text = f"{self.min_circularity:.2f} - {self.max_circularity:.2f}"
        elif filter_name in ["min_aspect_ratio", "max_aspect_ratio"]:
            key = "Aspect ratio:"
            text = f"{self.min_aspect_ratio:.2f} - {self.max_aspect_ratio:.2f}"
        elif filter_name == "min_solidity":
            key = "Solidity:"
            text = f"≥ {self.min_solidity:.2f}"
        
        if key in self.param_labels:
            self.param_labels[key].config(text=text)




    def toggle_invert(self, event):
        """Toggle image inversion"""
        self.invert_image = not self.invert_image
        
        invert_text = f'INVERT\n{"ON" if self.invert_image else "OFF"}'
        invert_color = self.COLORS['success'] if self.invert_image else self.COLORS['gray']
        self.btn_invert.label.set_text(invert_text)
        self.btn_invert.color = invert_color
        
        self.update_visualization()
    
    def on_image_click(self, event):
        """Handle click on image to analyze particle"""
        if event.inaxes != self.ax_image or event.xdata is None or event.ydata is None:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        
        clicked_contour = None
        for cnt in self.contours:
            if cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0:
                clicked_contour = cnt
                break
        
        if clicked_contour is None:
            self.target_analysis_label.config(
                text="No particle found at click location",
                foreground="red"
            )
            return
        
        self._analyze_particle(clicked_contour)
    
    def _analyze_particle(self, contour: np.ndarray):
        """Analyze a specific particle"""
        area_px = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        circularity = 4 * np.pi * area_px / (perimeter ** 2) if perimeter > 0 else 0
        
        um2_per_px2 = self.pixel_size_um ** 2
        area_um2 = area_px * um2_per_px2
        
        suggestions = self._generate_suggestions(area_px, circularity, aspect_ratio)
        self.current_suggestions = suggestions
        
        analysis_text = (
            f"🎯 Target Particle Analysis:\n\n"
            f"Area: {area_px:.1f} px² ({area_um2:.2f} µm²)\n"
            f"Perimeter: {perimeter:.1f} px\n"
            f"Aspect Ratio: {aspect_ratio:.2f}\n"
            f"Circularity: {circularity:.3f}\n\n"
            f"📊 Suggestions:\n"
        )
        
        for param, value in suggestions.items():
            if 'area' in param:
                value_um2 = value * um2_per_px2
                analysis_text += f"• {param}: {value} px ({value_um2:.2f} µm²)\n"
            else:
                analysis_text += f"• {param}: {value}\n"
        
        self.target_analysis_label.config(text=analysis_text, foreground="black")
    
    def _generate_suggestions(self, area: float, circularity: float,
                             aspect_ratio: float) -> Dict[str, Any]:
        """Generate parameter suggestions"""
        suggestions: Dict[str, Any] = {}
        
        suggestions["min_area"] = max(10, int(area * 0.3))
        suggestions["max_area"] = min(20000, int(area * 3.0))
        
        if circularity < 0.6:
            suggestions["dilate_iterations"] = min(3, self.params["dilate_iterations"] + 1)
            suggestions["erode_iterations"] = min(3, self.params["erode_iterations"] + 1)
        
        if aspect_ratio > 2.5 or aspect_ratio < 0.4:
            suggestions["gaussian_sigma"] = min(10.0, self.params["gaussian_sigma"] + 1.0)
        
        return suggestions
    
    def apply_suggestions(self, event):
        """Apply suggested parameters"""
        if not self.current_suggestions:
            print("⚠ No suggestions available. Click on a particle first.")
            return
        
        print("\n✅ Applying suggestions:")
        for param, value in self.current_suggestions.items():
            if param in self.params:
                old_value = self.params[param]
                self.params[param] = value
                if param in self.sliders:
                    self.sliders[param].set_val(value)
                print(f"   {param}: {old_value} → {value}")
        
        self.update_visualization()
    
    def load_new_image(self):
        """Load a new image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            new_path = Path(file_path)
            
            if not validate_path_encoding(new_path):
                messagebox.showerror(
                    "Error",
                    f"Path contains problematic characters:\n{file_path}"
                )
                return
            
            self.original_image = self._load_image(new_path)
            self.image_path = new_path
            self.pixel_size_um, self.has_metadata = self._load_pixel_size()
            
            self.update_visualization()
            
            messagebox.showinfo(
                "Success",
                f"Image loaded!\n\n{new_path.name}\n"
                f"Pixel size: {self.pixel_size_um:.6f} µm/px\n"
                f"Source: {'Metadata' if self.has_metadata else 'Fallback'}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            print(f"❌ Error loading image: {e}")
    

# ==================================================
# SECTION 6: Main Menu (Pathogen Config Manager)
# ==================================================

class PathogenConfigManager:
    """Main menu for managing pathogen configurations"""
    
    PATHOGENS = {
        "Proteus mirabilis": {
            "config_key": "proteus_mirabilis",
            "description": "Rod-shaped, flagellated bacterium",
            "common_in": "Catheter-associated infections"
        },
        "Klebsiella pneumoniae": {
            "config_key": "klebsiella_pneumoniae",
            "description": "Gram-negative, encapsulated bacterium",
            "common_in": "Healthcare-associated infections"
        },
        "Streptococcus mitis": {
            "config_key": "streptococcus_mitis",
            "description": "Gram-positive cocci in chains",
            "common_in": "Touch contamination"
        }
    }
    
    COLORS = {
        'bg': '#1e1e1e',
        'fg': '#ffffff',
        'accent': '#007acc',
        'button': '#2d2d2d',
        'button_hover': '#3e3e3e',
        'success': '#4ec9b0',
        'warning': '#ce9178',
        'error': '#f48771',
        'header': '#569cd6'
    }
    
    def __init__(self, root: tk.Tk):
        """Initialize the pathogen configuration manager"""
        self.root = root
        self.root.title("🦠 Pathogen Configuration Manager")
        self.root.geometry("900x700")
        self.root.resizable(False, False)
        self.root.configure(bg=self.COLORS['bg'])
        
        self._create_ui()
        self._center_window()
        
    def _center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
    def _create_ui(self):
        """Create the main user interface"""
        main_frame = tk.Frame(self.root, bg=self.COLORS['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self._create_header(main_frame)
        self._create_pathogen_cards(main_frame)
        self._create_footer(main_frame)
        
    def _create_header(self, parent):
        """Create header section for main menu"""
        header_frame = tk.Frame(parent, bg=self.COLORS['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 30))
        
        # Main title
        title = tk.Label(
            header_frame,
            text="🦠 Pathogen Configuration Manager",
            font=('Segoe UI', 24, 'bold'),
            bg=self.COLORS['bg'],
            fg=self.COLORS['header']
        )
        title.pack(pady=(0, 10))
        
        # Subtitle
        subtitle = tk.Label(
            header_frame,
            text="Configure image analysis parameters for peritoneal dialysis pathogens",
            font=('Segoe UI', 11),
            bg=self.COLORS['bg'],
            fg=self.COLORS['fg']
        )
        subtitle.pack()
        
    def _create_pathogen_cards(self, parent):
        """Create cards for each pathogen"""
        cards_frame = tk.Frame(parent, bg=self.COLORS['bg'])
        cards_frame.pack(fill=tk.BOTH, expand=True)
        
        for pathogen_name, info in self.PATHOGENS.items():
            card = self._create_pathogen_card(cards_frame, pathogen_name, info)
            card.pack(fill=tk.X, pady=(0, 15))
            
    def _create_pathogen_card(self, parent, pathogen_name: str, info: dict):
        """Create a card for a single pathogen"""
        card = tk.Frame(
            parent,
            bg=self.COLORS['button'],
            relief=tk.RAISED,
            borderwidth=1
        )
        
        card.bind('<Enter>', lambda e: card.configure(bg=self.COLORS['button_hover']))
        card.bind('<Leave>', lambda e: card.configure(bg=self.COLORS['button']))
        
        content = tk.Frame(card, bg=self.COLORS['button'])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        left_frame = tk.Frame(content, bg=self.COLORS['button'])
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        name_label = tk.Label(
            left_frame,
            text=f"🔬 {pathogen_name}",
            font=('Segoe UI', 16, 'bold'),
            bg=self.COLORS['button'],
            fg=self.COLORS['success'],
            anchor=tk.W
        )
        name_label.pack(anchor=tk.W)
        
        desc_label = tk.Label(
            left_frame,
            text=info['description'],
            font=('Segoe UI', 10),
            bg=self.COLORS['button'],
            fg=self.COLORS['fg'],
            anchor=tk.W
        )
        desc_label.pack(anchor=tk.W, pady=(5, 0))
        
        common_label = tk.Label(
            left_frame,
            text=f"Common in: {info['common_in']}",
            font=('Segoe UI', 9, 'italic'),
            bg=self.COLORS['button'],
            fg=self.COLORS['warning'],
            anchor=tk.W
        )
        common_label.pack(anchor=tk.W, pady=(3, 0))
        
        config_label = tk.Label(
            left_frame,
            text=f"📄 Config: {info['config_key']}",
            font=('Segoe UI', 8),
            bg=self.COLORS['button'],
            fg=self.COLORS['fg'],
            anchor=tk.W
        )
        config_label.pack(anchor=tk.W, pady=(8, 0))
        
        right_frame = tk.Frame(content, bg=self.COLORS['button'])
        right_frame.pack(side=tk.RIGHT, padx=(20, 0))
        
        seg_btn = tk.Button(
            right_frame,
            text="🎨 Segmentation",
            font=('Segoe UI', 10, 'bold'),
            bg=self.COLORS['accent'],
            fg='white',
            activebackground=self.COLORS['header'],
            activeforeground='white',
            relief=tk.FLAT,
            cursor='hand2',
            padx=15,
            pady=8,
            command=lambda p=pathogen_name: self._launch_segmentation_tuner(p)
        )
        seg_btn.pack(pady=(0, 8))
        
        return card
        
    def _create_footer(self, parent):
        """Create footer with utility buttons"""
        footer_frame = tk.Frame(parent, bg=self.COLORS['bg'])
        footer_frame.pack(fill=tk.X, pady=(30, 0))
        
        separator = tk.Frame(footer_frame, height=2, bg=self.COLORS['accent'])
        separator.pack(fill=tk.X, pady=(0, 15))
        
        buttons_frame = tk.Frame(footer_frame, bg=self.COLORS['bg'])
        buttons_frame.pack()
        
        info_btn = tk.Button(
            buttons_frame,
            text="ℹ️ About",
            font=('Segoe UI', 10),
            bg=self.COLORS['button'],
            fg=self.COLORS['fg'],
            activebackground=self.COLORS['button_hover'],
            activeforeground=self.COLORS['fg'],
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=10,
            command=self._show_about
        )
        info_btn.pack(side=tk.LEFT, padx=5)
        
        exit_btn = tk.Button(
            buttons_frame,
            text="❌ Exit",
            font=('Segoe UI', 10),
            bg=self.COLORS['error'],
            fg='white',
            activebackground='#d67060',
            activeforeground='white',
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=10,
            command=self._exit_application
        )
        exit_btn.pack(side=tk.RIGHT, padx=5)
        
    def _launch_segmentation_tuner(self, pathogen_name: str):
        """Launch segmentation tuner with setup dialog"""
        print(f"\n🎨 Launching segmentation tuner for {pathogen_name}...")
        
        # Create setup dialog
        setup_dialog = tk.Toplevel(self.root)
        setup_dialog.title(f"Tuner Setup - {pathogen_name}")
        setup_dialog.geometry("500x350")
        setup_dialog.resizable(False, False)
        setup_dialog.configure(bg='white')
        setup_dialog.transient(self.root)
        setup_dialog.grab_set()
        
        # Variables
        image_path_var = tk.StringVar()
        structure_var = tk.StringVar(value="bacteria")
        mode_var = tk.StringVar(value="DARK")
        
        # Main frame
        main_frame = ttk.Frame(setup_dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(
            main_frame,
            text=f"🔬 {pathogen_name} Tuner Setup",
            font=("Segoe UI", 12, "bold")
        ).pack(pady=(0, 20))
        
        # Image selection
        ttk.Label(
            main_frame,
            text="1. Select Image",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 5))
        
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Entry(image_frame, textvariable=image_path_var, width=40).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        def browse_image():
            filename = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                    ("All files", "*.*")
                ]
            )
            if filename:
                image_path_var.set(filename)
        
        ttk.Button(image_frame, text="Browse...", command=browse_image).pack(side=tk.LEFT)
        
        # Structure selection
        ttk.Label(
            main_frame,
            text="2. Select Structure",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 5))
        
        structure_frame = ttk.Frame(main_frame)
        structure_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Radiobutton(
            structure_frame,
            text="Bacteria",
            variable=structure_var,
            value="bacteria"
        ).pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Radiobutton(
            structure_frame,
            text="Inclusions",
            variable=structure_var,
            value="inclusions"
        ).pack(side=tk.LEFT)
        
        # Mode selection
        ttk.Label(
            main_frame,
            text="3. Select Mode",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 5))
        
        mode_frame = ttk.Frame(main_frame)
        mode_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Radiobutton(
            mode_frame,
            text="DARK particles",
            variable=mode_var,
            value="DARK"
        ).pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Radiobutton(
            mode_frame,
            text="BRIGHT particles",
            variable=mode_var,
            value="BRIGHT"
        ).pack(side=tk.LEFT)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        def start_tuner():
            if not image_path_var.get():
                messagebox.showerror("Error", "Please select an image file", parent=setup_dialog)
                return
            
            setup_dialog.destroy()
            self.root.destroy()
            
            try:
                tuner_root = tk.Tk()
                
                def return_to_menu():
                    main()
                
                tuner = SegmentationTuner(
                    root=tuner_root,
                    image_path=image_path_var.get(),
                    bacterium=pathogen_name,
                    structure=structure_var.get(),
                    mode=mode_var.get(),
                    return_callback=return_to_menu
                )
                tuner_root.mainloop()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start tuner:\n{str(e)}")
                import traceback
                traceback.print_exc()
        
        ttk.Button(
            button_frame,
            text="❌ Cancel",
            command=setup_dialog.destroy
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="✅ Start Tuner",
            command=start_tuner
        ).pack(side=tk.LEFT)
        
    def _show_about(self):
        """Show about dialog"""
        about_text = """
Peritoneal Dialysis Pathogen Configuration Manager
Version 2.0

This application manages image analysis configurations 
for three common peritoneal dialysis pathogens:

• Proteus mirabilis
• Klebsiella pneumoniae  
• Streptococcus mitis

Features:
🎨 Segmentation Tuner - Adjust image segmentation parameters
📊 Histogram Analysis - Visualize particle distributions
💾 Config Management - Save and apply configurations

© 2026 Pathogen Analysis Suite
        """
        messagebox.showinfo("About", about_text.strip())
        
    def _exit_application(self):
        """Exit the application"""
        if messagebox.askyesno("Confirm Exit", "Are you sure you want to exit?"):
            print("\n👋 Exiting Pathogen Configuration Manager")
            self.root.quit()
            self.root.destroy()


# ==================================================
# SECTION 7: Main Entry Point
# ==================================================

def main():
    """Main entry point"""
    root = tk.Tk()
    app = PathogenConfigManager(root)
    root.mainloop()


if __name__ == "__main__":
    main()