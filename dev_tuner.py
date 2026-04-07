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
from zmq import has

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

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        self.width_scale = screen_width / self.base_width
        self.height_scale = screen_height / self.base_height
        self.scale = min(self.width_scale, self.height_scale)

        print(f" Screen: {screen_width}x{screen_height}")
        print(f" Scale factor: {self.scale:.2f}")

    def scale_font(self, base_size: int) -> int:
        return max(8, int(base_size * self.scale))

    def scale_dimension(self, base_dim: int) -> int:
        return int(base_dim * self.scale)

    def get_window_size(self) -> tuple[int, int]:
        width = int(self.base_width * 0.95)
        height = int(self.base_height * 0.90)
        return width, height


# ==================================================
# SECTION 0.6: Modern Parameter Panel (Tkinter-based)
# ==================================================

class ParameterPanel(ttk.Frame):
    """Modern collapsible parameter panel with sliders and input boxes"""

    def _format_value(self, value: float, resolution: float) -> str:
        try:
            value = float(value)
            if resolution >= 1:
                return f"{int(round(value))}"
            else:
                decimals = len(str(resolution).split('.')[-1])
                return f"{value:.{decimals}f}"
        except (ValueError, TypeError):
            return str(value)

    def __init__(self, parent, tuner_instance):
        super().__init__(parent)
        self.tuner = tuner_instance

        print(f"🐛 ParameterPanel DEBUG: tuner.morph_kernel_size = {self.tuner.morph_kernel_size}")

        self.sliders = {}
        self.value_labels = {}
        self.input_boxes = {}
        self.section_frames = {}

        # Store the INITIAL pathogen-specific values as defaults
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
            'invert_image': self.tuner.invert_image,
            'use_intensity_threshold': self.tuner.use_intensity_threshold,
            'intensity_threshold_value': self.tuner.intensity_threshold_value,
        }

        self.debounce_delay = 150
        self._updating_slider = False
        self.configure(relief=tk.RIDGE, borderwidth=1, width=420)
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

        self.canvas = tk.Canvas(canvas_container, bg='#f8f9fa', highlightthickness=0)

        scrollbar_frame = tk.Frame(canvas_container, bg='#cccccc', width=16)
        scrollbar_frame.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_frame.pack_propagate(False)

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

        self.scrollable_frame = tk.Frame(self.canvas, bg='#f8f9fa')

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.canvas.bind('<Configure>', self._on_canvas_configure)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

        self._create_sections()

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _snap_morph_kernel(self, v: float, min_val: int = 1, max_val: int = 15) -> int:
        v = int(round(v))
        v = max(min_val, min(max_val, v))
        if v % 2 == 0:
            down = v - 1
            up = v + 1
            if down < min_val:
                v = up
            elif up > max_val:
                v = down
            else:
                v = down
        return v

    def _on_mousewheel(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _create_sections(self):
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
                                    self.tuner.morph_kernel_size, 1, 15, 1)

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
                                    self.tuner.params['max_area'], 100, 30000, 100)

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

        tk.Frame(self.scrollable_frame, bg='#f8f9fa', height=20).pack()

    def _create_collapsible_section(self, title: str, color: str, expanded: bool = True):
        container = tk.Frame(self.scrollable_frame, bg='#f8f9fa')
        container.pack(fill=tk.X, padx=8, pady=(8, 2))

        header_frame = tk.Frame(container, bg=color, cursor="hand2", height=32)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        arrow_var = tk.StringVar(value="▼" if expanded else "▶")
        arrow_label = tk.Label(
            header_frame, textvariable=arrow_var,
            font=("Segoe UI", 10, "bold"), bg=color, fg="white", width=2
        )
        arrow_label.pack(side=tk.LEFT, padx=(8, 2))

        title_label = tk.Label(
            header_frame, text=title,
            font=("Segoe UI", 10, "bold"), bg=color, fg="white", anchor="w"
        )
        title_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 8))

        content_frame = tk.Frame(container, bg='white', relief=tk.SUNKEN, borderwidth=1)
        if expanded:
            content_frame.pack(fill=tk.X, padx=2, pady=(0, 2))

        def toggle(event=None):
            if content_frame.winfo_ismapped():
                content_frame.pack_forget()
                arrow_var.set("▶")
            else:
                content_frame.pack(fill=tk.X, padx=2, pady=(0, 2))
                arrow_var.set("▼")

        header_frame.bind("<Button-1>", toggle)
        arrow_label.bind("<Button-1>", toggle)
        title_label.bind("<Button-1>", toggle)

        return content_frame

    def _add_slider_with_input(self, parent, label: str, param_name: str,
                               initial_value: float, min_val: float,
                               max_val: float, resolution: float = 1.0):
        row_frame = ttk.Frame(parent, style='ParamRow.TFrame')
        row_frame.pack(fill='x', padx=10, pady=5)

        label_widget = ttk.Label(
            row_frame, text=f"{label}:",
            font=('Segoe UI', 9), foreground='#E0E0E0', background='#2D2D30'
        )
        label_widget.pack(side='left', padx=(0, 10))

        slider_frame = ttk.Frame(row_frame, style='ParamRow.TFrame')
        slider_frame.pack(side='left', fill='x', expand=True)
        slider_frame.columnconfigure(0, weight=1)

        if param_name == 'morph_kernel_size':
            initial_value = self._snap_morph_kernel(initial_value, min_val=int(min_val), max_val=int(max_val))

        value_label = ttk.Label(
            slider_frame,
            text=self._format_value(initial_value, resolution),
            font=('Consolas', 9, 'bold'), foreground='#4EC9B0',
            background='#2D2D30', width=6
        )
        value_label.grid(row=0, column=1)

        entry_var = tk.StringVar(value=str(int(initial_value) if resolution >= 1 else initial_value))

        updating = {'flag': False}

        slider = ttk.Scale(
            slider_frame, from_=min_val, to=max_val,
            orient=tk.HORIZONTAL, style='Custom.Horizontal.TScale'
        )
        slider.grid(row=0, column=0, sticky='ew', padx=(0, 10))

        self.sliders[param_name] = slider
        self.value_labels[param_name] = value_label

        slider.set(initial_value)

        def on_slider_change(v):
            if updating['flag']:
                return
            updating['flag'] = True
            try:
                self._on_slider_change(param_name, v, resolution, entry_var)
            finally:
                updating['flag'] = False

        slider.configure(command=on_slider_change)

        entry = ttk.Entry(row_frame, textvariable=entry_var, width=8)
        entry.pack(side='right', padx=(10, 0))

        def on_entry_change(*args):
            if updating['flag']:
                return
            updating['flag'] = True
            try:
                value_str = entry_var.get().strip()
                if not value_str:
                    return

                value = float(value_str)

                if not (min_val <= value <= max_val):
                    return

                if resolution >= 1:
                    value = int(round(value / resolution) * resolution)
                else:
                    value = round(value / resolution) * resolution

                if param_name == 'morph_kernel_size':
                    value = self._snap_morph_kernel(value, min_val=int(min_val), max_val=int(max_val))

                if param_name in self.value_labels:
                    self.value_labels[param_name].config(
                        text=self._format_value(value, resolution)
                    )

                params_dict_keys = ['gaussian_sigma', 'min_area', 'max_area',
                                    'dilate_iterations', 'erode_iterations']
                if param_name in params_dict_keys:
                    self.tuner.params[param_name] = value
                elif hasattr(self.tuner, param_name):
                    setattr(self.tuner, param_name, value)

                # Sync intensity_threshold_value when in intensity mode
                if param_name == 'manual_threshold' and getattr(self.tuner, 'use_intensity_threshold', False):
                    self.tuner.intensity_threshold_value = float(value)

                slider.set(value)

                entry_var.set(str(int(value)) if resolution >= 1 else str(value))

                if not hasattr(self, 'update_timers'):
                    self.update_timers = {}
                if param_name in self.update_timers:
                    self.after_cancel(self.update_timers[param_name])
                self.update_timers[param_name] = self.after(
                    self.debounce_delay,
                    lambda p=param_name: self._execute_visualization_update(p)
                )

            except ValueError:
                pass
            finally:
                updating['flag'] = False

        entry_var.trace_add('write', on_entry_change)
        entry.bind('<Return>', lambda e: on_entry_change())
        entry.bind('<FocusOut>', lambda e: on_entry_change())

    def _add_threshold_controls(self, parent):
        """Add threshold mode controls — includes INTENSITY mode for pipeline compatibility"""
        container = tk.Frame(parent, bg='white')
        container.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(
            container, text="Threshold Method:",
            font=("Segoe UI", 9), bg='white', fg='#555', anchor='w'
        ).pack(anchor="w")

        # Threshold mode button — cycles through 4 modes
        self.thresh_mode_btn = tk.Button(
            container,
            text=f"{self.tuner.threshold_mode.upper()}",
            font=("Segoe UI", 9, "bold"),
            bg=SegmentationTuner.COLORS['primary'],
            fg="white", relief=tk.FLAT, cursor="hand2", pady=6,
            command=self._cycle_threshold_mode
        )
        self.thresh_mode_btn.pack(fill=tk.X, pady=2)

        # Intensity threshold description label
        self.intensity_info_label = tk.Label(
            container,
            text="⚠ INTENSITY mode: pixels DARKER than threshold → foreground\n"
                 "(matches pipeline use_intensity_threshold=True)",
            font=("Segoe UI", 8), bg='white', fg='#e74c3c',
            anchor='w', justify=tk.LEFT
        )
        if self.tuner.threshold_mode == "intensity":
            self.intensity_info_label.pack(fill=tk.X, pady=2)

        # Manual/Intensity threshold slider container
        self.manual_threshold_container = tk.Frame(container, bg='white')
        if self.tuner.threshold_mode in ("manual", "intensity"):
            self.manual_threshold_container.pack(fill=tk.X, pady=4)

        self._add_slider_with_input(
            self.manual_threshold_container, "Threshold",
            "manual_threshold",
            self.tuner.manual_threshold, 0, 255, 1
        )

    def _cycle_threshold_mode(self):
        """Cycle through threshold modes: otsu → manual → adaptive → intensity"""
        modes = ["otsu", "manual", "adaptive", "intensity"]
        current_idx = modes.index(self.tuner.threshold_mode) if self.tuner.threshold_mode in modes else 0
        next_mode = modes[(current_idx + 1) % len(modes)]

        self.tuner.threshold_mode = next_mode
        self.thresh_mode_btn.config(text=next_mode.upper())

        if next_mode == "intensity":
            self.tuner.use_intensity_threshold = True
            self.tuner.intensity_threshold_value = float(self.tuner.manual_threshold)
            self.manual_threshold_container.pack(fill=tk.X, pady=4)
            self.intensity_info_label.pack(fill=tk.X, pady=2)
        else:
            self.tuner.use_intensity_threshold = False
            self.intensity_info_label.pack_forget()
            if next_mode == "manual":
                self.manual_threshold_container.pack(fill=tk.X, pady=4)
            else:
                self.manual_threshold_container.pack_forget()

        self.tuner.update_visualization()

    def _create_control_buttons(self):
        control_section = tk.Frame(self.scrollable_frame, bg='#f8f9fa')
        control_section.pack(fill=tk.X, padx=8, pady=10)

        invert_color = SegmentationTuner.COLORS['success'] if self.tuner.invert_image else SegmentationTuner.COLORS['gray']
        self.invert_btn = tk.Button(
            control_section,
            text=f"INVERT: {'ON' if self.tuner.invert_image else 'OFF'}",
            font=("Segoe UI", 9, "bold"), bg=invert_color, fg="white",
            relief=tk.RAISED, cursor="hand2", pady=8,
            command=self._toggle_invert
        )
        self.invert_btn.pack(fill=tk.X, pady=3)

        apply_btn = tk.Button(
            control_section,
            text="✨ APPLY SUGGESTIONS",
            font=("Segoe UI", 9, "bold"),
            bg=SegmentationTuner.COLORS['warning'], fg="white",
            relief=tk.RAISED, cursor="hand2", pady=8,
            command=self._apply_suggestions
        )
        apply_btn.pack(fill=tk.X, pady=3)

        reset_btn = tk.Button(
            control_section,
            text="🔄 RESET TO PRESET",
            font=("Segoe UI", 9, "bold"),
            bg=SegmentationTuner.COLORS['danger'], fg="white",
            relief=tk.RAISED, cursor="hand2", pady=8,
            command=self._reset_to_default
        )
        reset_btn.pack(fill=tk.X, pady=3)

    def _on_slider_change(self, param_name: str, value: str, step: float, input_var: tk.StringVar):
        if hasattr(self, '_updating_slider') and self._updating_slider:
            return

        try:
            numeric_value = float(value)

            if step >= 1:
                numeric_value = int(round(numeric_value / step) * step)
            else:
                numeric_value = round(numeric_value / step) * step

            if param_name == 'morph_kernel_size':
                numeric_value = self._snap_morph_kernel(numeric_value, min_val=1, max_val=15)

            params_dict_keys = ['gaussian_sigma', 'min_area', 'max_area',
                                'dilate_iterations', 'erode_iterations']

            if param_name in params_dict_keys:
                current_value = self.tuner.params[param_name]
            elif hasattr(self.tuner, param_name):
                current_value = getattr(self.tuner, param_name)
            else:
                print(f"⚠️ Warning: self.tuner.{param_name} does not exist")
                return

            if numeric_value == current_value:
                return

            if param_name in params_dict_keys:
                self.tuner.params[param_name] = numeric_value
            elif hasattr(self.tuner, param_name):
                setattr(self.tuner, param_name, numeric_value)

            # Sync intensity_threshold_value when in intensity mode
            if param_name == 'manual_threshold' and getattr(self.tuner, 'use_intensity_threshold', False):
                self.tuner.intensity_threshold_value = float(numeric_value)

            print(f"🔄 {param_name}: {current_value} → {numeric_value}")

            if param_name in self.value_labels:
                formatted = self._format_value(numeric_value, step)
                self.value_labels[param_name].config(text=formatted)

            entry_value = str(int(numeric_value)) if step >= 1 else str(numeric_value)
            if input_var.get() != entry_value:
                input_var.set(entry_value)

            if param_name in self.sliders:
                current_slider_val = self.sliders[param_name].get()
                if abs(float(current_slider_val) - numeric_value) > 0.01:
                    self._updating_slider = True
                    self.sliders[param_name].set(numeric_value)
                    self._updating_slider = False

            if not hasattr(self, 'update_timers'):
                self.update_timers = {}

            if param_name in self.update_timers:
                self.after_cancel(self.update_timers[param_name])

            self.update_timers[param_name] = self.after(
                self.debounce_delay,
                lambda p=param_name: self._execute_visualization_update(p)
            )

        except (ValueError, AttributeError, TypeError) as e:
            print(f"❌ Error in _on_slider_change for {param_name}: {e}")
            import traceback
            traceback.print_exc()

    def _execute_visualization_update(self, param_name: str):
        try:
            if param_name in self.update_timers:
                del self.update_timers[param_name]

            if hasattr(self.tuner, 'update_visualization'):
                self.tuner.update_visualization()
            elif hasattr(self.tuner, 'update_preview'):
                self.tuner.update_preview()
            elif hasattr(self.tuner, 'process_frame'):
                self.tuner.process_frame()

        except Exception as e:
            print(f"❌ Error updating visualization for {param_name}: {e}")

    def _toggle_invert(self):
        self.tuner.invert_image = not self.tuner.invert_image

        invert_color = SegmentationTuner.COLORS['success'] if self.tuner.invert_image else SegmentationTuner.COLORS['gray']
        self.invert_btn.config(
            text=f"INVERT: {'ON' if self.tuner.invert_image else 'OFF'}",
            bg=invert_color
        )

        self.tuner.update_visualization()

    def _apply_suggestions(self):
        if not self.tuner.current_suggestions:
            messagebox.showinfo("Info", "Click on a particle first to get suggestions",
                                parent=self.winfo_toplevel())
            return

        self.tuner.apply_suggestions()


    def _reset_to_default(self):
        pathogen_name = getattr(self.tuner, 'current_pathogen', "preset")

        confirm = messagebox.askyesno(
            "Reset Parameters",
            f"Reset all parameters to {pathogen_name.upper()} preset?\n\n"
            f"This will discard ALL current changes.",
            parent=self.winfo_toplevel()
        )

        if not confirm:
            return

        config_key = pathogen_name.lower().replace(' ', '_').replace('.', '')
        config_file = Path("bacteria_configs") / f"{config_key}.json"

        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                if "config" in json_data:
                    config_data = json_data["config"]
                else:
                    config_data = json_data

                um2_per_px2 = self.tuner.pixel_size_um ** 2

                # Determine threshold mode from config
                use_intensity = bool(config_data.get('use_intensity_threshold', False))
                if use_intensity:
                    threshold_mode = 'intensity'
                    threshold_val = float(config_data.get('intensity_threshold', 80.0))
                    self.tuner.use_intensity_threshold = True
                    self.tuner.intensity_threshold_value = threshold_val
                else:
                    threshold_mode = config_data.get('threshold_mode', 'otsu')
                    threshold_val = config_data.get('manual_threshold', 127)
                    self.tuner.use_intensity_threshold = False

                slider_mappings = {
                    'gaussian_sigma': config_data.get('gaussian_sigma', 2.0),
                    'manual_threshold': threshold_val,
                    'morph_kernel_size': config_data.get('morph_kernel_size', 3),
                    'morph_iterations': config_data.get('morph_iterations', 1),
                    'min_area': config_data.get('min_area_um2', 3.0) / um2_per_px2,
                    'max_area': config_data.get('max_area_um2', 100.0) / um2_per_px2,
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
                self.tuner.threshold_mode = threshold_mode
                self.thresh_mode_btn.config(text=threshold_mode.upper())

                if threshold_mode in ("manual", "intensity"):
                    self.manual_threshold_container.pack(fill=tk.X, pady=4)
                else:
                    self.manual_threshold_container.pack_forget()

                if threshold_mode == "intensity":
                    self.intensity_info_label.pack(fill=tk.X, pady=2)
                else:
                    self.intensity_info_label.pack_forget()

                # Reset invert
                invert_value = config_data.get('invert_image', False)
                self.tuner.invert_image = invert_value
                invert_color = SegmentationTuner.COLORS['success'] if invert_value else SegmentationTuner.COLORS['gray']
                self.invert_btn.config(
                    text=f"INVERT: {'ON' if invert_value else 'OFF'}",
                    bg=invert_color
                )

                self.tuner.update_visualization()

                messagebox.showinfo("Reset Complete",
                                    f"Parameters reset to {pathogen_name.upper()} preset from config file!",
                                    parent=self.winfo_toplevel())
                return

            except Exception as e:
                print(f"❌ Failed to load config file for reset: {e}")
                import traceback
                traceback.print_exc()

        # FALLBACK: Use stored default_params
        print(f"ℹ️ No config file found, using default_params stored at initialization")

        for param_name, default_value in self.default_params.items():
            if param_name in self.sliders:
                self.sliders[param_name].set(default_value)
            elif param_name == 'threshold_mode':
                self.tuner.threshold_mode = default_value
                self.thresh_mode_btn.config(text=default_value.upper())
                if default_value in ("manual", "intensity"):
                    self.manual_threshold_container.pack(fill=tk.X, pady=4)
                else:
                    self.manual_threshold_container.pack_forget()
                if default_value == "intensity":
                    self.intensity_info_label.pack(fill=tk.X, pady=2)
                else:
                    self.intensity_info_label.pack_forget()
            elif param_name == 'invert_image':
                self.tuner.invert_image = default_value
                invert_color = SegmentationTuner.COLORS['success'] if default_value else SegmentationTuner.COLORS['gray']
                self.invert_btn.config(
                    text=f"INVERT: {'ON' if default_value else 'OFF'}",
                    bg=invert_color
                )
            elif param_name == 'use_intensity_threshold':
                self.tuner.use_intensity_threshold = default_value
            elif param_name == 'intensity_threshold_value':
                self.tuner.intensity_threshold_value = default_value

        self.tuner.update_visualization()

        messagebox.showinfo("Reset Complete",
                            f"Parameters reset to {pathogen_name.upper()} initial values!",
                            parent=self.winfo_toplevel())


# ==================================================
# SECTION 1: Configuration Data Classes
# ==================================================

@property
def min_area_px(self) -> float:
    um2_per_px2 = 0.012
    return self.min_area_um2 / um2_per_px2

@property
def max_area_px(self) -> float:
    um2_per_px2 = 0.012
    return self.max_area_um2 / um2_per_px2


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
    if bacteria_type not in _CONFIGS:
        print(f"[WARN] Unknown bacteria type '{bacteria_type}', using default")
        return DEFAULT_CONFIG
    return _CONFIGS[bacteria_type]


# ==================================================
# SECTION 2: Config File Manager (AST-based)
# ==================================================

class ConfigFileManager:
    def __init__(self, config_file: Path):
        self.config_file = config_file
        self.tree: Optional[ast.Module] = None
        self.source: Optional[str] = None

    def load(self) -> bool:
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.source = f.read()
            self.tree = ast.parse(self.source)
            return True
        except Exception as e:
            print(f"❌ Failed to load config file: {e}")
            return False

    def find_config_assignment(self, var_name: str) -> Optional[Tuple[int, ast.Assign]]:
        if self.tree is None:
            return None
        for idx, node in enumerate(self.tree.body):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        return idx, node
        return None

    def create_config_assignment(self, var_name: str, config_data: dict) -> ast.Assign:
        keywords = []
        for key, value in config_data.items():
            value_node = ast.Constant(value=value)
            keywords.append(ast.keyword(arg=key, value=value_node))

        config_call = ast.Call(
            func=ast.Name(id='SegmentationConfig', ctx=ast.Load()),
            args=[], keywords=keywords
        )
        assignment = ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=config_call
        )
        return assignment

    def update_config(self, var_name: str, config_data: dict) -> bool:
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
        result = self.find_config_assignment('DEFAULT')
        return result[0] if result else None

    def save(self, backup: bool = True) -> bool:
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
        if self.tree is None or astor is None:
            return False
        try:
            compile(astor.to_source(self.tree), str(self.config_file), 'exec')
            return True
        except SyntaxError as e:
            print(f"❌ Syntax error in generated code: {e}")
            return False


def config_to_dict(config: SegmentationConfig) -> dict:
    return {
        field.name: getattr(config, field.name)
        for field in fields(config)
    }


def update_bacteria_config(bacterium: str, config: SegmentationConfig, backup: bool = True) -> bool:
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
    v = elem.get(attr)
    if v is None:
        raise ValueError(f"Missing attribute '{attr}' in {context}")
    return v


def _parse_float(s: str) -> float:
    return float(s.strip().replace(",", "."))


def get_pixel_size_um(
    xml_props_path: Optional[Path],
    xml_main_path: Optional[Path],
) -> Tuple[float, float]:
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
        "morph_kernel_size": 1,
        "morph_iterations": 1,
    }

    DEFAULT_SHAPE_FILTERS = {
        "min_circularity": 0.7,
        "max_circularity": 0.97,
        "min_aspect_ratio": 0.75,
        "max_aspect_ratio": 1.60,
        "min_solidity": 0.88,
        "dilate_iterations": 0,
        "erode_iterations": 0
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

    def __init__(self, root: tk.Tk, image_path: str, bacterium: str,
                 structure: str, mode: str, return_callback=None):
        self.current_pathogen = bacterium
        self.master = root
        self.root = root

        self.ui_scaler = UIScaler(root)

        self.image_path = Path(image_path)
        self.bacterium = bacterium
        self.structure = structure
        self.mode = mode
        self.return_callback = return_callback

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

        # ── Pick / Reject / Normalize state ──────────────────────────
        self.selection_mode: Any = False          # False | 'pick_reject' | 'done'
        self.accepted_indices: set = set()
        self.rejected_indices: set = set()
        self._pre_broad_params: dict = {}         # snapshot before broad detect

        self.sliders: Dict[str, Slider] = {}
        self.param_labels: Dict[str, tk.Label] = {}

        self.setup_gui()

    def quit(self, event=None):
        if messagebox.askyesno("Quit", "Are you sure you want to quit?\n\nUnsaved changes will be lost."):
            self.master.quit()
            self.master.destroy()

    def back(self, event=None):
        if hasattr(self, 'image_index') and self.image_index > 0:
            self.image_index -= 1
            self.load_image_at_index(self.image_index)
        else:
            messagebox.showinfo("Info", "Already at the first image")

    def load_image_at_index(self, index: int):
        if not hasattr(self, 'image_list') or not self.image_list:
            messagebox.showwarning("Warning", "No image list available")
            return
        if 0 <= index < len(self.image_list):
            image_path = self.image_list[index]
            try:
                self.original_image = self._load_image(image_path)
                self.image_path = image_path
                self.image_index = index
                self.master.title(f"Tuner - [{index + 1}/{len(self.image_list)}] {image_path.name}")
                self.update_visualization()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{e}")
        else:
            messagebox.showwarning("Warning", f"Invalid image index: {index}")

    def save_and_apply(self, event=None):
        """Save parameters to JSON with PIPELINE-COMPATIBLE field names"""
        if not self.save():
            messagebox.showerror("Error", "Failed to save parameters")
            return

        try:
            um2_per_px2 = self.pixel_size_um ** 2

            use_intensity_threshold = self.use_intensity_threshold
            intensity_threshold = (
                float(self.intensity_threshold_value)
                if use_intensity_threshold
                else float(self.manual_threshold)
            )

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

                use_intensity_threshold=bool(use_intensity_threshold),
                intensity_threshold=float(intensity_threshold),
                morph_kernel_size=int(self.morph_kernel_size),
                morph_iterations=int(self.morph_iterations),
                pixel_size_um=float(self.pixel_size_um),
                last_modified=datetime.now().isoformat(),
                tuned_by="Interactive Tuner"
            )

            bacteria_key = self.bacterium.lower().replace(' ', '_').replace('.', '').replace('-', '_')

            success = _manager.update_config(bacteria_key, config)

            if success:
                config_file = _manager._get_config_path(bacteria_key)

                if use_intensity_threshold:
                    thresh_info = f"Intensity threshold: {intensity_threshold:.0f} (BINARY_INV)"
                else:
                    thresh_info = "Background subtraction + Otsu"

                messagebox.showinfo(
                    "Success",
                    f"✓ Configuration saved!\n\n"
                    f"Bacterium: {self.bacterium}\n"
                    f"Saved to: {config_file.name}\n"
                    f"Threshold: {thresh_info}\n\n"
                    f"Morph: OPEN({self.morph_iterations}) → "
                    f"CLOSE({self.morph_iterations + 1})\n"
                    f"Dilate: {self.params['dilate_iterations']}  "
                    f"Erode: {self.params['erode_iterations']}"
                )

                print(f"\n{'=' * 80}")
                print(f"CONFIGURATION SAVED")
                print(f"{'=' * 80}")
                print(f"Bacterium: {self.bacterium}")
                print(f"Key: {bacteria_key}")
                print(f"File: {config_file}")
                print(f"\nPipeline-critical parameters:")
                print(f"  use_intensity_threshold: {use_intensity_threshold}")
                print(f"  intensity_threshold: {intensity_threshold:.1f}")
                print(f"  Gaussian σ: {config.gaussian_sigma:.2f}")
                print(f"  Morph: OPEN({config.morph_iterations}) → CLOSE({config.morph_iterations + 1})")
                print(f"  Min Area: {config.min_area_um2:.2f} µm²")
                print(f"  Max Area: {config.max_area_um2:.2f} µm²")
                print(f"  Circularity: {config.min_circularity:.2f} - {config.max_circularity:.2f}")
                print(f"  Aspect Ratio: {config.min_aspect_ratio:.2f} - {config.max_aspect_ratio:.2f}")
                print(f"  Solidity: ≥ {config.min_solidity:.2f}")
                print(f"  Pixel size: {config.pixel_size_um:.6f} µm")
                print(f"{'=' * 80}\n")
            else:
                messagebox.showerror(
                    "Error",
                    "Failed to update configuration\nCheck console for details"
                )

        except Exception as e:
            print(f"✗ Error saving configuration: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to save configuration:\n{e}")

    def save(self) -> bool:
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
                    'threshold_mode': self.threshold_mode,
                    'manual_threshold': self.manual_threshold,
                    'morph_kernel_size': self.morph_kernel_size,
                    'morph_iterations': self.morph_iterations,
                    'invert_image': self.invert_image,
                    'use_intensity_threshold': self.use_intensity_threshold,
                    'intensity_threshold': self.intensity_threshold_value,
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
        for key, value in self.DEFAULT_SHAPE_FILTERS.items():
            setattr(self, key, value)

        self.threshold_mode = self.DEFAULT_THRESHOLD_PARAMS['threshold_mode']
        self.manual_threshold = self.DEFAULT_THRESHOLD_PARAMS['manual_threshold']
        self.morph_kernel_size = self.DEFAULT_THRESHOLD_PARAMS['morph_kernel_size']
        self.morph_iterations = self.DEFAULT_THRESHOLD_PARAMS['morph_iterations']
        self.invert_image = False

        # Pipeline-compatible intensity threshold state
        self.use_intensity_threshold = False
        self.intensity_threshold_value = 80.0

        config_key = self.bacterium.lower().replace(' ', '_').replace('.', '').replace('-', '_')

        # Priority 1: Load from bacteria_configs/{pathogen}.json (PERMANENT)
        config_file = Path("bacteria_configs") / f"{config_key}.json"

        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                if "config" in json_data:
                    config_data = json_data["config"]
                else:
                    config_data = json_data

                if 'morph_kernel_size' in config_data:
                    kernel_size = config_data['morph_kernel_size']
                    if kernel_size % 2 == 0:
                        print(f"⚠️ WARNING: morph_kernel_size={kernel_size} is even, correcting to {kernel_size + 1}")
                        config_data['morph_kernel_size'] = kernel_size + 1

                if 'pixel_size_um' in config_data and config_data['pixel_size_um'] is not None:
                    if not self.has_metadata:          # only override if no XML metadata was found
                        self.pixel_size_um = float(config_data['pixel_size_um'])
                        print(f"   • Pixel size restored from JSON: {self.pixel_size_um:.6f} µm/px")

                um2_per_px2 = self.pixel_size_um ** 2

                self.threshold_mode = config_data.get('threshold_mode', 'otsu')
                self.manual_threshold = config_data.get('manual_threshold', 127)
                self.morph_kernel_size = config_data.get('morph_kernel_size', 3)
                self.morph_iterations = config_data.get('morph_iterations', 1)

                # Load pipeline intensity threshold settings
                self.use_intensity_threshold = bool(config_data.get('use_intensity_threshold', False))
                self.intensity_threshold_value = float(config_data.get('intensity_threshold', 80.0))

                # If use_intensity_threshold is True, override threshold_mode for UI
                if self.use_intensity_threshold:
                    self.threshold_mode = "intensity"
                    self.manual_threshold = int(self.intensity_threshold_value)
                    print(f"   • Threshold: INTENSITY (max={self.intensity_threshold_value})")

                self.params = {
                    "gaussian_sigma": float(config_data.get('gaussian_sigma', 2.0)),
                    "min_area": float(config_data.get('min_area_um2', 3.0) / um2_per_px2),
                    "max_area": float(config_data.get('max_area_um2', 100.0) / um2_per_px2),
                    "dilate_iterations": int(config_data.get('dilate_iterations', 0)),
                    "erode_iterations": int(config_data.get('erode_iterations', 0)),
                }

                self.min_circularity = float(config_data.get('min_circularity', 0.0))
                self.max_circularity = float(config_data.get('max_circularity', 1.0))
                self.min_aspect_ratio = float(config_data.get('min_aspect_ratio', 0.2))
                self.max_aspect_ratio = float(config_data.get('max_aspect_ratio', 10.0))
                self.min_solidity = float(config_data.get('min_solidity', 0.3))

                self.invert_image = config_data.get('invert_image', False)

                print(f"✅ Loaded PERMANENT config from: {config_file}")
                print(f"   • Gaussian σ: {self.params['gaussian_sigma']:.2f}")
                print(f"   • Morph Kernel: {self.morph_kernel_size}x{self.morph_kernel_size}")
                print(f"   • Morph Iterations: {self.morph_iterations}")
                if not self.use_intensity_threshold:
                    print(f"   • Threshold: {self.threshold_mode.upper()}")
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

        # Priority 2: Load from session JSON (temporary tuning)
        json_filename = f"segmentation_params_{self.bacterium}_{self.structure}_{self.mode}.json"

        if Path(json_filename).exists():
            try:
                with open(json_filename, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)

                params_dict = saved_data['parameters'].copy()
                self.invert_image = params_dict.pop('invert_image', False)

                self.params = {
                    "gaussian_sigma": params_dict.get('gaussian_sigma', 2.0),
                    "min_area": params_dict.get('min_area', 20),
                    "max_area": params_dict.get('max_area', 5000),
                    "dilate_iterations": params_dict.get('dilate_iterations', 0),
                    "erode_iterations": params_dict.get('erode_iterations', 0),
                }

                self.min_circularity = params_dict.get('min_circularity', 0.0)
                self.max_circularity = params_dict.get('max_circularity', 1.0)
                self.min_aspect_ratio = params_dict.get('min_aspect_ratio', 0.2)
                self.max_aspect_ratio = params_dict.get('max_aspect_ratio', 10.0)
                self.min_solidity = params_dict.get('min_solidity', 0.3)

                self.threshold_mode = params_dict.get('threshold_mode', 'otsu')
                self.manual_threshold = params_dict.get('manual_threshold', 127)
                self.morph_kernel_size = params_dict.get('morph_kernel_size', 3)
                self.morph_iterations = params_dict.get('morph_iterations', 1)

                self.use_intensity_threshold = params_dict.get('use_intensity_threshold', False)
                self.intensity_threshold_value = params_dict.get('intensity_threshold', 80.0)
                if self.use_intensity_threshold:
                    self.threshold_mode = "intensity"

                print(f"✅ Restored TEMP session from: {json_filename}")
                return

            except Exception as e:
                print(f"⚠️ Could not load session JSON: {e}")

        # Priority 3: Load from bacteria_configs.py (legacy Python config)
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

                self.min_circularity = float(saved_config.min_circularity)
                self.max_circularity = float(saved_config.max_circularity)
                self.min_aspect_ratio = float(saved_config.min_aspect_ratio)
                self.max_aspect_ratio = float(saved_config.max_aspect_ratio)
                self.min_solidity = float(saved_config.min_solidity)

                self.invert_image = False

                print(f"✅ Loaded config for {self.bacterium} from bacteria_configs.py")
                return

            except Exception as e:
                print(f"⚠️ Error loading bacteria_configs.py: {e}")

        # Priority 4: Use defaults (nothing found)
        print(f"ℹ️ No saved config found for '{self.bacterium}' - using defaults")
        self.params = self.DEFAULT_PARAMS.copy()
        self.threshold_mode = self.DEFAULT_THRESHOLD_PARAMS['threshold_mode']
        self.manual_threshold = self.DEFAULT_THRESHOLD_PARAMS['manual_threshold']
        self.morph_kernel_size = self.DEFAULT_THRESHOLD_PARAMS['morph_kernel_size']
        self.morph_iterations = self.DEFAULT_THRESHOLD_PARAMS['morph_iterations']
        self.invert_image = False
        self.use_intensity_threshold = False
        self.intensity_threshold_value = 80.0

    def update_threshold(self, param_name: str, value: float):
        setattr(self, param_name, value)
        self.update_visualization()

    def update_morph(self, param_name: str, value: float):
        setattr(self, param_name, value)
        self.update_visualization()

    def cycle_threshold_mode(self, event):
        """Cycle through threshold modes (legacy matplotlib button callback)"""
        modes = ["otsu", "manual", "adaptive", "intensity"]
        if self.threshold_mode in modes:
            current_idx = modes.index(self.threshold_mode)
        else:
            current_idx = 0
        next_idx = (current_idx + 1) % len(modes)
        self.threshold_mode = modes[next_idx]

        if self.threshold_mode == "intensity":
            self.use_intensity_threshold = True
            self.intensity_threshold_value = float(self.manual_threshold)
        else:
            self.use_intensity_threshold = False

        if hasattr(self, 'btn_thresh_mode'):
            mode_text = f"THRESHOLD\n{self.threshold_mode.upper()}"
            self.btn_thresh_mode.label.set_text(mode_text)

        if 'manual_threshold' in self.sliders:
            self.sliders['manual_threshold'].set_active(
                self.threshold_mode in ("manual", "intensity")
            )

        self.update_visualization()

    def setup_gui(self):
        window_width, window_height = self.ui_scaler.get_window_size()

        self.root.title(f"Segmentation Tuner - {self.bacterium}")
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.minsize(1280, 720)
        self.root.configure(bg=self.COLORS['bg'])

        self.root.resizable(True, True)
        self.root.bind('<Configure>', self._on_window_resize)

        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (window_width // 2)
        y = (self.root.winfo_screenheight() // 2) - (window_height // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._create_header(main_container)

        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)

        self._create_vertical_slider_panel(content_frame)
        self._create_image_panel(content_frame)
        self._create_right_panel(content_frame)

        print("✅ GUI Setup Complete")
        self.update_visualization()
        print("✅ Initial visualization complete")

    def _create_vertical_slider_panel(self, parent: ttk.Frame):
        self.parameter_panel = ParameterPanel(parent, tuner_instance=self)
        self.parameter_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))

    def _on_window_resize(self, event):
        if event.widget == self.root:
            if hasattr(self, '_resize_timer'):
                self.root.after_cancel(self._resize_timer)
            self._resize_timer = self.root.after(100, self._refresh_layout)

    def _refresh_layout(self):
        if hasattr(self, 'canvas_image'):
            self.canvas_image.draw()
        if hasattr(self, 'canvas_hist'):
            self.canvas_hist.draw()
        if hasattr(self, 'canvas_sliders'):
            self.canvas_sliders.draw()

    def _create_header(self, parent: ttk.Frame):
        header_height = self.ui_scaler.scale_dimension(45)
        header_frame = tk.Frame(parent, bg=self.COLORS['header'], height=header_height)
        header_frame.pack(fill=tk.X, pady=(0, 5))
        header_frame.pack_propagate(False)

        title_font = ("Segoe UI", self.ui_scaler.scale_font(14), "bold")
        badge_font = ("Segoe UI", self.ui_scaler.scale_font(9))
        button_font = ("Segoe UI", self.ui_scaler.scale_font(9), "bold")

        title_text = f"🔬 {self.bacterium} - {self.structure}"
        tk.Label(
            header_frame, text=title_text, font=title_font,
            bg=self.COLORS['header'], fg="white"
        ).pack(side=tk.LEFT, padx=20, pady=6)

        mode_text = f"Mode: {self.mode} {'(Inverted)' if self.invert_image else ''}"
        tk.Label(
            header_frame, text=mode_text, font=badge_font,
            bg=self.COLORS['secondary'], fg="white",
            padx=12, pady=4, relief=tk.RAISED
        ).pack(side=tk.LEFT, pady=6)

        pixel_color = self.COLORS['success'] if self.has_metadata else self.COLORS['warning']
        pixel_text = f"Pixel: {self.pixel_size_um:.6f} µm"
        if not self.has_metadata:
            pixel_text += " (fallback)"
        tk.Label(
            header_frame, text=pixel_text, font=badge_font,
            bg=pixel_color, fg="white", padx=12, pady=4, relief=tk.RAISED
        ).pack(side=tk.LEFT, padx=10, pady=6)

        tk.Button(
            header_frame, text="📁 LOAD IMAGE", font=button_font,
            bg=self.COLORS['primary'], fg="white", padx=12, pady=4,
            relief=tk.RAISED, command=self.load_new_image, cursor="hand2"
        ).pack(side=tk.LEFT, padx=10, pady=6)

        # ── Pick / Reject / Normalize buttons ────────────────────────
        self.btn_pick_reject = tk.Button(
            header_frame,
            text="🎯 PICK/REJECT",
            font=button_font,
            bg=self.COLORS['warning'], fg="white",
            relief=tk.RAISED, cursor="hand2",
            command=self.enter_pick_reject_mode,
            padx=10, pady=4
        )
        self.btn_pick_reject.pack(side=tk.LEFT, padx=5, pady=6)

        self.btn_normalize = tk.Button(
            header_frame,
            text="✨ NORMALIZE",
            font=button_font,
            bg=self.COLORS['success'], fg="white",
            relief=tk.RAISED, cursor="hand2",
            command=self.normalize_from_selection,
            padx=10, pady=4,
            state=tk.DISABLED
        )
        self.btn_normalize.pack(side=tk.LEFT, padx=2, pady=6)

        self.btn_cancel_pr = tk.Button(
            header_frame,
            text="✖ CANCEL",
            font=button_font,
            bg=self.COLORS['danger'], fg="white",
            relief=tk.RAISED, cursor="hand2",
            command=self.cancel_pick_reject,
            padx=10, pady=4,
            state=tk.DISABLED
        )
        self.btn_cancel_pr.pack(side=tk.LEFT, padx=2, pady=6)

        # ── Right-side permanent buttons ─────────────────────────────
        self.contour_count_label = tk.Label(
            header_frame, text="Contours: 0", font=badge_font,
            bg=self.COLORS['success'], fg="white",
            padx=10, pady=4, relief=tk.RAISED
        )
        self.contour_count_label.pack(side=tk.RIGHT, padx=(10, 20), pady=6)

        button_specs = [
            ("❌ QUIT", self.COLORS['danger'], self.quit),
            ("✅ SAVE & APPLY", self.COLORS['success'], self.save_and_apply),
            ("💾 SAVE JSON", self.COLORS['secondary'], self.save),
            ("⬅ BACK", self.COLORS['secondary'], self.back),
        ]

        for text, color, command in button_specs:
            tk.Button(
                header_frame, text=text, font=button_font,
                bg=color, fg="white", activebackground=color,
                activeforeground="white", relief=tk.RAISED,
                command=command, cursor="hand2", padx=12, pady=4
            ).pack(side=tk.RIGHT, padx=5, pady=6)

    def _create_content_area(self, parent: ttk.Frame) -> ttk.Frame:
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True)
        self._create_image_panel(content_frame)
        self._create_right_panel(content_frame)
        return content_frame

    def _create_image_panel(self, parent: ttk.Frame):
        left_panel = ttk.Frame(parent, relief=tk.RIDGE, borderwidth=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        header = tk.Frame(left_panel, bg=self.COLORS['secondary'], height=30)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header, text="📷 IMAGE ANALYSIS - Original + Contours",
            font=("Segoe UI", 11, "bold"), bg=self.COLORS['secondary'], fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=5)

        canvas_frame = ttk.Frame(left_panel)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig_image = Figure(figsize=(11, 8), facecolor='white', dpi=100)
        self.ax_image = self.fig_image.add_subplot(111)
        self.canvas_image = FigureCanvasTkAgg(self.fig_image, canvas_frame)
        self.canvas_image.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_image.mpl_connect("button_press_event", self.on_image_click)

        instruction = tk.Frame(left_panel, bg=self.COLORS['primary'], height=28)
        instruction.pack(fill=tk.X)
        instruction.pack_propagate(False)
        self.instruction_label = tk.Label(
            instruction,
            text="💡 Click on a particle to analyze and get parameter suggestions",
            font=("Segoe UI", 9), bg=self.COLORS['primary'], fg="white"
        )
        self.instruction_label.pack(pady=4)

    def _create_right_panel(self, parent: ttk.Frame):
        right_panel = ttk.Frame(parent, width=420)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_panel.pack_propagate(False)

        self._create_parameters_section(right_panel)
        self._create_target_analysis_section(right_panel)
        self._create_histogram_section(right_panel)

    def _create_parameters_section(self, parent: ttk.Frame):
        header_height = self.ui_scaler.scale_dimension(28)
        header = tk.Frame(parent, bg=self.COLORS['secondary'], height=header_height)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        header_font = ("Segoe UI", self.ui_scaler.scale_font(10), "bold")
        tk.Label(
            header, text="⚙️ PARAMETERS", font=header_font,
            bg=self.COLORS['secondary'], fg="white"
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
        title_font = ("Segoe UI", self.ui_scaler.scale_font(9), "bold")
        label_font = ("Segoe UI", self.ui_scaler.scale_font(8))
        value_font = ("Segoe UI", self.ui_scaler.scale_font(8), "bold")
        label_width = self.ui_scaler.scale_dimension(15)

        tk.Label(
            parent, text=title, font=title_font,
            foreground=self.COLORS['header']
        ).pack(anchor="w", pady=(0, 4))

        for label_text, value_text, color in params:
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=1)

            tk.Label(
                frame, text=label_text, font=label_font,
                foreground="gray", width=label_width, anchor="w"
            ).pack(side=tk.LEFT)

            padx = self.ui_scaler.scale_dimension(6)

            if color:
                value_label = tk.Label(
                    frame, text=value_text, font=value_font,
                    foreground="white", bg=color,
                    padx=padx, pady=1, relief=tk.RAISED
                )
            else:
                value_label = tk.Label(
                    frame, text=value_text, font=value_font,
                    foreground=self.COLORS['header']
                )
            value_label.pack(side=tk.LEFT)

            self.param_labels[label_text] = value_label

    def _create_target_analysis_section(self, parent: ttk.Frame):
        header_height = self.ui_scaler.scale_dimension(26)
        header = tk.Frame(parent, bg=self.COLORS['warning'], height=header_height)
        header.pack(fill=tk.X, pady=(8, 0))
        header.pack_propagate(False)
        header_font = ("Segoe UI", self.ui_scaler.scale_font(9), "bold")
        tk.Label(
            header, text="🎯 TARGET ANALYSIS", font=header_font,
            bg=self.COLORS['warning'], fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=3)

        display = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        display.pack(fill=tk.X, padx=5, pady=(0, 5))

        text_frame = tk.Frame(display, bg="white")
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        label_font = ("Segoe UI", self.ui_scaler.scale_font(8))

        self.target_analysis_text = tk.Text(
            text_frame, font=label_font, wrap=tk.WORD, height=10,
            bg="white", fg="gray", padx=8, pady=8,
            yscrollcommand=scrollbar.set, relief=tk.FLAT
        )
        self.target_analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.target_analysis_text.yview)

        self.target_analysis_text.insert('1.0', "Click on image to analyze a particle or detect missed particles")
        self.target_analysis_text.config(state=tk.NORMAL)

    def _create_histogram_section(self, parent: ttk.Frame):
        header = tk.Frame(parent, bg=self.COLORS['info'], height=28)
        header.pack(fill=tk.X, pady=(5, 0))
        header.pack_propagate(False)
        tk.Label(
            header, text="📊 AREA DISTRIBUTION",
            font=("Segoe UI", 10, "bold"), bg=self.COLORS['info'], fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=4)

        canvas_frame = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        self.fig_hist = Figure(figsize=(5.2, 3.5), facecolor='white', dpi=80)
        self.ax_hist = self.fig_hist.add_subplot(111)
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, canvas_frame)
        self.canvas_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    # ------------------------------------------------------------------
    # Legacy matplotlib slider panel (kept for backwards compat, unused)
    # ------------------------------------------------------------------

    def _create_control_panel(self, parent: ttk.Frame):
        panel = ttk.Frame(parent)
        panel.pack(fill=tk.X, pady=(5, 0))

        header = tk.Frame(panel, bg=self.COLORS['secondary'], height=30)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header, text="🎚️ ADJUST PARAMETERS",
            font=("Segoe UI", 11, "bold"), bg=self.COLORS['secondary'], fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=5)

        slider_frame = ttk.Frame(panel, relief=tk.SUNKEN, borderwidth=1)
        slider_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        screen_width = self.root.winfo_screenwidth()
        available_width = screen_width * 0.92
        fig_width_inches = available_width / 100

        self.fig_sliders = Figure(figsize=(fig_width_inches, 2.5), facecolor='#f8f9fa', dpi=100)
        self.canvas_sliders = FigureCanvasTkAgg(self.fig_sliders, slider_frame)
        self.canvas_sliders.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._create_sliders()

    def _create_sliders(self):
        self.fig_sliders.clear()

        self.fig_sliders.text(0.18, 0.97, 'SEGMENTATION', ha='center', va='top',
                              fontsize=9, fontweight='bold', color=self.COLORS['header'])
        self.fig_sliders.text(0.50, 0.97, 'FILTERING & MORPHOLOGY', ha='center', va='top',
                              fontsize=9, fontweight='bold', color=self.COLORS['header'])
        self.fig_sliders.text(0.82, 0.97, 'SHAPE FILTERS', ha='center', va='top',
                              fontsize=9, fontweight='bold', color=self.COLORS['header'])

        slider_height = 0.10
        slider_width = 0.25

        col1_x = 0.04
        col2_x = 0.36
        col3_x = 0.68

        row1_y = 0.78
        row2_y = 0.62
        row3_y = 0.46
        row4_y = 0.30
        row5_y = 0.14

        ax = self.fig_sliders.add_axes((col1_x, row1_y, slider_width, slider_height))
        slider = Slider(ax, 'Gaussian σ', 0.5, 20.0,
                        valinit=self.params['gaussian_sigma'], color=self.COLORS['primary'])
        slider.on_changed(lambda val: self.update_parameter('gaussian_sigma', val))
        self.sliders['gaussian_sigma'] = slider

        ax = self.fig_sliders.add_axes((col1_x, row2_y, slider_width, slider_height))
        slider = Slider(ax, 'Threshold', 0, 255,
                        valinit=self.manual_threshold, valstep=1, color=self.COLORS['primary'])
        slider.on_changed(lambda val: self.update_threshold('manual_threshold', val))
        self.sliders['manual_threshold'] = slider

        ax = self.fig_sliders.add_axes((col1_x, row3_y, slider_width, slider_height))
        slider = Slider(ax, 'Morph Kernel', 3, 15,
                        valinit=self.morph_kernel_size, valstep=2, color=self.COLORS['primary'])
        slider.on_changed(lambda val: self.update_morph('morph_kernel_size', val))
        self.sliders['morph_kernel_size'] = slider

        ax = self.fig_sliders.add_axes((col1_x, row4_y, slider_width, slider_height))
        slider = Slider(ax, 'Morph Iter', 0, 5,
                        valinit=self.morph_iterations, valstep=1, color=self.COLORS['primary'])
        slider.on_changed(lambda val: self.update_morph('morph_iterations', val))
        self.sliders['morph_iterations'] = slider

        ax_thresh_mode = self.fig_sliders.add_axes((col1_x, row5_y, 0.13, slider_height * 1.5))
        mode_text = f"THRESHOLD\n{self.threshold_mode.upper()}"
        self.btn_thresh_mode = Button(ax_thresh_mode, mode_text,
                                      color=self.COLORS['primary'],
                                      hovercolor=self.COLORS['header'])
        self.btn_thresh_mode.on_clicked(self.cycle_threshold_mode)

        ax = self.fig_sliders.add_axes((col2_x, row1_y, slider_width, slider_height))
        slider = Slider(ax, 'Min Area (px)', 10, 5000,
                        valinit=self.params['min_area'], valstep=10, color=self.COLORS['info'])
        slider.on_changed(lambda val: self.update_parameter('min_area', val))
        self.sliders['min_area'] = slider

        ax = self.fig_sliders.add_axes((col2_x, row2_y, slider_width, slider_height))
        slider = Slider(ax, 'Max Area (px)', 100, 30000,
                        valinit=self.params['max_area'], valstep=100, color=self.COLORS['info'])
        slider.on_changed(lambda val: self.update_parameter('max_area', val))
        self.sliders['max_area'] = slider

        ax = self.fig_sliders.add_axes((col2_x, row3_y, slider_width, slider_height))
        slider = Slider(ax, 'Dilate Iter', 0, 5,
                        valinit=self.params['dilate_iterations'], valstep=1, color=self.COLORS['warning'])
        slider.on_changed(lambda val: self.update_parameter('dilate_iterations', val))
        self.sliders['dilate_iterations'] = slider

        ax = self.fig_sliders.add_axes((col2_x, row4_y, slider_width, slider_height))
        slider = Slider(ax, 'Erode Iter', 0, 5,
                        valinit=self.params['erode_iterations'], valstep=1, color=self.COLORS['warning'])
        slider.on_changed(lambda val: self.update_parameter('erode_iterations', val))
        self.sliders['erode_iterations'] = slider

        ax = self.fig_sliders.add_axes((col3_x, row1_y, slider_width, slider_height))
        slider = Slider(ax, 'Min Circular', 0.0, 1.0,
                        valinit=float(self.min_circularity), valstep=0.01, color=self.COLORS['purple'])
        slider.on_changed(lambda val: self.update_shape_filter('min_circularity', val))
        self.sliders['min_circularity'] = slider

        ax = self.fig_sliders.add_axes((col3_x, row2_y, slider_width, slider_height))
        slider = Slider(ax, 'Max Circular', 0.0, 1.0,
                        valinit=float(self.max_circularity), valstep=0.01, color=self.COLORS['purple'])
        slider.on_changed(lambda val: self.update_shape_filter('max_circularity', val))
        self.sliders['max_circularity'] = slider

        ax = self.fig_sliders.add_axes((col3_x, row3_y, slider_width, slider_height))
        slider = Slider(ax, 'Min Solidity', 0.0, 1.0,
                        valinit=float(self.min_solidity), valstep=0.01, color=self.COLORS['purple'])
        slider.on_changed(lambda val: self.update_shape_filter('min_solidity', val))
        self.sliders['min_solidity'] = slider

        invert_color = self.COLORS['success'] if self.invert_image else self.COLORS['gray']
        invert_text = f'INVERT\n{"ON" if self.invert_image else "OFF"}'
        ax_invert = self.fig_sliders.add_axes((col2_x, row5_y, 0.13, slider_height * 1.5))
        self.btn_invert = Button(ax_invert, invert_text, color=invert_color,
                                 hovercolor=self.COLORS['success'])
        self.btn_invert.on_clicked(self.toggle_invert)

        ax_apply = self.fig_sliders.add_axes((col2_x + 0.15, row5_y, 0.13, slider_height * 1.5))
        self.btn_apply = Button(ax_apply, "APPLY\nSUGGESTIONS",
                                color=self.COLORS['primary'], hovercolor='#5dade2')
        self.btn_apply.on_clicked(self.apply_suggestions)

    # ------------------------------------------------------------------
    # process_image — MATCHES PIPELINE (dual threshold path + OPEN/CLOSE)
    # ------------------------------------------------------------------

    def process_image(self):
        """Process image - MATCHES pipeline segment_particles_brightfield exactly"""
        img = self.original_image.copy()

        # NOTE: invert_image in the pipeline does NOT actually invert.
        # We replicate that here so tuner matches pipeline output.
        if self.invert_image:
            pass  # Pipeline no-op; toggle this if pipeline is fixed

        # Step 1: Gaussian blur (pipeline does this unconditionally)
        blur = cv2.GaussianBlur(
            img, (0, 0),
            sigmaX=self.params["gaussian_sigma"],
            sigmaY=self.params["gaussian_sigma"]
        )

        # Step 2: Thresholding — TWO PATHS matching pipeline
        if self.use_intensity_threshold:
            # PATH A: Direct intensity threshold (e.g., Klebsiella)
            # Pipeline: cv2.THRESH_BINARY_INV on the blurred image
            _, binary = cv2.threshold(
                blur,
                float(self.manual_threshold),
                255,
                cv2.THRESH_BINARY_INV
            )
            self.processed_image = blur
        else:
            # PATH B: Background subtraction + Otsu/Manual/Adaptive
            bg = cv2.GaussianBlur(
                img, (0, 0),
                sigmaX=self.params["gaussian_sigma"],
                sigmaY=self.params["gaussian_sigma"]
            )
            enhanced = cv2.subtract(bg, img)
            enhanced_blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

            self.processed_image = enhanced_blur

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
                    enhanced_blur, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
            else:
                _, binary = cv2.threshold(
                    enhanced_blur, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

        # Step 3: Morphological operations — MATCH PIPELINE EXACTLY
        # Pipeline does: MORPH_OPEN(iter) then MORPH_CLOSE(iter+1)
        kernel_size = int(self.morph_kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        morph_iter = int(self.morph_iterations)

        # OPEN: removes small noise (pipeline step 1)
        if morph_iter > 0:
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN, kernel,
                iterations=morph_iter
            )

        # CLOSE: fills holes and connects (pipeline step 2)
        # Pipeline uses morph_iterations + 1 for CLOSE
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel,
            iterations=morph_iter + 1
        )

        # Optional dilate (pipeline step 3)
        if int(self.params["dilate_iterations"]) > 0:
            binary = cv2.dilate(
                binary, kernel,
                iterations=int(self.params["dilate_iterations"])
            )

        # Optional erode (pipeline step 4)
        if int(self.params["erode_iterations"]) > 0:
            binary = cv2.erode(
                binary, kernel,
                iterations=int(self.params["erode_iterations"])
            )

        self.binary_mask = binary

        # Step 4: Contour detection and filtering
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        self.contours = []
        self.contour_areas = []

        H, W = img.shape[:2]
        img_area_px = float(H * W)
        max_big_area_px = 0.25 * img_area_px

        for cnt in contours:
            area_px = float(cv2.contourArea(cnt))

            if area_px <= 0:
                continue

            if not (self.params["min_area"] <= area_px <= self.params["max_area"]):
                continue

            if area_px >= max_big_area_px:
                continue

            perimeter = float(cv2.arcLength(cnt, True))
            if perimeter > 0:
                circularity = (4 * np.pi * area_px) / (perimeter ** 2)
            else:
                circularity = 0.0

            if not (self.min_circularity <= circularity <= self.max_circularity):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0.0

            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                continue

            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            solidity = area_px / hull_area if hull_area > 0 else 0.0

            if solidity < self.min_solidity:
                continue

            self.contours.append(cnt)
            self.contour_areas.append(area_px)

    def update_visualization(self):
        # In pick/reject mode, re-run broad detection but keep
        # accepted/rejected state and use the pick/reject display.
        if self.selection_mode == 'pick_reject':
            self._run_broad_detection()
            self._update_image_display_pick_reject()
            self._update_histogram()
            self._update_param_displays()
            self.contour_count_label.config(
                text=f"PICK/REJECT — {len(self.contours)} objects"
            )
            return

        self.process_image()
        self._update_image_display()
        self._update_histogram()
        self._update_param_displays()
        self.contour_count_label.config(text=f"Contours: {len(self.contours)}")

    def _update_image_display(self):
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
        invert_label = self.param_labels["Invert:"]
        invert_text = "ON" if self.invert_image else "OFF"
        invert_color = self.COLORS['success'] if self.invert_image else self.COLORS['gray']
        invert_label.config(text=invert_text, bg=invert_color)

        um2_per_px2 = self.pixel_size_um ** 2
        min_area_um2 = self.params['min_area'] * um2_per_px2
        max_area_um2 = self.params['max_area'] * um2_per_px2

        updates = {
            "Gaussian σ:": f"{self.params['gaussian_sigma']:.1f}",
            "Threshold:": self.threshold_mode.upper(),
            "Morph kernel:": f"{int(self.morph_kernel_size)}x{int(self.morph_kernel_size)}",
            "Morph iter:": str(int(self.morph_iterations)),
            "Min area:": f"{self.params['min_area']:.0f} px ({min_area_um2:.2f} µm²)",
            "Max area:": f"{self.params['max_area']:.0f} px ({max_area_um2:.2f} µm²)",
            "Dilate iter:": str(int(self.params["dilate_iterations"])),
            "Erode iter:": str(int(self.params["erode_iterations"])),
            "Circularity:": f"{self.min_circularity:.2f} - {self.max_circularity:.2f}",
            "Aspect ratio:": f"{self.min_aspect_ratio:.2f} - {self.max_aspect_ratio:.2f}",
            "Solidity:": f"≥ {self.min_solidity:.2f}",
        }

        for key, value in updates.items():
            if key in self.param_labels:
                self.param_labels[key].config(text=value)

    def update_parameter(self, param_name: str, value: float):
        self.params[param_name] = value
        self.update_visualization()

    def update_shape_filter(self, filter_name: str, value: float):
        setattr(self, filter_name, value)
        self.update_visualization()

        if filter_name in ["min_circularity", "max_circularity"]:
            key = "Circularity:"
            text = f"{self.min_circularity:.2f} - {self.max_circularity:.2f}"
        elif filter_name in ["min_aspect_ratio", "max_aspect_ratio"]:
            key = "Aspect ratio:"
            text = f"{self.min_aspect_ratio:.2f} - {self.max_aspect_ratio:.2f}"
        elif filter_name == "min_solidity":
            key = "Solidity:"
            text = f"≥ {self.min_solidity:.2f}"
        else:
            return

        if key in self.param_labels:
            self.param_labels[key].config(text=text)

    def toggle_invert(self, event):
        """Toggle image inversion (legacy matplotlib button callback)"""
        self.invert_image = not self.invert_image

        if hasattr(self, 'btn_invert'):
            invert_text = f'INVERT\n{"ON" if self.invert_image else "OFF"}'
            invert_color = self.COLORS['success'] if self.invert_image else self.COLORS['gray']
            self.btn_invert.label.set_text(invert_text)
            self.btn_invert.color = invert_color

        self.update_visualization()

    # ==================================================================
    # PICK / REJECT / NORMALIZE
    # ==================================================================

    def _run_broad_detection(self):
        """Run process_image with the currently stored (loose) params.

        Called both when first entering pick/reject mode AND on every
        update_visualization call while in that mode, so the contour
        list stays consistent with any slider changes the user makes
        while browsing.
        """
        self.process_image()

    def enter_pick_reject_mode(self):
        """Step 1 — broad detection with very loose params, enter pick/reject."""
        H, W = self.original_image.shape[:2]

        # Snapshot current params so we can restore on cancel
        self._pre_broad_params = {
            'params': self.params.copy(),
            'min_circularity':  self.min_circularity,
            'max_circularity':  self.max_circularity,
            'min_aspect_ratio': self.min_aspect_ratio,
            'max_aspect_ratio': self.max_aspect_ratio,
            'min_solidity':     self.min_solidity,
        }

        # Very loose bounds — catch every dark object
        self.params['min_area']   = 5.0
        self.params['max_area']   = float(H * W) * 0.15   # up to 15 % of image
        self.min_circularity      = 0.0
        self.max_circularity      = 1.0
        self.min_aspect_ratio     = 0.01
        self.max_aspect_ratio     = 50.0
        self.min_solidity         = 0.0

        # Sync loose values into the parameter panel sliders so the
        # user can see them (and tweak if needed without leaving mode)
        if hasattr(self, 'parameter_panel'):
            pp = self.parameter_panel
            for name, val in [
                ('min_area',         self.params['min_area']),
                ('max_area',         self.params['max_area']),
                ('min_circularity',  self.min_circularity),
                ('max_circularity',  self.max_circularity),
                ('min_aspect_ratio', self.min_aspect_ratio),
                ('max_aspect_ratio', self.max_aspect_ratio),
                ('min_solidity',     self.min_solidity),
            ]:
                if name in pp.sliders:
                    pp.sliders[name].set(val)

        self._run_broad_detection()

        # Reset selection state for a fresh session
        self.accepted_indices = set()
        self.rejected_indices = set()
        self.selection_mode   = 'pick_reject'

        self._update_image_display_pick_reject()
        self._update_pick_reject_buttons(active=True)

        n = len(self.contours)
        self.contour_count_label.config(
            text=f"PICK/REJECT — {n} objects found"
        )

        # Update instruction bar
        if hasattr(self, 'instruction_label'):
            self.instruction_label.config(
                text="🖱 LEFT-click = ✅ Accept  |  RIGHT-click = ❌ Reject  "
                     "|  Click accepted to toggle back to unclassified",
                bg=self.COLORS['warning']
            )

        self._show_pick_reject_status()

    def _update_image_display_pick_reject(self):
        """Redraw image with colour-coded pick/reject contours."""
        self.ax_image.clear()

        if self.original_image.ndim == 2:
            display = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB)
        else:
            display = self.original_image.copy()

        for i, cnt in enumerate(self.contours):
            if i in self.accepted_indices:
                color, thickness = (0, 220, 0), 3       # green  = accepted
            elif i in self.rejected_indices:
                color, thickness = (220, 40, 40), 2     # red    = rejected
            else:
                color, thickness = (255, 200, 0), 1     # yellow = unclassified

            cv2.drawContours(display, [cnt], -1, color, thickness)

            # Small index label near each centroid
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(
                    display, str(i), (cx - 4, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
                    cv2.LINE_AA
                )

        n_acc = len(self.accepted_indices)
        n_rej = len(self.rejected_indices)
        n_unc = len(self.contours) - n_acc - n_rej

        self.ax_image.imshow(display)
        self.ax_image.set_title(
            f"PICK/REJECT  —  Left-click ✅ Accept    Right-click ❌ Reject\n"
            f"🟢 Accepted: {n_acc}   🔴 Rejected: {n_rej}   🟡 Unclassified: {n_unc}",
            fontsize=10, fontweight='bold', color='#1a5276', pad=8
        )
        self.ax_image.axis('off')
        self.canvas_image.draw()

    def _update_pick_reject_buttons(self, active: bool):
        """Enable/disable the Normalize and Cancel buttons."""
        state = tk.NORMAL if active else tk.DISABLED
        self.btn_normalize.config(state=state)
        self.btn_cancel_pr.config(state=state)
        self.btn_pick_reject.config(
            text="⏳ Selecting…" if active else "🎯 PICK/REJECT",
            state=tk.DISABLED if active else tk.NORMAL
        )

    def _show_pick_reject_status(self):
        """Update the target-analysis text box with pick/reject counts."""
        n_acc = len(self.accepted_indices)
        n_rej = len(self.rejected_indices)
        n_tot = len(self.contours)
        n_unc = n_tot - n_acc - n_rej

        msg = (
            f"🎯 PICK / REJECT MODE\n\n"
            f"Total objects detected: {n_tot}\n\n"
            f"🟢 Accepted (left-click):    {n_acc}\n"
            f"🔴 Rejected (right-click):   {n_rej}\n"
            f"🟡 Unclassified:             {n_unc}\n\n"
            f"When done, click  ✨ NORMALIZE  in the header\n"
            f"to compute optimal parameters from your selection.\n\n"
            f"Click  ✖ CANCEL  to restore original parameters."
        )

        self.target_analysis_text.delete('1.0', tk.END)
        self.target_analysis_text.insert('1.0', msg)

    def normalize_from_selection(self):
        """Step 3 — compute optimal parameter bounds from accepted / rejected objects."""
        if not self.accepted_indices:
            messagebox.showwarning(
                "No selection",
                "Please accept at least one particle (left-click) before normalizing.",
                parent=self.root
            )
            return

        acc_props = self._compute_contour_properties(
            [self.contours[i] for i in self.accepted_indices]
        )
        rej_props = self._compute_contour_properties(
            [self.contours[i] for i in self.rejected_indices]
        ) if self.rejected_indices else []

        def bounds(values, lo_pct=5, hi_pct=95, margin=0.20):
            arr = np.array(values)
            lo = np.percentile(arr, lo_pct) * (1.0 - margin)
            hi = np.percentile(arr, hi_pct) * (1.0 + margin)
            return lo, hi

        acc_areas = [p['area']        for p in acc_props]
        acc_circs = [p['circularity'] for p in acc_props]
        acc_asps  = [p['aspect_ratio']for p in acc_props]
        acc_sols  = [p['solidity']    for p in acc_props]

        new_min_area, new_max_area = bounds(acc_areas)
        new_min_circ, new_max_circ = bounds(acc_circs, margin=0.05)
        new_min_asp,  new_max_asp  = bounds(acc_asps,  margin=0.10)
        new_min_sol,  _            = bounds(acc_sols,  hi_pct=100, margin=0)

        # ── Refine against rejected objects ──────────────────────────
        if rej_props:
            rej_areas = np.array([p['area'] for p in rej_props])
            mean_acc  = float(np.mean(acc_areas))

            # Push min_area up if small rejected objects sit below acc range
            rej_small = rej_areas[rej_areas < mean_acc]
            if len(rej_small):
                new_min_area = max(new_min_area, float(np.max(rej_small)) * 1.05)

            # Push max_area down if large rejected objects sit above acc range
            rej_large = rej_areas[rej_areas > mean_acc]
            if len(rej_large):
                new_max_area = min(new_max_area, float(np.min(rej_large)) * 0.95)

        # ── Clamp to sane ranges ─────────────────────────────────────
        H, W = self.original_image.shape[:2]
        new_min_area = max(1.0, new_min_area)
        new_max_area = min(float(H * W) * 0.25, new_max_area)
        new_min_circ = max(0.0, new_min_circ)
        new_max_circ = min(1.0, new_max_circ)
        new_min_asp  = max(0.0, new_min_asp)
        new_max_asp  = min(50.0, new_max_asp)
        new_min_sol  = max(0.0, new_min_sol)

        # ── Apply ────────────────────────────────────────────────────
        self.params['min_area']  = new_min_area
        self.params['max_area']  = new_max_area
        self.min_circularity     = new_min_circ
        self.max_circularity     = new_max_circ
        self.min_aspect_ratio    = new_min_asp
        self.max_aspect_ratio    = new_max_asp
        self.min_solidity        = new_min_sol

        # Sync to parameter panel sliders
        if hasattr(self, 'parameter_panel'):
            pp = self.parameter_panel
            for name, val in [
                ('min_area',         new_min_area),
                ('max_area',         new_max_area),
                ('min_circularity',  new_min_circ),
                ('max_circularity',  new_max_circ),
                ('min_aspect_ratio', new_min_asp),
                ('max_aspect_ratio', new_max_asp),
                ('min_solidity',     new_min_sol),
            ]:
                if name in pp.sliders:
                    pp.sliders[name].set(val)

        # Exit pick/reject mode and run normal detection with new bounds
        self.selection_mode = False
        self.accepted_indices.clear()
        self.rejected_indices.clear()
        self._update_pick_reject_buttons(active=False)

        # Restore instruction bar
        if hasattr(self, 'instruction_label'):
            self.instruction_label.config(
                text="💡 Click on a particle to analyze and get parameter suggestions",
                bg=self.COLORS['primary']
            )

        self.update_visualization()

        um2 = self.pixel_size_um ** 2
        messagebox.showinfo(
            "✨ Normalized",
            f"Parameters set from {len(self.accepted_indices | {-1}) - 1 + len(acc_props)} "
            f"accepted  /  {len(rej_props)} rejected objects:\n\n"
            f"Area:          {new_min_area:.0f} – {new_max_area:.0f} px\n"
            f"               ({new_min_area * um2:.2f} – {new_max_area * um2:.2f} µm²)\n"
            f"Circularity:   {new_min_circ:.2f} – {new_max_circ:.2f}\n"
            f"Aspect ratio:  {new_min_asp:.2f} – {new_max_asp:.2f}\n"
            f"Solidity ≥     {new_min_sol:.2f}\n\n"
            f"You can now fine-tune with the parameter panel.",
            parent=self.root
        )

    def cancel_pick_reject(self):
        """Restore original params and exit pick/reject mode."""
        if self._pre_broad_params:
            self.params           = self._pre_broad_params['params'].copy()
            self.min_circularity  = self._pre_broad_params['min_circularity']
            self.max_circularity  = self._pre_broad_params['max_circularity']
            self.min_aspect_ratio = self._pre_broad_params['min_aspect_ratio']
            self.max_aspect_ratio = self._pre_broad_params['max_aspect_ratio']
            self.min_solidity     = self._pre_broad_params['min_solidity']

            # Sync restored values back to parameter panel sliders
            if hasattr(self, 'parameter_panel'):
                pp = self.parameter_panel
                for name, val in [
                    ('min_area',         self.params['min_area']),
                    ('max_area',         self.params['max_area']),
                    ('min_circularity',  self.min_circularity),
                    ('max_circularity',  self.max_circularity),
                    ('min_aspect_ratio', self.min_aspect_ratio),
                    ('max_aspect_ratio', self.max_aspect_ratio),
                    ('min_solidity',     self.min_solidity),
                ]:
                    if name in pp.sliders:
                        pp.sliders[name].set(val)

        self.selection_mode = False
        self.accepted_indices.clear()
        self.rejected_indices.clear()
        self._update_pick_reject_buttons(active=False)

        # Restore instruction bar
        if hasattr(self, 'instruction_label'):
            self.instruction_label.config(
                text="💡 Click on a particle to analyze and get parameter suggestions",
                bg=self.COLORS['primary']
            )

        self.update_visualization()

    def _compute_contour_properties(self, contours: list) -> list:
        """Return a list of shape-property dicts for the given contours."""
        props = []
        for cnt in contours:
            area      = float(cv2.contourArea(cnt))
            if area <= 0:
                continue
            perimeter = float(cv2.arcLength(cnt, True))
            circ      = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
            x, y, w, h = cv2.boundingRect(cnt)
            aspect    = float(w) / h if h > 0 else 0.0
            hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
            solidity  = area / hull_area if hull_area > 0 else 0.0
            props.append({
                'area':         area,
                'circularity':  circ,
                'aspect_ratio': aspect,
                'solidity':     solidity,
            })
        return props

    # ==================================================================
    # Image click handler — normal + pick/reject
    # ==================================================================

    def on_image_click(self, event):
        if event.inaxes != self.ax_image or event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        # ── Pick / Reject mode ────────────────────────────────────────
        if self.selection_mode == 'pick_reject':
            hit = False
            for i, cnt in enumerate(self.contours):
                if cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0:
                    if event.button == 1:           # left-click  → accept
                        if i in self.accepted_indices:
                            # Toggle back to unclassified
                            self.accepted_indices.discard(i)
                        else:
                            self.accepted_indices.add(i)
                            self.rejected_indices.discard(i)
                    elif event.button == 3:         # right-click → reject
                        if i in self.rejected_indices:
                            # Toggle back to unclassified
                            self.rejected_indices.discard(i)
                        else:
                            self.rejected_indices.add(i)
                            self.accepted_indices.discard(i)
                    hit = True
                    break

            self._update_image_display_pick_reject()
            self._show_pick_reject_status()
            n_acc = len(self.accepted_indices)
            n_rej = len(self.rejected_indices)
            self.contour_count_label.config(
                text=f"PICK/REJECT — ✅{n_acc}  ❌{n_rej}  🟡{len(self.contours)-n_acc-n_rej}"
            )
            return

        # ── Normal mode ───────────────────────────────────────────────
        clicked_contour = None
        for cnt in self.contours:
            if cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0:
                clicked_contour = cnt
                break

        if clicked_contour is not None:
            self._analyze_particle(clicked_contour)
        else:
            self._analyze_missed_particle(x, y)

    def _analyze_missed_particle(self, x: int, y: int):
        roi_size = 50
        h, w = self.original_image.shape[:2]

        x1 = max(0, x - roi_size)
        x2 = min(w, x + roi_size)
        y1 = max(0, y - roi_size)
        y2 = min(h, y + roi_size)

        roi_original  = self.original_image[y1:y2, x1:x2]
        roi_binary    = self.binary_mask[y1:y2, x1:x2]
        roi_processed = self.processed_image[y1:y2, x1:x2]

        analysis = self._analyze_roi_characteristics(
            roi_original, roi_binary, roi_processed, (x - x1, y - y1)
        )

        if analysis is None:
            self.target_analysis_text.delete('1.0', tk.END)
            self.target_analysis_text.insert('1.0', "❌ No particle-like structure detected in clicked region")
            self.target_analysis_text.tag_config("error", foreground="red")
            self.target_analysis_text.tag_add("error", "1.0", "end")
            return

        suggestions = self._generate_missed_particle_suggestions(analysis)
        self.current_suggestions = suggestions
        self._display_missed_particle_analysis(analysis, suggestions)

    def _analyze_roi_characteristics(
        self,
        roi_original: np.ndarray,
        roi_binary: np.ndarray,
        roi_processed: np.ndarray,
        click_offset: Tuple[int, int]
    ) -> Optional[Dict[str, Any]]:

        mean_intensity = float(np.mean(roi_original))
        std_intensity  = float(np.std(roi_original))

        if std_intensity < 10:
            return None

        if self.threshold_mode == "otsu" and not self.use_intensity_threshold:
            _, test_binary = cv2.threshold(
                roi_processed, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif self.use_intensity_threshold:
            test_threshold = max(0, int(self.manual_threshold) + 20)
            _, test_binary = cv2.threshold(
                roi_original, test_threshold, 255, cv2.THRESH_BINARY_INV
            )
        else:
            test_threshold = max(0, self.manual_threshold - 30)
            _, test_binary = cv2.threshold(
                roi_processed, test_threshold, 255, cv2.THRESH_BINARY
            )

        test_contours, _ = cv2.findContours(
            test_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not test_contours:
            return None

        click_x, click_y = click_offset
        min_dist        = float('inf')
        closest_contour = None

        for cnt in test_contours:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx   = int(M['m10'] / M['m00'])
                cy   = int(M['m01'] / M['m00'])
                dist = np.sqrt((cx - click_x) ** 2 + (cy - click_y) ** 2)
                if dist < min_dist:
                    min_dist        = dist
                    closest_contour = cnt

        if closest_contour is None or cv2.contourArea(closest_contour) < 5:
            return None

        area_px   = float(cv2.contourArea(closest_contour))
        perimeter = float(cv2.arcLength(closest_contour, True))

        x, y, w, h = cv2.boundingRect(closest_contour)
        aspect_ratio = float(w) / h if h > 0 else 0.0

        if perimeter > 0:
            circularity = (4 * np.pi * area_px) / (perimeter ** 2)
        else:
            circularity = 0.0

        hull      = cv2.convexHull(closest_contour)
        hull_area = float(cv2.contourArea(hull))
        solidity  = area_px / hull_area if hull_area > 0 else 0.0

        mask = np.zeros(roi_original.shape, dtype=np.uint8)
        cv2.drawContours(mask, [closest_contour], -1, 255, -1)
        particle_mean = float(np.mean(roi_original[mask > 0]))

        return {
            'area_px':           area_px,
            'perimeter':         perimeter,
            'aspect_ratio':      aspect_ratio,
            'circularity':       circularity,
            'solidity':          solidity,
            'mean_intensity':    mean_intensity,
            'particle_intensity':particle_mean,
            'std_intensity':     std_intensity,
            'test_contour':      closest_contour,
            'test_binary':       test_binary,
            'roi_size':          roi_original.shape
        }

    def _generate_missed_particle_suggestions(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:

        suggestions    = {}
        area_px        = analysis['area_px']
        circularity    = analysis['circularity']
        aspect_ratio   = analysis['aspect_ratio']
        solidity       = analysis['solidity']
        std_intensity  = analysis['std_intensity']

        current_min = self.params['min_area']
        current_max = self.params['max_area']

        if area_px < current_min:
            suggestions['min_area'] = int(area_px * 0.7)
        if area_px > current_max:
            suggestions['max_area'] = int(area_px * 1.3)
        
        # ── Shape filters ─────────────────────────────────────────────
        if circularity < self.min_circularity:
            suggestions['min_circularity'] = max(0.0, circularity - 0.1)
        if aspect_ratio > self.max_aspect_ratio:
            suggestions['max_aspect_ratio'] = min(20.0, aspect_ratio + 1.0)
        elif aspect_ratio < self.min_aspect_ratio:
            suggestions['min_aspect_ratio'] = max(0.1, aspect_ratio - 0.1)
        if solidity < self.min_solidity:
            suggestions['min_solidity'] = max(0.0, solidity - 0.1)
        if circularity < 0.5:
            suggestions['dilate_iterations'] = min(5, self.params['dilate_iterations'] + 1)

        # ── Gaussian — only when contrast is very low ─────────────────
        if std_intensity < 20:
            suggestions['gaussian_sigma'] = min(20.0, self.params['gaussian_sigma'] + 2.0)

        # ── Threshold — only when NO filter issue was found ───────────
        # If area/shape suggestions already exist, thresholding is fine;
        # the particle is visible but being rejected by the filters.
        # OTSU/adaptive adapt automatically — never suggest switching modes.
        has_filter_issue = any(k in suggestions for k in (
            'min_area', 'max_area', 'min_circularity',
            'max_aspect_ratio', 'min_aspect_ratio', 'min_solidity'
        ))

        if not has_filter_issue:
            if self.use_intensity_threshold:
                # Raise threshold to catch slightly darker objects
                suggestions['manual_threshold'] = min(255, int(self.manual_threshold) + 15)
            elif self.threshold_mode == "manual":
                # Lower manual threshold to increase sensitivity
                suggestions['manual_threshold'] = max(0, self.manual_threshold - 20)
            # OTSU / adaptive: trust the algorithm — it already adapts to the
            # image histogram. Only the gaussian suggestion above applies here.

        return suggestions


    def _display_missed_particle_analysis(
        self,
        analysis: Dict[str, Any],
        suggestions: Dict[str, Any]
    ):
        area_px  = analysis['area_px']
        um2_per_px2 = self.pixel_size_um ** 2
        area_um2 = area_px * um2_per_px2

        analysis_text = (
            f"🔍 MISSED PARTICLE DETECTED\n\n"
            f"📊 Characteristics:\n"
            f"• Area: {area_px:.1f} px² ({area_um2:.2f} µm²)\n"
            f"• Perimeter: {analysis['perimeter']:.1f} px\n"
            f"• Aspect Ratio: {analysis['aspect_ratio']:.2f}\n"
            f"• Circularity: {analysis['circularity']:.3f}\n"
            f"• Solidity: {analysis['solidity']:.3f}\n"
            f"• Mean Intensity: {analysis['mean_intensity']:.1f}\n"
            f"• Contrast (std): {analysis['std_intensity']:.1f}\n\n"
            f"💡 SUGGESTIONS TO DETECT:\n"
        )

        for param, value in suggestions.items():
            if 'area' in param:
                value_um2 = value * um2_per_px2
                analysis_text += f"• {param}: {value} px ({value_um2:.2f} µm²)\n"
            else:
                analysis_text += f"• {param}: {value}\n"

        analysis_text += f"\n✅ Click 'APPLY SUGGESTIONS' to enable detection"

        self.target_analysis_text.delete('1.0', tk.END)
        self.target_analysis_text.insert('1.0', analysis_text)
        self.target_analysis_text.tag_add("warning", "1.0", "end")
        self.target_analysis_text.tag_config("warning", foreground=self.COLORS['warning'])

        self._highlight_missed_particle(analysis['test_contour'])

    def _highlight_missed_particle(self, contour: np.ndarray):
        self.ax_image.clear()

        if len(self.original_image.shape) == 2:
            display = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB)
        else:
            display = self.original_image.copy()

        cv2.drawContours(display, self.contours, -1, (0, 255, 0), 2)
        cv2.drawContours(display, [contour], -1, (255, 165, 0), 3)

        self.ax_image.imshow(display)
        self.ax_image.set_title(
            "Original + Contours (Orange = Missed Particle)",
            fontsize=12, fontweight='bold', pad=10
        )
        self.ax_image.axis("off")
        self.canvas_image.draw()

    def _analyze_particle(self, contour: np.ndarray):
        area_px   = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        circularity  = 4 * np.pi * area_px / (perimeter ** 2) if perimeter > 0 else 0

        hull      = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity  = area_px / hull_area if hull_area > 0 else 0.0

        um2_per_px2 = self.pixel_size_um ** 2
        area_um2    = area_px * um2_per_px2

        suggestions = self._generate_suggestions(area_px, circularity, aspect_ratio)
        self.current_suggestions = suggestions

        self.target_analysis_text.delete('1.0', tk.END)

        self.target_analysis_text.insert('end', "🎯 Target Particle Analysis\n\n", "header")
        self.target_analysis_text.insert('end', "📏 Measurements:\n", "section")
        self.target_analysis_text.insert('end', f"• Area: {area_px:.1f} px² ({area_um2:.2f} µm²)\n")
        self.target_analysis_text.insert('end', f"• Perimeter: {perimeter:.1f} px\n")
        self.target_analysis_text.insert('end', f"• Aspect Ratio: {aspect_ratio:.2f}\n")
        self.target_analysis_text.insert('end', f"• Circularity: {circularity:.3f}\n")
        self.target_analysis_text.insert('end', f"• Solidity: {solidity:.3f}\n\n")

        if suggestions:
            self.target_analysis_text.insert('end', "📊 Suggestions:\n", "section")
            for param, value in suggestions.items():
                if 'area' in param:
                    value_um2 = value * um2_per_px2
                    self.target_analysis_text.insert('end', f"• {param}: {value} px ({value_um2:.2f} µm²)\n", "suggestion")
                else:
                    self.target_analysis_text.insert('end', f"• {param}: {value}\n", "suggestion")

        self.target_analysis_text.tag_config("header",     foreground=self.COLORS['success'],  font=("Segoe UI", 10, "bold"))
        self.target_analysis_text.tag_config("section",    foreground=self.COLORS['header'],   font=("Segoe UI", 9,  "bold"))
        self.target_analysis_text.tag_config("suggestion", foreground=self.COLORS['primary'],  font=("Segoe UI", 8,  "bold"))

    def _generate_suggestions(
        self,
        area_px: float,
        circularity: float,
        aspect_ratio: float
    ) -> Dict[str, Any]:
        suggestions = {}

        current_min = self.params['min_area']
        current_max = self.params['max_area']

        if area_px < current_min:
            suggestions['min_area'] = int(area_px * 0.8)
        if area_px > current_max:
            suggestions['max_area'] = int(area_px * 1.2)
        if circularity < self.min_circularity:
            suggestions['min_circularity'] = max(0.0, circularity - 0.05)
        if circularity > self.max_circularity:
            suggestions['max_circularity'] = min(1.0, circularity + 0.05)
        if aspect_ratio > self.max_aspect_ratio:
            suggestions['max_aspect_ratio'] = min(20.0, aspect_ratio + 0.5)
        elif aspect_ratio < self.min_aspect_ratio:
            suggestions['min_aspect_ratio'] = max(0.1, aspect_ratio - 0.5)

        return suggestions


    def apply_suggestions(self, event=None):
        if not self.current_suggestions:
            self.target_analysis_text.delete('1.0', tk.END)
            self.target_analysis_text.insert('1.0', "No suggestions to apply")
            return

        panel_sliders = {}
        if hasattr(self, 'parameter_panel'):
            panel_sliders = self.parameter_panel.sliders

        for param, value in self.current_suggestions.items():
            if param == 'threshold_mode':
                # ── Set internal state ────────────────────────────────────
                self.threshold_mode = value
                self.use_intensity_threshold = (value == "intensity")
                if self.use_intensity_threshold:
                    self.intensity_threshold_value = float(self.manual_threshold)

                # ── Update ParameterPanel tkinter widgets ─────────────────
                if hasattr(self, 'parameter_panel'):
                    pp = self.parameter_panel
                    if hasattr(pp, 'thresh_mode_btn'):
                        pp.thresh_mode_btn.config(text=value.upper())
                    if hasattr(pp, 'manual_threshold_container'):
                        if value in ("manual", "intensity"):
                            pp.manual_threshold_container.pack(fill=tk.X, pady=4)
                        else:
                            pp.manual_threshold_container.pack_forget()
                    if hasattr(pp, 'intensity_info_label'):
                        if value == "intensity":
                            pp.intensity_info_label.pack(fill=tk.X, pady=2)
                        else:
                            pp.intensity_info_label.pack_forget()

                # ── Update legacy matplotlib button if it exists ──────────
                if hasattr(self, 'btn_thresh_mode'):
                    self.btn_thresh_mode.label.set_text(
                        f"THRESHOLD\n{value.upper()}"
                    )

            elif param in panel_sliders:
                panel_sliders[param].set(value)
            elif param in self.sliders:
                self.sliders[param].set_val(value)
            elif param in ['min_circularity', 'max_circularity', 'min_aspect_ratio',
                        'max_aspect_ratio', 'min_solidity']:
                setattr(self, param, value)
            elif param == 'manual_threshold':
                self.manual_threshold = value
                if self.use_intensity_threshold:
                    self.intensity_threshold_value = float(value)
                if 'manual_threshold' in panel_sliders:
                    panel_sliders['manual_threshold'].set(value)
            elif param == 'gaussian_sigma':
                self.params['gaussian_sigma'] = value
                if 'gaussian_sigma' in panel_sliders:
                    panel_sliders['gaussian_sigma'].set(value)
            elif param in ['dilate_iterations', 'erode_iterations', 'min_area', 'max_area']:
                self.params[param] = value
                if param in panel_sliders:
                    panel_sliders[param].set(value)

        self.current_suggestions = {}

        self.target_analysis_text.delete('1.0', tk.END)
        self.target_analysis_text.insert('1.0', "✅ Suggestions applied! Processing...", "success")
        self.target_analysis_text.tag_config("success",
                                            foreground=self.COLORS['success'],
                                            font=("Segoe UI", 9, "bold"))
        self.update_visualization()


    def load_new_image(self):
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
            self.image_path      = new_path
            self.pixel_size_um, self.has_metadata = self._load_pixel_size()

            # Reset pick/reject state when a new image is loaded
            if self.selection_mode == 'pick_reject':
                self.cancel_pick_reject()
            else:
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
# SECTION 6: Main Menu (Pathogen Selection + Tuner Setup)
# ==================================================

class PathogenConfigManager:
    """Unified pathogen configuration manager with integrated tuner setup"""

    PATHOGENS = {
        "Proteus mirabilis": {
            "config_key": "proteus_mirabilis",
            "description": "Rod-shaped, flagellated bacterium",
            "common_in": "Catheter-associated infections",
        },
        "Klebsiella pneumoniae": {
            "config_key": "klebsiella_pneumoniae",
            "description": "Gram-negative, encapsulated bacterium",
            "common_in": "Healthcare-associated infections",
        },
        "Streptococcus mitis": {
            "config_key": "streptococcus_mitis",
            "description": "Gram-positive cocci in chains",
            "common_in": "Touch contamination",
        },
    }

    COLORS = {
        "bg": "#1e1e1e",
        "fg": "#ffffff",
        "accent": "#007acc",
        "button": "#2d2d2d",
        "button_hover": "#3e3e3e",
        "success": "#4ec9b0",
        "warning": "#ce9178",
        "error": "#f48771",
        "header": "#569cd6",
        "selected": "#094771",
        "muted": "#b9b9b9",
        "panel": "#232323",
        "panel_border": "#2f2f2f",
    }

    def __init__(self, root: "tk.Tk"):
        self.root = root
        self.root.title("🦠 Pathogen Segmentation Tuner Setup")
        self.root.geometry("980x760")
        self.root.minsize(980, 760)
        self.root.maxsize(980, 760)
        self.root.configure(**{"bg": self.COLORS["bg"]})

        self.selected_pathogen: "Optional[str]" = None
        self.image_path_var  = tk.StringVar()
        self.structure_var   = tk.StringVar(value="bacteria")
        self.mode_var        = tk.StringVar(value="DARK")

        self.pathogen_cards:      "Dict[str, tk.Frame]" = {}
        self.card_indicators:     "Dict[str, tk.Label]" = {}
        self.card_contents:       "Dict[str, tk.Frame]" = {}
        self.card_left_frames:    "Dict[str, tk.Frame]" = {}
        self.card_right_frames:   "Dict[str, tk.Frame]" = {}

        self.FONT_TITLE = ("Segoe UI", 22, "bold")
        self.FONT_H2    = ("Segoe UI", 11, "bold")
        self.FONT_BODY  = ("Segoe UI", 10)
        self.FONT_SMALL = ("Segoe UI", 9)
        self.FONT_TINY  = ("Segoe UI", 8, "italic")

        self._create_ui()
        self._center_window()

    def _center_window(self):
        self.root.update_idletasks()
        width  = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth()  // 2) - (width  // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _panel(self, parent, padx=14, pady=12):
        outer = tk.Frame(parent)
        outer["bg"] = self.COLORS["panel_border"]
        inner = tk.Frame(outer)
        inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        inner["bg"] = self.COLORS["panel"]
        content = tk.Frame(inner)
        content.pack(fill=tk.BOTH, expand=True, padx=padx, pady=pady)
        content["bg"] = self.COLORS["panel"]
        return outer, content

    def _hsep(self, parent, pady=14):
        sep = tk.Frame(parent, height=1)
        sep.pack(fill=tk.X, pady=pady)
        sep["bg"] = self.COLORS["panel_border"]

    def _section_title(self, parent, step_text: str, title: str):
        row = tk.Frame(parent)
        row.pack(fill=tk.X, pady=(0, 10))
        row["bg"] = parent["bg"] if isinstance(parent, tk.Widget) else self.COLORS["panel"]

        step = tk.Label(row, text=step_text, font=self.FONT_H2)
        step.pack(side=tk.LEFT)
        step["bg"] = row["bg"]
        step["fg"] = self.COLORS["success"]

        ttl = tk.Label(row, text=title, font=self.FONT_H2)
        ttl.pack(side=tk.LEFT, padx=(8, 0))
        ttl["bg"] = row["bg"]
        ttl["fg"] = self.COLORS["fg"]
        return row

    def _create_ui(self):
        root_pad = tk.Frame(self.root)
        root_pad.pack(fill=tk.BOTH, expand=True, padx=18, pady=16)
        root_pad["bg"] = self.COLORS["bg"]

        self._create_header(root_pad)

        main_grid = tk.Frame(root_pad)
        main_grid.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        main_grid["bg"] = self.COLORS["bg"]

        main_grid.grid_columnconfigure(0, weight=3, uniform="cols")
        main_grid.grid_columnconfigure(1, weight=2, uniform="cols")
        main_grid.grid_rowconfigure(0, weight=1)

        left_outer, left_panel = self._panel(main_grid, padx=14, pady=12)
        left_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left_panel["bg"] = self.COLORS["panel"]
        self._create_pathogen_section(left_panel)

        right_outer, right_panel = self._panel(main_grid, padx=14, pady=12)
        right_outer.grid(row=0, column=1, sticky="nsew")
        right_panel["bg"] = self.COLORS["panel"]
        self._create_config_section(right_panel)

        self._hsep(root_pad, pady=14)

        image_outer, image_panel = self._panel(root_pad, padx=14, pady=12)
        image_outer.pack(fill=tk.X)
        self._create_image_section(image_panel)

        self._hsep(root_pad, pady=14)
        self._create_action_buttons(root_pad)

    def _create_header(self, parent):
        header = tk.Frame(parent)
        header.pack(fill=tk.X)
        header["bg"] = self.COLORS["bg"]

        title_row = tk.Frame(header)
        title_row.pack()
        title_row["bg"] = self.COLORS["bg"]

        title = tk.Label(title_row, text="🦠  Pathogen Segmentation Tuner", font=self.FONT_TITLE)
        title.pack()
        title["bg"] = self.COLORS["bg"]
        title["fg"] = self.COLORS["header"]

        subtitle = tk.Label(
            header,
            text="Configure image analysis parameters for peritoneal dialysis pathogens",
            font=self.FONT_BODY,
        )
        subtitle.pack(pady=(6, 0))
        subtitle["bg"] = self.COLORS["bg"]
        subtitle["fg"] = self.COLORS["muted"]

    def _create_pathogen_section(self, parent):
        parent["bg"] = self.COLORS["panel"]
        self._section_title(parent, "1", "Select pathogen")

        hint = tk.Label(parent, text="Click a card to choose the pathogen profile used for tuning.", font=self.FONT_SMALL)
        hint.pack(anchor=tk.W, pady=(0, 12))
        hint["bg"] = self.COLORS["panel"]
        hint["fg"] = self.COLORS["muted"]

        cards_frame = tk.Frame(parent)
        cards_frame.pack(fill=tk.BOTH, expand=True)
        cards_frame["bg"] = self.COLORS["panel"]

        for pathogen_name, info in self.PATHOGENS.items():
            card = self._create_pathogen_card(cards_frame, pathogen_name, info)
            card.pack(fill=tk.X, pady=(0, 10))

    def _create_pathogen_card(self, parent, pathogen_name: str, info: dict):
        card = tk.Frame(parent, relief=tk.FLAT, borderwidth=0, cursor="hand2")
        card["bg"] = self.COLORS["button"]
        self.pathogen_cards[pathogen_name] = card

        def select_pathogen(event=None):
            self._select_pathogen(pathogen_name)

        border = tk.Frame(card)
        border.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        border["bg"] = self.COLORS["panel_border"]
        border.bind("<Button-1>", select_pathogen)

        content = tk.Frame(border)
        content.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        content["bg"] = self.COLORS["button"]
        content.bind("<Button-1>", select_pathogen)

        card.bind("<Enter>", lambda e: self._hover_card(pathogen_name, True))
        card.bind("<Leave>", lambda e: self._hover_card(pathogen_name, False))
        border.bind("<Enter>", lambda e: self._hover_card(pathogen_name, True))
        border.bind("<Leave>", lambda e: self._hover_card(pathogen_name, False))
        content.bind("<Enter>", lambda e: self._hover_card(pathogen_name, True))
        content.bind("<Leave>", lambda e: self._hover_card(pathogen_name, False))

        self.card_contents[pathogen_name] = content

        left = tk.Frame(content)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left["bg"] = self.COLORS["button"]
        left.bind("<Button-1>", select_pathogen)
        self.card_left_frames[pathogen_name] = left

        name = tk.Label(left, text=f"🔬 {pathogen_name}", font=("Segoe UI", 12, "bold"), anchor=tk.W)
        name.pack(anchor=tk.W)
        name["bg"] = self.COLORS["button"]
        name["fg"] = self.COLORS["success"]
        name.bind("<Button-1>", select_pathogen)

        desc = tk.Label(left, text=info["description"], font=self.FONT_SMALL, anchor=tk.W)
        desc.pack(anchor=tk.W, pady=(4, 0))
        desc["bg"] = self.COLORS["button"]
        desc["fg"] = self.COLORS["fg"]
        desc.bind("<Button-1>", select_pathogen)

        common = tk.Label(left, text=f"📌 {info['common_in']}", font=self.FONT_TINY, anchor=tk.W)
        common.pack(anchor=tk.W, pady=(3, 0))
        common["bg"] = self.COLORS["button"]
        common["fg"] = self.COLORS["warning"]
        common.bind("<Button-1>", select_pathogen)

        right = tk.Frame(content)
        right.pack(side=tk.RIGHT, padx=(10, 0))
        right["bg"] = self.COLORS["button"]
        right.bind("<Button-1>", select_pathogen)
        self.card_right_frames[pathogen_name] = right

        indicator = tk.Label(right, text="○", font=("Segoe UI", 18))
        indicator.pack(padx=6)
        indicator["bg"] = self.COLORS["button"]
        indicator["fg"] = self.COLORS["muted"]
        indicator.bind("<Button-1>", select_pathogen)
        self.card_indicators[pathogen_name] = indicator

        return card

    def _hover_card(self, pathogen_name: str, is_hover: bool):
        if self.selected_pathogen == pathogen_name:
            return
        bg      = self.COLORS["button_hover"] if is_hover else self.COLORS["button"]
        card    = self.pathogen_cards[pathogen_name]
        content = self.card_contents[pathogen_name]
        left    = self.card_left_frames[pathogen_name]
        right   = self.card_right_frames[pathogen_name]

        card["bg"]    = bg
        content["bg"] = bg
        left["bg"]    = bg
        right["bg"]   = bg
        for w in left.winfo_children():
            try:
                w["bg"] = bg
            except Exception:
                pass
        for w in right.winfo_children():
            try:
                w["bg"] = bg
            except Exception:
                pass

    def _select_pathogen(self, pathogen_name: str):
        if self.selected_pathogen:
            old = self.selected_pathogen
            for part in (self.pathogen_cards[old], self.card_contents[old],
                         self.card_left_frames[old], self.card_right_frames[old]):
                part["bg"] = self.COLORS["button"]
            self.card_indicators[old]["text"] = "○"
            self.card_indicators[old]["fg"]   = self.COLORS["muted"]
            for w in self.card_left_frames[old].winfo_children():
                try:
                    w["bg"] = self.COLORS["button"]
                except Exception:
                    pass
            for w in self.card_right_frames[old].winfo_children():
                try:
                    w["bg"] = self.COLORS["button"]
                except Exception:
                    pass

        self.selected_pathogen = pathogen_name
        for part in (self.pathogen_cards[pathogen_name], self.card_contents[pathogen_name],
                     self.card_left_frames[pathogen_name], self.card_right_frames[pathogen_name]):
            part["bg"] = self.COLORS["selected"]
        self.card_indicators[pathogen_name]["text"] = "●"
        self.card_indicators[pathogen_name]["fg"]   = self.COLORS["success"]
        for w in self.card_left_frames[pathogen_name].winfo_children():
            try:
                w["bg"] = self.COLORS["selected"]
            except Exception:
                pass
        for w in self.card_right_frames[pathogen_name].winfo_children():
            try:
                w["bg"] = self.COLORS["selected"]
            except Exception:
                pass
        print(f"✓ Selected pathogen: {pathogen_name}")

    def _create_image_section(self, parent):
        parent["bg"] = self.COLORS["panel"]
        self._section_title(parent, "2", "Select image")

        row = tk.Frame(parent)
        row.pack(fill=tk.X)
        row["bg"] = self.COLORS["panel"]
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=0)

        entry = tk.Entry(
            row, textvariable=self.image_path_var, font=self.FONT_BODY,
            insertbackground=self.COLORS["fg"], relief=tk.FLAT, bd=6,
        )
        entry.grid(row=0, column=0, sticky="ew", padx=(0, 10), ipady=6)
        entry["bg"] = self.COLORS["button"]
        entry["fg"] = self.COLORS["fg"]

        browse = tk.Button(
            row, text="📁 Browse…", font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, cursor="hand2", padx=18, pady=10,
            command=self._browse_image,
        )
        browse.grid(row=0, column=1, sticky="e")
        browse["bg"]             = self.COLORS["accent"]
        browse["fg"]             = "white"
        browse["activebackground"] = self.COLORS["header"]
        browse["activeforeground"] = "white"

        self.file_info_label = tk.Label(
            parent, text="No image selected", font=self.FONT_TINY, anchor=tk.W,
        )
        self.file_info_label.pack(anchor=tk.W, pady=(8, 0))
        self.file_info_label["bg"] = self.COLORS["panel"]
        self.file_info_label["fg"] = self.COLORS["warning"]

    def _browse_image(self):
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.image_path_var.set(filename)
            path = Path(filename)
            self.file_info_label["text"] = f"✓ {path.name} ({path.stat().st_size / 1024:.1f} KB)"
            self.file_info_label["fg"]   = self.COLORS["success"]

    def _create_config_section(self, parent):
        parent["bg"] = self.COLORS["panel"]
        self._section_title(parent, "3", "Configure options")

        g1 = tk.Frame(parent)
        g1.pack(fill=tk.X, pady=(0, 12))
        g1["bg"] = self.COLORS["panel"]

        l1 = tk.Label(g1, text="Structure type", font=self.FONT_H2)
        l1.pack(anchor=tk.W, pady=(0, 8))
        l1["bg"] = self.COLORS["panel"]
        l1["fg"] = self.COLORS["fg"]

        box1 = tk.Frame(g1, relief=tk.FLAT, borderwidth=0)
        box1.pack(fill=tk.X)
        box1["bg"] = self.COLORS["button"]

        rb1 = tk.Radiobutton(box1, text="🦠 Bacteria", variable=self.structure_var,
                              value="bacteria", font=self.FONT_BODY,
                              selectcolor=self.COLORS["selected"], cursor="hand2")
        rb1.pack(anchor=tk.W, padx=10, pady=8)
        rb1["bg"] = self.COLORS["button"]
        rb1["fg"] = self.COLORS["fg"]
        rb1["activebackground"] = self.COLORS["button_hover"]
        rb1["activeforeground"] = self.COLORS["fg"]

        rb2 = tk.Radiobutton(box1, text="🔵 Inclusions", variable=self.structure_var,
                              value="inclusions", font=self.FONT_BODY,
                              selectcolor=self.COLORS["selected"], cursor="hand2")
        rb2.pack(anchor=tk.W, padx=10, pady=(0, 10))
        rb2["bg"] = self.COLORS["button"]
        rb2["fg"] = self.COLORS["fg"]
        rb2["activebackground"] = self.COLORS["button_hover"]
        rb2["activeforeground"] = self.COLORS["fg"]

        g2 = tk.Frame(parent)
        g2.pack(fill=tk.X)
        g2["bg"] = self.COLORS["panel"]

        l2 = tk.Label(g2, text="Particle mode", font=self.FONT_H2)
        l2.pack(anchor=tk.W, pady=(0, 8))
        l2["bg"] = self.COLORS["panel"]
        l2["fg"] = self.COLORS["fg"]

        box2 = tk.Frame(g2, relief=tk.FLAT, borderwidth=0)
        box2.pack(fill=tk.X)
        box2["bg"] = self.COLORS["button"]

        rb3 = tk.Radiobutton(box2, text="⚫ Dark particles on bright background",
                              variable=self.mode_var, value="DARK", font=self.FONT_BODY,
                              selectcolor=self.COLORS["selected"], cursor="hand2")
        rb3.pack(anchor=tk.W, padx=10, pady=8)
        rb3["bg"] = self.COLORS["button"]
        rb3["fg"] = self.COLORS["fg"]
        rb3["activebackground"] = self.COLORS["button_hover"]
        rb3["activeforeground"] = self.COLORS["fg"]

        rb4 = tk.Radiobutton(box2, text="⚪ Bright particles on dark background",
                              variable=self.mode_var, value="BRIGHT", font=self.FONT_BODY,
                              selectcolor=self.COLORS["selected"], cursor="hand2")
        rb4.pack(anchor=tk.W, padx=10, pady=(0, 10))
        rb4["bg"] = self.COLORS["button"]
        rb4["fg"] = self.COLORS["fg"]
        rb4["activebackground"] = self.COLORS["button_hover"]
        rb4["activeforeground"] = self.COLORS["fg"]

    def _create_action_buttons(self, parent):
        row = tk.Frame(parent)
        row.pack(fill=tk.X)
        row["bg"] = self.COLORS["bg"]

        info_btn = tk.Button(
            row, text="ℹ️ About", font=self.FONT_BODY,
            relief=tk.FLAT, cursor="hand2", padx=18, pady=10,
            command=self._show_about,
        )
        info_btn.pack(side=tk.LEFT)
        info_btn["bg"] = self.COLORS["button"]
        info_btn["fg"] = self.COLORS["fg"]
        info_btn["activebackground"] = self.COLORS["button_hover"]
        info_btn["activeforeground"] = self.COLORS["fg"]

        right = tk.Frame(row)
        right.pack(side=tk.RIGHT)
        right["bg"] = self.COLORS["bg"]

        exit_btn = tk.Button(
            right, text="❌ Exit", font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, cursor="hand2", padx=18, pady=10,
            command=self._exit_application,
        )
        exit_btn.pack(side=tk.LEFT, padx=(0, 10))
        exit_btn["bg"] = self.COLORS["error"]
        exit_btn["fg"] = "white"
        exit_btn["activebackground"] = "#d67060"
        exit_btn["activeforeground"] = "white"

        start_btn = tk.Button(
            right, text="✅ Start tuner", font=("Segoe UI", 11, "bold"),
            relief=tk.RAISED, cursor="hand2", padx=22, pady=10,
            command=self._start_tuner,
        )
        start_btn.pack(side=tk.LEFT)
        start_btn["bg"] = self.COLORS["success"]
        start_btn["fg"] = "white"
        start_btn["activebackground"] = "#3da88a"
        start_btn["activeforeground"] = "white"

    def _validate_inputs(self) -> bool:
        errors = []
        if not self.selected_pathogen:
            errors.append("• Please select a pathogen")
        if not self.image_path_var.get():
            errors.append("• Please select an image file")
        elif not Path(self.image_path_var.get()).exists():
            errors.append("• Selected image file does not exist")

        if errors:
            messagebox.showerror("Validation Error", "Please fix:\n\n" + "\n".join(errors))
            return False
        return True

    def _start_tuner(self):
        if not self._validate_inputs():
            return

        assert self.selected_pathogen is not None

        print(f"\n{'=' * 60}")
        print("STARTING SEGMENTATION TUNER")
        print(f"{'=' * 60}")
        print(f"Pathogen: {self.selected_pathogen}")
        print(f"Image: {Path(self.image_path_var.get()).name}")
        print(f"Structure: {self.structure_var.get()}")
        print(f"Mode: {self.mode_var.get()}")
        print(f"{'=' * 60}\n")

        self.root.destroy()

        try:
            tuner_root = tk.Tk()

            def return_to_menu():
                main()

            _tuner = SegmentationTuner(
                root=tuner_root,
                image_path=self.image_path_var.get(),
                bacterium=self.selected_pathogen,
                structure=self.structure_var.get(),
                mode=self.mode_var.get(),
                return_callback=return_to_menu,
            )
            tuner_root.mainloop()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start tuner:\n{str(e)}")
            import traceback
            traceback.print_exc()
            main()

    def _show_about(self):
        about_text = """
Peritoneal Dialysis Pathogen Segmentation Tuner
Version 2.0

Interactive parameter tuning for image analysis of:

• Proteus mirabilis
• Klebsiella pneumoniae
• Streptococcus mitis

Features:
🎨 Real-time segmentation preview
📊 Histogram analysis
🎯 Click-to-analyze particles
🎯 Pick / Reject / Normalize workflow
💾 Save configurations

© 2026 Pathogen Analysis Suite
        """
        messagebox.showinfo("About", about_text.strip())

    def _exit_application(self):
        if messagebox.askyesno("Confirm Exit", "Are you sure you want to exit?"):
            print("\n👋 Exiting Pathogen Configuration Manager")
            self.root.quit()
            self.root.destroy()


# ==================================================
# SECTION 7: Main Entry Point
# ==================================================

def main():
    root = tk.Tk()
    app  = PathogenConfigManager(root)
    root.mainloop()


if __name__ == "__main__":
    main()