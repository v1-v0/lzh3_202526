"""
Integrated Pathogen Configuration Manager
Combines main menu, config management, and segmentation tuner
"""

from logging import config
import os
import sys
import json
import re
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
from bacteria_registry import registry as _bacteria_registry


# ==================================================
# SECTION 0.5: Responsive UI Configuration
# ==================================================

class UIScaler:
    """Manages responsive UI scaling"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.base_width = 1920
        self.base_height = 1080
        self.min_width = 1440
        self.min_height = 900

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
        height = int(self.base_height * 0.95)
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

        _um2 = self.tuner.pixel_size_um ** 2
        self.default_params = {
            'gaussian_sigma':         self.tuner.params['gaussian_sigma'],
            'manual_threshold':       self.tuner.manual_threshold,
            'morph_kernel_size':      self.tuner.morph_kernel_size,
            'morph_iterations':       self.tuner.morph_iterations,
            'min_area':               self.tuner.params['min_area'] * _um2,   # µm²
            'max_area':               self.tuner.params['max_area'] * _um2,   # µm²
            'dilate_iterations':      self.tuner.params['dilate_iterations'],
            'erode_iterations':       self.tuner.params['erode_iterations'],
            'min_circularity':        self.tuner.min_circularity,
            'max_circularity':        self.tuner.max_circularity,
            'min_aspect_ratio':       self.tuner.min_aspect_ratio,
            'max_aspect_ratio':       self.tuner.max_aspect_ratio,
            'min_solidity':           self.tuner.min_solidity,
            'max_mean_intensity':     self.tuner.max_mean_intensity,
            'threshold_mode':         self.tuner.threshold_mode,
            'invert_image':           self.tuner.invert_image,
            'use_intensity_threshold':self.tuner.use_intensity_threshold,
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

        # Scrollable canvas
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
            background='#888888', troughcolor='#e0e0e0',
            bordercolor='#cccccc', arrowcolor='white',
            darkcolor='#666666', lightcolor='#aaaaaa', gripcount=0
        )
        style.map(
            "Custom.Vertical.TScrollbar",
            background=[('active', '#555555'), ('!active', '#888888')]
        )

        scrollbar = ttk.Scrollbar(
            scrollbar_frame, orient="vertical",
            command=self.canvas.yview,
            style="Custom.Vertical.TScrollbar"
        )
        scrollbar.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.scrollable_frame = tk.Frame(self.canvas, bg='#f8f9fa')
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>",   self._on_mousewheel)
        self.canvas.bind_all("<Button-5>",   self._on_mousewheel)

        self._create_sections()

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _snap_morph_kernel(self, v: float, min_val: int = 1, max_val: int = 15) -> int:
        v = int(round(v))
        v = max(min_val, min(max_val, v))
        if v % 2 == 0:
            down, up = v - 1, v + 1
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
            self.canvas.yview_scroll(1,  "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _create_sections(self):
        """
        Build all collapsible sections.
        Slider ranges and steps are taken directly from SegmentationTuner.SLIDER_META
        so that suggestion clamping and display precision are always in sync.
        """
        M   = SegmentationTuner.SLIDER_META
        _um2 = self.tuner.pixel_size_um ** 2

        # ── SEGMENTATION ─────────────────────────────────────────────
        seg = self._create_collapsible_section(
            "SEGMENTATION", SegmentationTuner.COLORS['primary'], expanded=True
        )

        self._add_slider_with_input(
            seg, "Gaussian σ", "gaussian_sigma",
            self.tuner.params['gaussian_sigma'],
            M['gaussian_sigma']['min'], M['gaussian_sigma']['max'],
            M['gaussian_sigma']['step']
        )

        self._add_threshold_controls(seg)

        self._add_slider_with_input(
            seg, "Morph Kernel", "morph_kernel_size",
            self.tuner.morph_kernel_size,
            M['morph_kernel_size']['min'], M['morph_kernel_size']['max'],
            M['morph_kernel_size']['step']
        )

        self._add_slider_with_input(
            seg, "Morph Iterations", "morph_iterations",
            self.tuner.morph_iterations,
            M['morph_iterations']['min'], M['morph_iterations']['max'],
            M['morph_iterations']['step']
        )

        # ── FILTERING ────────────────────────────────────────────────
        filt = self._create_collapsible_section(
            "FILTERING", SegmentationTuner.COLORS['info'], expanded=True
        )

        self._add_slider_with_input(
            filt, "Min Area (µm²)", "min_area",
            self.tuner.params['min_area'] * _um2,
            M['min_area']['min'], M['min_area']['max'],
            M['min_area']['step']
        )

        self._add_slider_with_input(
            filt, "Max Area (µm²)", "max_area",
            self.tuner.params['max_area'] * _um2,
            M['max_area']['min'], M['max_area']['max'],
            M['max_area']['step']
        )

        self._add_slider_with_input(
            filt, "Dilate Iterations", "dilate_iterations",
            self.tuner.params['dilate_iterations'],
            M['dilate_iterations']['min'], M['dilate_iterations']['max'],
            M['dilate_iterations']['step']
        )

        self._add_slider_with_input(
            filt, "Erode Iterations", "erode_iterations",
            self.tuner.params['erode_iterations'],
            M['erode_iterations']['min'], M['erode_iterations']['max'],
            M['erode_iterations']['step']
        )

        # ── SHAPE FILTERS ─────────────────────────────────────────────
        shape = self._create_collapsible_section(
            "SHAPE FILTERS", SegmentationTuner.COLORS['purple'], expanded=True
        )

        self._add_slider_with_input(
            shape, "Min Circularity", "min_circularity",
            self.tuner.min_circularity,
            M['min_circularity']['min'], M['min_circularity']['max'],
            M['min_circularity']['step']
        )

        self._add_slider_with_input(
            shape, "Max Circularity", "max_circularity",
            self.tuner.max_circularity,
            M['max_circularity']['min'], M['max_circularity']['max'],
            M['max_circularity']['step']
        )

        self._add_slider_with_input(
            shape, "Min Aspect Ratio", "min_aspect_ratio",
            self.tuner.min_aspect_ratio,
            M['min_aspect_ratio']['min'], M['min_aspect_ratio']['max'],
            M['min_aspect_ratio']['step']
        )

        self._add_slider_with_input(
            shape, "Max Aspect Ratio", "max_aspect_ratio",
            self.tuner.max_aspect_ratio,
            M['max_aspect_ratio']['min'], M['max_aspect_ratio']['max'],
            M['max_aspect_ratio']['step']
        )

        self._add_slider_with_input(
            shape, "Min Solidity", "min_solidity",
            self.tuner.min_solidity,
            M['min_solidity']['min'], M['min_solidity']['max'],
            M['min_solidity']['step']
        )

        self._add_slider_with_input(
            shape, "Max BF Intensity", "max_mean_intensity",
            self.tuner.max_mean_intensity,
            M['max_mean_intensity']['min'], M['max_mean_intensity']['max'],
            M['max_mean_intensity']['step']
        )

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
        arrow_label.bind("<Button-1>",  toggle)
        title_label.bind("<Button-1>",  toggle)

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
            initial_value = self._snap_morph_kernel(
                initial_value, min_val=int(min_val), max_val=int(max_val)
            )

        value_label = ttk.Label(
            slider_frame,
            text=self._format_value(initial_value, resolution),
            font=('Consolas', 9, 'bold'), foreground='#4EC9B0',
            background='#2D2D30', width=7
        )
        value_label.grid(row=0, column=1)

        entry_var = tk.StringVar(
            value=str(int(initial_value) if resolution >= 1 else initial_value)
        )

        updating = {'flag': False}

        slider = ttk.Scale(
            slider_frame, from_=min_val, to=max_val,
            orient=tk.HORIZONTAL, style='Custom.Horizontal.TScale'
        )
        slider.grid(row=0, column=0, sticky='ew', padx=(0, 10))

        self.sliders[param_name]      = slider
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
                    value = round(round(value / resolution) * resolution, 10)

                if param_name == 'morph_kernel_size':
                    value = self._snap_morph_kernel(
                        value, min_val=int(min_val), max_val=int(max_val)
                    )

                if param_name in self.value_labels:
                    self.value_labels[param_name].config(
                        text=self._format_value(value, resolution)
                    )

                _um2 = self.tuner.pixel_size_um ** 2
                params_dict_keys = ['gaussian_sigma', 'min_area', 'max_area',
                                    'dilate_iterations', 'erode_iterations']

                if param_name in ('min_area', 'max_area'):
                    self.tuner.params[param_name] = value / _um2       # µm² → px
                elif param_name in params_dict_keys:
                    self.tuner.params[param_name] = value
                elif hasattr(self.tuner, param_name):
                    setattr(self.tuner, param_name, value)

                if param_name == 'manual_threshold' and getattr(
                    self.tuner, 'use_intensity_threshold', False
                ):
                    self.tuner.intensity_threshold_value = float(value)

                slider.set(value)
                entry_var.set(
                    str(int(value)) if resolution >= 1 else str(value)
                )

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
        entry.bind('<Return>',   lambda e: on_entry_change())
        entry.bind('<FocusOut>', lambda e: on_entry_change())

    def _add_threshold_controls(self, parent):
        container = tk.Frame(parent, bg='white')
        container.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(
            container, text="Threshold Method:",
            font=("Segoe UI", 9), bg='white', fg='#555', anchor='w'
        ).pack(anchor="w")

        self.thresh_mode_btn = tk.Button(
            container,
            text=f"{self.tuner.threshold_mode.upper()}",
            font=("Segoe UI", 9, "bold"),
            bg=SegmentationTuner.COLORS['primary'],
            fg="white", relief=tk.FLAT, cursor="hand2", pady=6,
            command=self._cycle_threshold_mode
        )
        self.thresh_mode_btn.pack(fill=tk.X, pady=2)

        self.intensity_info_label = tk.Label(
            container,
            text="⚠ INTENSITY mode: pixels DARKER than threshold → foreground\n"
                 "(matches pipeline use_intensity_threshold=True)",
            font=("Segoe UI", 8), bg='white', fg='#e74c3c',
            anchor='w', justify=tk.LEFT
        )
        if self.tuner.threshold_mode == "intensity":
            self.intensity_info_label.pack(fill=tk.X, pady=2)

        self.manual_threshold_container = tk.Frame(container, bg='white')
        if self.tuner.threshold_mode in ("manual", "intensity"):
            self.manual_threshold_container.pack(fill=tk.X, pady=4)

        M = SegmentationTuner.SLIDER_META
        self._add_slider_with_input(
            self.manual_threshold_container, "Threshold",
            "manual_threshold",
            self.tuner.manual_threshold,
            M['manual_threshold']['min'], M['manual_threshold']['max'],
            M['manual_threshold']['step']
        )

    def _cycle_threshold_mode(self):
        modes = ["otsu", "manual", "adaptive", "intensity"]
        current_idx = (
            modes.index(self.tuner.threshold_mode)
            if self.tuner.threshold_mode in modes else 0
        )
        next_mode = modes[(current_idx + 1) % len(modes)]

        self.tuner.threshold_mode = next_mode
        self.thresh_mode_btn.config(text=next_mode.upper())

        if next_mode == "intensity":
            self.tuner.use_intensity_threshold   = True
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

        invert_color = (
            SegmentationTuner.COLORS['success']
            if self.tuner.invert_image
            else SegmentationTuner.COLORS['gray']
        )
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

    def _on_slider_change(self, param_name: str, value: str,
                          step: float, input_var: tk.StringVar):
        if hasattr(self, '_updating_slider') and self._updating_slider:
            return
        try:
            numeric_value = float(value)

            if step >= 1:
                numeric_value = int(round(numeric_value / step) * step)
            else:
                numeric_value = round(round(numeric_value / step) * step, 10)

            if param_name == 'morph_kernel_size':
                numeric_value = self._snap_morph_kernel(numeric_value, min_val=1, max_val=15)

            params_dict_keys = ['gaussian_sigma', 'min_area', 'max_area',
                                'dilate_iterations', 'erode_iterations']
            um2 = self.tuner.pixel_size_um ** 2

            if param_name in ('min_area', 'max_area'):
                current_value = self.tuner.params[param_name] * um2    # px → µm²
            elif param_name in params_dict_keys:
                current_value = self.tuner.params[param_name]
            elif hasattr(self.tuner, param_name):
                current_value = getattr(self.tuner, param_name)
            else:
                print(f"⚠️ Warning: self.tuner.{param_name} does not exist")
                return

            if numeric_value == current_value:
                return

            if param_name in ('min_area', 'max_area'):
                self.tuner.params[param_name] = numeric_value / um2    # µm² → px
            elif param_name in params_dict_keys:
                self.tuner.params[param_name] = numeric_value
            elif hasattr(self.tuner, param_name):
                setattr(self.tuner, param_name, numeric_value)

            if param_name == 'manual_threshold' and getattr(
                self.tuner, 'use_intensity_threshold', False
            ):
                self.tuner.intensity_threshold_value = float(numeric_value)

            print(f"🔄 {param_name}: {current_value} → {numeric_value}")

            if param_name in self.value_labels:
                self.value_labels[param_name].config(
                    text=self._format_value(numeric_value, step)
                )

            entry_value = (
                str(int(numeric_value)) if step >= 1 else str(numeric_value)
            )
            if input_var.get() != entry_value:
                input_var.set(entry_value)

            if param_name in self.sliders:
                current_sv = self.sliders[param_name].get()
                if abs(float(current_sv) - numeric_value) > 0.001:
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
            import traceback; traceback.print_exc()

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
        invert_color = (
            SegmentationTuner.COLORS['success']
            if self.tuner.invert_image
            else SegmentationTuner.COLORS['gray']
        )
        self.invert_btn.config(
            text=f"INVERT: {'ON' if self.tuner.invert_image else 'OFF'}",
            bg=invert_color
        )
        self.tuner.update_visualization()

    def _apply_suggestions(self):
        if not self.tuner.current_suggestions:
            messagebox.showinfo(
                "Info", "Click on a particle first to get suggestions",
                parent=self.winfo_toplevel()
            )
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

        config_key  = self.tuner.config_key
        config_file = Path("bacteria_configs") / f"{config_key}.json"

        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                config_data = json_data.get("config", json_data)

                use_intensity = bool(config_data.get('use_intensity_threshold', False))
                if use_intensity:
                    threshold_mode = 'intensity'
                    threshold_val  = float(config_data.get('intensity_threshold', 80.0))
                    self.tuner.use_intensity_threshold   = True
                    self.tuner.intensity_threshold_value = threshold_val
                else:
                    threshold_mode = config_data.get('threshold_mode', 'otsu')
                    threshold_val  = config_data.get('manual_threshold', 127)
                    self.tuner.use_intensity_threshold = False

                slider_mappings = {
                    'gaussian_sigma':     config_data.get('gaussian_sigma', 2.0),
                    'manual_threshold':   threshold_val,
                    'morph_kernel_size':  config_data.get('morph_kernel_size', 3),
                    'morph_iterations':   config_data.get('morph_iterations', 1),
                    'min_area':           config_data.get('min_area_um2', 3.0),      # µm²
                    'max_area':           config_data.get('max_area_um2', 100.0),    # µm²
                    'dilate_iterations':  config_data.get('dilate_iterations', 0),
                    'erode_iterations':   config_data.get('erode_iterations', 0),
                    'min_circularity':    config_data.get('min_circularity', 0.0),
                    'max_circularity':    config_data.get('max_circularity', 1.0),
                    'min_aspect_ratio':   config_data.get('min_aspect_ratio', 0.2),
                    'max_aspect_ratio':   config_data.get('max_aspect_ratio', 10.0),
                    'min_solidity':       config_data.get('min_solidity', 0.3),
                    'max_mean_intensity': config_data.get(
                        'max_mean_intensity_bf',
                        config_data.get('max_mean_intensity', 255.0)
                    ),
                }

                for pname, val in slider_mappings.items():
                    if pname in self.sliders:
                        self.sliders[pname].set(val)

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

                invert_val   = config_data.get('invert_image', False)
                self.tuner.invert_image = invert_val
                invert_color = (
                    SegmentationTuner.COLORS['success'] if invert_val
                    else SegmentationTuner.COLORS['gray']
                )
                self.invert_btn.config(
                    text=f"INVERT: {'ON' if invert_val else 'OFF'}",
                    bg=invert_color
                )

                self.tuner.update_visualization()
                messagebox.showinfo(
                    "Reset Complete",
                    f"Parameters reset to {pathogen_name.upper()} preset from config file!",
                    parent=self.winfo_toplevel()
                )
                return

            except Exception as e:
                print(f"❌ Failed to load config file for reset: {e}")
                import traceback; traceback.print_exc()

        # FALLBACK: use stored default_params
        print("ℹ️ No config file found, using default_params stored at initialization")

        for pname, default_value in self.default_params.items():
            if pname in self.sliders:
                self.sliders[pname].set(default_value)
            elif pname == 'threshold_mode':
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
            elif pname == 'invert_image':
                self.tuner.invert_image = default_value
                ic = (
                    SegmentationTuner.COLORS['success'] if default_value
                    else SegmentationTuner.COLORS['gray']
                )
                self.invert_btn.config(
                    text=f"INVERT: {'ON' if default_value else 'OFF'}",
                    bg=ic
                )
            elif pname == 'use_intensity_threshold':
                self.tuner.use_intensity_threshold = default_value
            elif pname == 'intensity_threshold_value':
                self.tuner.intensity_threshold_value = default_value

        self.tuner.update_visualization()
        messagebox.showinfo(
            "Reset Complete",
            f"Parameters reset to {pathogen_name.upper()} initial values!",
            parent=self.winfo_toplevel()
        )


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
        "threshold_mode":   "otsu",
        "manual_threshold": 127,
        "morph_kernel_size": 1,
        "morph_iterations":  1,
    }

    DEFAULT_SHAPE_FILTERS = {
        "min_circularity":  0.7,
        "max_circularity":  0.97,
        "min_aspect_ratio": 0.75,
        "max_aspect_ratio": 1.60,
        "min_solidity":     0.88,
        "dilate_iterations": 0,
        "erode_iterations":  0,
        "max_mean_intensity": 255.0,
    }

    FALLBACK_UM_PER_PX = 0.109492

    COLORS = {
        'bg':        '#f0f0f0',
        'header':    '#2c3e50',
        'primary':   '#3498db',
        'success':   '#27ae60',
        'warning':   '#e67e22',
        'info':      '#16a085',
        'secondary': '#34495e',
        'danger':    '#e74c3c',
        'purple':    '#9b59b6',
        'gray':      '#95a5a6',
    }

    # ----------------------------------------------------------------
    # Single source of truth for slider bounds, steps, and units.
    # ParameterPanel._create_sections and the suggestion helpers all
    # reference this dict so they stay in sync automatically.
    # ----------------------------------------------------------------
    SLIDER_META: Dict[str, Dict] = {
        'gaussian_sigma':     {'min': 0.5,  'max': 20.0,   'step': 0.1,  'unit': ''},
        'manual_threshold':   {'min': 0,    'max': 255,    'step': 1,    'unit': ''},
        'morph_kernel_size':  {'min': 1,    'max': 15,     'step': 1,    'unit': ''},
        'morph_iterations':   {'min': 0,    'max': 5,      'step': 1,    'unit': ''},
        # area sliders work in µm²; internal params['min/max_area'] stay in px
        'min_area':           {'min': 0.1,  'max': 500.0,  'step': 0.1,  'unit': ' µm²'},
        'max_area':           {'min': 1.0,  'max': 5000.0, 'step': 0.1,  'unit': ' µm²'},
        'dilate_iterations':  {'min': 0,    'max': 5,      'step': 1,    'unit': ''},
        'erode_iterations':   {'min': 0,    'max': 5,      'step': 1,    'unit': ''},
        'min_circularity':    {'min': 0.0,  'max': 1.0,    'step': 0.01, 'unit': ''},
        'max_circularity':    {'min': 0.0,  'max': 1.0,    'step': 0.01, 'unit': ''},
        'min_aspect_ratio':   {'min': 0.0,  'max': 10.0,   'step': 0.1,  'unit': ''},
        'max_aspect_ratio':   {'min': 0.0,  'max': 20.0,   'step': 0.1,  'unit': ''},
        'min_solidity':       {'min': 0.0,  'max': 1.0,    'step': 0.01, 'unit': ''},
        'max_mean_intensity': {'min': 0.0,  'max': 255.0,  'step': 1.0,  'unit': ''},
    }

    # ----------------------------------------------------------------
    # Suggestion margin: how close (as a fraction) to a threshold
    # a detected particle's value must be before a suggestion fires.
    # ----------------------------------------------------------------
    _SUGGESTION_MARGIN = 0.15   # 15 % of the threshold value

    def __init__(self, root: tk.Tk, image_path: str, bacterium: str,
                 structure: str, mode: str,
                 config_key: Optional[str] = None,
                 return_callback=None):
        self.current_pathogen = bacterium
        self.master = root
        self.root   = root

        self.ui_scaler = UIScaler(root)

        self.image_path = Path(image_path)
        self.bacterium  = bacterium
        self.structure  = structure
        self.mode       = mode
        self.return_callback = return_callback

        if config_key is not None:
            self.config_key = config_key
        else:
            _key = re.sub(r'[^a-z0-9]+', '_', bacterium.lower()).strip('_')
            self.config_key = re.sub(r'_+', '_', _key)

        print(f"  Tuner config_key: {self.config_key}")

        self.image_list  = []
        self.image_index = 0

        if not validate_path_encoding(self.image_path):
            raise ValueError(f"Path contains problematic characters: {image_path}")

        self.original_image = self._load_image(self.image_path)
        self.pixel_size_um, self.has_metadata = self._load_pixel_size()

        self._initialize_parameters()

        self.processed_image: np.ndarray     = np.zeros_like(self.original_image)
        self.binary_mask:     np.ndarray     = np.zeros_like(self.original_image)
        self.contours:        List[np.ndarray] = []
        self.contour_areas:   List[float]    = []
        self.current_suggestions: Dict[str, Any] = {}

        self.selection_mode:   Any  = False
        self.accepted_indices: set  = set()
        self.rejected_indices: set  = set()
        self._pre_broad_params: dict = {}

        self.sliders:      Dict[str, Slider]   = {}
        self.param_labels: Dict[str, tk.Label] = {}

        self.setup_gui()

    # ------------------------------------------------------------------
    # Helper: snap a suggestion value to the correct slider step and
    # clamp it within the slider's legal bounds.
    # ------------------------------------------------------------------
    @classmethod
    def _snap_to_slider(
        cls,
        param_name: str,
        value: "float | np.floating[Any]",
    ) -> Optional[float]:
        
        
        """
        Round *value* to the slider step for *param_name* and clamp it
        within [min, max].  Returns None if the param is not in SLIDER_META.
        """
        meta = cls.SLIDER_META.get(param_name)
        if meta is None:
            return float(value)
        step  = meta['step']
        lo    = meta['min']
        hi    = meta['max']
        snapped = round(round(float(value) / step) * step, 10)
        return float(max(lo, min(hi, snapped)))

    def quit(self, event=None):
        if messagebox.askyesno(
            "Quit", "Are you sure you want to quit?\n\nUnsaved changes will be lost."
        ):
            self.master.quit()
            self.master.destroy()

    def back(self, event=None):
        if messagebox.askyesno(
            "Back to Setup",
            "Return to the Pathogen Setup screen?\n\nUnsaved changes will be lost."
        ):
            callback = self.return_callback   # capture before destroy clears state
            self.master.quit()
            self.master.destroy()
            if callable(callback):
                callback()

    def load_image_at_index(self, index: int):
        if not hasattr(self, 'image_list') or not self.image_list:
            messagebox.showwarning("Warning", "No image list available")
            return
        if 0 <= index < len(self.image_list):
            image_path = self.image_list[index]
            try:
                self.original_image = self._load_image(image_path)
                self.image_path      = image_path
                self.image_index     = index
                self.master.title(
                    f"Tuner - [{index + 1}/{len(self.image_list)}] {image_path.name}"
                )
                self.update_visualization()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{e}")
        else:
            messagebox.showwarning("Warning", f"Invalid image index: {index}")

    def save_and_apply(self, event=None):
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
                description=(
                    f"{self.structure} segmentation - Tuned "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                ),
                gaussian_sigma=float(self.params['gaussian_sigma']),
                min_area_um2=float(self.params['min_area']) * um2_per_px2,
                max_area_um2=float(self.params['max_area']) * um2_per_px2,
                dilate_iterations=int(self.params['dilate_iterations']),
                erode_iterations=int(self.params['erode_iterations']),
                min_circularity=float(self.min_circularity),
                max_circularity=float(self.max_circularity),
                min_aspect_ratio=float(self.min_aspect_ratio),
                max_aspect_ratio=float(self.max_aspect_ratio),
                min_mean_intensity_bf=0,
                max_mean_intensity_bf=float(self.max_mean_intensity),
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

            bacteria_key = self.config_key
            success      = _manager.update_config(bacteria_key, config)

            if success:
                if _bacteria_registry.key_exists(bacteria_key):
                    _bacteria_registry.set_validated(bacteria_key, True)
                    print(f"[Registry] '{bacteria_key}' marked as validated after tuning.")

                config_file = _manager._get_config_path(bacteria_key)
                thresh_info = (
                    f"Intensity threshold: {intensity_threshold:.0f} (BINARY_INV)"
                    if use_intensity_threshold
                    else "Background subtraction + Otsu"
                )

                messagebox.showinfo(
                    "Success",
                    f"✓ Configuration saved!\n\n"
                    f"Bacterium: {self.bacterium}\n"
                    f"Saved to: {config_file.name}\n"
                    f"Threshold: {thresh_info}\n\n"
                    f"Morph: OPEN({self.morph_iterations}) → "
                    f"CLOSE({self.morph_iterations + 1})\n"
                    f"Dilate: {self.params['dilate_iterations']}  "
                    f"Erode: {self.params['erode_iterations']}\n\n"
                    f"✓ Marked as validated in registry."
                )
            else:
                messagebox.showerror(
                    "Error",
                    "Failed to update configuration\nCheck console for details"
                )

        except Exception as e:
            print(f"✗ Error saving configuration: {e}")
            import traceback; traceback.print_exc()
            messagebox.showerror("Error", f"Failed to save configuration:\n{e}")

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
        for key, value in self.DEFAULT_SHAPE_FILTERS.items():
            setattr(self, key, value)

        self.threshold_mode   = self.DEFAULT_THRESHOLD_PARAMS['threshold_mode']
        self.manual_threshold = self.DEFAULT_THRESHOLD_PARAMS['manual_threshold']
        self.morph_kernel_size = self.DEFAULT_THRESHOLD_PARAMS['morph_kernel_size']
        self.morph_iterations  = self.DEFAULT_THRESHOLD_PARAMS['morph_iterations']
        self.invert_image      = False

        self.use_intensity_threshold   = False
        self.intensity_threshold_value = 80.0

        config_key  = self.config_key
        config_file = Path("bacteria_configs") / f"{config_key}.json"

        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                config_data = json_data.get("config", json_data)

                if 'morph_kernel_size' in config_data:
                    ks = config_data['morph_kernel_size']
                    if ks % 2 == 0:
                        print(f"WARNING: morph_kernel_size={ks} is even, correcting to {ks + 1}")
                        config_data['morph_kernel_size'] = ks + 1

                if config_data.get('pixel_size_um') and not self.has_metadata:
                    self.pixel_size_um = float(config_data['pixel_size_um'])
                    print(f"   • Pixel size restored from JSON: {self.pixel_size_um:.6f} µm/px")

                um2_per_px2 = self.pixel_size_um ** 2

                self.threshold_mode    = config_data.get('threshold_mode',    'otsu')
                self.manual_threshold  = config_data.get('manual_threshold',  127)
                self.morph_kernel_size = config_data.get('morph_kernel_size', 3)
                self.morph_iterations  = config_data.get('morph_iterations',  1)

                self.use_intensity_threshold   = bool(config_data.get('use_intensity_threshold', False))
                self.intensity_threshold_value = float(config_data.get('intensity_threshold', 80.0))

                if self.use_intensity_threshold:
                    self.threshold_mode   = "intensity"
                    self.manual_threshold = int(self.intensity_threshold_value)

                self.params = {
                    "gaussian_sigma":    float(config_data.get('gaussian_sigma', 2.0)),
                    "min_area":          float(config_data.get('min_area_um2', 3.0) / um2_per_px2),
                    "max_area":          float(config_data.get('max_area_um2', 100.0) / um2_per_px2),
                    "dilate_iterations": int(config_data.get('dilate_iterations', 0)),
                    "erode_iterations":  int(config_data.get('erode_iterations', 0)),
                }

                self.min_circularity  = float(config_data.get('min_circularity',  0.0))
                self.max_circularity  = float(config_data.get('max_circularity',  1.0))
                self.min_aspect_ratio = float(config_data.get('min_aspect_ratio', 0.2))
                self.max_aspect_ratio = float(config_data.get('max_aspect_ratio', 10.0))
                self.min_solidity     = float(config_data.get('min_solidity',     0.3))
                self.max_mean_intensity = float(config_data.get(
                    'max_mean_intensity_bf',
                    config_data.get('max_mean_intensity', 255.0)
                ))
                self.invert_image = config_data.get('invert_image', False)

                print(f"✅ Loaded PERMANENT config from: {config_file}")
                return

            except Exception as e:
                print(f"⚠️ Could not load bacteria_configs JSON: {e}")
                import traceback; traceback.print_exc()

        json_filename = (
            f"segmentation_params_{self.bacterium}_{self.structure}_{self.mode}.json"
        )
        if Path(json_filename).exists():
            try:
                with open(json_filename, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)

                params_dict = saved_data['parameters'].copy()
                self.invert_image = params_dict.pop('invert_image', False)

                self.params = {
                    "gaussian_sigma":    params_dict.get('gaussian_sigma', 2.0),
                    "min_area":          params_dict.get('min_area',  20),
                    "max_area":          params_dict.get('max_area',  5000),
                    "dilate_iterations": params_dict.get('dilate_iterations', 0),
                    "erode_iterations":  params_dict.get('erode_iterations',  0),
                }

                self.min_circularity    = params_dict.get('min_circularity',  0.0)
                self.max_circularity    = params_dict.get('max_circularity',  1.0)
                self.min_aspect_ratio   = params_dict.get('min_aspect_ratio', 0.2)
                self.max_aspect_ratio   = params_dict.get('max_aspect_ratio', 10.0)
                self.min_solidity       = params_dict.get('min_solidity',     0.3)
                self.threshold_mode     = params_dict.get('threshold_mode',   'otsu')
                self.manual_threshold   = params_dict.get('manual_threshold', 127)
                self.morph_kernel_size  = params_dict.get('morph_kernel_size',3)
                self.morph_iterations   = params_dict.get('morph_iterations', 1)
                self.use_intensity_threshold   = params_dict.get('use_intensity_threshold', False)
                self.intensity_threshold_value = params_dict.get('intensity_threshold', 80.0)
                if self.use_intensity_threshold:
                    self.threshold_mode = "intensity"
                self.max_mean_intensity = params_dict.get('max_mean_intensity', 255.0)

                print(f"✅ Restored TEMP session from: {json_filename}")
                return

            except Exception as e:
                print(f"⚠️ Could not load session JSON: {e}")

        if config_key in _CONFIGS:
            try:
                saved_config = _CONFIGS[config_key]
                um2_per_px2  = self.pixel_size_um ** 2

                self.params = {
                    "gaussian_sigma":    float(saved_config.gaussian_sigma),
                    "min_area":          float(saved_config.min_area_um2 / um2_per_px2),
                    "max_area":          float(saved_config.max_area_um2 / um2_per_px2),
                    "dilate_iterations": int(saved_config.dilate_iterations),
                    "erode_iterations":  int(saved_config.erode_iterations),
                }
                self.min_circularity  = float(saved_config.min_circularity)
                self.max_circularity  = float(saved_config.max_circularity)
                self.min_aspect_ratio = float(saved_config.min_aspect_ratio)
                self.max_aspect_ratio = float(saved_config.max_aspect_ratio)
                self.min_solidity     = float(saved_config.min_solidity)
                self.max_mean_intensity = float(
                    getattr(saved_config, 'max_mean_intensity_bf', 255.0)
                )
                self.invert_image = False
                print(f"✅ Loaded config for {self.bacterium} from bacteria_configs.py")
                return
            except Exception as e:
                print(f"⚠️ Error loading bacteria_configs.py: {e}")

        print(f"No saved config found for '{self.bacterium}' - using defaults")
        self.params = self.DEFAULT_PARAMS.copy()
        self.threshold_mode    = self.DEFAULT_THRESHOLD_PARAMS['threshold_mode']
        self.manual_threshold  = self.DEFAULT_THRESHOLD_PARAMS['manual_threshold']
        self.morph_kernel_size = self.DEFAULT_THRESHOLD_PARAMS['morph_kernel_size']
        self.morph_iterations  = self.DEFAULT_THRESHOLD_PARAMS['morph_iterations']
        self.invert_image      = False
        self.use_intensity_threshold   = False
        self.intensity_threshold_value = 80.0
        self.max_mean_intensity        = 255.0

    def update_threshold(self, param_name: str, value: float):
        setattr(self, param_name, value)
        self.update_visualization()

    def update_morph(self, param_name: str, value: float):
        setattr(self, param_name, value)
        self.update_visualization()

    def cycle_threshold_mode(self, event):
        modes = ["otsu", "manual", "adaptive", "intensity"]
        current_idx = modes.index(self.threshold_mode) if self.threshold_mode in modes else 0
        self.threshold_mode = modes[(current_idx + 1) % len(modes)]

        if self.threshold_mode == "intensity":
            self.use_intensity_threshold   = True
            self.intensity_threshold_value = float(self.manual_threshold)
        else:
            self.use_intensity_threshold = False

        if hasattr(self, 'parameter_panel') and hasattr(self.parameter_panel, 'thresh_mode_btn'):
            self.parameter_panel.thresh_mode_btn.config(
                text=self.threshold_mode.upper()
            )
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
        x = (self.root.winfo_screenwidth()  // 2) - (window_width  // 2)
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
        if hasattr(self, 'canvas_image'): self.canvas_image.draw()
        if hasattr(self, 'canvas_hist'):  self.canvas_hist.draw()
        if hasattr(self, 'canvas_sliders'): self.canvas_sliders.draw()

    def _create_header(self, parent: ttk.Frame):
        header_height = self.ui_scaler.scale_dimension(45)
        header_frame  = tk.Frame(parent, bg=self.COLORS['header'], height=header_height)
        header_frame.pack(fill=tk.X, pady=(0, 5))
        header_frame.pack_propagate(False)

        title_font  = ("Segoe UI", self.ui_scaler.scale_font(14), "bold")
        badge_font  = ("Segoe UI", self.ui_scaler.scale_font(9))
        button_font = ("Segoe UI", self.ui_scaler.scale_font(9), "bold")

        tk.Label(
            header_frame, text=f"🔬 {self.bacterium} - {self.structure}",
            font=title_font, bg=self.COLORS['header'], fg="white"
        ).pack(side=tk.LEFT, padx=20, pady=6)

        tk.Label(
            header_frame,
            text=f"Mode: {self.mode} {'(Inverted)' if self.invert_image else ''}",
            font=badge_font, bg=self.COLORS['secondary'], fg="white",
            padx=12, pady=4, relief=tk.RAISED
        ).pack(side=tk.LEFT, pady=6)

        pixel_color = self.COLORS['success'] if self.has_metadata else self.COLORS['warning']
        pixel_text  = f"Pixel: {self.pixel_size_um:.6f} µm"
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

        self.btn_pick_reject = tk.Button(
            header_frame, text="🎯 PICK/REJECT", font=button_font,
            bg=self.COLORS['warning'], fg="white",
            relief=tk.RAISED, cursor="hand2",
            command=self.enter_pick_reject_mode, padx=10, pady=4
        )
        self.btn_pick_reject.pack(side=tk.LEFT, padx=5, pady=6)

        self.btn_normalize = tk.Button(
            header_frame, text="✨ NORMALIZE", font=button_font,
            bg=self.COLORS['success'], fg="white",
            relief=tk.RAISED, cursor="hand2",
            command=self.normalize_from_selection,
            padx=10, pady=4, state=tk.DISABLED
        )
        self.btn_normalize.pack(side=tk.LEFT, padx=2, pady=6)

        self.btn_cancel_pr = tk.Button(
            header_frame, text="✖ CANCEL", font=button_font,
            bg=self.COLORS['danger'], fg="white",
            relief=tk.RAISED, cursor="hand2",
            command=self.cancel_pick_reject,
            padx=10, pady=4, state=tk.DISABLED
        )
        self.btn_cancel_pr.pack(side=tk.LEFT, padx=2, pady=6)

        self.contour_count_label = tk.Label(
            header_frame, text="Contours: 0", font=badge_font,
            bg=self.COLORS['success'], fg="white",
            padx=10, pady=4, relief=tk.RAISED
        )
        self.contour_count_label.pack(side=tk.RIGHT, padx=(10, 20), pady=6)

        for text, color, command in [
            ("❌ QUIT",         self.COLORS['danger'],    self.quit),
            ("✅ SAVE & APPLY", self.COLORS['success'],   self.save_and_apply),
            ("⬅ BACK",         self.COLORS['secondary'], self.back),
        ]:
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
            font=("Segoe UI", 11, "bold"),
            bg=self.COLORS['secondary'], fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=5)

        canvas_frame = ttk.Frame(left_panel)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig_image = Figure(figsize=(11, 8), facecolor='white', dpi=100)
        self.ax_image  = self.fig_image.add_subplot(111)
        self.canvas_image = FigureCanvasTkAgg(self.fig_image, canvas_frame)
        self.canvas_image.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_image.mpl_connect("button_press_event", self.on_image_click)

        instruction = tk.Frame(left_panel, bg=self.COLORS['primary'], height=28)
        instruction.pack(fill=tk.X)
        instruction.pack_propagate(False)
        self.instruction_label = tk.Label(
            instruction,
            text="💡 Click a detected particle (green) to inspect it  |  "
                 "Click empty space to diagnose missed particles",
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
        tk.Label(
            header, text="⚙️ PARAMETERS",
            font=("Segoe UI", self.ui_scaler.scale_font(10), "bold"),
            bg=self.COLORS['secondary'], fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=4)

        display = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        display.pack(fill=tk.X, padx=5, pady=(5, 0))

        inner = ttk.Frame(display)
        inner.pack(fill=tk.X, padx=8, pady=8)

        metadata_status = "✓ From metadata" if self.has_metadata else "⚠ Fallback"
        self._add_param_section(inner, "Basic Information", [
            ("Pathogen:",   self.bacterium, self.COLORS['danger']),
            ("Structure:",  self.structure, self.COLORS['purple']),
            ("Mode:",       f"{self.mode} particles", self.COLORS['primary']),
            ("Pixel size:", f"{self.pixel_size_um:.6f} µm/px",
             self.COLORS['success'] if self.has_metadata else self.COLORS['warning']),
            ("Metadata:",   metadata_status,
             self.COLORS['success'] if self.has_metadata else self.COLORS['warning']),
        ])

        ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        self._add_param_section(inner, "Segmentation", [
            ("Invert:",      "ON" if self.invert_image else "OFF",
             self.COLORS['success'] if self.invert_image else self.COLORS['gray']),
            ("Gaussian σ:",  f"{self.params['gaussian_sigma']:.1f}", None),
            ("Threshold:",   self.threshold_mode.upper(), self.COLORS['primary']),
            ("Morph kernel:",f"{self.morph_kernel_size}x{self.morph_kernel_size}", None),
            ("Morph iter:",  str(int(self.morph_iterations)), None),
        ])

        ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        um2_per_px2  = self.pixel_size_um ** 2
        min_area_um2 = self.params['min_area'] * um2_per_px2
        max_area_um2 = self.params['max_area'] * um2_per_px2

        self._add_param_section(inner, "Filtering & Morphology", [
            ("Min area:",    f"{min_area_um2:.2f} µm²", None),
            ("Max area:",    f"{max_area_um2:.2f} µm²", None),
            ("Dilate iter:", str(self.params["dilate_iterations"]), None),
            ("Erode iter:",  str(self.params["erode_iterations"]),  None),
        ])

        ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        self._add_param_section(inner, "Shape Filters", [
            ("Circularity:",  f"{self.min_circularity:.2f} – {self.max_circularity:.2f}", None),
            ("Aspect ratio:", f"{self.min_aspect_ratio:.2f} – {self.max_aspect_ratio:.2f}", None),
            ("Solidity:",     f"≥ {self.min_solidity:.2f}", None),
            ("Max BF int.:",  f"≤ {self.max_mean_intensity:.0f}", None),
        ])

    def _add_param_section(self, parent: ttk.Frame, title: str,
                           params: List[Tuple[str, str, Optional[str]]]):
        title_font  = ("Segoe UI", self.ui_scaler.scale_font(9), "bold")
        label_font  = ("Segoe UI", self.ui_scaler.scale_font(8))
        value_font  = ("Segoe UI", self.ui_scaler.scale_font(8), "bold")
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
        tk.Label(
            header, text="🎯 TARGET ANALYSIS",
            font=("Segoe UI", self.ui_scaler.scale_font(9), "bold"),
            bg=self.COLORS['warning'], fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=3)

        display = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        display.pack(fill=tk.X, padx=5, pady=(0, 5))

        text_frame = tk.Frame(display, bg="white")
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.target_analysis_text = tk.Text(
            text_frame,
            font=("Segoe UI", self.ui_scaler.scale_font(8)),
            wrap=tk.WORD, height=12,
            bg="white", fg="gray", padx=8, pady=8,
            yscrollcommand=scrollbar.set, relief=tk.FLAT
        )
        self.target_analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.target_analysis_text.yview)

        self.target_analysis_text.insert(
            '1.0',
            "Click a green contour to inspect its filter details.\n"
            "Click empty space to diagnose a missed particle."
        )

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
        self.ax_hist  = self.fig_hist.add_subplot(111)
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, canvas_frame)
        self.canvas_hist.get_tk_widget().pack(
            fill=tk.BOTH, expand=True, padx=2, pady=2
        )

    def _create_control_panel(self, parent: ttk.Frame):
        panel = ttk.Frame(parent)
        panel.pack(fill=tk.X, pady=(5, 0))

        header = tk.Frame(panel, bg=self.COLORS['secondary'], height=30)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header, text="🎚️ ADJUST PARAMETERS",
            font=("Segoe UI", 11, "bold"),
            bg=self.COLORS['secondary'], fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=5)

        slider_frame = ttk.Frame(panel, relief=tk.SUNKEN, borderwidth=1)
        slider_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        screen_width    = self.root.winfo_screenwidth()
        fig_width_inches = (screen_width * 0.92) / 100

        self.fig_sliders = Figure(
            figsize=(fig_width_inches, 2.5), facecolor='#f8f9fa', dpi=100
        )
        self.canvas_sliders = FigureCanvasTkAgg(self.fig_sliders, slider_frame)
        self.canvas_sliders.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._create_sliders()

    def _create_sliders(self):
        """Legacy matplotlib slider panel — not used in the current UI."""
        self.fig_sliders.clear()

        slider_height = 0.10
        slider_width  = 0.25
        col1_x = 0.04; col2_x = 0.36; col3_x = 0.68
        row1_y = 0.78; row2_y = 0.62; row3_y = 0.46
        row4_y = 0.30; row5_y = 0.14

        for label, col, row, key, callback in [
            ('Gaussian σ',  col1_x, row1_y, 'gaussian_sigma', lambda v: self.update_parameter('gaussian_sigma', v)),
            ('Threshold',   col1_x, row2_y, 'manual_threshold', lambda v: self.update_threshold('manual_threshold', v)),
            ('Morph Kernel',col1_x, row3_y, 'morph_kernel_size', lambda v: self.update_morph('morph_kernel_size', v)),
            ('Morph Iter',  col1_x, row4_y, 'morph_iterations', lambda v: self.update_morph('morph_iterations', v)),
            ('Min Area(px)',col2_x, row1_y, 'min_area', lambda v: self.update_parameter('min_area', v)),
            ('Max Area(px)',col2_x, row2_y, 'max_area', lambda v: self.update_parameter('max_area', v)),
            ('Dilate Iter', col2_x, row3_y, 'dilate_iterations', lambda v: self.update_parameter('dilate_iterations', v)),
            ('Erode Iter',  col2_x, row4_y, 'erode_iterations',  lambda v: self.update_parameter('erode_iterations', v)),
            ('Min Circular',col3_x, row1_y, 'min_circularity', lambda v: self.update_shape_filter('min_circularity', v)),
            ('Max Circular',col3_x, row2_y, 'max_circularity', lambda v: self.update_shape_filter('max_circularity', v)),
            ('Min Solidity',col3_x, row3_y, 'min_solidity',    lambda v: self.update_shape_filter('min_solidity', v)),
        ]:
            meta    = self.SLIDER_META.get(key, {'min': 0, 'max': 1, 'step': 0.01})
            valinit = (
                self.params.get(key, getattr(self, key, meta['min']))
                if key in self.params else getattr(self, key, meta['min'])
            )
            ax = self.fig_sliders.add_axes((col, row, slider_width, slider_height))
            slider = Slider(ax, label, meta['min'], meta['max'],
                            valinit=valinit, color=self.COLORS['primary'])
            slider.on_changed(callback)
            self.sliders[key] = slider

    def process_image(self):
        img  = self.original_image.copy()

        if self.invert_image:
            img = cv2.bitwise_not(img)

        blur = cv2.GaussianBlur(
            img, (0, 0),
            sigmaX=self.params["gaussian_sigma"],
            sigmaY=self.params["gaussian_sigma"]
        )

        if self.use_intensity_threshold:
            _, binary = cv2.threshold(
                blur, float(self.manual_threshold), 255, cv2.THRESH_BINARY_INV
            )
            self.processed_image = blur
        else:
            bg       = cv2.GaussianBlur(img, (0, 0),
                                        sigmaX=self.params["gaussian_sigma"],
                                        sigmaY=self.params["gaussian_sigma"])
            enhanced      = cv2.subtract(bg, img)
            enhanced_blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
            self.processed_image = enhanced_blur

            if self.threshold_mode == "otsu":
                _, binary = cv2.threshold(
                    enhanced_blur, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            elif self.threshold_mode == "manual":
                _, binary = cv2.threshold(
                    enhanced_blur, self.manual_threshold, 255, cv2.THRESH_BINARY
                )
            elif self.threshold_mode == "adaptive":
                binary = cv2.adaptiveThreshold(
                    enhanced_blur, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
            else:
                _, binary = cv2.threshold(
                    enhanced_blur, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

        kernel_size = int(self.morph_kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel     = np.ones((kernel_size, kernel_size), np.uint8)
        morph_iter = int(self.morph_iterations)

        if morph_iter > 0:
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN, kernel, iterations=morph_iter
            )
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel, iterations=morph_iter + 1
        )
        if int(self.params["dilate_iterations"]) > 0:
            binary = cv2.dilate(
                binary, kernel, iterations=int(self.params["dilate_iterations"])
            )
        if int(self.params["erode_iterations"]) > 0:
            binary = cv2.erode(
                binary, kernel, iterations=int(self.params["erode_iterations"])
            )

        self.binary_mask = binary
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        self.contours      = []
        self.contour_areas = []

        H, W = img.shape[:2]
        max_big_area_px = 0.25 * float(H * W)

        for cnt in contours:
            area_px = float(cv2.contourArea(cnt))
            if area_px <= 0:
                continue
            if not (self.params["min_area"] <= area_px <= self.params["max_area"]):
                continue
            if area_px >= max_big_area_px:
                continue

            perimeter   = float(cv2.arcLength(cnt, True))
            circularity = (4 * np.pi * area_px / perimeter ** 2) if perimeter > 0 else 0.0
            if not (self.min_circularity <= circularity <= self.max_circularity):
                continue

            x, y, w, h   = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0.0
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                continue

            hull      = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            solidity  = area_px / hull_area if hull_area > 0 else 0.0
            if solidity < self.min_solidity:
                continue

            _mask_int = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(_mask_int, [cnt], -1, 255, cv2.FILLED)
            _pixels  = img[_mask_int > 0]
            mean_int = float(np.mean(_pixels.astype(np.float64))) if _pixels.size > 0 else 0.0
            if mean_int > self.max_mean_intensity:
                continue

            self.contours.append(cnt)
            self.contour_areas.append(area_px)

    def update_visualization(self):
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
        display = (
            cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB)
            if self.original_image.ndim == 2
            else self.original_image.copy()
        )
        cv2.drawContours(display, self.contours, -1, (0, 255, 0), 2)
        self.ax_image.imshow(display)
        self.ax_image.set_title(
            "Original + Contours", fontsize=12, fontweight='bold', pad=10
        )
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
            areas_px  = np.array(self.contour_areas)
            um2       = self.pixel_size_um ** 2
            areas_um2 = areas_px * um2

            self.ax_hist.hist(
                areas_um2, bins=30,
                color=self.COLORS['primary'], alpha=0.7, edgecolor='black'
            )
            for val, color, lbl in [
                (float(np.median(areas_um2)), 'orange',  f"Median: {np.median(areas_um2):.2f} µm²"),
                (float(np.mean(areas_um2)),   'red',     f"Mean: {np.mean(areas_um2):.2f} µm²"),
                (float(np.min(areas_um2)),    'green',   f"Min: {np.min(areas_um2):.2f} µm²"),
                (float(np.max(areas_um2)),    'purple',  f"Max: {np.max(areas_um2):.2f} µm²"),
            ]:
                self.ax_hist.axvline(
                    val, color=color, linestyle='--', linewidth=1.5, label=lbl
                )

            self.ax_hist.set_xlabel("Area (µm²)", fontsize=9)
            self.ax_hist.set_ylabel("Count", fontsize=9)
            self.ax_hist.set_title(
                f"Distribution  (n={len(areas_um2)})", fontsize=10, fontweight='bold'
            )
            self.ax_hist.legend(fontsize=7, loc='upper right')
            self.ax_hist.grid(True, alpha=0.3)

        self.canvas_hist.draw()

    def _update_param_displays(self):
        if "Invert:" in self.param_labels:
            ic = (
                self.COLORS['success'] if self.invert_image else self.COLORS['gray']
            )
            self.param_labels["Invert:"].config(
                text="ON" if self.invert_image else "OFF", bg=ic
            )

        um2          = self.pixel_size_um ** 2
        min_area_um2 = self.params['min_area'] * um2
        max_area_um2 = self.params['max_area'] * um2

        updates = {
            "Gaussian σ:":  f"{self.params['gaussian_sigma']:.1f}",
            "Threshold:":   self.threshold_mode.upper(),
            "Morph kernel:":f"{int(self.morph_kernel_size)}x{int(self.morph_kernel_size)}",
            "Morph iter:":  str(int(self.morph_iterations)),
            "Min area:":    f"{min_area_um2:.2f} µm²",
            "Max area:":    f"{max_area_um2:.2f} µm²",
            "Dilate iter:": str(int(self.params["dilate_iterations"])),
            "Erode iter:":  str(int(self.params["erode_iterations"])),
            "Circularity:": f"{self.min_circularity:.2f} – {self.max_circularity:.2f}",
            "Aspect ratio:":f"{self.min_aspect_ratio:.2f} – {self.max_aspect_ratio:.2f}",
            "Solidity:":    f"≥ {self.min_solidity:.2f}",
            "Max BF int.:": f"≤ {self.max_mean_intensity:.0f}",
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

        if filter_name in ("min_circularity", "max_circularity"):
            key  = "Circularity:"
            text = f"{self.min_circularity:.2f} – {self.max_circularity:.2f}"
        elif filter_name in ("min_aspect_ratio", "max_aspect_ratio"):
            key  = "Aspect ratio:"
            text = f"{self.min_aspect_ratio:.2f} – {self.max_aspect_ratio:.2f}"
        elif filter_name == "min_solidity":
            key  = "Solidity:"
            text = f"≥ {self.min_solidity:.2f}"
        else:
            return

        if key in self.param_labels:
            self.param_labels[key].config(text=text)

    def toggle_invert(self, event):
        self.invert_image = not self.invert_image
        if hasattr(self, 'parameter_panel') and hasattr(self.parameter_panel, 'invert_btn'):
            self.parameter_panel.invert_btn.config(
                text=f'INVERT\n{"ON" if self.invert_image else "OFF"}',
                bg=self.COLORS['success'] if self.invert_image else self.COLORS['gray']
            )

        self.update_visualization()

    # ==================================================================
    # PICK / REJECT / NORMALIZE
    # ==================================================================

    def _run_broad_detection(self):
        self.process_image()

    def enter_pick_reject_mode(self):
        H, W = self.original_image.shape[:2]

        self._pre_broad_params = {
            'params':           self.params.copy(),
            'min_circularity':  self.min_circularity,
            'max_circularity':  self.max_circularity,
            'min_aspect_ratio': self.min_aspect_ratio,
            'max_aspect_ratio': self.max_aspect_ratio,
            'min_solidity':     self.min_solidity,
        }

        self.params['min_area']   = 5.0
        self.params['max_area']   = float(H * W) * 0.15
        self.min_circularity      = 0.0
        self.max_circularity      = 1.0
        self.min_aspect_ratio     = 0.01
        self.max_aspect_ratio     = 50.0
        self.min_solidity         = 0.0

        if hasattr(self, 'parameter_panel'):
            pp  = self.parameter_panel
            um2 = self.pixel_size_um ** 2
            for name, val in [
                ('min_area',         self.params['min_area'] * um2),
                ('max_area',         self.params['max_area'] * um2),
                ('min_circularity',  self.min_circularity),
                ('max_circularity',  self.max_circularity),
                ('min_aspect_ratio', self.min_aspect_ratio),
                ('max_aspect_ratio', self.max_aspect_ratio),
                ('min_solidity',     self.min_solidity),
            ]:
                if name in pp.sliders:
                    pp.sliders[name].set(val)

        self._run_broad_detection()

        self.accepted_indices = set()
        self.rejected_indices = set()
        self.selection_mode   = 'pick_reject'

        self._update_image_display_pick_reject()
        self._update_pick_reject_buttons(active=True)

        self.contour_count_label.config(
            text=f"PICK/REJECT — {len(self.contours)} objects found"
        )
        if hasattr(self, 'instruction_label'):
            self.instruction_label.config(
                text="🖱 LEFT-click = ✅ Accept  |  RIGHT-click = ❌ Reject  "
                     "|  Click accepted again to toggle back to unclassified",
                bg=self.COLORS['warning']
            )
        self._show_pick_reject_status()

    def _update_image_display_pick_reject(self):
        self.ax_image.clear()
        display = (
            cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB)
            if self.original_image.ndim == 2
            else self.original_image.copy()
        )

        for i, cnt in enumerate(self.contours):
            if i in self.accepted_indices:
                color, thickness = (0, 220, 0), 3
            elif i in self.rejected_indices:
                color, thickness = (220, 40, 40), 2
            else:
                color, thickness = (255, 200, 0), 1

            cv2.drawContours(display, [cnt], -1, color, thickness)
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(
                    display, str(i), (cx - 4, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA
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
        state = tk.NORMAL if active else tk.DISABLED
        self.btn_normalize.config(state=state)
        self.btn_cancel_pr.config(state=state)
        self.btn_pick_reject.config(
            text="⏳ Selecting…" if active else "🎯 PICK/REJECT",
            state=tk.DISABLED if active else tk.NORMAL
        )

    def _show_pick_reject_status(self):
        n_acc = len(self.accepted_indices)
        n_rej = len(self.rejected_indices)
        n_tot = len(self.contours)
        n_unc = n_tot - n_acc - n_rej

        msg = (
            f"🎯 PICK / REJECT MODE\n\n"
            f"Total objects detected:  {n_tot}\n\n"
            f"🟢 Accepted (left-click):  {n_acc}\n"
            f"🔴 Rejected (right-click): {n_rej}\n"
            f"🟡 Unclassified:           {n_unc}\n\n"
            f"When done, click  ✨ NORMALIZE  in the header\n"
            f"to compute optimal parameters from your selection.\n\n"
            f"Click  ✖ CANCEL  to restore original parameters."
        )
        self.target_analysis_text.delete('1.0', tk.END)
        self.target_analysis_text.insert('1.0', msg)

    def normalize_from_selection(self):
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
            return (
                np.percentile(arr, lo_pct) * (1.0 - margin),
                np.percentile(arr, hi_pct) * (1.0 + margin),
            )

        acc_areas = [p['area']        for p in acc_props]
        acc_circs = [p['circularity'] for p in acc_props]
        acc_asps  = [p['aspect_ratio']for p in acc_props]
        acc_sols  = [p['solidity']    for p in acc_props]

        new_min_area, new_max_area = bounds(acc_areas)
        new_min_circ, new_max_circ = bounds(acc_circs, margin=0.05)
        new_min_asp,  new_max_asp  = bounds(acc_asps,  margin=0.10)
        new_min_sol,  _            = bounds(acc_sols,  hi_pct=100, margin=0)

        if rej_props:
            rej_areas = np.array([p['area'] for p in rej_props])
            mean_acc  = float(np.mean(acc_areas))
            rej_small = rej_areas[rej_areas < mean_acc]
            if len(rej_small):
                new_min_area = max(new_min_area, float(np.max(rej_small)) * 1.05)
            rej_large = rej_areas[rej_areas > mean_acc]
            if len(rej_large):
                new_max_area = min(new_max_area, float(np.min(rej_large)) * 0.95)

        H, W = self.original_image.shape[:2]
        new_min_area = max(1.0,             new_min_area)
        new_max_area = min(float(H*W)*0.25, new_max_area)
        new_min_circ = max(0.0,  new_min_circ)
        new_max_circ = min(1.0,  new_max_circ)
        new_min_asp  = max(0.0,  new_min_asp)
        new_max_asp  = min(50.0, new_max_asp)
        new_min_sol  = max(0.0,  new_min_sol)

        self.params['min_area']   = new_min_area
        self.params['max_area']   = new_max_area
        self.min_circularity      = new_min_circ
        self.max_circularity      = new_max_circ
        self.min_aspect_ratio     = new_min_asp
        self.max_aspect_ratio     = new_max_asp
        self.min_solidity         = new_min_sol

        um2 = self.pixel_size_um ** 2

        if hasattr(self, 'parameter_panel'):
            pp = self.parameter_panel
            for name, val in [
                ('min_area',         new_min_area * um2),
                ('max_area',         new_max_area * um2),
                ('min_circularity',  new_min_circ),
                ('max_circularity',  new_max_circ),
                ('min_aspect_ratio', new_min_asp),
                ('max_aspect_ratio', new_max_asp),
                ('min_solidity',     new_min_sol),
            ]:
                if name in pp.sliders:
                    pp.sliders[name].set(val)

        self.selection_mode = False
        self.accepted_indices.clear()
        self.rejected_indices.clear()
        self._update_pick_reject_buttons(active=False)

        if hasattr(self, 'instruction_label'):
            self.instruction_label.config(
                text="💡 Click a green contour to inspect  |  "
                     "Click empty space to diagnose missed particles",
                bg=self.COLORS['primary']
            )

        self.update_visualization()

        messagebox.showinfo(
            "✨ Normalized",
            f"Parameters set from {len(acc_props)} accepted / {len(rej_props)} rejected:\n\n"
            f"Area:         {new_min_area * um2:.2f} – {new_max_area * um2:.2f} µm²\n"
            f"Circularity:  {new_min_circ:.2f} – {new_max_circ:.2f}\n"
            f"Aspect ratio: {new_min_asp:.2f} – {new_max_asp:.2f}\n"
            f"Solidity ≥    {new_min_sol:.2f}\n\n"
            f"Fine-tune with the parameter panel.",
            parent=self.root
        )

    def cancel_pick_reject(self):
        if self._pre_broad_params:
            self.params           = self._pre_broad_params['params'].copy()
            self.min_circularity  = self._pre_broad_params['min_circularity']
            self.max_circularity  = self._pre_broad_params['max_circularity']
            self.min_aspect_ratio = self._pre_broad_params['min_aspect_ratio']
            self.max_aspect_ratio = self._pre_broad_params['max_aspect_ratio']
            self.min_solidity     = self._pre_broad_params['min_solidity']

            if hasattr(self, 'parameter_panel'):
                pp  = self.parameter_panel
                um2 = self.pixel_size_um ** 2
                for name, val in [
                    ('min_area',         self.params['min_area'] * um2),
                    ('max_area',         self.params['max_area'] * um2),
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

        if hasattr(self, 'instruction_label'):
            self.instruction_label.config(
                text="💡 Click a green contour to inspect  |  "
                     "Click empty space to diagnose missed particles",
                bg=self.COLORS['primary']
            )
        self.update_visualization()

    def _compute_contour_properties(self, contours: list) -> list:
        props = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area <= 0:
                continue
            perimeter  = float(cv2.arcLength(cnt, True))
            circ       = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
            x, y, w, h = cv2.boundingRect(cnt)
            aspect     = float(w) / h if h > 0 else 0.0
            hull_area  = float(cv2.contourArea(cv2.convexHull(cnt)))
            solidity   = area / hull_area if hull_area > 0 else 0.0
            props.append({
                'area': area, 'circularity': circ,
                'aspect_ratio': aspect, 'solidity': solidity,
            })
        return props

    # ==================================================================
    # Image click handler
    # ==================================================================

    def on_image_click(self, event):
        if event.inaxes != self.ax_image or event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.selection_mode == 'pick_reject':
            for i, cnt in enumerate(self.contours):
                if cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0:
                    if event.button == 1:
                        if i in self.accepted_indices:
                            self.accepted_indices.discard(i)
                        else:
                            self.accepted_indices.add(i)
                            self.rejected_indices.discard(i)
                    elif event.button == 3:
                        if i in self.rejected_indices:
                            self.rejected_indices.discard(i)
                        else:
                            self.rejected_indices.add(i)
                            self.accepted_indices.discard(i)
                    break

            self._update_image_display_pick_reject()
            self._show_pick_reject_status()
            n_acc = len(self.accepted_indices)
            n_rej = len(self.rejected_indices)
            self.contour_count_label.config(
                text=f"PICK/REJECT — ✅{n_acc}  ❌{n_rej}  "
                     f"🟡{len(self.contours) - n_acc - n_rej}"
            )
            return

        clicked_contour = None
        for cnt in self.contours:
            if cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0:
                clicked_contour = cnt
                break

        if clicked_contour is not None:
            self._analyze_particle(clicked_contour)
        else:
            self._analyze_missed_particle(x, y)

    # ==================================================================
    # TARGET ANALYSIS — detected particle
    # ==================================================================

    def _analyze_particle(self, contour: np.ndarray):
        """
        Full filter-breakdown analysis for a DETECTED particle.

        Shows every measured property alongside its current threshold,
        marks each as pass (✓) or fail (✗), computes proximity to each
        boundary, and suggests tightening a filter when the particle sits
        close to an edge — helping the user identify borderline detections.
        """
        area_px   = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))

        x, y, w, h   = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0.0
        circularity  = (
            4 * np.pi * area_px / perimeter ** 2 if perimeter > 0 else 0.0
        )
        hull      = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity  = area_px / hull_area if hull_area > 0 else 0.0

        _mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(_mask, [contour], -1, 255, cv2.FILLED)
        _pix      = self.original_image[_mask > 0]
        mean_int  = float(np.mean(_pix)) if _pix.size > 0 else 0.0

        um2          = self.pixel_size_um ** 2
        area_um2     = area_px * um2
        min_area_um2 = self.params['min_area'] * um2
        max_area_um2 = self.params['max_area'] * um2

        # ── per-filter pass/fail and proximity ────────────────────────
        def _check(val, lo, hi):
            """Return (passes, icon, margin_pct_of_range)."""
            passes = lo <= val <= hi
            span   = max(hi - lo, 1e-9)
            margin = min(abs(val - lo), abs(val - hi)) / span
            return passes, "✓" if passes else "✗", margin

        area_ok,  area_ic,  area_margin  = _check(area_um2,     min_area_um2,          max_area_um2)
        circ_ok,  circ_ic,  circ_margin  = _check(circularity,  self.min_circularity,  self.max_circularity)
        asp_ok,   asp_ic,   asp_margin   = _check(aspect_ratio, self.min_aspect_ratio, self.max_aspect_ratio)
        sol_ok,   sol_ic,   sol_margin   = _check(solidity,     self.min_solidity,     1.0)
        int_ok    = mean_int <= self.max_mean_intensity
        int_ic    = "✓" if int_ok else "✗"
        int_margin= abs(mean_int - self.max_mean_intensity) / max(self.max_mean_intensity, 1.0)

        suggestions = self._generate_suggestions(
            area_px=area_px,
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            solidity=solidity,
            mean_intensity=mean_int,
        )
        self.current_suggestions = suggestions

        # ── build display text ─────────────────────────────────────────
        T = self.target_analysis_text
        T.delete('1.0', tk.END)

        T.insert('end', "▶ DETECTED PARTICLE\n\n", "header")

        # — Measurements —
        T.insert('end', "📏  Measurements\n", "section")
        T.insert('end',
            f"  Area:          {area_um2:.2f} µm²"
            f"   ({area_px:.0f} px²)\n"
            f"  Perimeter:     {perimeter:.1f} px\n"
            f"  Aspect Ratio:  {aspect_ratio:.3f}   (w={w} h={h})\n"
            f"  Circularity:   {circularity:.3f}\n"
            f"  Solidity:      {solidity:.3f}\n"
            f"  Mean BF int.:  {mean_int:.1f}\n\n"
        )

        # — Filter table —
        T.insert('end', "🔍  Filter Check  (✓ pass  ✗ fail)\n", "section")

        def _near(margin):
            """Return a ⚠ warning string when the value is within 15 % of a boundary."""
            return "  ⚠ borderline" if margin < self._SUGGESTION_MARGIN else ""

        T.insert('end',
            f"  {area_ic} Area:        {area_um2:>8.2f} µm²"
            f"  [{min_area_um2:.2f} – {max_area_um2:.2f}]"
            f"{_near(area_margin)}\n",
            "pass_row" if area_ok else "fail_row",
        )
        T.insert('end',
            f"  {circ_ic} Circularity: {circularity:>8.3f}"
            f"  [{self.min_circularity:.2f} – {self.max_circularity:.2f}]"
            f"{_near(circ_margin)}\n",
            "pass_row" if circ_ok else "fail_row",
        )
        T.insert('end',
            f"  {asp_ic} Aspect ratio:{aspect_ratio:>8.3f}"
            f"  [{self.min_aspect_ratio:.2f} – {self.max_aspect_ratio:.2f}]"
            f"{_near(asp_margin)}\n",
            "pass_row" if asp_ok else "fail_row",
        )
        T.insert('end',
            f"  {sol_ic} Solidity:    {solidity:>8.3f}"
            f"  ≥ {self.min_solidity:.2f}"
            f"{_near(sol_margin)}\n",
            "pass_row" if sol_ok else "fail_row",
        )
        T.insert('end',
            f"  {int_ic} BF intensity:{mean_int:>8.1f}"
            f"  ≤ {self.max_mean_intensity:.0f}"
            f"{_near(int_margin)}\n\n",
            "pass_row" if int_ok else "fail_row",
        )

        # — Suggestions —
        if suggestions:
            T.insert('end', "💡  Suggested adjustments\n", "section")
            for param, value in suggestions.items():
                meta  = self.SLIDER_META.get(param, {})
                unit  = meta.get('unit', '')
                fmt   = f"{value:.2f}" if meta.get('step', 1) < 1 else f"{value:.1f}"
                T.insert('end', f"  • {param}: {fmt}{unit}\n", "suggestion")
            T.insert('end', "\n")
            T.insert('end',
                "  Click  ✨ APPLY SUGGESTIONS  to use these values.\n",
                "hint"
            )
        else:
            T.insert('end', "✅  All filters passed with comfortable margins.\n", "pass_row")

        # — Tags —
        T.tag_config("header",     foreground=self.COLORS['success'],
                     font=("Segoe UI", 10, "bold"))
        T.tag_config("section",    foreground=self.COLORS['header'],
                     font=("Segoe UI", 9,  "bold"))
        T.tag_config("pass_row",   foreground="#27ae60",
                     font=("Courier New", 8))
        T.tag_config("fail_row",   foreground=self.COLORS['danger'],
                     font=("Courier New", 8, "bold"))
        T.tag_config("suggestion", foreground=self.COLORS['primary'],
                     font=("Segoe UI", 8, "bold"))
        T.tag_config("hint",       foreground=self.COLORS['warning'],
                     font=("Segoe UI", 8, "italic"))

    # ==================================================================
    # Suggestion generators
    # ==================================================================

    def _generate_suggestions(
        self,
        area_px:       float,
        circularity:   float,
        aspect_ratio:  float,
        solidity:      float = 1.0,
        mean_intensity:float = 0.0,
    ) -> Dict[str, Any]:
        """
        Generate parameter suggestions for a DETECTED particle.

        Fires when a measured value is within _SUGGESTION_MARGIN of a
        threshold (borderline) OR when the particle actually violates a
        filter (should only happen for detected contours if process_image
        was run with different settings).

        All area values in µm²; all returned values are clamped and
        rounded to their SLIDER_META step so they can be set directly.
        """
        suggestions: Dict[str, Any] = {}
        um2      = self.pixel_size_um ** 2
        area_um2 = area_px * um2

        min_area_um2 = self.params['min_area'] * um2
        max_area_um2 = self.params['max_area'] * um2

        def _near_lo(val, lo, span):
            return (lo - val) / max(span, 1e-9) < self._SUGGESTION_MARGIN or val < lo

        def _near_hi(val, hi, span):
            return (val - hi) / max(span, 1e-9) < self._SUGGESTION_MARGIN or val > hi

        area_span = max(max_area_um2 - min_area_um2, 1e-9)

        # Area
        if _near_lo(area_um2, min_area_um2, area_span):
            new_min = min_area_um2 * (1.0 - self._SUGGESTION_MARGIN * 2)
            v = self._snap_to_slider('min_area', float(new_min))
            if v is not None and v < min_area_um2:
                suggestions['min_area'] = v

        if _near_hi(area_um2, max_area_um2, area_span):
            new_max = max_area_um2 * (1.0 + self._SUGGESTION_MARGIN * 2)
            v = self._snap_to_slider('max_area', float(new_max))
            if v is not None and v > max_area_um2:
                suggestions['max_area'] = v

        # Circularity
        circ_span = max(self.max_circularity - self.min_circularity, 1e-9)
        if _near_lo(circularity, self.min_circularity, circ_span):
            v = self._snap_to_slider('min_circularity',
                                     self.min_circularity * (1.0 - self._SUGGESTION_MARGIN * 2))
            if v is not None and v < self.min_circularity:
                suggestions['min_circularity'] = v
        if _near_hi(circularity, self.max_circularity, circ_span):
            v = self._snap_to_slider('max_circularity',
                                     min(1.0, self.max_circularity * (1.0 + self._SUGGESTION_MARGIN)))
            if v is not None and v > self.max_circularity:
                suggestions['max_circularity'] = v

        # Aspect ratio
        asp_span = max(self.max_aspect_ratio - self.min_aspect_ratio, 1e-9)
        if _near_lo(aspect_ratio, self.min_aspect_ratio, asp_span):
            v = self._snap_to_slider('min_aspect_ratio',
                                     max(0.0, aspect_ratio - 0.2))
            if v is not None and v < self.min_aspect_ratio:
                suggestions['min_aspect_ratio'] = v
        if _near_hi(aspect_ratio, self.max_aspect_ratio, asp_span):
            v = self._snap_to_slider('max_aspect_ratio', aspect_ratio + 0.3)
            if v is not None and v > self.max_aspect_ratio:
                suggestions['max_aspect_ratio'] = v

        # Solidity
        if _near_lo(solidity, self.min_solidity, self.min_solidity or 1.0):
            v = self._snap_to_slider('min_solidity',
                                     float(max(0.0, self.min_solidity - self._SUGGESTION_MARGIN)))
            if v is not None and v < self.min_solidity:
                suggestions['min_solidity'] = v

        # BF intensity
        if mean_intensity > self.max_mean_intensity:
            v = self._snap_to_slider('max_mean_intensity', mean_intensity * 1.1)
            if v is not None:
                suggestions['max_mean_intensity'] = v
        elif _near_hi(mean_intensity, self.max_mean_intensity,
                      max(self.max_mean_intensity, 1.0)):
            v = self._snap_to_slider('max_mean_intensity', mean_intensity * 1.15)
            if v is not None and v > self.max_mean_intensity:
                suggestions['max_mean_intensity'] = v

        return suggestions

    def _generate_missed_particle_suggestions(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate suggestions to LOOSEN filters so that a missed particle
        would be detected.  Only fires for filters the particle actually
        fails.  Values are clamped and rounded via SLIDER_META.
        """
        suggestions: Dict[str, Any] = {}

        area_px      = analysis['area_px']
        circularity  = analysis['circularity']
        aspect_ratio = analysis['aspect_ratio']
        solidity     = analysis['solidity']
        std_intensity= analysis['std_intensity']

        um2      = self.pixel_size_um ** 2
        area_um2 = area_px * um2

        min_area_um2 = self.params['min_area'] * um2
        max_area_um2 = self.params['max_area'] * um2

        # ── Area ──────────────────────────────────────────────────────
        if area_um2 < min_area_um2:
            new_min = area_um2 * 0.7
            v = self._snap_to_slider('min_area', new_min)
            if v is not None:
                suggestions['min_area'] = v

        if area_um2 > max_area_um2:
            new_max = area_um2 * 1.3
            v = self._snap_to_slider('max_area', new_max)
            if v is not None:
                suggestions['max_area'] = v

        # ── Circularity ────────────────────────────────────────────────
        if circularity < self.min_circularity:
            v = self._snap_to_slider('min_circularity', circularity - 0.05)
            if v is not None:
                suggestions['min_circularity'] = v
        if circularity > self.max_circularity:
            v = self._snap_to_slider('max_circularity', circularity + 0.05)
            if v is not None:
                suggestions['max_circularity'] = v

        # ── Aspect ratio ───────────────────────────────────────────────
        if aspect_ratio > self.max_aspect_ratio:
            v = self._snap_to_slider('max_aspect_ratio', aspect_ratio + 0.5)
            if v is not None:
                suggestions['max_aspect_ratio'] = v
        elif aspect_ratio < self.min_aspect_ratio:
            v = self._snap_to_slider('min_aspect_ratio', max(0.0, aspect_ratio - 0.1))
            if v is not None:
                suggestions['min_aspect_ratio'] = v

        # ── Solidity ────────────────────────────────────────────────────
        if solidity < self.min_solidity:
            v = self._snap_to_slider('min_solidity', max(0.0, solidity - 0.05))
            if v is not None:
                suggestions['min_solidity'] = v
            # Low solidity → try an extra dilate pass to fill gaps
            if int(self.params['dilate_iterations']) < self.SLIDER_META['dilate_iterations']['max']:
                suggestions['dilate_iterations'] = int(self.params['dilate_iterations']) + 1

        # ── Segmentation (fallback when no geometric filter fails) ─────
        has_geometric_issue = any(
            k in suggestions
            for k in ('min_area', 'max_area', 'min_circularity', 'max_circularity',
                      'min_aspect_ratio', 'max_aspect_ratio', 'min_solidity')
        )

        if not has_geometric_issue:
            if std_intensity < 20:
                # Low contrast → more smoothing helps
                v = self._snap_to_slider(
                    'gaussian_sigma',
                    self.params['gaussian_sigma'] + 2.0
                )
                if v is not None:
                    suggestions['gaussian_sigma'] = v
            else:
                # Threshold is the likely culprit
                if self.use_intensity_threshold:
                    v = self._snap_to_slider(
                        'manual_threshold',
                        int(self.manual_threshold) + 15
                    )
                elif self.threshold_mode == "manual":
                    v = self._snap_to_slider(
                        'manual_threshold',
                        max(0, self.manual_threshold - 20)
                    )
                else:
                    v = None
                if v is not None:
                    suggestions['manual_threshold'] = v

        return suggestions

    # ==================================================================
    # TARGET ANALYSIS — missed particle
    # ==================================================================

    def _analyze_missed_particle(self, x: int, y: int):
        roi_size = 50
        h, w = self.original_image.shape[:2]

        x1 = max(0, x - roi_size); x2 = min(w, x + roi_size)
        y1 = max(0, y - roi_size); y2 = min(h, y + roi_size)

        roi_original  = self.original_image[y1:y2, x1:x2]
        roi_binary    = self.binary_mask[y1:y2, x1:x2]
        roi_processed = self.processed_image[y1:y2, x1:x2]

        analysis = self._analyze_roi_characteristics(
            roi_original, roi_binary, roi_processed, (x - x1, y - y1)
        )

        if analysis is None:
            self.target_analysis_text.delete('1.0', tk.END)
            self.target_analysis_text.insert(
                '1.0',
                "❌ No particle-like structure found near click.\n\n"
                "Try clicking closer to the centre of a dim/faint object,\n"
                "or reduce Gaussian σ to preserve low-contrast features."
            )
            self.target_analysis_text.tag_config("error", foreground="red")
            self.target_analysis_text.tag_add("error", "1.0", "end")
            return

        suggestions = self._generate_missed_particle_suggestions(analysis)
        self.current_suggestions = suggestions
        self._display_missed_particle_analysis(analysis, suggestions)

    def _analyze_roi_characteristics(
        self,
        roi_original:  np.ndarray,
        roi_binary:    np.ndarray,
        roi_processed: np.ndarray,
        click_offset:  Tuple[int, int]
    ) -> Optional[Dict[str, Any]]:

        mean_intensity = float(np.mean(roi_original))
        std_intensity  = float(np.std(roi_original))

        if std_intensity < 10:
            return None

        if self.use_intensity_threshold:
            thr = max(0, int(self.manual_threshold) + 20)
            _, test_binary = cv2.threshold(
                roi_original, thr, 255, cv2.THRESH_BINARY_INV
            )
        elif self.threshold_mode == "otsu":
            _, test_binary = cv2.threshold(
                roi_processed, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            thr = max(0, self.manual_threshold - 30)
            _, test_binary = cv2.threshold(
                roi_processed, thr, 255, cv2.THRESH_BINARY
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
                dist = np.hypot(cx - click_x, cy - click_y)
                if dist < min_dist:
                    min_dist        = dist
                    closest_contour = cnt

        if closest_contour is None or cv2.contourArea(closest_contour) < 5:
            return None

        area_px   = float(cv2.contourArea(closest_contour))
        perimeter = float(cv2.arcLength(closest_contour, True))
        x, y, w, h = cv2.boundingRect(closest_contour)
        aspect_ratio = float(w) / h if h > 0 else 0.0
        circularity  = (
            4 * np.pi * area_px / perimeter ** 2 if perimeter > 0 else 0.0
        )
        hull      = cv2.convexHull(closest_contour)
        hull_area = float(cv2.contourArea(hull))
        solidity  = area_px / hull_area if hull_area > 0 else 0.0

        mask = np.zeros(roi_original.shape, dtype=np.uint8)
        cv2.drawContours(mask, [closest_contour], -1, 255, -1)
        _roi_pixels   = roi_original[mask > 0].astype(np.float32)
        particle_mean = float(np.mean(_roi_pixels)) if _roi_pixels.size > 0 else 0.0

        return {
            'area_px':            area_px,
            'perimeter':          perimeter,
            'aspect_ratio':       aspect_ratio,
            'circularity':        circularity,
            'solidity':           solidity,
            'mean_intensity':     mean_intensity,
            'particle_intensity': particle_mean,
            'std_intensity':      std_intensity,
            'test_contour':       closest_contour,
            'test_binary':        test_binary,
            'roi_size':           roi_original.shape,
        }

    def _display_missed_particle_analysis(
        self,
        analysis:    Dict[str, Any],
        suggestions: Dict[str, Any]
    ):
        """
        Show a structured filter-breakdown for a MISSED particle.

        For every filter the particle fails, the display shows the measured
        value, the current threshold, and the gap that needs to be closed.
        """
        area_px      = analysis['area_px']
        circularity  = analysis['circularity']
        aspect_ratio = analysis['aspect_ratio']
        solidity     = analysis['solidity']

        um2          = self.pixel_size_um ** 2
        area_um2     = area_px * um2
        min_area_um2 = self.params['min_area'] * um2
        max_area_um2 = self.params['max_area'] * um2

        # ── per-filter pass/fail ───────────────────────────────────────
        area_ok = min_area_um2 <= area_um2 <= max_area_um2
        circ_ok = self.min_circularity <= circularity <= self.max_circularity
        asp_ok  = self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio
        sol_ok  = solidity >= self.min_solidity

        failing = []
        if not area_ok:
            if area_um2 < min_area_um2:
                failing.append(
                    f"  ✗ Area too small:  {area_um2:.2f} µm²"
                    f"  (min = {min_area_um2:.2f}, gap = {min_area_um2 - area_um2:.2f} µm²)"
                )
            else:
                failing.append(
                    f"  ✗ Area too large:  {area_um2:.2f} µm²"
                    f"  (max = {max_area_um2:.2f}, gap = {area_um2 - max_area_um2:.2f} µm²)"
                )
        if not circ_ok:
            if circularity < self.min_circularity:
                failing.append(
                    f"  ✗ Circularity low: {circularity:.3f}"
                    f"  (min = {self.min_circularity:.2f}, gap = {self.min_circularity - circularity:.3f})"
                )
            else:
                failing.append(
                    f"  ✗ Circularity high:{circularity:.3f}"
                    f"  (max = {self.max_circularity:.2f}, gap = {circularity - self.max_circularity:.3f})"
                )
        if not asp_ok:
            if aspect_ratio < self.min_aspect_ratio:
                failing.append(
                    f"  ✗ Aspect too low:  {aspect_ratio:.3f}"
                    f"  (min = {self.min_aspect_ratio:.2f})"
                )
            else:
                failing.append(
                    f"  ✗ Aspect too high: {aspect_ratio:.3f}"
                    f"  (max = {self.max_aspect_ratio:.2f})"
                )
        if not sol_ok:
            failing.append(
                f"  ✗ Solidity low:    {solidity:.3f}"
                f"  (min = {self.min_solidity:.2f}, gap = {self.min_solidity - solidity:.3f})"
            )

        passing = []
        if area_ok:   passing.append(f"  ✓ Area:        {area_um2:.2f} µm²")
        if circ_ok:   passing.append(f"  ✓ Circularity: {circularity:.3f}")
        if asp_ok:    passing.append(f"  ✓ Aspect:      {aspect_ratio:.3f}")
        if sol_ok:    passing.append(f"  ✓ Solidity:    {solidity:.3f}")

        T = self.target_analysis_text
        T.delete('1.0', tk.END)

        T.insert('end', "🔍 MISSED PARTICLE\n\n", "header")

        # — Measurements —
        T.insert('end', "📏  Estimated Measurements\n", "section")
        T.insert('end',
            f"  Area:          {area_um2:.2f} µm²   ({area_px:.0f} px²)\n"
            f"  Perimeter:     {analysis['perimeter']:.1f} px\n"
            f"  Aspect Ratio:  {aspect_ratio:.3f}\n"
            f"  Circularity:   {circularity:.3f}\n"
            f"  Solidity:      {solidity:.3f}\n"
            f"  Contrast (σ):  {analysis['std_intensity']:.1f}\n\n"
        )

        # — Failing filters —
        if failing:
            T.insert('end', "🚫  Failing Filters\n", "section")
            for line in failing:
                T.insert('end', line + "\n", "fail_row")
            T.insert('end', "\n")
        else:
            T.insert('end',
                "⚠  All geometric filters pass — the issue is likely\n"
                "   in the thresholding step (see suggestions below).\n\n",
                "warn_row"
            )

        # — Passing filters —
        if passing:
            T.insert('end', "✅  Passing Filters\n", "section")
            for line in passing:
                T.insert('end', line + "\n", "pass_row")
            T.insert('end', "\n")

        # — Suggestions —
        if suggestions:
            T.insert('end', "💡  Suggested Adjustments\n", "section")
            for param, value in suggestions.items():
                meta = self.SLIDER_META.get(param, {})
                unit = meta.get('unit', '')
                step = meta.get('step', 1)
                fmt  = f"{value:.2f}" if step < 1 else f"{value:.1f}"
                T.insert('end', f"  • {param}: {fmt}{unit}\n", "suggestion")
            T.insert('end', "\n")
            T.insert('end',
                "  Click  ✨ APPLY SUGGESTIONS  to try these settings.\n",
                "hint"
            )
        else:
            T.insert('end',
                "  No automatic suggestions available.\n"
                "  Try lowering Gaussian σ or switching threshold mode.\n",
                "warn_row"
            )

        # — Tags —
        T.tag_config("header",     foreground=self.COLORS['warning'],
                     font=("Segoe UI", 10, "bold"))
        T.tag_config("section",    foreground=self.COLORS['header'],
                     font=("Segoe UI", 9,  "bold"))
        T.tag_config("pass_row",   foreground="#27ae60",
                     font=("Courier New", 8))
        T.tag_config("fail_row",   foreground=self.COLORS['danger'],
                     font=("Courier New", 8, "bold"))
        T.tag_config("warn_row",   foreground=self.COLORS['warning'],
                     font=("Segoe UI", 8, "italic"))
        T.tag_config("suggestion", foreground=self.COLORS['primary'],
                     font=("Segoe UI", 8, "bold"))
        T.tag_config("hint",       foreground=self.COLORS['warning'],
                     font=("Segoe UI", 8, "italic"))

        self._highlight_missed_particle(analysis['test_contour'])

    def _highlight_missed_particle(self, contour: np.ndarray):
        self.ax_image.clear()
        display = (
            cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB)
            if self.original_image.ndim == 2
            else self.original_image.copy()
        )
        cv2.drawContours(display, self.contours, -1, (0, 255, 0),  2)
        cv2.drawContours(display, [contour],     -1, (255, 165, 0), 3)
        self.ax_image.imshow(display)
        self.ax_image.set_title(
            "Original + Contours  (🟠 = missed particle candidate)",
            fontsize=12, fontweight='bold', pad=10
        )
        self.ax_image.axis("off")
        self.canvas_image.draw()

    # ==================================================================
    # Apply suggestions
    # ==================================================================

    def apply_suggestions(self, event=None):
        if not self.current_suggestions:
            self.target_analysis_text.delete('1.0', tk.END)
            self.target_analysis_text.insert('1.0', "No suggestions to apply.")
            return

        panel_sliders: Dict[str, Any] = {}
        if hasattr(self, 'parameter_panel'):
            panel_sliders = self.parameter_panel.sliders

        for param, value in self.current_suggestions.items():
            if param == 'threshold_mode':
                self.threshold_mode            = value
                self.use_intensity_threshold   = (value == "intensity")
                if self.use_intensity_threshold:
                    self.intensity_threshold_value = float(self.manual_threshold)

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

            elif param in ('min_area', 'max_area'):
                # suggestions already in µm² (from _generate_*); convert to px
                um2 = self.pixel_size_um ** 2
                self.params[param] = float(value) / um2
                if param in panel_sliders:
                    panel_sliders[param].set(value)      # slider shows µm²

            elif param in panel_sliders:
                panel_sliders[param].set(value)

            elif param in self.sliders:
                self.sliders[param].set_val(value)

            elif param in ('min_circularity', 'max_circularity',
                           'min_aspect_ratio', 'max_aspect_ratio',
                           'min_solidity', 'max_mean_intensity'):
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

            elif param in ('dilate_iterations', 'erode_iterations'):
                self.params[param] = value
                if param in panel_sliders:
                    panel_sliders[param].set(value)

        self.current_suggestions = {}

        self.target_analysis_text.delete('1.0', tk.END)
        self.target_analysis_text.insert(
            '1.0', "✅ Suggestions applied — re-running segmentation…", "success"
        )
        self.target_analysis_text.tag_config(
            "success",
            foreground=self.COLORS['success'],
            font=("Segoe UI", 9, "bold")
        )
        self.update_visualization()

    def load_new_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("All files", "*.*"),
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
# SECTION 6: Main Menu (Pathogen Selection + Tuner Setup)
# ==================================================

class PathogenConfigManager:
    """Unified pathogen configuration manager with integrated tuner setup"""

    # NOTE: PATHOGENS hard-coded dict removed.
    # The bacteria list is now loaded dynamically from the registry
    # (bacteria_configs/registry.json) via _bacteria_registry.all().

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
        self.root.minsize(980, 800)
        self.root.maxsize(1440, 900)
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

        # Container widget for all pathogen cards; populated by
        # _create_pathogen_section and rebuilt by _rebuild_pathogen_cards.
        self._cards_outer: Optional[tk.Frame] = None

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

    # ------------------------------------------------------------------
    # Pathogen section — driven entirely by the registry
    # ------------------------------------------------------------------

    def _create_pathogen_section(self, parent: tk.Frame) -> None:
        """Build the scrollable bacteria card list from the registry."""
        parent["bg"] = self.COLORS["panel"]
        self._section_title(parent, "1", "Select pathogen")

        # ── toolbar: hint + Register button ──────────────────────────
        ctrl = tk.Frame(parent, bg=self.COLORS["panel"])
        ctrl.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            ctrl,
            text="Click a card to select  ·  use  ✕  to remove",
            font=("Segoe UI", 9),
            bg=self.COLORS["panel"], fg=self.COLORS["muted"],
        ).pack(side=tk.LEFT)

        tk.Button(
            ctrl,
            text="➕ Register New",
            font=("Segoe UI", 9, "bold"),
            bg=self.COLORS["success"], fg="white",
            relief=tk.FLAT, cursor="hand2", padx=10, pady=4,
            command=self._open_register_dialog,
        ).pack(side=tk.RIGHT)

        # ── scrollable card container ─────────────────────────────────
        self._cards_outer = tk.Frame(parent, bg=self.COLORS["panel"])
        self._cards_outer.pack(fill=tk.BOTH, expand=True)

        self._rebuild_pathogen_cards()

    def _rebuild_pathogen_cards(self) -> None:
        """Destroy all existing cards and recreate them from the registry."""
        if self._cards_outer is None:
            return

        for widget in self._cards_outer.winfo_children():
            widget.destroy()

        self.pathogen_cards    = {}
        self.card_indicators   = {}
        self.card_contents     = {}
        self.card_left_frames  = {}
        self.card_right_frames = {}

        bacteria = _bacteria_registry.all()

        if not bacteria:
            tk.Label(
                self._cards_outer,
                text="No bacteria registered.\nClick  ➕ Register New  to add one.",
                font=("Segoe UI", 10, "italic"),
                bg=self.COLORS["panel"], fg=self.COLORS["muted"],
                justify=tk.CENTER,
            ).pack(pady=24)
            return

        for config_key, meta in bacteria.items():
            display_name = meta["display_name"]
            card = self._create_pathogen_card(
                self._cards_outer,
                display_name,
                {
                    "config_key":  config_key,
                    "description": meta.get("description", ""),
                    "common_in":   meta.get("common_in", ""),
                    "validated":   meta.get("validated", False),
                },
            )
            card.pack(fill=tk.X, pady=(0, 10))

        # Re-apply selection highlight if the previously chosen name still exists
        if self.selected_pathogen and self.selected_pathogen in self.pathogen_cards:
            self._select_pathogen(self.selected_pathogen)

    def _create_pathogen_card(self, parent: tk.Frame, pathogen_name: str, info: dict) -> tk.Frame:
        """Build one bacteria card with a  ✕  remove button."""
        C = self.COLORS
        validated = bool(info.get("validated", False))

        card = tk.Frame(parent, relief=tk.FLAT, borderwidth=0, cursor="hand2")
        card["bg"] = C["button"]
        self.pathogen_cards[pathogen_name] = card

        def select_this(event=None):
            self._select_pathogen(pathogen_name)

        border = tk.Frame(card)
        border.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        border["bg"] = C["panel_border"]
        border.bind("<Button-1>", select_this)

        content = tk.Frame(border)
        content.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        content["bg"] = C["button"]
        content.bind("<Button-1>", select_this)

        for w in (card, border, content):
            w.bind("<Enter>", lambda e, n=pathogen_name: self._hover_card(n, True))
            w.bind("<Leave>", lambda e, n=pathogen_name: self._hover_card(n, False))

        self.card_contents[pathogen_name] = content

        # ── Left side: name + description ────────────────────────────
        left = tk.Frame(content, bg=C["button"])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left.bind("<Button-1>", select_this)
        self.card_left_frames[pathogen_name] = left

        checkmark = "  ✓" if validated else ""
        name_lbl = tk.Label(
            left,
            text=f"🔬 {pathogen_name}{checkmark}",
            font=("Segoe UI", 12, "bold"),
            anchor=tk.W,
        )
        name_lbl.pack(anchor=tk.W)
        name_lbl["bg"] = C["button"]
        name_lbl["fg"] = C["success"] if validated else C["fg"]
        name_lbl.bind("<Button-1>", select_this)

        for text, fg_color in [
            (info.get("description", ""),           C["fg"]),
            (f"📌 {info.get('common_in', '')}",     C["warning"]),
        ]:
            body = text.replace("📌 ", "").strip()
            if body:
                lbl = tk.Label(left, text=text, font=("Segoe UI", 9), anchor=tk.W)
                lbl.pack(anchor=tk.W, pady=(3, 0))
                lbl["bg"] = C["button"]
                lbl["fg"] = fg_color
                lbl.bind("<Button-1>", select_this)

        if not validated:
            tk.Label(
                left,
                text="⚠ Not yet validated for multi-scan",
                font=("Segoe UI", 8, "italic"),
                anchor=tk.W,
                bg=C["button"], fg=C["error"],
            ).pack(anchor=tk.W, pady=(2, 0))

        # ── Right side: selection indicator + remove button ───────────
        right = tk.Frame(content, bg=C["button"])
        right.pack(side=tk.RIGHT, padx=(10, 0))
        right.bind("<Button-1>", select_this)
        self.card_right_frames[pathogen_name] = right

        indicator = tk.Label(right, text="○", font=("Segoe UI", 18))
        indicator.pack()
        indicator["bg"] = C["button"]
        indicator["fg"] = C["muted"]
        indicator.bind("<Button-1>", select_this)
        self.card_indicators[pathogen_name] = indicator

        tk.Button(
            right,
            text="✕",
            font=("Segoe UI", 9, "bold"),
            bg=C["button"], fg=C["error"],
            relief=tk.FLAT, cursor="hand2",
            padx=4, pady=2,
            activebackground=C["button_hover"],
            activeforeground=C["error"],
            command=lambda n=pathogen_name: self._confirm_remove_bacteria(n),
        ).pack(pady=(6, 0))

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

    # ------------------------------------------------------------------
    # Registration dialog
    # ------------------------------------------------------------------

    def _open_register_dialog(self) -> None:
        """Open RegisterBacteriaDialog and handle its result."""

        def on_registered(config_key: str, open_tuner: bool) -> None:
            self._rebuild_pathogen_cards()

            entry = _bacteria_registry.get(config_key)
            if entry:
                display_name = entry["display_name"]
                if display_name in self.pathogen_cards:
                    self._select_pathogen(display_name)

            if open_tuner:
                image_ok = (
                    bool(self.image_path_var.get())
                    and Path(self.image_path_var.get()).exists()
                )
                name = entry["display_name"] if entry else config_key

                if image_ok:
                    messagebox.showinfo(
                        "Registered",
                        f"'{name}' has been registered.\n\n"
                        f"Launching tuner to configure segmentation parameters…",
                        parent=self.root,
                    )
                    self._start_tuner()
                else:
                    messagebox.showinfo(
                        "Registered",
                        f"'{name}' has been registered.\n\n"
                        f"Now select an image file, then click\n"
                        f"✅ Start tuner  to configure segmentation parameters.",
                        parent=self.root,
                    )

        RegisterBacteriaDialog(self.root, on_success=on_registered)

    # ------------------------------------------------------------------
    # Remove bacterium
    # ------------------------------------------------------------------

    def _confirm_remove_bacteria(self, pathogen_name: str) -> None:
        """Two-step confirmation: remove from registry, optionally delete JSON."""

        config_key: Optional[str] = None
        for key, meta in _bacteria_registry.all().items():
            if meta["display_name"] == pathogen_name:
                config_key = key
                break

        if config_key is None:
            messagebox.showerror(
                "Error",
                f"Could not find a registry entry for '{pathogen_name}'.",
                parent=self.root,
            )
            return

        if len(_bacteria_registry.all()) <= 1:
            messagebox.showwarning(
                "Cannot Remove",
                "At least one bacterium must remain in the registry.",
                parent=self.root,
            )
            return

        confirmed = messagebox.askyesno(
            "Remove Bacterium",
            f"Remove  '{pathogen_name}'  from the bacteria registry?\n\n"
            f"It will disappear from the Tuner list and the\n"
            f"automated multi-scan whitelist.",
            parent=self.root,
        )
        if not confirmed:
            return

        delete_json = False
        if _bacteria_registry.has_json_config(config_key):
            delete_json = messagebox.askyesno(
                "Delete Config File?",
                f"Also delete the tuned parameter file?\n\n"
                f"  bacteria_configs/{config_key}.json\n\n"
                f"Click  YES  to permanently delete it.\n"
                f"Click  NO   to keep it for future use.",
                parent=self.root,
            )

        _bacteria_registry.remove(config_key, delete_json=delete_json)

        if self.selected_pathogen == pathogen_name:
            self.selected_pathogen = None

        self._rebuild_pathogen_cards()

        detail = " Config file also deleted." if delete_json else ""
        messagebox.showinfo(
            "Removed",
            f"'{pathogen_name}' has been removed from the registry.{detail}",
            parent=self.root,
        )

    # ------------------------------------------------------------------
    # Image section, config section, action buttons
    # ------------------------------------------------------------------

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

        # ── Look up the canonical key from the registry ───────────────
        config_key: Optional[str] = None
        for key, meta in _bacteria_registry.all().items():
            if meta["display_name"] == self.selected_pathogen:
                config_key = key
                break

        # Fallback: derive from display name (should not normally be needed)
        if config_key is None:
            config_key = re.sub(r'[^a-z0-9]+', '_', self.selected_pathogen.lower()).strip('_')
            config_key = re.sub(r'_+', '_', config_key)

        print(f"\n{'=' * 60}")
        print("STARTING SEGMENTATION TUNER")
        print(f"{'=' * 60}")
        print(f"Pathogen:   {self.selected_pathogen}")
        print(f"Config key: {config_key}")
        print(f"Image:      {Path(self.image_path_var.get()).name}")
        print(f"Structure:  {self.structure_var.get()}")
        print(f"Mode:       {self.mode_var.get()}")
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
                config_key=config_key,
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
Version 2.1

Interactive parameter tuning for image analysis of:

• Proteus mirabilis
• Klebsiella pneumoniae
• Streptococcus mitis
• Any newly registered bacterium

Features:
🎨 Real-time segmentation preview
📊 Histogram analysis
🎯 Click-to-analyze particles
🎯 Pick / Reject / Normalize workflow
➕ Register new bacteria on-the-fly
🗑  Remove bacteria from the registry
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
# Register Bacteria Dialog
# Placed after PathogenConfigManager so it can reference
# PathogenConfigManager.COLORS at instantiation time.
# ==================================================

class RegisterBacteriaDialog(tk.Toplevel):
    """Modal dialog that registers a new bacterium into the persistent registry."""

    def __init__(
        self,
        parent: tk.Misc,
        on_success: Optional[object] = None,
    ) -> None:
        super().__init__(parent)
        self._C = PathogenConfigManager.COLORS
        self.on_success = on_success
        self.result_key: Optional[str] = None

        self.title("➕ Register New Bacterium")
        self.geometry("560x530")
        self.resizable(False, False)
        self.grab_set()
        self.configure(bg=self._C["bg"])

        self._build_ui()
        self._center(parent)

    def _center(self, parent: tk.Misc) -> None:
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width()  - 560) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - 530) // 2
        self.geometry(f"560x530+{x}+{y}")

    def _lbl(self, parent: tk.Widget, text: str, bold: bool = False) -> tk.Label:
        lbl = tk.Label(
            parent, text=text,
            font=("Segoe UI", 10, "bold" if bold else "normal"),
            bg=parent["bg"], fg=self._C["fg"],
        )
        lbl.pack(anchor="w", pady=(8, 2))
        return lbl

    def _entry(self, parent: tk.Widget, var: tk.StringVar) -> tk.Entry:
        e = tk.Entry(
            parent, textvariable=var,
            font=("Segoe UI", 10),
            bg=self._C["button"], fg=self._C["fg"],
            insertbackground=self._C["fg"],
            relief=tk.FLAT, bd=6,
        )
        e.pack(fill=tk.X, pady=(0, 2))
        return e

    def _build_ui(self) -> None:
        C = self._C

        pad = tk.Frame(self, bg=C["bg"], padx=18, pady=14)
        pad.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            pad, text="➕  Register New Bacterium",
            font=("Segoe UI", 14, "bold"),
            bg=C["bg"], fg=C["success"],
        ).pack(anchor="w", pady=(0, 10))

        pnl   = tk.Frame(pad, bg=C["panel_border"], relief=tk.FLAT, bd=1)
        pnl.pack(fill=tk.BOTH, expand=True)
        inner = tk.Frame(pnl, bg=C["panel"], padx=14, pady=10)
        inner.pack(fill=tk.BOTH, expand=True)

        self._lbl(inner, "Display Name  (e.g. Escherichia coli) *", bold=True)
        self.name_var = tk.StringVar()
        name_e = self._entry(inner, self.name_var)
        name_e.bind("<KeyRelease>", self._auto_fill_key)
        name_e.focus_set()

        self._lbl(inner, "Config Key  (auto-filled · must be unique) *")
        self.key_var = tk.StringVar()
        key_e = self._entry(inner, self.key_var)
        key_e.bind("<KeyRelease>", lambda _e: self._check_key())

        self.key_status_lbl = tk.Label(
            inner, text="",
            font=("Segoe UI", 8, "italic"),
            bg=C["panel"], fg=C["warning"],
        )
        self.key_status_lbl.pack(anchor="w")

        self._lbl(inner, "Short description")
        self.desc_var = tk.StringVar()
        self._entry(inner, self.desc_var)

        self._lbl(inner, "Commonly associated with")
        self.common_var = tk.StringVar()
        self._entry(inner, self.common_var)

        self.validated_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            inner,
            text="Include in automated multi-scan  (mark as validated)",
            variable=self.validated_var,
            font=("Segoe UI", 9),
            bg=C["panel"], fg=C["fg"],
            selectcolor=C["selected"],
            activebackground=C["panel"],
            cursor="hand2",
        ).pack(anchor="w", pady=(12, 2))

        tk.Label(
            inner,
            text="⚠  Only enable after segmentation parameters have been tuned and validated.",
            font=("Segoe UI", 8, "italic"),
            bg=C["panel"], fg=C["warning"],
        ).pack(anchor="w")

        btn_row = tk.Frame(pad, bg=C["bg"])
        btn_row.pack(fill=tk.X, pady=(14, 0))

        for text, color, open_t in [
            ("✅ Register + Open Tuner", C["success"], True),
            ("✅ Register Only",          C["accent"],  False),
        ]:
            tk.Button(
                btn_row, text=text,
                font=("Segoe UI", 10, "bold"),
                bg=color, fg="white",
                relief=tk.FLAT, cursor="hand2", padx=12, pady=8,
                command=lambda ot=open_t: self._submit(open_tuner=ot),
            ).pack(side=tk.RIGHT, padx=(6, 0))

        tk.Button(
            btn_row, text="Cancel",
            font=("Segoe UI", 10),
            bg=C["button"], fg=C["fg"],
            relief=tk.FLAT, cursor="hand2", padx=12, pady=8,
            command=self.destroy,
        ).pack(side=tk.RIGHT)

    def _auto_fill_key(self, _event=None) -> None:
        raw = self.name_var.get()
        key = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
        self.key_var.set(key)
        self._check_key()

    def _check_key(self) -> bool:
        key = self.key_var.get().strip()
        C   = self._C

        if not key:
            self.key_status_lbl.config(text="", fg=C["warning"])
            return False

        if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            self.key_status_lbl.config(
                text="✗  Must start with a letter; only  a-z  0-9  _  allowed.",
                fg=C["error"],
            )
            return False

        if _bacteria_registry.key_exists(key):
            self.key_status_lbl.config(
                text=f"✗  '{key}'  is already registered.",
                fg=C["error"],
            )
            return False

        self.key_status_lbl.config(
            text=f"✓  '{key}'  is available.",
            fg=C["success"],
        )
        return True

    def _submit(self, *, open_tuner: bool = True) -> None:
        name = self.name_var.get().strip()
        key  = self.key_var.get().strip()

        if not name:
            messagebox.showerror("Missing field", "Display name is required.", parent=self)
            return

        if not self._check_key():
            messagebox.showerror(
                "Invalid key",
                "The config key is invalid or already taken.\n"
                "Edit it and try again.",
                parent=self,
            )
            return

        try:
            _bacteria_registry.register(
                display_name=name,
                description=self.desc_var.get().strip(),
                common_in=self.common_var.get().strip(),
                validated=self.validated_var.get(),
                config_key=key,
            )
            self.result_key = key

            if callable(self.on_success):
                self.on_success(key, open_tuner)

            self.destroy()

        except ValueError as exc:
            messagebox.showerror("Registration error", str(exc), parent=self)


# ==================================================
# SECTION 7: Main Entry Point
# ==================================================

def main():
    root = tk.Tk()
    app  = PathogenConfigManager(root)
    root.mainloop()


if __name__ == "__main__":
    main()