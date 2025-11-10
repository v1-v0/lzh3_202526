#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive Bacteria Segmentation Parameter Tuner
- Real-time preview
- Entry boxes for input + Progress bars for visual feedback
- Tooltips
- Reset to defaults
- All images auto-fit (zoom out)
"""

import cv2
import numpy as np
from pathlib import Path
from scipy import ndimage
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from typing import cast, List, Tuple, Dict


# --------------------------------------------------------------------- #
# ToolTip class
# --------------------------------------------------------------------- #
class ToolTip:
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(tw, text=self.text, justify=tk.LEFT,
                          background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                          font=("Segoe UI", 9), padding=(5, 3))
        label.pack()

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class SegmentationViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Bacteria Segmentation Parameter Tuner")
        self.root.geometry("1400x900")

        self.original_image: np.ndarray | None = None

        # Default parameter values
        self.default_params = {
            'use_otsu': False,
            'manual_threshold': 60,
            'enable_clahe': False,
            'clahe_clip': 2.0,
            'clahe_tile': 8,
            'open_kernel': 1,
            'close_kernel': 5,
            'open_iter': 1,
            'close_iter': 1,
            'min_area': 50,
            'watershed_dilate': 20,
        }

        # Tkinter variables
        self.params: Dict[str, tk.Variable] = {
            'use_otsu': tk.BooleanVar(value=self.default_params['use_otsu']),
            'manual_threshold': tk.IntVar(value=self.default_params['manual_threshold']),
            'enable_clahe': tk.BooleanVar(value=self.default_params['enable_clahe']),
            'clahe_clip': tk.DoubleVar(value=self.default_params['clahe_clip']),
            'clahe_tile': tk.IntVar(value=self.default_params['clahe_tile']),
            'open_kernel': tk.IntVar(value=self.default_params['open_kernel']),
            'close_kernel': tk.IntVar(value=self.default_params['close_kernel']),
            'open_iter': tk.IntVar(value=self.default_params['open_iter']),
            'close_iter': tk.IntVar(value=self.default_params['close_iter']),
            'min_area': tk.IntVar(value=self.default_params['min_area']),
            'watershed_dilate': tk.IntVar(value=self.default_params['watershed_dilate']),
        }

        self.entries: Dict[str, tk.Entry] = {}
        self.progressbars: Dict[str, ttk.Progressbar] = {}

        self.setup_ui()
        self.root.bind("<Configure>", lambda e: self.root.after_idle(self.update_preview))

    # --------------------------------------------------------------------- #
    # UI construction
    # --------------------------------------------------------------------- #
    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel: controls
        control_frame = ttk.Frame(main_frame, width=380)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # Buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        load_btn = ttk.Button(btn_frame, text="Load Brightfield Image", command=self.load_image)
        load_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ToolTip(load_btn, "Select a brightfield TIFF image.")

        reset_btn = ttk.Button(btn_frame, text="Reset to Defaults", command=self.reset_to_defaults)
        reset_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        ToolTip(reset_btn, "Restore all parameters to default values.")

        # Scrollable controls
        canvas = tk.Canvas(control_frame)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Build controls
        self.create_threshold_controls(scrollable_frame)
        self.create_clahe_controls(scrollable_frame)
        self.create_morphology_controls(scrollable_frame)
        self.create_watershed_controls(scrollable_frame)

        # Right panel: image tabs
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(image_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_original = ttk.Frame(self.notebook)
        self.tab_enhanced = ttk.Frame(self.notebook)
        self.tab_threshold = ttk.Frame(self.notebook)
        self.tab_morphology = ttk.Frame(self.notebook)
        self.tab_contours = ttk.Frame(self.notebook)

        tabs = [
            ("Original", self.tab_original),
            ("CLAHE Enhanced", self.tab_enhanced),
            ("Threshold", self.tab_threshold),
            ("Morphology", self.tab_morphology),
            ("Final Contours", self.tab_contours),
        ]
        for name, tab in tabs:
            self.notebook.add(tab, text=name)

        self.canvas_original = tk.Canvas(self.tab_original, bg='gray')
        self.canvas_enhanced = tk.Canvas(self.tab_enhanced, bg='gray')
        self.canvas_threshold = tk.Canvas(self.tab_threshold, bg='gray')
        self.canvas_morphology = tk.Canvas(self.tab_morphology, bg='gray')
        self.canvas_contours = tk.Canvas(self.tab_contours, bg='gray')

        for canvas in [self.canvas_original, self.canvas_enhanced,
                       self.canvas_threshold, self.canvas_morphology,
                       self.canvas_contours]:
            canvas.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Load an image to begin")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # --------------------------------------------------------------------- #
    # Entry + Progressbar row
    # --------------------------------------------------------------------- #
    def add_entry_with_progress(self, parent, label_text, tooltip_text, var, min_val, max_val,
                                resolution=1.0, is_float=False):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)

        label = ttk.Label(frame, text=label_text, width=24, anchor=tk.W)
        label.pack(side=tk.LEFT)
        ToolTip(label, tooltip_text)

        var_name = next(k for k, v in self.params.items() if v == var)

        entry = ttk.Entry(frame, width=8, justify=tk.RIGHT)
        entry.pack(side=tk.LEFT, padx=(5, 5))
        entry.insert(0, str(var.get()))
        entry.bind('<Return>', lambda e: self.sync_entry(var_name, entry, min_val, max_val,
                                                         resolution, is_float))
        entry.bind('<FocusOut>', lambda e: self.sync_entry(var_name, entry, min_val, max_val,
                                                          resolution, is_float))
        ToolTip(entry, tooltip_text)

        pb = ttk.Progressbar(frame, orient=tk.HORIZONTAL, mode='determinate', length=150)
        pb.pack(side=tk.RIGHT, padx=(0, 5))
        ToolTip(pb, tooltip_text)

        self.entries[var_name] = entry
        self.progressbars[var_name] = pb
        self.update_progressbar(var_name)

        return entry

    # --------------------------------------------------------------------- #
    # Sync entry → variable + progressbar
    # --------------------------------------------------------------------- #
    def sync_entry(self, var_name: str, entry: tk.Entry, min_val: float, max_val: float,
                   resolution: float, is_float: bool) -> None:
        try:
            raw = entry.get().strip()
            value = float(raw) if is_float else int(float(raw))
            value = max(min_val, min(max_val, value))

            if var_name in ('open_kernel', 'close_kernel'):
                value = int(value)
                if value % 2 == 0:
                    lower = value - 1
                    if lower >= min_val:
                        value = lower
                    else:
                        value = value + 1
                value = max(min_val, min(max_val, value))

            elif resolution != 1.0:
                value = round(value / resolution) * resolution
                if is_float and resolution == 0.1:
                    value = round(value, 1)

            self.params[var_name].set(value)
            entry.delete(0, tk.END)
            entry.insert(0, str(int(value) if not is_float else value))
            self.update_progressbar(var_name)
            self.update_preview()

        except ValueError:
            messagebox.showwarning("Invalid Input",
                                   f"Enter a valid number for {var_name.replace('_', ' ')}.")
            entry.delete(0, tk.END)
            entry.insert(0, str(self.params[var_name].get()))

    # --------------------------------------------------------------------- #
    # Update progress bar
    # --------------------------------------------------------------------- #
    def update_progressbar(self, var_name: str) -> None:
        pb = self.progressbars[var_name]
        value = self.params[var_name].get()

        ranges = {
            'manual_threshold': (0, 255),
            'clahe_clip': (1.0, 10.0),
            'clahe_tile': (4, 32),
            'open_kernel': (1, 15),
            'close_kernel': (1, 15),
            'open_iter': (1, 5),
            'close_iter': (1, 5),
            'min_area': (10, 500),
            'watershed_dilate': (1, 20),
        }
        min_val, max_val = ranges[var_name]
        percent = (value - min_val) / (max_val - min_val) * 100
        pb['value'] = percent

    # --------------------------------------------------------------------- #
    # Control groups
    # --------------------------------------------------------------------- #
    def create_threshold_controls(self, parent):
        f = ttk.LabelFrame(parent, text="Thresholding", padding=10)
        f.pack(fill=tk.X, pady=5)
        cb = ttk.Checkbutton(f, text="Use Otsu (Auto)", variable=self.params['use_otsu'],
                             command=self.update_preview)
        cb.pack(anchor=tk.W)
        ToolTip(cb, "Auto threshold using Otsu's method.")
        self.add_entry_with_progress(f, "Manual Threshold:", "0–255. Lower = more foreground.",
                                     self.params['manual_threshold'], 0, 255, resolution=1, is_float=False)

    def create_clahe_controls(self, parent):
        f = ttk.LabelFrame(parent, text="CLAHE Enhancement", padding=10)
        f.pack(fill=tk.X, pady=5)
        cb = ttk.Checkbutton(f, text="Enable CLAHE", variable=self.params['enable_clahe'],
                             command=self.update_preview)
        cb.pack(anchor=tk.W)
        ToolTip(cb, "Improves contrast in low-light regions.")
        self.add_entry_with_progress(f, "Clip Limit:", "1.0–10.0. Higher = stronger contrast.",
                                     self.params['clahe_clip'], 1.0, 10.0, resolution=0.1, is_float=True)
        self.add_entry_with_progress(f, "Tile Size:", "4–32. Smaller = more local enhancement.",
                                     self.params['clahe_tile'], 4, 32, resolution=1, is_float=False)

    def create_morphology_controls(self, parent):
        f = ttk.LabelFrame(parent, text="Morphological Operations", padding=10)
        f.pack(fill=tk.X, pady=5)
        self.add_entry_with_progress(f, "Opening Kernel (odd):", "1,3,5,...,15. Removes noise.",
                                     self.params['open_kernel'], 1, 15, resolution=2, is_float=False)
        self.add_entry_with_progress(f, "Closing Kernel (odd):", "1,3,5,...,15. Fills holes.",
                                     self.params['close_kernel'], 1, 15, resolution=2, is_float=False)
        self.add_entry_with_progress(f, "Opening Iterations:", "1–5. More = stronger.",
                                     self.params['open_iter'], 1, 5, resolution=1, is_float=False)
        self.add_entry_with_progress(f, "Closing Iterations:", "1–5. More = stronger.",
                                     self.params['close_iter'], 1, 5, resolution=1, is_float=False)

    def create_watershed_controls(self, parent):
        f = ttk.LabelFrame(parent, text="Watershed & Filtering", padding=10)
        f.pack(fill=tk.X, pady=5)
        self.add_entry_with_progress(f, "Minimum Area:", "10–500 px². Ignore small objects.",
                                     self.params['min_area'], 10, 500, resolution=1, is_float=False)
        self.add_entry_with_progress(f, "Watershed Threshold %:", "1–20. Higher = more merging.",
                                     self.params['watershed_dilate'], 1, 20, resolution=1, is_float=False)

    # --------------------------------------------------------------------- #
    # Reset
    # --------------------------------------------------------------------- #
    def reset_to_defaults(self):
        if messagebox.askyesno("Reset", "Reset all parameters to defaults?"):
            for key, val in self.default_params.items():
                self.params[key].set(val)
                if key in self.entries:
                    self.entries[key].delete(0, tk.END)
                    self.entries[key].insert(0, str(val))
                    self.update_progressbar(key)
            self.update_preview()
            self.status_var.set("Parameters reset")

    # --------------------------------------------------------------------- #
    # Load image
    # --------------------------------------------------------------------- #
    def load_image(self):
        filepath = filedialog.askopenfilename(
            title="Select Brightfield Image",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if not filepath:
            return
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.status_var.set(f"Error: Cannot read {filepath}")
            return
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.original_image = img
        self.status_var.set(f"Loaded: {Path(filepath).name} ({img.shape[1]}x{img.shape[0]})")
        self.update_preview()

    # --------------------------------------------------------------------- #
    # Segmentation
    # --------------------------------------------------------------------- #
    def segment_bacteria(self, gray_bf: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        if self.params['enable_clahe'].get():
            bf8 = cv2.convertScaleAbs(gray_bf)
            clahe = cv2.createCLAHE(clipLimit=self.params['clahe_clip'].get(),
                                    tileGridSize=(self.params['clahe_tile'].get(),)*2)
            enhanced = clahe.apply(bf8)
        else:
            enhanced = cv2.convertScaleAbs(gray_bf)

        if self.params['use_otsu'].get():
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(enhanced, self.params['manual_threshold'].get(),
                                      255, cv2.THRESH_BINARY_INV)

        open_k_size = self.params['open_kernel'].get()
        if open_k_size % 2 == 0:
            open_k_size = max(1, open_k_size - 1)
        close_k_size = self.params['close_kernel'].get()
        if close_k_size % 2 == 0:
            close_k_size = max(1, close_k_size - 1)

        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k_size, open_k_size))
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k_size, close_k_size))

        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k,
                                 iterations=self.params['open_iter'].get())
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_k,
                                  iterations=self.params['close_iter'].get())

        distance = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(distance,
                                   self.params['watershed_dilate'].get() * distance.max() / 100.0,
                                   255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, open_k, iterations=1)

        markers, _ = cast(Tuple[np.ndarray, int], ndimage.label(sure_fg))
        markers = markers.astype(np.int32)
        markers += 1
        markers[cleaned == 0] = 0

        watershed_input = cv2.cvtColor(gray_bf, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(watershed_input, markers)

        contour_mask = (markers > 1).astype(np.uint8) * 255
        res = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cast(List[np.ndarray], res[-2])

        bacteria = [c for c in contours if cv2.contourArea(c) >= self.params['min_area'].get()]

        return enhanced, thresh, cleaned, bacteria

    # --------------------------------------------------------------------- #
    # Update preview + progress bars
    # --------------------------------------------------------------------- #
    def update_preview(self) -> None:
        if self.original_image is None:
            return
        try:
            enhanced, thresh, cleaned, bacteria = self.segment_bacteria(self.original_image)
            self.status_var.set(f"Detected {len(bacteria)} bacteria")

            for key in self.progressbars:
                self.update_progressbar(key)

            self.display_image(self.original_image, self.canvas_original)
            self.display_image(enhanced, self.canvas_enhanced)
            self.display_image(thresh, self.canvas_threshold)
            self.display_image(cleaned, self.canvas_morphology)

            contour_img = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_img, bacteria, -1, (0, 255, 0), 2)
            self.display_image(contour_img, self.canvas_contours)
        except Exception as e:
            self.status_var.set(f"Error: {e}")

    # --------------------------------------------------------------------- #
    # Display image: FIT TO CANVAS
    # --------------------------------------------------------------------- #
    def display_image(self, img, canvas):
        if len(img.shape) == 2:
            display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw > 1 and ch > 1:
            h, w = display_img.shape[:2]
            scale = min(cw / w, ch / h) * 0.95
            if scale < 1:
                nw, nh = int(w * scale), int(h * scale)
                display_img = cv2.resize(display_img, (nw, nh), interpolation=cv2.INTER_AREA)

        photo = ImageTk.PhotoImage(Image.fromarray(display_img))
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor=tk.CENTER)
        canvas.image = photo


if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationViewer(root)
    root.mainloop()