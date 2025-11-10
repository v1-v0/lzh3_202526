#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive Bacteria Segmentation Parameter Tuner
- Real-time preview of segmentation results
- Adjust all parameters with sliders
"""

import cv2
import numpy as np
from pathlib import Path
from scipy import ndimage
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from typing import cast, List, Tuple


class SegmentationViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Bacteria Segmentation Parameter Tuner")
        self.root.geometry("1400x900")

        self.original_image = None

        # Default parameters
        self.params = {
            'use_otsu': tk.BooleanVar(value=False),
            'manual_threshold': tk.IntVar(value=60),
            'enable_clahe': tk.BooleanVar(value=False),
            'clahe_clip': tk.DoubleVar(value=2.0),
            'clahe_tile': tk.IntVar(value=8),
            'open_kernel': tk.IntVar(value=1),
            'close_kernel': tk.IntVar(value=5),
            'open_iter': tk.IntVar(value=1),
            'close_iter': tk.IntVar(value=1),
            'min_area': tk.IntVar(value=50),
            'watershed_dilate': tk.IntVar(value=20),
        }

        self.setup_ui()

    # --------------------------------------------------------------------- #
    # UI construction
    # --------------------------------------------------------------------- #
    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ----- left panel (controls) -----
        control_frame = ttk.Frame(main_frame, width=350)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # load button
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(btn_frame, text="Load Brightfield Image",
                   command=self.load_image).pack(fill=tk.X)

        # scrollable parameter area
        canvas = tk.Canvas(control_frame)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # build each control group
        self.create_threshold_controls(scrollable_frame)
        self.create_clahe_controls(scrollable_frame)
        self.create_morphology_controls(scrollable_frame)
        self.create_watershed_controls(scrollable_frame)

        # ----- right panel (image tabs) -----
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(image_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_original   = ttk.Frame(self.notebook)
        self.tab_enhanced   = ttk.Frame(self.notebook)
        self.tab_threshold  = ttk.Frame(self.notebook)
        self.tab_morphology = ttk.Frame(self.notebook)
        self.tab_contours   = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_original,   text="Original")
        self.notebook.add(self.tab_enhanced,   text="CLAHE Enhanced")
        self.notebook.add(self.tab_threshold,  text="Threshold")
        self.notebook.add(self.tab_morphology, text="Morphology")
        self.notebook.add(self.tab_contours,   text="Final Contours")

        # canvases
        self.canvas_original   = tk.Canvas(self.tab_original,   bg='gray')
        self.canvas_enhanced   = tk.Canvas(self.tab_enhanced,   bg='gray')
        self.canvas_threshold  = tk.Canvas(self.tab_threshold,  bg='gray')
        self.canvas_morphology = tk.Canvas(self.tab_morphology, bg='gray')
        self.canvas_contours   = tk.Canvas(self.tab_contours,   bg='gray')

        for c in (self.canvas_original, self.canvas_enhanced,
                  self.canvas_threshold, self.canvas_morphology,
                  self.canvas_contours):
            c.pack(fill=tk.BOTH, expand=True)

        # status bar
        self.status_var = tk.StringVar(value="Load an image to begin")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # --------------------------------------------------------------------- #
    # Control groups
    # --------------------------------------------------------------------- #
    def create_threshold_controls(self, parent):
        f = ttk.LabelFrame(parent, text="Thresholding", padding=10)
        f.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(f, text="Use Otsu (Auto)",
                        variable=self.params['use_otsu'],
                        command=self.update_preview).pack(anchor=tk.W)

        ttk.Label(f, text="Manual Threshold (0-255):").pack(anchor=tk.W, pady=(5, 0))
        ttk.Scale(f, from_=0, to=255, variable=self.params['manual_threshold'],
                  command=lambda _: self.update_preview(),
                  orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(f, textvariable=self.params['manual_threshold']).pack(anchor=tk.W)

    def create_clahe_controls(self, parent):
        f = ttk.LabelFrame(parent, text="CLAHE Enhancement", padding=10)
        f.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(f, text="Enable CLAHE",
                        variable=self.params['enable_clahe'],
                        command=self.update_preview).pack(anchor=tk.W)

        ttk.Label(f, text="Clip Limit (1.0-10.0):").pack(anchor=tk.W, pady=(5, 0))
        ttk.Scale(f, from_=1.0, to=10.0, variable=self.params['clahe_clip'],
                  command=lambda _: self.update_preview(),
                  orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(f, textvariable=self.params['clahe_clip']).pack(anchor=tk.W)

        ttk.Label(f, text="Tile Size (4-32):").pack(anchor=tk.W, pady=(5, 0))
        ttk.Scale(f, from_=4, to=32, variable=self.params['clahe_tile'],
                  command=lambda _: self.update_preview(),
                  orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(f, textvariable=self.params['clahe_tile']).pack(anchor=tk.W)

    def create_morphology_controls(self, parent):
        f = ttk.LabelFrame(parent, text="Morphological Operations", padding=10)
        f.pack(fill=tk.X, pady=5)

        ttk.Label(f, text="Opening Kernel (1-15, odd):").pack(anchor=tk.W)
        ttk.Scale(f, from_=1, to=15, variable=self.params['open_kernel'],
                  command=lambda _: self.update_preview(),
                  orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(f, textvariable=self.params['open_kernel']).pack(anchor=tk.W)

        ttk.Label(f, text="Closing Kernel (1-15, odd):").pack(anchor=tk.W, pady=(5, 0))
        ttk.Scale(f, from_=1, to=15, variable=self.params['close_kernel'],
                  command=lambda _: self.update_preview(),
                  orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(f, textvariable=self.params['close_kernel']).pack(anchor=tk.W)

        ttk.Label(f, text="Opening Iterations (1-5):").pack(anchor=tk.W, pady=(5, 0))
        ttk.Scale(f, from_=1, to=5, variable=self.params['open_iter'],
                  command=lambda _: self.update_preview(),
                  orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(f, textvariable=self.params['open_iter']).pack(anchor=tk.W)

        ttk.Label(f, text="Closing Iterations (1-5):").pack(anchor=tk.W, pady=(5, 0))
        ttk.Scale(f, from_=1, to=5, variable=self.params['close_iter'],
                  command=lambda _: self.update_preview(),
                  orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(f, textvariable=self.params['close_iter']).pack(anchor=tk.W)

    def create_watershed_controls(self, parent):
        f = ttk.LabelFrame(parent, text="Watershed & Filtering", padding=10)
        f.pack(fill=tk.X, pady=5)

        ttk.Label(f, text="Minimum Area (10-500):").pack(anchor=tk.W)
        ttk.Scale(f, from_=10, to=500, variable=self.params['min_area'],
                  command=lambda _: self.update_preview(),
                  orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(f, textvariable=self.params['min_area']).pack(anchor=tk.W)

        ttk.Label(f, text="Watershed Threshold % (1-20):").pack(anchor=tk.W, pady=(5, 0))
        ttk.Scale(f, from_=1, to=20, variable=self.params['watershed_dilate'],
                  command=lambda _: self.update_preview(),
                  orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(f, textvariable=self.params['watershed_dilate']).pack(anchor=tk.W)

    # --------------------------------------------------------------------- #
    # Image loading
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
    # Core segmentation
    # --------------------------------------------------------------------- #
    def segment_bacteria(self, gray_bf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        # ---- CLAHE ----
        if self.params['enable_clahe'].get():
            bf8 = cv2.convertScaleAbs(gray_bf)
            clahe = cv2.createCLAHE(
                clipLimit=self.params['clahe_clip'].get(),
                tileGridSize=(self.params['clahe_tile'].get(), self.params['clahe_tile'].get())
            )
            enhanced = clahe.apply(bf8)
        else:
            enhanced = cv2.convertScaleAbs(gray_bf)

        # ---- Thresholding ----
        if self.params['use_otsu'].get():
            _, thresh = cv2.threshold(enhanced, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(enhanced, self.params['manual_threshold'].get(),
                                      255, cv2.THRESH_BINARY_INV)

        # ---- Morphology ----
        open_k_size = self.params['open_kernel'].get()
        if open_k_size % 2 == 0:
            open_k_size += 1
        close_k_size = self.params['close_kernel'].get()
        if close_k_size % 2 == 0:
            close_k_size += 1

        open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k_size, open_k_size))
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k_size, close_k_size))

        opened  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  open_k,
                                   iterations=self.params['open_iter'].get())
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_k,
                                   iterations=self.params['close_iter'].get())

        # ---- Watershed ----
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

        # ---- Contours (Pylance-safe) ----
        contour_mask = (markers > 1).astype(np.uint8) * 255
        res = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # res[-2] -> contours for both OpenCV 3.x and 4.x; cast for the type checker
        contours = cast(List[np.ndarray], res[-2])

        # ---- Area filtering ----
        bacteria: List[np.ndarray] = []
        min_area = self.params['min_area'].get()
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                bacteria.append(cnt)

        return enhanced, thresh, cleaned, bacteria

    # --------------------------------------------------------------------- #
    # UI update
    # --------------------------------------------------------------------- #
    def update_preview(self):
        if self.original_image is None:
            return

        try:
            enhanced, thresh, cleaned, bacteria = self.segment_bacteria(self.original_image)

            self.status_var.set(f"Detected {len(bacteria)} bacteria")

            self.display_image(self.original_image, self.canvas_original)
            self.display_image(enhanced,          self.canvas_enhanced)
            self.display_image(thresh,            self.canvas_threshold)
            self.display_image(cleaned,           self.canvas_morphology)

            contour_img = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
            for cnt in bacteria:
                cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)
            self.display_image(contour_img, self.canvas_contours)

        except Exception as e:
            self.status_var.set(f"Error: {e}")

    # --------------------------------------------------------------------- #
    # Image display helper
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
            nw, nh = int(w * scale), int(h * scale)
            display_img = cv2.resize(display_img, (nw, nh), interpolation=cv2.INTER_AREA)

        photo = ImageTk.PhotoImage(Image.fromarray(display_img))
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor=tk.CENTER)
        canvas.image = photo   # keep reference


if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationViewer(root)
    root.mainloop()