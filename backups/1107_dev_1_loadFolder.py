#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive Bacteria Segmentation Parameter Tuner
- Real-time preview
- Collapsible Config and Measurement panels
- Navigation panel with Load, Reset, Exit, Previous/Next buttons
- Click Original tab → probe + measurement
- Ctrl+Click → auto-tune
- File name in title
- Exit button with proper cleanup
- Previous/Next navigation through source folder (FIXED for subfolders + AppleDouble files)
- Brightfield (_ch00.tif) and Fluorescence (_ch01.tif) pair support
- Fluorescence displayed in RED color with enhanced visibility
- Auto-loads first image on startup (disable by default)
- Configurable arrows and labels on Final Contours view with smart positioning
- SMART POSITIONING: Finds most empty direction, avoids borders and other objects
- Statistics list showing BF area, fluorescence intensity, and combined overlay view
- Compact statistics list below configuration panel
- Navigate through individual bacteria with arrow buttons (merged into statistics panel)
- BF-Fluor Overlay uses original brightfield as background
- ALL INDICES NOW USE VIEWING INDEX (1, 2, 3...) - NO GAPS
- VIEWING BACTERIUM INFO MERGED INTO QUICK STATISTICS
"""

import cv2
import numpy as np
from pathlib import Path
from scipy import ndimage
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
from typing import cast, List, Tuple, Dict, Optional


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


# --------------------------------------------------------------------- #
# Collapsible Frame class
# --------------------------------------------------------------------- #
class CollapsibleFrame(ttk.Frame):
    def __init__(self, parent, title="", **kwargs):
        super().__init__(parent, **kwargs)
        
        self.is_collapsed = False
        
        # Header frame with toggle button
        self.header = ttk.Frame(self, relief=tk.RAISED, borderwidth=1)
        self.header.pack(fill=tk.X, padx=2, pady=2)
        
        # Toggle button (▼ when expanded, ▶ when collapsed)
        self.toggle_btn = ttk.Button(
            self.header, 
            text="▼", 
            width=3,
            command=self.toggle
        )
        self.toggle_btn.pack(side=tk.LEFT, padx=2)
        
        # Title label
        self.title_label = ttk.Label(
            self.header, 
            text=title, 
            font=("Segoe UI", 10, "bold")
        )
        self.title_label.pack(side=tk.LEFT, padx=5)
        
        # Content frame
        self.content = ttk.Frame(self, padding=10)
        self.content.pack(fill=tk.BOTH, expand=True)
        
    def toggle(self):
        """Toggle between collapsed and expanded states."""
        if self.is_collapsed:
            self.expand()
        else:
            self.collapse()
    
    def collapse(self):
        """Collapse the frame to show only the header."""
        self.content.pack_forget()
        self.toggle_btn.config(text="▶")
        self.is_collapsed = True
    
    def expand(self):
        """Expand the frame to show the content."""
        self.content.pack(fill=tk.BOTH, expand=True)
        self.toggle_btn.config(text="▼")
        self.is_collapsed = False
    
    def get_content_frame(self):
        """Return the content frame for adding widgets."""
        return self.content


class SegmentationViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Bacteria Segmentation Tuner")
        self.root.geometry("1500x950")
        
        # Add window close protocol
        self.root.protocol("WM_DELETE_WINDOW", self.exit_application)

        self.original_image: Optional[np.ndarray] = None
        self.fluorescence_image: Optional[np.ndarray] = None
        self.current_contours: List[np.ndarray] = []
        self.probe_point: Optional[Tuple[int, int]] = None
        self.probe_canvas_ids: List[int] = []
        self.current_file: Optional[Path] = None
        
        # Statistics data
        self.bacteria_stats: List[Dict] = []
        self.current_bacteria_index: int = -1  # For navigating through bacteria
        
        # Navigation
        self.source_dir = Path("source/")
        self.image_files: List[Path] = []
        self.current_index: int = -1

        # Default parameter values
        self.default_params = {
            'use_otsu': False,
            'manual_threshold': 50,
            'enable_clahe': True,
            'clahe_clip': 5.0,
            'clahe_tile': 16,
            'open_kernel': 3,
            'close_kernel': 5,
            'open_iter': 3,
            'close_iter': 2,
            'min_area': 50,
            'watershed_dilate': 15,
            'fluor_brightness': 2.0,
            'fluor_gamma': 0.5,
            'show_labels': True,
            'label_font_size': 20,
            'arrow_length': 60,
            'label_offset': 15,
            'min_fluor_per_area': 10.0,
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
            'fluor_brightness': tk.DoubleVar(value=self.default_params['fluor_brightness']),
            'fluor_gamma': tk.DoubleVar(value=self.default_params['fluor_gamma']),
            'show_labels': tk.BooleanVar(value=self.default_params['show_labels']),
            'label_font_size': tk.IntVar(value=self.default_params['label_font_size']),
            'arrow_length': tk.IntVar(value=self.default_params['arrow_length']),
            'label_offset': tk.IntVar(value=self.default_params['label_offset']),
            'min_fluor_per_area': tk.DoubleVar(value=self.default_params['min_fluor_per_area']),
        }

        self.entries: Dict[str, tk.Entry] = {}
        self.progressbars: Dict[str, ttk.Progressbar] = {}

        # Measurement labels
        self.measure_labels: Dict[str, ttk.Label] = {}
        
        # Compact statistics labels
        self.compact_stats_labels: Dict[str, ttk.Label] = {}

        self.setup_ui()
        self.root.bind("<Configure>", lambda e: self.root.after_idle(self.update_preview))
        
        # Scan for images on startup
        self.scan_source_folder()
        
        # AUTO-LOAD FIRST IMAGE ON STARTUP
        #self.root.after(100, self.load_first_image_on_startup)

    def load_first_image_on_startup(self):
        """Automatically load the first image when the application starts."""
        if self.image_files:
            print("\n=== Auto-loading first image on startup ===")
            self.load_image_by_index(0)
        else:
            self.status_var.set("No images found in source folder. Please load an image manually.")

    # --------------------------------------------------------------------- #
    # Image navigation methods
    # --------------------------------------------------------------------- #
    def is_valid_image_file(self, filepath: Path) -> bool:
        """
        Check if a file is a valid image file (not an AppleDouble or hidden file).
        
        Returns True if:
        - Filename doesn't start with '._' (AppleDouble files)
        - Filename doesn't start with '.' (hidden files)
        - File ends with _ch00.tif (brightfield)
        """
        filename = filepath.name
        
        # Filter out AppleDouble files (._filename)
        if filename.startswith('._'):
            return False
        
        # Filter out hidden files (.filename)
        if filename.startswith('.'):
            return False
        
        # Must be a brightfield file
        if not filename.endswith('_ch00.tif'):
            return False
        
        return True
    
    def scan_source_folder(self):
        """Scan the source folder for all brightfield image files (_ch00.tif) recursively."""
        self.image_files = []
        if self.source_dir.exists():
            # Find all _ch00.tif files recursively (brightfield only)
            all_files = list(self.source_dir.glob('**/*_ch00.tif'))
            
            # Filter out AppleDouble and hidden files
            valid_files = [f for f in all_files if self.is_valid_image_file(f)]
            
            # Sort by string representation of path for consistent ordering
            self.image_files = sorted(valid_files, key=lambda p: str(p))
        
        if self.image_files:
            self.status_var.set(f"Found {len(self.image_files)} brightfield images in source folder")
            # Debug: print all files to verify order
            print(f"\n=== Found {len(self.image_files)} valid images ===")
            for i, f in enumerate(self.image_files):
                print(f"  [{i:3d}] {f.relative_to(self.source_dir)}")
        else:
            self.status_var.set("No brightfield images (_ch00.tif) found in source folder")
    
    def get_fluorescence_path(self, brightfield_path: Path) -> Optional[Path]:
        """Get the corresponding fluorescence image path for a brightfield image."""
        if not brightfield_path.name.endswith('_ch00.tif'):
            return None
        
        fluor_path = brightfield_path.parent / brightfield_path.name.replace('_ch00.tif', '_ch01.tif')
        
        # Check if it's a valid file (not AppleDouble)
        if fluor_path.exists() and not fluor_path.name.startswith('._'):
            return fluor_path
        return None
    
    def find_file_index(self, filepath: Path) -> int:
        """Find the index of a file in the image_files list. Returns -1 if not found."""
        try:
            # Try exact match first
            return self.image_files.index(filepath)
        except ValueError:
            # Try matching by absolute path
            abs_path = filepath.resolve()
            for i, img_file in enumerate(self.image_files):
                if img_file.resolve() == abs_path:
                    return i
            return -1
    
    def load_image_by_index(self, index: int):
        """Load an image by its index in the image_files list."""
        if not self.image_files:
            self.status_var.set("No images available")
            return
        
        # Ensure index is within bounds
        if index < 0:
            index = 0
        elif index >= len(self.image_files):
            index = len(self.image_files) - 1
        
        if 0 <= index < len(self.image_files):
            filepath = self.image_files[index]
            
            # Debug: print navigation info
            print(f"\n=== Navigating to index {index}/{len(self.image_files)-1} ===")
            print(f"  Loading: {filepath.relative_to(self.source_dir)}")
            
            # Update current index BEFORE loading to ensure it's set correctly
            self.current_index = index
            
            # Load the image
            self.load_image_from_path(filepath)
    
    def load_previous_image(self):
        """Load the previous image in the list."""
        if not self.image_files:
            self.status_var.set("No images available")
            return
        
        print(f"\n=== Previous Button Clicked ===")
        print(f"  Current index: {self.current_index}")
        print(f"  Total images: {len(self.image_files)}")
        
        if self.current_index > 0:
            new_index = self.current_index - 1
            print(f"  Moving to index: {new_index}")
            self.load_image_by_index(new_index)
        else:
            self.status_var.set("Already at first image")
            print(f"  Already at first image")
    
    def load_next_image(self):
        """Load the next image in the list."""
        if not self.image_files:
            self.status_var.set("No images available")
            return
        
        print(f"\n=== Next Button Clicked ===")
        print(f"  Current index: {self.current_index}")
        print(f"  Total images: {len(self.image_files)}")
        
        if self.current_index < len(self.image_files) - 1:
            new_index = self.current_index + 1
            print(f"  Moving to index: {new_index}")
            self.load_image_by_index(new_index)
        else:
            self.status_var.set("Already at last image")
            print(f"  Already at last image")
    
    def update_navigation_buttons(self):
        """Enable/disable navigation buttons based on current position."""
        if not hasattr(self, 'prev_btn') or not hasattr(self, 'next_btn'):
            return
            
        has_images = len(self.image_files) > 0
        
        # Update button states
        if has_images and self.current_index >= 0:
            prev_enabled = self.current_index > 0
            next_enabled = self.current_index < len(self.image_files) - 1
            
            self.prev_btn.config(state=tk.NORMAL if prev_enabled else tk.DISABLED)
            self.next_btn.config(state=tk.NORMAL if next_enabled else tk.DISABLED)
            
            self.index_label.config(text=f"{self.current_index + 1}/{len(self.image_files)}")
            
            # Debug output
            print(f"\n=== Navigation State Updated ===")
            print(f"  Index: {self.current_index + 1}/{len(self.image_files)}")
            print(f"  Previous button: {'enabled' if prev_enabled else 'disabled'}")
            print(f"  Next button: {'enabled' if next_enabled else 'disabled'}")
        else:
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.index_label.config(text="-/-")
            print(f"\n=== Navigation Disabled (no images) ===")

    # --------------------------------------------------------------------- #
    # Bacteria navigation methods
    # --------------------------------------------------------------------- #
    def navigate_to_bacteria(self, index: int):
        """Navigate to a specific bacterium and highlight it."""
        if not self.bacteria_stats or index < 0 or index >= len(self.bacteria_stats):
            return
        
        if self.original_image is None:
            return
        
        self.current_bacteria_index = index
        stat = self.bacteria_stats[index]
        
        # Get centroid of the bacterium
        M = cv2.moments(stat['contour'])
        if M["m00"] == 0:
            return
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Update probe point to this bacterium
        self.probe_point = (cx, cy)
        
        # Update measurement panel
        pixel_value = int(self.original_image[cy, cx])
        self.update_measurement_panel(cx, cy, pixel_value)
        
        # Update compact statistics display with current bacteria info
        self.update_compact_statistics()
        
        # Refresh the preview to show the highlighted bacterium
        self.update_preview()
        
        # Update status to show which bacterium is selected (viewing index only)
        self.update_bacterium_status(index)
    
    def navigate_previous_bacteria(self):
        """Navigate to the previous bacterium."""
        if not self.bacteria_stats:
            return
        
        if self.current_bacteria_index <= 0:
            # Wrap around to last
            new_index = len(self.bacteria_stats) - 1
        else:
            new_index = self.current_bacteria_index - 1
        
        self.navigate_to_bacteria(new_index)
    
    def navigate_next_bacteria(self):
        """Navigate to the next bacterium."""
        if not self.bacteria_stats:
            return
        
        if self.current_bacteria_index >= len(self.bacteria_stats) - 1:
            # Wrap around to first
            new_index = 0
        else:
            new_index = self.current_bacteria_index + 1
        
        self.navigate_to_bacteria(new_index)
    
    def update_bacteria_navigation_buttons(self):
        """Enable/disable bacteria navigation buttons based on available bacteria."""
        if not hasattr(self, 'bacteria_prev_btn') or not hasattr(self, 'bacteria_next_btn'):
            return
        
        has_bacteria = len(self.bacteria_stats) > 0
        
        if has_bacteria:
            self.bacteria_prev_btn.config(state=tk.NORMAL)
            self.bacteria_next_btn.config(state=tk.NORMAL)
        else:
            self.bacteria_prev_btn.config(state=tk.DISABLED)
            self.bacteria_next_btn.config(state=tk.DISABLED)

    # --------------------------------------------------------------------- #
    # Helper method for status updates - REDUCES CODE DUPLICATION
    # --------------------------------------------------------------------- #
    def update_bacterium_status(self, index: int):
        """Update status bar with current bacterium info."""
        if 0 <= index < len(self.bacteria_stats):
            stat = self.bacteria_stats[index]
            self.status_var.set(
                f"Viewing bacterium #{index + 1}/{len(self.bacteria_stats)} - "
                f"BF: {stat['bf_area']:.1f} px² | F/A: {stat['fluor_per_area']:.3f}"
            )

    # --------------------------------------------------------------------- #
    # UI construction
    # --------------------------------------------------------------------- #
    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Left panel: navigation + measurement + config panels
        left_panel = ttk.Frame(main_frame, width=420)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_panel.pack_propagate(False)

        # === NAVIGATION PANEL (TOP) ===
        nav_panel = ttk.LabelFrame(left_panel, text=" Navigation ", padding=10)
        nav_panel.pack(fill=tk.X, pady=(0, 10))

        # Row 1: Load, Reset, Exit
        btn_row1 = ttk.Frame(nav_panel)
        btn_row1.pack(fill=tk.X, pady=(0, 8))

        load_btn = ttk.Button(btn_row1, text="Load Image", command=self.load_image)
        load_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ToolTip(load_btn, "Select a brightfield TIFF image (_ch00.tif).")

        reset_btn = ttk.Button(btn_row1, text="Reset", command=self.reset_to_defaults)
        reset_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 2))
        ToolTip(reset_btn, "Restore all parameters to default values.")

        exit_btn = ttk.Button(btn_row1, text="Exit", command=self.exit_application)
        exit_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        ToolTip(exit_btn, "Close the application.")

        # Row 2: Previous, index, Next
        btn_row2 = ttk.Frame(nav_panel)
        btn_row2.pack(fill=tk.X, pady=(0, 0))
        btn_row2.grid_columnconfigure(0, weight=1)
        btn_row2.grid_columnconfigure(2, weight=1)

        self.prev_btn = ttk.Button(btn_row2, text="◀ Previous", command=self.load_previous_image, state=tk.DISABLED)
        self.prev_btn.grid(row=0, column=0, sticky='ew')
        ToolTip(self.prev_btn, "Load previous image in source folder")

        self.index_label = ttk.Label(btn_row2, text="-/-", anchor=tk.CENTER)
        self.index_label.grid(row=0, column=1, sticky='ew')

        self.next_btn = ttk.Button(btn_row2, text="Next ▶", command=self.load_next_image, state=tk.DISABLED)
        self.next_btn.grid(row=0, column=2, sticky='ew')
        ToolTip(self.next_btn, "Load next image in source folder")

        # === COLLAPSIBLE MEASUREMENT PANEL (ABOVE CONFIG) ===
        self.measure_panel = CollapsibleFrame(left_panel, title="Measurement on Click")
        self.measure_panel.pack(fill=tk.X, pady=(0, 10))
        
        measure_content = self.measure_panel.get_content_frame()

        labels = [
            ("pixel_coord", "Pixel: -, -"),
            ("pixel_value", "Value: -"),
            ("inside_contour", "Inside Contour: -"),
            ("contour_area", "Contour Area: - px²"),
        ]
        for key, text in labels:
            row = ttk.Frame(measure_content)
            row.pack(fill=tk.X, pady=2)
            label = ttk.Label(row, text=text, font=("Consolas", 10), foreground="#2c3e50")
            label.pack(anchor=tk.W)
            self.measure_labels[key] = label

        # === COLLAPSIBLE CONFIG PANEL (BELOW MEASUREMENT) ===
        self.config_panel = CollapsibleFrame(left_panel, title="Configuration Parameters")
        self.config_panel.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        config_content = self.config_panel.get_content_frame()

        # Scrollable config
        cfg_canvas = tk.Canvas(config_content, height=380)
        cfg_scrollbar = ttk.Scrollbar(config_content, orient="vertical", command=cfg_canvas.yview)
        scrollable_cfg = ttk.Frame(cfg_canvas)

        scrollable_cfg.bind(
            "<Configure>",
            lambda e: cfg_canvas.configure(scrollregion=cfg_canvas.bbox("all"))
        )
        cfg_canvas.create_window((0, 0), window=scrollable_cfg, anchor="nw")
        cfg_canvas.configure(yscrollcommand=cfg_scrollbar.set)

        cfg_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cfg_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Build controls
        self.create_threshold_controls(scrollable_cfg)
        self.create_clahe_controls(scrollable_cfg)
        self.create_morphology_controls(scrollable_cfg)
        self.create_watershed_controls(scrollable_cfg)
        self.create_fluorescence_controls(scrollable_cfg)
        self.create_label_controls(scrollable_cfg)

        # === COMPACT STATISTICS PANEL (BELOW CONFIG) WITH BACTERIA NAVIGATION ===
        self.compact_stats_panel = ttk.LabelFrame(left_panel, text=" Quick Statistics ", padding=8)
        self.compact_stats_panel.pack(fill=tk.X, pady=(0, 0))
        
        # Statistics labels - FIXED: Added separate viewing info line
        compact_stats = [
            ("bacteria_count", "Bacteria: -"),
            ("current_viewing", ""),  # NEW: Dedicated line for viewing info
            ("avg_bf_area", "Avg BF Area: - px²"),
            ("avg_fluor_area", "Avg Fluor/Area: -"),
        ]
        for key, text in compact_stats:
            row = ttk.Frame(self.compact_stats_panel)
            row.pack(fill=tk.X, pady=1)
            label = ttk.Label(row, text=text, font=("Consolas", 9), foreground="#34495e")
            label.pack(anchor=tk.W)
            self.compact_stats_labels[key] = label
        
        # Hide the "current_viewing" label initially (will be shown when needed)
        self.compact_stats_labels["current_viewing"].pack_forget()
        
        # Separator
        ttk.Separator(self.compact_stats_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Bacteria navigation controls (merged into statistics panel)
        bacteria_nav_frame = ttk.Frame(self.compact_stats_panel)
        bacteria_nav_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(bacteria_nav_frame, text="Navigate:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        
        self.bacteria_prev_btn = ttk.Button(
            bacteria_nav_frame, 
            text="◀", 
            width=3,
            command=self.navigate_previous_bacteria, 
            state=tk.DISABLED
        )
        self.bacteria_prev_btn.pack(side=tk.LEFT, padx=(0, 2))
        ToolTip(self.bacteria_prev_btn, "Previous bacterium")
        
        self.bacteria_next_btn = ttk.Button(
            bacteria_nav_frame, 
            text="▶", 
            width=3,
            command=self.navigate_next_bacteria, 
            state=tk.DISABLED
        )
        self.bacteria_next_btn.pack(side=tk.LEFT, padx=(2, 0))
        ToolTip(self.bacteria_next_btn, "Next bacterium")

        # Right panel: image tabs
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(image_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.tab_original = ttk.Frame(self.notebook)
        self.tab_fluorescence = ttk.Frame(self.notebook)
        self.tab_enhanced = ttk.Frame(self.notebook)
        self.tab_threshold = ttk.Frame(self.notebook)
        self.tab_morphology = ttk.Frame(self.notebook)
        self.tab_contours = ttk.Frame(self.notebook)
        self.tab_overlay = ttk.Frame(self.notebook)
        self.tab_statistics = ttk.Frame(self.notebook)

        tabs = [
            ("Original (BF)", self.tab_original),
            ("Fluorescence", self.tab_fluorescence),
            ("CLAHE Enhanced", self.tab_enhanced),
            ("Threshold", self.tab_threshold),
            ("Morphology", self.tab_morphology),
            ("Final Contours", self.tab_contours),
            ("BF+Fluor Overlay", self.tab_overlay),
            ("Statistics List", self.tab_statistics),
        ]
        for name, tab in tabs:
            self.notebook.add(tab, text=name)

        self.canvas_original = tk.Canvas(self.tab_original, bg='#f8f9fa', highlightthickness=0)
        self.canvas_fluorescence = tk.Canvas(self.tab_fluorescence, bg='#000000', highlightthickness=0)
        self.canvas_enhanced = tk.Canvas(self.tab_enhanced, bg='#f8f9fa', highlightthickness=0)
        self.canvas_threshold = tk.Canvas(self.tab_threshold, bg='#f8f9fa', highlightthickness=0)
        self.canvas_morphology = tk.Canvas(self.tab_morphology, bg='#f8f9fa', highlightthickness=0)
        self.canvas_contours = tk.Canvas(self.tab_contours, bg='#f8f9fa', highlightthickness=0)
        self.canvas_overlay = tk.Canvas(self.tab_overlay, bg='#f8f9fa', highlightthickness=0)

        for canvas in [self.canvas_original, self.canvas_fluorescence, self.canvas_enhanced,
                       self.canvas_threshold, self.canvas_morphology,
                       self.canvas_contours, self.canvas_overlay]:
            canvas.pack(fill=tk.BOTH, expand=True)

        # Setup statistics table
        self.setup_statistics_table()

        # Bind click only to Original tab
        self.canvas_original.bind("<Button-1>", self.on_canvas_click)
        self.canvas_original.bind("<Button-3>", self.clear_probe)

        # Status bar
        self.status_var = tk.StringVar(value="Load an image. Click to measure, Ctrl+Click to auto-tune.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # --------------------------------------------------------------------- #
    # Update compact statistics panel - COMPLETE FIX
    # --------------------------------------------------------------------- #
    def update_compact_statistics(self):
        """Update the compact statistics panel with overview data and viewing info."""
        if self.bacteria_stats:
            # Show overview statistics
            count = len(self.bacteria_stats)
            avg_bf = np.mean([s['bf_area'] for s in self.bacteria_stats])
            avg_fluor_area = np.mean([s['fluor_per_area'] for s in self.bacteria_stats])
            
            # Always show total count in green
            self.compact_stats_labels["bacteria_count"].config(
                text=f"Bacteria: {count}",
                foreground="#27ae60"  # Green color
            )
            
            # Show which bacterium is being viewed (if any)
            if self.current_bacteria_index >= 0 and self.current_bacteria_index < len(self.bacteria_stats):
                stat = self.bacteria_stats[self.current_bacteria_index]
                viewing_text = (f"→ Viewing #{self.current_bacteria_index + 1} | "
                              f"BF: {stat['bf_area']:.1f} px² | "
                              f"F/A: {stat['fluor_per_area']:.3f}")
                self.compact_stats_labels["current_viewing"].config(
                    text=viewing_text,
                    foreground="#e67e22"  # Orange color for currently viewed
                )
                # Make sure it's visible
                if not self.compact_stats_labels["current_viewing"].winfo_ismapped():
                    self.compact_stats_labels["current_viewing"].pack(fill=tk.X, pady=1)
            else:
                # Hide the viewing info when no bacterium is selected
                self.compact_stats_labels["current_viewing"].pack_forget()
            
            self.compact_stats_labels["avg_bf_area"].config(text=f"Avg BF Area: {avg_bf:.1f} px²")
            self.compact_stats_labels["avg_fluor_area"].config(text=f"Avg Fluor/Area: {avg_fluor_area:.3f}")
        else:
            self.compact_stats_labels["bacteria_count"].config(text="Bacteria: -", foreground="#34495e")
            self.compact_stats_labels["current_viewing"].pack_forget()
            self.compact_stats_labels["avg_bf_area"].config(text="Avg BF Area: - px²")
            self.compact_stats_labels["avg_fluor_area"].config(text="Avg Fluor/Area: -")
        
        # Update bacteria navigation buttons
        self.update_bacteria_navigation_buttons()

    # --------------------------------------------------------------------- #
    # Setup statistics table
    # --------------------------------------------------------------------- #
    def setup_statistics_table(self):
        """Create the statistics table with scrollbars."""
        # Create a frame for the table
        table_frame = ttk.Frame(self.tab_statistics)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create Treeview with scrollbars
        columns = ("Index", "BF Area (px²)", "Fluor Mean", "Fluor Total", "Fluor/Area")
        
        self.stats_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=20)
        
        # Define column headings and widths
        self.stats_tree.heading("Index", text="Index")
        self.stats_tree.heading("BF Area (px²)", text="BF Area (px²)")
        self.stats_tree.heading("Fluor Mean", text="Fluor Mean")
        self.stats_tree.heading("Fluor Total", text="Fluor Total")
        self.stats_tree.heading("Fluor/Area", text="Fluor/Area")
        
        self.stats_tree.column("Index", width=60, anchor=tk.CENTER)
        self.stats_tree.column("BF Area (px²)", width=120, anchor=tk.CENTER)
        self.stats_tree.column("Fluor Mean", width=120, anchor=tk.CENTER)
        self.stats_tree.column("Fluor Total", width=120, anchor=tk.CENTER)
        self.stats_tree.column("Fluor/Area", width=120, anchor=tk.CENTER)
        
        # Add scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.stats_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.stats_tree.xview)
        self.stats_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout
        self.stats_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Add summary label at bottom
        self.stats_summary = ttk.Label(table_frame, text="No data", font=("Segoe UI", 10, "bold"))
        self.stats_summary.grid(row=2, column=0, columnspan=2, pady=5)

    # --------------------------------------------------------------------- #
    # Calculate statistics for all bacteria
    # --------------------------------------------------------------------- #
    def calculate_bacteria_statistics(self, contours, bf_image, fluor_image=None):
        """
        Calculate statistics for each detected bacterium.
        
        Returns:
            List of dictionaries containing statistics for each bacterium
        """
        stats = []
        
        for contour in contours:
            # Calculate brightfield area
            bf_area = cv2.contourArea(contour)
            
            # Create mask for this contour
            mask = np.zeros(bf_image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Calculate fluorescence statistics if available
            if fluor_image is not None:
                # Extract fluorescence values within the contour
                fluor_values = fluor_image[mask == 255]
                
                if len(fluor_values) > 0:
                    fluor_mean = float(np.mean(fluor_values))
                    fluor_total = float(np.sum(fluor_values))
                    fluor_per_area = fluor_total / bf_area if bf_area > 0 else 0
                else:
                    fluor_mean = 0
                    fluor_total = 0
                    fluor_per_area = 0
            else:
                fluor_mean = 0
                fluor_total = 0
                fluor_per_area = 0
            
            stats.append({
                'bf_area': bf_area,
                'fluor_mean': fluor_mean,
                'fluor_total': fluor_total,
                'fluor_per_area': fluor_per_area,
                'contour': contour
            })
        
        return stats

    # --------------------------------------------------------------------- #
    # Update statistics table
    # --------------------------------------------------------------------- #
    def update_statistics_table(self):
        """Update the statistics table with current data."""
        # Clear existing items
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        
        # Populate with new data using viewing index
        if self.bacteria_stats:
            for idx, stat in enumerate(self.bacteria_stats, 1):
                self.stats_tree.insert('', 'end', values=(
                    idx,  # Viewing index (1, 2, 3, 4...)
                    f"{stat['bf_area']:.1f}",
                    f"{stat['fluor_mean']:.2f}",
                    f"{stat['fluor_total']:.1f}",
                    f"{stat['fluor_per_area']:.3f}"
                ))
            
            # Update summary
            total_bacteria = len(self.bacteria_stats)
            avg_fluor_per_area = np.mean([s['fluor_per_area'] for s in self.bacteria_stats])
            
            summary_text = (f"Total: {total_bacteria} bacteria | "
                          f"Avg Fluor/Area: {avg_fluor_per_area:.3f}")
            self.stats_summary.config(text=summary_text)
        else:
            self.stats_summary.config(text="No bacteria detected or no fluorescence data")

    # --------------------------------------------------------------------- #
    # Create overlay image - UPDATED TO USE ORIGINAL BRIGHTFIELD
    # --------------------------------------------------------------------- #
    def create_overlay_image(self, bf_image, fluor_image, contours):
        """
        Create an overlay of brightfield and fluorescence images with contours.
        Uses original brightfield as background with red fluorescence overlay.
        
        Args:
            bf_image: Grayscale brightfield image
            fluor_image: Grayscale fluorescence image (or None)
            contours: List of contours
        
        Returns:
            RGB overlay image
        """
        # Start with original brightfield as RGB
        overlay = cv2.cvtColor(bf_image, cv2.COLOR_GRAY2RGB)
        
        if fluor_image is not None:
            # Process fluorescence with gamma and brightness
            fluor_float = fluor_image.astype(np.float32)
            gamma = self.params['fluor_gamma'].get()
            brightness = self.params['fluor_brightness'].get()
            
            if fluor_image.dtype == np.uint16:
                fluor_max = fluor_float.max()
                if fluor_max > 0:
                    fluor_float = fluor_float / fluor_max
            else:
                fluor_float = fluor_float / 255.0
            
            # Apply gamma and brightness
            fluor_processed = np.power(fluor_float, gamma) * brightness
            fluor_processed = np.clip(fluor_processed, 0, 1)
            
            # Convert to uint8
            fluor_uint8 = (fluor_processed * 255).astype(np.uint8)
            
            # Create red overlay channel
            # Add red fluorescence on top of the original brightfield
            red_channel = overlay[:, :, 2].astype(np.float32)  # Red channel in BGR
            
            # Blend: increase red where there's fluorescence
            red_channel = np.clip(red_channel + fluor_uint8.astype(np.float32), 0, 255)
            overlay[:, :, 2] = red_channel.astype(np.uint8)
        
        # Draw contours in cyan (visible against both gray and red)
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)  # Yellow/cyan contours
        
        return overlay

    # --------------------------------------------------------------------- #
    # Click handler: probe + refresh measurement
    # --------------------------------------------------------------------- #
    def on_canvas_click(self, event):
        if self.original_image is None:
            return

        cw = self.canvas_original.winfo_width()
        ch = self.canvas_original.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        if self.original_image is None:
            return
        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
        else:
            return
        scale = min(cw / w, ch / h) * 0.95
        if scale >= 1.0:
            scale = 1.0
        offset_x = (cw - int(w * scale)) // 2
        offset_y = (ch - int(h * scale)) // 2

        img_x = int((event.x - offset_x) / scale)
        img_y = int((event.y - offset_y) / scale)

        if not (0 <= img_x < w and 0 <= img_y < h):
            return

        pixel_value = int(self.original_image[img_y, img_x])
        self.probe_point = (img_x, img_y)

        # Draw crosshair
        self.clear_probe()
        cx = event.x
        cy = event.y
        self.probe_canvas_ids = [
            self.canvas_original.create_line(cx - 12, cy, cx + 12, cy, fill="red", width=3),
            self.canvas_original.create_line(cx, cy - 12, cx, cy + 12, fill="red", width=3)
        ]

        # Update measurement panel (auto-expand if collapsed)
        if self.measure_panel.is_collapsed:
            self.measure_panel.expand()
        
        self.update_measurement_panel(img_x, img_y, pixel_value)
        
        # Find which bacterium was clicked and update current_bacteria_index
        if self.bacteria_stats:
            found = False
            for index, stat in enumerate(self.bacteria_stats):
                if cv2.pointPolygonTest(stat['contour'], (img_x, img_y), False) >= 0:
                    self.current_bacteria_index = index
                    found = True
                    # Update status to show which bacterium was clicked (viewing index only)
                    self.update_bacterium_status(index)
                    break
            
            if not found:
                self.current_bacteria_index = -1
            
            self.update_bacteria_navigation_buttons()
            self.update_compact_statistics()

        # Auto-tune ONLY if Ctrl key is held
        if event.state & 0x4:  # Ctrl key modifier
            self.auto_tune_from_point(img_x, img_y, pixel_value)
            self.update_preview()
        else:
            # Just update the preview without changing parameters
            self.update_preview()

    def clear_probe(self, event=None):
        for cid in self.probe_canvas_ids:
            self.canvas_original.delete(cid)
        self.probe_canvas_ids = []
        self.probe_point = None
        self.current_bacteria_index = -1
        self.update_bacteria_navigation_buttons()
        self.reset_measurement_panel()
        # Update compact stats to show overview
        self.update_compact_statistics()
        if self.current_file:
            self.status_var.set(f"Ready. Loaded: {self.current_file.name}")

    # --------------------------------------------------------------------- #
    # Update measurement panel
    # --------------------------------------------------------------------- #
    def update_measurement_panel(self, x: int, y: int, value: int):
        self.measure_labels["pixel_coord"].config(text=f"Pixel: ({x}, {y})")
        self.measure_labels["pixel_value"].config(text=f"Value: {value}")

        inside = False
        area = 0.0
        if self.current_contours:
            for contour in self.current_contours:
                if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                    inside = True
                    area = cv2.contourArea(contour)
                    break

        self.measure_labels["inside_contour"].config(
            text=f"Inside Contour: {'Yes' if inside else 'No'}",
            foreground="#27ae60" if inside else "#e74c3c"
        )
        self.measure_labels["contour_area"].config(
            text=f"Contour Area: {int(area)} px²" if inside else "Contour Area: - px²"
        )

    def reset_measurement_panel(self):
        for key in self.measure_labels:
            default = {
                "pixel_coord": "Pixel: -, -",
                "pixel_value": "Value: -",
                "inside_contour": "Inside Contour: -",
                "contour_area": "Contour Area: - px²",
            }[key]
            self.measure_labels[key].config(text=default, foreground="black")

    # --------------------------------------------------------------------- #
    # Auto-tune
    # --------------------------------------------------------------------- #
    def auto_tune_from_point(self, x: int, y: int, pixel_value: int):
        if not self.current_contours:
            self.status_var.set("No contours found - cannot auto-tune")
            return

        for contour in self.current_contours:
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                area = cv2.contourArea(contour)
                if area < 10:
                    self.status_var.set("Contour too small - cannot auto-tune")
                    continue

                self.params['use_otsu'].set(False)
                self.params['manual_threshold'].set(max(5, pixel_value - 18))
                self.params['min_area'].set(int(area * 0.75))
                self.params['watershed_dilate'].set(14)

                for key in ['use_otsu', 'manual_threshold', 'min_area', 'watershed_dilate']:
                    if key in self.entries:
                        self.entries[key].delete(0, tk.END)
                        self.entries[key].insert(0, str(self.params[key].get()))
                        self.update_progressbar(key)

                self.status_var.set(f"✓ Auto-tuned: Threshold={pixel_value-18}, Min Area={int(area*0.75)}")
                return
        
        self.status_var.set("Click inside a contour to auto-tune")

    # --------------------------------------------------------------------- #
    # Entry + Progressbar
    # --------------------------------------------------------------------- #
    def add_entry_with_progress(self, parent, label_text, tooltip_text, var, min_val, max_val,
                                resolution=1.0, is_float=False):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=1)

        label = ttk.Label(frame, text=label_text, width=20, anchor=tk.W)
        label.pack(side=tk.LEFT)
        ToolTip(label, tooltip_text)

        var_name = next(k for k, v in self.params.items() if v == var)

        entry = ttk.Entry(frame, width=7, justify=tk.RIGHT, font=("Consolas", 10))
        entry.pack(side=tk.LEFT, padx=(2, 4))
        entry.insert(0, str(var.get()))
        entry.bind('<Return>', lambda e: self.sync_entry(var_name, entry, min_val, max_val, resolution, is_float))
        entry.bind('<FocusOut>', lambda e: self.sync_entry(var_name, entry, min_val, max_val, resolution, is_float))
        ToolTip(entry, tooltip_text)

        pb = ttk.Progressbar(frame, orient=tk.HORIZONTAL, mode='determinate', length=150)
        pb.pack(side=tk.RIGHT, padx=(0, 2))
        ToolTip(pb, tooltip_text)

        self.entries[var_name] = entry
        self.progressbars[var_name] = pb
        self.update_progressbar(var_name)
        return entry

    # --------------------------------------------------------------------- #
    # Sync entry
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
            messagebox.showwarning("Invalid Input", f"Enter a valid number for {var_name.replace('_', ' ')}.")
            entry.delete(0, tk.END)
            entry.insert(0, str(self.params[var_name].get()))

    # --------------------------------------------------------------------- #
    # Update progress bar
    # --------------------------------------------------------------------- #
    def update_progressbar(self, var_name: str) -> None:
        if var_name not in self.progressbars:
            return
            
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
            'fluor_brightness': (0.5, 5.0),
            'fluor_gamma': (0.2, 2.0),
            'label_font_size': (10, 60),
            'arrow_length': (20, 100),
            'label_offset': (5, 50),
            'min_fluor_per_area': (0.0, 255.0),
        }
        min_val, max_val = ranges[var_name]
        percent = (value - min_val) / (max_val - min_val) * 100
        pb['value'] = percent

    # --------------------------------------------------------------------- #
    # Control groups
    # --------------------------------------------------------------------- #
    def create_threshold_controls(self, parent):
        f = ttk.LabelFrame(parent, text="Thresholding", padding=10)
        f.pack(fill=tk.X, pady=4)
        cb = ttk.Checkbutton(f, text="Use Otsu (Auto)", variable=self.params['use_otsu'], command=self.update_preview)
        cb.pack(anchor=tk.W)
        ToolTip(cb, "Auto threshold using Otsu's method.")
        self.add_entry_with_progress(f, "Manual Threshold:", "0–255. Lower = more foreground.",
                                     self.params['manual_threshold'], 0, 255, resolution=1, is_float=False)

    def create_clahe_controls(self, parent):
        f = ttk.LabelFrame(parent, text="CLAHE Enhancement", padding=10)
        f.pack(fill=tk.X, pady=4)
        cb = ttk.Checkbutton(f, text="Enable CLAHE", variable=self.params['enable_clahe'], command=self.update_preview)
        cb.pack(anchor=tk.W)
        ToolTip(cb, "Improves contrast in low-light regions.")
        self.add_entry_with_progress(f, "Clip Limit:", "1.0–10.0. Higher = stronger contrast.",
                                     self.params['clahe_clip'], 1.0, 10.0, resolution=0.1, is_float=True)
        self.add_entry_with_progress(f, "Tile Size:", "4–32. Smaller = more local enhancement.",
                                     self.params['clahe_tile'], 4, 32, resolution=1, is_float=False)

    def create_morphology_controls(self, parent):
        f = ttk.LabelFrame(parent, text="Morphological Operations", padding=10)
        f.pack(fill=tk.X, pady=4)
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
        f.pack(fill=tk.X, pady=4)
        self.add_entry_with_progress(f, "Minimum Area:", "10–500 px². Ignore small objects.",
                                     self.params['min_area'], 10, 500, resolution=1, is_float=False)
        self.add_entry_with_progress(f, "Watershed Threshold %:", "1–20. Higher = more merging.",
                                     self.params['watershed_dilate'], 1, 20, resolution=1, is_float=False)
        self.add_entry_with_progress(f, "Min Fluor/Area:", "0.0–255.0. Filter low fluorescence bacteria.",
                                     self.params['min_fluor_per_area'], 0.0, 255.0, resolution=0.1, is_float=True)

    def create_fluorescence_controls(self, parent):
        """Fluorescence visualization controls"""
        f = ttk.LabelFrame(parent, text="Fluorescence Display", padding=10)
        f.pack(fill=tk.X, pady=4)
        self.add_entry_with_progress(f, "Brightness:", "0.5–5.0. Higher = brighter.",
                                     self.params['fluor_brightness'], 0.5, 5.0, resolution=0.1, is_float=True)
        self.add_entry_with_progress(f, "Gamma:", "0.2–2.0. Lower = enhance dim signals.",
                                     self.params['fluor_gamma'], 0.2, 2.0, resolution=0.1, is_float=True)

    def create_label_controls(self, parent):
        """Label and arrow controls"""
        f = ttk.LabelFrame(parent, text="Labels & Arrows", padding=10)
        f.pack(fill=tk.X, pady=4)
        
        cb = ttk.Checkbutton(f, text="Show Labels", variable=self.params['show_labels'], command=self.update_preview)
        cb.pack(anchor=tk.W)
        ToolTip(cb, "Show/hide bacteria labels on Final Contours view.")
        
        self.add_entry_with_progress(f, "Font Size:", "10–60. Label text size.",
                                     self.params['label_font_size'], 10, 60, resolution=1, is_float=False)
        self.add_entry_with_progress(f, "Arrow Length:", "20–100 px. Arrow line length.",
                                     self.params['arrow_length'], 20, 100, resolution=1, is_float=False)
        self.add_entry_with_progress(f, "Label Offset:", "5–50 px. Label distance from arrow.",
                                     self.params['label_offset'], 5, 50, resolution=1, is_float=False)

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
            self.clear_probe()
            self.update_preview()
            self.status_var.set("Parameters reset")

    # --------------------------------------------------------------------- #
    # Load image
    # --------------------------------------------------------------------- #
    def load_image(self):
        # Set default directory
        default_dir = "source/"
        
        filepath = filedialog.askopenfilename(
            title="Select Brightfield Image",
            initialdir=default_dir,
            filetypes=[("Brightfield files", "*_ch00.tif"), ("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if not filepath:
            return
        
        filepath_obj = Path(filepath)
        
        # Check if it's a valid file (not AppleDouble)
        if not self.is_valid_image_file(filepath_obj):
            messagebox.showerror("Invalid File", 
                               "Selected file appears to be a system/hidden file.\n"
                               "Please select a valid brightfield image (_ch00.tif).")
            return
        
        # Rescan to ensure we have the latest file list
        print("\n=== Rescanning source folder ===")
        self.scan_source_folder()
        
        # Find the index of the loaded file
        index = self.find_file_index(filepath_obj)
        
        if index >= 0:
            print(f"File found at index {index}")
            self.current_index = index
            self.load_image_from_path(filepath_obj)
            # Update navigation buttons AFTER loading
            self.update_navigation_buttons()
        else:
            print(f"Warning: File not in scan list, loading anyway")
            # File not in list, but load it anyway
            self.current_index = -1
            self.load_image_from_path(filepath_obj)
            self.update_navigation_buttons()
    
    def load_image_from_path(self, filepath: Path):
        """Load a brightfield image and its corresponding fluorescence image."""
        print(f"\n=== Loading image from path ===")
        print(f"  Path: {filepath}")
        
        # Verify it's a valid file
        if not self.is_valid_image_file(filepath):
            self.status_var.set(f"Error: Invalid file (system/hidden file): {filepath.name}")
            return
        
        # Load brightfield image
        img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if img is None:
            self.status_var.set(f"Error: Cannot read {filepath}")
            return
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.original_image = img
        self.current_file = filepath
        self.current_index = self.find_file_index(filepath)
        
        # Try to load corresponding fluorescence image
        fluor_path = self.get_fluorescence_path(filepath)
        if fluor_path:
            fluor_img = cv2.imread(str(fluor_path), cv2.IMREAD_UNCHANGED)
            if fluor_img is not None:
                if len(fluor_img.shape) == 3:
                    fluor_img = cv2.cvtColor(fluor_img, cv2.COLOR_BGR2GRAY)
                self.fluorescence_image = fluor_img
                self.status_var.set(f"Loaded: {self.current_file.name} + {fluor_path.name} ({img.shape[1]}x{img.shape[0]})")
            else:
                self.fluorescence_image = None
                self.status_var.set(f"Loaded: {self.current_file.name} (no fluorescence pair) ({img.shape[1]}x{img.shape[0]})")
        else:
            self.fluorescence_image = None
            self.status_var.set(f"Loaded: {self.current_file.name} (no fluorescence pair) ({img.shape[1]}x{img.shape[0]})")
        
        self.root.title(f"{self.current_file.name} - Bacteria Segmentation Tuner")
        self.clear_probe()
        
        # Update navigation buttons to reflect the new state
        self.update_navigation_buttons()
        
        # Update the preview
        self.update_preview()
        
        print(f"  Successfully loaded. Current index: {self.current_index}")

    # --------------------------------------------------------------------- #
    # Segmentation
    # --------------------------------------------------------------------- #
    def segment_bacteria(self, gray_bf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
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
            _, thresh = cv2.threshold(enhanced, self.params['manual_threshold'].get(), 255, cv2.THRESH_BINARY_INV)

        open_k_size = self.params['open_kernel'].get()
        if open_k_size % 2 == 0: open_k_size = max(1, open_k_size - 1)
        close_k_size = self.params['close_kernel'].get()
        if close_k_size % 2 == 0: close_k_size = max(1, close_k_size - 1)

        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k_size, open_k_size))
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k_size, close_k_size))

        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k, iterations=self.params['open_iter'].get())
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_k, iterations=self.params['close_iter'].get())

        distance = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(distance, self.params['watershed_dilate'].get() * distance.max() / 100.0, 255, 0)
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
        contours: List[np.ndarray] = res[-2]

        bacteria = [c for c in contours if cv2.contourArea(c) >= self.params['min_area'].get()]

        self.current_contours = bacteria
        return enhanced, thresh, cleaned, bacteria

    # --------------------------------------------------------------------- #
    # Smart label positioning with obstacle avoidance
    # --------------------------------------------------------------------- #
    def create_occupancy_map(self, img_shape, contours, margin=20):
        """
        Create a binary occupancy map showing where objects and borders are.
        
        Args:
            img_shape: (height, width) of the image
            contours: list of contours representing bacteria
            margin: safety margin around objects and borders
        
        Returns:
            occupancy_map: binary map (255 = occupied, 0 = free)
        """
        h, w = img_shape
        occupancy = np.zeros((h, w), dtype=np.uint8)
        
        # Mark all contours as occupied (with margin)
        for contour in contours:
            # Draw filled contour
            cv2.drawContours(occupancy, [contour], -1, 255, -1)
            # Dilate to add margin around bacteria
            if margin > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin*2, margin*2))
                occupancy = cv2.dilate(occupancy, kernel, iterations=1)
        
        # Mark borders as occupied
        border_margin = margin
        occupancy[:border_margin, :] = 255  # Top border
        occupancy[-border_margin:, :] = 255  # Bottom border
        occupancy[:, :border_margin] = 255  # Left border
        occupancy[:, -border_margin:] = 255  # Right border
        
        return occupancy
    
    def calculate_direction_score(self, centroid, angle, arrow_len, label_size, 
                                  occupancy_map, offset):
        """
        Calculate a score for placing a label in a given direction.
        Lower score = better placement.
        
        Args:
            centroid: (x, y) center of bacteria
            angle: angle in degrees to test
            arrow_len: length of the arrow
            label_size: (width, height) of the label
            occupancy_map: binary map showing occupied areas
            offset: additional offset from arrow tip
        
        Returns:
            score: float (lower is better, inf if invalid)
        """
        cx, cy = centroid
        h, w = occupancy_map.shape
        label_w, label_h = label_size
        
        # Calculate arrow end point
        rad = np.deg2rad(angle)
        arrow_x = int(cx + arrow_len * np.cos(rad))
        arrow_y = int(cy - arrow_len * np.sin(rad))
        
        # Calculate label position (centered on arrow tip with offset)
        label_x = int(arrow_x + offset * np.cos(rad) - label_w / 2)
        label_y = int(arrow_y - offset * np.sin(rad) - label_h / 2)
        
        # Check if label is out of bounds
        if (label_x < 0 or label_x + label_w >= w or
            label_y < 0 or label_y + label_h >= h):
            return float('inf')  # Invalid position
        
        # Calculate score based on occupancy in the label area
        label_region = occupancy_map[label_y:label_y+label_h, label_x:label_x+label_w]
        
        # Count occupied pixels in label area
        occupied_pixels = np.sum(label_region > 0)
        total_pixels = label_region.size
        
        # Prevent division by zero
        if total_pixels == 0:
            return float('inf')
        
        # Also check arrow path for obstacles
        arrow_score = 0
        num_samples = 10
        for i in range(num_samples):
            t = i / num_samples
            sample_x = int(cx + t * arrow_len * np.cos(rad))
            sample_y = int(cy - t * arrow_len * np.sin(rad))
            
            if 0 <= sample_x < w and 0 <= sample_y < h:
                if occupancy_map[sample_y, sample_x] > 0:
                    arrow_score += 10  # Penalty for arrow crossing objects
        
        # Combined score: prefer directions with less occupancy
        # and penalize arrow crossing objects
        score = (occupied_pixels / total_pixels) * 100 + arrow_score
        
        return score
    
    def find_best_label_position(self, centroid, img_shape, arrow_len, label_size, 
                                 offset, occupancy_map):
        """
        Find the best position for a label by testing multiple directions.
        
        Returns:
            (arrow_end_x, arrow_end_y, label_x, label_y, angle_deg) or None
        """
        cx, cy = centroid
        h, w = img_shape
        label_w, label_h = label_size
        
        # Test 16 directions for better coverage
        angles = [i * 22.5 for i in range(16)]
        
        best_score = float('inf')
        best_position = None
        
        for angle in angles:
            score = self.calculate_direction_score(
                centroid, angle, arrow_len, label_size, 
                occupancy_map, offset
            )
            
            if score < best_score:
                best_score = score
                
                # Calculate position for this angle
                rad = np.deg2rad(angle)
                arrow_x = int(cx + arrow_len * np.cos(rad))
                arrow_y = int(cy - arrow_len * np.sin(rad))
                label_x = int(arrow_x + offset * np.cos(rad) - label_w / 2)
                label_y = int(arrow_y - offset * np.sin(rad) - label_h / 2)
                
                # Clamp to valid range (should already be valid from score calculation)
                label_x = max(0, min(label_x, w - label_w))
                label_y = max(0, min(label_y, h - label_h))
                
                best_position = (arrow_x, arrow_y, label_x, label_y, angle)
        
        return best_position

    # --------------------------------------------------------------------- #
    # Get font with current size - SIMPLIFIED
    # --------------------------------------------------------------------- #
    def get_label_font(self):
        """Get a font with the current font size setting."""
        font_size = self.params['label_font_size'].get()
        
        # List of font paths to try (in order of preference)
        font_paths = [
            "arial.ttf",  # Windows/current directory
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS alternative
            "C:\\Windows\\Fonts\\arial.ttf",  # Windows absolute
        ]
        
        # Try each font path
        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, font_size)
            except (OSError, IOError):
                continue
        
        # Last resort: use default font
        return ImageFont.load_default()

    # --------------------------------------------------------------------- #
    # Draw labels with arrows using smart positioning
    # --------------------------------------------------------------------- #
    def draw_labels_on_contours(self, img_bgr, contours):
        """
        Draw labels with arrows on detected bacteria using intelligent positioning.
        Labels show viewing index (1, 2, 3...) not original IDs.
        """
        if not self.params['show_labels'].get() or not contours:
            return img_bgr
        
        # Convert BGR to RGB for PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # Get parameters
        arrow_len = self.params['arrow_length'].get()
        label_offset = self.params['label_offset'].get()
        font = self.get_label_font()
        
        h, w = img_bgr.shape[:2]
        occupancy_map = self.create_occupancy_map((h, w), contours, margin=20)
        
        # Draw labels for each contour using viewing index
        for stat_idx, stat in enumerate(self.bacteria_stats):
            contour = stat['contour']
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Get label text (use viewing index, 1-based)
            label_text = str(stat_idx + 1)  # Shows 1, 2, 3, 4... (viewing order)
            
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # Find best label position
            result = self.find_best_label_position(
                (cx, cy), 
                (h, w), 
                arrow_len, 
                (text_w, text_h), 
                label_offset,
                occupancy_map
            )
            
            if result is None:
                continue
            
            arrow_x, arrow_y, label_x, label_y, angle = result
            
            # Highlight the currently selected bacterium
            is_selected = (self.current_bacteria_index == stat_idx)
            arrow_color = (255, 128, 0) if is_selected else (255, 255, 0)
            arrow_width = 3 if is_selected else 2
            
            # Draw arrow line
            draw.line([(cx, cy), (arrow_x, arrow_y)], fill=arrow_color, width=arrow_width)
            
            # Draw arrow head
            head_len = 10 if is_selected else 8
            head_angle = 25
            angle_rad = np.deg2rad(angle)
            
            left_angle = angle_rad + np.deg2rad(180 - head_angle)
            left_x = int(arrow_x + head_len * np.cos(left_angle))
            left_y = int(arrow_y - head_len * np.sin(left_angle))
            draw.line([(arrow_x, arrow_y), (left_x, left_y)], fill=arrow_color, width=arrow_width)
            
            right_angle = angle_rad + np.deg2rad(180 + head_angle)
            right_x = int(arrow_x + head_len * np.cos(right_angle))
            right_y = int(arrow_y - head_len * np.sin(right_angle))
            draw.line([(arrow_x, arrow_y), (right_x, right_y)], fill=arrow_color, width=arrow_width)
            
            # Draw text background
            padding = 4
            bg_rect = [
                label_x - padding,
                label_y - padding,
                label_x + text_w + padding,
                label_y + text_h + padding
            ]
            draw.rectangle(bg_rect, fill=(0, 0, 0, 200))
            
            # Draw text
            draw.text((label_x, label_y), label_text, font=font, fill=arrow_color)
            
            # Mark this label area as occupied
            occupancy_map[label_y:label_y+text_h, label_x:label_x+text_w] = 255
        
        # Convert back to BGR
        img_rgb_array = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb_array, cv2.COLOR_RGB2BGR)
        
        return img_bgr

    # --------------------------------------------------------------------- #
    # Update preview
    # --------------------------------------------------------------------- #
    def update_preview(self) -> None:
        if self.original_image is None:
            return
        try:
            enhanced, thresh, cleaned, bacteria = self.segment_bacteria(self.original_image)
            
            # Calculate statistics for ALL bacteria
            all_bacteria_stats = self.calculate_bacteria_statistics(
                bacteria, 
                self.original_image, 
                self.fluorescence_image
            )
            
            # Filter based on min_fluor_per_area
            min_fpa = self.params['min_fluor_per_area'].get()
            if self.fluorescence_image is not None and min_fpa > 0:
                # Keep only bacteria that pass the threshold
                self.bacteria_stats = [s for s in all_bacteria_stats if s['fluor_per_area'] >= min_fpa]
                
                # Extract filtered contours (maintaining order)
                bacteria = [stat['contour'] for stat in self.bacteria_stats]
                self.current_contours = bacteria
                
                # Update status to show filtering
                total_count = len(all_bacteria_stats)
                filtered_count = len(self.bacteria_stats)
                base_msg = f"Detected {filtered_count}/{total_count} bacteria (min fluor/area: {min_fpa:.1f})"
            else:
                # No filtering
                self.bacteria_stats = all_bacteria_stats
                bacteria = [stat['contour'] for stat in self.bacteria_stats]
                self.current_contours = bacteria
                count = len(bacteria)
                base_msg = f"Detected {count} bacteria"
            
            if self.probe_point is None and self.current_file:
                fluor_status = " (with fluorescence)" if self.fluorescence_image is not None else ""
                self.status_var.set(f"{base_msg} | Loaded: {self.current_file.name}{fluor_status}")

            # Update statistics table and compact statistics
            self.update_statistics_table()
            self.update_compact_statistics()

            for key in self.progressbars:
                self.update_progressbar(key)

            self.display_image(self.original_image, self.canvas_original)
            
            # Display fluorescence image if available (in RED with enhanced visibility)
            if self.fluorescence_image is not None:
                self.display_fluorescence_image(self.fluorescence_image, self.canvas_fluorescence)
            else:
                # Clear fluorescence canvas if no image
                self.canvas_fluorescence.delete("all")
                cw = self.canvas_fluorescence.winfo_width()
                ch = self.canvas_fluorescence.winfo_height()
                if cw > 1 and ch > 1:
                    self.canvas_fluorescence.create_text(
                        cw // 2, ch // 2, 
                        text="No fluorescence image available", 
                        font=("Segoe UI", 12), 
                        fill="#666"
                    )
            
            self.display_image(enhanced, self.canvas_enhanced)
            self.display_image(thresh, self.canvas_threshold)
            self.display_image(cleaned, self.canvas_morphology)

            # Create contour image with labels (using viewing indices)
            contour_img = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_img, bacteria, -1, (0, 255, 0), 2)

            # Highlight probed contour if exists
            if self.probe_point and bacteria:
                x, y = self.probe_point
                for contour in bacteria:
                    if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                        cv2.drawContours(contour_img, [contour], -1, (0, 0, 255), 3)
                        break

            # Draw labels with smart positioning - using viewing indices
            contour_img = self.draw_labels_on_contours(contour_img, bacteria)

            self.display_image(contour_img, self.canvas_contours)
            
            # Create and display overlay image
            overlay_img = self.create_overlay_image(
                self.original_image,
                self.fluorescence_image,
                bacteria
            )
            self.display_image(overlay_img, self.canvas_overlay)
            
        except Exception as e:
            import traceback
            self.status_var.set(f"Error: {e}")
            traceback.print_exc()

    # --------------------------------------------------------------------- #
    # Display fluorescence image in RED with enhanced visibility
    # --------------------------------------------------------------------- #
    def display_fluorescence_image(self, img, canvas):
        """Display fluorescence image with RED colormap and enhanced visibility."""
        # Convert to float for processing
        img_float = img.astype(np.float32)
        
        # Apply gamma correction (lower gamma enhances dim signals)
        gamma = self.params['fluor_gamma'].get()
        if img.dtype == np.uint16:
            # For 16-bit images, normalize first
            img_max = img_float.max()
            if img_max > 0:
                img_float = img_float / img_max
        else:
            # For 8-bit images
            img_float = img_float / 255.0
        
        # Apply gamma correction
        img_gamma = np.power(img_float, gamma)
        
        # Apply brightness multiplier
        brightness = self.params['fluor_brightness'].get()
        img_enhanced = img_gamma * brightness
        
        # Clip to valid range
        img_enhanced = np.clip(img_enhanced, 0, 1)
        
        # Convert to 8-bit
        img_8bit = (img_enhanced * 255).astype(np.uint8)
        
        # Create RGB image with RED channel only
        red_img = np.zeros((img_8bit.shape[0], img_8bit.shape[1], 3), dtype=np.uint8)
        red_img[:, :, 2] = img_8bit  # Red channel (BGR format for OpenCV)
        
        # Convert to RGB for PIL
        red_img_rgb = cv2.cvtColor(red_img, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw > 1 and ch > 1:
            h, w = red_img_rgb.shape[:2]
            scale = min(cw / w, ch / h) * 0.95
            if scale < 1:
                nw, nh = int(w * scale), int(h * scale)
                red_img_rgb = cv2.resize(red_img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        # Display
        photo = ImageTk.PhotoImage(Image.fromarray(red_img_rgb))
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor=tk.CENTER)
        canvas.image = photo

    # --------------------------------------------------------------------- #
    # Display image
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

        # Re-draw crosshair (only if original image is loaded)
        if canvas == self.canvas_original and self.probe_point and self.original_image is not None:
            h, w = self.original_image.shape[:2]
            scale = min(cw / w, ch / h) * 0.95
            if scale >= 1:
                scale = 1
            offset_x = (cw - int(w * scale)) // 2
            offset_y = (ch - int(h * scale)) // 2
            img_x, img_y = self.probe_point
            cx = int(img_x * scale) + offset_x
            cy = int(img_y * scale) + offset_y
            self.probe_canvas_ids = [
                canvas.create_line(cx - 12, cy, cx + 12, cy, fill="red", width=3),
                canvas.create_line(cx, cy - 12, cx, cy + 12, fill="red", width=3)
            ]

    # --------------------------------------------------------------------- #
    # Exit application with cleanup
    # --------------------------------------------------------------------- #
    def exit_application(self):
        """Clean up resources and exit the application."""
        try:
            # Clear probe markers
            self.clear_probe()
            
            # Clear canvas images to release memory
            for canvas in [self.canvas_original, self.canvas_fluorescence, self.canvas_enhanced,
                           self.canvas_threshold, self.canvas_morphology,
                           self.canvas_contours, self.canvas_overlay]:
                canvas.delete("all")
                if hasattr(canvas, 'image'):
                    delattr(canvas, 'image')
            
            # Clear image arrays
            if self.original_image is not None:
                del self.original_image
            self.original_image = None
            
            if self.fluorescence_image is not None:
                del self.fluorescence_image
            self.fluorescence_image = None
            
            # Clear contours and statistics
            self.current_contours.clear()
            self.bacteria_stats.clear()
            
            # Unbind all events
            try:
                self.canvas_original.unbind("<Button-1>")
                self.canvas_original.unbind("<Button-3>")
                self.root.unbind("<Configure>")
            except:
                pass  # Ignore if already unbound
            
            # Destroy the window
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            # If cleanup fails, force exit anyway
            print(f"Cleanup error: {e}")
            try:
                self.root.destroy()
            except:
                pass


if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationViewer(root)
    root.mainloop()