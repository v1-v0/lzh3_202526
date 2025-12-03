#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main application window.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import json

from config.parameters import ParameterManager, PARAMETER_RANGES
from config.themes import ThemeManager
from core.image_processing import ImageProcessor
from core.segmentation import BacteriaSegmenter
from analysis.statistics import BacteriaStatistics
from analysis.visualization import ResultVisualizer
from utils.file_utils import ImageFileManager
from utils.platform_utils import PlatformHelper
from .controls import ParameterControls
from .canvas_manager import CanvasManager
from .statistics_panel import StatisticsPanel
from .widgets import ToolTip


class MainWindow:
    """Main application window."""
    
    def __init__(self, root: tk.Tk):
        """Initialize main window.
        
        Args:
            root: Root Tk instance
        """
        self.root = root
        self.root.title("Bacteria Analyzer v2.0")
        self.root.geometry("1600x900")
        
        # Initialize managers
        self.param_manager = ParameterManager()
        self.theme_manager = ThemeManager()
        self.canvas_manager = CanvasManager()
        self.file_manager = ImageFileManager()
        
        # State variables
        self.current_folder: Optional[Path] = None
        self.current_file: Optional[Path] = None
        self.image_files: List[Path] = []
        self.current_index: int = 0
        
        self.bf_image: Optional[np.ndarray] = None
        self.fluor_image: Optional[np.ndarray] = None
        self.labeled_image: Optional[np.ndarray] = None
        self.contours: List = []
        self.stats: List[Dict] = []
        
        # UI variables
        self.dark_mode_var = tk.BooleanVar(value=False)
        self.recursive_var = tk.BooleanVar(value=False)
        
        # Build UI
        self._create_menu()
        self._create_widgets()
        self._apply_theme()
        
        # Bind events
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.bind('<Left>', lambda e: self._navigate_image(-1))
        self.root.bind('<Right>', lambda e: self._navigate_image(1))
        self.root.bind('<Control-o>', lambda e: self._open_folder())
        self.root.bind('<Control-s>', lambda e: self._save_results())
        self.root.bind('<Control-e>', lambda e: self._export_all())
    
    def _create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Folder...", 
                            command=self._open_folder,
                            accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Save Current Results...",
                            command=self._save_results,
                            accelerator="Ctrl+S")
        file_menu.add_command(label="Export All to CSV...",
                            command=self._export_all,
                            accelerator="Ctrl+E")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)
        
        # Parameters menu
        param_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Parameters", menu=param_menu)
        param_menu.add_command(label="Load Parameters...",
                             command=self._load_parameters)
        param_menu.add_command(label="Save Parameters...",
                             command=self._save_parameters)
        param_menu.add_separator()
        param_menu.add_command(label="Reset to Defaults",
                             command=self._reset_parameters)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Dark Mode",
                                 variable=self.dark_mode_var,
                                 command=self._toggle_theme)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="Keyboard Shortcuts", 
                            command=self._show_shortcuts)
    
    def _create_widgets(self):
        """Create main window widgets."""
        # Main container
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        left_panel = self._create_left_panel()
        main_paned.add(left_panel, weight=1)
        
        # Right panel - Images and Stats
        right_panel = self._create_right_panel()
        main_paned.add(right_panel, weight=4)
    
    def _create_left_panel(self) -> ttk.Frame:
        """Create left control panel."""
        panel = ttk.Frame(self.root, width=300)
        
        # Scrollable frame for parameters
        canvas = tk.Canvas(panel, highlightthickness=0)
        scrollbar = ttk.Scrollbar(panel, orient="vertical", 
                                 command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Folder selection
        folder_frame = ttk.LabelFrame(scrollable_frame, 
                                     text=" Folder Selection ", 
                                     padding=8)
        folder_frame.pack(fill=tk.X, padx=5, pady=5)
        
        btn_frame = ttk.Frame(folder_frame)
        btn_frame.pack(fill=tk.X)
        
        open_btn = ttk.Button(btn_frame, text="📁 Open Folder",
                            command=self._open_folder)
        open_btn.pack(side=tk.LEFT, padx=(0, 5))
        ToolTip(open_btn, "Select folder containing microscopy images")
        
        ttk.Checkbutton(btn_frame, text="Recursive",
                       variable=self.recursive_var).pack(side=tk.LEFT)
        
        self.folder_label = ttk.Label(folder_frame, text="No folder selected",
                                     wraplength=270, foreground="gray")
        self.folder_label.pack(fill=tk.X, pady=(5, 0))
        
        # Image info
        info_frame = ttk.LabelFrame(scrollable_frame,
                                   text=" Current Image ",
                                   padding=8)
        info_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        self.image_info_label = ttk.Label(info_frame, text="No image loaded",
                                         wraplength=270)
        self.image_info_label.pack(fill=tk.X)
        
        # Navigation
        nav_frame = ttk.Frame(info_frame)
        nav_frame.pack(fill=tk.X, pady=(5, 0))
        
        prev_btn = ttk.Button(nav_frame, text="◀ Previous",
                            command=lambda: self._navigate_image(-1))
        prev_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ToolTip(prev_btn, "Previous image (Left Arrow)")
        
        next_btn = ttk.Button(nav_frame, text="Next ▶",
                            command=lambda: self._navigate_image(1))
        next_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
        ToolTip(next_btn, "Next image (Right Arrow)")
        
        # Parameter controls - Pass PARAMETER_RANGES to controls
        params = self.param_manager.get_tk_variables()
        self.param_controls = ParameterControls(
            scrollable_frame, params, self._on_parameter_change,
        )
        
        self.param_controls.create_threshold_controls()
        self.param_controls.create_clahe_controls()
        self.param_controls.create_morphology_controls()
        self.param_controls.create_watershed_controls()
        self.param_controls.create_fluorescence_controls()
        self.param_controls.create_label_controls()
        
        # Action buttons
        action_frame = ttk.LabelFrame(scrollable_frame,
                                     text=" Actions ",
                                     padding=8)
        action_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        process_btn = ttk.Button(action_frame, text="🔄 Reprocess",
                               command=self._process_current_image)
        process_btn.pack(fill=tk.X, pady=2)
        ToolTip(process_btn, "Reprocess current image with current parameters")
        
        save_btn = ttk.Button(action_frame, text="💾 Save Results",
                            command=self._save_results)
        save_btn.pack(fill=tk.X, pady=2)
        ToolTip(save_btn, "Save current analysis results (Ctrl+S)")
        
        export_btn = ttk.Button(action_frame, text="📊 Export All to CSV",
                              command=self._export_all)
        export_btn.pack(fill=tk.X, pady=2)
        ToolTip(export_btn, "Export all images' statistics to CSV (Ctrl+E)")
        
        # Pack scrollable components
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Mousewheel scrolling
        def on_mousewheel(event):
            delta = PlatformHelper.get_mousewheel_delta(event)
            canvas.yview_scroll(delta, "units")
        
        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
            canvas.bind_all("<Button-4>", on_mousewheel)
            canvas.bind_all("<Button-5>", on_mousewheel)
        
        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")
        
        PlatformHelper.bind_mousewheel(canvas, on_mousewheel)

        return panel
    
    def _create_right_panel(self) -> ttk.Frame:
        """Create right panel with images and statistics."""
        panel = ttk.Frame(self.root)
        
        # Notebook for different views
        notebook = ttk.Notebook(panel)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Images tab
        images_frame = ttk.Frame(notebook)
        notebook.add(images_frame, text="Images")
        
        # Create 2x2 grid for images
        for i in range(2):
            images_frame.grid_rowconfigure(i, weight=1)
            images_frame.grid_columnconfigure(i, weight=1)
        
        # Bright-field
        bf_frame = ttk.LabelFrame(images_frame, text=" Bright-field ",
                                 padding=5)
        bf_frame.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)
        self.bf_canvas = tk.Canvas(bf_frame, bg='black')
        self.bf_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Fluorescence
        fluor_frame = ttk.LabelFrame(images_frame, text=" Fluorescence ",
                                    padding=5)
        fluor_frame.grid(row=0, column=1, sticky='nsew', padx=2, pady=2)
        self.fluor_canvas = tk.Canvas(fluor_frame, bg='black')
        self.fluor_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Segmentation
        seg_frame = ttk.LabelFrame(images_frame, text=" Segmentation ",
                                  padding=5)
        seg_frame.grid(row=1, column=0, sticky='nsew', padx=2, pady=2)
        self.seg_canvas = tk.Canvas(seg_frame, bg='black')
        self.seg_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Overlay
        overlay_frame = ttk.LabelFrame(images_frame, text=" Overlay ",
                                      padding=5)
        overlay_frame.grid(row=1, column=1, sticky='nsew', padx=2, pady=2)
        self.overlay_canvas = tk.Canvas(overlay_frame, bg='black')
        self.overlay_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Statistics tab
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Statistics")
        
        self.stats_panel = StatisticsPanel(
            stats_frame,
            self._on_bacteria_select,
            self.dark_mode_var
        )
        
        return panel
    
    def _apply_theme(self):
        """Apply current theme to UI."""
        is_dark = self.dark_mode_var.get()
        
        if is_dark:
            self.theme_manager.apply_dark_theme(self.root)
        else:
            self.theme_manager.apply_light_theme(self.root)
        
        # Update canvas backgrounds
        bg = '#1e1e1e' if is_dark else 'black'
        for canvas in [self.bf_canvas, self.fluor_canvas,
                      self.seg_canvas, self.overlay_canvas]:
            canvas.config(bg=bg)
    
    def _toggle_theme(self):
        """Toggle between light and dark themes."""
        self._apply_theme()
        if hasattr(self, 'stats_panel'):
            self.stats_panel._update_histograms()
    
    def _open_folder(self):
        """Open folder dialog and scan for images."""
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder:
            return
        
        self.current_folder = Path(folder)
        recursive = self.recursive_var.get()
        
        self.image_files = self.file_manager.scan_folder(
            self.current_folder, recursive
        )
        
        if not self.image_files:
            messagebox.showwarning(
                "No Images",
                "No valid bright-field images (*_ch00.tif) found in the selected folder."
            )
            self.folder_label.config(text="No valid images found")
            return
        
        self.current_index = 0
        folder_text = str(self.current_folder)
        if len(folder_text) > 40:
            folder_text = "..." + folder_text[-37:]
        
        self.folder_label.config(
            text=f"{folder_text}\n{len(self.image_files)} images found",
            foreground="green"
        )
        
        self._load_image(self.image_files[0])
    
    def _load_image(self, filepath: Path):
        """Load and process image.
        
        Args:
            filepath: Path to bright-field image
        """
        self.current_file = filepath
        
        # Load bright-field
        self.bf_image = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        if self.bf_image is None:
            messagebox.showerror("Error", f"Failed to load image:\n{filepath}")
            return
        
        # Load fluorescence
        fluor_path = self.file_manager.get_fluorescence_path(filepath)
        
        if fluor_path:
            self.fluor_image = cv2.imread(str(fluor_path), cv2.IMREAD_GRAYSCALE)
            if self.fluor_image is None:
                messagebox.showwarning(
                    "Warning",
                    f"Failed to load fluorescence image:\n{fluor_path}"
                )
        else:
            self.fluor_image = None
        
        # Update info
        self._update_image_info()
        
        # Process
        self._process_current_image()
    
    def _process_current_image(self):
        """Process current image with current parameters."""
        if self.bf_image is None:
            return
        
        try:
            # Get parameters
            params = self.param_manager.get_values()
            
            print(f"DEBUG: Processing with watershed_dilate={params['watershed_dilate']}")
            
            # Preprocess
            processed = ImageProcessor.preprocess_brightfield(
                self.bf_image,
                enable_clahe=params['enable_clahe'],
                clahe_clip=params['clahe_clip'],
                clahe_tile=params['clahe_tile']
            )
            
            # Threshold
            if params['use_otsu']:
                binary = ImageProcessor.threshold_otsu(processed)
            else:
                binary = ImageProcessor.threshold_manual(
                    processed, params['manual_threshold']
                )
            
            # Morphology
            morphed = ImageProcessor.morphology_operations(
                binary,
                open_kernel=params['open_kernel'],
                open_iter=params['open_iter'],
                close_kernel=params['close_kernel'],
                close_iter=params['close_iter']
            )
            
            print(f"DEBUG: Binary sum before watershed: {morphed.sum()}")
            
            # Segment
            self.labeled_image, self.contours = BacteriaSegmenter.watershed_segmentation(
                morphed,
                watershed_dilate=params['watershed_dilate']
            )
            
            print(f"DEBUG: Found {len(self.contours)} contours before filtering")
            
            # Filter
            self.contours = BacteriaSegmenter.filter_bacteria(
                self.contours, min_area=params['min_area']
            )
            
            print(f"DEBUG: {len(self.contours)} contours after filtering (min_area={params['min_area']})")
            
            # Calculate statistics
            self.stats = BacteriaStatistics.calculate_stats(
                self.bf_image,
                self.fluor_image,
                self.contours,
                pixel_size_um=params['pixel_size_um'],
                min_fluor_per_area=params['min_fluor_per_area']
            )
            
            # Visualize
            self._update_displays()
            
            # Update statistics panel
            self.stats_panel.update_data(self.stats)
            
        except Exception as e:
            messagebox.showerror("Processing Error", str(e))
            import traceback
            traceback.print_exc()
    
    def _update_displays(self):
        """Update all canvas displays."""
        if self.bf_image is None:
            return
        
        params = self.param_manager.get_values()
        
        # Bright-field
        self.canvas_manager.display_image(
            self.bf_image, self.bf_canvas
        )
        
        # Fluorescence
        if self.fluor_image is not None:
            self.canvas_manager.display_fluorescence(
                self.fluor_image,
                self.fluor_canvas,
                params['fluor_brightness'],
                params['fluor_gamma']
            )
        else:
            self.fluor_canvas.delete("all")
        
        # Segmentation
        seg_vis = ResultVisualizer.draw_segmentation(
            self.bf_image, self.contours
        )
        self.canvas_manager.display_image(seg_vis, self.seg_canvas)
        
        # Overlay
        overlay = ResultVisualizer.create_overlay(
            self.bf_image,
            self.fluor_image,
            self.contours,
            self.stats,
            fluor_brightness=params['fluor_brightness'],
            fluor_gamma=params['fluor_gamma'],
            show_labels=params['show_labels'],
            label_font_size=params['label_font_size'],
            arrow_length=params['arrow_length'],
            label_offset=params['label_offset'],
            show_scale_bar=params['show_scale_bar'],
            pixel_size_um=params['pixel_size_um']
        )
        self.canvas_manager.display_image(overlay, self.overlay_canvas)
    
    def _update_image_info(self):
        """Update image info label."""
        if not self.current_file:
            self.image_info_label.config(text="No image loaded")
            return
        
        idx = self.current_index + 1
        total = len(self.image_files)
        name = self.current_file.name
        
        h, w = self.bf_image.shape if self.bf_image is not None else (0, 0)
        fluor_status = "✓" if self.fluor_image is not None else "✗"
        
        info_text = (
            f"Image {idx} of {total}\n"
            f"{name}\n"
            f"Size: {w}×{h} px\n"
            f"Fluorescence: {fluor_status}"
        )
        
        self.image_info_label.config(text=info_text)
    
    def _navigate_image(self, delta: int):
        """Navigate to next/previous image.
        
        Args:
            delta: Direction (-1 for previous, +1 for next)
        """
        if not self.image_files:
            return
        
        new_index = self.current_index + delta
        if 0 <= new_index < len(self.image_files):
            self.current_index = new_index
            self._load_image(self.image_files[self.current_index])
    
    def _on_parameter_change(self):
        """Handle parameter change event."""
        # Auto-reprocess if image is loaded
        if self.bf_image is not None:
            self._process_current_image()
    
    def _on_bacteria_select(self, index: int):
        """Handle bacteria selection from statistics table.
        
        Args:
            index: Selected bacteria index
        """
        if not self.stats or index >= len(self.stats):
            return
        
        # Highlight selected bacterium in overlay
        params = self.param_manager.get_values()
        
        overlay = ResultVisualizer.create_overlay(
            self.bf_image,
            self.fluor_image,
            self.contours,
            self.stats,
            fluor_brightness=params['fluor_brightness'],
            fluor_gamma=params['fluor_gamma'],
            show_labels=params['show_labels'],
            label_font_size=params['label_font_size'],
            arrow_length=params['arrow_length'],
            label_offset=params['label_offset'],
            highlight_index=index,
            show_scale_bar=params['show_scale_bar'],
            pixel_size_um=params['pixel_size_um']
        )
        
        self.canvas_manager.display_image(overlay, self.overlay_canvas)
    
    def _save_results(self):
        """Save current analysis results."""
        if not self.current_file or not self.stats:
            messagebox.showinfo("No Data", "No results to save. Process an image first.")
            return
        
        if self.current_folder is None:
            messagebox.showerror("Error", "No folder selected.")
            return

        output_dir = self.current_folder / "analysis_output"
        output_dir.mkdir(exist_ok=True)
        
        base_name = self.current_file.stem.replace('_ch00', '')
        
        # Save overlay image
        params = self.param_manager.get_values()
        overlay = ResultVisualizer.create_overlay(
            self.bf_image,
            self.fluor_image,
            self.contours,
            self.stats,
            fluor_brightness=params['fluor_brightness'],
            fluor_gamma=params['fluor_gamma'],
            show_labels=params['show_labels'],
            label_font_size=params['label_font_size'],
            arrow_length=params['arrow_length'],
            label_offset=params['label_offset'],
            show_scale_bar=params['show_scale_bar'],
            pixel_size_um=params['pixel_size_um']
        )
        
        overlay_path = output_dir / f"{base_name}_overlay.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Save statistics CSV
        csv_path = output_dir / f"{base_name}_stats.csv"
        BacteriaStatistics.export_to_csv([self.stats], [base_name], csv_path)
        
        messagebox.showinfo(
            "Success",
            f"Results saved to:\n{output_dir}\n\n"
            f"• {overlay_path.name}\n"
            f"• {csv_path.name}"
        )
    
    def _export_all(self):
        """Export all images' statistics to single CSV."""
        if not self.image_files:
            messagebox.showinfo("No Data", "No images loaded. Open a folder first.")
            return
        
        output_file = filedialog.asksaveasfilename(
            title="Export All Statistics",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="all_bacteria_stats.csv"
        )
        
        if not output_file:
            return
        
        try:
            all_stats = []
            all_names = []
            
            # Progress tracking
            total = len(self.image_files)
            
            for i, img_path in enumerate(self.image_files, 1):
                # Load images
                bf = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if bf is None:
                    continue
                
                fluor_path = self.file_manager.get_fluorescence_path(img_path)
                fluor = None
                if fluor_path:
                    fluor = cv2.imread(str(fluor_path), cv2.IMREAD_GRAYSCALE)
                
                # Process
                params = self.param_manager.get_values()
                
                processed = ImageProcessor.preprocess_brightfield(
                    bf, params['enable_clahe'],
                    params['clahe_clip'], params['clahe_tile']
                )
                
                if params['use_otsu']:
                    binary = ImageProcessor.threshold_otsu(processed)
                else:
                    binary = ImageProcessor.threshold_manual(
                        processed, params['manual_threshold']
                    )
                
                morphed = ImageProcessor.morphology_operations(
                    binary, params['open_kernel'], params['open_iter'],
                    params['close_kernel'], params['close_iter']
                )
                
                _, contours = BacteriaSegmenter.watershed_segmentation(
                    morphed, params['watershed_dilate']
                )
                
                contours = BacteriaSegmenter.filter_bacteria(
                    contours, params['min_area']
                )
                
                stats = BacteriaStatistics.calculate_stats(
                    bf, fluor, contours,
                    params['pixel_size_um'],
                    params['min_fluor_per_area']
                )
                
                all_stats.append(stats)
                all_names.append(img_path.stem.replace('_ch00', ''))
                
                # Update progress
                print(f"Processed {i}/{total}: {img_path.name}")
            
            # Export
            BacteriaStatistics.export_to_csv(
                all_stats, all_names, Path(output_file)
            )
            
            messagebox.showinfo(
                "Success",
                f"Exported statistics for {len(all_stats)} images to:\n{output_file}"
            )
            
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
            import traceback
            traceback.print_exc()
    
    def _load_parameters(self):
        """Load parameters from JSON file."""
        filepath = filedialog.askopenfilename(
            title="Load Parameters",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            self.param_manager.load_from_file(str(filepath))
            
            # Update UI
            params = self.param_manager.get_values()
            self.param_controls.reset_to_defaults(params)
            
            # Reprocess
            if self.bf_image is not None:
                self._process_current_image()
            
            messagebox.showinfo("Success", "Parameters loaded successfully.")
            
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
    
    def _save_parameters(self):
        """Save parameters to JSON file."""
        filepath = filedialog.asksaveasfilename(
            title="Save Parameters",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="bacteria_params.json"
        )
        
        if not filepath:
            return
        
        try:
            self.param_manager.save_to_file(str(filepath))
            messagebox.showinfo("Success", f"Parameters saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))
    
    def _reset_parameters(self):
        """Reset all parameters to defaults."""
        result = messagebox.askyesno(
            "Reset Parameters",
            "Reset all parameters to default values?"
        )
        
        if result:
            defaults = self.param_manager.get_defaults()
            self.param_controls.reset_to_defaults(defaults)
            self.param_manager.reset_to_defaults()
            
            if self.bf_image is not None:
                self._process_current_image()
    
    def _show_about(self):
        """Show about dialog."""
        about_text = (
            "Bacteria Analyzer v2.0\n\n"
            "A comprehensive tool for analyzing bacterial microscopy images.\n\n"
            "Features:\n"
            "• Automated bacteria segmentation\n"
            "• Fluorescence analysis\n"
            "• Statistical analysis and export\n"
            "• Batch processing\n\n"
            "Developed with Python, OpenCV, and tkinter"
        )
        
        messagebox.showinfo("About", about_text)
    
    def _show_shortcuts(self):
        """Show keyboard shortcuts dialog."""
        shortcuts = (
            "Keyboard Shortcuts\n\n"
            "Navigation:\n"
            "  Left Arrow    - Previous image\n"
            "  Right Arrow   - Next image\n\n"
            "File Operations:\n"
            "  Ctrl+O        - Open folder\n"
            "  Ctrl+S        - Save current results\n"
            "  Ctrl+E        - Export all to CSV\n"
        )
        
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)
    
    def _on_closing(self):
        """Handle window closing."""
        result = messagebox.askyesnocancel(
            "Exit",
            "Do you want to save parameters before exiting?"
        )
        
        if result is None:  # Cancel
            return
        elif result:  # Yes
            self._save_parameters()
        
        self.root.destroy()