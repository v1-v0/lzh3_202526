import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import json

@dataclass
class SegmentationParams:
    """Parameters for bacteria segmentation"""
    gaussian_kernel: int = 5
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    threshold_block: int = 21
    threshold_c: int = 5
    morph_kernel: int = 3
    morph_iterations: int = 2
    min_area: int = 50
    max_area: int = 5000
    min_fluor_per_area: float = 0.0
    show_labels: bool = True
    arrow_length: int = 60
    label_offset: int = 15

class CollapsiblePanel(ttk.Frame):
    """A collapsible panel widget"""
    def __init__(self, parent, title="", **kwargs):
        super().__init__(parent, **kwargs)
        self.is_collapsed = False
        
        # Header frame with click binding
        self.header = ttk.Frame(self, relief="raised", borderwidth=1)
        self.header.pack(fill="x", padx=2, pady=2)
        self.header.bind("<Button-1>", self.toggle)
        
        # Arrow label
        self.arrow_label = ttk.Label(self.header, text="▼", width=2)
        self.arrow_label.pack(side="left")
        self.arrow_label.bind("<Button-1>", self.toggle)
        
        # Title label
        self.title_label = ttk.Label(self.header, text=title, font=("Segoe UI", 10, "bold"))
        self.title_label.pack(side="left", fill="x", expand=True)
        self.title_label.bind("<Button-1>", self.toggle)
        
        # Content frame
        self.content = ttk.Frame(self)
        self.content.pack(fill="both", expand=True, padx=5, pady=5)
    
    def toggle(self, event=None):
        """Toggle panel collapse state"""
        if self.is_collapsed:
            self.expand()
        else:
            self.collapse()
    
    def collapse(self):
        """Collapse the panel"""
        self.content.pack_forget()
        self.arrow_label.config(text="▶")
        self.is_collapsed = True
    
    def expand(self):
        """Expand the panel"""
        self.content.pack(fill="both", expand=True, padx=5, pady=5)
        self.arrow_label.config(text="▼")
        self.is_collapsed = False

class BacteriaSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bacteria Segmentation Tool")
        
        # Keep references to PhotoImage objects to prevent garbage collection
        from typing import Any
        self._canvas_images: dict[tk.Canvas, Any] = {}
        
        # Initialize variables
        self.bf_files: List[Path] = []
        self.current_index = 0
        
        # Data
        self.original_image: Optional[np.ndarray] = None
        self.fluorescence_image: Optional[np.ndarray] = None
        self.current_file: Optional[Path] = None
        self.current_fluor_file: Optional[Path] = None
        self.current_contours: List[np.ndarray] = []
        self.bacteria_stats: List[Dict] = []
        self.current_bacteria_index: int = -1
        
        # Probe state
        self.probe_point: Optional[Tuple[int, int]] = None
        self.probe_canvas_ids: List[int] = []
        
        # Parameters
        self.params = {
            'gaussian_kernel': tk.IntVar(value=5),
            'clahe_clip': tk.DoubleVar(value=2.0),
            'clahe_grid': tk.IntVar(value=8),
            'threshold_block': tk.IntVar(value=21),
            'threshold_c': tk.IntVar(value=5),
            'morph_kernel': tk.IntVar(value=3),
            'morph_iterations': tk.IntVar(value=2),
            'min_area': tk.IntVar(value=50),
            'max_area': tk.IntVar(value=5000),
            'min_fluor_per_area': tk.DoubleVar(value=0.0),
            'show_labels': tk.BooleanVar(value=True),
            'arrow_length': tk.IntVar(value=60),
            'label_offset': tk.IntVar(value=15),
        }
        
        # Progress bar values
        self.progressbars = {}
        
        self.setup_ui()
        self.load_last_settings()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Left panel (controls)
        left_panel = ttk.Frame(main_container, width=320)
        left_panel.pack(side="left", fill="y", padx=(0, 5))
        left_panel.pack_propagate(False)
        
        # Right panel (image display)
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side="left", fill="both", expand=True)
        
        # Setup left panel sections
        self.setup_file_section(left_panel)
        self.setup_measurement_section(left_panel)
        self.setup_navigation_section(left_panel)
        self.setup_parameters_section(left_panel)
        self.setup_export_section(left_panel)
        
        # Setup right panel
        self.setup_image_display(right_panel)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")
    
    def setup_file_section(self, parent):
        """Setup file loading section"""
        panel = CollapsiblePanel(parent, title="📁 Load Images")
        panel.pack(fill="x", pady=(0, 5))
        
        # Brightfield image
        bf_frame = ttk.LabelFrame(panel.content, text="Brightfield Image", padding=5)
        bf_frame.pack(fill="x", pady=(0, 5))
        
        ttk.Button(bf_frame, text="Load BF Image", command=self.load_image).pack(fill="x")
        self.bf_label = ttk.Label(bf_frame, text="No image loaded", foreground="gray")
        self.bf_label.pack(fill="x", pady=(5, 0))
        
        # Fluorescence image
        fluor_frame = ttk.LabelFrame(panel.content, text="Fluorescence Image (Optional)", padding=5)
        fluor_frame.pack(fill="x")
        
        ttk.Button(fluor_frame, text="Load Fluorescence", command=self.load_fluorescence).pack(fill="x")
        self.fluor_label = ttk.Label(fluor_frame, text="No fluorescence loaded", foreground="gray")
        self.fluor_label.pack(fill="x", pady=(5, 0))
    
    def setup_measurement_section(self, parent):
        """Setup measurement panel"""
        self.measure_panel = CollapsiblePanel(parent, title="🔍 Pixel Probe")
        self.measure_panel.pack(fill="x", pady=(0, 5))
        self.measure_panel.collapse()  # Start collapsed
        
        # Info label
        info_label = ttk.Label(
            self.measure_panel.content, 
            text="Click on image to probe\nCtrl+Click to auto-tune",
            font=("Segoe UI", 9),
            foreground="gray"
        )
        info_label.pack(pady=(0, 10))
        
        # Measurement display
        self.measure_text = tk.Text(
            self.measure_panel.content, 
            height=8, 
            width=35,
            font=("Consolas", 9),
            state="disabled",
            background="#f0f0f0"
        )
        self.measure_text.pack(fill="both", expand=True)
    
    def setup_navigation_section(self, parent):
        """Setup bacteria navigation section"""
        self.nav_panel = CollapsiblePanel(parent, title="🧬 Bacteria Navigation")
        self.nav_panel.pack(fill="x", pady=(0, 5))
        self.nav_panel.collapse()  # Start collapsed
        
        # Navigation info
        self.nav_info_label = ttk.Label(
            self.nav_panel.content,
            text="No bacteria detected",
            font=("Segoe UI", 9)
        )
        self.nav_info_label.pack(pady=(0, 5))
        
        # Navigation buttons
        nav_buttons = ttk.Frame(self.nav_panel.content)
        nav_buttons.pack(fill="x", pady=5)
        
        self.btn_first = ttk.Button(nav_buttons, text="⏮ First", command=self.goto_first_bacterium, state="disabled")
        self.btn_first.pack(side="left", fill="x", expand=True, padx=(0, 2))
        
        self.btn_prev = ttk.Button(nav_buttons, text="◀ Prev", command=self.goto_prev_bacterium, state="disabled")
        self.btn_prev.pack(side="left", fill="x", expand=True, padx=2)
        
        self.btn_next = ttk.Button(nav_buttons, text="Next ▶", command=self.goto_next_bacterium, state="disabled")
        self.btn_next.pack(side="left", fill="x", expand=True, padx=2)
        
        self.btn_last = ttk.Button(nav_buttons, text="Last ⏭", command=self.goto_last_bacterium, state="disabled")
        self.btn_last.pack(side="left", fill="x", expand=True, padx=(2, 0))
        
        # Compact statistics display
        stats_frame = ttk.LabelFrame(self.nav_panel.content, text="Selected Bacterium", padding=5)
        stats_frame.pack(fill="both", expand=True, pady=(5, 0))
        
        self.compact_stats_text = tk.Text(
            stats_frame,
            height=6,
            width=35,
            font=("Consolas", 9),
            state="disabled",
            background="#f0f0f0"
        )
        self.compact_stats_text.pack(fill="both", expand=True)
    
    def setup_parameters_section(self, parent):
        """Setup parameters section with collapsible subsections"""
        # Create scrollable frame for parameters
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Preprocessing parameters
        preproc_panel = CollapsiblePanel(scrollable_frame, title="⚙️ Preprocessing")
        preproc_panel.pack(fill="x", pady=(0, 5))
        
        self.add_slider(preproc_panel.content, "Gaussian Kernel", "gaussian_kernel", 1, 15, 2)
        self.add_slider(preproc_panel.content, "CLAHE Clip Limit", "clahe_clip", 0.5, 10.0, 0.5)
        self.add_slider(preproc_panel.content, "CLAHE Grid Size", "clahe_grid", 2, 16, 2)
        
        # Thresholding parameters
        thresh_panel = CollapsiblePanel(scrollable_frame, title="🎯 Thresholding")
        thresh_panel.pack(fill="x", pady=(0, 5))
        
        self.add_slider(thresh_panel.content, "Block Size", "threshold_block", 3, 51, 2)
        self.add_slider(thresh_panel.content, "Constant C", "threshold_c", -20, 20, 1)
        
        # Morphology parameters
        morph_panel = CollapsiblePanel(scrollable_frame, title="🔄 Morphology")
        morph_panel.pack(fill="x", pady=(0, 5))
        
        self.add_slider(morph_panel.content, "Kernel Size", "morph_kernel", 1, 15, 2)
        self.add_slider(morph_panel.content, "Iterations", "morph_iterations", 1, 10, 1)
        
        # Filtering parameters
        filter_panel = CollapsiblePanel(scrollable_frame, title="🔬 Size Filtering")
        filter_panel.pack(fill="x", pady=(0, 5))
        
        self.add_slider(filter_panel.content, "Min Area (px²)", "min_area", 10, 500, 10)
        self.add_slider(filter_panel.content, "Max Area (px²)", "max_area", 500, 10000, 100)
        
        # Fluorescence filtering
        fluor_panel = CollapsiblePanel(scrollable_frame, title="💡 Fluorescence Filter")
        fluor_panel.pack(fill="x", pady=(0, 5))
        
        self.add_slider(fluor_panel.content, "Min Fluor/Area", "min_fluor_per_area", 0, 100, 0.5)
        
        # Label & Arrow configuration
        label_panel = CollapsiblePanel(scrollable_frame, title="🏷️ Labels & Arrows")
        label_panel.pack(fill="x", pady=(0, 5))
        
        # Show labels checkbox
        show_labels_frame = ttk.Frame(label_panel.content)
        show_labels_frame.pack(fill="x", pady=(0, 5))
        ttk.Checkbutton(
            show_labels_frame,
            text="Show Labels",
            variable=self.params['show_labels'],
            command=self.update_preview
        ).pack(anchor="w")
        
        self.add_slider(label_panel.content, "Arrow Length", "arrow_length", 20, 150, 5)
        self.add_slider(label_panel.content, "Label Offset", "label_offset", 5, 50, 5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def setup_export_section(self, parent):
        """Setup export section"""
        panel = CollapsiblePanel(parent, title="💾 Export & Statistics")
        panel.pack(fill="x", pady=(0, 5))
        
        # Statistics table
        stats_frame = ttk.LabelFrame(panel.content, text="Statistics List", padding=5)
        stats_frame.pack(fill="both", expand=True, pady=(0, 5))
        
        # Create treeview with scrollbar
        tree_scroll = ttk.Scrollbar(stats_frame)
        tree_scroll.pack(side="right", fill="y")
        
        self.stats_tree = ttk.Treeview(
            stats_frame,
            columns=("ID", "Area", "Fluor", "F/A"),
            show="headings",
            height=8,
            yscrollcommand=tree_scroll.set
        )
        tree_scroll.config(command=self.stats_tree.yview)
        
        self.stats_tree.heading("ID", text="ID")
        self.stats_tree.heading("Area", text="Area")
        self.stats_tree.heading("Fluor", text="Fluor")
        self.stats_tree.heading("F/A", text="F/A")
        
        self.stats_tree.column("ID", width=40, anchor="center")
        self.stats_tree.column("Area", width=60, anchor="e")
        self.stats_tree.column("Fluor", width=60, anchor="e")
        self.stats_tree.column("F/A", width=60, anchor="e")
        
        self.stats_tree.pack(fill="both", expand=True)
        self.stats_tree.bind("<Double-Button-1>", self.on_stats_tree_double_click)
        
        # Export button
        ttk.Button(panel.content, text="Export Statistics (CSV)", command=self.export_statistics).pack(fill="x", pady=(5, 0))
    
    def add_slider(self, parent, label, param_key, min_val, max_val, increment):
        """Add a slider with label and value display"""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=3)
        
        # Label
        lbl = ttk.Label(frame, text=label, width=18, anchor="w")
        lbl.pack(side="left")
        
        # Value display
        value_var = self.params[param_key]
        if isinstance(min_val, float):
            value_label = ttk.Label(frame, text=f"{value_var.get():.1f}", width=6, anchor="e")
        else:
            value_label = ttk.Label(frame, text=f"{value_var.get()}", width=6, anchor="e")
        value_label.pack(side="right")
        
        # Progress bar for visual feedback
        progress = ttk.Progressbar(frame, length=100, mode='determinate')
        progress.pack(side="right", padx=5)
        self.progressbars[param_key] = progress
        
        # Slider
        if isinstance(min_val, float):
            resolution = increment
            slider = ttk.Scale(
                frame,
                from_=min_val,
                to=max_val,
                variable=value_var,
                orient="horizontal",
                command=lambda v, k=param_key, vl=value_label, r=resolution: self.on_slider_change(k, v, vl, r)
            )
        else:
            slider = ttk.Scale(
                frame,
                from_=min_val,
                to=max_val,
                variable=value_var,
                orient="horizontal",
                command=lambda v, k=param_key, vl=value_label: self.on_slider_change(k, v, vl)
            )
        slider.pack(side="right", fill="x", expand=True)
    
    def update_progressbar(self, param_key):
        """Update progress bar based on parameter value"""
        if param_key not in self.progressbars:
            return
        
        progress = self.progressbars[param_key]
        value_var = self.params[param_key]
        value = value_var.get()
        
        # Define ranges for each parameter
        ranges = {
            'gaussian_kernel': (1, 15),
            'clahe_clip': (0.5, 10.0),
            'clahe_grid': (2, 16),
            'threshold_block': (3, 51),
            'threshold_c': (-20, 20),
            'morph_kernel': (1, 15),
            'morph_iterations': (1, 10),
            'min_area': (10, 500),
            'max_area': (500, 10000),
            'min_fluor_per_area': (0, 100),
            'arrow_length': (20, 150),
            'label_offset': (5, 50),
        }
        
        if param_key in ranges:
            min_val, max_val = ranges[param_key]
            percentage = ((value - min_val) / (max_val - min_val)) * 100
            progress['value'] = percentage
    
    def on_slider_change(self, param_key, value, value_label, resolution=None):
        """Handle slider value change"""
        if resolution:
            val = float(value)
            value_label.config(text=f"{val:.1f}")
        else:
            val = int(float(value))
            self.params[param_key].set(val)
            value_label.config(text=f"{val}")
        
        self.update_progressbar(param_key)
        self.update_preview()
    
    def setup_image_display(self, parent):
        """Setup image display grid"""
        # Create 3x3 grid
        for i in range(3):
            parent.grid_rowconfigure(i, weight=1)
            parent.grid_columnconfigure(i, weight=1)
        
        # Image titles and canvases
        titles = [
            "Original BF", "Enhanced", "Thresholded",
            "Morphology Cleaned", "Final Contours", "BF+Fluor Overlay",
            "Fluorescence (Red)", "", "Statistics"
        ]
        
        self.canvases = {}
        canvas_keys = [
            "original", "enhanced", "threshold",
            "morphology", "contours", "overlay",
            "fluorescence", None, None
        ]
        
        for idx, (title, key) in enumerate(zip(titles, canvas_keys)):
            row = idx // 3
            col = idx % 3
            
            frame = ttk.LabelFrame(parent, text=title, padding=2)
            frame.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
            
            if key:
                canvas = tk.Canvas(frame, bg="black", highlightthickness=0)
                canvas.pack(fill="both", expand=True)
                self.canvases[key] = canvas
                
                # Bind click event only to original canvas
                if key == "original":
                    canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Store canvas references for easy access
        self.canvas_original = self.canvases["original"]
        self.canvas_enhanced = self.canvases["enhanced"]
        self.canvas_threshold = self.canvases["threshold"]
        self.canvas_morphology = self.canvases["morphology"]
        self.canvas_contours = self.canvases["contours"]
        self.canvas_overlay = self.canvases["overlay"]
        self.canvas_fluorescence = self.canvases["fluorescence"]
    
    def load_image(self):
        """Load brightfield image"""
        file_path = filedialog.askopenfilename(
            title="Select Brightfield Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_file = Path(file_path)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                messagebox.showerror("Error", "Failed to load image")
                return
            
            self.original_image = img
            self.bf_label.config(text=self.current_file.name, foreground="black")
            self.status_var.set(f"Loaded: {self.current_file.name}")
            
            # Try to auto-load matching fluorescence image
            self.auto_load_fluorescence()
            
            self.update_preview()
    
    def load_fluorescence(self):
        """Load fluorescence image"""
        file_path = filedialog.askopenfilename(
            title="Select Fluorescence Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_fluor_file = Path(file_path)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                messagebox.showerror("Error", "Failed to load fluorescence image")
                return
            
            # Check if dimensions match
            if self.original_image is not None:
                if img.shape != self.original_image.shape:
                    messagebox.showwarning(
                        "Warning",
                        f"Fluorescence image dimensions {img.shape} don't match BF image {self.original_image.shape}"
                    )
            
            self.fluorescence_image = img
            self.fluor_label.config(text=self.current_fluor_file.name, foreground="black")
            
            if self.original_image is not None:
                self.update_preview()
    
    def auto_load_fluorescence(self):
        """Try to automatically load matching fluorescence image"""
        if self.current_file is None:
            return
        
        # Common fluorescence naming patterns
        bf_name = self.current_file.stem
        bf_dir = self.current_file.parent
        
        patterns = [
            f"{bf_name}_fluor",
            f"{bf_name}_fluorescence",
            f"{bf_name}_fl",
            f"{bf_name.replace('BF', 'FL')}",
            f"{bf_name.replace('bf', 'fl')}",
        ]
        
        for pattern in patterns:
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                fluor_path = bf_dir / f"{pattern}{ext}"
                if fluor_path.exists():
                    img = cv2.imread(str(fluor_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None and self.original_image is not None and img.shape == self.original_image.shape:
                        self.fluorescence_image = img
                        self.current_fluor_file = fluor_path
                        self.fluor_label.config(text=fluor_path.name, foreground="black")
                        self.status_var.set(f"Auto-loaded fluorescence: {fluor_path.name}")
                        return
    
    def segment_bacteria(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Segment bacteria from image"""
        # Get parameters
        gauss_k = self.params['gaussian_kernel'].get()
        if gauss_k % 2 == 0:
            gauss_k += 1
        
        clahe_clip = self.params['clahe_clip'].get()
        clahe_grid = self.params['clahe_grid'].get()
        
        thresh_block = self.params['threshold_block'].get()
        if thresh_block % 2 == 0:
            thresh_block += 1
        thresh_c = self.params['threshold_c'].get()
        
        morph_k = self.params['morph_kernel'].get()
        morph_iter = self.params['morph_iterations'].get()
        
        min_area = self.params['min_area'].get()
        max_area = self.params['max_area'].get()
        
        # 1. Gaussian blur
        blurred = cv2.GaussianBlur(image, (gauss_k, gauss_k), 0)
        
        # 2. CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
        enhanced = clahe.apply(blurred)
        
        # 3. Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            thresh_block,
            thresh_c
        )
        
        # 4. Morphological operations
        kernel = np.ones((morph_k, morph_k), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 5. Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 6. Filter by area
        filtered_contours = [
            cnt for cnt in contours
            if min_area <= cv2.contourArea(cnt) <= max_area
        ]
        
        return enhanced, thresh, cleaned, filtered_contours
    
    def calculate_bacteria_statistics(self, contours: List[np.ndarray], bf_image: np.ndarray, fluor_image: Optional[np.ndarray]) -> List[Dict]:
        """Calculate statistics for each bacterium"""
        stats = []
        
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Calculate circularity
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Create mask for this bacterium
            mask = np.zeros(bf_image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Calculate mean intensity in brightfield
            bf_mean = cv2.mean(bf_image, mask=mask)[0]
            
            # Calculate fluorescence statistics if available
            fluor_total = 0
            fluor_mean = 0
            fluor_per_area = 0
            
            if fluor_image is not None:
                fluor_total = cv2.sumElems(cv2.bitwise_and(fluor_image, fluor_image, mask=mask))[0]
                fluor_mean = cv2.mean(fluor_image, mask=mask)[0]
                fluor_per_area = fluor_total / area if area > 0 else 0
            
            stats.append({
                'id': idx + 1,
                'contour': contour,
                'area': area,
                'perimeter': perimeter,
                'aspect_ratio': aspect_ratio,
                'circularity': circularity,
                'bf_mean': bf_mean,
                'fluor_total': fluor_total,
                'fluor_mean': fluor_mean,
                'fluor_per_area': fluor_per_area,
                'bbox': (x, y, w, h)
            })
        
        return stats
    
    def create_overlay_image(self, bf_image: np.ndarray, fluor_image: Optional[np.ndarray], contours: List[np.ndarray]) -> np.ndarray:
        """Create overlay image with BF and fluorescence"""
        # Convert BF to BGR
        overlay = cv2.cvtColor(bf_image, cv2.COLOR_GRAY2BGR)
        
        if fluor_image is not None:
            # Normalize fluorescence to 0-255 range
            fluor_norm = cv2.normalize(fluor_image, None, 0, 255, cv2.NORM_MINMAX)
            
            # Create colored fluorescence (green channel)
            fluor_colored = np.zeros_like(overlay)
            fluor_colored[:, :, 1] = fluor_norm  # Green channel
            
            # Blend
            overlay = cv2.addWeighted(overlay, 0.7, fluor_colored, 0.3, 0)
        
        # Draw contours
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)  # Yellow contours
        
        return overlay
    
    def display_image(self, img: np.ndarray, canvas: tk.Canvas):
        """Display image on canvas"""
        if img is None:
            return
        
        # Get canvas size
        canvas.update()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        
        if cw <= 1 or ch <= 1:
            return
        
        # Convert to RGB if needed
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        h, w = img_rgb.shape[:2]
        scale = min(cw / w, ch / h) * 0.95
        
        if scale >= 1.0:
            scale = 1.0
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Display on canvas
        canvas.delete("all")
        x_offset = (cw - new_w) // 2
        y_offset = (ch - new_h) // 2
        canvas.create_image(x_offset, y_offset, anchor="nw", image=img_tk)
        self._canvas_images[canvas] = img_tk  # Keep reference
    
    def display_fluorescence_image(self, img: np.ndarray, canvas: tk.Canvas):
        """Display fluorescence image in RED with enhanced visibility"""
        if img is None:
            return
        
        canvas.update()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        
        if cw <= 1 or ch <= 1:
            return
        
        # Normalize to enhance visibility
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        # Create red-colored image
        img_colored = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        img_colored[:, :, 0] = img_norm  # Red channel
        
        # Resize to fit canvas
        h, w = img_colored.shape[:2]
        scale = min(cw / w, ch / h) * 0.95
        
        if scale >= 1.0:
            scale = 1.0
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img_resized = cv2.resize(img_colored, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Display on canvas
        canvas.delete("all")
        x_offset = (cw - new_w) // 2
        y_offset = (ch - new_h) // 2
        canvas.create_image(x_offset, y_offset, anchor="nw", image=img_tk)
        self._canvas_images[canvas] = img_tk
    
    def get_label_font(self):
        """Get font for labels"""
        try:
            return ImageFont.truetype("arial.ttf", 16)
        except:
            return ImageFont.load_default()
    
    def create_occupancy_map(self, image_shape: Tuple[int, int], contours: List[np.ndarray], margin: int = 20) -> np.ndarray:
        """Create occupancy map showing where contours are located"""
        h, w = image_shape
        occupancy = np.zeros((h, w), dtype=np.uint8)
        
        for contour in contours:
            cv2.drawContours(occupancy, [contour], -1, 255, -1)
            # Add margin around contour
            cv2.drawContours(occupancy, [contour], -1, 255, margin)
        
        return occupancy
    
    def find_best_label_position(
        self,
        centroid: Tuple[int, int],
        image_shape: Tuple[int, int],
        arrow_length: int,
        text_size: Tuple[int, int],
        label_offset: int,
        occupancy_map: np.ndarray
    ) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Find best position for label with arrow.
        Returns: (arrow_x, arrow_y, label_x, label_y, angle) or None
        """
        cx, cy = centroid
        h, w = image_shape
        text_w, text_h = text_size
        
        # Try angles in preferred order
        preferred_angles = [0, 45, 315, 90, 270, 135, 225, 180]
        
        for angle in preferred_angles:
            angle_rad = np.deg2rad(angle)
            
            # Calculate arrow endpoint
            arrow_x = int(cx + arrow_length * np.cos(angle_rad))
            arrow_y = int(cy - arrow_length * np.sin(angle_rad))
            
            # Check if arrow endpoint is within image
            if not (0 <= arrow_x < w and 0 <= arrow_y < h):
                continue
            
            # Calculate label position (offset from arrow endpoint)
            label_x = int(arrow_x + label_offset * np.cos(angle_rad))
            label_y = int(arrow_y - label_offset * np.sin(angle_rad))
            
            # Adjust label position based on angle to keep it readable
            if 45 < angle <= 135:  # Top
                label_y -= text_h
            elif 225 < angle <= 315:  # Bottom
                label_y += text_h
            
            if 135 < angle <= 225:  # Left
                label_x -= text_w
            
            # Check if label fits in image
            if not (0 <= label_x < w - text_w and 0 <= label_y < h - text_h):
                continue
            
            # Check if label area is unoccupied
            label_area = occupancy_map[label_y:label_y+text_h, label_x:label_x+text_w]
            if np.sum(label_area) == 0:  # Area is free
                return arrow_x, arrow_y, label_x, label_y, angle
        
        # If no perfect position found, use first valid position
        for angle in preferred_angles:
            angle_rad = np.deg2rad(angle)
            arrow_x = int(cx + arrow_length * np.cos(angle_rad))
            arrow_y = int(cy - arrow_length * np.sin(angle_rad))
            
            if 0 <= arrow_x < w and 0 <= arrow_y < h:
                label_x = int(arrow_x + label_offset * np.cos(angle_rad))
                label_y = int(arrow_y - label_offset * np.sin(angle_rad))
                
                # Adjust for readability
                if 45 < angle <= 135:
                    label_y -= text_h
                elif 225 < angle <= 315:
                    label_y += text_h
                if 135 < angle <= 225:
                    label_x -= text_w
                
                if 0 <= label_x < w - text_w and 0 <= label_y < h - text_h:
                    return arrow_x, arrow_y, label_x, label_y, angle
        
        return None
    
    def draw_labels_on_contours(self, img_bgr, contours):
        """
        Draw labels with arrows on detected bacteria using intelligent positioning.
        
        Args:
            img_bgr: BGR image from OpenCV
            contours: list of contours to label (may be filtered subset)
        
        Returns:
            img_bgr with labels drawn
        """
        if not self.params['show_labels'].get() or not contours:
            return img_bgr
        
        # Convert BGR to RGB for PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # Get parameters from config
        arrow_len = self.params['arrow_length'].get()
        label_offset = self.params['label_offset'].get()
        font = self.get_label_font()
        
        h, w = img_bgr.shape[:2]
        occupancy_map = self.create_occupancy_map((h, w), contours, margin=20)
        
        # Create a mapping from contour to stat for quick lookup
        # Compare contours by their array data
        contour_to_stat = {}
        for stat in self.bacteria_stats:
            # Use id(stat['contour']) or compare arrays
            for i, c in enumerate(contours):
                if np.array_equal(stat['contour'], c):
                    contour_to_stat[i] = stat
                    break
        
        # Draw labels for each contour using ORIGINAL ID from bacteria_stats
        for contour_idx, contour in enumerate(contours):
            # Get the stat for this contour
            if contour_idx not in contour_to_stat:
                continue
                
            stat = contour_to_stat[contour_idx]
            original_id = stat['id']  # Use original ID
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Get label text (use original ID)
            label_text = str(original_id)
            bbox = draw.textbbox((0, 0), label_text, font=font)
            #text_w = bbox[2] - bbox[0]
            #text_h = bbox[3] - bbox[1]
            text_w = int(bbox[2] - bbox[0])
            text_h = int(bbox[3] - bbox[1])
            
            # Find best label position using config parameters
            result = self.find_best_label_position(
                (cx, cy), 
                (h, w), 
                arrow_len,      # From config
                (text_w, text_h), 
                label_offset,   # From config
                occupancy_map
            )
            
            if result is None:
                continue
            
            arrow_x, arrow_y, label_x, label_y, angle = result
            
            # Highlight the currently selected bacterium
            # Find the index of this stat in bacteria_stats
            stat_idx = -1
            for idx, s in enumerate(self.bacteria_stats):
                if s['id'] == original_id:
                    stat_idx = idx
                    break
            
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
            
            # Filter based on min_fluor_per_area but PRESERVE original IDs
            min_fpa = self.params['min_fluor_per_area'].get()
            if self.fluorescence_image is not None and min_fpa > 0:
                # Keep only bacteria that pass the threshold
                self.bacteria_stats = [s for s in all_bacteria_stats if s['fluor_per_area'] >= min_fpa]
                
                # Extract filtered contours (maintaining original order)
                filtered_bacteria = [stat['contour'] for stat in self.bacteria_stats]
                
                # Update status to show filtering
                total_count = len(all_bacteria_stats)
                filtered_count = len(self.bacteria_stats)
                base_msg = f"Detected {filtered_count}/{total_count} bacteria (min fluor/area: {min_fpa:.1f})"
            else:
                # No filtering
                self.bacteria_stats = all_bacteria_stats
                filtered_bacteria = bacteria
                count = len(bacteria)
                base_msg = f"Detected {count} bacteria"
            
            # Keep ALL bacteria for intermediate views
            self.current_contours = bacteria  # ALL bacteria for measurements
            
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

            # ============================================================
            # FINAL CONTOURS VIEW - Show ONLY filtered bacteria with labels
            # ============================================================
            contour_img = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_img, filtered_bacteria, -1, (0, 255, 0), 2)

            # Highlight probed contour if exists (check if it's in filtered list)
            if self.probe_point and filtered_bacteria:
                x, y = self.probe_point
                for contour in filtered_bacteria:
                    if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                        cv2.drawContours(contour_img, [contour], -1, (0, 0, 255), 3)
                        break

            # Draw labels with smart positioning - USE ORIGINAL IDs from filtered bacteria
            contour_img = self.draw_labels_on_contours(contour_img, filtered_bacteria)

            self.display_image(contour_img, self.canvas_contours)
            
            # ============================================================
            # BF+FLUOR OVERLAY VIEW - Show ONLY filtered bacteria with labels
            # ============================================================
            overlay_img = self.create_overlay_image(
                self.original_image,
                self.fluorescence_image,
                filtered_bacteria  # Use filtered bacteria
            )
            
            # Add labels to overlay image
            overlay_img = self.draw_labels_on_contours(overlay_img, filtered_bacteria)
            
            self.display_image(overlay_img, self.canvas_overlay)
            
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def update_statistics_table(self):
        """Update the statistics table with current bacteria data"""
        # Clear existing items
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        
        # Add new items (only filtered bacteria)
        for stat in self.bacteria_stats:
            self.stats_tree.insert("", "end", values=(
                stat['id'],
                f"{stat['area']:.0f}",
                f"{stat['fluor_total']:.0f}",
                f"{stat['fluor_per_area']:.2f}"
            ))
    
    def update_compact_statistics(self):
        """Update compact statistics display for selected bacterium"""
        self.compact_stats_text.config(state="normal")
        self.compact_stats_text.delete(1.0, tk.END)
        
        if 0 <= self.current_bacteria_index < len(self.bacteria_stats):
            stat = self.bacteria_stats[self.current_bacteria_index]
            
            stats_text = f"""ID: {stat['id']}
Area: {stat['area']:.1f} px²
Fluor Total: {stat['fluor_total']:.1f}
Fluor/Area: {stat['fluor_per_area']:.2f}
Circularity: {stat['circularity']:.3f}
Aspect: {stat['aspect_ratio']:.2f}"""
            
            self.compact_stats_text.insert(1.0, stats_text)
        else:
            self.compact_stats_text.insert(1.0, "No bacterium selected")
        
        self.compact_stats_text.config(state="disabled")
    
    def on_stats_tree_double_click(self, event):
        """Handle double-click on statistics table"""
        selection = self.stats_tree.selection()
        if not selection:
            return
        
        item = self.stats_tree.item(selection[0])
        bact_id = int(item['values'][0])
        
        # Find the index in bacteria_stats
        for idx, stat in enumerate(self.bacteria_stats):
            if stat['id'] == bact_id:
                self.current_bacteria_index = idx
                self.update_bacteria_navigation_buttons()
                self.update_compact_statistics()
                self.update_preview()
                break
    
    def goto_first_bacterium(self):
        """Navigate to first bacterium"""
        if self.bacteria_stats:
            self.current_bacteria_index = 0
            self.update_bacteria_navigation_buttons()
            self.update_compact_statistics()
            self.update_preview()
    
    def goto_prev_bacterium(self):
        """Navigate to previous bacterium"""
        if self.bacteria_stats and self.current_bacteria_index > 0:
            self.current_bacteria_index -= 1
            self.update_bacteria_navigation_buttons()
            self.update_compact_statistics()
            self.update_preview()
    
    def goto_next_bacterium(self):
        """Navigate to next bacterium"""
        if self.bacteria_stats and self.current_bacteria_index < len(self.bacteria_stats) - 1:
            self.current_bacteria_index += 1
            self.update_bacteria_navigation_buttons()
            self.update_compact_statistics()
            self.update_preview()
    
    def goto_last_bacterium(self):
        """Navigate to last bacterium"""
        if self.bacteria_stats:
            self.current_bacteria_index = len(self.bacteria_stats) - 1
            self.update_bacteria_navigation_buttons()
            self.update_compact_statistics()
            self.update_preview()
    
    def update_bacteria_navigation_buttons(self):
        """Update navigation button states"""
        if not self.bacteria_stats:
            self.btn_first['state'] = 'disabled'
            self.btn_prev['state'] = 'disabled'
            self.btn_next['state'] = 'disabled'
            self.btn_last['state'] = 'disabled'
            self.nav_info_label.config(text="No bacteria detected")
            return
        
        total = len(self.bacteria_stats)
        current = self.current_bacteria_index + 1
        
        self.nav_info_label.config(text=f"Bacterium {current} of {total}")
        
        self.btn_first['state'] = 'normal' if self.current_bacteria_index > 0 else 'disabled'
        self.btn_prev['state'] = 'normal' if self.current_bacteria_index > 0 else 'disabled'
        self.btn_next['state'] = 'normal' if self.current_bacteria_index < total - 1 else 'disabled'
        self.btn_last['state'] = 'normal' if self.current_bacteria_index < total - 1 else 'disabled'
        
        # Expand navigation panel if bacteria are detected
        if self.nav_panel.is_collapsed:
            self.nav_panel.expand()
    
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
        
        # Find which bacterium was clicked in FILTERED bacteria_stats
        if self.bacteria_stats:
            for idx, stat in enumerate(self.bacteria_stats):
                if cv2.pointPolygonTest(stat['contour'], (img_x, img_y), False) >= 0:
                    self.current_bacteria_index = idx
                    self.update_bacteria_navigation_buttons()
                    break

        # Auto-tune ONLY if Ctrl key is held
        if event.state & 0x4:  # Ctrl key modifier
            self.auto_tune_from_point(img_x, img_y, pixel_value)
            self.update_preview()
        else:
            # Just update the preview without changing parameters
            self.update_preview()
    
    def clear_probe(self):
        """Clear probe visualization"""
        for canvas_id in self.probe_canvas_ids:
            self.canvas_original.delete(canvas_id)
        self.probe_canvas_ids = []
    
    def update_measurement_panel(self, x: int, y: int, pixel_value: int):
        """Update measurement panel with probe information"""
        self.measure_text.config(state="normal")
        self.measure_text.delete(1.0, tk.END)
        
        text = f"Position: ({x}, {y})\n"
        text += f"BF Intensity: {pixel_value}\n"
        
        if self.fluorescence_image is not None:
            fluor_value = int(self.fluorescence_image[y, x])
            text += f"Fluor Intensity: {fluor_value}\n"
        
        # Check if point is inside any bacterium
        for stat in self.bacteria_stats:
            if cv2.pointPolygonTest(stat['contour'], (x, y), False) >= 0:
                text += f"\n--- Bacterium #{stat['id']} ---\n"
                text += f"Area: {stat['area']:.1f} px²\n"
                text += f"Fluor Total: {stat['fluor_total']:.1f}\n"
                text += f"Fluor/Area: {stat['fluor_per_area']:.2f}\n"
                break
        
        self.measure_text.insert(1.0, text)
        self.measure_text.config(state="disabled")
    
    def auto_tune_from_point(self, x: int, y: int, pixel_value: int):
        """Auto-tune parameters based on clicked point"""
        # Simple auto-tuning logic
        if pixel_value < 100:
            # Dark region - might need lower threshold
            self.params['threshold_c'].set(max(-20, self.params['threshold_c'].get() - 2))
        elif pixel_value > 200:
            # Bright region - might need higher threshold
            self.params['threshold_c'].set(min(20, self.params['threshold_c'].get() + 2))
        
        self.status_var.set(f"Auto-tuned at ({x}, {y}), intensity={pixel_value}")
    
    def export_statistics(self):
        """Export statistics to CSV file"""
        if not self.bacteria_stats:
            messagebox.showwarning("Warning", "No bacteria detected to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Statistics",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    # Header
                    f.write("ID,Area,Perimeter,AspectRatio,Circularity,BF_Mean,Fluor_Total,Fluor_Mean,Fluor_Per_Area\n")
                    
                    # Data
                    for stat in self.bacteria_stats:
                        f.write(f"{stat['id']},{stat['area']:.2f},{stat['perimeter']:.2f},")
                        f.write(f"{stat['aspect_ratio']:.3f},{stat['circularity']:.3f},")
                        f.write(f"{stat['bf_mean']:.2f},{stat['fluor_total']:.2f},")
                        f.write(f"{stat['fluor_mean']:.2f},{stat['fluor_per_area']:.2f}\n")
                
                self.status_var.set(f"Exported statistics to {Path(file_path).name}")
                messagebox.showinfo("Success", f"Statistics exported to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export statistics:\n{e}")
    
    def save_settings(self):
        """Save current parameters to file"""
        settings = {key: var.get() for key, var in self.params.items()}
        
        try:
            with open("bacteria_segmentation_settings.json", 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Failed to save settings: {e}")
    
    def load_last_settings(self):
        """Load last saved parameters"""
        try:
            if Path("bacteria_segmentation_settings.json").exists():
                with open("bacteria_segmentation_settings.json", 'r') as f:
                    settings = json.load(f)
                    for key, value in settings.items():
                        if key in self.params:
                            self.params[key].set(value)
        except Exception as e:
            print(f"Failed to load settings: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        self.save_settings()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = BacteriaSegmentationApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()