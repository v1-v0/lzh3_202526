import os
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider, Button
from typing import Dict, List, Tuple, Optional, Any

from bacteria_configs import bacteria_configs
from bacteria_configs import SegmentationConfig
from datetime import datetime

class SegmentationTuner:
    """Interactive segmentation parameter tuner with matplotlib/tkinter GUI"""
    
    # Default parameters
    DEFAULT_PARAMS = {
        "gaussian_sigma": 2.0,
        "brightness_adjust": 0,
        "contrast_adjust": 1.0,
        "threshold_offset": 0,
        "min_area": 20,
        "max_area": 5000,
        "dilate_iterations": 0,
        "erode_iterations": 0,
    }
    
    # GUI styling constants
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
    
    def __init__(self, image_path: str, bacterium: str, structure: str, mode: str):
        """
        Initialize the segmentation tuner
        
        Args:
            image_path: Path to the image file
            bacterium: Name of the bacterium
            structure: Structure type ('bacteria' or 'inclusions')
            mode: Segmentation mode ('DARK' or 'BRIGHT')
        """
        self.image_path = image_path
        self.bacterium = bacterium
        self.structure = structure
        self.mode = mode
        
        # Load image
        self.original_image = self._load_image(image_path)
        
        # Initialize parameters and state
        self._initialize_parameters()
        
        # Initialize processing results
        self.processed_image: np.ndarray = np.zeros_like(self.original_image)
        self.binary_mask: np.ndarray = np.zeros_like(self.original_image)
        self.contours: List[np.ndarray] = []
        self.contour_areas: List[float] = []
        self.current_suggestions: Dict[str, Any] = {}
        
        # Initialize GUI components
        self.sliders: Dict[str, Slider] = {}
        self.param_labels: Dict[str, tk.Label] = {}
        
        # Setup GUI
        self.setup_gui()
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and validate image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        print(f"✅ Loaded image: {os.path.basename(image_path)}")
        print(f"   Shape: {image.shape}, Dtype: {image.dtype}")
        return image
    
    def _initialize_parameters(self):
        """Initialize parameters from config or defaults"""
        self.params = self.DEFAULT_PARAMS.copy()
        self.invert_image = False
        
        # Try to load from existing config
        config_key = self.bacterium
        if config_key in bacteria_configs:
            self._load_from_config(bacteria_configs[config_key])
        else:
            print(f"ℹ No existing config for '{self.bacterium}', using defaults")
    
    def _load_from_config(self, config: Dict[str, Any]):
        """Load parameters from bacteria config"""
        print(f"📂 Loading existing configuration for '{self.bacterium}'")
        
        # Determine parameter key
        if self.structure == "bacteria":
            params_key = "bacteria_segmentation"
        else:
            params_key = f"inclusion_{self.mode.lower()}_segmentation"
        
        if params_key not in config:
            print(f"   ⚠ No {params_key} found, using defaults")
            return
        
        loaded_params = config[params_key]
        print(f"   ✓ Found {params_key} parameters")
        
        # Update parameters
        for key in self.params.keys():
            if key in loaded_params:
                self.params[key] = loaded_params[key]
        
        self.invert_image = loaded_params.get("invert_image", False)
        
        print(f"   ✓ Loaded parameters:")
        for key, value in self.params.items():
            print(f"      - {key}: {value}")
        print(f"      - invert_image: {self.invert_image}")
    
    def setup_gui(self):
        """Setup the matplotlib/tkinter GUI"""
        self.root = tk.Tk()
        self.root.title(f"Segmentation Tuner - {self.bacterium}")
        self.root.geometry("1920x1080")
        self.root.minsize(1600, 900)
        self.root.configure(bg=self.COLORS['bg'])
        
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Build GUI sections
        self._create_header(main_container)
        content_frame = self._create_content_area(main_container)
        self._create_control_panel(main_container)
        self._create_action_buttons(main_container)
        
        print("✅ GUI Setup Complete")
        
        # Initial update
        self.update_visualization()
    
    def _create_header(self, parent: ttk.Frame):
        """Create header section"""
        header_frame = tk.Frame(parent, bg=self.COLORS['header'], height=45)
        header_frame.pack(fill=tk.X, pady=(0, 5))
        header_frame.pack_propagate(False)
        
        # Title
        title_text = f"🔬 {self.bacterium} - {self.structure}"
        tk.Label(
            header_frame,
            text=title_text,
            font=("Segoe UI", 14, "bold"),
            bg=self.COLORS['header'],
            fg="white"
        ).pack(side=tk.LEFT, padx=20, pady=6)
        
        # Mode badge
        mode_text = f"Mode: {self.mode} {'(Inverted)' if self.invert_image else ''}"
        tk.Label(
            header_frame,
            text=mode_text,
            font=("Segoe UI", 9),
            bg=self.COLORS['secondary'],
            fg="white",
            padx=12,
            pady=4,
            relief=tk.RAISED
        ).pack(side=tk.LEFT, pady=6)
        
        # Load Image button
        tk.Button(
            header_frame,
            text="📁 LOAD IMAGE",
            font=("Segoe UI", 9, "bold"),
            bg=self.COLORS['primary'],
            fg="white",
            padx=12,
            pady=4,
            relief=tk.RAISED,
            command=self.load_new_image,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=10, pady=6)
        
        # Contour count
        self.contour_count_label = tk.Label(
            header_frame,
            text="Contours: 0",
            font=("Segoe UI", 9),
            bg=self.COLORS['success'],
            fg="white",
            padx=10,
            pady=4,
            relief=tk.RAISED
        )
        self.contour_count_label.pack(side=tk.RIGHT, padx=20, pady=6)
    
    def _create_content_area(self, parent: ttk.Frame) -> ttk.Frame:
        """Create main content area"""
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Image display
        self._create_image_panel(content_frame)
        
        # Right panel - Controls
        self._create_right_panel(content_frame)
        
        return content_frame
    
    def _create_image_panel(self, parent: ttk.Frame):
        """Create left image panel"""
        left_panel = ttk.Frame(parent, relief=tk.RIDGE, borderwidth=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Header
        header = tk.Frame(left_panel, bg=self.COLORS['secondary'], height=28)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="📷 IMAGE ANALYSIS - Original + Contours",
            font=("Segoe UI", 10, "bold"),
            bg=self.COLORS['secondary'],
            fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=4)
        
        # Canvas
        canvas_frame = ttk.Frame(left_panel)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_image = Figure(figsize=(13, 9.5), facecolor='white')
        self.ax_image = self.fig_image.add_subplot(111)
        self.canvas_image = FigureCanvasTkAgg(self.fig_image, canvas_frame)
        self.canvas_image.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_image.mpl_connect("button_press_event", self.on_image_click)
        
        # Instruction
        instruction = tk.Frame(left_panel, bg=self.COLORS['primary'], height=25)
        instruction.pack(fill=tk.X)
        instruction.pack_propagate(False)
        
        tk.Label(
            instruction,
            text="💡 Click on a particle to analyze and get parameter suggestions",
            font=("Segoe UI", 8),
            bg=self.COLORS['primary'],
            fg="white"
        ).pack(pady=3)
    
    def _create_right_panel(self, parent: ttk.Frame):
        """Create right control panel"""
        right_panel = ttk.Frame(parent, width=380)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        # Parameters section
        self._create_parameters_section(right_panel)
        
        # Target analysis section
        self._create_target_analysis_section(right_panel)
        
        # Histogram section
        self._create_histogram_section(right_panel)
    
    def _create_parameters_section(self, parent: ttk.Frame):
        """Create parameters display section"""
        # Header
        header = tk.Frame(parent, bg=self.COLORS['secondary'], height=28)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="⚙️ PARAMETERS",
            font=("Segoe UI", 10, "bold"),
            bg=self.COLORS['secondary'],
            fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=4)
        
        # Display area
        display = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        display.pack(fill=tk.X, padx=5, pady=(5, 0))
        
        inner = ttk.Frame(display)
        inner.pack(fill=tk.X, padx=8, pady=8)
        
        # Basic info
        self._add_param_section(inner, "Basic Information", [
            ("Pathogen:", self.bacterium, self.COLORS['danger']),
            ("Structure:", self.structure, self.COLORS['purple']),
            ("Mode:", f"{self.mode} particles", self.COLORS['primary']),
        ])
        
        ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        
        # Preprocessing
        self._add_param_section(inner, "Preprocessing", [
            ("Invert:", "ON" if self.invert_image else "OFF",
             self.COLORS['success'] if self.invert_image else self.COLORS['gray']),
            ("Gaussian σ:", f"{self.params['gaussian_sigma']:.1f}", None),
            ("Brightness:", f"{self.params['brightness_adjust']:+.0f}", None),
            ("Contrast:", f"{self.params['contrast_adjust']:.2f}", None),
            ("Threshold Δ:", f"{self.params['threshold_offset']:+.0f}", None),
        ])
        
        ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        
        # Filtering
        self._add_param_section(inner, "Filtering & Morphology", [
            ("Min area:", f"{self.params['min_area']:.0f} px", None),
            ("Max area:", f"{self.params['max_area']:.0f} px", None),
            ("Dilate iter:", str(self.params["dilate_iterations"]), None),
            ("Erode iter:", str(self.params["erode_iterations"]), None),
        ])
    
    def _add_param_section(self, parent: ttk.Frame, title: str,
                           params: List[Tuple[str, str, Optional[str]]]):
        """Add a parameter display section"""
        # Title
        tk.Label(
            parent,
            text=title,
            font=("Segoe UI", 9, "bold"),
            foreground=self.COLORS['header']
        ).pack(anchor="w", pady=(0, 4))
        
        # Parameters
        for label_text, value_text, color in params:
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=1)
            
            tk.Label(
                frame,
                text=label_text,
                font=("Segoe UI", 8),
                foreground="gray",
                width=15,
                anchor="w"
            ).pack(side=tk.LEFT)
            
            if color:
                value_label = tk.Label(
                    frame,
                    text=value_text,
                    font=("Segoe UI", 8, "bold"),
                    foreground="white",
                    bg=color,
                    padx=6,
                    pady=1,
                    relief=tk.RAISED
                )
            else:
                value_label = tk.Label(
                    frame,
                    text=value_text,
                    font=("Segoe UI", 8, "bold"),
                    foreground=self.COLORS['header']
                )
            value_label.pack(side=tk.LEFT)
            
            # Store reference
            self.param_labels[label_text] = value_label
    
    def _create_target_analysis_section(self, parent: ttk.Frame):
        """Create target analysis section"""
        header = tk.Frame(parent, bg=self.COLORS['warning'], height=26)
        header.pack(fill=tk.X, pady=(8, 0))
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="🎯 TARGET ANALYSIS",
            font=("Segoe UI", 9, "bold"),
            bg=self.COLORS['warning'],
            fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=3)
        
        display = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        display.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        self.target_analysis_label = tk.Label(
            display,
            text="Click on image to analyze a particle",
            font=("Segoe UI", 8),
            foreground="gray",
            justify=tk.LEFT,
            wraplength=360,
            bg="white",
            anchor="w",
            padx=8,
            pady=8
        )
        self.target_analysis_label.pack(fill=tk.BOTH, expand=True)
    
    def export_complete_config(self):
        """Export comprehensive configuration for dev2.py compatibility"""
        
        
        
        if not self.bacterium or not self.structure or not self.mode:
            messagebox.showerror("Error", "Please set bacterium, structure, and mode first")
            return
        
        # Get current segmentation parameters
        seg_params = {
            'invert_image': self.invert_image,
            'gaussian_sigma': self.params['gaussian_sigma'],
            'brightness_adjust': self.params['brightness_adjust'],
            'contrast_adjust': self.params['contrast_adjust'],
            'threshold_offset': self.params['threshold_offset'],
            'min_area': self.params['min_area'],
            'max_area': self.params['max_area'],
            'dilate_iterations': self.params['dilate_iterations'],
            'erode_iterations': self.params['erode_iterations'],
        }
        
        # Calculate derived parameters for dev2.py
        # Assuming 0.109 µm/px (update if you have actual pixel size)
        um_per_px = 0.109
        um2_per_px2 = um_per_px ** 2
        
        min_area_um2 = seg_params['min_area'] * um2_per_px2
        max_area_um2 = seg_params['max_area'] * um2_per_px2
        
        # Build complete configuration
        complete_config = {
            "bacterium": self.bacterium,
            "structure": self.structure,
            "mode": self.mode,
            "export_version": "2.0",
            "compatible_with": "dev2.py",
            "pixel_size_um": um_per_px,
            
            # Original feedback_tuner parameters
            "parameters": seg_params,
            
            # Extended parameters for dev2.py
            "extended_parameters": {
                # Size filtering (in µm²)
                "min_area_um2": min_area_um2,
                "max_area_um2": max_area_um2,
                
                # Morphological operations
                "morph_kernel_size": 3,
                "morph_iterations": 1,
                "dilate_iterations": seg_params['dilate_iterations'],
                "erode_iterations": seg_params['erode_iterations'],
                
                # Shape filters (reasonable defaults)
                "min_circularity": 0.0,
                "max_circularity": 1.0,
                "min_aspect_ratio": 0.2,
                "max_aspect_ratio": 10.0,
                "min_solidity": 0.3,
                
                # Intensity filters
                "min_mean_intensity": 0,
                "max_mean_intensity": 255,
                "max_edge_gradient": 200,
                
                # Image processing
                "max_fraction_of_image": 0.25,
                
                # Fluorescence parameters
                "fluor_gaussian_sigma": 1.5,
                "fluor_morph_kernel_size": 3,
                "fluor_min_area_um2": 3.0,
                "fluor_match_min_intersection_px": 5.0,
            },
            
            # Metadata
            "notes": f"Exported from Feedback Tuner on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "tuner_version": "1.0",
        }
        
        # Save file
        default_name = f"segmentation_params_{self.bacterium}_{self.structure}_{self.mode}_COMPLETE.json"
        file_path = filedialog.asksaveasfilename(
            title="Export Complete Configuration",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(complete_config, f, indent=2)
                
                messagebox.showinfo(
                    "Success",
                    f"Complete configuration exported:\n{os.path.basename(file_path)}\n\n"
                    f"Compatible with dev2.py\n"
                    f"Includes {len(complete_config['extended_parameters'])} extended parameters"
                )
                print(f"✓ Exported complete config: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export:\n{e}")







    def _create_histogram_section(self, parent: ttk.Frame):
        """Create histogram section"""
        header = tk.Frame(parent, bg=self.COLORS['info'], height=26)
        header.pack(fill=tk.X, pady=(5, 0))
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="📊 AREA DISTRIBUTION",
            font=("Segoe UI", 9, "bold"),
            bg=self.COLORS['info'],
            fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=3)
        
        canvas_frame = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        
        self.fig_hist = Figure(figsize=(5, 4), facecolor='white')
        self.ax_hist = self.fig_hist.add_subplot(111)
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, canvas_frame)
        self.canvas_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    def _create_control_panel(self, parent: ttk.Frame):
        """Create slider control panel"""
        panel = ttk.Frame(parent)
        panel.pack(fill=tk.X, pady=(5, 0))
        
        # Header
        header = tk.Frame(panel, bg=self.COLORS['secondary'], height=28)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="🎚️ ADJUST PARAMETERS",
            font=("Segoe UI", 10, "bold"),
            bg=self.COLORS['secondary'],
            fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=4)
        
        # Sliders
        slider_frame = ttk.Frame(panel, relief=tk.SUNKEN, borderwidth=1)
        slider_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        self.fig_sliders = Figure(figsize=(18, 2.2), facecolor='#f8f9fa')
        self.canvas_sliders = FigureCanvasTkAgg(self.fig_sliders, slider_frame)
        self.canvas_sliders.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self._create_sliders()
    
    def _create_sliders(self):
        """Create parameter sliders"""
        # Headers
        self.fig_sliders.text(0.195, 0.95, 'PREPROCESSING', ha='center', va='top',
                             fontsize=9, fontweight='bold', color=self.COLORS['header'])
        self.fig_sliders.text(0.695, 0.95, 'FILTERING & MORPHOLOGY', ha='center', va='top',
                             fontsize=9, fontweight='bold', color=self.COLORS['header'])
        
        # Slider configurations
        slider_configs = {
            'left': [
                ("gaussian_sigma", "Gaussian σ", 0.5, 10.0, 0.72, 0.04),
                ("brightness_adjust", "Brightness", -100, 100, 0.51, 0.04),
                ("contrast_adjust", "Contrast", 0.5, 2.0, 0.30, 0.04),
                ("threshold_offset", "Threshold Δ", -50, 50, 0.09, 0.04),
            ],
            'right': [
                ("min_area", "Min Area (px)", 10, 2000, 0.72, 0.51),
                ("max_area", "Max Area (px)", 100, 10000, 0.51, 0.51),
                ("dilate_iterations", "Dilate", 0, 5, 0.30, 0.51),
                ("erode_iterations", "Erode", 0, 5, 0.09, 0.51),
            ]
        }
        
        slider_height = 0.13
        slider_width = 0.38
        
        # Create left sliders (preprocessing)
        for param_key, label, vmin, vmax, y_pos, x_start in slider_configs['left']:
            ax = self.fig_sliders.add_axes((x_start, y_pos, slider_width, slider_height))
            valstep = 1 if "Brightness" in label or "Threshold" in label else None
            slider = Slider(ax, label, vmin, vmax, valinit=self.params[param_key],
                          valstep=valstep, color=self.COLORS['primary'])
            slider.on_changed(lambda val, key=param_key: self.update_parameter(key, val))
            self.sliders[param_key] = slider
        
        # Create right sliders (filtering)
        for param_key, label, vmin, vmax, y_pos, x_start in slider_configs['right']:
            ax = self.fig_sliders.add_axes((x_start, y_pos, slider_width, slider_height))
            valstep = 1 if "iter" in param_key else None
            slider = Slider(ax, label, vmin, vmax, valinit=self.params[param_key],
                          valstep=valstep, color=self.COLORS['info'])
            slider.on_changed(lambda val, key=param_key: self.update_parameter(key, val))
            self.sliders[param_key] = slider
        
        # Action buttons
        invert_color = self.COLORS['success'] if self.invert_image else self.COLORS['gray']
        invert_text = f'INVERT\n{"ON" if self.invert_image else "OFF"}'
        
        ax_invert = self.fig_sliders.add_axes((0.915, 0.51, 0.07, 0.34))
        self.btn_invert = Button(ax_invert, invert_text, color=invert_color,
                                hovercolor=self.COLORS['success'])
        self.btn_invert.on_clicked(self.toggle_invert)
        
        ax_apply = self.fig_sliders.add_axes((0.915, 0.09, 0.07, 0.34))
        self.btn_apply = Button(ax_apply, "APPLY\nSUGGESTIONS",
                               color=self.COLORS['primary'],
                               hovercolor='#5dade2')
        self.btn_apply.on_clicked(self.apply_suggestions)
    

    
    def process_image(self):
        """Process image with current parameters"""
        img = self.original_image.copy()
        
        # Invert if needed
        if self.invert_image:
            img = cv2.bitwise_not(img)
        
        # Brightness and contrast
        img = cv2.convertScaleAbs(
            img,
            alpha=self.params["contrast_adjust"],
            beta=self.params["brightness_adjust"]
        )
        
        # Gaussian blur
        if self.params["gaussian_sigma"] > 0:
            ksize = int(2 * np.ceil(2 * self.params["gaussian_sigma"]) + 1)
            img = cv2.GaussianBlur(img, (ksize, ksize), self.params["gaussian_sigma"])
        
        self.processed_image = img
        
        # Thresholding
        threshold_value = float(img.mean()) + self.params["threshold_offset"]
        thresh_type = cv2.THRESH_BINARY_INV if self.mode == "DARK" else cv2.THRESH_BINARY
        _, binary = cv2.threshold(img, threshold_value, 255, thresh_type)
        
        # Morphological operations
        if self.params["dilate_iterations"] > 0:
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.dilate(binary, kernel,
                              iterations=int(self.params["dilate_iterations"]))
        
        if self.params["erode_iterations"] > 0:
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.erode(binary, kernel,
                             iterations=int(self.params["erode_iterations"]))
        
        self.binary_mask = binary
        
        # Find and filter contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.contours = []
        self.contour_areas = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.params["min_area"] <= area <= self.params["max_area"]:
                self.contours.append(cnt)
                self.contour_areas.append(area)
    
    def update_visualization(self):
        """Update all visualizations"""
        self.process_image()
        
        # Update image display
        self._update_image_display()
        
        # Update histogram
        self._update_histogram()
        
        # Update parameter displays
        self._update_param_displays()
        
        # Update contour count
        self.contour_count_label.config(text=f"Contours: {len(self.contours)}")
    
    def _update_image_display(self):
        """Update main image display"""
        self.ax_image.clear()
        
        # Convert to RGB and draw contours
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
            areas = np.array(self.contour_areas)
            
            self.ax_hist.hist(areas, bins=30, color=self.COLORS['primary'],
                            alpha=0.7, edgecolor='black')
            
            # Statistics
            median = float(np.median(areas))
            mean = float(np.mean(areas))
            min_area = float(np.min(areas))
            max_area = float(np.max(areas))
            
            self.ax_hist.axvline(median, color='orange', linestyle='--',
                               linewidth=2, label=f'Median: {median:.1f}')
            self.ax_hist.axvline(mean, color='red', linestyle='--',
                               linewidth=2, label=f'Mean: {mean:.1f}')
            self.ax_hist.axvline(min_area, color='green', linestyle=':',
                               linewidth=1.5, label=f'Min: {min_area:.0f}')
            self.ax_hist.axvline(max_area, color='purple', linestyle=':',
                               linewidth=1.5, label=f'Max: {max_area:.0f}')
            
            self.ax_hist.set_xlabel("Area (pixels)", fontsize=9)
            self.ax_hist.set_ylabel("Count", fontsize=9)
            self.ax_hist.set_title(f"Distribution (n={len(areas)})",
                                  fontsize=10, fontweight='bold')
            self.ax_hist.legend(fontsize=7, loc='upper right')
            self.ax_hist.grid(True, alpha=0.3)
        
        self.canvas_hist.draw()
    
    def _update_param_displays(self):
        """Update parameter display values"""
        # Update invert
        invert_label = self.param_labels["Invert:"]
        invert_text = "ON" if self.invert_image else "OFF"
        invert_color = self.COLORS['success'] if self.invert_image else self.COLORS['gray']
        invert_label.config(text=invert_text, bg=invert_color)
        
        # Update numeric parameters
        updates = {
            "Gaussian σ:": f"{self.params['gaussian_sigma']:.1f}",
            "Brightness:": f"{self.params['brightness_adjust']:+.0f}",
            "Contrast:": f"{self.params['contrast_adjust']:.2f}",
            "Threshold Δ:": f"{self.params['threshold_offset']:+.0f}",
            "Min area:": f"{self.params['min_area']:.0f} px",
            "Max area:": f"{self.params['max_area']:.0f} px",
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
    
    def update_sliders_from_params(self):
        """Update slider positions to match current parameters"""
        for param_key, slider in self.sliders.items():
            if param_key in self.params:
                slider.set_val(self.params[param_key])
        
        # Update invert button
        if hasattr(self, 'btn_invert'):
            invert_color = self.COLORS['success'] if self.invert_image else self.COLORS['gray']
            invert_text = f'INVERT\n{"ON" if self.invert_image else "OFF"}'
            self.btn_invert.color = invert_color
            self.btn_invert.label.set_text(invert_text)
        
        print("✅ Sliders updated from parameters")
    
    def toggle_invert(self, event):
        """Toggle image inversion"""
        self.invert_image = not self.invert_image
        
        # Update button appearance
        invert_text = f'INVERT\n{"ON" if self.invert_image else "OFF"}'
        invert_color = self.COLORS['success'] if self.invert_image else self.COLORS['gray']
        self.btn_invert.label.set_text(invert_text)
        self.btn_invert.color = invert_color
        
        self.update_visualization()
    
    def on_image_click(self, event):
        """Handle click on image to analyze particle"""
        if event.inaxies != self.ax_image or event.xdata is None or event.ydata is None:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        
        # Find clicked contour
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
        
        # Analyze particle
        self._analyze_particle(clicked_contour)
    
    def _analyze_particle(self, contour: np.ndarray):
        """Analyze a specific particle and generate suggestions"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Generate suggestions
        suggestions = self._generate_suggestions(area, circularity, aspect_ratio)
        self.current_suggestions = suggestions
        
        # Display analysis
        analysis_text = (
            f"🎯 Target Particle Analysis:\n\n"
            f"Area: {area:.1f} px²\n"
            f"Perimeter: {perimeter:.1f} px\n"
            f"Aspect Ratio: {aspect_ratio:.2f}\n"
            f"Circularity: {circularity:.3f}\n\n"
            f"📊 Suggestions:\n"
        )
        
        for param, value in suggestions.items():
            analysis_text += f"• {param}: {value}\n"
        
        self.target_analysis_label.config(text=analysis_text, foreground="black")
        
        print(f"\n🎯 Particle Analysis:")
        print(f"   Area: {area:.1f} px², Circularity: {circularity:.3f}")
        print(f"   Suggestions: {suggestions}")
    
    def _generate_suggestions(self, area: float, circularity: float,
                             aspect_ratio: float) -> Dict[str, Any]:
        """Generate parameter suggestions based on particle analysis"""
        suggestions: Dict[str, Any] = {}
        
        # Area-based suggestions
        suggestions["min_area"] = max(10, int(area * 0.3))
        suggestions["max_area"] = min(10000, int(area * 3.0))
        
        # Circularity-based suggestions
        if circularity < 0.6:
            suggestions["dilate_iterations"] = min(3, self.params["dilate_iterations"] + 1)
            suggestions["erode_iterations"] = min(3, self.params["erode_iterations"] + 1)
        
        # Aspect ratio-based suggestions
        if aspect_ratio > 2.5 or aspect_ratio < 0.4:
            suggestions["gaussian_sigma"] = min(5.0, self.params["gaussian_sigma"] + 0.5)
        
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
            self.original_image = self._load_image(file_path)
            self.image_path = file_path
            
            # Update visualization with current parameters
            self.update_visualization()
            
            messagebox.showinfo("Success",
                              f"Image loaded!\n\n{os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            print(f"❌ Error loading image: {e}")
    
    def save(self, event=None) -> bool:
        """Save parameters to JSON"""
        try:
            config = {
                "bacterium": self.bacterium,
                "structure": self.structure,
                "mode": self.mode,
                "parameters": {
                    "invert_image": self.invert_image,
                    **self.params
                }
            }
            
            filename = f"segmentation_params_{self.bacterium}_{self.structure}_{self.mode}.json"
            
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"\n💾 Parameters saved to: {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving parameters: {e}")
            return False
    
    def save_and_apply(self, event=None):
        
        """Save parameters and update bacteria_configs.py file"""
        if not self.save():
            messagebox.showerror("Error", "Failed to save parameters")
            return
        
        # Update bacteria_configs.py file
        try:
            from bacteria_configs import SegmentationConfig
            
            # Calculate area conversion (pixels² to µm²)
            um2_per_px2 = 0.012  # Adjust if your pixel size is different
            
            # Create SegmentationConfig from tuned parameters
            config = SegmentationConfig(
                name=f"{self.bacterium}",
                description=f"{self.structure} segmentation - Tuned {datetime.now().strftime('%Y-%m-%d')}",
                
                # Core segmentation
                gaussian_sigma=float(self.params['gaussian_sigma']),
                
                # Size filtering (convert pixels² to µm²)
                min_area_um2=float(self.params['min_area']) * um2_per_px2,
                max_area_um2=float(self.params['max_area']) * um2_per_px2,
                
                # Morphology
                dilate_iterations=int(self.params['dilate_iterations']),
                erode_iterations=int(self.params['erode_iterations']),
                morph_kernel_size=3,
                morph_iterations=1,
                
                # Shape filters (keep defaults)
                min_circularity=0.0,
                max_circularity=1.0,
                min_aspect_ratio=0.2,
                max_aspect_ratio=10.0,
                
                # Intensity filters (keep defaults)
                min_mean_intensity=0,
                max_mean_intensity=255,
                max_edge_gradient=200,
                
                # Other
                min_solidity=0.3,
                max_fraction_of_image=0.25,
                
                # Fluorescence
                fluor_min_area_um2=3.0,
                fluor_match_min_intersection_px=5.0,
            )
            
            # Write to bacteria_configs.py
            self._write_config_to_file(config)
            
            print(f"\n✅ Configuration saved and applied: {self.bacterium}")
            messagebox.showinfo("Success", 
                f"Parameters saved and applied!\n\n"
                f"Configuration updated in bacteria_configs.py\n"
                f"Next run of dev2.py will use these parameters automatically.")
            
        except Exception as e:
            print(f"❌ Error updating bacteria_configs.py: {e}")
            messagebox.showerror("Error", 
                f"Parameters saved to JSON but failed to update bacteria_configs.py:\n{e}")


    def _write_config_to_file(self, config: 'SegmentationConfig') -> None:
        """Write configuration to bacteria_configs.py file
        
        Args:
            config: SegmentationConfig object to write
        """
        from pathlib import Path
        
        config_file = Path(__file__).parent / "bacteria_configs.py"
        
        # Read current file
        with open(config_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find the config block to replace
        bacteria_key = self.bacterium.upper().replace(' ', '_')
        start_marker = f"{bacteria_key} = SegmentationConfig("
        end_marker = ")"
        
        # Find start and end indices
        start_idx = None
        end_idx = None
        indent_level = 0
        
        for i, line in enumerate(lines):
            if start_marker in line:
                start_idx = i
                indent_level = len(line) - len(line.lstrip())
            elif start_idx is not None:
                # Count parentheses to find matching closing paren
                indent_level += line.count('(') - line.count(')')
                if indent_level == 0:
                    end_idx = i
                    break
        
        # Generate new config block
        new_config_lines = self._generate_config_block(config, bacteria_key)
        
        if start_idx is not None and end_idx is not None:
            # Replace existing config
            lines[start_idx:end_idx+1] = new_config_lines
            print(f"  ✓ Updated existing {bacteria_key} configuration")
        else:
            # Add new config before DEFAULT
            default_idx = None
            for i, line in enumerate(lines):
                if "DEFAULT = SegmentationConfig(" in line:
                    default_idx = i
                    break
            
            if default_idx:
                lines[default_idx:default_idx] = new_config_lines + ["\n"]
                print(f"  ✓ Added new {bacteria_key} configuration")
            else:
                print(f"  ⚠ Could not find insertion point, appending to end")
                lines.extend(["\n"] + new_config_lines)
        
        # Write back to file
        with open(config_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"  ✓ bacteria_configs.py updated successfully")


    def _generate_config_block(self, config: 'SegmentationConfig', var_name: str) -> list[str]:
        """Generate Python code for SegmentationConfig
        
        Args:
            config: SegmentationConfig object
            var_name: Variable name (e.g., 'KLEBSIELLA_PNEUMONIAE')
            
        Returns:
            List of code lines
        """
        lines = [
            f"{var_name} = SegmentationConfig(\n",
            f'    name="{config.name}",\n',
            f'    description="{config.description}",\n',
            f'    \n',
            f'    # Core segmentation\n',
            f'    gaussian_sigma={config.gaussian_sigma:.2f},\n',
            f'    \n',
            f'    # Size filtering (in µm²)\n',
            f'    min_area_um2={config.min_area_um2:.2f},\n',
            f'    max_area_um2={config.max_area_um2:.2f},\n',
            f'    \n',
            f'    # Morphology\n',
            f'    dilate_iterations={config.dilate_iterations},\n',
            f'    erode_iterations={config.erode_iterations},\n',
            f'    morph_kernel_size={config.morph_kernel_size},\n',
            f'    morph_iterations={config.morph_iterations},\n',
            f'    \n',
            f'    # Shape filters\n',
            f'    min_circularity={config.min_circularity:.2f},\n',
            f'    max_circularity={config.max_circularity:.2f},\n',
            f'    min_aspect_ratio={config.min_aspect_ratio:.2f},\n',
            f'    max_aspect_ratio={config.max_aspect_ratio:.2f},\n',
            f'    \n',
            f'    # Intensity filters\n',
            f'    min_mean_intensity={config.min_mean_intensity:.0f},\n',
            f'    max_mean_intensity={config.max_mean_intensity:.0f},\n',
            f'    max_edge_gradient={config.max_edge_gradient:.0f},\n',
            f'    \n',
            f'    # Other\n',
            f'    min_solidity={config.min_solidity:.2f},\n',
            f'    max_fraction_of_image={config.max_fraction_of_image:.2f},\n',
            f'    \n',
            f'    # Fluorescence\n',
            f'    fluor_min_area_um2={config.fluor_min_area_um2:.2f},\n',
            f'    fluor_match_min_intersection_px={config.fluor_match_min_intersection_px:.2f},\n',
            f')\n',
        ]
        
        return lines


    def _create_action_buttons(self, parent: ttk.Frame):
        """Create bottom action buttons"""
        action_frame = tk.Frame(parent, bg=self.COLORS['header'], height=55)
        action_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        action_frame.pack_propagate(False)
        
        button_container = tk.Frame(action_frame, bg=self.COLORS['header'])
        button_container.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        style = ttk.Style()
        style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), padding=10)
        
        # Define buttons - conditionally include Back button
        buttons = [
            ("⬅ BACK", self.back, 13, None),
            ("💾 SAVE JSON", self.save, 15, None),
            ("📋 EXPORT COMPLETE", self.export_complete_config, 20, None),  # ← NEW
            ("✅ SAVE & APPLY", self.save_and_apply, 18, self.COLORS['success']),
            ("❌ QUIT", self.quit, 13, None),
        ]
        

        
        for text, command, width, highlight in buttons:
            if highlight:
                frame = tk.Frame(button_container, bg=highlight, bd=3, relief=tk.RAISED)
                frame.pack(side=tk.LEFT, padx=6)
                btn = ttk.Button(frame, text=text, command=command, width=width,
                            style="Action.TButton")
                btn.pack(padx=2, pady=2)
            else:
                btn = ttk.Button(button_container, text=text, command=command,
                            width=width, style="Action.TButton")
                btn.pack(side=tk.LEFT, padx=5)

    def _can_launch_main_menu(self) -> bool:
        """Check if main menu can be launched"""
        try:
            import importlib.util
            return importlib.util.find_spec("pathogen_config_manager") is not None
        except:
            return False

    def back(self):
        """Close the tuner"""
        if messagebox.askyesno("Confirm", "Close tuner?\n\nUnsaved changes will be lost."):
            print("🔙 Closing tuner...")
            self.root.destroy()





    def quit(self, event=None):
        """Quit application"""
        if messagebox.askyesno("Confirm", "Quit application?"):
            print("\n❌ Exiting application")
            self.root.quit()
            self.root.destroy()
    
    def run(self):
        """Start the GUI"""
        print(f"\n🚀 Starting Segmentation Tuner")
        print(f"   Bacterium: {self.bacterium}")
        print(f"   Structure: {self.structure}")
        print(f"   Mode: {self.mode}")
        self.root.mainloop()


def launch_tuner_with_setup():
    """Launch tuner with interactive setup dialog"""
    setup_root = tk.Tk()
    setup_root.title("Segmentation Tuner Setup")
    setup_root.geometry("500x400")
    setup_root.resizable(False, False)
    
    style = ttk.Style()
    style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"))
    style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"))
    
    # Variables
    image_path_var = tk.StringVar()
    bacterium_var = tk.StringVar()
    structure_var = tk.StringVar(value="bacteria")
    mode_var = tk.StringVar(value="DARK")
    
    # Main frame
    main_frame = ttk.Frame(setup_root, padding=20)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title
    ttk.Label(main_frame, text="🔬 Segmentation Tuner Setup",
             style="Title.TLabel").pack(pady=(0, 20))
    
    # Image selection
    ttk.Label(main_frame, text="1. Select Image",
             style="Section.TLabel").pack(anchor="w", pady=(0, 5))
    
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
    
    # Bacterium selection
    ttk.Label(main_frame, text="2. Select Bacterium",
             style="Section.TLabel").pack(anchor="w", pady=(0, 5))
    
    bacterium_combo = ttk.Combobox(
        main_frame,
        textvariable=bacterium_var,
        values=list(bacteria_configs.keys()),
        width=37,
        state="readonly"
    )
    bacterium_combo.pack(fill=tk.X, pady=(0, 15))
    if bacteria_configs:
        bacterium_combo.current(0)
    
    # Structure selection
    ttk.Label(main_frame, text="3. Select Structure",
             style="Section.TLabel").pack(anchor="w", pady=(0, 5))
    
    structure_frame = ttk.Frame(main_frame)
    structure_frame.pack(fill=tk.X, pady=(0, 15))
    
    ttk.Radiobutton(structure_frame, text="Bacteria",
                   variable=structure_var, value="bacteria").pack(side=tk.LEFT, padx=(0, 20))
    ttk.Radiobutton(structure_frame, text="Inclusions",
                   variable=structure_var, value="inclusions").pack(side=tk.LEFT)
    
    # Mode selection
    ttk.Label(main_frame, text="4. Select Mode",
             style="Section.TLabel").pack(anchor="w", pady=(0, 5))
    
    mode_frame = ttk.Frame(main_frame)
    mode_frame.pack(fill=tk.X, pady=(0, 20))
    
    ttk.Radiobutton(mode_frame, text="DARK particles",
                   variable=mode_var, value="DARK").pack(side=tk.LEFT, padx=(0, 20))
    ttk.Radiobutton(mode_frame, text="BRIGHT particles",
                   variable=mode_var, value="BRIGHT").pack(side=tk.LEFT)
    
    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X)
    
    def start_tuner():
        if not image_path_var.get():
            messagebox.showerror("Error", "Please select an image file")
            return
        if not bacterium_var.get():
            messagebox.showerror("Error", "Please select a bacterium")
            return
        
        setup_root.destroy()
        
        try:
            tuner = SegmentationTuner(
                image_path=image_path_var.get(),
                bacterium=bacterium_var.get(),
                structure=structure_var.get(),
                mode=mode_var.get()
            )
            tuner.run()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start tuner:\n{str(e)}")
    
    ttk.Button(button_frame, text="❌ Cancel",
              command=setup_root.destroy).pack(side=tk.LEFT, padx=(0, 10))
    ttk.Button(button_frame, text="✅ Start Tuner",
              command=start_tuner).pack(side=tk.LEFT)
    
    setup_root.mainloop()


def launch_tuner(image_path: str, bacterium: str, structure: str, mode: str):
    """Launch the segmentation tuner directly"""
    tuner = SegmentationTuner(image_path, bacterium, structure, mode)
    tuner.run()


if __name__ == "__main__":
    launch_tuner_with_setup()