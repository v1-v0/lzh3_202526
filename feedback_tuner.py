import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure  # Fix: Import Figure from correct module
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import ttk
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys


class SegmentationTuner:
    def __init__(self, image_path: str, bacterium: str, structure: str, mode: str = "DARK"):
        """
        Interactive tuner for segmentation parameters
        
        Args:
            image_path: Path to the microscopy image
            bacterium: Name of the bacterium
            structure: Type of structure (bacteria, organelle, etc.)
            mode: "DARK" for dark particles on light background, "LIGHT" for light particles on dark background
        """
        self.image_path = image_path
        self.bacterium = bacterium
        self.structure = structure
        self.mode = mode
        self.invert_image = False
        
        # Load image
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Initialize parameters with defaults
        self.params = {
            'gaussian_sigma': 4.0,
            'min_area': 1980.0,
            'max_area': 9950.0,
            'dilate_iterations': 1,
            'erode_iterations': 1,
            'edge_gradient_threshold': 41
        }
        
        # Target info for clicked region
        self.target_info: Optional[Dict] = None
        self.target_marker: Optional[Tuple[int, int]] = None
        
        # Create the GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the matplotlib/tkinter GUI with improved layout"""
        # Create tkinter root
        self.root = tk.Tk()
        self.root.title(f"Segmentation Tuner - {self.bacterium}")
        self.root.geometry("1400x900")
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text=f"{self.bacterium} - {self.structure}", 
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=(0, 10))
        
        # Top section - Image and Parameters side by side
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Image
        image_frame = ttk.Frame(top_frame)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Create matplotlib figure for image - Fix: Use Figure class
        self.fig_image = Figure(figsize=(8, 8))
        self.ax_image = self.fig_image.add_subplot(111)
        self.canvas_image = FigureCanvasTkAgg(self.fig_image, image_frame)
        self.canvas_image.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Enable click on image
        self.canvas_image.mpl_connect('button_press_event', self.on_image_click)
        
        # Right side - Parameters and Histogram
        right_frame = ttk.Frame(top_frame, relief=tk.RIDGE, borderwidth=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        # Parameters section
        params_label = ttk.Label(right_frame, text="PARAMETERS", font=('Arial', 12, 'bold'))
        params_label.pack(pady=(10, 5))
        
        params_inner = ttk.Frame(right_frame)
        params_inner.pack(fill=tk.X, padx=10)
        
        # Create parameter labels with better formatting
        self.param_labels = {}
        param_data = [
            ("Bacterium:", self.bacterium),
            ("Structure:", self.structure),
            ("Mode:", f"{self.mode} particles"),
            ("", ""),  # Spacer
            ("Gaussian σ:", f"{self.params['gaussian_sigma']:.1f}"),
            ("Min area:", f"{self.params['min_area']:.0f} µm²"),
            ("Max area:", f"{self.params['max_area']:.0f} µm²"),
            ("Dilate iter:", str(self.params['dilate_iterations'])),
            ("Erode iter:", str(self.params['erode_iterations'])),
            ("Edge grad:", str(self.params['edge_gradient_threshold'])),
        ]
        
        for i, (label_text, value_text) in enumerate(param_data):
            if label_text:  # Skip spacer
                row_frame = ttk.Frame(params_inner)
                row_frame.pack(fill=tk.X, pady=2)
                
                label = ttk.Label(row_frame, text=label_text, width=15, anchor='w')
                label.pack(side=tk.LEFT)
                
                value = ttk.Label(row_frame, text=value_text, anchor='w', font=('Arial', 9))
                value.pack(side=tk.LEFT, fill=tk.X, expand=True)
                
                # Store value labels for updating
                if label_text in ["Gaussian σ:", "Min area:", "Max area:", 
                                  "Dilate iter:", "Erode iter:", "Edge grad:"]:
                    self.param_labels[label_text] = value
        
        # Histogram section
        hist_label = ttk.Label(right_frame, text="Contour Area Distribution", 
                               font=('Arial', 11, 'bold'))
        hist_label.pack(pady=(15, 5))
        
        # Fix: Use Figure class
        self.fig_hist = Figure(figsize=(5, 4))
        self.ax_hist = self.fig_hist.add_subplot(111)
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, right_frame)
        self.canvas_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Bottom section - Sliders
        slider_frame = ttk.LabelFrame(main_frame, text="Adjust Parameters", padding=10)
        slider_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Create sliders with better layout
        slider_height = 0.08
        slider_configs = [
            ('gaussian_sigma', 'Gaussian σ', 0.5, 10.0, self.params['gaussian_sigma'], 0.85),
            ('min_area', 'Min Area (µm²)', 100, 5000, self.params['min_area'], 0.70),
            ('max_area', 'Max Area (µm²)', 1000, 15000, self.params['max_area'], 0.55),
            ('dilate_iterations', 'Dilate Iter', 0, 5, self.params['dilate_iterations'], 0.40),
            ('erode_iterations', 'Erode Iter', 0, 5, self.params['erode_iterations'], 0.25),
            ('edge_gradient_threshold', 'Edge Gradient', 0, 100, self.params['edge_gradient_threshold'], 0.10),
        ]
        
        # Fix: Use Figure class
        self.fig_sliders = Figure(figsize=(12, 4))
        self.canvas_sliders = FigureCanvasTkAgg(self.fig_sliders, slider_frame)
        self.canvas_sliders.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.sliders = {}
        for param_key, label, vmin, vmax, vinit, y_pos in slider_configs:
            # Fix: Use tuple instead of list for add_axes
            ax = self.fig_sliders.add_axes((0.15, y_pos, 0.65, slider_height - 0.02))
            slider = Slider(ax, label, vmin, vmax, valinit=vinit, valstep=1 if 'iter' in param_key or 'gradient' in param_key else None)
            slider.on_changed(lambda val, key=param_key: self.update_parameter(key, val))
            self.sliders[param_key] = slider
        
        # Invert checkbox - Fix: Use tuple instead of list
        ax_invert = self.fig_sliders.add_axes((0.82, 0.10, 0.15, slider_height - 0.02))
        self.btn_invert = Button(ax_invert, 'Invert: OFF', color='lightgray', hovercolor='gray')
        self.btn_invert.on_clicked(self.toggle_invert)
        
        # Bottom buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Create button container for centering
        button_container = ttk.Frame(button_frame)
        button_container.pack(expand=True)
        
        style = ttk.Style()
        style.configure('Back.TButton', font=('Arial', 10))
        style.configure('Save.TButton', font=('Arial', 10, 'bold'))
        style.configure('Quit.TButton', font=('Arial', 10))
        
        btn_back = ttk.Button(button_container, text="BACK", command=self.back, 
                             style='Back.TButton', width=15)
        btn_back.pack(side=tk.LEFT, padx=5)
        
        btn_save = ttk.Button(button_container, text="SAVE", command=self.save, 
                             style='Save.TButton', width=15)
        btn_save.pack(side=tk.LEFT, padx=5)
        
        btn_quit = ttk.Button(button_container, text="QUIT", command=self.quit, 
                             style='Quit.TButton', width=15)
        btn_quit.pack(side=tk.LEFT, padx=5)
        
        # Initial update
        self.update_visualization()
        
    def update_parameter(self, param_key: str, value: float):
        """Update parameter and refresh visualization"""
        self.params[param_key] = value
        
        # Update parameter display
        label_map = {
            'gaussian_sigma': 'Gaussian σ:',
            'min_area': 'Min area:',
            'max_area': 'Max area:',
            'dilate_iterations': 'Dilate iter:',
            'erode_iterations': 'Erode iter:',
            'edge_gradient_threshold': 'Edge grad:'
        }
        
        if param_key in label_map:
            label_key = label_map[param_key]
            if label_key in self.param_labels:
                if 'area' in param_key:
                    self.param_labels[label_key].config(text=f"{value:.0f} µm²")
                elif 'sigma' in param_key:
                    self.param_labels[label_key].config(text=f"{value:.1f}")
                else:
                    self.param_labels[label_key].config(text=str(int(value)))
        
        self.update_visualization()
        
    def toggle_invert(self, event):
        """Toggle image inversion"""
        self.invert_image = not self.invert_image
        self.btn_invert.label.set_text(f'Invert: {"ON" if self.invert_image else "OFF"}')
        self.btn_invert.color = 'lightgreen' if self.invert_image else 'lightgray'
        self.update_visualization()
        
    def segment_image(self) -> Tuple[np.ndarray, list]:
        """Apply current parameters and segment the image"""
        # Fix: Ensure original_image is not None
        if self.original_image is None:
            return np.array([]), []
        
        # Apply Gaussian blur - Fix: Cast to proper type
        blurred = cv2.GaussianBlur(
            self.original_image.astype(np.uint8), 
            (0, 0), 
            float(self.params['gaussian_sigma'])
        )
        
        # Threshold using Otsu's method
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed
        if self.invert_image:
            binary = cv2.bitwise_not(binary)
        
        # Morphological operations
        if self.params['dilate_iterations'] > 0:
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=int(self.params['dilate_iterations']))
        
        if self.params['erode_iterations'] > 0:
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.erode(binary, kernel, iterations=int(self.params['erode_iterations']))
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.params['min_area'] <= area <= self.params['max_area']:
                filtered_contours.append(contour)
        
        return binary, filtered_contours
    
    def update_visualization(self):
        """Update all visualizations"""
        # Segment image
        binary, contours = self.segment_image()
        
        # Fix: Ensure original_image is not None
        if self.original_image is None:
            return
        
        # Draw on image - Fix: Cast to proper type
        display_image = cv2.cvtColor(self.original_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        cv2.drawContours(display_image, contours, -1, (0, 255, 0), 2)
        
        # Draw target marker if exists
        if self.target_marker:
            x, y = self.target_marker
            cv2.circle(display_image, (x, y), 5, (255, 0, 0), -1)
            cv2.circle(display_image, (x, y), 50, (255, 0, 0), 2)
            
            # Add text label
            cv2.putText(display_image, "TARGET", (x - 30, y - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(display_image, f"({x}, {y})", (x - 30, y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            if self.target_info and 'estimated_area' in self.target_info:
                cv2.putText(display_image, f"{self.target_info['estimated_area']:.0f} px",
                           (x - 30, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Update image display
        self.ax_image.clear()
        self.ax_image.imshow(display_image)
        self.ax_image.set_title(f'Detected Contours (n={len(contours)})', fontsize=12, fontweight='bold')
        self.ax_image.axis('off')
        self.canvas_image.draw()
        
        # Update histogram
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            
            self.ax_hist.clear()
            self.ax_hist.hist(areas, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
            
            # Add statistics lines - Fix: Convert numpy types to float
            median_area = float(np.median(areas))
            mean_area = float(np.mean(areas))
            
            self.ax_hist.axvline(median_area, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_area:.1f}')
            self.ax_hist.axvline(mean_area, color='darkorange', linestyle='--', linewidth=2, label=f'Mean: {mean_area:.1f}')
            self.ax_hist.axvline(self.params['min_area'], color='blue', linestyle=':', linewidth=1.5, label=f"Min: {self.params['min_area']:.0f}")
            self.ax_hist.axvline(self.params['max_area'], color='blue', linestyle=':', linewidth=1.5, label=f"Max: {self.params['max_area']:.0f}")
            
            # Add target area if exists
            if self.target_info and 'estimated_area' in self.target_info:
                target_area = self.target_info['estimated_area']
                self.ax_hist.axvline(target_area, color='red', linestyle='-', linewidth=2.5, label=f'Target: {target_area:.0f}')
            
            self.ax_hist.set_xlabel('Area (µm²)', fontsize=10)
            self.ax_hist.set_ylabel('Count', fontsize=10)
            self.ax_hist.set_title(f'Contour Area Distribution (n={len(contours)})', fontsize=10, fontweight='bold')
            self.ax_hist.legend(fontsize=8, loc='upper right')
            self.ax_hist.grid(True, alpha=0.3)
        else:
            self.ax_hist.clear()
            self.ax_hist.text(0.5, 0.5, 'No contours detected', 
                            ha='center', va='center', fontsize=12, transform=self.ax_hist.transAxes)
            self.ax_hist.set_title('Contour Area Distribution (n=0)', fontsize=10, fontweight='bold')
        
        self.fig_hist.tight_layout()
        self.canvas_hist.draw()
    
    def on_image_click(self, event):
        """Handle click on image to select target region"""
        if event.inaxes != self.ax_image:
            return
        
        if event.xdata is None or event.ydata is None:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        
        # Store marker position
        self.target_marker = (x, y)
        
        print(f"\n🎯 Target selected at ({x}, {y})")
        
        # Analyze the region
        self.analyze_target_region(x, y)
        
        # Update visualization
        self.update_visualization()
    
    def analyze_target_region(self, x: int, y: int, radius: int = 50):
        """Analyze the region around the clicked point"""
        if self.original_image is None:
            return
        
        h, w = self.original_image.shape
        
        # Extract region
        y1, y2 = max(0, y-radius), min(h, y+radius)
        x1, x2 = max(0, x-radius), min(w, x+radius)
        region = self.original_image[y1:y2, x1:x2]
        
        # Calculate region statistics with explicit type conversion
        region_float: np.ndarray = region.astype(np.float64)
        mean_intensity = float(np.mean(region_float))
        std_intensity = float(np.std(region_float))
        min_intensity = int(np.min(region))
        max_intensity = int(np.max(region))
        
        # Store target info
        self.target_info = {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'min_intensity': min_intensity,
            'max_intensity': max_intensity,
        }
        
        # Estimate particle size
        try:
            blurred_region = cv2.GaussianBlur(region.astype(np.uint8), (0, 0), float(self.params['gaussian_sigma']))
            _, binary_region = cv2.threshold(blurred_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            if self.invert_image:
                binary_region = cv2.bitwise_not(binary_region)
            
            contours, _ = cv2.findContours(binary_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                center = (radius, radius) if x1 + radius <= w and y1 + radius <= h else (x - x1, y - y1)
                
                closest_contour = None
                min_distance = float('inf')
                
                for contour in contours:
                    dist = abs(cv2.pointPolygonTest(contour, center, True))
                    if dist < min_distance:
                        min_distance = dist
                        closest_contour = contour
                
                if closest_contour is not None:
                    estimated_area = float(cv2.contourArea(closest_contour))
                    perimeter = float(cv2.arcLength(closest_contour, True))
                    circularity = 4 * np.pi * estimated_area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    self.target_info['estimated_area'] = estimated_area
                    self.target_info['circularity'] = circularity
                    
                    print(f"  📊 Target region analysis:")
                    print(f"     Position: ({x}, {y})")
                    print(f"     Intensity: {mean_intensity:.1f} ± {std_intensity:.1f}")
                    print(f"     Range: {min_intensity} - {max_intensity}")
                    print(f"     Estimated area: {estimated_area:.0f} pixels")
                    print(f"     Circularity: {circularity:.2f}")
                    
                    self.suggest_parameters(mean_intensity, estimated_area, min_intensity)
                    
        except Exception as e:
            print(f"  ⚠ Error analyzing region: {e}")
    
    def suggest_parameters(self, mean_intensity: float, estimated_area: float, min_intensity: int):
        """Suggest parameter adjustments based on target"""
        print(f"\n  💡 Parameter suggestions:")
        
        # Suggest area range
        suggested_min = max(100, estimated_area * 0.5)
        suggested_max = min(15000, estimated_area * 2.0)
        print(f"     Min area: {suggested_min:.0f} µm²")
        print(f"     Max area: {suggested_max:.0f} µm²")
        
        # Suggest Gaussian sigma based on particle size
        if estimated_area < 1000:
            suggested_sigma = 2.0
        elif estimated_area < 3000:
            suggested_sigma = 4.0
        else:
            suggested_sigma = 6.0
        print(f"     Gaussian σ: {suggested_sigma:.1f}")
        
        # Check if inversion might help
        if self.mode == "DARK" and mean_intensity > 128:
            print(f"     ⚠ Consider enabling 'Invert' (bright region detected)")
        elif self.mode == "LIGHT" and mean_intensity < 128:
            print(f"     ⚠ Consider enabling 'Invert' (dark region detected)")
    
    def save(self):
        """Save current parameters"""
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / f"{self.bacterium}_{self.structure}_segmentation.json"
        
        config = {
            'bacterium': self.bacterium,
            'structure': self.structure,
            'mode': self.mode,
            'invert': self.invert_image,
            'parameters': self.params,
            'target_info': self.target_info
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Parameters saved to {config_file}")
        print(f"   Parameters: {self.params}")
        
    def back(self):
        """Go back (close window)"""
        self.root.quit()
        self.root.destroy()
        
    def quit(self):
        """Quit application"""
        self.root.quit()
        self.root.destroy()
        sys.exit(0)
        
    def run(self):
        """Start the GUI"""
        self.root.mainloop()


def main():
    """Example usage"""
    import argparse
    import os
    from glob import glob
    
    # Try to parse command line arguments
    parser = argparse.ArgumentParser(description='Interactive Segmentation Parameter Tuner')
    parser.add_argument('image_path', type=str, nargs='?', help='Path to the microscopy image')
    parser.add_argument('--bacterium', type=str, default='Unknown Bacterium', 
                       help='Name of the bacterium')
    parser.add_argument('--structure', type=str, default='bacteria', 
                       help='Type of structure (bacteria, organelle, etc.)')
    parser.add_argument('--mode', type=str, choices=['DARK', 'LIGHT'], default='DARK',
                       help='DARK for dark particles on light background, LIGHT for light particles')
    
    args = parser.parse_args()
    
    # If no image path provided, try to find one
    if args.image_path is None:
        print("No image path provided. Searching for images...")
        
        # Search for images in common locations
        search_patterns = [
            "source/**/*ch00.tif",
        ]
        
        found_images = []
        for pattern in search_patterns:
            found_images.extend(glob(pattern, recursive=True))
        
        if not found_images:
            print(f"❌ Error: No images found in data/ directory or current directory")
            print(f"\nCurrent directory: {os.getcwd()}")
            print(f"\nUsage:")
            print(f"  python feedback_tuner.py path/to/image.tif")
            print(f"  python feedback_tuner.py path/to/image.tif --bacterium 'E. coli' --structure bacteria")
            sys.exit(1)
        
        # Use the first found image
        args.image_path = found_images[0]
        print(f"Found {len(found_images)} image(s). Using: {args.image_path}")
        
        # Try to extract bacterium name from path
        if 'klebsiella' in args.image_path.lower():
            args.bacterium = "Klebsiella Pneumoniae"
        elif 'ecoli' in args.image_path.lower() or 'e_coli' in args.image_path.lower():
            args.bacterium = "E. coli"
    
    # Check if file exists
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image file not found: {args.image_path}")
        print(f"\nCurrent directory: {os.getcwd()}")
        print(f"\nPlease provide a valid image path.")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Loading Segmentation Tuner")
    print(f"{'='*60}")
    print(f"Image:     {args.image_path}")
    print(f"Bacterium: {args.bacterium}")
    print(f"Structure: {args.structure}")
    print(f"Mode:      {args.mode}")
    print(f"{'='*60}\n")
    
    try:
        tuner = SegmentationTuner(
            image_path=args.image_path,
            bacterium=args.bacterium,
            structure=args.structure,
            mode=args.mode
        )
        
        tuner.run()
        
    except Exception as e:
        print(f"\n❌ Error creating tuner: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()