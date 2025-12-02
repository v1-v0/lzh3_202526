import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
from pathlib import Path
import platform

class ParticleDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Particle Detector - Advanced Image Analysis")
        self.root.geometry("1400x900")
        
        # Initialize variables
        self.original_image = None
        self.preprocessed_image = None
        self.contours = None
        self.final_contours = None
        self.hierarchy = None
        self.measurements = None
        self.current_folder = None
        self.image_files = []
        self.current_image_index = 0
        
        # Processing parameters
        self.blur_kernel = tk.IntVar(value=5)
        self.threshold_value = tk.IntVar(value=127)
        self.min_area = tk.IntVar(value=100)
        self.max_area = tk.IntVar(value=10000)
        self.circularity_threshold = tk.DoubleVar(value=0.5)
        self.pixels_per_mm = tk.DoubleVar(value=10.0)
        
        # Display variables
        self.show_labels = tk.BooleanVar(value=True)
        self.show_scale = tk.BooleanVar(value=True)
        self.contour_color_var = tk.StringVar(value="Green")
        self.line_thickness_var = tk.IntVar(value=2)
        self.final_contour_color_var = tk.StringVar(value="Red")
        self.final_line_thickness_var = tk.IntVar(value=2)
        self.overlay_contour_color_var = tk.StringVar(value="Red")
        self.overlay_line_thickness_var = tk.IntVar(value=2)
        
        # Tab display variables
        self.tab_display_vars = {}
        
        self.setup_ui()
    
    def create_tab_controls(self, parent, tab_name):
        """Create controls for each tab including label and scale toggles."""
        control_frame = ttk.Frame(parent)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Left side - existing controls
        left_frame = ttk.Frame(control_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Right side - label and scale controls
        right_frame = ttk.Frame(control_frame)
        right_frame.pack(side=tk.RIGHT)
        
        # Create variables for this tab
        show_labels_var = tk.BooleanVar(value=True)
        show_scale_var = tk.BooleanVar(value=True)
        
        # Store variables in a dictionary for each tab
        self.tab_display_vars[tab_name] = {
            'show_labels': show_labels_var,
            'show_scale': show_scale_var
        }
        
        # Add checkboxes
        ttk.Checkbutton(
            right_frame,
            text="Show Labels",
            variable=show_labels_var,
            command=lambda: self.update_tab_display(tab_name)
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Checkbutton(
            right_frame,
            text="Show Scale",
            variable=show_scale_var,
            command=lambda: self.update_tab_display(tab_name)
        ).pack(side=tk.LEFT, padx=5)
        
        return control_frame, left_frame
    
    def update_tab_display(self, tab_name):
        """Update the display for a specific tab based on its control settings."""
        if tab_name == "Original":
            self.display_original_image()
        elif tab_name == "Preprocessed":
            self.display_preprocessed_image()
        elif tab_name == "Contours":
            self.display_contours()
        elif tab_name == "Final Contours":
            self.display_final_contours()
        elif tab_name == "Overlay":
            self.display_overlay()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        left_panel = ttk.Frame(main_container, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        # Right panel for image display
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Setup left panel components
        self.setup_file_controls(left_panel)
        self.setup_processing_controls(left_panel)
        self.setup_measurement_controls(left_panel)
        self.setup_action_buttons(left_panel)
        
        # Setup right panel - Notebook with tabs
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Original Image Tab
        original_tab = ttk.Frame(self.notebook)
        self.notebook.add(original_tab, text="Original")
        control_frame, left_frame = self.create_tab_controls(original_tab, "Original")
        
        self.original_canvas = tk.Canvas(original_tab, bg='white')
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Preprocessed Image Tab
        preprocessed_tab = ttk.Frame(self.notebook)
        self.notebook.add(preprocessed_tab, text="Preprocessed")
        control_frame, left_frame = self.create_tab_controls(preprocessed_tab, "Preprocessed")
        
        self.preprocessed_canvas = tk.Canvas(preprocessed_tab, bg='white')
        self.preprocessed_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Contours Tab
        contours_tab = ttk.Frame(self.notebook)
        self.notebook.add(contours_tab, text="Contours")
        control_frame, left_frame = self.create_tab_controls(contours_tab, "Contours")
        
        # Add contour-specific controls to left_frame
        ttk.Label(left_frame, text="Contour Color:").pack(side=tk.LEFT, padx=(0, 5))
        color_menu = ttk.Combobox(left_frame, textvariable=self.contour_color_var,
                                 values=["Green", "Red", "Blue", "Yellow", "Cyan", "Magenta"],
                                 state="readonly", width=10)
        color_menu.pack(side=tk.LEFT, padx=5)
        color_menu.bind("<<ComboboxSelected>>", lambda e: self.display_contours())
        
        ttk.Label(left_frame, text="Line Thickness:").pack(side=tk.LEFT, padx=(10, 5))
        thickness_spinbox = ttk.Spinbox(left_frame, from_=1, to=10, 
                                       textvariable=self.line_thickness_var,
                                       width=5, command=self.display_contours)
        thickness_spinbox.pack(side=tk.LEFT, padx=5)
        
        self.contours_canvas = tk.Canvas(contours_tab, bg='white')
        self.contours_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Final Contours Tab
        final_tab = ttk.Frame(self.notebook)
        self.notebook.add(final_tab, text="Final Contours")
        control_frame, left_frame = self.create_tab_controls(final_tab, "Final Contours")
        
        # Add final contour-specific controls to left_frame
        ttk.Label(left_frame, text="Contour Color:").pack(side=tk.LEFT, padx=(0, 5))
        final_color_menu = ttk.Combobox(left_frame, textvariable=self.final_contour_color_var,
                                       values=["Green", "Red", "Blue", "Yellow", "Cyan", "Magenta"],
                                       state="readonly", width=10)
        final_color_menu.pack(side=tk.LEFT, padx=5)
        final_color_menu.bind("<<ComboboxSelected>>", lambda e: self.display_final_contours())
        
        ttk.Label(left_frame, text="Line Thickness:").pack(side=tk.LEFT, padx=(10, 5))
        final_thickness_spinbox = ttk.Spinbox(left_frame, from_=1, to=10,
                                             textvariable=self.final_line_thickness_var,
                                             width=5, command=self.display_final_contours)
        final_thickness_spinbox.pack(side=tk.LEFT, padx=5)
        
        self.final_canvas = tk.Canvas(final_tab, bg='white')
        self.final_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Overlay Tab
        overlay_tab = ttk.Frame(self.notebook)
        self.notebook.add(overlay_tab, text="Overlay")
        control_frame, left_frame = self.create_tab_controls(overlay_tab, "Overlay")
        
        # Add overlay-specific controls to left_frame
        ttk.Label(left_frame, text="Contour Color:").pack(side=tk.LEFT, padx=(0, 5))
        overlay_color_menu = ttk.Combobox(left_frame, textvariable=self.overlay_contour_color_var,
                                         values=["Green", "Red", "Blue", "Yellow", "Cyan", "Magenta"],
                                         state="readonly", width=10)
        overlay_color_menu.pack(side=tk.LEFT, padx=5)
        overlay_color_menu.bind("<<ComboboxSelected>>", lambda e: self.display_overlay())
        
        ttk.Label(left_frame, text="Line Thickness:").pack(side=tk.LEFT, padx=(10, 5))
        overlay_thickness_spinbox = ttk.Spinbox(left_frame, from_=1, to=10,
                                               textvariable=self.overlay_line_thickness_var,
                                               width=5, command=self.display_overlay)
        overlay_thickness_spinbox.pack(side=tk.LEFT, padx=5)
        
        self.overlay_canvas = tk.Canvas(overlay_tab, bg='white')
        self.overlay_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_file_controls(self, parent):
        """Setup file selection controls."""
        file_frame = ttk.LabelFrame(parent, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="📁 Select Folder", 
                  command=self.select_folder, width=25).pack(pady=2)
        
        ttk.Button(file_frame, text="📄 Select Single Image", 
                  command=self.select_single_image, width=25).pack(pady=2)
        
        # Navigation frame
        nav_frame = ttk.Frame(file_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        self.prev_btn = ttk.Button(nav_frame, text="◀ Previous", 
                                   command=self.previous_image, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=2)
        
        self.next_btn = ttk.Button(nav_frame, text="Next ▶", 
                                   command=self.next_image, state=tk.DISABLED)
        self.next_btn.pack(side=tk.RIGHT, padx=2)
        
        self.image_counter_label = ttk.Label(file_frame, text="No images loaded")
        self.image_counter_label.pack(pady=2)
    
    def setup_processing_controls(self, parent):
        """Setup image processing parameter controls."""
        proc_frame = ttk.LabelFrame(parent, text="Processing Parameters", padding=10)
        proc_frame.pack(fill=tk.X, pady=5)
        
        # Blur Kernel Size
        ttk.Label(proc_frame, text="Blur Kernel Size:").pack(anchor=tk.W)
        blur_frame = ttk.Frame(proc_frame)
        blur_frame.pack(fill=tk.X, pady=2)
        ttk.Scale(blur_frame, from_=1, to=15, variable=self.blur_kernel, 
                 orient=tk.HORIZONTAL, command=lambda x: self.blur_kernel.set(int(float(x)) // 2 * 2 + 1)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(blur_frame, textvariable=self.blur_kernel, width=5).pack(side=tk.RIGHT)
        
        # Threshold Value
        ttk.Label(proc_frame, text="Threshold Value:").pack(anchor=tk.W, pady=(5, 0))
        threshold_frame = ttk.Frame(proc_frame)
        threshold_frame.pack(fill=tk.X, pady=2)
        ttk.Scale(threshold_frame, from_=0, to=255, variable=self.threshold_value, 
                 orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(threshold_frame, textvariable=self.threshold_value, width=5).pack(side=tk.RIGHT)
        
        # Min Area
        ttk.Label(proc_frame, text="Min Area (pixels):").pack(anchor=tk.W, pady=(5, 0))
        min_area_frame = ttk.Frame(proc_frame)
        min_area_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(min_area_frame, textvariable=self.min_area, width=10).pack(side=tk.LEFT)
        
        # Max Area
        ttk.Label(proc_frame, text="Max Area (pixels):").pack(anchor=tk.W, pady=(5, 0))
        max_area_frame = ttk.Frame(proc_frame)
        max_area_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(max_area_frame, textvariable=self.max_area, width=10).pack(side=tk.LEFT)
        
        # Circularity
        ttk.Label(proc_frame, text="Circularity Threshold:").pack(anchor=tk.W, pady=(5, 0))
        circ_frame = ttk.Frame(proc_frame)
        circ_frame.pack(fill=tk.X, pady=2)
        ttk.Scale(circ_frame, from_=0, to=1, variable=self.circularity_threshold, 
                 orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(circ_frame, textvariable=self.circularity_threshold, width=5).pack(side=tk.RIGHT)
    
    def setup_measurement_controls(self, parent):
        """Setup measurement calibration controls."""
        meas_frame = ttk.LabelFrame(parent, text="Measurement Calibration", padding=10)
        meas_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(meas_frame, text="Pixels per mm:").pack(anchor=tk.W)
        calib_frame = ttk.Frame(meas_frame)
        calib_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(calib_frame, textvariable=self.pixels_per_mm, width=10).pack(side=tk.LEFT)
        ttk.Label(calib_frame, text="px/mm").pack(side=tk.LEFT, padx=5)
    
    def setup_action_buttons(self, parent):
        """Setup action buttons."""
        action_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        action_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(action_frame, text="🔄 Process Image", 
                  command=self.process_image, width=25).pack(pady=2)
        
        ttk.Button(action_frame, text="💾 Export Results", 
                  command=self.export_results, width=25).pack(pady=2)
        
        ttk.Button(action_frame, text="📊 Batch Process Folder", 
                  command=self.batch_process, width=25).pack(pady=2)
    
    def select_folder(self):
        """Select a folder containing images."""
        initial_dir = str(Path.home())
        
        if platform.system() == "Linux":
            folder_path = self._linux_folder_dialog(initial_dir)
        else:
            folder_path = filedialog.askdirectory(initialdir=initial_dir)
        
        if folder_path:
            self.current_folder = Path(folder_path)
            self.image_files = sorted([
                f for f in self.current_folder.glob("*")
                if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
            ])
            
            if self.image_files:
                self.current_image_index = 0
                self.load_image(self.image_files[self.current_image_index])
                self.update_navigation_buttons()
                self.update_image_counter()
                self.status_bar.config(text=f"Loaded folder: {self.current_folder} ({len(self.image_files)} images)")
            else:
                messagebox.showwarning("No Images", "No image files found in the selected folder.")
    
    def _linux_folder_dialog(self, initial_dir: str) -> str:
        """Custom folder selection dialog optimized for Linux."""
        selected_folder = tk.StringVar(value="")
        current_path = Path(initial_dir)
        
        def on_folder_click(event):
            selection = listbox.curselection()
            if selection:
                idx = selection[0]
                folder_name = listbox.get(idx)
                if folder_name == "..":
                    return
                else:
                    potential_path = current_path / folder_name
                    path_entry.delete(0, tk.END)
                    path_entry.insert(0, str(potential_path))
        
        def on_folder_double_click(event):
            nonlocal current_path
            selection = listbox.curselection()
            if selection:
                idx = selection[0]
                folder_name = listbox.get(idx)
                if folder_name == "..":
                    current_path = current_path.parent
                    refresh_folder_list()
                else:
                    potential_path = current_path / folder_name
                    if potential_path.is_dir():
                        current_path = potential_path
                        refresh_folder_list()
                        path_entry.delete(0, tk.END)
                        path_entry.insert(0, str(current_path))
        
        def refresh_folder_list():
            nonlocal current_path
            listbox.delete(0, tk.END)
            
            if current_path.parent != current_path:
                listbox.insert(tk.END, "..")
            
            try:
                items = sorted([
                    item for item in current_path.iterdir() 
                    if item.is_dir() and not item.name.startswith('.')
                ], key=lambda p: p.name.lower())
                
                for item in items:
                    listbox.insert(tk.END, item.name)
                
                current_path_label.config(text=f"Current: {current_path}")
                path_entry.delete(0, tk.END)
                path_entry.insert(0, str(current_path))
                
            except PermissionError:
                messagebox.showerror("Permission Denied", f"Cannot access: {current_path}")
                current_path = current_path.parent
                refresh_folder_list()
        
        def on_select():
            path_text = path_entry.get().strip()
            if path_text:
                selected_path = Path(path_text)
                if selected_path.is_dir():
                    selected_folder.set(str(selected_path))
                    dialog.destroy()
                else:
                    messagebox.showwarning("Invalid Path", "Please select a valid folder")
            else:
                selected_folder.set(str(current_path))
                dialog.destroy()
        
        def on_cancel():
            selected_folder.set("")
            dialog.destroy()
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Folder")
        dialog.geometry("700x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        current_path_label = ttk.Label(dialog, text=f"Current: {current_path}", 
                                       font=("Segoe UI", 9), foreground="#666")
        current_path_label.pack(pady=(10, 5), padx=10, anchor=tk.W)
        
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, 
                            font=("Monospace", 10), height=20)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        listbox.bind('<ButtonRelease-1>', on_folder_click)
        listbox.bind('<Double-Button-1>', on_folder_double_click)
        
        path_frame = ttk.Frame(dialog)
        path_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        
        ttk.Label(path_frame, text="Selected path:").pack(side=tk.LEFT, padx=(0, 5))
        path_entry = ttk.Entry(path_frame, font=("Monospace", 9))
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        path_entry.insert(0, str(current_path))
        path_entry.bind('<Return>', lambda e: on_select())
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Select This Folder", command=on_select, 
                  width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=on_cancel, 
                  width=12).pack(side=tk.LEFT, padx=5)
        
        instructions = ttk.Label(dialog, 
            text="💡 Single-click: Preview path | Double-click: Navigate into folder | '..' = Go up",
            font=("Segoe UI", 8), foreground="#888")
        instructions.pack(pady=(0, 10))
        
        refresh_folder_list()
        
        listbox.focus_set()
        if listbox.size() > 0:
            listbox.selection_set(0)
        
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        dialog.wait_window()
        return selected_folder.get()
    
    def select_single_image(self):
        """Select a single image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_folder = Path(file_path).parent
            self.image_files = [Path(file_path)]
            self.current_image_index = 0
            self.load_image(Path(file_path))
            self.update_navigation_buttons()
            self.update_image_counter()
            self.status_bar.config(text=f"Loaded: {Path(file_path).name}")
    
    def load_image(self, image_path):
        """Load an image from the specified path."""
        try:
            self.original_image = cv2.imread(str(image_path))
            if self.original_image is None:
                raise ValueError("Failed to load image")
            
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.preprocessed_image = None
            self.contours = None
            self.final_contours = None
            self.measurements = None
            
            self.display_original_image()
            self.status_bar.config(text=f"Loaded: {image_path.name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def previous_image(self):
        """Load the previous image in the folder."""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image(self.image_files[self.current_image_index])
            self.update_navigation_buttons()
            self.update_image_counter()
    
    def next_image(self):
        """Load the next image in the folder."""
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image(self.image_files[self.current_image_index])
            self.update_navigation_buttons()
            self.update_image_counter()
    
    def update_navigation_buttons(self):
        """Update the state of navigation buttons."""
        if len(self.image_files) <= 1:
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
        else:
            self.prev_btn.config(state=tk.NORMAL if self.current_image_index > 0 else tk.DISABLED)
            self.next_btn.config(state=tk.NORMAL if self.current_image_index < len(self.image_files) - 1 else tk.DISABLED)
    
    def update_image_counter(self):
        """Update the image counter label."""
        if self.image_files:
            self.image_counter_label.config(
                text=f"Image {self.current_image_index + 1} of {len(self.image_files)}"
            )
        else:
            self.image_counter_label.config(text="No images loaded")
    
    def process_image(self):
        """Process the current image to detect particles."""
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            kernel_size = self.blur_kernel.get()
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            
            # Apply threshold
            _, thresh = cv2.threshold(blurred, self.threshold_value.get(), 255, cv2.THRESH_BINARY_INV)
            self.preprocessed_image = thresh
            
            # Find contours
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.contours = contours
            self.hierarchy = hierarchy
            
            # Filter contours
            self.filter_contours()
            
            # Calculate measurements
            self.calculate_measurements()
            
            # Display results
            self.display_preprocessed_image()
            self.display_contours()
            self.display_final_contours()
            self.display_overlay()
            
            num_particles = len(self.final_contours) if self.final_contours is not None else 0
            self.status_bar.config(text=f"Processing complete. Found {num_particles} particles.")
            
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred: {str(e)}")
    
    def filter_contours(self):
        """Filter contours based on area and circularity."""
        if self.contours is None:
            return
        
        filtered_contours = []
        min_area = self.min_area.get()
        max_area = self.max_area.get()
        circ_thresh = self.circularity_threshold.get()
        
        for contour in self.contours:
            area = cv2.contourArea(contour)
            
            if min_area <= area <= max_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity >= circ_thresh:
                        filtered_contours.append(contour)
        
        self.final_contours = filtered_contours
    
    def calculate_measurements(self):
        """Calculate measurements for each detected particle."""
        if self.final_contours is None or len(self.final_contours) == 0:
            self.measurements = None
            return
        
        measurements = []
        ppmm = self.pixels_per_mm.get()
        
        for i, contour in enumerate(self.final_contours):
            area_px = cv2.contourArea(contour)
            area_mm = area_px / (ppmm ** 2)
            
            perimeter_px = cv2.arcLength(contour, True)
            perimeter_mm = perimeter_px / ppmm
            
            # Equivalent diameter
            equiv_diameter_px = np.sqrt(4 * area_px / np.pi)
            equiv_diameter_mm = equiv_diameter_px / ppmm
            
            # Circularity
            circularity = 4 * np.pi * area_px / (perimeter_px ** 2) if perimeter_px > 0 else 0
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            measurements.append({
                'Particle_ID': i + 1,
                'Area_px': area_px,
                'Area_mm2': area_mm,
                'Perimeter_px': perimeter_px,
                'Perimeter_mm': perimeter_mm,
                'Equiv_Diameter_px': equiv_diameter_px,
                'Equiv_Diameter_mm': equiv_diameter_mm,
                'Circularity': circularity,
                'BoundingBox_X': x,
                'BoundingBox_Y': y,
                'BoundingBox_Width': w,
                'BoundingBox_Height': h
            })
        
        self.measurements = pd.DataFrame(measurements)
    
    def display_image(self, image, canvas):
        """Display an image on the specified canvas."""
        # Get canvas dimensions
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Convert image to PIL format
        if len(image.shape) == 2:  # Grayscale
            pil_image = Image.fromarray(image)
        else:  # Color
            pil_image = Image.fromarray(image)
        
        # Calculate scaling to fit canvas
        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Display on canvas
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER)
        canvas.image = photo  # Keep a reference
    
    def add_labels_to_image(self, image, reference_image, num_contours=None):
        """Add labels to the image."""
        img_with_labels = image.copy()
        h, w = reference_image.shape[:2]
        
        # Add title
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = f"Particle Detection"
        if num_contours is not None:
            title += f" - {num_contours} particles"
        
        cv2.putText(img_with_labels, title, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add scale info
        scale_text = f"Scale: {self.pixels_per_mm.get():.2f} px/mm"
        cv2.putText(img_with_labels, scale_text, (10, h - 10), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        return img_with_labels
    
    def add_scale_bar(self, image):
        """Add a scale bar to the image."""
        img_with_scale = image.copy()
        h, w = img_with_scale.shape[:2]
        
        # Scale bar parameters
        bar_length_mm = 1.0  # 1 mm
        bar_length_px = int(bar_length_mm * self.pixels_per_mm.get())
        bar_height = 10
        bar_x = w - bar_length_px - 20
        bar_y = h - 40
        
        # Draw scale bar
        cv2.rectangle(img_with_scale, (bar_x, bar_y), (bar_x + bar_length_px, bar_y + bar_height), 
                     (255, 255, 255), -1)
        cv2.rectangle(img_with_scale, (bar_x, bar_y), (bar_x + bar_length_px, bar_y + bar_height), 
                     (0, 0, 0), 2)
        
        # Add scale text
        scale_text = f"{bar_length_mm} mm"
        cv2.putText(img_with_scale, scale_text, (bar_x, bar_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return img_with_scale
    
    def display_original_image(self):
        """Display the original image on the canvas."""
        if self.original_image is None:
            return
        
        display_vars = self.tab_display_vars.get("Original", {})
        show_labels = display_vars.get('show_labels', tk.BooleanVar(value=True)).get()
        show_scale = display_vars.get('show_scale', tk.BooleanVar(value=True)).get()
        
        img_display = self.original_image.copy()
        
        if show_labels:
            img_display = self.add_labels_to_image(img_display, self.original_image)
        
        if show_scale:
            img_display = self.add_scale_bar(img_display)
        
        self.display_image(img_display, self.original_canvas)
    
    def display_preprocessed_image(self):
        """Display the preprocessed image."""
        if self.preprocessed_image is None:
            return
        
        display_vars = self.tab_display_vars.get("Preprocessed", {})
        show_labels = display_vars.get('show_labels', tk.BooleanVar(value=True)).get()
        show_scale = display_vars.get('show_scale', tk.BooleanVar(value=True)).get()
        
        img_display = cv2.cvtColor(self.preprocessed_image, cv2.COLOR_GRAY2BGR)
        
        if show_labels:
            img_display = self.add_labels_to_image(img_display, self.preprocessed_image)
        
        if show_scale:
            img_display = self.add_scale_bar(img_display)
        
        self.display_image(img_display, self.preprocessed_canvas)
    
    def display_contours(self):
        """Display all detected contours."""
        if self.original_image is None or self.contours is None:
            return
        
        display_vars = self.tab_display_vars.get("Contours", {})
        show_labels = display_vars.get('show_labels', tk.BooleanVar(value=True)).get()
        show_scale = display_vars.get('show_scale', tk.BooleanVar(value=True)).get()
        
        img_display = self.original_image.copy()
        
        color_map = {
            "Green": (0, 255, 0), "Red": (255, 0, 0), "Blue": (0, 0, 255),
            "Yellow": (255, 255, 0), "Cyan": (0, 255, 255), "Magenta": (255, 0, 255)
        }
        color = color_map.get(self.contour_color_var.get(), (0, 255, 0))
        thickness = self.line_thickness_var.get()
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        cv2.drawContours(img_bgr, self.contours, -1, color, thickness)
        img_display = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if show_labels:
            img_display = self.add_labels_to_image(img_display, self.original_image, 
                                                   num_contours=len(self.contours))
        
        if show_scale:
            img_display = self.add_scale_bar(img_display)
        
        self.display_image(img_display, self.contours_canvas)
    
    def display_final_contours(self):
        """Display filtered contours with measurements."""
        if self.original_image is None or self.final_contours is None:
            return
        
        display_vars = self.tab_display_vars.get("Final Contours", {})
        show_labels = display_vars.get('show_labels', tk.BooleanVar(value=True)).get()
        show_scale = display_vars.get('show_scale', tk.BooleanVar(value=True)).get()
        
        img_display = self.original_image.copy()
        
        color_map = {
            "Green": (0, 255, 0), "Red": (255, 0, 0), "Blue": (0, 0, 255),
            "Yellow": (255, 255, 0), "Cyan": (0, 255, 255), "Magenta": (255, 0, 255)
        }
        color = color_map.get(self.final_contour_color_var.get(), (255, 0, 0))
        thickness = self.final_line_thickness_var.get()
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        cv2.drawContours(img_bgr, self.final_contours, -1, color, thickness)
        img_display = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if show_labels:
            img_display = self.add_labels_to_image(img_display, self.original_image, 
                                                   num_contours=len(self.final_contours))
        
        if show_scale:
            img_display = self.add_scale_bar(img_display)
        
        self.display_image(img_display, self.final_canvas)
    
    def display_overlay(self):
        """Display original image with contours overlay."""
        if self.original_image is None or self.final_contours is None:
            return
        
        display_vars = self.tab_display_vars.get("Overlay", {})
        show_labels = display_vars.get('show_labels', tk.BooleanVar(value=True)).get()
        show_scale = display_vars.get('show_scale', tk.BooleanVar(value=True)).get()
        
        img_display = self.original_image.copy()
        
        color_map = {
            "Green": (0, 255, 0), "Red": (255, 0, 0), "Blue": (0, 0, 255),
            "Yellow": (255, 255, 0), "Cyan": (0, 255, 255), "Magenta": (255, 0, 255)
        }
        color = color_map.get(self.overlay_contour_color_var.get(), (255, 0, 0))
        thickness = self.overlay_line_thickness_var.get()
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        cv2.drawContours(img_bgr, self.final_contours, -1, color, thickness)
        img_display = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if show_labels:
            img_display = self.add_labels_to_image(img_display, self.original_image, 
                                                   num_contours=len(self.final_contours))
        
        if show_scale:
            img_display = self.add_scale_bar(img_display)
        
        self.display_image(img_display, self.overlay_canvas)
    
    def export_results(self):
        """Export measurement results to CSV."""
        if self.measurements is None or self.measurements.empty:
            messagebox.showwarning("No Results", "No measurements to export. Please process an image first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.measurements.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
    
    def batch_process(self):
        """Process all images in the current folder."""
        if not self.image_files:
            messagebox.showwarning("No Folder", "Please select a folder first.")
            return
        
        output_folder = filedialog.askdirectory(title="Select Output Folder")
        if not output_folder:
            return
        
        output_path = Path(output_folder)
        all_measurements = []
        
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Batch Processing")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        
        ttk.Label(progress_window, text="Processing images...").pack(pady=10)
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, padx=20, pady=10)
        
        status_label = ttk.Label(progress_window, text="")
        status_label.pack(pady=5)
        
        def process_batch():
            for i, image_file in enumerate(self.image_files):
                status_label.config(text=f"Processing {image_file.name}")
                progress_var.set((i / len(self.image_files)) * 100)
                progress_window.update()
                
                # Load and process image
                self.load_image(image_file)
                self.process_image()
                
                # Save results
                if self.measurements is not None and not self.measurements.empty:
                    measurements_copy = self.measurements.copy()
                    measurements_copy['Image_File'] = image_file.name
                    all_measurements.append(measurements_copy)
                
                # Save overlay image
                if self.final_contours is not None:
                    # If the image failed to load, self.original_image may be None; skip overlay in that case
                    if self.original_image is None:
                        continue
                    overlay_img = self.original_image.copy()
                    img_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
                    cv2.drawContours(img_bgr, self.final_contours, -1, (0, 0, 255), 2)
                    overlay_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    overlay_img = self.add_labels_to_image(overlay_img, self.original_image, 
                                                           num_contours=len(self.final_contours))
                    overlay_img = self.add_scale_bar(overlay_img)
                    
                    output_file = output_path / f"{image_file.stem}_detected{image_file.suffix}"
                    cv2.imwrite(str(output_file), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
            
            # Combine all measurements
            if all_measurements:
                combined_df = pd.concat(all_measurements, ignore_index=True)
                csv_path = output_path / "batch_results.csv"
                combined_df.to_csv(csv_path, index=False)
            
            progress_var.set(100)
            status_label.config(text="Processing complete!")
            progress_window.after(2000, progress_window.destroy)
            messagebox.showinfo("Batch Complete", 
                              f"Processed {len(self.image_files)} images.\nResults saved to {output_path}")
        
        progress_window.after(100, process_batch)

def main():
    root = tk.Tk()
    app = ParticleDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()