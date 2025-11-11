import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
from scipy import ndimage

class BacteriaSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bacteria Segmentation Tool")
        self.root.geometry("1500x950")
        
        # Image variables
        self.current_folder = None
        self.image_files = []
        self.current_image_index = 0
        self.bf_image = None
        self.fluor_image = None
        self.current_filename = ""
        
        # Segmentation results
        self.contours = []
        self.stats_data = []
        self.bacteria_index = -1
        
        # Probe point
        self.probe_point = None
        self.probe_marker = None
        
        # Default parameters
        self.use_clahe = tk.BooleanVar(value=True)
        self.clahe_clip = tk.DoubleVar(value=2.0)
        self.clahe_tile = tk.IntVar(value=8)
        
        self.use_otsu = tk.BooleanVar(value=True)
        self.manual_threshold = tk.IntVar(value=128)
        
        self.open_kernel = tk.IntVar(value=3)
        self.open_iter = tk.IntVar(value=2)
        self.close_kernel = tk.IntVar(value=5)
        self.close_iter = tk.IntVar(value=2)
        
        self.watershed_dilate = tk.IntVar(value=3)
        self.min_area = tk.IntVar(value=50)
        self.min_fluor_per_area = tk.DoubleVar(value=0.0)
        
        self.gamma_value = tk.DoubleVar(value=1.0)
        self.brightness_value = tk.DoubleVar(value=1.0)
        
        # Create UI
        self.create_ui()
        
        # Status
        self.update_status("Click 'Load Folder' to start")
    
    def create_ui(self):
        # Main container
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel
        left_panel = tk.Frame(main_container, width=300, relief=tk.RAISED, borderwidth=1)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        # Right panel with tabs
        right_panel = tk.Frame(main_container)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.create_left_panel(left_panel)
        self.create_right_panel(right_panel)
    
    def create_left_panel(self, parent):
        # Scrollable frame
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # File operations
        file_frame = tk.LabelFrame(scrollable_frame, text="File Operations", font=("Arial", 9, "bold"))
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(file_frame, text="Load Folder", command=self.load_folder, width=20).pack(pady=3)
        
        # Display current folder path
        self.folder_path_label = tk.Label(file_frame, text="No folder loaded", 
                                          font=("Arial", 7), wraplength=280, justify=tk.LEFT)
        self.folder_path_label.pack(pady=3)
        
        nav_frame = tk.Frame(file_frame)
        nav_frame.pack(pady=3)
        self.prev_btn = tk.Button(nav_frame, text="◀ Previous", command=self.prev_image, width=10)
        self.prev_btn.pack(side=tk.LEFT, padx=2)
        self.next_btn = tk.Button(nav_frame, text="Next ▶", command=self.next_image, width=10)
        self.next_btn.pack(side=tk.LEFT, padx=2)
        
        self.image_counter_label = tk.Label(file_frame, text="0/0", font=("Arial", 9))
        self.image_counter_label.pack(pady=3)
        
        # Quick statistics
        stats_frame = tk.LabelFrame(scrollable_frame, text="Quick Statistics", font=("Arial", 9, "bold"))
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.quick_stats_text = tk.Label(stats_frame, text="No data", justify=tk.LEFT, font=("Arial", 8))
        self.quick_stats_text.pack(padx=5, pady=5, anchor="w")
        
        # Bacteria navigation
        bacteria_frame = tk.LabelFrame(scrollable_frame, text="Bacteria Navigation", font=("Arial", 9, "bold"))
        bacteria_frame.pack(fill=tk.X, padx=5, pady=5)
        
        bact_nav_frame = tk.Frame(bacteria_frame)
        bact_nav_frame.pack(pady=3)
        self.prev_bact_btn = tk.Button(bact_nav_frame, text="◀ Previous", command=self.prev_bacterium, width=10)
        self.prev_bact_btn.pack(side=tk.LEFT, padx=2)
        self.next_bact_btn = tk.Button(bact_nav_frame, text="Next ▶", command=self.next_bacterium, width=10)
        self.next_bact_btn.pack(side=tk.LEFT, padx=2)
        
        # CLAHE parameters
        clahe_frame = tk.LabelFrame(scrollable_frame, text="CLAHE Enhancement", font=("Arial", 9, "bold"))
        clahe_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Checkbutton(clahe_frame, text="Enable CLAHE", variable=self.use_clahe, 
                      command=self.on_param_change).pack(anchor="w", padx=5)
        
        self.create_param_control(clahe_frame, "Clip Limit:", self.clahe_clip, 0.1, 10.0, 0.1)
        self.create_param_control(clahe_frame, "Tile Size:", self.clahe_tile, 4, 16, 1, is_int=True)
        
        # Threshold parameters
        thresh_frame = tk.LabelFrame(scrollable_frame, text="Thresholding", font=("Arial", 9, "bold"))
        thresh_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Checkbutton(thresh_frame, text="Use Otsu Threshold", variable=self.use_otsu,
                      command=self.on_param_change).pack(anchor="w", padx=5)
        
        self.create_param_control(thresh_frame, "Manual Threshold:", self.manual_threshold, 0, 255, 1, is_int=True)
        
        # Morphology parameters
        morph_frame = tk.LabelFrame(scrollable_frame, text="Morphological Operations", font=("Arial", 9, "bold"))
        morph_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.create_param_control(morph_frame, "Opening Kernel:", self.open_kernel, 1, 15, 2, is_int=True, odd_only=True)
        self.create_param_control(morph_frame, "Opening Iterations:", self.open_iter, 1, 10, 1, is_int=True)
        self.create_param_control(morph_frame, "Closing Kernel:", self.close_kernel, 1, 15, 2, is_int=True, odd_only=True)
        self.create_param_control(morph_frame, "Closing Iterations:", self.close_iter, 1, 10, 1, is_int=True)
        
        # Watershed parameters
        water_frame = tk.LabelFrame(scrollable_frame, text="Watershed Segmentation", font=("Arial", 9, "bold"))
        water_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.create_param_control(water_frame, "Watershed Dilate:", self.watershed_dilate, 1, 20, 1, is_int=True)
        
        # Filtering parameters
        filter_frame = tk.LabelFrame(scrollable_frame, text="Filtering", font=("Arial", 9, "bold"))
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.create_param_control(filter_frame, "Min Area (pixels):", self.min_area, 10, 1000, 10, is_int=True)
        self.create_param_control(filter_frame, "Min Fluor/Area:", self.min_fluor_per_area, 0.0, 10.0, 0.1)
        
        # Fluorescence display
        fluor_frame = tk.LabelFrame(scrollable_frame, text="Fluorescence Display", font=("Arial", 9, "bold"))
        fluor_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.create_param_control(fluor_frame, "Gamma:", self.gamma_value, 0.1, 3.0, 0.1)
        self.create_param_control(fluor_frame, "Brightness:", self.brightness_value, 0.1, 3.0, 0.1)
        
        # Measurement panel
        measure_frame = tk.LabelFrame(scrollable_frame, text="Point Measurement", font=("Arial", 9, "bold"))
        measure_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.measure_text = tk.Label(measure_frame, text="Click on image to measure\n(Ctrl+Click to auto-tune)", 
                                     justify=tk.LEFT, font=("Arial", 8))
        self.measure_text.pack(padx=5, pady=5, anchor="w")
        
        # Control buttons
        control_frame = tk.Frame(scrollable_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=10)
        
        tk.Button(control_frame, text="Reset to Defaults", command=self.reset_defaults, width=20).pack(pady=3)
        tk.Button(control_frame, text="Exit", command=self.exit_app, width=20).pack(pady=3)
        
        # Status bar
        self.status_label = tk.Label(scrollable_frame, text="Ready", relief=tk.SUNKEN, anchor="w",
                                     font=("Arial", 8), bg="lightgray")
        self.status_label.pack(fill=tk.X, padx=5, pady=5)
    
    def create_param_control(self, parent, label, variable, min_val, max_val, step, is_int=False, odd_only=False):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=3)
        
        tk.Label(frame, text=label, font=("Arial", 8)).pack(side=tk.LEFT)
        
        entry = tk.Entry(frame, textvariable=variable, width=8, font=("Arial", 8))
        entry.pack(side=tk.RIGHT)
        entry.bind("<Return>", lambda e: self.on_param_change())
        
        if is_int:
            scale = ttk.Scale(frame, from_=min_val, to=max_val, variable=variable,
                            command=lambda v: self.on_scale_change(variable, float(v), is_int, odd_only))
        else:
            scale = ttk.Scale(frame, from_=min_val, to=max_val, variable=variable,
                            command=lambda v: self.on_scale_change(variable, float(v), is_int, odd_only))
        
        scale.pack(fill=tk.X, padx=5)
    
    def on_scale_change(self, variable, value, is_int, odd_only):
        if is_int:
            value = int(round(value))
            if odd_only and value % 2 == 0:
                value += 1
        variable.set(value)
        self.on_param_change()
    
    def on_param_change(self):
        if self.bf_image is not None:
            self.process_and_display()
    
    def create_right_panel(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create 8 tabs
        self.tab_original = tk.Frame(self.notebook)
        self.tab_fluor = tk.Frame(self.notebook)
        self.tab_clahe = tk.Frame(self.notebook)
        self.tab_thresh = tk.Frame(self.notebook)
        self.tab_morph = tk.Frame(self.notebook)
        self.tab_contours = tk.Frame(self.notebook)
        self.tab_overlay = tk.Frame(self.notebook)
        self.tab_stats = tk.Frame(self.notebook)
        
        self.notebook.add(self.tab_original, text="1. Original (BF)")
        self.notebook.add(self.tab_fluor, text="2. Fluorescence")
        self.notebook.add(self.tab_clahe, text="3. CLAHE Enhanced")
        self.notebook.add(self.tab_thresh, text="4. Threshold")
        self.notebook.add(self.tab_morph, text="5. Morphology")
        self.notebook.add(self.tab_contours, text="6. Final Contours")
        self.notebook.add(self.tab_overlay, text="7. BF+Fluor Overlay")
        self.notebook.add(self.tab_stats, text="8. Statistics List")
        
        # Create canvases for image display
        self.canvas_original = self.create_canvas(self.tab_original)
        self.canvas_fluor = self.create_canvas(self.tab_fluor)
        self.canvas_clahe = self.create_canvas(self.tab_clahe)
        self.canvas_thresh = self.create_canvas(self.tab_thresh)
        self.canvas_morph = self.create_canvas(self.tab_morph)
        self.canvas_contours = self.create_canvas(self.tab_contours)
        self.canvas_overlay = self.create_canvas(self.tab_overlay)
        
        # Bind click events to original canvas
        self.canvas_original.bind("<Button-1>", self.on_canvas_click)
        self.canvas_original.bind("<Control-Button-1>", self.on_ctrl_click)
        self.canvas_original.bind("<Button-3>", self.clear_probe)
        
        # Create statistics table
        self.create_stats_table(self.tab_stats)
    
    def create_canvas(self, parent):
        canvas = tk.Canvas(parent, bg="gray")
        canvas.pack(fill=tk.BOTH, expand=True)
        return canvas
    
    def create_stats_table(self, parent):
        # Create treeview for statistics
        columns = ("Index", "Area", "Mean Fluor", "Total Fluor", "Fluor/Area")
        self.stats_tree = ttk.Treeview(parent, columns=columns, show="headings", height=20)
        
        for col in columns:
            self.stats_tree.heading(col, text=col, command=lambda c=col: self.sort_stats(c))
            self.stats_tree.column(col, width=100, anchor="center")
        
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=scrollbar.set)
        
        self.stats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def sort_stats(self, column):
        # Simple sorting (can be enhanced)
        pass
    
    def load_folder(self):
        # Force the dialog to start fresh - use home directory
        initial_dir = os.path.expanduser("~")
        
        folder_path = filedialog.askdirectory(
            title="Select Folder Containing Images",
            initialdir=initial_dir,
            mustexist=True,
            parent=self.root
        )
        
        if not folder_path:
            return
        
        print(f"DEBUG: Selected folder path: {folder_path}")
        
        # Scan for .tif files directly in selected folder
        try:
            all_files = os.listdir(folder_path)
            print(f"DEBUG: Files in folder: {all_files[:10]}...")  # Show first 10 files
        except Exception as e:
            messagebox.showerror("Error", f"Cannot read folder: {str(e)}")
            return
        
        tif_files = [f for f in all_files
                     if f.lower().endswith(('.tif', '.tiff')) 
                     and '_ch00' in f 
                     and not f.startswith('.')]
        
        print(f"DEBUG: Found {len(tif_files)} _ch00 .tif files directly in folder")
        
        # If no files found directly, check subfolders and let user choose
        if not tif_files:
            subfolders = [item for item in all_files
                          if os.path.isdir(os.path.join(folder_path, item)) 
                          and not item.startswith('.')]
            
            print(f"DEBUG: Found {len(subfolders)} subfolders: {subfolders}")
            
            if subfolders:
                # Show dialog to select subfolder
                selected_subfolder = self.choose_subfolder(subfolders)
                
                if not selected_subfolder:
                    return  # User cancelled
                
                folder_path = os.path.join(folder_path, selected_subfolder)
                
                print(f"DEBUG: User selected subfolder: {folder_path}")
                
                tif_files = [f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.tif', '.tiff')) 
                             and '_ch00' in f 
                             and not f.startswith('.')]
                
                print(f"DEBUG: Found {len(tif_files)} _ch00 .tif files in subfolder")
        
        if not tif_files:
            messagebox.showwarning("No Images", 
                                  f"No valid .tif files with '_ch00' found in:\n{folder_path}")
            return
        
        # Continue with loading images
        self.current_folder = folder_path
        self.image_files = sorted(tif_files)
        self.current_image_index = 0
        
        # Update folder path display
        self.folder_path_label.config(text=f"Folder: {folder_path}")
        
        self.load_current_image()
        
        self.update_status(f"Loaded {len(self.image_files)} images from {os.path.basename(folder_path)}")
        self.update_navigation_buttons()
    
    def choose_subfolder(self, subfolders):
        """Show dialog for user to choose which subfolder to use"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Subfolder")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        selected_folder = [None]  # Use list to allow modification in nested function
        
        tk.Label(dialog, text="Multiple subfolders found.\nPlease select which one to use:",
                font=("Arial", 10, "bold")).pack(pady=10)
        
        # Create listbox with scrollbar
        frame = tk.Frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, font=("Arial", 10))
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Add subfolders to listbox
        for folder in sorted(subfolders):
            listbox.insert(tk.END, folder)
        
        # Select first item by default
        listbox.selection_set(0)
        listbox.activate(0)
        
        def on_ok():
            selection = listbox.curselection()
            if selection:
                selected_folder[0] = listbox.get(selection[0])
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        def on_double_click(event):
            on_ok()
        
        listbox.bind("<Double-Button-1>", on_double_click)
        
        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="OK", command=on_ok, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side=tk.LEFT, padx=5)
        
        # Wait for dialog to close
        self.root.wait_window(dialog)
        
        return selected_folder[0]
    
    def load_current_image(self):
        if not self.image_files:
            return
        
        filename = self.image_files[self.current_image_index]
        self.current_filename = filename
        
        # Load bright-field image
        bf_path = os.path.join(self.current_folder, filename)
        print(f"DEBUG: Loading BF image: {bf_path}")
        self.bf_image = cv2.imread(bf_path, cv2.IMREAD_GRAYSCALE)
        
        if self.bf_image is None:
            messagebox.showerror("Error", f"Failed to load image: {filename}")
            return
        
        # Load fluorescence image
        fluor_filename = filename.replace('_ch00', '_ch01')
        fluor_path = os.path.join(self.current_folder, fluor_filename)
        
        print(f"DEBUG: Looking for fluorescence image: {fluor_path}")
        if os.path.exists(fluor_path):
            self.fluor_image = cv2.imread(fluor_path, cv2.IMREAD_GRAYSCALE)
            print(f"DEBUG: Fluorescence image loaded successfully")
        else:
            self.fluor_image = None
            print(f"DEBUG: Fluorescence image not found")
        
        # Reset probe and bacteria index
        self.probe_point = None
        self.bacteria_index = -1
        self.clear_probe(None)
        
        # Update title
        self.root.title(f"Bacteria Segmentation Tool - {filename}")
        
        # Update counter
        self.image_counter_label.config(text=f"{self.current_image_index + 1}/{len(self.image_files)}")
        
        # Process and display
        self.process_and_display()
    
    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
            self.update_navigation_buttons()
    
    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
            self.update_navigation_buttons()
    
    def update_navigation_buttons(self):
        self.prev_btn.config(state=tk.NORMAL if self.current_image_index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_image_index < len(self.image_files) - 1 else tk.DISABLED)
        
        self.prev_bact_btn.config(state=tk.NORMAL if len(self.contours) > 0 else tk.DISABLED)
        self.next_bact_btn.config(state=tk.NORMAL if len(self.contours) > 0 else tk.DISABLED)
    
    def process_and_display(self):
        if self.bf_image is None:
            return
        
        # Step 1: CLAHE Enhancement
        if self.use_clahe.get():
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip.get(), 
                                   tileGridSize=(self.clahe_tile.get(), self.clahe_tile.get()))
            enhanced = clahe.apply(self.bf_image)
        else:
            enhanced = self.bf_image.copy()
        
        # Step 2: Thresholding
        if self.use_otsu.get():
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(enhanced, self.manual_threshold.get(), 255, cv2.THRESH_BINARY_INV)
        
        # Step 3: Morphological operations
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                (self.open_kernel.get(), self.open_kernel.get()))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=self.open_iter.get())
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (self.close_kernel.get(), self.close_kernel.get()))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=self.close_iter.get())
        
        # Step 4: Watershed segmentation
        dist_transform = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
        
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                  (self.watershed_dilate.get(), self.watershed_dilate.get()))
        sure_fg = cv2.dilate(dist_transform, kernel_dilate, iterations=1)
        _, sure_fg = cv2.threshold(sure_fg, 0.3 * sure_fg.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        unknown = cv2.subtract(closed, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Convert to BGR for watershed
        bf_bgr = cv2.cvtColor(self.bf_image, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(bf_bgr, markers)
        
        # Step 5: Extract contours
        mask = np.uint8(markers > 1) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        self.contours = [c for c in contours if cv2.contourArea(c) >= self.min_area.get()]
        
        # Step 6: Calculate statistics and filter by fluorescence
        self.calculate_statistics()
        
        # Display all tabs
        self.display_original()
        self.display_fluorescence()
        self.display_clahe(enhanced)
        self.display_threshold(binary)
        self.display_morphology(closed)
        self.display_contours()
        self.display_overlay()
        self.display_statistics()
        
        # Update quick stats
        self.update_quick_stats()
        
        # Update navigation
        self.update_navigation_buttons()
    
    def calculate_statistics(self):
        self.stats_data = []
        
        if self.fluor_image is None:
            for i, contour in enumerate(self.contours):
                area = cv2.contourArea(contour)
                self.stats_data.append({
                    'index': i,
                    'area': area,
                    'mean_fluor': 0,
                    'total_fluor': 0,
                    'fluor_per_area': 0
                })
            return
        
        # Filter contours by fluorescence
        filtered_contours = []
        
        for i, contour in enumerate(self.contours):
            mask = np.zeros(self.bf_image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            area = cv2.contourArea(contour)
            fluor_values = self.fluor_image[mask > 0]
            
            if len(fluor_values) > 0:
                mean_fluor = np.mean(fluor_values)
                total_fluor = np.sum(fluor_values)
                fluor_per_area = total_fluor / area if area > 0 else 0
                
                if fluor_per_area >= self.min_fluor_per_area.get():
                    filtered_contours.append(contour)
                    self.stats_data.append({
                        'index': len(filtered_contours) - 1,
                        'area': area,
                        'mean_fluor': mean_fluor,
                        'total_fluor': total_fluor,
                        'fluor_per_area': fluor_per_area
                    })
        
        self.contours = filtered_contours
    
    def display_original(self):
        img_display = cv2.cvtColor(self.bf_image, cv2.COLOR_GRAY2BGR)
        
        # Draw probe point
        if self.probe_point is not None:
            x, y = self.probe_point
            cv2.drawMarker(img_display, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        
        self.display_image(self.canvas_original, img_display)
    
    def display_fluorescence(self):
        if self.fluor_image is None:
            self.canvas_fluor.delete("all")
            self.canvas_fluor.create_text(self.canvas_fluor.winfo_width() // 2,
                                         self.canvas_fluor.winfo_height() // 2,
                                         text="No fluorescence image", fill="white", font=("Arial", 14))
            return
        
        # Apply gamma and brightness
        fluor_adjusted = self.fluor_image.astype(np.float32)
        fluor_adjusted = np.power(fluor_adjusted / 255.0, 1.0 / self.gamma_value.get()) * 255.0
        fluor_adjusted = fluor_adjusted * self.brightness_value.get()
        fluor_adjusted = np.clip(fluor_adjusted, 0, 255).astype(np.uint8)
        
        # Display as red channel
        fluor_bgr = np.zeros((fluor_adjusted.shape[0], fluor_adjusted.shape[1], 3), dtype=np.uint8)
        fluor_bgr[:, :, 2] = fluor_adjusted
        
        self.display_image(self.canvas_fluor, fluor_bgr)
    
    def display_clahe(self, enhanced):
        img_display = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        self.display_image(self.canvas_clahe, img_display)
    
    def display_threshold(self, binary):
        img_display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.display_image(self.canvas_thresh, img_display)
    
    def display_morphology(self, morph):
        img_display = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
        self.display_image(self.canvas_morph, img_display)
    
    def display_contours(self):
        img_display = cv2.cvtColor(self.bf_image, cv2.COLOR_GRAY2BGR)
        
        # Draw all contours in green
        cv2.drawContours(img_display, self.contours, -1, (0, 255, 0), 2)
        
        # Highlight selected bacterium in red
        if 0 <= self.bacteria_index < len(self.contours):
            cv2.drawContours(img_display, [self.contours[self.bacteria_index]], -1, (0, 0, 255), 3)
        
        # Add smart labels
        self.add_smart_labels(img_display)
        
        self.display_image(self.canvas_contours, img_display)
    
    def add_smart_labels(self, img):
        if len(self.contours) == 0:
            return
        
        h, w = img.shape[:2]
        occupancy = np.zeros((h, w), dtype=np.uint8)
        
        # Mark bacteria positions with margin
        for contour in self.contours:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
            mask = cv2.dilate(mask, kernel)
            occupancy = cv2.bitwise_or(occupancy, mask)
        
        # Place labels
        for i, contour in enumerate(self.contours):
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Find best label position
            best_score = float('inf')
            best_angle = 0
            
            for angle_deg in range(0, 360, 22):
                angle_rad = np.deg2rad(angle_deg)
                distance = 50
                
                ex = int(cx + distance * np.cos(angle_rad))
                ey = int(cy + distance * np.sin(angle_rad))
                
                if ex < 0 or ex >= w - 30 or ey < 0 or ey >= h - 20:
                    continue
                
                # Check occupancy
                score = np.sum(occupancy[max(0, ey-10):min(h, ey+10), max(0, ex-15):min(w, ex+15)])
                
                if score < best_score:
                    best_score = score
                    best_angle = angle_rad
            
            # Draw arrow and label
            ex = int(cx + 50 * np.cos(best_angle))
            ey = int(cy + 50 * np.sin(best_angle))
            
            cv2.arrowedLine(img, (cx, cy), (ex, ey), (255, 255, 0), 1, tipLength=0.3)
            
            label = str(i + 1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(img, (ex - 2, ey - th - 2), (ex + tw + 2, ey + 2), (0, 0, 0), -1)
            cv2.putText(img, label, (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def display_overlay(self):
        if self.fluor_image is None:
            self.canvas_overlay.delete("all")
            self.canvas_overlay.create_text(self.canvas_overlay.winfo_width() // 2,
                                           self.canvas_overlay.winfo_height() // 2,
                                           text="No fluorescence image", fill="white", font=("Arial", 14))
            return
        
        # Create overlay
        img_display = cv2.cvtColor(self.bf_image, cv2.COLOR_GRAY2BGR)
        
        # Apply gamma to fluorescence
        fluor_adjusted = self.fluor_image.astype(np.float32)
        fluor_adjusted = np.power(fluor_adjusted / 255.0, 1.0 / self.gamma_value.get()) * 255.0
        fluor_adjusted = fluor_adjusted * self.brightness_value.get()
        fluor_adjusted = np.clip(fluor_adjusted, 0, 255).astype(np.uint8)
        
        # Overlay fluorescence in red channel
        img_display[:, :, 2] = cv2.addWeighted(img_display[:, :, 2], 0.5, fluor_adjusted, 0.5, 0)
        
        # Draw contours in yellow
        cv2.drawContours(img_display, self.contours, -1, (0, 255, 255), 2)
        
        self.display_image(self.canvas_overlay, img_display)
    
    def display_statistics(self):
        # Clear existing items
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        
        # Add data
        for stat in self.stats_data:
            self.stats_tree.insert("", "end", values=(
                stat['index'] + 1,
                f"{stat['area']:.1f}",
                f"{stat['mean_fluor']:.2f}",
                f"{stat['total_fluor']:.1f}",
                f"{stat['fluor_per_area']:.3f}"
            ))
    
    def display_image(self, canvas, img):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Resize to fit canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img_pil.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(img_pil)
        
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=img_tk)
        canvas.image = img_tk  # Keep a reference
    
    def on_canvas_click(self, event):
        if self.bf_image is None:
            return
        
        # Get canvas and image dimensions
        canvas_width = self.canvas_original.winfo_width()
        canvas_height = self.canvas_original.winfo_height()
        
        img_height, img_width = self.bf_image.shape
        
        # Calculate scale
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        # Calculate offset (image is centered)
        offset_x = (canvas_width - img_width * scale) // 2
        offset_y = (canvas_height - img_height * scale) // 2
        
        # Convert canvas coordinates to image coordinates
        img_x = int((event.x - offset_x) / scale)
        img_y = int((event.y - offset_y) / scale)
        
        # Validate coordinates
        if img_x < 0 or img_x >= img_width or img_y < 0 or img_y >= img_height:
            return
        
        self.probe_point = (img_x, img_y)
        
        # Update measurement display
        pixel_value = self.bf_image[img_y, img_x]
        
        # Check if inside any contour
        inside_contour = False
        contour_area = 0
        
        for i, contour in enumerate(self.contours):
            if cv2.pointPolygonTest(contour, (float(img_x), float(img_y)), False) >= 0:
                inside_contour = True
                contour_area = cv2.contourArea(contour)
                self.bacteria_index = i
                break
        
        measure_info = f"Coordinates: ({img_x}, {img_y})\n"
        measure_info += f"Pixel Value: {pixel_value}\n"
        measure_info += f"Inside Contour: {'Yes' if inside_contour else 'No'}\n"
        if inside_contour:
            measure_info += f"Contour Area: {contour_area:.1f} pixels"
        
        self.measure_text.config(text=measure_info)
        
        # Update display
        self.process_and_display()
        
        # Update status
        if inside_contour and self.bacteria_index < len(self.stats_data):
            stat = self.stats_data[self.bacteria_index]
            status = f"Bacterium #{stat['index']+1}: Area={stat['area']:.1f}, Fluor/Area={stat['fluor_per_area']:.3f}"
            self.update_status(status)
    
    def on_ctrl_click(self, event):
        if self.bf_image is None:
            return
        
        # Get image coordinates (same as on_canvas_click)
        canvas_width = self.canvas_original.winfo_width()
        canvas_height = self.canvas_original.winfo_height()
        img_height, img_width = self.bf_image.shape
        scale = min(canvas_width / img_width, canvas_height / img_height)
        offset_x = (canvas_width - img_width * scale) // 2
        offset_y = (canvas_height - img_height * scale) // 2
        img_x = int((event.x - offset_x) / scale)
        img_y = int((event.y - offset_y) / scale)
        
        if img_x < 0 or img_x >= img_width or img_y < 0 or img_y >= img_height:
            return
        
        # Check if inside contour
        for contour in self.contours:
            if cv2.pointPolygonTest(contour, (float(img_x), float(img_y)), False) >= 0:
                pixel_value = self.bf_image[img_y, img_x]
                area = cv2.contourArea(contour)
                
                # Auto-tune parameters
                self.use_otsu.set(False)
                self.manual_threshold.set(max(0, pixel_value - 18))
                self.min_area.set(int(area * 0.75))
                self.watershed_dilate.set(14)
                
                self.update_status("Auto-tuned parameters based on selected bacterium")
                self.process_and_display()
                return
        
        messagebox.showinfo("Auto-tune", "Please Ctrl+Click inside a bacterium")
    
    def clear_probe(self, event):
        self.probe_point = None
        self.bacteria_index = -1
        self.measure_text.config(text="Click on image to measure\n(Ctrl+Click to auto-tune)")
        
        if self.bf_image is not None:
            self.process_and_display()
    
    def prev_bacterium(self):
        if len(self.contours) == 0:
            return
        
        self.bacteria_index = (self.bacteria_index - 1) % len(self.contours)
        self.goto_bacterium(self.bacteria_index)
    
    def next_bacterium(self):
        if len(self.contours) == 0:
            return
        
        self.bacteria_index = (self.bacteria_index + 1) % len(self.contours)
        self.goto_bacterium(self.bacteria_index)
    
    def goto_bacterium(self, index):
        if 0 <= index < len(self.contours):
            contour = self.contours[index]
            M = cv2.moments(contour)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                self.probe_point = (cx, cy)
                
                # Update measurement
                pixel_value = self.bf_image[cy, cx]
                area = cv2.contourArea(contour)
                
                measure_info = f"Bacterium #{index + 1}\n"
                measure_info += f"Coordinates: ({cx}, {cy})\n"
                measure_info += f"Pixel Value: {pixel_value}\n"
                measure_info += f"Area: {area:.1f} pixels"
                
                self.measure_text.config(text=measure_info)
                
                # Update status
                if index < len(self.stats_data):
                    stat = self.stats_data[index]
                    status = f"Bacterium #{stat['index']+1}: Area={stat['area']:.1f}, Fluor/Area={stat['fluor_per_area']:.3f}"
                    self.update_status(status)
                
                self.process_and_display()
    
    def update_quick_stats(self):
        if len(self.stats_data) == 0:
            self.quick_stats_text.config(text="No bacteria detected")
            return
        
        total = len(self.stats_data)
        avg_area = np.mean([s['area'] for s in self.stats_data])
        avg_fluor = np.mean([s['fluor_per_area'] for s in self.stats_data])
        
        stats_text = f"Total Bacteria: {total}\n"
        if self.bacteria_index >= 0:
            stats_text += f"Current: #{self.bacteria_index + 1}\n"
        stats_text += f"Avg Area: {avg_area:.1f} pixels\n"
        stats_text += f"Avg Fluor/Area: {avg_fluor:.3f}"
        
        self.quick_stats_text.config(text=stats_text)
    
    def reset_defaults(self):
        if messagebox.askyesno("Reset", "Reset all parameters to defaults?"):
            self.use_clahe.set(True)
            self.clahe_clip.set(2.0)
            self.clahe_tile.set(8)
            
            self.use_otsu.set(True)
            self.manual_threshold.set(128)
            
            self.open_kernel.set(3)
            self.open_iter.set(2)
            self.close_kernel.set(5)
            self.close_iter.set(2)
            
            self.watershed_dilate.set(3)
            self.min_area.set(50)
            self.min_fluor_per_area.set(0.0)
            
            self.gamma_value.set(1.0)
            self.brightness_value.set(1.0)
            
            self.probe_point = None
            self.bacteria_index = -1
            
            self.update_status("Parameters reset to defaults")
            
            if self.bf_image is not None:
                self.process_and_display()
    
    def update_status(self, message):
        self.status_label.config(text=message)
    
    def exit_app(self):
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = BacteriaSegmentationApp(root)
    root.mainloop()