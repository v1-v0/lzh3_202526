"""
Cross-Platform Clinical Results Viewer
Compatible with Windows, macOS, and Linux
Now includes Processing Steps Viewer with Control Group Support
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
from pathlib import Path
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sys
from typing import Optional, Dict, List
import webbrowser

# Try to import reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ClinicalResultsViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Particle-Scout Clinical Results Viewer")
        self.root.geometry("1920x1080")
        
        # Data storage
        self.current_folder: Optional[Path] = None
        self.clinical_data: Optional[pd.DataFrame] = None
        self.gplus_data: Optional[pd.DataFrame] = None
        self.gminus_data: Optional[pd.DataFrame] = None
        self.selected_group: Optional[str] = None
        self.dataset_name: str = ""
        
        # Photo reference storage - prevents garbage collection
        self.photo_refs: Dict[int, ImageTk.PhotoImage] = {}
        
        # Processing steps viewer - image storage
        self.current_pos_image: Optional[Image.Image] = None
        self.current_neg_image: Optional[Image.Image] = None
        self.current_pos_path: Optional[Path] = None
        self.current_neg_path: Optional[Path] = None
        self.photo_positive: Optional[ImageTk.PhotoImage] = None
        self.photo_negative: Optional[ImageTk.PhotoImage] = None
        
        # Color scheme
        self.colors = {
            'positive': '#ffcccc',      # Light red
            'negative': '#ccffcc',      # Light green
            'no_bacteria': '#ffffcc',   # Light yellow
            'mixed': '#ffd4b4',         # Light orange
            'missing': '#e0e0e0',       # Light gray
            'control': '#e0e0ff',       # Light blue
            'header': '#4472c4',        # Blue
            'text': '#000000',          # Black
        }
        
        # Configure root window
        self.setup_styles()
        
        # Create UI
        self.create_menu()
        self.create_toolbar()
        self.create_main_panels()
        self.create_status_bar()
        
        # Try to auto-load if run from outputs directory
        self.auto_load_recent()
    
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        
        # Try to use modern theme if available
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'alt' in available_themes:
            style.theme_use('alt')
        
        # Configure custom styles
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Status.TLabel', font=('Arial', 9))
    
    def create_menu(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Output Folder...", 
                             command=self.load_results, 
                             accelerator="Ctrl+O")
        file_menu.add_command(label="Open Recent Folder", 
                             command=self.auto_load_recent,
                             accelerator="Ctrl+R")
        file_menu.add_separator()
        file_menu.add_command(label="Refresh View", 
                             command=self.refresh_view,
                             accelerator="F5")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", 
                             command=self.root.quit,
                             accelerator="Ctrl+Q")
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Export menu
        export_menu = tk.Menu(menubar, tearoff=0)
        export_menu.add_command(label="Export Summary to CSV", 
                               command=self.export_csv)
        export_menu.add_command(label="Export Report to PDF", 
                               command=self.export_pdf)
        export_menu.add_separator()
        export_menu.add_command(label="Open Output Folder", 
                               command=self.open_output_folder)
        menubar.add_cascade(label="Export", menu=export_menu)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Expand All Groups", 
                             command=self.expand_all)
        view_menu.add_command(label="Collapse All Groups", 
                             command=self.collapse_all)
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", 
                             command=self.show_about)
        help_menu.add_command(label="User Guide", 
                             command=self.show_help)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.load_results())
        self.root.bind('<Control-r>', lambda e: self.auto_load_recent())
        self.root.bind('<F5>', lambda e: self.refresh_view())
        self.root.bind('<Control-q>', lambda e: self.root.quit())
    
    def create_toolbar(self):
        """Create toolbar with main actions"""
        toolbar = ttk.Frame(self.root, relief=tk.RAISED, borderwidth=1)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
        
        # Load button
        ttk.Button(toolbar, text="📁 Load Results", 
                command=self.load_results).pack(side=tk.LEFT, padx=2, pady=2)
        
        # Refresh button
        ttk.Button(toolbar, text="🔄 Refresh", 
                command=self.refresh_view).pack(side=tk.LEFT, padx=2, pady=2)
        
        # Separator
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)
        
        # Filter controls
        ttk.Label(toolbar, text="Filter:").pack(side=tk.LEFT, padx=(10,5))
        
        self.filter_var = tk.StringVar(value="All")
        filter_combo = ttk.Combobox(toolbar, textvariable=self.filter_var,
                                    values=["All", "POSITIVE", "NEGATIVE", 
                                        "NO OBVIOUS BACTERIA", "MIXED/CONTRADICTORY", "CONTROL"],
                                    state="readonly", width=20)
        filter_combo.pack(side=tk.LEFT, padx=2)
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_filter())
        
        # Separator
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)
        
        # Export buttons
        ttk.Button(toolbar, text="💾 Export CSV", 
                command=self.export_csv).pack(side=tk.LEFT, padx=2, pady=2)
        
        if REPORTLAB_AVAILABLE:
            ttk.Button(toolbar, text="📄 Export PDF", 
                    command=self.export_pdf).pack(side=tk.LEFT, padx=2, pady=2)
        
        # Separator
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)
        
        # Close button
        ttk.Button(toolbar, text="❌ Close", 
                command=self.close_application).pack(side=tk.LEFT, padx=2, pady=2)
        
        # Dataset label (right side)
        self.dataset_label = ttk.Label(toolbar, text="No dataset loaded", 
                                    style='Status.TLabel')
        self.dataset_label.pack(side=tk.RIGHT, padx=10)
    
    def close_application(self):
        """Close the application"""
        self.root.quit()

    def create_main_panels(self):
        """Create main content area with panels"""
        # Main container with paned window
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ========== Left Panel - Results Tree ==========
        left_panel = ttk.Frame(main_container, width=350)
        main_container.add(left_panel, weight=1)
        
        # Title
        ttk.Label(left_panel, text="Clinical Results", 
                 style='Header.TLabel').pack(pady=5, padx=5, anchor=tk.W)
        
        # Tree frame with scrollbar
        tree_frame = ttk.Frame(left_panel)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview
        self.tree = ttk.Treeview(tree_frame, 
                                columns=("type", "gplus", "gminus", "result"), 
                                show="tree headings",
                                selectmode="browse")
        
        self.tree.heading("#0", text="Group")
        self.tree.heading("type", text="Type")
        self.tree.heading("gplus", text="G+")
        self.tree.heading("gminus", text="G-")
        self.tree.heading("result", text="Result")
        
        self.tree.column("#0", width=100, minwidth=80)
        self.tree.column("type", width=70, minwidth=50)
        self.tree.column("gplus", width=50, minwidth=40)
        self.tree.column("gminus", width=50, minwidth=40)
        self.tree.column("result", width=80, minwidth=60)
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Bind selection event
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        
        # Configure tags for colors
        self.tree.tag_configure("positive", background=self.colors['positive'])
        self.tree.tag_configure("negative", background=self.colors['negative'])
        self.tree.tag_configure("no_bacteria", background=self.colors['no_bacteria'])
        self.tree.tag_configure("mixed", background=self.colors['mixed'])
        self.tree.tag_configure("missing", background=self.colors['missing'])
        self.tree.tag_configure("control", background=self.colors['control'])
        
        # Legend
        legend_frame = ttk.LabelFrame(left_panel, text="Legend", padding=5)
        legend_frame.pack(fill=tk.X, padx=5, pady=5)
        
        legends = [
            ("POSITIVE", self.colors['positive']),
            ("NEGATIVE", self.colors['negative']),
            ("NO BACTERIA", self.colors['no_bacteria']),
            ("MIXED", self.colors['mixed']),
            ("CONTROL", self.colors['control']),
        ]
        
        for text, color in legends:
            frame = tk.Frame(legend_frame, bg=color, relief=tk.SOLID, borderwidth=1)
            frame.pack(fill=tk.X, pady=2)
            tk.Label(frame, text=text, bg=color, font=('Arial', 8)).pack(pady=2)
        
        # ========== Right Panel - Details Notebook ==========
        right_panel = ttk.Notebook(main_container)
        main_container.add(right_panel, weight=3)
        
        # Tab 1: Overview
        self.overview_tab = ttk.Frame(right_panel)
        right_panel.add(self.overview_tab, text="📊 Overview")
        self.create_overview_tab()
        
        # Tab 2: G+ Details
        self.gplus_tab = ttk.Frame(right_panel)
        right_panel.add(self.gplus_tab, text="🔴 G+ Details")
        
        # Tab 3: G- Details
        self.gminus_tab = ttk.Frame(right_panel)
        right_panel.add(self.gminus_tab, text="🔵 G- Details")
        
        # Tab 4: Processing Steps Viewer
        self.processing_tab = ttk.Frame(right_panel)
        right_panel.add(self.processing_tab, text="🔬 Processing Steps")
        self.create_processing_steps_tab()
        
        # Tab 5: Comparison Plots
        self.plots_tab = ttk.Frame(right_panel)
        right_panel.add(self.plots_tab, text="📈 Plots")
        
        # Tab 6: Raw Data
        self.data_tab = ttk.Frame(right_panel)
        right_panel.add(self.data_tab, text="📋 Raw Data")
    
    def create_overview_tab(self):
        """Create overview tab content"""
        # Scrolled frame for overview
        canvas = tk.Canvas(self.overview_tab)
        scrollbar = ttk.Scrollbar(self.overview_tab, orient=tk.VERTICAL, 
                                 command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Content
        self.overview_content = scrollable_frame
        
        # Initial message
        ttk.Label(self.overview_content, 
                 text="Select a group from the list to view details",
                 font=('Arial', 11)).pack(pady=20)
    
    def create_processing_steps_tab(self):
        """Create the processing steps viewer tab"""
        # Main horizontal split
        main_split = ttk.PanedWindow(self.processing_tab, orient=tk.HORIZONTAL)
        main_split.pack(fill=tk.BOTH, expand=True)
        
        # Left: Processing steps tree
        left_frame = ttk.Frame(main_split, width=250)
        main_split.add(left_frame, weight=1)
        
        ttk.Label(left_frame, text="Processing Steps", 
                 style='Header.TLabel').pack(pady=5, padx=5, anchor=tk.W)
        
        # Steps tree with scrollbar
        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.steps_tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set,
                                       columns=('description',), show='tree',
                                       selectmode='browse')
        self.steps_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.config(command=self.steps_tree.yview)
        
        self.steps_tree.bind('<<TreeviewSelect>>', self.on_step_selected)
        
        # Populate processing steps
        self.populate_processing_steps()
        
        # Right: Dual image view (Positive | Negative)
        right_frame = ttk.Frame(main_split)
        main_split.add(right_frame, weight=3)
        
        # Split into Positive (left) and Negative (right)
        dual_split = ttk.PanedWindow(right_frame, orient=tk.HORIZONTAL)
        dual_split.pack(fill=tk.BOTH, expand=True)
        
        # === POSITIVE COLUMN ===
        positive_frame = ttk.Frame(dual_split)
        dual_split.add(positive_frame, weight=1)
        
        # Positive header
        pos_header = ttk.Frame(positive_frame)
        pos_header.pack(fill=tk.X, pady=(5, 5), padx=5)
        
        ttk.Label(pos_header, text="POSITIVE (G+)", font=('Arial', 11, 'bold'),
                 foreground='#C62828').pack(side=tk.LEFT)
        
        self.pos_title = ttk.Label(pos_header, text="", font=('Arial', 9))
        self.pos_title.pack(side=tk.LEFT, padx=(10, 0))
        
        # Positive canvas
        pos_canvas_frame = ttk.Frame(positive_frame, relief='sunken', borderwidth=2)
        pos_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5)
        
        self.pos_canvas = tk.Canvas(pos_canvas_frame, bg='#f0f0f0', cursor='cross')
        self.pos_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Positive controls
        pos_controls = ttk.Frame(positive_frame)
        pos_controls.pack(fill=tk.X, pady=(5, 0), padx=5)
        
        ttk.Button(pos_controls, text="Fit to Window", 
                  command=lambda: self.fit_to_window('positive')).pack(side=tk.LEFT, padx=2)
        
        self.pos_zoom_label = ttk.Label(pos_controls, text="100%")
        self.pos_zoom_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(pos_controls, text="Save Image",
                  command=lambda: self.save_processing_image('positive')).pack(side=tk.RIGHT, padx=2)
        
        # Positive description
        pos_desc_frame = ttk.LabelFrame(positive_frame, text="Description", padding=5)
        pos_desc_frame.pack(fill=tk.X, pady=(5, 5), padx=5)
        
        self.pos_description = tk.Text(pos_desc_frame, height=3, wrap='word', 
                                       font=('Arial', 9), state='disabled')
        self.pos_description.pack(fill=tk.X)
        
        # === NEGATIVE COLUMN ===
        negative_frame = ttk.Frame(dual_split)
        dual_split.add(negative_frame, weight=1)
        
        # Negative header
        neg_header = ttk.Frame(negative_frame)
        neg_header.pack(fill=tk.X, pady=(5, 5), padx=5)
        
        ttk.Label(neg_header, text="NEGATIVE (G-)", font=('Arial', 11, 'bold'),
                 foreground='#2E7D32').pack(side=tk.LEFT)
        
        self.neg_title = ttk.Label(neg_header, text="", font=('Arial', 9))
        self.neg_title.pack(side=tk.LEFT, padx=(10, 0))
        
        # Negative canvas
        neg_canvas_frame = ttk.Frame(negative_frame, relief='sunken', borderwidth=2)
        neg_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5)
        
        self.neg_canvas = tk.Canvas(neg_canvas_frame, bg='#f0f0f0', cursor='cross')
        self.neg_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Negative controls
        neg_controls = ttk.Frame(negative_frame)
        neg_controls.pack(fill=tk.X, pady=(5, 0), padx=5)
        
        ttk.Button(neg_controls, text="Fit to Window",
                  command=lambda: self.fit_to_window('negative')).pack(side=tk.LEFT, padx=2)
        
        self.neg_zoom_label = ttk.Label(neg_controls, text="100%")
        self.neg_zoom_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(neg_controls, text="Save Image",
                  command=lambda: self.save_processing_image('negative')).pack(side=tk.RIGHT, padx=2)
        
        # Negative description
        neg_desc_frame = ttk.LabelFrame(negative_frame, text="Description", padding=5)
        neg_desc_frame.pack(fill=tk.X, pady=(5, 5), padx=5)
        
        self.neg_description = tk.Text(neg_desc_frame, height=3, wrap='word',
                                       font=('Arial', 9), state='disabled')
        self.neg_description.pack(fill=tk.X)
    
    def populate_processing_steps(self):
        """Populate the processing steps tree"""
        # Clear existing items
        for item in self.steps_tree.get_children():
            self.steps_tree.delete(item)
        
        # Define processing pipeline
        steps = {
            "Phase 1: Brightfield Preprocessing": [
                ("01_gray_8bit.png", "Original grayscale image (8-bit)"),
                ("02_enhanced.png", "CLAHE enhanced image for improved contrast"),
                ("03_enhanced_blur.png", "Gaussian blur applied to reduce noise"),
            ],
            "Phase 2: Brightfield Segmentation": [
                ("04_thresh_raw.png", "Otsu thresholding for binary segmentation"),
                ("05_closed.png", "Morphological closing to fill gaps"),
            ],
            "Phase 3: Particle Detection & Filtering": [
                ("10_contours_all.png", "All detected contours from segmentation"),
                ("11_contours_rejected_orange_accepted_yellow_ids_green.png", "Filtered contours (rejected=orange, accepted=yellow, IDs=green)"),
                ("12_mask_all.png", "Binary mask of all detected particles"),
                ("13_mask_accepted.png", "Binary mask of accepted particles only"),
                ("13_mask_accepted_ids.png", "Accepted particles with ID labels"),
            ],
            "Phase 4: Fluorescence Channel Processing": [
                ("20_fluorescence_aligned_raw.png", "Raw fluorescence channel aligned to brightfield"),
                ("20_fluorescence_8bit.png", "8-bit converted fluorescence image"),
                ("21_fluorescence_overlay.png", "Fluorescence overlaid on brightfield"),
                ("22_fluorescence_mask_global.png", "Global thresholded fluorescence mask"),
                ("22_fluorescence_mask_global_ids.png", "Fluorescence mask with particle IDs"),
                ("23_fluorescence_contours_global.png", "Fluorescence contours detected"),
                ("23_fluorescence_contours_global_ids.png", "Fluorescence contours with IDs"),
            ],
            "Phase 5: BF-Fluorescence Matching": [
                ("24_bf_fluor_matching_overlay.png", "Brightfield and fluorescence matching visualization"),
                ("24_bf_fluor_matching_overlay_ids.png", "BF-Fluorescence matching with particle IDs"),
            ],
        }
        
        for phase, phase_steps in steps.items():
            phase_item = self.steps_tree.insert("", "end", text=phase, tags=('phase',))
            for filename, description in phase_steps:
                self.steps_tree.insert(phase_item, "end", text=f"○ {filename}", 
                                      values=(description,), tags=('step',))
        
        # Expand all
        for item in self.steps_tree.get_children():
            self.steps_tree.item(item, open=True)
        
        # Configure tags
        self.steps_tree.tag_configure('phase', font=('Arial', 10, 'bold'))
    
    def on_step_selected(self, event):
        """Display selected processing step for both Positive and Negative"""
        selection = self.steps_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        item_text = self.steps_tree.item(item, "text")
        
        # Extract filename
        filename = item_text.replace("○ ", "").replace("✓ ", "").replace("◐ ", "").replace("✗ ", "")
        
        # Skip phase headers
        if not filename.endswith(".png"):
            return
        
        # Get description
        values = self.steps_tree.item(item, "values")
        description = values[0] if values else ""
        
        if self.selected_group is None or self.current_folder is None:
            self.update_processing_description("Please select a group first", 'both')
            return
        
        # Load Positive image
        self.load_processing_image('positive', filename, description)
        
        # Load Negative image
        self.load_processing_image('negative', filename, description)
    
    
    def load_processing_image(self, img_type, filename, description):
        """Load processing image for specific type (positive or negative)"""
        if self.current_folder is None or self.selected_group is None:
            self.update_processing_description("Please select a group first", img_type)
            self.clear_processing_canvas(img_type)
            return
        
        type_name = "Positive" if img_type == 'positive' else "Negative"
        
        # === CORRECTED: Handle "Control group" folder name ===
        # Control folder is named "Control group" (with space)
        folder_name = "Control group" if self.selected_group.lower() == 'control' else self.selected_group
        type_folder = self.current_folder / type_name / folder_name
        # === END CORRECTION ===
        
        if not type_folder.exists():
            self.update_processing_description(f"Folder not found:\n{type_folder}", img_type)
            self.clear_processing_canvas(img_type)
            return
        
        # Find first image folder
        image_folders = [d for d in type_folder.iterdir() if d.is_dir()]
        
        if not image_folders:
            self.update_processing_description(f"No image folders found", img_type)
            self.clear_processing_canvas(img_type)
            return
        
        first_image_folder = sorted(image_folders)[0]
        image_path = first_image_folder / filename
        
        # Update title
        display_name = "Control" if self.selected_group.lower() == 'control' else self.selected_group
        if img_type == 'positive':
            self.pos_title.config(text=f"{display_name} / {first_image_folder.name}")
        else:
            self.neg_title.config(text=f"{display_name} / {first_image_folder.name}")
        
        if not image_path.exists():
            self.update_processing_description(f"Image not found:\n{filename}", img_type)
            self.clear_processing_canvas(img_type)
            return
        
        # Display image
        try:
            self.display_processing_step_image(image_path, img_type)
            self.update_processing_description(description, img_type)
            
            # Store current path
            if img_type == 'positive':
                self.current_pos_path = image_path
            else:
                self.current_neg_path = image_path
                
        except Exception as e:
            self.update_processing_description(f"Error loading image:\n{str(e)}", img_type)
            self.clear_processing_canvas(img_type)


    def display_processing_step_image(self, image_path, img_type):
        """Display a processing step image on the appropriate canvas"""
        canvas = self.pos_canvas if img_type == 'positive' else self.neg_canvas
        
        # Load image
        image = Image.open(image_path)
        
        # Store original
        if img_type == 'positive':
            self.current_pos_image = image
        else:
            self.current_neg_image = image
        
        # Fit to canvas
        self.fit_image_to_canvas(image, canvas, img_type)
    
    def fit_image_to_canvas(self, image, canvas, img_type):
        """Fit image to canvas size"""
        # Get canvas dimensions
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 400
            canvas_height = 400
        
        # Calculate scaling
        img_width, img_height = image.size
        scale_w = canvas_width / img_width
        scale_h = canvas_height / img_height
        scale = min(scale_w, scale_h) * 0.95  # 95% to leave some margin
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(resized)
        
        # Store reference
        if img_type == 'positive':
            self.photo_positive = photo
        else:
            self.photo_negative = photo
        
        # Display on canvas
        canvas.delete("all")
        x = canvas_width // 2
        y = canvas_height // 2
        canvas.create_image(x, y, image=photo, anchor='center')
        
        # Update zoom label
        zoom_pct = int(scale * 100)
        if img_type == 'positive':
            self.pos_zoom_label.config(text=f"{zoom_pct}%")
        else:
            self.neg_zoom_label.config(text=f"{zoom_pct}%")
    
    def fit_to_window(self, img_type):
        """Fit image to window"""
        image = self.current_pos_image if img_type == 'positive' else self.current_neg_image
        canvas = self.pos_canvas if img_type == 'positive' else self.neg_canvas
        
        if image:
            self.fit_image_to_canvas(image, canvas, img_type)
    
    def clear_processing_canvas(self, img_type):
        """Clear the specified processing canvas"""
        canvas = self.pos_canvas if img_type == 'positive' else self.neg_canvas
        canvas.delete("all")
        
        # Clear stored image and path
        if img_type == 'positive':
            self.current_pos_image = None
            self.photo_positive = None
            self.current_pos_path = None
        else:
            self.current_neg_image = None
            self.photo_negative = None
            self.current_neg_path = None
    
    def update_processing_description(self, text, img_type):
        """Update processing step description text"""
        if img_type == 'both':
            text_widgets = [self.pos_description, self.neg_description]
        elif img_type == 'positive':
            text_widgets = [self.pos_description]
        else:
            text_widgets = [self.neg_description]
        
        for widget in text_widgets:
            widget.config(state='normal')
            widget.delete('1.0', tk.END)
            widget.insert('1.0', text)
            widget.config(state='disabled')
    
    def save_processing_image(self, img_type):
        """Save the current processing step image"""
        image_path = self.current_pos_path if img_type == 'positive' else self.current_neg_path
        
        if not image_path:
            messagebox.showwarning("No Image", "No image to save")
            return
        
        if self.selected_group is None:
            messagebox.showwarning("No Group", "No group selected")
            return
        
        # Suggest filename
        type_label = "Positive" if img_type == 'positive' else "Negative"
        default_name = f"{self.selected_group}_{type_label}_{image_path.name}"
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                image = self.current_pos_image if img_type == 'positive' else self.current_neg_image
                if image:
                    image.save(filename)
                    messagebox.showinfo("Success", f"Image saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")
    
    def create_status_bar(self):
        """Create status bar at bottom"""
        status_frame = ttk.Frame(self.root, relief=tk.SUNKEN, borderwidth=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                style='Status.TLabel')
        status_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Info label on right
        self.info_var = tk.StringVar(value="")
        info_label = ttk.Label(status_frame, textvariable=self.info_var,
                              style='Status.TLabel')
        info_label.pack(side=tk.RIGHT, padx=5, pady=2)
    
    # ==================== Data Loading ====================
    
    def load_results(self):
        """Open file dialog to load results folder"""
        initial_dir = Path.cwd() / "outputs"
        if not initial_dir.exists():
            initial_dir = Path.cwd()
        
        folder = filedialog.askdirectory(
            title="Select Output Folder",
            initialdir=str(initial_dir)
        )
        
        if folder:
            try:
                folder_path = Path(folder)
                # Test if path is accessible
                folder_path.exists()
                self.load_results_from_folder(folder_path)
            except (OSError, UnicodeDecodeError) as e:
                messagebox.showerror("Path Error", 
                    f"Cannot access folder with non-ASCII characters:\n{folder}\n\nError: {str(e)}")
    
    def auto_load_recent(self):
        """Try to load the most recent output folder"""
        outputs_dir = Path.cwd() / "outputs"
        
        if not outputs_dir.exists():
            self.status_var.set("No outputs folder found")
            return
        
        # Find most recent folder
        folders = [f for f in outputs_dir.iterdir() if f.is_dir()]
        if not folders:
            self.status_var.set("No output folders found")
            return
        
        # Sort by modification time
        most_recent = max(folders, key=lambda f: f.stat().st_mtime)
        self.load_results_from_folder(most_recent)
    
    def load_results_from_folder(self, folder_path: Path):
        """Load and parse results from folder"""
        try:
            self.status_var.set(f"Loading from {folder_path.name}...")
            self.current_folder = folder_path
            
            # Clear existing data
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Clear photo references
            self.photo_refs.clear()
            
            # Try to load final clinical results
            clinical_matrix = folder_path / "final_clinical_results.csv"
            
            if clinical_matrix.exists():
                self.load_batch_results(folder_path, clinical_matrix)
            else:
                # Try single mode
                self.load_single_results(folder_path)
            
            self.dataset_label.config(text=f"Dataset: {self.dataset_name}")
            self.status_var.set(f"Loaded: {folder_path.name}")
            self.info_var.set(f"{len(self.tree.get_children())} groups loaded")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load results:\n{str(e)}")
            self.status_var.set("Error loading results")
    
    def load_batch_results(self, folder_path: Path, clinical_matrix: Path):
        """Load batch mode results (G+ and G-)"""
        try:
            self.clinical_data = pd.read_csv(clinical_matrix, encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to GB18030 for Simplified Chinese
            self.clinical_data = pd.read_csv(clinical_matrix, encoding='gb18030')
        
        self.dataset_name = folder_path.name
        
        # Load G+ and G- classifications
        gplus_csv = folder_path / "Positive" / "clinical_classification_positive.csv"
        gminus_csv = folder_path / "Negative" / "clinical_classification_negative.csv"
        
        if gplus_csv.exists():
            try:
                self.gplus_data = pd.read_csv(gplus_csv, encoding='utf-8')
            except UnicodeDecodeError:
                self.gplus_data = pd.read_csv(gplus_csv, encoding='gb18030')
        
        if gminus_csv.exists():
            try:
                self.gminus_data = pd.read_csv(gminus_csv, encoding='utf-8')
            except UnicodeDecodeError:
                self.gminus_data = pd.read_csv(gminus_csv, encoding='gb18030')
        
        # Populate tree
        for _, row in self.clinical_data.iterrows():
            group = str(row['Group'])
            status = row['Final_Classification']
            
            # Get G+ and G- values
            gplus_val = row['G+_Detection'] if 'G+_Detection' in row else '-'
            gminus_val = row['G-_Detection'] if 'G-_Detection' in row else '-'
            
            # Determine color tag
            tag = self.get_status_tag(status)
            
            # Insert into tree
            self.tree.insert("", tk.END, 
                           text=f"Group {group}", 
                           values=("Combined", gplus_val, gminus_val, status),
                           tags=(tag,))
    
    def load_single_results(self, folder_path: Path):
        """Load single mode results (G+ or G- only)"""
        # Try to find classification file
        classification_files = list(folder_path.glob("clinical_classification_*.csv"))
        
        if not classification_files:
            raise FileNotFoundError("No clinical classification file found")
        
        classification_csv = classification_files[0]
        try:
            self.clinical_data = pd.read_csv(classification_csv, encoding='utf-8')
        except UnicodeDecodeError:
            self.clinical_data = pd.read_csv(classification_csv, encoding='gb18030')
        
        self.dataset_name = folder_path.name
        
        # Determine type from filename
        if "positive" in classification_csv.name.lower():
            self.gplus_data = self.clinical_data
            microgel_type = "G+"
        else:
            self.gminus_data = self.clinical_data
            microgel_type = "G-"
        
        # === ADD CONTROL GROUP BY SCANNING FOLDERS ===
        # Check if Control group folder exists
        control_folder = folder_path / "Control group"
        if not control_folder.exists():
            # Try alternative names
            control_folder = folder_path / "Control"
        
        if control_folder.exists() and control_folder.is_dir():
            # Try to get Control group statistics from group_statistics_summary.csv
            stats_summary_csv = folder_path / "group_statistics_summary.csv"
            control_mean = None
            control_n = None
            control_std = None
            
            if stats_summary_csv.exists():
                try:
                    stats_data = pd.read_csv(stats_summary_csv, encoding='utf-8')
                except UnicodeDecodeError:
                    stats_data = pd.read_csv(stats_summary_csv, encoding='gb18030')
                
                # Find Control row (case-insensitive)
                control_rows = stats_data[stats_data['Group'].astype(str).str.lower() == 'control']
                if not control_rows.empty:
                    control_row = control_rows.iloc[0]
                    control_mean = control_row.get('Mean', None)
                    control_n = control_row.get('N', None)
                    control_std = control_row.get('Std_Dev', None)
            
            # If we couldn't find statistics, use placeholder values
            if control_mean is None:
                # Get control_mean from clinical_data if available
                if 'Control_Mean' in self.clinical_data.columns and not self.clinical_data.empty:
                    control_mean = self.clinical_data.iloc[0]['Control_Mean']
                else:
                    control_mean = 0
            
            # Check if Control is already in clinical_data
            if 'Control' not in self.clinical_data['Group'].astype(str).values and \
               'control' not in self.clinical_data['Group'].astype(str).str.lower().values:
                # Add Control group entry
                control_entry = {
                    'Group': 'Control',
                    'N': control_n if control_n is not None else 0,
                    'Mean': control_mean,
                    'Std_Dev': control_std if control_std is not None else 0,
                    'Control_Mean': control_mean,
                    'Threshold': control_mean,
                    'Diff_from_Threshold': 0,
                    'Diff_from_Control': 0,
                    'Pct_Diff_from_Control': 0,
                    'Classification': 'CONTROL (Reference)'
                }
                # Append to clinical_data
                self.clinical_data = pd.concat([self.clinical_data, pd.DataFrame([control_entry])], 
                                            ignore_index=True)
        # === END OF CONTROL GROUP ADDITION ===
        
        # Populate tree
        for _, row in self.clinical_data.iterrows():
            group = str(row['Group'])
            status = row['Classification']
            
            tag = self.get_status_tag(status)
            
            self.tree.insert("", tk.END,
                        text=f"Group {group}",
                        values=(microgel_type, "-", "-", status),
                        tags=(tag,))
    
    def get_status_tag(self, status: str) -> str:
        """Determine color tag from status"""
        status_upper = status.upper()
        
        # CHECK CONTROL FIRST
        if "CONTROL" in status_upper:
            return "control"
        
        if "POSITIVE" in status_upper and "NO OBVIOUS" not in status_upper:
            return "positive"
        elif "NEGATIVE" in status_upper and "NO OBVIOUS" not in status_upper:
            return "negative"
        elif "NO OBVIOUS BACTERIA" in status_upper:
            return "no_bacteria"
        elif "MIXED" in status_upper or "CONTRADICTORY" in status_upper:
            return "mixed"
        else:
            return "missing"
    
    # ==================== UI Actions ====================
    
    def on_select(self, event):
        """Handle tree selection"""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = selection[0]
        group_text = self.tree.item(item, "text")
        
        # Extract group number
        self.selected_group = group_text.replace("Group ", "")
        
        # Update all tabs
        self.display_overview()
        self.display_gplus_details()
        self.display_gminus_details()
        self.display_plots()
        self.display_raw_data()
        
        # Update processing steps availability
        self.update_step_availability()
        
        self.info_var.set(f"Selected: {group_text}")
    
    def update_step_availability(self):
        """Update step tree to show which images are available"""
        if self.selected_group is None or self.current_folder is None:
            return
        
        # === CORRECTED: Handle "Control group" folder name ===
        # Control folder is named "Control group" (with space)
        folder_name = "Control group" if self.selected_group.lower() == 'control' else self.selected_group
        pos_folder = self.current_folder / "Positive" / folder_name
        neg_folder = self.current_folder / "Negative" / folder_name
        # === END CORRECTION ===
        
        # Get first image folders
        pos_images = None
        neg_images = None
        
        if pos_folder.exists():
            pos_image_folders = [d for d in pos_folder.iterdir() if d.is_dir()]
            if pos_image_folders:
                pos_images = sorted(pos_image_folders)[0]
        
        if neg_folder.exists():
            neg_image_folders = [d for d in neg_folder.iterdir() if d.is_dir()]
            if neg_image_folders:
                neg_images = sorted(neg_image_folders)[0]
        
        # Update tree items
        for phase_item in self.steps_tree.get_children():
            for step_item in self.steps_tree.get_children(phase_item):
                current_text = self.steps_tree.item(step_item, "text")
                filename = current_text.replace("○ ", "").replace("✓ ", "").replace("◐ ", "").replace("✗ ", "")
                
                # Check if file exists in either location
                pos_exists = pos_images is not None and (pos_images / filename).exists()
                neg_exists = neg_images is not None and (neg_images / filename).exists()
                
                if pos_exists and neg_exists:
                    self.steps_tree.item(step_item, text=f"✓ {filename}", 
                                    tags=('step', 'both'))
                elif pos_exists or neg_exists:
                    self.steps_tree.item(step_item, text=f"◐ {filename}",
                                    tags=('step', 'partial'))
                else:
                    self.steps_tree.item(step_item, text=f"✗ {filename}",
                                    tags=('step', 'missing'))
        
        # Configure tags
        self.steps_tree.tag_configure('both', foreground='green')
        self.steps_tree.tag_configure('partial', foreground='orange')
        self.steps_tree.tag_configure('missing', foreground='gray')



    def display_overview(self):
        """Display overview for selected group"""
        # Clear existing content
        for widget in self.overview_content.winfo_children():
            widget.destroy()
        
        if self.clinical_data is None or self.selected_group is None:
            ttk.Label(self.overview_content, 
                     text="No data available",
                     font=('Arial', 11)).pack(pady=20)
            return
        
        # Get group data
        group_data = self.clinical_data[
            self.clinical_data['Group'].astype(str) == self.selected_group
        ]
        
        if group_data.empty:
            ttk.Label(self.overview_content,
                     text=f"No data found for Group {self.selected_group}",
                     font=('Arial', 11)).pack(pady=20)
            return
        
        row = group_data.iloc[0]
        
        # Title
        title_frame = ttk.Frame(self.overview_content)
        title_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(title_frame, 
                 text=f"Group {self.selected_group} - Clinical Summary",
                 style='Title.TLabel').pack(anchor=tk.W)
        
        # Result card
        result_color = self.colors[self.get_status_tag(row['Final_Classification'] if 'Final_Classification' in row else row['Classification'])]
        result_frame = tk.Frame(self.overview_content, 
                               bg=result_color,
                               relief=tk.RAISED,
                               borderwidth=2)
        result_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(result_frame, 
                text="FINAL CLASSIFICATION",
                bg=result_color,
                font=('Arial', 10, 'bold')).pack(pady=(10,5))
        
        classification_text = row['Final_Classification'] if 'Final_Classification' in row else row['Classification']
        tk.Label(result_frame,
                text=classification_text,
                bg=result_color,
                font=('Arial', 16, 'bold')).pack(pady=(5,10))
        
        # Details table
        details_frame = ttk.LabelFrame(self.overview_content, 
                                      text="Detection Details",
                                      padding=10)
        details_frame.pack(fill=tk.X, padx=20, pady=10)
        
        details = [
            ("G+ Microgel Mean:", f"{row.get('G+_Mean', 'N/A')}"),
            ("G+ Detection:", row.get('G+_Detection', 'N/A')),
            ("", ""),
            ("G- Microgel Mean:", f"{row.get('G-_Mean', 'N/A')}"),
            ("G- Detection:", row.get('G-_Detection', 'N/A')),
        ]
        
        for i, (label, value) in enumerate(details):
            if label:
                tk.Label(details_frame, text=label, 
                        font=('Arial', 10, 'bold'),
                        anchor=tk.W).grid(row=i, column=0, sticky=tk.W, pady=2, padx=5)
                tk.Label(details_frame, text=str(value),
                        font=('Arial', 10),
                        anchor=tk.W).grid(row=i, column=1, sticky=tk.W, pady=2, padx=5)
        
        # Interpretation guide
        guide_frame = ttk.LabelFrame(self.overview_content,
                                    text="Interpretation Guide",
                                    padding=10)
        guide_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        interpretation = self.get_interpretation(classification_text)
        
        text_widget = tk.Text(guide_frame, wrap=tk.WORD, height=8,
                             font=('Arial', 10))
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert('1.0', interpretation)
        text_widget.config(state=tk.DISABLED)
    
    def get_interpretation(self, classification: str) -> str:
        """Get clinical interpretation text"""
        interpretations = {
            'POSITIVE': """
POSITIVE RESULT - Bacteria Detected

This sample shows fluorescence patterns consistent with bacterial presence. 
Both G+ and G- microgels indicate detection.

Clinical Action:
• Consider this result as indicative of bacterial infection
• May warrant antibiotic therapy
• Correlate with clinical symptoms and other laboratory findings
            """,
            'NEGATIVE': """
NEGATIVE RESULT - No Bacteria Detected

This sample shows minimal fluorescence, indicating absence of bacteria.
Both G+ and G- microgels show no detection.

Clinical Action:
• This result suggests no bacterial infection
• Consider alternative diagnoses
• If clinical suspicion remains high, consider repeat testing
            """,
            'NO OBVIOUS BACTERIA': """
NO OBVIOUS BACTERIA

The fluorescence levels are within normal control range for both microgel types.

Clinical Action:
• No clear bacterial signal detected
• May indicate very low bacterial load or absence of infection
• Clinical correlation recommended
            """,
            'MIXED/CONTRADICTORY': """
MIXED/CONTRADICTORY RESULT

The G+ and G- microgels show conflicting results.

Clinical Action:
• This may indicate:
  - Mixed infection with both Gram-positive and Gram-negative bacteria
  - Technical variability
  - Borderline bacterial load
• Recommend repeat testing
• Consider additional confirmatory tests
            """,
            'CONTROL (Reference)': """
CONTROL GROUP - Reference Standard

This is the control group used as baseline for comparison.

Purpose:
• Establishes normal fluorescence levels
• Used to calculate detection thresholds
• Reference for interpreting test samples
• Quality control for the assay
            """
        }
        
        # Match partial strings
        for key, value in interpretations.items():
            if key.upper() in classification.upper():
                return value
        
        return "No interpretation available for this result."
    
    def display_gplus_details(self):
        """Display G+ microgel details"""
        for widget in self.gplus_tab.winfo_children():
            widget.destroy()
        
        if self.gplus_data is None:
            ttk.Label(self.gplus_tab,
                     text="G+ data not available for this dataset",
                     font=('Arial', 11)).pack(pady=20)
            return
        
        # Get group data
        group_data = self.gplus_data[
            self.gplus_data['Group'].astype(str) == self.selected_group
        ]
        
        if group_data.empty:
            ttk.Label(self.gplus_tab,
                     text=f"No G+ data for Group {self.selected_group}",
                     font=('Arial', 11)).pack(pady=20)
            return
        
        self.create_microgel_detail_view(self.gplus_tab, group_data.iloc[0], "G+ (Positive)")
    
    def display_gminus_details(self):
        """Display G- microgel details"""
        for widget in self.gminus_tab.winfo_children():
            widget.destroy()
        
        if self.gminus_data is None:
            ttk.Label(self.gminus_tab,
                     text="G- data not available for this dataset",
                     font=('Arial', 11)).pack(pady=20)
            return
        
        # Get group data
        group_data = self.gminus_data[
            self.gminus_data['Group'].astype(str) == self.selected_group
        ]
        
        if group_data.empty:
            ttk.Label(self.gminus_tab,
                     text=f"No G- data for Group {self.selected_group}",
                     font=('Arial', 11)).pack(pady=20)
            return
        
        self.create_microgel_detail_view(self.gminus_tab, group_data.iloc[0], "G- (Negative)")
    
    def create_microgel_detail_view(self, parent, data, title):
        """Create detailed view for a microgel type"""
        # Scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Title
        ttk.Label(scrollable_frame,
                 text=f"{title} Microgel Analysis",
                 style='Title.TLabel').pack(pady=10, padx=20, anchor=tk.W)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(scrollable_frame, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X, padx=20, pady=10)
        
        stats = [
            ("Sample Size (N):", data.get('N', 'N/A')),
            ("Mean Fluorescence:", f"{data.get('Mean', 'N/A'):.2f}" if isinstance(data.get('Mean'), (int, float)) else 'N/A'),
            ("Std Deviation:", f"{data.get('Std_Dev', 'N/A'):.2f}" if isinstance(data.get('Std_Dev'), (int, float)) else 'N/A'),
            ("Control Mean:", f"{data.get('Control_Mean', 'N/A'):.2f}" if isinstance(data.get('Control_Mean'), (int, float)) else 'N/A'),
            ("Threshold:", f"{data.get('Threshold', 'N/A'):.2f}" if isinstance(data.get('Threshold'), (int, float)) else 'N/A'),
            ("Diff from Threshold:", f"{data.get('Diff_from_Threshold', 'N/A'):.2f}" if isinstance(data.get('Diff_from_Threshold'), (int, float)) else 'N/A'),
            ("Diff from Control:", f"{data.get('Diff_from_Control', 'N/A'):.2f}" if isinstance(data.get('Diff_from_Control'), (int, float)) else 'N/A'),
            ("% Diff from Control:", f"{data.get('Pct_Diff_from_Control', 'N/A'):.1f}%" if isinstance(data.get('Pct_Diff_from_Control'), (int, float)) else 'N/A'),
        ]
        
        for i, (label, value) in enumerate(stats):
            tk.Label(stats_frame, text=label,
                    font=('Arial', 10, 'bold'),
                    anchor=tk.W).grid(row=i, column=0, sticky=tk.W, pady=2, padx=5)
            tk.Label(stats_frame, text=str(value),
                    font=('Arial', 10),
                    anchor=tk.W).grid(row=i, column=1, sticky=tk.W, pady=2, padx=5)
        
        # Classification
        class_frame = ttk.LabelFrame(scrollable_frame, text="Classification", padding=10)
        class_frame.pack(fill=tk.X, padx=20, pady=10)
        
        classification = data.get('Classification', 'N/A')
        tag = self.get_status_tag(classification)
        color = self.colors.get(tag, '#ffffff')
        
        result_label = tk.Label(class_frame,
                               text=classification,
                               font=('Arial', 14, 'bold'),
                               bg=color,
                               relief=tk.RAISED,
                               borderwidth=2,
                               padx=20, pady=10)
        result_label.pack(fill=tk.X)
    
    def display_plots(self):
        """Display comparison plots"""
        for widget in self.plots_tab.winfo_children():
            widget.destroy()
        
        # Clear old photo references for plots tab
        self.photo_refs = {k: v for k, v in self.photo_refs.items() 
                          if k not in [id(w) for w in self.plots_tab.winfo_children()]}
        
        if self.current_folder is None:
            ttk.Label(self.plots_tab,
                     text="No plots available",
                     font=('Arial', 11)).pack(pady=20)
            return
        
        # Create notebook for different plot types
        plot_notebook = ttk.Notebook(self.plots_tab)
        plot_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Find plot files
        plot_files = list(self.current_folder.rglob("*.png"))
        
        # Filter for comparison plots
        comparison_plots = [p for p in plot_files if "comparison" in p.name.lower()]
        
        if comparison_plots:
            for plot_path in comparison_plots[:3]:  # Limit to first 3 plots
                self.add_plot_tab(plot_notebook, plot_path)
        
        # Add group-specific plots if available
        if self.selected_group:
            group_plots = [p for p in plot_files 
                          if f"Group_{self.selected_group}" in p.name or
                             f"group_{self.selected_group}" in p.name.lower()]
            
            for plot_path in group_plots[:3]:
                self.add_plot_tab(plot_notebook, plot_path)
        
        if not comparison_plots and not (self.selected_group and group_plots):
            ttk.Label(self.plots_tab,
                     text="No plots found in output folder",
                     font=('Arial', 11)).pack(pady=20)
    
    def add_plot_tab(self, notebook, plot_path: Path):
        """Add a plot as a tab"""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=plot_path.stem[:20])
        
        try:
            # Ensure path is properly encoded
            img = Image.open(str(plot_path))
            
            # Calculate size to fit in window
            max_width = 1000
            max_height = 700
            
            img_width, img_height = img.size
            scale = min(max_width/img_width, max_height/img_height, 1.0)
            
            new_size = (int(img_width * scale), int(img_height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            
            # Create label with image
            label = tk.Label(tab, image=photo)
            
            # Store photo reference using label's id as key
            self.photo_refs[id(label)] = photo
            
            label.pack(expand=True)
            
        except (OSError, UnicodeDecodeError) as e:
            ttk.Label(tab, text=f"Error loading plot (path encoding issue):\n{str(e)}").pack(pady=20)
        except Exception as e:
            ttk.Label(tab, text=f"Error loading plot:\n{str(e)}").pack(pady=20)
    
    def display_raw_data(self):
        """Display raw data table"""
        for widget in self.data_tab.winfo_children():
            widget.destroy()
        
        if self.clinical_data is None:
            ttk.Label(self.data_tab,
                     text="No data available",
                     font=('Arial', 11)).pack(pady=20)
            return
        
        # Create frame for table
        table_frame = ttk.Frame(self.data_tab)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview
        columns = list(self.clinical_data.columns)
        
        data_tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        # Configure columns
        for col in columns:
            data_tree.heading(col, text=col)
            data_tree.column(col, width=100, minwidth=80)
        
        # Add scrollbars
        vsb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=data_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=data_tree.xview)
        data_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout
        data_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Insert data
        for _, row in self.clinical_data.iterrows():
            values = [str(row[col]) for col in columns]
            data_tree.insert('', tk.END, values=values)
    
    def apply_filter(self):
        """Apply filter to tree view"""
        filter_value = self.filter_var.get()
        
        # Clear tree
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if self.clinical_data is None:
            return
        
        # Re-populate with filter
        for _, row in self.clinical_data.iterrows():
            status = row['Final_Classification'] if 'Final_Classification' in row else row['Classification']
            
            # Apply filter
            if filter_value != "All" and filter_value not in status:
                continue
            
            group = str(row['Group'])
            gplus_val = row.get('G+_Detection', '-')
            gminus_val = row.get('G-_Detection', '-')
            
            tag = self.get_status_tag(status)
            
            if 'Final_Classification' in row:
                # Batch mode
                self.tree.insert("", tk.END,
                               text=f"Group {group}",
                               values=("Combined", gplus_val, gminus_val, status),
                               tags=(tag,))
            else:
                # Single mode
                microgel_type = "G+" if self.gplus_data is not None else "G-"
                self.tree.insert("", tk.END,
                               text=f"Group {group}",
                               values=(microgel_type, "-", "-", status),
                               tags=(tag,))
        
        count = len(self.tree.get_children())
        self.info_var.set(f"{count} groups (filtered)")
    
    def refresh_view(self):
        """Refresh current view"""
        if self.current_folder:
            self.load_results_from_folder(self.current_folder)
        else:
            self.auto_load_recent()
    
    def expand_all(self):
        """Expand all tree items"""
        for item in self.tree.get_children():
            self.tree.item(item, open=True)
    
    def collapse_all(self):
        """Collapse all tree items"""
        for item in self.tree.get_children():
            self.tree.item(item, open=False)
    
    # ==================== Export Functions ====================
    
    def export_csv(self):
        """Export current data to CSV"""
        if self.clinical_data is None:
            messagebox.showwarning("No Data", "No data loaded to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{self.dataset_name}_export.csv"
        )
        
        if filename:
            try:
                # UTF-8 with BOM for Excel compatibility with Chinese characters
                self.clinical_data.to_csv(filename, index=False, encoding='utf-8-sig')
                messagebox.showinfo("Success", f"Data exported to:\n{filename}")
                self.status_var.set(f"Exported to {Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def export_pdf(self):
        """Export report to PDF"""
        if not REPORTLAB_AVAILABLE:
            messagebox.showwarning("PDF Export Unavailable",
                                 "ReportLab library not installed.\n"
                                 "Install with: pip install reportlab")
            return
        
        if self.clinical_data is None:
            messagebox.showwarning("No Data", "No data loaded to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            initialfile=f"{self.dataset_name}_report.pdf"
        )
        
        if filename:
            try:
                self.generate_pdf_report(filename)
                messagebox.showinfo("Success", f"Report exported to:\n{filename}")
                self.status_var.set(f"PDF exported to {Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Error", f"PDF export failed:\n{str(e)}")
    
    def generate_pdf_report(self, filename: str):
        """Generate PDF report using ReportLab"""
        if self.clinical_data is None:
            return
            
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        
        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph(f"<b>Clinical Results Report</b><br/>{self.dataset_name}", 
                         styles['Title'])
        story.append(title)
        story.append(Spacer(1, 0.3*inch))
        
        # Summary
        summary_text = f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}<br/>"
        summary_text += f"Total Groups: {len(self.clinical_data)}<br/>"
        
        summary = Paragraph(summary_text, styles['Normal'])
        story.append(summary)
        story.append(Spacer(1, 0.2*inch))
        
        # Results table
        table_data = [list(self.clinical_data.columns)]
        
        for _, row in self.clinical_data.iterrows():
            table_data.append([str(row[col]) for col in self.clinical_data.columns])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        
        doc.build(story)
    
    def open_output_folder(self):
        """Open the current output folder in file explorer"""
        if self.current_folder is None:
            messagebox.showwarning("No Folder", "No output folder loaded")
            return
        
        try:
            if sys.platform == 'win32':
                import os
                # Use os.startfile for better Unicode support on Windows
                os.startfile(str(self.current_folder))
            elif sys.platform == 'darwin':
                import subprocess
                subprocess.Popen(['open', str(self.current_folder)])
            else:  # linux
                import subprocess
                subprocess.Popen(['xdg-open', str(self.current_folder)])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder:\n{str(e)}")
    
    # ==================== Help Functions ====================
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
Particle-Scout Clinical Results Viewer
Version 2.0 - Now with Control Group Support

A cross-platform application for reviewing
bacterial detection results using microgel
fluorescence analysis.

Features:
• Batch processing (G+ and G-)
• Clinical classification
• Control group visualization
• Processing steps visualization
• Data export (CSV/PDF)

© 2026 - Clinical Diagnostics
        """
        
        messagebox.showinfo("About", about_text.strip())
    
    def show_help(self):
        """Show help dialog"""
        help_window = tk.Toplevel(self.root)
        help_window.title("User Guide")
        help_window.geometry("700x700")
        
        # Scrolled text for help
        help_text = scrolledtext.ScrolledText(help_window, wrap=tk.WORD,
                                              font=('Arial', 10))
        help_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        help_content = """
PARTICLE-SCOUT CLINICAL RESULTS VIEWER - USER GUIDE

1. LOADING RESULTS
   • Click "Load Results" or use Ctrl+O
   • Navigate to your output folder
   • The application will automatically detect batch or single mode
   • Use "Open Recent Folder" (Ctrl+R) for quick access

2. VIEWING RESULTS
   • Groups are displayed in the left panel
   • Color coding indicates classification:
     - Red: POSITIVE (bacteria detected)
     - Green: NEGATIVE (no bacteria)
     - Yellow: NO OBVIOUS BACTERIA
     - Orange: MIXED/CONTRADICTORY
     - Light Blue: CONTROL (reference group)
   
3. CONTROL GROUP (NEW!)
   • Control group appears in light blue
   • Used as reference standard for calculations
   • View processing steps same as other groups
   • Shows baseline fluorescence levels

4. DETAILED ANALYSIS
   • Click any group to view details
   • Overview tab: Summary and interpretation
   • G+/G- tabs: Individual microgel statistics
   • Processing Steps tab: View processing pipeline images
   • Plots tab: Visual comparisons
   • Raw Data tab: Complete dataset

5. PROCESSING STEPS VIEWER
   • View processing pipeline images side-by-side for G+ and G-
   • Control group shows same images for both columns
   • Select a processing step from the tree
   • Images show up for both Positive and Negative samples
   • Save individual images if needed
   • Color indicators show availability:
     ✓ Green: Available for both G+ and G-
     ◐ Orange: Available for one type only
     ✗ Gray: Not available

6. FILTERING
   • Use the filter dropdown to show specific classifications
   • Filter by: All, POSITIVE, NEGATIVE, NO OBVIOUS BACTERIA, MIXED, CONTROL

7. EXPORTING
   • Export to CSV: Raw data table
   • Export to PDF: Complete formatted report
   • Open Output Folder: View all files

8. KEYBOARD SHORTCUTS
   • Ctrl+O: Open results folder
   • Ctrl+R: Open recent folder
   • F5: Refresh view
   • Ctrl+Q: Quit application

9. INTERPRETATION
   • POSITIVE: Bacteria detected, consider treatment
   • NEGATIVE: No bacteria detected
   • NO OBVIOUS BACTERIA: Within control range
   • MIXED: Conflicting results, repeat testing recommended
   • CONTROL: Reference standard for comparison

10. TROUBLESHOOTING
    • If plots don't display: Check image files in output folder
    • If data is missing: Ensure complete processing pipeline
    • For batch mode: Both G+ and G- folders must exist
    • Processing images: Must be in subfolder structure
    • Control group: Folder must be named "Control group" or "Control"

For additional support, refer to the processing pipeline documentation.
        """
        
        help_text.insert('1.0', help_content)
        help_text.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(help_window, text="Close",
                  command=help_window.destroy).pack(pady=10)


def launch_viewer(initial_folder: Optional[Path] = None):
    """Launch the GUI viewer"""
    root = tk.Tk()
    
    # Set window icon if available
    try:
        # You can add an icon file here
        # root.iconbitmap('icon.ico')
        pass
    except:
        pass
    
    app = ClinicalResultsViewer(root)
    
    # Load initial folder if provided
    if initial_folder and initial_folder.exists():
        app.load_results_from_folder(initial_folder)
    
    root.mainloop()


if __name__ == "__main__":
    # Only run if not being imported by PyInstaller
    if not getattr(sys, 'frozen', False):
        launch_viewer()