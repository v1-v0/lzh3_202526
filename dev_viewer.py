"""
Cross-Platform Clinical Results Viewer
Compatible with Windows, macOS, and Linux
Now includes Processing Steps Viewer with Control Group Support and Multi-Sample Navigation
Version 2.2 - Enhanced multi-sample browsing
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
import json
from typing import Optional, Dict, List, Set
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
        self.root.title("Particle-Scout Clinical Results Viewer v2.2")
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
        
        # Multi-sample navigation
        self.pos_sample_folders: List[Path] = []
        self.neg_sample_folders: List[Path] = []
        self.current_pos_sample_index: int = 0
        self.current_neg_sample_index: int = 0
        self.current_step_filename: Optional[str] = None
        self.current_step_description: str = ""
        
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
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Load preferences
        self.load_preferences()

    def find_group_folders(self, parent_dir: Path, img_type: Optional[str] = None) -> List[Path]:
        """Find group folders in both single and batch mode structures
        
        Args:
            parent_dir: Root output directory
            img_type: 'positive', 'negative', or None for single mode
            
        Returns:
            List of group folder paths
        """
        group_folders = []
        
        # Check if batch mode (has Positive/Negative subfolders)
        positive_dir = parent_dir / "Positive"
        negative_dir = parent_dir / "Negative"
        
        is_batch_mode = positive_dir.exists() or negative_dir.exists()
        
        if is_batch_mode:
            # Batch mode: look inside Positive or Negative folder
            if img_type == "positive" and positive_dir.exists():
                search_dir = positive_dir
            elif img_type == "negative" and negative_dir.exists():
                search_dir = negative_dir
            else:
                print(f"[WARN] {img_type} folder not found in batch mode")
                return []
            
            # Find all group folders inside the type folder
            for item in search_dir.iterdir():
                if item.is_dir():
                    # Check if it contains sample subfolders with images
                    sample_folders = [d for d in item.iterdir() 
                                    if d.is_dir() and any(d.glob("*.png"))]
                    if sample_folders:
                        group_folders.append(item)
        else:
            # Single mode: groups are directly in output_dir
            for item in parent_dir.iterdir():
                if item.is_dir():
                    # Check if it contains sample subfolders with images
                    sample_folders = [d for d in item.iterdir() 
                                    if d.is_dir() and any(d.glob("*.png"))]
                    if sample_folders:
                        group_folders.append(item)
        
        return sorted(group_folders)


    def find_group_folder(self, parent_dir: Path, group_name: str) -> Optional[Path]:
        """Find group folder with flexible naming (handles 'Control group' vs 'Control')
        
        Args:
            parent_dir: Directory to search in (either root, Positive, or Negative folder)
            group_name: Group name to find (e.g., '1', '2', 'Control')
            
        Returns:
            Path to group folder, or None if not found
        """
        if not parent_dir.exists():
            return None
        
        # Normalize group name
        group_normalized = group_name.lower().strip()
        
        # For control group, check multiple variations
        if 'control' in group_normalized:
            for folder in parent_dir.iterdir():
                if folder.is_dir():
                    folder_normalized = folder.name.lower().strip()
                    # Match any variation: "Control", "Control group", "control_group", etc.
                    if 'control' in folder_normalized:
                        return folder
            return None
        
        # For numbered groups, try exact match first
        exact_match = parent_dir / group_name
        if exact_match.exists():
            return exact_match
        
        # Try case-insensitive match
        for folder in parent_dir.iterdir():
            if folder.is_dir() and folder.name.lower() == group_name.lower():
                return folder
        
        return None


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
                             command=self.on_closing,
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
        self.root.bind('<Control-q>', lambda e: self.on_closing())
    
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
                command=self.on_closing).pack(side=tk.LEFT, padx=2, pady=2)
        
        # Dataset label (right side)
        self.dataset_label = ttk.Label(toolbar, text="No dataset loaded", 
                                    style='Status.TLabel')
        self.dataset_label.pack(side=tk.RIGHT, padx=10)
    
    def close_application(self):
        """Close the application"""
        self.on_closing()

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
        
        # Sample navigation (left side)
        pos_nav_frame = ttk.Frame(pos_controls)
        pos_nav_frame.pack(side=tk.LEFT)
        
        ttk.Button(pos_nav_frame, text="◀ Prev Sample", 
                  command=lambda: self.navigate_sample('positive', -1),
                  width=12).pack(side=tk.LEFT, padx=2)
        
        self.pos_sample_label = ttk.Label(pos_nav_frame, text="1/1")
        self.pos_sample_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(pos_nav_frame, text="Next Sample ▶",
                  command=lambda: self.navigate_sample('positive', 1),
                  width=12).pack(side=tk.LEFT, padx=2)
        
        # Zoom/Save controls (right side)
        pos_zoom_frame = ttk.Frame(pos_controls)
        pos_zoom_frame.pack(side=tk.RIGHT)
        
        ttk.Button(pos_zoom_frame, text="Save Image",
                  command=lambda: self.save_processing_image('positive')).pack(side=tk.RIGHT, padx=2)
        
        self.pos_zoom_label = ttk.Label(pos_zoom_frame, text="100%")
        self.pos_zoom_label.pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(pos_zoom_frame, text="Fit to Window", 
                  command=lambda: self.fit_to_window('positive')).pack(side=tk.RIGHT, padx=2)
        
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
        
        # Sample navigation (left side)
        neg_nav_frame = ttk.Frame(neg_controls)
        neg_nav_frame.pack(side=tk.LEFT)
        
        ttk.Button(neg_nav_frame, text="◀ Prev Sample",
                  command=lambda: self.navigate_sample('negative', -1),
                  width=12).pack(side=tk.LEFT, padx=2)
        
        self.neg_sample_label = ttk.Label(neg_nav_frame, text="1/1")
        self.neg_sample_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(neg_nav_frame, text="Next Sample ▶",
                  command=lambda: self.navigate_sample('negative', 1),
                  width=12).pack(side=tk.LEFT, padx=2)
        
        # Zoom/Save controls (right side)
        neg_zoom_frame = ttk.Frame(neg_controls)
        neg_zoom_frame.pack(side=tk.RIGHT)
        
        ttk.Button(neg_zoom_frame, text="Save Image",
                  command=lambda: self.save_processing_image('negative')).pack(side=tk.RIGHT, padx=2)
        
        self.neg_zoom_label = ttk.Label(neg_zoom_frame, text="100%")
        self.neg_zoom_label.pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(neg_zoom_frame, text="Fit to Window",
                  command=lambda: self.fit_to_window('negative')).pack(side=tk.RIGHT, padx=2)
        
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
        
        # Store current step info
        self.current_step_filename = filename
        self.current_step_description = description
        
        if self.selected_group is None or self.current_folder is None:
            self.update_processing_description("Please select a group first", 'both')
            return
        
        # Reset sample indices when changing steps
        self.current_pos_sample_index = 0
        self.current_neg_sample_index = 0
        
        # Load both images
        self.load_processing_image_by_index('positive', filename, description, 0)
        self.load_processing_image_by_index('negative', filename, description, 0)
    

    def navigate_sample(self, img_type: str, direction: int):
        """Navigate to next/previous sample (independent navigation)"""
        # Get the appropriate folder list
        if img_type == 'positive':
            folders = self.pos_sample_folders
            current_idx = self.current_pos_sample_index
        else:
            folders = self.neg_sample_folders
            current_idx = self.current_neg_sample_index
        
        if not folders:
            messagebox.showwarning("No Samples", f"No {img_type} samples found for this group")
            return
        
        # Calculate new index with wrap-around
        new_idx = current_idx + direction
        
        if new_idx < 0:
            new_idx = len(folders) - 1
        elif new_idx >= len(folders):
            new_idx = 0
        
        # **FIX: Update only the relevant index (independent navigation)**
        if img_type == 'positive':
            self.current_pos_sample_index = new_idx
        else:
            self.current_neg_sample_index = new_idx
        
        # Reload only the selected type with bounds checking
        if self.current_step_filename:
            self.load_processing_image_by_index(img_type, 
                                                self.current_step_filename,
                                                self.current_step_description,
                                                new_idx)



    def load_processing_image(self, img_type: str, filename: str, description: str):
        """Load processing image for specific type (uses current index)"""
        if img_type == 'positive':
            idx = self.current_pos_sample_index
        else:
            idx = self.current_neg_sample_index
        
        self.load_processing_image_by_index(img_type, filename, description, idx)


    def load_processing_image_by_index(self, img_type: str, filename: str, 
                                    description: str, sample_idx: int):
        """Load processing image for specific type and sample index - BATCH MODE COMPATIBLE
        
        Args:
            img_type: 'positive' or 'negative'
            filename: Image filename to load
            description: Step description
            sample_idx: Sample index within group
        """
        if self.current_folder is None or self.selected_group is None:
            self.update_processing_description("Please select a group first", img_type)
            self.clear_processing_canvas(img_type)
            self.update_sample_label(img_type, 0, 0)
            return
        
        # **DEBUG: Print current state**
        print(f"\n=== DEBUG load_processing_image_by_index ===")
        print(f"img_type: {img_type}")
        print(f"filename: {filename}")
        print(f"selected_group: {self.selected_group}")
        print(f"sample_idx: {sample_idx}")
        
        # **FIX: Use pre-populated sample folder lists**
        sample_folders = self.pos_sample_folders if img_type == 'positive' else self.neg_sample_folders
        
        print(f"sample_folders count: {len(sample_folders)}")
        
        # Check if sample folders exist
        if not sample_folders:
            msg = f"No {img_type} samples found for Group {self.selected_group}"
            print(f"ERROR: {msg}")
            self.update_processing_description(msg, img_type)
            self.clear_processing_canvas(img_type)
            self.update_sample_label(img_type, 0, 0)
            
            # Update title to show no samples
            if img_type == 'positive':
                self.pos_title.config(text="(No samples)")
            else:
                self.neg_title.config(text="(No samples)")
            return
        
        # **FIX: Strict bounds checking**
        if sample_idx >= len(sample_folders):
            sample_idx = len(sample_folders) - 1
        if sample_idx < 0:
            sample_idx = 0
        
        # **FIX: Update ONLY the appropriate index**
        if img_type == 'positive':
            self.current_pos_sample_index = sample_idx
        else:
            self.current_neg_sample_index = sample_idx
        
        # Get the sample folder
        sample_folder = sample_folders[sample_idx]
        image_path = sample_folder / filename
        
        print(f"sample_folder: {sample_folder}")
        print(f"image_path: {image_path}")
        print(f"image_path.exists(): {image_path.exists()}")
        
        # Update title with sample name
        sample_name = sample_folder.name
        if img_type == 'positive':
            self.pos_title.config(text=f"[{sample_name}]")
        else:
            self.neg_title.config(text=f"[{sample_name}]")
        
        # Check if image exists
        if not image_path.exists():
            # **DEBUG: List what files ARE in the folder**
            print(f"Available files in {sample_folder}:")
            try:
                for file in sorted(sample_folder.glob("*.png"))[:10]:  # First 10 PNG files
                    print(f"  - {file.name}")
            except Exception as e:
                print(f"Error listing files: {e}")
            
            msg = f"Image not found: {filename}\n"
            msg += f"{description}\n"
            msg += f"Sample: {sample_name} ({sample_idx + 1}/{len(sample_folders)})\n"
            msg += f"Path: {image_path}"
            self.update_processing_description(msg, img_type)
            self.clear_processing_canvas(img_type)
            self.update_sample_label(img_type, sample_idx + 1, len(sample_folders))
            return
        
        try:
            # Load image
            print(f"Loading image: {image_path}")
            img = Image.open(image_path)
            print(f"Image loaded successfully: {img.size}")
            
            # Store original image reference
            if img_type == 'positive':
                self.current_pos_image = img.copy()
                self.current_pos_path = image_path
                canvas = self.pos_canvas
            else:
                self.current_neg_image = img.copy()
                self.current_neg_path = image_path
                canvas = self.neg_canvas
            
            # Get canvas dimensions
            canvas.update_idletasks()  # Force update to get actual size
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            # Use minimum size if canvas not yet rendered
            if canvas_width <= 1:
                canvas_width = 600
            if canvas_height <= 1:
                canvas_height = 400
            
            # Calculate scaling to fit canvas while maintaining aspect ratio
            img_width, img_height = img.size
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            scale = min(scale_w, scale_h, 1.0) * 0.95  # 95% to leave margin
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize image with high-quality resampling
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img_resized)
            
            # Store reference to prevent garbage collection
            self.photo_refs[id(photo)] = photo
            
            if img_type == 'positive':
                self.photo_positive = photo
            else:
                self.photo_negative = photo
            
            # Clear canvas and display image
            canvas.delete("all")
            x = canvas_width // 2
            y = canvas_height // 2
            canvas.create_image(x, y, image=photo, anchor=tk.CENTER)
            
            # Update zoom label
            zoom_pct = int(scale * 100)
            if img_type == 'positive':
                self.pos_zoom_label.config(text=f"{zoom_pct}%")
            else:
                self.neg_zoom_label.config(text=f"{zoom_pct}%")
            
            # Update description with full information
            desc_text = f"{description}\n"
            desc_text += f"Sample: {sample_name} ({sample_idx + 1}/{len(sample_folders)})\n"
            desc_text += f"Image: {filename}\n"
            desc_text += f"Size: {img_width} × {img_height} px"
            
            self.update_processing_description(desc_text, img_type)
            
            # Update sample counter
            self.update_sample_label(img_type, sample_idx + 1, len(sample_folders))
            
        except FileNotFoundError:
            msg = f"File not found: {filename}\n"
            msg += f"Path: {image_path}\n"
            msg += f"Sample: {sample_name} ({sample_idx + 1}/{len(sample_folders)})"
            self.update_processing_description(msg, img_type)
            self.clear_processing_canvas(img_type)
            self.update_sample_label(img_type, sample_idx + 1, len(sample_folders))
            
        except (OSError, UnicodeDecodeError) as e:
            msg = f"Error accessing file (encoding issue):\n{filename}\n"
            msg += f"Error: {str(e)}\n"
            msg += f"Sample: {sample_name} ({sample_idx + 1}/{len(sample_folders)})"
            self.update_processing_description(msg, img_type)
            self.clear_processing_canvas(img_type)
            self.update_sample_label(img_type, sample_idx + 1, len(sample_folders))
            
        except Exception as e:
            msg = f"Error loading image: {str(e)}\n"
            msg += f"File: {filename}\n"
            msg += f"Sample: {sample_name} ({sample_idx + 1}/{len(sample_folders)})\n"
            msg += f"Path: {image_path}"
            self.update_processing_description(msg, img_type)
            self.clear_processing_canvas(img_type)
            self.update_sample_label(img_type, sample_idx + 1, len(sample_folders))

    def prev_pos_sample(self):
        """Navigate to previous positive sample (updates both)"""
        if not self.current_step_filename or not self.current_step_description:
            return
        
        new_idx = self.current_pos_sample_index - 1
        if new_idx < 0:
            new_idx = max(len(self.pos_sample_folders) - 1, 0)
        
        # Load both positive and negative with the same index
        self.load_processing_image_by_index('positive', 
                                            self.current_step_filename,
                                            self.current_step_description,
                                            new_idx)
        self.load_processing_image_by_index('negative',
                                            self.current_step_filename,
                                            self.current_step_description,
                                            new_idx)

    def next_pos_sample(self):
        """Navigate to next positive sample (updates both)"""
        if not self.current_step_filename or not self.current_step_description:
            return
        
        new_idx = self.current_pos_sample_index + 1
        if new_idx >= len(self.pos_sample_folders):
            new_idx = 0
        
        # Load both positive and negative with the same index
        self.load_processing_image_by_index('positive',
                                            self.current_step_filename,
                                            self.current_step_description,
                                            new_idx)
        self.load_processing_image_by_index('negative',
                                            self.current_step_filename,
                                            self.current_step_description,
                                            new_idx)

    def prev_neg_sample(self):
        """Navigate to previous negative sample (updates both)"""
        if not self.current_step_filename or not self.current_step_description:
            return
        
        new_idx = self.current_neg_sample_index - 1
        if new_idx < 0:
            new_idx = max(len(self.neg_sample_folders) - 1, 0)
        
        # Load both positive and negative with the same index
        self.load_processing_image_by_index('positive',
                                            self.current_step_filename,
                                            self.current_step_description,
                                            new_idx)
        self.load_processing_image_by_index('negative',
                                            self.current_step_filename,
                                            self.current_step_description,
                                            new_idx)

    def next_neg_sample(self):
        """Navigate to next negative sample (updates both)"""
        if not self.current_step_filename or not self.current_step_description:
            return
        
        new_idx = self.current_neg_sample_index + 1
        if new_idx >= len(self.neg_sample_folders):
            new_idx = 0
        
        # Load both positive and negative with the same index
        self.load_processing_image_by_index('positive',
                                            self.current_step_filename,
                                            self.current_step_description,
                                            new_idx)
        self.load_processing_image_by_index('negative',
                                            self.current_step_filename,
                                            self.current_step_description,
                                            new_idx)

    def update_sample_label(self, img_type: str, current: int, total: int):
        """Update the sample counter label"""
        label = self.pos_sample_label if img_type == 'positive' else self.neg_sample_label
        label.config(text=f"{current}/{total}")

    def display_processing_step_image(self, image_path: Path, img_type: str):
        """Display a processing step image on the appropriate canvas"""
        canvas = self.pos_canvas if img_type == 'positive' else self.neg_canvas
        
        # Load image
        image = Image.open(str(image_path))
        
        # Store original
        if img_type == 'positive':
            self.current_pos_image = image
        else:
            self.current_neg_image = image
        
        # Fit to canvas
        self.fit_image_to_canvas(image, canvas, img_type)
    
    def fit_image_to_canvas(self, image: Image.Image, canvas: tk.Canvas, img_type: str):
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
    
    def fit_to_window(self, img_type: str):
        """Fit image to window"""
        image = self.current_pos_image if img_type == 'positive' else self.current_neg_image
        canvas = self.pos_canvas if img_type == 'positive' else self.neg_canvas
        
        if image:
            self.fit_image_to_canvas(image, canvas, img_type)
    
    def clear_processing_canvas(self, img_type: str):
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
    
    def update_processing_description(self, text: str, img_type: str):
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
    
    def save_processing_image(self, img_type: str):
        """Save the current processing step image"""
        image_path = self.current_pos_path if img_type == 'positive' else self.current_neg_path
        
        if not image_path:
            messagebox.showwarning("No Image", "No image to save")
            return
        
        if self.selected_group is None:
            messagebox.showwarning("No Group", "No group selected")
            return
        
        # Get sample info
        if img_type == 'positive':
            sample_idx = self.current_pos_sample_index
            sample_folders = self.pos_sample_folders
        else:
            sample_idx = self.current_neg_sample_index
            sample_folders = self.neg_sample_folders
        
        sample_name = sample_folders[sample_idx].name if sample_folders else "sample"
        
        # Suggest filename
        type_label = "Positive" if img_type == 'positive' else "Negative"
        default_name = f"{self.selected_group}_{type_label}_{sample_name}_{image_path.name}"
        
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
    
    def safe_read_csv(self, path: Path) -> pd.DataFrame:
        """Read CSV with multiple encoding attempts"""
        encodings = ['utf-8', 'utf-8-sig', 'gb18030', 'gbk', 'latin1']
        
        for encoding in encodings:
            try:
                return pd.read_csv(path, encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        raise ValueError(f"Could not read CSV with any supported encoding: {path.name}")
    
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
    
    def debug_print_structure(self):
        """Debug: Print folder structure"""
        if self.current_folder is None:
            print("No folder loaded")
            return
        
        print("\n=== FOLDER STRUCTURE ===")
        print(f"Root: {self.current_folder}")
        
        for item in sorted(self.current_folder.iterdir())[:20]:  # First 20 items
            if item.is_dir():
                print(f"\n📁 {item.name}/")
                for subitem in sorted(item.iterdir())[:5]:  # First 5 subitems
                    if subitem.is_dir():
                        png_count = len(list(subitem.glob("*.png")))
                        print(f"  📁 {subitem.name}/ ({png_count} PNGs)")
                    else:
                        print(f"  📄 {subitem.name}")
    
    
    
    
    def load_results_from_folder(self, folder_path: Path):
        """Load and parse results from folder"""
        try:
            self.status_var.set(f"Loading from {folder_path.name}...")
            self.current_folder = folder_path

            # **DEBUG: Print structure**
            self.debug_print_structure()
            
            # Clear existing data
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Clean up photo references
            self.cleanup_photo_refs()
            
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
            
            # Auto-select first item
            if self.tree.get_children():
                first_item = self.tree.get_children()[0]
                self.tree.selection_set(first_item)
                self.tree.focus(first_item)
                self.on_select(None)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load results:\n{str(e)}")
            self.status_var.set("Error loading results")
    
    def load_batch_results(self, folder_path: Path, clinical_matrix: Path):
        """Load batch mode results (G+ and G-)"""
        self.clinical_data = self.safe_read_csv(clinical_matrix)
        self.dataset_name = folder_path.name
        
        # Load G+ and G- classifications
        gplus_csv = folder_path / "Positive" / "clinical_classification_positive_default.csv"
        gminus_csv = folder_path / "Negative" / "clinical_classification_negative_default.csv"
        
        # Try with bacteria profile suffix if not found
        if not gplus_csv.exists():
            gplus_csv = list((folder_path / "Positive").glob("clinical_classification_positive*.csv"))
            gplus_csv = gplus_csv[0] if gplus_csv else None
        
        if not gminus_csv.exists():
            gminus_csv = list((folder_path / "Negative").glob("clinical_classification_negative*.csv"))
            gminus_csv = gminus_csv[0] if gminus_csv else None
        
        if gplus_csv and gplus_csv.exists():
            self.gplus_data = self.safe_read_csv(gplus_csv)
        
        if gminus_csv and gminus_csv.exists():
            self.gminus_data = self.safe_read_csv(gminus_csv)
        
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
        self.clinical_data = self.safe_read_csv(classification_csv)
        
        self.dataset_name = folder_path.name
        
        # Determine type from filename
        if "positive" in classification_csv.name.lower():
            self.gplus_data = self.clinical_data
            microgel_type = "G+"
        else:
            self.gminus_data = self.clinical_data
            microgel_type = "G-"
        
        # Check for Control folder
        control_folder = folder_path / "Control"
        
        if control_folder.exists() and control_folder.is_dir():
            # Try to get Control group statistics
            stats_summary_csv = folder_path / "group_statistics_summary.csv"
            
            if stats_summary_csv.exists():
                try:
                    stats_data = self.safe_read_csv(stats_summary_csv)
                    
                    # Find Control row
                    control_rows = stats_data[stats_data['Group'].astype(str).str.lower() == 'control']
                    
                    if not control_rows.empty:
                        control_row = control_rows.iloc[0]
                        control_mean = control_row.get('Mean', None)
                        control_n = control_row.get('N', None)
                        control_std = control_row.get('Std_Dev', None)
                        
                        # Only add if we have valid data
                        if control_mean is not None and control_mean > 0:
                            # Check if Control is already in data
                            if 'Control' not in self.clinical_data['Group'].astype(str).values and \
                               'control' not in self.clinical_data['Group'].astype(str).str.lower().values:
                                
                                control_entry = {
                                    'Group': 'Control',
                                    'N': control_n if control_n is not None else 0,
                                    'Mean': control_mean,
                                    'Std_Dev': control_std if control_std is not None else 0,
                                    'Control_Mean': control_mean,
                                    'Threshold': control_mean,
                                    'Diff_from_Threshold': 0,
                                    'Classification': 'CONTROL (Reference)'
                                }
                                
                                self.clinical_data = pd.concat([
                                    self.clinical_data, 
                                    pd.DataFrame([control_entry])
                                ], ignore_index=True)
                except Exception as e:
                    print(f"Warning: Could not load control statistics: {e}")
        
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
    
    def cleanup_photo_refs(self):
        """Clean up old photo references to prevent memory leaks"""
        # Get all current widget IDs
        current_ids = set()
        for widget in self.root.winfo_children():
            current_ids.update(self._get_widget_ids(widget))
        
        # Remove old references
        self.photo_refs = {k: v for k, v in self.photo_refs.items() if k in current_ids}
    
    def _get_widget_ids(self, widget) -> Set[int]:
        """Recursively get all widget IDs"""
        ids = {id(widget)}
        for child in widget.winfo_children():
            ids.update(self._get_widget_ids(child))
        return ids
    
    def on_select(self, event):
        """Handle tree selection"""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = selection[0]
        group_text = self.tree.item(item, "text")
        
        # Extract group number
        self.selected_group = group_text.replace("Group ", "")
        
        # Reset sample indices
        self.current_pos_sample_index = 0
        self.current_neg_sample_index = 0
        
        # Clean up old photos before loading new data
        self.cleanup_photo_refs()
        
        # Update all tabs
        self.display_overview()
        self.display_gplus_details()
        self.display_gminus_details()
        self.display_plots()
        self.display_raw_data()
        
        # Update processing steps availability
        self.update_step_availability()
        
        # Reload current processing step if one was selected
        if self.current_step_filename:
            self.load_processing_image_by_index('positive', 
                                                self.current_step_filename,
                                                self.current_step_description, 
                                                0)
            self.load_processing_image_by_index('negative',
                                                self.current_step_filename,
                                                self.current_step_description,
                                                0)
        
        self.info_var.set(f"Selected: {group_text}")
    


    def update_step_availability(self):
        """Update step tree to show which images are available"""
        if self.selected_group is None or self.current_folder is None:
            return
        
        # **FIX: Detect batch mode and navigate correctly**
        positive_dir = self.current_folder / "Positive"
        negative_dir = self.current_folder / "Negative"
        is_batch_mode = positive_dir.exists() or negative_dir.exists()
        
        if is_batch_mode:
            # Batch mode: groups are inside Positive/Negative folders
            pos_group_folder = self.find_group_folder(positive_dir, self.selected_group)
            neg_group_folder = self.find_group_folder(negative_dir, self.selected_group)
        else:
            # Single mode: groups are at root level
            pos_group_folder = self.find_group_folder(self.current_folder, self.selected_group)
            neg_group_folder = None  # No separate negative in single mode
        
        # **CRITICAL FIX: Populate sample folder lists**
        self.pos_sample_folders = []
        self.neg_sample_folders = []
        
        if pos_group_folder and pos_group_folder.exists():
            # Get all subdirectories that contain images
            self.pos_sample_folders = sorted([d for d in pos_group_folder.iterdir() 
                                            if d.is_dir() and any(d.glob("*.png"))])
            
            print(f"DEBUG: Found {len(self.pos_sample_folders)} positive samples in {pos_group_folder}")
            for folder in self.pos_sample_folders[:3]:  # Print first 3
                print(f"  - {folder.name}")
        else:
            print(f"DEBUG: Positive group folder not found for {self.selected_group}")
        
        if neg_group_folder and neg_group_folder.exists():
            # Get negative samples
            self.neg_sample_folders = sorted([d for d in neg_group_folder.iterdir() 
                                            if d.is_dir() and any(d.glob("*.png"))])
            
            print(f"DEBUG: Found {len(self.neg_sample_folders)} negative samples in {neg_group_folder}")
            for folder in self.neg_sample_folders[:3]:
                print(f"  - {folder.name}")
        else:
            print(f"DEBUG: Negative group folder not found for {self.selected_group}")
        
        # Get first image folders for checking availability
        pos_images = self.pos_sample_folders[0] if self.pos_sample_folders else None
        neg_images = self.neg_sample_folders[0] if self.neg_sample_folders else None
        
        # Update tree items
        for phase_item in self.steps_tree.get_children():
            for step_item in self.steps_tree.get_children(phase_item):
                current_text = self.steps_tree.item(step_item, "text")
                filename = current_text.replace("○ ", "").replace("✓ ", "").replace("◐ ", "").replace("✗ ", "")
                
                # Check if file exists
                pos_exists = pos_images is not None and (pos_images / filename).exists()
                neg_exists = neg_images is not None and (neg_images / filename).exists()
                
                if pos_exists and neg_exists:
                    self.steps_tree.item(step_item, text=f"✓ {filename}", 
                                    tags=('step', 'available'))
                elif pos_exists or neg_exists:
                    self.steps_tree.item(step_item, text=f"◐ {filename}",
                                    tags=('step', 'partial'))
                else:
                    self.steps_tree.item(step_item, text=f"✗ {filename}",
                                    tags=('step', 'missing'))
        
        # Configure tags
        self.steps_tree.tag_configure('available', foreground='green')
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
            ("Z-Score:", f"{data.get('Z_Score', 'N/A'):.2f}" if isinstance(data.get('Z_Score'), (int, float)) else 'N/A'),
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
        
        # Clean up old photo references for plots tab
        plt.close('all')  # Close any matplotlib figures
        
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
    
    # ==================== Preferences ====================
    
    def save_preferences(self):
        """Save user preferences"""
        prefs = {
            'last_folder': str(self.current_folder) if self.current_folder else None,
            'filter': self.filter_var.get(),
            'window_geometry': self.root.geometry()
        }
        
        prefs_file = Path.home() / '.particle_scout_viewer.json'
        try:
            with open(prefs_file, 'w') as f:
                json.dump(prefs, f)
        except Exception:
            pass
    
    def load_preferences(self):
        """Load user preferences"""
        prefs_file = Path.home() / '.particle_scout_viewer.json'
        if prefs_file.exists():
            try:
                with open(prefs_file, 'r') as f:
                    prefs = json.load(f)
                    
                if prefs.get('last_folder'):
                    last_folder = Path(prefs['last_folder'])
                    if last_folder.exists():
                        # Don't auto-load on startup, just store the path
                        # User can use Ctrl+R to load recent
                        pass
                
                if prefs.get('window_geometry'):
                    try:
                        self.root.geometry(prefs['window_geometry'])
                    except:
                        pass
                    
            except Exception:
                pass
    
    def on_closing(self):
        """Handle window close event"""
        self.save_preferences()
        self.root.quit()
        self.root.destroy()
    
    # ==================== Help Functions ====================
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
Particle-Scout Clinical Results Viewer
Version 2.2 - Multi-Sample Navigation

A cross-platform application for reviewing
bacterial detection results using microgel
fluorescence analysis.

Features:
• Batch processing (G+ and G-)
• Clinical classification
• Control group visualization
• Multi-sample browsing
• Processing steps visualization
• Data export (CSV/PDF)
• Adaptive thresholding support

© 2026 - Clinical Diagnostics
        """
        
        messagebox.showinfo("About", about_text.strip())
    
    def show_help(self):
        """Show help dialog"""
        help_window = tk.Toplevel(self.root)
        help_window.title("User Guide")
        help_window.geometry("700x800")
        
        # Scrolled text for help
        help_text = scrolledtext.ScrolledText(help_window, wrap=tk.WORD,
                                              font=('Arial', 10))
        help_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        help_content = """
PARTICLE-SCOUT CLINICAL RESULTS VIEWER - USER GUIDE v2.2

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
   
3. CONTROL GROUP
   • Control group appears in light blue
   • Used as reference standard for calculations
   • View processing steps same as other groups
   • Shows baseline fluorescence levels
   • Folder must be named "Control" (no space)

4. DETAILED ANALYSIS
   • Click any group to view details
   • Overview tab: Summary and interpretation
   • G+/G- tabs: Individual microgel statistics
   • Processing Steps tab: View processing pipeline images
   • Plots tab: Visual comparisons
   • Raw Data tab: Complete dataset

5. PROCESSING STEPS VIEWER - MULTI-SAMPLE NAVIGATION
   • View processing pipeline images side-by-side for G+ and G-
   • Navigate through multiple samples using arrow buttons:
     - "◀ Prev Sample" / "Next Sample ▶"
     - Sample counter shows current position (e.g., "2/5")
   • Independent navigation for G+ and G-
   • Select a processing step from the tree to view
   • Color indicators show availability:
     ✓ Green: Available for both G+ and G-
     ◐ Orange: Available for one type only
     ✗ Gray: Not available
   • Save individual images for any sample

6. SAMPLE NAVIGATION
   • Each group may contain multiple samples (biological replicates)
   • Use navigation buttons to browse all samples
   • Sample name displayed in title (e.g., "Sample 1/3 - image_001.tif")
   • Navigation wraps around (first ↔ last)
   • G+ and G- samples can be navigated independently

7. FILTERING
   • Use the filter dropdown to show specific classifications
   • Filter by: All, POSITIVE, NEGATIVE, NO OBVIOUS BACTERIA, MIXED, CONTROL

8. EXPORTING
   • Export to CSV: Raw data table
   • Export to PDF: Complete formatted report (requires reportlab)
   • Open Output Folder: View all files

9. KEYBOARD SHORTCUTS
   • Ctrl+O: Open results folder
   • Ctrl+R: Open recent folder
   • F5: Refresh view
   • Ctrl+Q: Quit application

10. INTERPRETATION
    • POSITIVE: Bacteria detected, consider treatment
    • NEGATIVE: No bacteria detected
    • NO OBVIOUS BACTERIA: Within control range
    • MIXED: Conflicting results, repeat testing recommended
    • CONTROL: Reference standard for comparison

11. TROUBLESHOOTING
    • If plots don't display: Check image files in output folder
    • If data is missing: Ensure complete processing pipeline
    • For batch mode: Both G+ and G- folders must exist
    • Processing images: Must be in subfolder structure
    • Control group: Folder must be named "Control" (no space)
    • Multiple samples: Each sample in separate subfolder
    • Unicode paths: Ensure proper encoding in folder names

12. NEW IN VERSION 2.2
    • Multi-sample navigation in Processing Steps viewer
    • Independent sample browsing for G+ and G-
    • Sample counter display
    • Enhanced sample information in titles
    • Wrap-around navigation
    • Automatic sample detection
    • Improved memory management

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