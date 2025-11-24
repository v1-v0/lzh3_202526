#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive Bacteria Segmentation Parameter Tuner
- Enhanced folder selection with subfolder picker
- Smart label positioning and bacteria navigation
- All bugs fixed
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
        self.header = ttk.Frame(self, relief=tk.RAISED, borderwidth=1)
        self.header.pack(fill=tk.X, padx=2, pady=2)
        self.toggle_btn = ttk.Button(self.header, text="▼", width=3, command=self.toggle)
        self.toggle_btn.pack(side=tk.LEFT, padx=2)
        self.title_label = ttk.Label(self.header, text=title, font=("Segoe UI", 10, "bold"))
        self.title_label.pack(side=tk.LEFT, padx=5)
        self.content = ttk.Frame(self, padding=10)
        self.content.pack(fill=tk.BOTH, expand=True)

    def toggle(self):
        if self.is_collapsed:
            self.expand()
        else:
            self.collapse()

    def collapse(self):
        self.content.pack_forget()
        self.toggle_btn.config(text="▶")
        self.is_collapsed = True

    def expand(self):
        self.content.pack(fill=tk.BOTH, expand=True)
        self.toggle_btn.config(text="▼")
        self.is_collapsed = False

    def get_content_frame(self):
        return self.content


class SegmentationViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Bacteria Segmentation Tuner")
        self.root.geometry("1500x950")
        self.root.protocol("WM_DELETE_WINDOW", self.exit_application)

        self.original_image: Optional[np.ndarray] = None
        self.fluorescence_image: Optional[np.ndarray] = None
        self.current_contours: List[np.ndarray] = []
        self.probe_point: Optional[Tuple[int, int]] = None
        self.probe_canvas_ids: List[int] = []
        self.current_file: Optional[Path] = None

        self.bacteria_stats: List[Dict] = []
        self.current_bacteria_index: int = -1

        # Start with current directory instead of hardcoded "source/"
        self.source_dir = Path.cwd()
        self.image_files: List[Path] = []
        self.current_index: int = -1

        # Default parameters -------------------------------------------------
        self.default_params = {
            'use_otsu': False, 'manual_threshold': 50, 'enable_clahe': True,
            'clahe_clip': 5.0, 'clahe_tile': 16, 'open_kernel': 3, 'close_kernel': 5,
            'open_iter': 3, 'close_iter': 2, 'min_area': 50, 'watershed_dilate': 15,
            'fluor_brightness': 2.0, 'fluor_gamma': 0.5, 'show_labels': True,
            'label_font_size': 20, 'arrow_length': 60, 'label_offset': 15,
            'min_fluor_per_area': 10.0,
        }

        self.params: Dict[str, tk.Variable] = {
            k: tk.BooleanVar(value=v) if isinstance(v, bool) else
            tk.IntVar(value=v) if isinstance(v, int) else
            tk.DoubleVar(value=v)
            for k, v in self.default_params.items()
        }

        self.entries: Dict[str, tk.Entry] = {}
        self.progressbars: Dict[str, ttk.Progressbar] = {}
        self.measure_labels: Dict[str, ttk.Label] = {}
        self.compact_stats_labels: Dict[str, ttk.Label] = {}

        self.setup_ui()
        self.root.bind("<Configure>", lambda e: self.root.after_idle(self.update_preview))

    # --------------------------------------------------------------------- #
    # Folder / file handling
    # --------------------------------------------------------------------- #
    def is_valid_image_file(self, filepath: Path) -> bool:
        """Return True only for real bright-field files."""
        name = filepath.name
        ok = (
            not name.startswith('._') and      # skip AppleDouble files
            not name.startswith('.') and       # skip hidden files
            name.endswith('_ch00.tif')         # must be bright-field
        )
        return ok

    def choose_and_load_folder(self):
        """
        Open folder selection dialog and load images.
        
        Enhanced with subfolder detection:
        - If selected folder contains only subfolders → prompt to choose one
        - If selected folder contains images → scan directly
        """
        # Let user choose starting directory
        if self.source_dir.exists() and self.source_dir.is_dir():
            initial_dir = str(self.source_dir)
        else:
            initial_dir = str(Path.cwd())
        
        print(f"\n🔍 Opening folder selection dialog...")
        print(f"   Initial directory: {initial_dir}")
        
        folder = filedialog.askdirectory(
            title="Select Folder containing _ch00.tif files (or parent of subfolders)",
            initialdir=initial_dir
        )
        
        # If user cancelled (empty string returned)
        if not folder:
            print("ℹ️  Folder selection cancelled by user\n")
            return
        
        # Convert string path to Path object
        selected_path = Path(folder)
        
        print(f"\n{'='*70}")
        print(f"📂 USER SELECTED FOLDER")
        print(f"{'='*70}")
        print(f"Folder name: {selected_path.name}")
        print(f"Absolute path: {selected_path.resolve()}")
        print(f"{'='*70}\n")
        
        # Check what's inside the selected folder
        try:
            items = list(selected_path.iterdir())
            subfolders = [item for item in items if item.is_dir() and not item.name.startswith('.')]
            tif_files = [item for item in items if item.is_file() and item.suffix.lower() in ['.tif', '.tiff']]
            
            print(f"📊 Contents analysis:")
            print(f"   Subfolders: {len(subfolders)}")
            print(f"   .tif files: {len(tif_files)}")
            
            # Case 1: Has .tif files directly → scan this folder
            if tif_files:
                print(f"\n✅ Found .tif files directly in selected folder")
                self.source_dir = selected_path
                self.scan_source_folder()
                return
            
            # Case 2: Has subfolders but no .tif files → let user pick a subfolder
            if subfolders:
                print(f"\n📂 Found {len(subfolders)} subfolders, no direct .tif files")
                print(f"   Showing subfolder selection dialog...")
                
                subfolder = self.choose_subfolder(selected_path, subfolders)
                if subfolder:
                    self.source_dir = subfolder
                    self.scan_source_folder()
                else:
                    print("ℹ️  Subfolder selection cancelled\n")
                return
            
            # Case 3: Empty folder or no valid content
            msg = f"Selected folder contains no .tif files or subfolders:\n{selected_path}"
            print(f"⚠️ {msg}")
            self.status_var.set("No images or subfolders found")
            messagebox.showwarning("Empty Folder", msg)
            
        except PermissionError as e:
            msg = f"Permission denied accessing folder:\n{selected_path}\n\n{str(e)}"
            print(f"❌ PERMISSION ERROR: {e}")
            self.status_var.set("Permission denied")
            messagebox.showerror("Permission Error", msg)
        except Exception as e:
            msg = f"Error analyzing folder:\n{selected_path}\n\n{str(e)}"
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Error analyzing folder")
            messagebox.showerror("Analysis Error", msg)

    def choose_subfolder(self, parent_path: Path, subfolders: List[Path]) -> Optional[Path]:
        """
        Show dialog to choose from available subfolders.
        
        Args:
            parent_path: Parent directory containing subfolders
            subfolders: List of subfolder Path objects
        
        Returns:
            Selected subfolder Path or None if cancelled
        """
        # Show subfolder picker dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select subfolder from source")
        dialog.geometry("300x400")
        dialog.transient(self.root)
        dialog.grab_set()

        selected_var = tk.StringVar()

        tk.Label(dialog, text="Select a subfolder:", font=("Arial", 10, "bold")).pack(pady=10)

        for subfolder in subfolders:
            tk.Radiobutton(dialog, text=subfolder, variable=selected_var, 
                        value=subfolder, font=("Arial", 9)).pack(anchor="w", padx=20)

        def on_ok():
            if selected_var.get():
                dialog.result = selected_var.get()
                dialog.destroy()
            else:
                messagebox.showwarning("No selection", "Please select a subfolder")

        def on_cancel():
            dialog.result = None
            dialog.destroy()

        tk.Button(dialog, text="OK", command=on_ok, width=10).pack(side="left", padx=20, pady=20)
        tk.Button(dialog, text="Cancel", command=on_cancel, width=10).pack(side="right", padx=20, pady=20)

        dialog.wait_window()

        if hasattr(dialog, 'result') and dialog.result:
            folder_path = os.path.join(folder_path, dialog.result)
            # NOW it will load the .tif files from this subfolder
            # The rest of your existing code continues here...
        else:
            return  # User cancelled

        
        # Header
        ttk.Label(dialog, text=f"Multiple subfolders found in:", 
                 font=("Segoe UI", 10, "bold")).pack(pady=(10, 0))
        ttk.Label(dialog, text=str(parent_path), 
                 font=("Segoe UI", 9), foreground="#666").pack(pady=(0, 10))
        ttk.Label(dialog, text="Select a subfolder to scan:", 
                 font=("Segoe UI", 10)).pack(pady=(0, 5))
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, 
                            font=("Consolas", 10), height=15)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Populate listbox
        sorted_folders = sorted(subfolders, key=lambda p: p.name.lower())
        for folder in sorted_folders:
            listbox.insert(tk.END, folder.name)
        
        listbox.bind('<Double-Button-1>', on_double_click)
        if sorted_folders:
            listbox.selection_set(0)
            listbox.focus_set()
        
        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Select", command=on_select, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=on_cancel, width=12).pack(side=tk.LEFT, padx=5)
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        dialog.wait_window()
        return selected_folder

    def scan_source_folder(self):
        """
        Scan **only the selected folder** (no sub-folders) for _ch00.tif files.
        """
        self.image_files = []
        
        print(f"\n{'='*60}")
        print(f"SCANNING INSIDE: {self.source_dir}")
        print(f"ABSOLUTE PATH: {self.source_dir.resolve()}")
        print(f"{'='*60}")
        
        # Verify the path we're actually scanning
        if not self.source_dir.exists():
            msg = f"Selected folder does not exist: {self.source_dir}"
            self.status_var.set(msg)
            print(f"❌ {msg}")
            messagebox.showerror("Folder Error", msg)
            return

        if not self.source_dir.is_dir():
            msg = f"Path is not a directory: {self.source_dir}"
            self.status_var.set(msg)
            print(f"❌ {msg}")
            messagebox.showerror("Folder Error", msg)
            return

        try:
            # List ALL files INSIDE the selected folder (direct children only)
            all_files = list(self.source_dir.iterdir())
            print(f"\n📁 Contents of {self.source_dir.name}/:")
            print(f"   Total items found: {len(all_files)}")
            
            # Show first few items for debugging
            print(f"\n   First 10 items:")
            for i, item in enumerate(all_files[:10]):
                item_type = "DIR" if item.is_dir() else "FILE"
                print(f"     {i+1}. [{item_type}] {item.name}")
            if len(all_files) > 10:
                print(f"     ... and {len(all_files) - 10} more items")
            
            # Filter for .tif files
            tif_files = [f for f in all_files if f.is_file() and f.suffix.lower() in ['.tif', '.tiff']]
            print(f"\n📄 Total .tif/.tiff files: {len(tif_files)}")
            
            # Show all .tif files found
            if tif_files:
                print("\n.TIF files found:")
                for f in sorted(tif_files):
                    print(f"  • {f.name}")
            
            # Filter for _ch00.tif files
            ch00_files = [f for f in tif_files if '_ch00' in f.name.lower()]
            print(f"\n🔍 Files containing '_ch00': {len(ch00_files)}")
            
            if ch00_files:
                print("_ch00 files found:")
                for f in sorted(ch00_files):
                    print(f"  • {f.name}")

            # Filter valid files
            valid = []
            rejected = []
            
            for f in ch00_files:
                if self.is_valid_image_file(f):
                    valid.append(f)
                    print(f"✅ ACCEPTED: {f.name}")
                else:
                    reasons = []
                    if f.name.startswith('._'): reasons.append("AppleDouble")
                    if f.name.startswith('.'): reasons.append("hidden")
                    if not f.name.endswith('_ch00.tif'): reasons.append("wrong suffix")
                    rejected.append((f, reasons))
                    print(f"❌ REJECTED: {f.name} - {', '.join(reasons)}")

            self.image_files = sorted(valid, key=lambda p: p.name)

            # Update UI
            self.update_navigation_buttons()
            
            print(f"\n{'='*60}")
            print(f"SCAN RESULTS: {len(self.image_files)} valid file(s)")
            print(f"{'='*60}\n")
            
            if self.image_files:
                self.status_var.set(
                    f"Found {len(self.image_files)} image(s) in: {self.source_dir.name}/"
                )
                # auto-load the first one
                self.load_image_by_index(0)
            else:
                # Detailed message when nothing matches
                msg_parts = []
                msg_parts.append(f"No valid _ch00.tif files found in:\n{self.source_dir}\n")
                
                if not tif_files:
                    msg_parts.append("❌ No .tif files found at all.")
                elif not ch00_files:
                    msg_parts.append(f"❌ Found {len(tif_files)} .tif file(s) but none contain '_ch00'.")
                elif rejected:
                    msg_parts.append(f"❌ Found {len(ch00_files)} _ch00.tif file(s) but all were rejected:")
                    for f, reasons in rejected[:5]:
                        msg_parts.append(f"  • {f.name}: {', '.join(reasons)}")
                
                msg_parts.append("\n✓ Valid files must:")
                msg_parts.append("  • End with '_ch00.tif'")
                msg_parts.append("  • Not start with '.' or '._'")
                
                msg = "\n".join(msg_parts)
                self.status_var.set("No valid images found")
                messagebox.showinfo("No Images Found", msg)
                
        except PermissionError as e:
            msg = f"Permission denied accessing folder:\n{self.source_dir}\n\n{str(e)}"
            print(f"❌ PERMISSION ERROR: {e}")
            self.status_var.set("Permission denied")
            messagebox.showerror("Permission Error", msg)
        except Exception as e:
            msg = f"Error scanning folder:\n{self.source_dir}\n\n{str(e)}"
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Error scanning folder")
            messagebox.showerror("Scan Error", msg)

    def get_fluorescence_path(self, bf_path: Path) -> Optional[Path]:
        """
        Given a bright-field image path (*_ch00.tif), find matching fluorescence (*_ch01.tif).
        
        Args:
            bf_path: Path to bright-field image (must end with _ch00.tif)
        
        Returns:
            Path to fluorescence image if exists, None otherwise
        """
        if not bf_path.name.endswith('_ch00.tif'):
            return None
        fluor_path = bf_path.parent / bf_path.name.replace('_ch00.tif', '_ch01.tif')
        return fluor_path if fluor_path.exists() and not fluor_path.name.startswith('._') else None

    def find_file_index(self, filepath: Path) -> int:
        """Find index of filepath in self.image_files list."""
        try:
            return self.image_files.index(filepath)
        except ValueError:
            abs_path = filepath.resolve()
            for i, f in enumerate(self.image_files):
                if f.resolve() == abs_path:
                    return i
            return -1

    def load_image_by_index(self, index: int):
        """Load image at given index in self.image_files."""
        if not (0 <= index < len(self.image_files)):
            return
        self.load_image_from_path(self.image_files[index])

    def load_previous_image(self):
        """Load previous image in the list."""
        if self.current_index > 0:
            self.load_image_by_index(self.current_index - 1)
        else:
            self.status_var.set("Already at first image")

    def load_next_image(self):
        """Load next image in the list."""
        if self.current_index < len(self.image_files) - 1:
            self.load_image_by_index(self.current_index + 1)
        else:
            self.status_var.set("Already at last image")

    def update_navigation_buttons(self):
        """Enable/disable Previous/Next buttons based on current position."""
        if not hasattr(self, 'prev_btn') or not hasattr(self, 'next_btn'):
            return
        has = len(self.image_files) > 0
        if has and self.current_index >= 0:
            self.prev_btn.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
            self.next_btn.config(state=tk.NORMAL if self.current_index < len(self.image_files) - 1 else tk.DISABLED)
            self.index_label.config(text=f"{self.current_index + 1}/{len(self.image_files)}")
        else:
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.index_label.config(text="-/-")

    # --------------------------------------------------------------------- #
    # Bacteria navigation
    # --------------------------------------------------------------------- #
    def navigate_to_bacteria(self, index: int):
        """Navigate to specific bacterium by index."""
        if not self.bacteria_stats or not (0 <= index < len(self.bacteria_stats)):
            return
        self.current_bacteria_index = index
        stat = self.bacteria_stats[index]
        M = cv2.moments(stat['contour'])
        if M["m00"] == 0:
            return
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        self.probe_point = (cx, cy)
        self.update_measurement_panel(cx, cy, int(self.original_image[cy, cx]))
        self.update_compact_statistics()
        self.update_preview()
        self.update_bacterium_status(index)

    def navigate_previous_bacteria(self):
        """Navigate to previous bacterium (wraps around)."""
        if self.bacteria_stats:
            self.navigate_to_bacteria((self.current_bacteria_index - 1) % len(self.bacteria_stats))

    def navigate_next_bacteria(self):
        """Navigate to next bacterium (wraps around)."""
        if self.bacteria_stats:
            self.navigate_to_bacteria((self.current_bacteria_index + 1) % len(self.bacteria_stats))

    def update_bacteria_navigation_buttons(self):
        """Enable/disable bacteria navigation buttons."""
        if not hasattr(self, 'bacteria_prev_btn'):
            return
        state = tk.NORMAL if self.bacteria_stats else tk.DISABLED
        self.bacteria_prev_btn.config(state=state)
        self.bacteria_next_btn.config(state=state)

    def update_bacterium_status(self, idx: int):
        """Update status bar with current bacterium info."""
        if 0 <= idx < len(self.bacteria_stats):
            s = self.bacteria_stats[idx]
            self.status_var.set(
                f"Viewing #{idx+1}/{len(self.bacteria_stats)} – BF: {s['bf_area']:.1f} px² | F/A: {s['fluor_per_area']:.3f}"
            )

    # --------------------------------------------------------------------- #
    # UI construction
    # --------------------------------------------------------------------- #
    def setup_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        left = ttk.Frame(main, width=420)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left.pack_propagate(False)

        # ---- Navigation -------------------------------------------------
        nav = ttk.LabelFrame(left, text=" Navigation ", padding=10)
        nav.pack(fill=tk.X, pady=(0, 10))

        row1 = ttk.Frame(nav)
        row1.pack(fill=tk.X, pady=(0, 8))

        load_btn = ttk.Button(row1, text="Load Folder", command=self.choose_and_load_folder)
        load_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ToolTip(load_btn, "Select a folder – all _ch00.tif images inside it will be scanned.")

        reset_btn = ttk.Button(row1, text="Reset", command=self.reset_to_defaults)
        reset_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 2))
        ToolTip(reset_btn, "Restore default parameters.")

        exit_btn = ttk.Button(row1, text="Exit", command=self.exit_application)
        exit_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        ToolTip(exit_btn, "Close the application.")

        row2 = ttk.Frame(nav)
        row2.pack(fill=tk.X)
        row2.grid_columnconfigure(0, weight=1)
        row2.grid_columnconfigure(2, weight=1)

        self.prev_btn = ttk.Button(row2, text="Previous", command=self.load_previous_image, state=tk.DISABLED)
        self.prev_btn.grid(row=0, column=0, sticky='ew')
        ToolTip(self.prev_btn, "Previous image in folder")

        self.index_label = ttk.Label(row2, text="-/-", anchor=tk.CENTER)
        self.index_label.grid(row=0, column=1, sticky='ew')

        self.next_btn = ttk.Button(row2, text="Next", command=self.load_next_image, state=tk.DISABLED)
        self.next_btn.grid(row=0, column=2, sticky='ew')
        ToolTip(self.next_btn, "Next image in folder")

        # ---- Measurement ------------------------------------------------
        self.measure_panel = CollapsibleFrame(left, title="Measurement on Click")
        self.measure_panel.pack(fill=tk.X, pady=(0, 10))
        mc = self.measure_panel.get_content_frame()
        for key, txt in [("pixel_coord", "Pixel: -, -"), ("pixel_value", "Value: -"),
                         ("inside_contour", "Inside Contour: -"), ("contour_area", "Contour Area: - px²")]:
            r = ttk.Frame(mc)
            r.pack(fill=tk.X, pady=2)
            lbl = ttk.Label(r, text=txt, font=("Consolas", 10), foreground="#2c3e50")
            lbl.pack(anchor=tk.W)
            self.measure_labels[key] = lbl

        # ---- Config ----------------------------------------------------
        self.config_panel = CollapsibleFrame(left, title="Configuration Parameters")
        self.config_panel.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        cfg = self.config_panel.get_content_frame()

        canvas = tk.Canvas(cfg, height=380)
        vsb = ttk.Scrollbar(cfg, orient="vertical", command=canvas.yview)
        scroll = ttk.Frame(canvas)

        scroll.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self.create_threshold_controls(scroll)
        self.create_clahe_controls(scroll)
        self.create_morphology_controls(scroll)
        self.create_watershed_controls(scroll)
        self.create_fluorescence_controls(scroll)
        self.create_label_controls(scroll)

        # ---- Compact stats + bacteria navigation -----------------------
        stats_panel = ttk.LabelFrame(left, text=" Quick Statistics ", padding=8)
        stats_panel.pack(fill=tk.X)

        for key, txt in [("bacteria_count", "Bacteria: -"), ("current_viewing", ""),
                         ("avg_bf_area", "Avg BF Area: - px²"), ("avg_fluor_area", "Avg Fluor/Area: -")]:
            r = ttk.Frame(stats_panel)
            r.pack(fill=tk.X, pady=1)
            lbl = ttk.Label(r, text=txt, font=("Consolas", 9), foreground="#34495e")
            lbl.pack(anchor=tk.W)
            self.compact_stats_labels[key] = lbl
        self.compact_stats_labels["current_viewing"].pack_forget()

        ttk.Separator(stats_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        nav_bac = ttk.Frame(stats_panel)
        nav_bac.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(nav_bac, text="Navigate:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        self.bacteria_prev_btn = ttk.Button(nav_bac, text="Previous", width=3,
                                            command=self.navigate_previous_bacteria, state=tk.DISABLED)
        self.bacteria_prev_btn.pack(side=tk.LEFT, padx=(0, 2))
        ToolTip(self.bacteria_prev_btn, "Previous bacterium")
        self.bacteria_next_btn = ttk.Button(nav_bac, text="Next", width=3,
                                            command=self.navigate_next_bacteria, state=tk.DISABLED)
        self.bacteria_next_btn.pack(side=tk.LEFT, padx=(2, 0))
        ToolTip(self.bacteria_next_btn, "Next bacterium")

        # ---- Image tabs -------------------------------------------------
        img_frame = ttk.Frame(main)
        img_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(img_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        tabs = [
            ("Original (BF)", "tab_original"), ("Fluorescence", "tab_fluorescence"),
            ("CLAHE Enhanced", "tab_enhanced"), ("Threshold", "tab_threshold"),
            ("Morphology", "tab_morphology"), ("Final Contours", "tab_contours"),
            ("BF+Fluor Overlay", "tab_overlay"), ("Statistics List", "tab_statistics")
        ]
        for name, attr in tabs:
            tab = ttk.Frame(self.notebook)
            setattr(self, attr, tab)
            self.notebook.add(tab, text=name)

        for attr in ["canvas_original", "canvas_fluorescence", "canvas_enhanced",
                     "canvas_threshold", "canvas_morphology", "canvas_contours", "canvas_overlay"]:
            canvas = tk.Canvas(getattr(self, attr.replace("canvas_", "tab_")), bg='#f8f9fa', highlightthickness=0)
            setattr(self, attr, canvas)
            canvas.pack(fill=tk.BOTH, expand=True)

        self.setup_statistics_table()
        self.canvas_original.bind("<Button-1>", self.on_canvas_click)
        self.canvas_original.bind("<Button-3>", self.clear_probe)

        self.status_var = tk.StringVar(value="Click 'Load Folder' to start")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN,
                  anchor=tk.W, padding=(5, 2)).pack(side=tk.BOTTOM, fill=tk.X)

    # --------------------------------------------------------------------- #
    # Compact statistics
    # --------------------------------------------------------------------- #
    def update_compact_statistics(self):
        """Update quick statistics panel."""
        if self.bacteria_stats:
            count = len(self.bacteria_stats)
            avg_bf = np.mean([s['bf_area'] for s in self.bacteria_stats])
            avg_fpa = np.mean([s['fluor_per_area'] for s in self.bacteria_stats])

            self.compact_stats_labels["bacteria_count"].config(text=f"Bacteria: {count}", foreground="#27ae60")
            if 0 <= self.current_bacteria_index < len(self.bacteria_stats):
                s = self.bacteria_stats[self.current_bacteria_index]
                txt = (f"Viewing #{self.current_bacteria_index+1} | BF: {s['bf_area']:.1f} px² | "
                       f"F/A: {s['fluor_per_area']:.3f}")
                self.compact_stats_labels["current_viewing"].config(text=txt, foreground="#e67e22")
                if not self.compact_stats_labels["current_viewing"].winfo_ismapped():
                    self.compact_stats_labels["current_viewing"].pack(fill=tk.X, pady=1)
            else:
                self.compact_stats_labels["current_viewing"].pack_forget()

            self.compact_stats_labels["avg_bf_area"].config(text=f"Avg BF Area: {avg_bf:.1f} px²")
            self.compact_stats_labels["avg_fluor_area"].config(text=f"Avg Fluor/Area: {avg_fpa:.3f}")
        else:
            for k in self.compact_stats_labels:
                self.compact_stats_labels[k].config(text=self.compact_stats_labels[k].cget("text").split(":")[0] + ": -",
                                                    foreground="#34495e")
            self.compact_stats_labels["current_viewing"].pack_forget()
        self.update_bacteria_navigation_buttons()

    # --------------------------------------------------------------------- #
    # Statistics table
    # --------------------------------------------------------------------- #
    def setup_statistics_table(self):
        """Setup the statistics table in the Statistics List tab."""
        frame = ttk.Frame(self.tab_statistics)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        cols = ("Index", "BF Area (px²)", "Fluor Mean", "Fluor Total", "Fluor/Area")
        self.stats_tree = ttk.Treeview(frame, columns=cols, show='headings', height=20)
        for c in cols:
            self.stats_tree.heading(c, text=c)
            self.stats_tree.column(c, width=120, anchor=tk.CENTER)
        self.stats_tree.column("Index", width=60)

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.stats_tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.stats_tree.xview)
        self.stats_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.stats_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.stats_summary = ttk.Label(frame, text="No data", font=("Segoe UI", 10, "bold"))
        self.stats_summary.grid(row=2, column=0, columnspan=2, pady=5)

    def calculate_bacteria_statistics(self, contours, bf_img, fluor_img=None):
        """Calculate statistics for each bacterium contour."""
        stats = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            mask = np.zeros(bf_img.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            f_mean = f_total = f_per = 0.0
            if fluor_img is not None:
                vals = fluor_img[mask == 255]
                if len(vals):
                    f_mean = float(np.mean(vals))
                    f_total = float(np.sum(vals))
                    f_per = f_total / area if area else 0.0
            stats.append({
                'bf_area': area, 'fluor_mean': f_mean, 'fluor_total': f_total,
                'fluor_per_area': f_per, 'contour': cnt
            })
        return stats

    def update_statistics_table(self):
        """Update the statistics table with current bacteria data."""
        for i in self.stats_tree.get_children():
            self.stats_tree.delete(i)
        if self.bacteria_stats:
            for i, s in enumerate(self.bacteria_stats, 1):
                self.stats_tree.insert('', 'end', values=(
                    i, f"{s['bf_area']:.1f}", f"{s['fluor_mean']:.2f}",
                    f"{s['fluor_total']:.1f}", f"{s['fluor_per_area']:.3f}"
                ))
            total = len(self.bacteria_stats)
            avg = np.mean([s['fluor_per_area'] for s in self.bacteria_stats])
            self.stats_summary.config(text=f"Total: {total} bacteria | Avg Fluor/Area: {avg:.3f}")
        else:
            self.stats_summary.config(text="No bacteria detected")

    # --------------------------------------------------------------------- #
    # Overlay
    # --------------------------------------------------------------------- #
    def create_overlay_image(self, bf, fluor, contours):
        """Create overlay image combining bright-field, fluorescence, and contours."""
        ov = cv2.cvtColor(bf, cv2.COLOR_GRAY2RGB)
        if fluor is not None:
            f = fluor.astype(np.float32)
            gamma = self.params['fluor_gamma'].get()
            bright = self.params['fluor_brightness'].get()
            f = f / (f.max() if f.max() else 1)
            f = np.power(f, gamma) * bright
            f = np.clip(f, 0, 1)
            f8 = (f * 255).astype(np.uint8)
            red = ov[:, :, 2].astype(np.float32)
            red = np.clip(red + f8.astype(np.float32), 0, 255)
            ov[:, :, 2] = red.astype(np.uint8)
        cv2.drawContours(ov, contours, -1, (255, 255, 0), 2)
        return ov

    # --------------------------------------------------------------------- #
    # Click handling
    # --------------------------------------------------------------------- #
    def on_canvas_click(self, event):
        """Handle mouse click on canvas for measurement and auto-tune."""
        if self.original_image is None:
            return
        cw, ch = self.canvas_original.winfo_width(), self.canvas_original.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        h, w = self.original_image.shape[:2]
        scale = min(cw / w, ch / h) * 0.95
        scale = min(scale, 1.0)
        ox = (cw - int(w * scale)) // 2
        oy = (ch - int(h * scale)) // 2
        ix = int((event.x - ox) / scale)
        iy = int((event.y - oy) / scale)
        if not (0 <= ix < w and 0 <= iy < h):
            return

        self.probe_point = (ix, iy)
        self.clear_probe()
        self.probe_canvas_ids = [
            self.canvas_original.create_line(event.x - 12, event.y, event.x + 12, event.y, fill="red", width=3),
            self.canvas_original.create_line(event.x, event.y - 12, event.x, event.y + 12, fill="red", width=3)
        ]
        if self.measure_panel.is_collapsed:
            self.measure_panel.expand()
        self.update_measurement_panel(ix, iy, int(self.original_image[iy, ix]))

        # Run segmentation first
        self.update_preview()

        # Auto-tune only on Ctrl+Click
        if event.state & 0x4:
            self.auto_tune_from_point(ix, iy, int(self.original_image[iy, ix]))

        # Highlight clicked bacterium
        if self.bacteria_stats:
            for idx, s in enumerate(self.bacteria_stats):
                if cv2.pointPolygonTest(s['contour'], (ix, iy), False) >= 0:
                    self.current_bacteria_index = idx
                    self.update_bacterium_status(idx)
                    break
            else:
                self.current_bacteria_index = -1
            self.update_bacteria_navigation_buttons()
            self.update_compact_statistics()

    def clear_probe(self, event=None):
        """Clear probe point and reset measurement panel."""
        for cid in self.probe_canvas_ids:
            self.canvas_original.delete(cid)
        self.probe_canvas_ids = []
        self.probe_point = None
        self.current_bacteria_index = -1
        self.update_bacteria_navigation_buttons()
        self.reset_measurement_panel()
        self.update_compact_statistics()
        if self.current_file:
            self.status_var.set(f"Ready – {self.current_file.name}")

    def update_measurement_panel(self, x, y, val):
        """Update measurement panel with clicked pixel information."""
        self.measure_labels["pixel_coord"].config(text=f"Pixel: ({x}, {y})")
        self.measure_labels["pixel_value"].config(text=f"Value: {val}")
        inside = area = 0
        if self.current_contours:
            for c in self.current_contours:
                if cv2.pointPolygonTest(c, (x, y), False) >= 0:
                    inside = True
                    area = cv2.contourArea(c)
                    break
        self.measure_labels["inside_contour"].config(
            text=f"Inside Contour: {'Yes' if inside else 'No'}",
            foreground="#27ae60" if inside else "#e74c3c")
        self.measure_labels["contour_area"].config(
            text=f"Contour Area: {int(area)} px²" if inside else "Contour Area: - px²")

    def reset_measurement_panel(self):
        """Reset measurement panel to default values."""
        defaults = {"pixel_coord": "Pixel: -, -", "pixel_value": "Value: -",
                    "inside_contour": "Inside Contour: -", "contour_area": "Contour Area: - px²"}
        for k, txt in defaults.items():
            self.measure_labels[k].config(text=txt, foreground="#2c3e50")

    # --------------------------------------------------------------------- #
    # Auto-tune
    # --------------------------------------------------------------------- #
    def auto_tune_from_point(self, x, y, val):
        """Auto-tune parameters based on clicked point (Ctrl+Click)."""
        if not self.current_contours:
            self.status_var.set("No contours – cannot auto-tune")
            return
        for c in self.current_contours:
            if cv2.pointPolygonTest(c, (x, y), False) >= 0:
                area = cv2.contourArea(c)
                if area < 10:
                    self.status_var.set("Contour too small")
                    continue
                self.params['use_otsu'].set(False)
                self.params['manual_threshold'].set(max(5, val - 18))
                self.params['min_area'].set(int(area * 0.75))
                self.params['watershed_dilate'].set(14)
                for k in ['use_otsu', 'manual_threshold', 'min_area', 'watershed_dilate']:
                    if k in self.entries:
                        self.entries[k].delete(0, tk.END)
                        self.entries[k].insert(0, str(self.params[k].get()))
                        self.update_progressbar(k)
                self.status_var.set(f"Auto-tuned: Th={val-18}, MinArea={int(area*0.75)}")
                return
        self.status_var.set("Click inside a contour to auto-tune")

    # --------------------------------------------------------------------- #
    # Parameter controls
    # --------------------------------------------------------------------- #
    def add_entry_with_progress(self, parent, label, tip, var, mn, mx, res=1.0, fl=False):
        """Add parameter entry field with progress bar."""
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=1)
        ttk.Label(f, text=label, width=20, anchor=tk.W).pack(side=tk.LEFT)
        ToolTip(f.winfo_children()[-1], tip)
        name = next(k for k, v in self.params.items() if v == var)
        e = ttk.Entry(f, width=7, justify=tk.RIGHT, font=("Consolas", 10))
        e.pack(side=tk.LEFT, padx=(2, 4))
        e.insert(0, str(var.get()))
        e.bind('<Return>', lambda ev: self.sync_entry(name, e, mn, mx, res, fl))
        e.bind('<FocusOut>', lambda ev: self.sync_entry(name, e, mn, mx, res, fl))
        ToolTip(e, tip)
        pb = ttk.Progressbar(f, orient=tk.HORIZONTAL, mode='determinate', length=150)
        pb.pack(side=tk.RIGHT, padx=(0, 2))
        ToolTip(pb, tip)
        self.entries[name] = e
        self.progressbars[name] = pb
        self.update_progressbar(name)
        return e

    def sync_entry(self, name, entry, mn, mx, res, fl):
        """Synchronize entry value with parameter variable."""
        try:
            v = (float if fl else int)(float(entry.get().strip()))
            v = max(mn, min(mx, v))
            if name in ('open_kernel', 'close_kernel'):
                v = int(v)
                if v % 2 == 0:
                    v = v - 1 if v - 1 >= mn else v + 1
            elif res != 1.0:
                v = round(v / res) * res
            self.params[name].set(v)
            entry.delete(0, tk.END)
            entry.insert(0, str(v))
            self.update_progressbar(name)
            self.update_preview()
        except ValueError:
            messagebox.showwarning("Invalid", f"Enter a number for {name.replace('_', ' ')}.")
            entry.delete(0, tk.END)
            entry.insert(0, str(self.params[name].get()))

    def update_progressbar(self, name):
        """Update progress bar to reflect current parameter value."""
        if name not in self.progressbars:
            return
        pb = self.progressbars[name]
        v = self.params[name].get()
        rng = {
            'manual_threshold': (0, 255), 'clahe_clip': (1, 10), 'clahe_tile': (4, 32),
            'open_kernel': (1, 15), 'close_kernel': (1, 15), 'open_iter': (1, 5),
            'close_iter': (1, 5), 'min_area': (10, 500), 'watershed_dilate': (1, 20),
            'fluor_brightness': (0.5, 5), 'fluor_gamma': (0.2, 2),
            'label_font_size': (10, 60), 'arrow_length': (20, 100),
            'label_offset': (5, 50), 'min_fluor_per_area': (0, 255),
        }
        mn, mx = rng[name]
        pb['value'] = (v - mn) / (mx - mn) * 100

    def create_threshold_controls(self, parent):
        """Create threshold parameter controls."""
        lf = ttk.LabelFrame(parent, text=" Threshold ", padding=8)
        lf.pack(fill=tk.X, pady=(0, 5))
        
        cb = ttk.Checkbutton(lf, text="Use Otsu", variable=self.params['use_otsu'],
                             command=self.update_preview)
        cb.pack(anchor=tk.W, pady=2)
        ToolTip(cb, "Automatically calculate threshold using Otsu's method")
        
        self.add_entry_with_progress(lf, "Manual Threshold:", 
                                     "Threshold value (0-255)", 
                                     self.params['manual_threshold'], 0, 255)

    def create_clahe_controls(self, parent):
        """Create CLAHE parameter controls."""
        lf = ttk.LabelFrame(parent, text=" CLAHE Enhancement ", padding=8)
        lf.pack(fill=tk.X, pady=(0, 5))
        
        cb = ttk.Checkbutton(lf, text="Enable CLAHE", variable=self.params['enable_clahe'],
                             command=self.update_preview)
        cb.pack(anchor=tk.W, pady=2)
        ToolTip(cb, "Apply Contrast Limited Adaptive Histogram Equalization")
        
        self.add_entry_with_progress(lf, "Clip Limit:", 
                                     "CLAHE clip limit (1-10)",
                                     self.params['clahe_clip'], 1, 10, 0.1, True)
        self.add_entry_with_progress(lf, "Tile Size:", 
                                     "CLAHE tile grid size (4-32)",
                                     self.params['clahe_tile'], 4, 32)

    def create_morphology_controls(self, parent):
        """Create morphology parameter controls."""
        lf = ttk.LabelFrame(parent, text=" Morphology ", padding=8)
        lf.pack(fill=tk.X, pady=(0, 5))
        
        self.add_entry_with_progress(lf, "Open Kernel:", 
                                     "Opening kernel size (odd, 1-15)",
                                     self.params['open_kernel'], 1, 15)
        self.add_entry_with_progress(lf, "Open Iterations:", 
                                     "Opening iterations (1-5)",
                                     self.params['open_iter'], 1, 5)
        self.add_entry_with_progress(lf, "Close Kernel:", 
                                     "Closing kernel size (odd, 1-15)",
                                     self.params['close_kernel'], 1, 15)
        self.add_entry_with_progress(lf, "Close Iterations:", 
                                     "Closing iterations (1-5)",
                                     self.params['close_iter'], 1, 5)

    def create_watershed_controls(self, parent):
        """Create watershed and filtering parameter controls."""
        lf = ttk.LabelFrame(parent, text=" Watershed & Filtering ", padding=8)
        lf.pack(fill=tk.X, pady=(0, 5))
        
        self.add_entry_with_progress(lf, "Watershed Dilate:", 
                                     "Watershed marker dilation (1-20)",
                                     self.params['watershed_dilate'], 1, 20)
        self.add_entry_with_progress(lf, "Min Area (px²):", 
                                     "Minimum bacteria area in pixels (10-500)",
                                     self.params['min_area'], 10, 500)

    def create_fluorescence_controls(self, parent):
        """Create fluorescence parameter controls."""
        lf = ttk.LabelFrame(parent, text=" Fluorescence ", padding=8)
        lf.pack(fill=tk.X, pady=(0, 5))
        
        self.add_entry_with_progress(lf, "Brightness:", 
                                     "Fluorescence brightness multiplier (0.5-5)",
                                     self.params['fluor_brightness'], 0.5, 5, 0.1, True)
        self.add_entry_with_progress(lf, "Gamma:", 
                                     "Fluorescence gamma correction (0.2-2)",
                                     self.params['fluor_gamma'], 0.2, 2, 0.1, True)
        self.add_entry_with_progress(lf, "Min Fluor/Area:", 
                                     "Minimum fluorescence per area ratio (0-255)",
                                     self.params['min_fluor_per_area'], 0, 255, 0.1, True)

    def create_label_controls(self, parent):
        """Create label display parameter controls."""
        lf = ttk.LabelFrame(parent, text=" Labels ", padding=8)
        lf.pack(fill=tk.X, pady=(0, 5))
        
        cb = ttk.Checkbutton(lf, text="Show Labels", variable=self.params['show_labels'],
                             command=self.update_preview)
        cb.pack(anchor=tk.W, pady=2)
        ToolTip(cb, "Display numbered labels for bacteria")
        
        self.add_entry_with_progress(lf, "Font Size:", 
                                     "Label font size (10-60)",
                                     self.params['label_font_size'], 10, 60)
        self.add_entry_with_progress(lf, "Arrow Length:", 
                                     "Arrow length in pixels (20-100)",
                                     self.params['arrow_length'], 20, 100)
        self.add_entry_with_progress(lf, "Label Offset:", 
                                     "Label offset from arrow (5-50)",
                                     self.params['label_offset'], 5, 50)

    # --------------------------------------------------------------------- #
    # Reset
    # --------------------------------------------------------------------- #
    def reset_to_defaults(self):
        """Reset all parameters to default values."""
        if messagebox.askyesno("Reset", "Restore all defaults?"):
            for k, v in self.default_params.items():
                self.params[k].set(v)
                if k in self.entries:
                    self.entries[k].delete(0, tk.END)
                    self.entries[k].insert(0, str(v))
                    self.update_progressbar(k)
            self.clear_probe()
            self.update_preview()
            self.status_var.set("Parameters reset")

    # --------------------------------------------------------------------- #
    # Load image
    # --------------------------------------------------------------------- #
    def load_image_from_path(self, path: Path):
        """Load bright-field image and matching fluorescence image if available."""
        print(f"\n📸 Loading image: {path}")
        
        bf = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bf is None:
            msg = f"Cannot read {path.name}"
            self.status_var.set(msg)
            print(f"❌ {msg}")
            messagebox.showerror("Load Error", f"Failed to load image:\n{path}")
            return
            
        if len(bf.shape) == 3:
            bf = cv2.cvtColor(bf, cv2.COLOR_BGR2GRAY)
        self.original_image = bf
        self.current_file = path
        self.current_index = self.find_file_index(path)

        fluor_path = self.get_fluorescence_path(path)
        self.fluorescence_image = None
        if fluor_path:
            print(f"  🔍 Looking for fluorescence: {fluor_path.name}")
            fluor = cv2.imread(str(fluor_path), cv2.IMREAD_UNCHANGED)
            if fluor is not None:
                if len(fluor.shape) == 3:
                    fluor = cv2.cvtColor(fluor, cv2.COLOR_BGR2GRAY)
                self.fluorescence_image = fluor
                print(f"  ✅ Fluorescence loaded")
            else:
                print(f"  ⚠️ Fluorescence file exists but couldn't be read")
        else:
            print(f"  ℹ️ No fluorescence file found")

        self.probe_point = None
        self.clear_probe()
        self.bacteria_stats = []
        self.current_bacteria_index = -1
        self.update_compact_statistics()
        self.update_bacteria_navigation_buttons()

        h, w = bf.shape[:2]
        fmsg = " + fluorescence" if self.fluorescence_image is not None else ""
        self.status_var.set(f"Loaded: {path.name}{fmsg} ({w}x{h})")
        self.root.title(f"{path.name} - Bacteria Segmentation Tuner")
        self.update_navigation_buttons()
        self.update_preview()
        print(f"✅ Image loaded successfully: {w}x{h}{fmsg}\n")

    # --------------------------------------------------------------------- #
    # SEGMENTATION
    # --------------------------------------------------------------------- #
    def segment_bacteria(self, gray_bf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Segment bacteria from bright-field image.
        
        Returns:
            Tuple of (enhanced, threshold, cleaned, bacteria_contours)
        """
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
    # Smart label positioning
    # --------------------------------------------------------------------- #
    def create_occupancy_map(self, img_shape, contours, margin=20):
        """Create a map showing occupied regions (bacteria + margins)."""
        h, w = img_shape
        occupancy = np.zeros((h, w), dtype=np.uint8)
        for contour in contours:
            cv2.drawContours(occupancy, [contour], -1, 255, -1)
            if margin > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin*2, margin*2))
                occupancy = cv2.dilate(occupancy, kernel, iterations=1)
        border_margin = margin
        occupancy[:border_margin, :] = 255
        occupancy[-border_margin:, :] = 255
        occupancy[:, :border_margin] = 255
        occupancy[:, -border_margin:] = 255
        return occupancy

    def calculate_direction_score(self, centroid, angle, arrow_len, label_size,
                                  occupancy_map, offset):
        """Calculate quality score for label placement in given direction."""
        cx, cy = centroid
        h, w = occupancy_map.shape
        label_w, label_h = label_size
        rad = np.deg2rad(angle)
        arrow_x = int(cx + arrow_len * np.cos(rad))
        arrow_y = int(cy - arrow_len * np.sin(rad))
        label_x = int(arrow_x + offset * np.cos(rad) - label_w / 2)
        label_y = int(arrow_y - offset * np.sin(rad) - label_h / 2)
        if (label_x < 0 or label_x + label_w >= w or
            label_y < 0 or label_y + label_h >= h):
            return float('inf')
        label_region = occupancy_map[label_y:label_y+label_h, label_x:label_x+label_w]
        occupied_pixels = np.sum(label_region > 0)
        total_pixels = label_region.size
        if total_pixels == 0:
            return float('inf')
        arrow_score = 0
        num_samples = 10
        for i in range(num_samples):
            t = i / num_samples
            sx = int(cx + t * arrow_len * np.cos(rad))
            sy = int(cy - t * arrow_len * np.sin(rad))
            if 0 <= sx < w and 0 <= sy < h and occupancy_map[sy, sx] > 0:
                arrow_score += 10
        score = (occupied_pixels / total_pixels) * 100 + arrow_score
        return score

    def find_best_label_position(self, centroid, img_shape, arrow_len, label_size,
                                 offset, occupancy_map):
        """Find best position for label by testing multiple angles."""
        cx, cy = centroid
        h, w = img_shape
        label_w, label_h = label_size
        angles = [i * 22.5 for i in range(16)]
        best_score = float('inf')
        best_pos = None
        for angle in angles:
            score = self.calculate_direction_score(
                centroid, angle, arrow_len, label_size, occupancy_map, offset
            )
            if score < best_score:
                best_score = score
                rad = np.deg2rad(angle)
                arrow_x = int(cx + arrow_len * np.cos(rad))
                arrow_y = int(cy - arrow_len * np.sin(rad))
                label_x = int(arrow_x + offset * np.cos(rad) - label_w / 2)
                label_y = int(arrow_y - offset * np.sin(rad) - label_h / 2)
                label_x = max(0, min(label_x, w - label_w))
                label_y = max(0, min(label_y, h - label_h))
                best_pos = (arrow_x, arrow_y, label_x, label_y, angle)
        return best_pos

    def get_label_font(self):
        """Get font for labels, trying multiple system font paths."""
        font_size = self.params['label_font_size'].get()
        font_paths = [
            "arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]
        for fp in font_paths:
            try:
                return ImageFont.truetype(fp, font_size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()

    def draw_labels_on_contours(self, img_bgr, contours):
        """Draw numbered labels on bacteria contours with smart positioning."""
        if not self.params['show_labels'].get() or not contours:
            return img_bgr
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        arrow_len = self.params['arrow_length'].get()
        label_offset = self.params['label_offset'].get()
        font = self.get_label_font()
        h, w = img_bgr.shape[:2]
        occupancy_map = self.create_occupancy_map((h, w), contours, margin=20)
        for stat_idx, stat in enumerate(self.bacteria_stats):
            contour = stat['contour']
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            label_text = str(stat_idx + 1)
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            result = self.find_best_label_position(
                (cx, cy), (h, w), arrow_len, (text_w, text_h), label_offset, occupancy_map
            )
            if result is None:
                continue
            arrow_x, arrow_y, label_x, label_y, angle = result
            is_selected = (self.current_bacteria_index == stat_idx)
            arrow_color = (255, 128, 0) if is_selected else (255, 255, 0)
            arrow_width = 3 if is_selected else 2
            draw.line([(cx, cy), (arrow_x, arrow_y)], fill=arrow_color, width=arrow_width)
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
            padding = 4
            bg_rect = [
                label_x - padding, label_y - padding,
                label_x + text_w + padding, label_y + text_h + padding
            ]
            draw.rectangle(bg_rect, fill=(0, 0, 0, 200))
            draw.text((label_x, label_y), label_text, font=font, fill=arrow_color)
            occupancy_map[label_y:label_y+text_h, label_x:label_x+text_w] = 255
        img_rgb_array = np.array(pil_img)
        return cv2.cvtColor(img_rgb_array, cv2.COLOR_RGB2BGR)

    # --------------------------------------------------------------------- #
    # Preview update
    # --------------------------------------------------------------------- #
    def update_preview(self):
        """Update all preview tabs with current segmentation results."""
        if self.original_image is None:
            return
        try:
            enhanced, thresh, cleaned, bacteria = self.segment_bacteria(self.original_image)

            all_stats = self.calculate_bacteria_statistics(bacteria, self.original_image, self.fluorescence_image)
            min_fpa = self.params['min_fluor_per_area'].get()

            if self.fluorescence_image is not None and min_fpa > 0:
                self.bacteria_stats = [s for s in all_stats if s['fluor_per_area'] >= min_fpa]
                bacteria = [s['contour'] for s in self.bacteria_stats]
            else:
                self.bacteria_stats = all_stats
            self.current_contours = bacteria

            base = f"Detected {len(bacteria)} bacteria"
            if self.fluorescence_image is not None and min_fpa > 0:
                base = f"Detected {len(self.bacteria_stats)}/{len(all_stats)} (min F/A {min_fpa:.1f})"
            if not self.probe_point and self.current_file:
                fstat = " (with fluorescence)" if self.fluorescence_image is not None else ""
                self.status_var.set(f"{base} | {self.current_file.name}{fstat}")

            self.update_statistics_table()
            self.update_compact_statistics()
            for k in self.progressbars:
                self.update_progressbar(k)

            self.display_image(self.original_image, self.canvas_original)
            if self.fluorescence_image is not None:
                self.display_fluorescence_image(self.fluorescence_image, self.canvas_fluorescence)
            else:
                self.canvas_fluorescence.delete("all")

            self.display_image(enhanced, self.canvas_enhanced)
            self.display_image(thresh, self.canvas_threshold)
            self.display_image(cleaned, self.canvas_morphology)

            cnt_img = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(cnt_img, bacteria, -1, (0, 255, 0), 2)
            if self.probe_point:
                px, py = self.probe_point
                for c in bacteria:
                    if cv2.pointPolygonTest(c, (px, py), False) >= 0:
                        cv2.drawContours(cnt_img, [c], -1, (0, 0, 255), 3)
                        break
            cnt_img = self.draw_labels_on_contours(cnt_img, bacteria)
            self.display_image(cnt_img, self.canvas_contours)

            ov = self.create_overlay_image(self.original_image, self.fluorescence_image, bacteria)
            self.display_image(ov, self.canvas_overlay)

        except Exception as e:
            import traceback
            self.status_var.set(f"Error: {e}")
            traceback.print_exc()

    # --------------------------------------------------------------------- #
    # Display helpers
    # --------------------------------------------------------------------- #
    def display_fluorescence_image(self, img, canvas):
        """Display fluorescence image with brightness/gamma adjustments."""
        f = img.astype(np.float32)
        gamma = self.params['fluor_gamma'].get()
        bright = self.params['fluor_brightness'].get()
        f = f / (f.max() if f.max() else 1)
        f = np.power(f, gamma) * bright
        f = np.clip(f, 0, 1)
        f8 = (f * 255).astype(np.uint8)
        red = np.zeros((f8.shape[0], f8.shape[1], 3), dtype=np.uint8)
        red[:, :, 2] = f8
        rgb = cv2.cvtColor(red, cv2.COLOR_BGR2RGB)
        self._show_resized(rgb, canvas)

    def display_image(self, img, canvas):
        """Display image on canvas (converts to RGB if needed)."""
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._show_resized(img, canvas)

    def _show_resized(self, img, canvas):
        """Resize and display image on canvas with proper scaling."""
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        h, w = img.shape[:2]
        scale = min(cw / w, ch / h) * 0.95
        if scale < 1:
            nw, nh = int(w * scale), int(h * scale)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor=tk.CENTER)
        canvas.image = photo

        if canvas == self.canvas_original and self.probe_point:
            h0, w0 = self.original_image.shape[:2]
            s = min(cw / w0, ch / h0) * 0.95
            s = min(s, 1.0)
            ox = (cw - int(w0 * s)) // 2
            oy = (ch - int(h0 * s)) // 2
            ix, iy = self.probe_point
            cx = int(ix * s) + ox
            cy = int(iy * s) + oy
            self.probe_canvas_ids = [
                canvas.create_line(cx - 12, cy, cx + 12, cy, fill="red", width=3),
                canvas.create_line(cx, cy - 12, cx, cy + 12, fill="red", width=3)
            ]

    # --------------------------------------------------------------------- #
    # Exit cleanup
    # --------------------------------------------------------------------- #
    def exit_application(self):
        """Clean up resources and exit application."""
        try:
            self.clear_probe()
            for c in [self.canvas_original, self.canvas_fluorescence, self.canvas_enhanced,
                      self.canvas_threshold, self.canvas_morphology,
                      self.canvas_contours, self.canvas_overlay]:
                c.delete("all")
                if hasattr(c, 'image'):
                    delattr(c, 'image')
            self.original_image = self.fluorescence_image = None
            self.current_contours.clear()
            self.bacteria_stats.clear()
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            print("Cleanup error:", e)
            try:
                self.root.destroy()
            except:
                pass


if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationViewer(root)
    root.mainloop()