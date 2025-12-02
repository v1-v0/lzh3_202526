#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive Bacteria Segmentation Parameter Tuner with Metadata Integration
- Enhanced folder selection with subfolder picker (IMPROVED for Linux)
- Smart label positioning
- Vertical scrollbar for entire left panel (OPTIMIZED for macOS)
- Numbered tabs with keyboard shortcuts
- Dark mode support (inline toggle button)
- Metadata integration for quantitative, reproducible analysis
- Physical unit conversions (µm, µm²)
- Reproducible display settings from acquisition metadata
- GRACEFUL HANDLING of folders without metadata
- Scale bar with physical units
- Histogram visualization in statistics tab
"""

import os
import cv2
import numpy as np
from pathlib import Path
from scipy import ndimage
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
from typing import cast, List, Tuple, Dict, Optional, Sequence
from cv2.typing import MatLike
import platform
import json
from datetime import datetime

# ✨ FIXED: Try importing matplotlib, provide fallback if not available
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not installed - histograms will be disabled")
    print("   Install with: pip install matplotlib")


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

        # Declare tab attributes for type checker
        self.tab_original: ttk.Frame
        self.tab_enhanced: ttk.Frame
        self.tab_threshold: ttk.Frame
        self.tab_morphology: ttk.Frame
        self.tab_contours: ttk.Frame
        self.tab_overlay: ttk.Frame
        self.tab_fluorescence: ttk.Frame
        self.tab_statistics: ttk.Frame

        # Declare canvas attributes for type checker
        self.canvas_original: tk.Canvas
        self.canvas_enhanced: tk.Canvas
        self.canvas_threshold: tk.Canvas
        self.canvas_morphology: tk.Canvas
        self.canvas_contours: tk.Canvas
        self.canvas_overlay: tk.Canvas
        self.canvas_fluorescence: tk.Canvas

        # Image and state variables ----------------------------------------
        self.original_image: Optional[np.ndarray] = None
        self.fluorescence_image: Optional[np.ndarray] = None
        self.current_contours: List[np.ndarray] = []
        self.probe_point: Optional[Tuple[int, int]] = None
        self.probe_canvas_ids: List[int] = []
        self.current_file: Optional[Path] = None
        self.current_metadata: Optional[Dict] = None
        self.has_metadata: bool = False

        self.bacteria_stats: List[Dict] = []
        self.current_bacteria_index: int = -1

        # Start with current directory instead of hardcoded "source/"
        self.source_dir = Path.cwd()
        self.image_files: List[Path] = []
        self.current_index: int = -1

        # Create status_var BEFORE setup_ui()
        self.status_var = tk.StringVar(value="Click 'Load Folder' to start")

        # Performance optimization flags
        self._scroll_update_pending = False
        self._preview_update_pending = False

        # Default parameters -------------------------------------------------
        self.default_params = {
            'use_otsu': False, 'manual_threshold': 110, 'enable_clahe': True,
            'clahe_clip': 5.0, 'clahe_tile': 32, 'open_kernel': 3, 'close_kernel': 5,
            'open_iter': 3, 'close_iter': 2, 'min_area': 67, 'watershed_dilate': 15,
            'fluor_brightness': 2.0, 'fluor_gamma': 0.5, 'show_labels': True,
            'label_font_size': 20, 'arrow_length': 60, 'label_offset': 15,
            'min_fluor_per_area': 10, 'show_scale_bar': True,
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

        # Statistics sorting state (default: Fluor/Area descending)
        self.stats_sort: Tuple[str, bool] = ("fluor_per_area", True)
        self.stats_col_map: Dict[str, str] = {}
        self.stats_columns: Tuple[str, ...] = tuple()

        self.setup_ui()
        self.setup_keyboard_shortcuts()
        self.root.bind("<Configure>", lambda e: self.root.after_idle(self.update_preview))

    # --------------------------------------------------------------------- #
    # Metadata handling (IMPROVED: graceful fallback)
    # --------------------------------------------------------------------- #
    def load_metadata_for_image(self, image_path: Path) -> Optional[Dict]:
        """Load metadata JSON for current image (returns None if not found)."""
        metadata_dir = Path.cwd() / "metadata_json"
        if not metadata_dir.exists():
            print(f"  ℹ️  No metadata_json folder found")
            return None
        
        sample_name = image_path.stem.replace('_ch00', '').replace('_ch01', '')
        metadata_file = metadata_dir / f"{sample_name}.json"
        if not metadata_file.exists():
            print(f"  ℹ️  No metadata file: {metadata_file.name}")
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"  ✅ Metadata loaded: {metadata_file.name}")
            return metadata
        except Exception as e:
            print(f"  ⚠️  Failed to load metadata: {e}")
            return None

    def apply_metadata_display_settings(self):
        """Apply brightness/gamma settings from metadata to fluorescence channel."""
        if not self.current_metadata or 'channels' not in self.current_metadata:
            print(f"  ℹ️  No channel metadata - keeping manual display settings")
            return
        
        channels = self.current_metadata['channels']
        if 'Red' in channels:
            red_channel = channels['Red']
            black_val = red_channel['normalized']['BlackValue']
            white_val = red_channel['normalized']['WhiteValue']
            gamma_val = red_channel.get('GammaValue', 1.0)
            
            brightness = 1.0 / white_val if white_val > 0 else 2.0
            brightness = max(0.5, min(5.0, brightness))
            gamma = max(0.2, min(2.0, gamma_val))
            
            self.params['fluor_brightness'].set(brightness)
            self.params['fluor_gamma'].set(gamma)
            
            if 'fluor_brightness' in self.entries:
                self.entries['fluor_brightness'].delete(0, tk.END)
                self.entries['fluor_brightness'].insert(0, f"{brightness:.2f}")
                self.update_progressbar('fluor_brightness')
            
            if 'fluor_gamma' in self.entries:
                self.entries['fluor_gamma'].delete(0, tk.END)
                self.entries['fluor_gamma'].insert(0, f"{gamma:.2f}")
                self.update_progressbar('fluor_gamma')
            
            print(f"  🎨 Applied metadata display settings: Brightness={brightness:.2f}, Gamma={gamma:.2f}")
        else:
            print(f"  ℹ️  No 'Red' channel in metadata - keeping manual settings")

    def get_pixel_size(self) -> float:
        """Get pixel size in µm from metadata or default (0.1289 µm)."""
        if self.current_metadata:
            pixel_size = self.current_metadata.get('pixel_size_um', 0.1289)
            return pixel_size
        return 0.1289

    # --------------------------------------------------------------------- #
    # Folder / file handling (IMPROVED for Linux)
    # --------------------------------------------------------------------- #
    def is_valid_image_file(self, filepath: Path) -> bool:
        """Return True only for real bright-field files."""
        name = filepath.name
        ok = (
            not name.startswith('._') and
            not name.startswith('.') and
            name.endswith('_ch00.tif')
        )
        return ok

    def choose_and_load_folder(self):
        """Open folder selection dialog and load images."""
        source_subfolder = Path.cwd() / "source"
        if source_subfolder.exists() and source_subfolder.is_dir():
            initial_dir = str(source_subfolder)
        else:
            initial_dir = str(Path.cwd())
        
        print(f"\n🔍 Opening folder selection dialog...")
        print(f"   Initial directory: {initial_dir}")
        
        # Platform-specific folder dialog with better Linux support
        system = platform.system()
        
        if system == "Linux":
            folder = self._linux_folder_dialog(initial_dir)
        else:
            folder = filedialog.askdirectory(
                title="Select Folder containing _ch00.tif files (or parent of subfolders)",
                initialdir=initial_dir
            )
        
        if not folder:
            print("ℹ️  Folder selection cancelled by user\n")
            return
        
        selected_path = Path(folder)
        
        print(f"\n{'='*70}")
        print(f"📂 USER SELECTED FOLDER")
        print(f"{'='*70}")
        print(f"Folder name: {selected_path.name}")
        print(f"Absolute path: {selected_path.resolve()}")
        print(f"{'='*70}\n")
        
        try:
            items = list(selected_path.iterdir())
            subfolders = [item for item in items if item.is_dir() and not item.name.startswith('.')]
            tif_files = [item for item in items if item.is_file() and item.suffix.lower() in ['.tif', '.tiff']]
            
            print(f"📊 Contents analysis:")
            print(f"   Subfolders: {len(subfolders)}")
            print(f"   .tif files: {len(tif_files)}")
            
            if tif_files:
                print(f"\n✅ Found .tif files directly in selected folder")
                self.source_dir = selected_path
                self.scan_source_folder()
                return
            
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
            import traceback
            msg = f"Error analyzing folder:\n{selected_path}\n\n{str(e)}"
            print(f"❌ ERROR: {e}")
            traceback.print_exc()
            self.status_var.set("Error analyzing folder")
            messagebox.showerror("Analysis Error", msg)

    def _linux_folder_dialog(self, initial_dir: str) -> str:
        """Custom folder selection dialog optimized for Linux."""
        selected_folder = tk.StringVar(value="")
        current_path = Path(initial_dir)  # Move initialization here
        
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
            nonlocal current_path  # Declare before use
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
            nonlocal current_path  # Declare before use
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

    def choose_subfolder(self, parent_path: Path, subfolders: List[Path]) -> Optional[Path]:
        """Show dialog to choose from available subfolders."""
        selected_folder = None
        
        def on_select():
            nonlocal selected_folder
            selection = listbox.curselection()
            if selection:
                idx = selection[0]
                selected_folder = sorted_folders[idx]
                dialog.destroy()
            else:
                messagebox.showwarning("No Selection", "Please select a subfolder")
        
        def on_cancel():
            nonlocal selected_folder
            selected_folder = None
            dialog.destroy()
        
        def on_double_click(event):
            nonlocal selected_folder
            selection = listbox.curselection()
            if selection:
                idx = selection[0]
                selected_folder = sorted_folders[idx]
                dialog.destroy()
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Select subfolder from source")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text=f"Multiple subfolders found in:", 
                font=("Segoe UI", 10, "bold")).pack(pady=(10, 0))
        ttk.Label(dialog, text=str(parent_path), 
                font=("Segoe UI", 9), foreground="#666").pack(pady=(0, 10))
        ttk.Label(dialog, text="Select a subfolder to scan:", 
                font=("Segoe UI", 10)).pack(pady=(0, 5))
        
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, 
                            font=("Consolas", 10), height=15)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        sorted_folders = sorted(subfolders, key=lambda p: p.name.lower())
        for folder in sorted_folders:
            listbox.insert(tk.END, folder.name)
        
        listbox.bind('<Double-Button-1>', on_double_click)
        if sorted_folders:
            listbox.selection_set(0)
            listbox.focus_set()
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Select", command=on_select, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=on_cancel, width=12).pack(side=tk.LEFT, padx=5)
        
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        dialog.wait_window()
        return selected_folder

    def scan_source_folder(self):
        """Scan only the selected folder for _ch00.tif files."""
        self.image_files = []
        
        print(f"\n{'='*60}")
        print(f"SCANNING INSIDE: {self.source_dir}")
        print(f"ABSOLUTE PATH: {self.source_dir.resolve()}")
        print(f"{'='*60}")
        
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
            all_files = list(self.source_dir.iterdir())
            print(f"\n📁 Contents of {self.source_dir.name}/:")
            print(f"   Total items found: {len(all_files)}")
            
            print(f"\n   First 10 items:")
            for i, item in enumerate(all_files[:10]):
                item_type = "DIR" if item.is_dir() else "FILE"
                print(f"     {i+1}. [{item_type}] {item.name}")
            if len(all_files) > 10:
                print(f"     ... and {len(all_files) - 10} more items")
            
            tif_files = [f for f in all_files if f.is_file() and f.suffix.lower() in ['.tif', '.tiff']]
            print(f"\n📄 Total .tif/.tiff files: {len(tif_files)}")
            
            if tif_files:
                print("\n.TIF files found:")
                for f in sorted(tif_files):
                    print(f"  • {f.name}")
            
            ch00_files = [f for f in tif_files if '_ch00' in f.name.lower()]
            print(f"\n🔍 Files containing '_ch00': {len(ch00_files)}")
            
            if ch00_files:
                print("_ch00 files found:")
                for f in sorted(ch00_files):
                    print(f"  • {f.name}")

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

            self.update_navigation_buttons()
            
            print(f"\n{'='*60}")
            print(f"SCAN RESULTS: {len(self.image_files)} valid file(s)")
            print(f"{'='*60}\n")
            
            if self.image_files:
                self.status_var.set(
                    f"Found {len(self.image_files)} image(s) in: {self.source_dir.name}/"
                )
                self.load_image_by_index(0)
            else:
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
        """Given a bright-field image path (*_ch00.tif), find matching fluorescence (*_ch01.tif)."""
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
    # Keyboard shortcuts
    # --------------------------------------------------------------------- #
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for tab navigation."""
        self.root.bind("<comma>", lambda e: self.move_to_previous_tab())
        self.root.bind("<period>", lambda e: self.move_to_next_tab())
        self.root.bind("l", lambda e: self.choose_and_load_folder())
        self.root.bind("d", lambda e: self.toggle_dark_mode())
        self.root.bind("<Escape>", lambda e: self.exit_with_confirmation())
        self.root.bind("<Left>", lambda e: self.load_previous_image())
        self.root.bind("<Right>", lambda e: self.load_next_image())

    def move_to_previous_tab(self):
        """Switch to the previous tab."""
        current_tab = self.notebook.index(self.notebook.select())
        total_tabs = self.notebook.index("end")
        previous_tab = (current_tab - 1) % total_tabs
        self.notebook.select(previous_tab)

    def move_to_next_tab(self):
        """Switch to the next tab."""
        current_tab = self.notebook.index(self.notebook.select())
        total_tabs = self.notebook.index("end")
        next_tab = (current_tab + 1) % total_tabs
        self.notebook.select(next_tab)
    
    def switch_to_tab(self, index: int):
        """Switch to tab at given index."""
        if 0 <= index < self.notebook.index('end'):
            self.notebook.select(index)

    # --------------------------------------------------------------------- #
    # Dark Mode / Light Mode
    # --------------------------------------------------------------------- #
    def apply_dark_mode(self):
        """Apply dark mode colors to the application."""
        dark_bg = "#2b2b2b"
        dark_fg = "#e0e0e0"
        dark_frame = "#3c3c3c"
        dark_button_bg = "#4a4a4a"
        dark_button_fg = "#ffffff"
        dark_canvas_bg = "#1e1e1e"
        
        self.root.configure(bg=dark_bg)
        
        style = ttk.Style()
        style.theme_use('default')
        
        style.configure("TNotebook", background=dark_bg, borderwidth=0)
        style.configure("TNotebook.Tab", 
            background=dark_frame, 
            foreground=dark_fg,
            padding=[10, 2])
        style.map("TNotebook.Tab", 
            background=[("selected", dark_button_bg)],
            foreground=[("selected", "#ffffff")])
        
        style.configure("TFrame", background=dark_bg)
        style.configure("TLabelframe", background=dark_bg, foreground=dark_fg)
        style.configure("TLabelframe.Label", background=dark_bg, foreground=dark_fg)
        style.configure("TLabel", background=dark_bg, foreground=dark_fg)
        
        style.configure("TButton", 
                       background=dark_button_bg, 
                       foreground=dark_button_fg,
                       borderwidth=1)
        style.map("TButton",
                 background=[("active", "#5a5a5a")],
                 foreground=[("active", "#ffffff")])
        
        style.configure("TCheckbutton", background=dark_bg, foreground=dark_fg)
        
        style.configure("TEntry", 
                       fieldbackground=dark_frame,
                       foreground=dark_fg,
                       insertcolor=dark_fg)
        
        style.configure("Vertical.TScrollbar", background=dark_button_bg)
        style.configure("Horizontal.TScrollbar", background=dark_button_bg)
        style.configure("TProgressbar", background="#4CAF50", troughcolor=dark_frame)
        
        for canvas_attr in ["canvas_original", "canvas_fluorescence", "canvas_enhanced",
                           "canvas_threshold", "canvas_morphology", "canvas_contours", 
                           "canvas_overlay", "left_canvas"]:
            if hasattr(self, canvas_attr):
                getattr(self, canvas_attr).configure(bg=dark_canvas_bg)
        
        for label in self.measure_labels.values():
            label.configure(foreground=dark_fg)
        
        style.configure("Treeview", 
                       background=dark_frame,
                       foreground=dark_fg,
                       fieldbackground=dark_frame)
        style.configure("Treeview.Heading", background=dark_button_bg, foreground=dark_fg)
        style.map("Treeview", 
                 background=[("selected", dark_button_bg)],
                 foreground=[("selected", "#ffffff")])
        
        self.dark_mode_btn.config(text="🌙 Dark")
        self.status_var.set("Dark mode activated")

    def apply_light_mode(self):
        """Apply light mode colors to the application."""
        light_bg = "#ffffff"
        light_fg = "#000000"
        light_frame = "#f0f0f0"
        light_button_bg = "#e1e1e1"
        light_canvas_bg = "#f8f9fa"
        
        self.root.configure(bg=light_bg)
        
        style = ttk.Style()
        style.theme_use('default')
        
        style.configure("TNotebook", background=light_bg, borderwidth=0)
        style.configure("TNotebook.Tab", 
                       background=light_frame, 
                       foreground=light_fg,
                       padding=[10, 2])
        style.map("TNotebook.Tab", 
                 background=[("selected", light_button_bg)],
                 foreground=[("selected", light_fg)])
        
        style.configure("TFrame", background=light_bg)
        style.configure("TLabelframe", background=light_bg, foreground=light_fg)
        style.configure("TLabelframe.Label", background=light_bg, foreground=light_fg)
        style.configure("TLabel", background=light_bg, foreground=light_fg)
        
        style.configure("TButton", 
                       background=light_button_bg, 
                       foreground=light_fg,
                       borderwidth=1)
        style.map("TButton",
                 background=[("active", "#d0d0d0")])
        
        style.configure("TCheckbutton", background=light_bg, foreground=light_fg)
        
        style.configure("TEntry", 
                       fieldbackground="white",
                       foreground=light_fg,
                       insertcolor=light_fg)
        
        style.configure("Vertical.TScrollbar", background=light_button_bg)
        style.configure("Horizontal.TScrollbar", background=light_button_bg)
        style.configure("TProgressbar", background="#4CAF50", troughcolor=light_frame)
        
        for canvas_attr in ["canvas_original", "canvas_fluorescence", "canvas_enhanced",
                           "canvas_threshold", "canvas_morphology", "canvas_contours", 
                           "canvas_overlay", "left_canvas"]:
            if hasattr(self, canvas_attr):
                getattr(self, canvas_attr).configure(bg=light_canvas_bg)
        
        for label in self.measure_labels.values():
            label.configure(foreground="#2c3e50")
        
        style.configure("Treeview", 
                       background="white",
                       foreground=light_fg,
                       fieldbackground="white")
        style.configure("Treeview.Heading", background=light_button_bg, foreground=light_fg)
        style.map("Treeview", 
                 background=[("selected", light_button_bg)])
        
        self.dark_mode_btn.config(text="☀️ Light")
        self.status_var.set("Light mode activated")

    def toggle_dark_mode(self):
        """Toggle between dark and light mode."""
        self.dark_mode_var.set(not self.dark_mode_var.get())
        if self.dark_mode_var.get():
            self.apply_dark_mode()
        else:
            self.apply_light_mode()

    # --------------------------------------------------------------------- #
    # UI construction (OPTIMIZED)
    # --------------------------------------------------------------------- #
    def setup_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # LEFT PANEL WITH SCROLLBAR
        left_container = ttk.Frame(main, width=420)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 15))
        left_container.pack_propagate(False)

        self.left_canvas = tk.Canvas(left_container, highlightthickness=0)
        self.left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=self.left_canvas.yview)
        self.scrollable_left = ttk.Frame(self.left_canvas)

        def update_scroll_region(event=None):
            if not self._scroll_update_pending:
                self._scroll_update_pending = True
                self.root.after(100, self._update_scroll_region_delayed)
        
        self.scrollable_left.bind("<Configure>", update_scroll_region)

        self.canvas_frame = self.left_canvas.create_window((0, 0), window=self.scrollable_left, anchor="nw")
        self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)

        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.left_canvas.bind('<Configure>', self._on_canvas_configure)
        self._bind_mousewheel()

        left = self.scrollable_left

        # ---- Navigation -------------------------------------------------
        nav = ttk.LabelFrame(left, text=" Navigation ", padding=10)
        nav.pack(fill=tk.X, pady=(0, 10), padx=5)

        row1 = ttk.Frame(nav)
        row1.pack(fill=tk.X, pady=(0, 8))
        row1.grid_columnconfigure(0, weight=1)
        row1.grid_columnconfigure(1, weight=1)
        row1.grid_columnconfigure(2, weight=1)
        row1.grid_columnconfigure(3, weight=1)

        load_btn = ttk.Button(row1, text="Load Folder", command=self.choose_and_load_folder)
        load_btn.grid(row=0, column=0, sticky='ew', padx=(0, 2))
        ToolTip(load_btn, "Select a folder – all _ch00.tif images inside it will be scanned. (Shortcut: L)")

        reset_btn = ttk.Button(row1, text="Reset", command=self.reset_to_defaults)
        reset_btn.grid(row=0, column=1, sticky='ew', padx=(2, 2))
        ToolTip(reset_btn, "Restore default parameters.")

        self.dark_mode_var = tk.BooleanVar(value=True)
        self.dark_mode_btn = ttk.Button(row1, text="☀️ Light", command=self.toggle_dark_mode)
        self.dark_mode_btn.grid(row=0, column=2, sticky='ew', padx=(2, 2))
        ToolTip(self.dark_mode_btn, "Toggle between light and dark mode (Shortcut: D)")

        exit_btn = ttk.Button(row1, text="Exit", command=self.exit_application)
        exit_btn.grid(row=0, column=3, sticky='ew', padx=(2, 0))
        ToolTip(exit_btn, "Close the application. (Shortcut: Esc)")

        row2 = ttk.Frame(nav)
        row2.pack(fill=tk.X)
        row2.grid_columnconfigure(0, weight=1)
        row2.grid_columnconfigure(2, weight=1)

        self.prev_btn = ttk.Button(row2, text="Previous (←)", command=self.load_previous_image, state=tk.DISABLED)
        self.prev_btn.grid(row=0, column=0, sticky='ew')
        ToolTip(self.prev_btn, "Previous image in folder (or press Left arrow)")

        self.index_label = ttk.Label(row2, text="-/-", anchor=tk.CENTER)
        self.index_label.grid(row=0, column=1, sticky='ew')

        self.next_btn = ttk.Button(row2, text="Next (→)", command=self.load_next_image, state=tk.DISABLED)
        self.next_btn.grid(row=0, column=2, sticky='ew')
        ToolTip(self.next_btn, "Next image in folder (or press Right arrow)")

        # ---- Metadata Info Panel ----------------------------------------
        self.metadata_panel = CollapsibleFrame(left, title="Image Metadata")
        self.metadata_panel.pack(fill=tk.X, pady=(0, 10), padx=5)
        mc = self.metadata_panel.get_content_frame()
        
        metadata_keys = [
            ("sample_name", "Sample: -"),
            ("pixel_size", "Pixel Size: -"),
            ("objective", "Objective: -"),
            ("acquired", "Acquired: -"),
            ("exposure_bf", "BF Exposure: -"),
            ("exposure_fluor", "Fluor Exposure: -")
        ]
        
        for key, txt in metadata_keys:
            r = ttk.Frame(mc)
            r.pack(fill=tk.X, pady=2)
            lbl = ttk.Label(r, text=txt, font=("Consolas", 9), foreground="#2c3e50")
            lbl.pack(anchor=tk.W)
            self.measure_labels[key] = lbl

        # ---- Measurement ------------------------------------------------
        self.measure_panel = CollapsibleFrame(left, title="Measurement on Click")
        self.measure_panel.pack(fill=tk.X, pady=(0, 10), padx=5)
        mc = self.measure_panel.get_content_frame()
        for key, txt in [("pixel_coord", "Pixel: -, -"), ("pixel_value", "Value: -"),
                         ("inside_contour", "Inside Contour: -"), 
                         ("contour_area_px", "Contour Area: - px²"),
                         ("contour_area_um", "Contour Area: - µm²")]:
            r = ttk.Frame(mc)
            r.pack(fill=tk.X, pady=2)
            lbl = ttk.Label(r, text=txt, font=("Consolas", 10), foreground="#2c3e50")
            lbl.pack(anchor=tk.W)
            self.measure_labels[key] = lbl

        # ---- Config ----------------------
        self.config_panel = CollapsibleFrame(left, title="Configuration Parameters")
        self.config_panel.pack(fill=tk.X, pady=(0, 10), padx=5)
        cfg = self.config_panel.get_content_frame()

        params_frame = ttk.Frame(cfg)
        params_frame.pack(fill=tk.X)

        self.create_threshold_controls(params_frame)
        self.create_clahe_controls(params_frame)
        self.create_morphology_controls(params_frame)
        self.create_watershed_controls(params_frame)
        self.create_fluorescence_controls(params_frame)
        self.create_label_controls(params_frame)

        # ---- Image tabs (REORGANIZED) -------------------------------------------------
        img_frame = ttk.Frame(main)
        img_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(img_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        tabs = [
            ("1_Original (BF)", "tab_original"), 
            ("2_CLAHE Enhanced", "tab_enhanced"), 
            ("3_Threshold", "tab_threshold"),
            ("4_Morphology", "tab_morphology"), 
            ("5_Final Contours", "tab_contours"),
            ("6_BF+Fluor Overlay", "tab_overlay"), 
            ("7_Fluorescence", "tab_fluorescence"),
            ("8_Statistics", "tab_statistics")
        ]
        
        for name, attr in tabs:
            tab = ttk.Frame(self.notebook)
            setattr(self, attr, tab)
            self.notebook.add(tab, text=name)

        # Create canvases for image tabs
        for attr in ["canvas_original", "canvas_enhanced", "canvas_threshold",
                     "canvas_morphology", "canvas_contours", "canvas_overlay", 
                     "canvas_fluorescence"]:
            tab_attr = attr.replace("canvas_", "tab_")
            canvas = tk.Canvas(getattr(self, tab_attr), bg='#f8f9fa', highlightthickness=0)
            setattr(self, attr, canvas)
            canvas.pack(fill=tk.BOTH, expand=True)

        self.setup_statistics_table()
        self.canvas_original.bind("<Button-1>", self.on_canvas_click)
        self.canvas_original.bind("<Button-3>", self.clear_probe)

        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN,
                  anchor=tk.W, padding=(5, 2)).pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        shortcut_hint = ttk.Label(status_frame, text="Keys: , . = Tabs | ← → = Images | L = Load | D = Dark | Esc = Exit", 
                                 relief=tk.SUNKEN, anchor=tk.E, padding=(5, 2), 
                                 foreground="#666", font=("Segoe UI", 8))
        shortcut_hint.pack(side=tk.RIGHT)

        if self.dark_mode_var.get():
            self.apply_dark_mode()

    # --------------------------------------------------------------------- #
    # OPTIMIZED scrollbar helpers
    # --------------------------------------------------------------------- #
    def _update_scroll_region_delayed(self):
        """Debounced scroll region update."""
        try:
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
        finally:
            self._scroll_update_pending = False

    def _on_canvas_configure(self, event):
        """Update the scrollable frame width when canvas is resized."""
        self.left_canvas.itemconfig(self.canvas_frame, width=event.width)

    def _bind_mousewheel(self):
        """Platform-specific mouse wheel binding."""
        system = platform.system()
        
        def on_enter(event):
            if system == "Darwin":
                self.left_canvas.bind("<MouseWheel>", self._on_mousewheel)
            elif system == "Linux":
                self.left_canvas.bind("<Button-4>", self._on_mousewheel)
                self.left_canvas.bind("<Button-5>", self._on_mousewheel)
            else:
                self.left_canvas.bind("<MouseWheel>", self._on_mousewheel)
        
        def on_leave(event):
            self.left_canvas.unbind("<MouseWheel>")
            self.left_canvas.unbind("<Button-4>")
            self.left_canvas.unbind("<Button-5>")
        
        self.left_canvas.bind("<Enter>", on_enter)
        self.left_canvas.bind("<Leave>", on_leave)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        system = platform.system()
        
        if system == "Darwin":
            delta = -1 if event.delta > 0 else 1
        elif event.num == 5 or event.delta < 0:
            delta = 1
        elif event.num == 4 or event.delta > 0:
            delta = -1
        else:
            return
        
        self.left_canvas.yview_scroll(delta, "units")

    # --------------------------------------------------------------------- #
    # Statistics table (WITH HISTOGRAM - ✨ FIXED)
    # --------------------------------------------------------------------- #
    def setup_statistics_table(self):
        """Setup the statistics table with histogram in the Statistics tab."""
        main_container = ttk.Frame(self.tab_statistics)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Top section: Table
        table_frame = ttk.Frame(main_container)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        cols = ("Index", "BF Area (px²)", "BF Area (µm²)", "Fluor Mean", "Fluor Total", "Fluor/Area")
        self.stats_columns = cols
        self.stats_col_map = {
            "Index": "index",
            "BF Area (px²)": "bf_area_px",
            "BF Area (µm²)": "bf_area_um2",
            "Fluor Mean": "fluor_mean",
            "Fluor Total": "fluor_total",
            "Fluor/Area": "fluor_per_area",
        }
        
        self.stats_tree = ttk.Treeview(table_frame, columns=cols, show='headings', height=12)
        for c in cols:
            key = self.stats_col_map.get(c, "")
            if key:
                self.stats_tree.heading(c, text=c, command=lambda col_key=key: self.on_stats_heading_click(col_key))
            else:
                self.stats_tree.heading(c, text=c)
            self.stats_tree.column(c, width=120, anchor=tk.CENTER)
        self.stats_tree.column("Index", width=60)

        self.stats_tree.bind('<<TreeviewSelect>>', self.on_stats_row_select)

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.stats_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.stats_tree.xview)
        self.stats_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.stats_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        # Summary and export row
        bottom_frame = ttk.Frame(table_frame)
        bottom_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky='ew')
        
        self.stats_summary = ttk.Label(bottom_frame, text="No data", font=("Segoe UI", 10, "bold"))
        self.stats_summary.pack(side=tk.LEFT, padx=(0, 10))
        
        export_btn = ttk.Button(bottom_frame, text="Export to CSV", command=self.export_stats_to_csv)
        export_btn.pack(side=tk.RIGHT)
        ToolTip(export_btn, "Export current statistics table to CSV file")
        
        # ✨ FIXED: Bottom section: Histogram (only if matplotlib available)
        if MATPLOTLIB_AVAILABLE:
            histogram_frame = ttk.LabelFrame(main_container, text=" Distribution Histograms ", padding=10)
            histogram_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
            
            self.hist_fig = Figure(figsize=(12, 3), dpi=80)
            self.hist_canvas_widget = FigureCanvasTkAgg(self.hist_fig, master=histogram_frame)
            self.hist_canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            # Show message if matplotlib not available
            no_hist_frame = ttk.LabelFrame(main_container, text=" Histograms ", padding=10)
            no_hist_frame.pack(fill=tk.X, pady=(10, 0))
            ttk.Label(no_hist_frame, 
                     text="⚠️  matplotlib not installed\nInstall with: pip install matplotlib",
                     font=("Segoe UI", 9), foreground="#e67e22").pack()
        
        self.update_stats_heading_arrows()

    def update_histograms(self):
        """Update histogram plots with current bacteria statistics."""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        if not self.bacteria_stats:
            self.hist_fig.clear()
            self.hist_canvas_widget.draw()
            return
        
        # Clear previous plots
        self.hist_fig.clear()
        
        # Extract data
        areas_um2 = [s['bf_area_um2'] for s in self.bacteria_stats]
        fluor_per_area = [s['fluor_per_area'] for s in self.bacteria_stats]
        fluor_total = [s['fluor_total'] for s in self.bacteria_stats]
        
        # Create subplots
        ax1 = self.hist_fig.add_subplot(131)
        ax2 = self.hist_fig.add_subplot(132)
        ax3 = self.hist_fig.add_subplot(133)
        
        # Determine if dark mode
        is_dark = self.dark_mode_var.get()
        bg_color = '#2b2b2b' if is_dark else 'white'
        fg_color = '#e0e0e0' if is_dark else 'black'
        grid_color = '#4a4a4a' if is_dark else '#e0e0e0'
        
        # ✨ FIXED: Use set_facecolor instead of patch.set_facecolor
        self.hist_fig.set_facecolor(bg_color)
        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor(bg_color)
            ax.tick_params(colors=fg_color)
            ax.spines['bottom'].set_color(fg_color)
            ax.spines['top'].set_color(fg_color)
            ax.spines['left'].set_color(fg_color)
            ax.spines['right'].set_color(fg_color)
            ax.xaxis.label.set_color(fg_color)
            ax.yaxis.label.set_color(fg_color)
            # ✨ FIXED: Use set_color instead of title.set_color
            ax.title.set_color(fg_color)
        
        # Plot 1: Area distribution
        ax1.hist(areas_um2, bins=20, color='#4CAF50', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Area (µm²)', fontsize=9)
        ax1.set_ylabel('Count', fontsize=9)
        ax1.set_title(f'BF Area (n={len(areas_um2)})', fontsize=10)
        ax1.grid(True, alpha=0.3, color=grid_color)
        ax1.tick_params(labelsize=8)
        
        # Plot 2: Fluorescence/Area distribution
        ax2.hist(fluor_per_area, bins=20, color='#FF5722', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Fluor/Area', fontsize=9)
        ax2.set_ylabel('Count', fontsize=9)
        ax2.set_title(f'Fluorescence per Area (n={len(fluor_per_area)})', fontsize=10)
        ax2.grid(True, alpha=0.3, color=grid_color)
        ax2.tick_params(labelsize=8)
        
        # Plot 3: Total fluorescence distribution
        ax3.hist(fluor_total, bins=20, color='#2196F3', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Total Fluorescence', fontsize=9)
        ax3.set_ylabel('Count', fontsize=9)
        ax3.set_title(f'Total Fluorescence (n={len(fluor_total)})', fontsize=10)
        ax3.grid(True, alpha=0.3, color=grid_color)
        ax3.tick_params(labelsize=8)
        
        self.hist_fig.tight_layout()
        self.hist_canvas_widget.draw()

    def calculate_bacteria_statistics(self, contours, bf_img, fluor_img=None):
        """Calculate statistics for each bacterium contour with physical units."""
        pixel_size = self.get_pixel_size()
        pixel_area = pixel_size ** 2
        
        stats = []
        for idx, cnt in enumerate(contours):
            area_px = cv2.contourArea(cnt)
            area_um2 = area_px * pixel_area
            
            mask = np.zeros(bf_img.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            f_mean = f_total = f_per = 0.0
            if fluor_img is not None:
                vals = fluor_img[mask == 255]
                if len(vals):
                    f_mean = float(np.mean(vals))
                    f_total = float(np.sum(vals))
                    f_per = f_total / area_px if area_px else 0.0
            
            stats.append({
                'orig_idx': idx,
                'bf_area_px': area_px,
                'bf_area_um2': area_um2,
                'fluor_mean': f_mean,
                'fluor_total': f_total,
                'fluor_per_area': f_per,
                'contour': cnt
            })
        return stats

    def apply_stats_sort(self):
        """Apply current sort to self.bacteria_stats in-place."""
        if not self.bacteria_stats:
            return
        key, desc = self.stats_sort
        if key == 'index':
            self.bacteria_stats.sort(key=lambda s: s.get('orig_idx', 0), reverse=desc)
        else:
            self.bacteria_stats.sort(key=lambda s: s.get(key, 0.0), reverse=desc)

    def on_stats_heading_click(self, key: str):
        """Handle click on a stats table header to sort by that column."""
        cur_key, cur_desc = self.stats_sort
        if key == cur_key:
            self.stats_sort = (key, not cur_desc)
        else:
            self.stats_sort = (key, True)
        self.update_stats_heading_arrows()
        self.update_preview()

    def update_stats_heading_arrows(self):
        """Update header texts to show sort direction arrows."""
        if not hasattr(self, 'stats_tree'):
            return
        key, desc = self.stats_sort
        for title in self.stats_columns:
            col_key = self.stats_col_map.get(title, "")
            arrow = ' ▼' if (col_key == key and desc) else (' ▲' if col_key == key else '')
            self.stats_tree.heading(title, text=f"{title}{arrow}")

    def on_stats_row_select(self, event):
        """Handle row selection in statistics table."""
        selection = self.stats_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        values = self.stats_tree.item(item, 'values')
        if not values:
            return
        
        selected_index = int(values[0]) - 1
        
        if 0 <= selected_index < len(self.bacteria_stats):
            self.current_bacteria_index = selected_index
            self.notebook.select(self.tab_contours)
            self.update_preview()

    def export_stats_to_csv(self):
        """Export statistics table to CSV file with metadata."""
        if not self.bacteria_stats:
            messagebox.showinfo("No Data", "No statistics to export. Load an image first.")
            return
        
        default_name = "bacteria_statistics.csv"
        if self.current_file:
            default_name = self.current_file.stem + "_statistics.csv"
        
        filepath = filedialog.asksaveasfilename(
            title="Save Statistics as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_name
        )
        
        if not filepath:
            return
        
        try:
            import csv
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                writer.writerow(["# Bacteria Segmentation Analysis"])
                writer.writerow([f"# Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
                writer.writerow([f"# Image: {self.current_file.name if self.current_file else 'Unknown'}"])
                
                if self.current_metadata and self.has_metadata:
                    writer.writerow([f"# Sample: {self.current_metadata.get('sample_name', 'N/A')}"])
                    writer.writerow([f"# Pixel Size: {self.current_metadata.get('pixel_size_um', 0.1289):.4f} µm"])
                    writer.writerow([f"# Objective: {self.current_metadata.get('objective', 'N/A')}"])
                    if 'acquired' in self.current_metadata:
                        writer.writerow([f"# Acquired: {self.current_metadata['acquired']}"])
                    if 'exposure_times' in self.current_metadata:
                        exp = self.current_metadata['exposure_times']
                        writer.writerow([f"# BF Exposure: {exp.get('brightfield_ms', 'N/A')} ms"])
                        writer.writerow([f"# Fluor Exposure: {exp.get('fluorescence_ms', 'N/A')} ms"])
                else:
                    writer.writerow([f"# ⚠️  No metadata available for this image"])
                    writer.writerow([f"# Pixel Size: 0.1289 µm (default fallback)"])
                
                writer.writerow(["# Analysis Parameters:"])
                writer.writerow([f"# Threshold: {'Otsu' if self.params['use_otsu'].get() else self.params['manual_threshold'].get()}"])
                writer.writerow([f"# CLAHE: {'Enabled' if self.params['enable_clahe'].get() else 'Disabled'}"])
                if self.params['enable_clahe'].get():
                    writer.writerow([f"# CLAHE Clip: {self.params['clahe_clip'].get()}"])
                    writer.writerow([f"# CLAHE Tile: {self.params['clahe_tile'].get()}"])
                writer.writerow([f"# Min Area: {self.params['min_area'].get()} px²"])
                writer.writerow([f"# Min Fluor/Area: {self.params['min_fluor_per_area'].get()}"])
                writer.writerow([f"# Watershed Dilate: {self.params['watershed_dilate'].get()}"])
                writer.writerow([])
                
                writer.writerow(["Index", "BF Area (px²)", "BF Area (µm²)", "Fluor Mean", "Fluor Total", "Fluor/Area"])
                
                for i, s in enumerate(self.bacteria_stats, 1):
                    writer.writerow([
                        i,
                        f"{s['bf_area_px']:.1f}",
                        f"{s['bf_area_um2']:.3f}",
                        f"{s['fluor_mean']:.2f}",
                        f"{s['fluor_total']:.1f}",
                        f"{s['fluor_per_area']:.3f}"
                    ])
            
            metadata_status = "with metadata" if self.has_metadata else "without metadata (defaults used)"
            self.status_var.set(f"Statistics exported {metadata_status} to {Path(filepath).name}")
            messagebox.showinfo("Export Successful", f"Statistics exported to:\n{filepath}\n\n{metadata_status.capitalize()}")
            print(f"✅ Statistics exported ({metadata_status}): {filepath}")
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to export statistics:\n{str(e)}"
            self.status_var.set("Export failed")
            messagebox.showerror("Export Error", error_msg)
            print(f"❌ Export error: {e}")
            traceback.print_exc()

    def update_statistics_table(self):
        """Update the statistics table and histograms with current bacteria data."""
        for i in self.stats_tree.get_children():
            self.stats_tree.delete(i)
        if self.bacteria_stats:
            for i, s in enumerate(self.bacteria_stats, 1):
                self.stats_tree.insert('', 'end', values=(
                    i,
                    f"{s['bf_area_px']:.1f}",
                    f"{s['bf_area_um2']:.3f}",
                    f"{s['fluor_mean']:.2f}",
                    f"{s['fluor_total']:.1f}",
                    f"{s['fluor_per_area']:.3f}"
                ))
            total = len(self.bacteria_stats)
            avg_um2 = np.mean([s['bf_area_um2'] for s in self.bacteria_stats])
            avg_fpa = np.mean([s['fluor_per_area'] for s in self.bacteria_stats])
            self.stats_summary.config(
                text=f"Total: {total} bacteria | Avg Area: {avg_um2:.3f} µm² | Avg F/A: {avg_fpa:.3f}"
            )
            self.update_histograms()
        else:
            self.stats_summary.config(text="No bacteria detected")
            if MATPLOTLIB_AVAILABLE and hasattr(self, 'hist_fig'):
                self.hist_fig.clear()
                self.hist_canvas_widget.draw()

    def update_metadata_panel(self):
        """Update metadata panel with current image metadata."""
        if not self.current_metadata or not self.has_metadata:
            self.measure_labels["sample_name"].config(
                text="⚠️  No metadata (using defaults)", 
                foreground="#e67e22"
            )
            self.measure_labels["pixel_size"].config(
                text="Pixel Size: 0.1289 µm (default)",
                foreground="#95a5a6"
            )
            self.measure_labels["objective"].config(text="Objective: -", foreground="#95a5a6")
            self.measure_labels["acquired"].config(text="Acquired: -", foreground="#95a5a6")
            self.measure_labels["exposure_bf"].config(text="BF Exposure: -", foreground="#95a5a6")
            self.measure_labels["exposure_fluor"].config(text="Fluor Exposure: -", foreground="#95a5a6")
            return
        
        normal_color = "#e0e0e0" if self.dark_mode_var.get() else "#2c3e50"
        
        sample = self.current_metadata.get('sample_name', '-')
        pixel_size = self.current_metadata.get('pixel_size_um', 0.1289)
        objective = self.current_metadata.get('objective', '-')
        acquired = self.current_metadata.get('acquired', '-')
        
        self.measure_labels["sample_name"].config(
            text=f"Sample: {sample}",
            foreground=normal_color
        )
        self.measure_labels["pixel_size"].config(
            text=f"Pixel Size: {pixel_size:.4f} µm",
            foreground=normal_color
        )
        self.measure_labels["objective"].config(
            text=f"Objective: {objective}",
            foreground=normal_color
        )
        
        if len(acquired) > 30:
            acquired = acquired[:27] + "..."
        self.measure_labels["acquired"].config(
            text=f"Acquired: {acquired}",
            foreground=normal_color
        )
        
        if 'exposure_times' in self.current_metadata:
            exp = self.current_metadata['exposure_times']
            bf_exp = exp.get('brightfield_ms', '-')
            fluor_exp = exp.get('fluorescence_ms', '-')
            self.measure_labels["exposure_bf"].config(
                text=f"BF Exposure: {bf_exp} ms",
                foreground=normal_color
            )
            self.measure_labels["exposure_fluor"].config(
                text=f"Fluor Exposure: {fluor_exp} ms",
                foreground=normal_color
            )
        else:
            self.measure_labels["exposure_bf"].config(text="BF Exposure: -", foreground="#95a5a6")
            self.measure_labels["exposure_fluor"].config(text="Fluor Exposure: -", foreground="#95a5a6")

    # --------------------------------------------------------------------- #
    # Scale bar
    # --------------------------------------------------------------------- #
    def draw_scale_bar(self, img_pil, pixel_size_um):
        """Draw scale bar on PIL image with physical units."""
        if not self.params['show_scale_bar'].get():
            return img_pil
        
        draw = ImageDraw.Draw(img_pil)
        img_width, img_height = img_pil.size
        
        # Scale bar parameters
        target_bar_length_um = 5.0  # Target 5 µm bar
        bar_length_px = int(target_bar_length_um / pixel_size_um)
        
        # Adjust if bar is too long for image
        max_bar_width = img_width * 0.25
        if bar_length_px > max_bar_width:
            bar_length_px = int(max_bar_width)
            target_bar_length_um = bar_length_px * pixel_size_um
        
        bar_thickness = 3
        margin = 15
        
        # Position at bottom-right
        x1 = img_width - margin - bar_length_px
        y1 = img_height - margin - bar_thickness - 20
        x2 = img_width - margin
        y2 = y1 + bar_thickness
        
        # Draw white bar with black outline
        draw.rectangle([x1-1, y1-1, x2+1, y2+1], fill='black')
        draw.rectangle([x1, y1, x2, y2], fill='white')
        
        # Draw text label
        font_size = 12
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        text = f"{target_bar_length_um:.1f} µm"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        text_x = x1 + (bar_length_px - text_w) // 2
        text_y = y2 + 3
        
        # Text with black outline
        for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            draw.text((text_x+dx, text_y+dy), text, font=font, fill='black')
        draw.text((text_x, text_y), text, font=font, fill='white')
        
        return img_pil

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

        self.update_preview()

        if event.state & 0x4:
            self.auto_tune_from_point(ix, iy, int(self.original_image[iy, ix]))

        if self.bacteria_stats:
            for idx, s in enumerate(self.bacteria_stats):
                if cv2.pointPolygonTest(s['contour'], (ix, iy), False) >= 0:
                    self.current_bacteria_index = idx
                    break
            else:
                self.current_bacteria_index = -1

    def clear_probe(self, event=None):
        """Clear probe point and reset measurement panel."""
        for cid in self.probe_canvas_ids:
            self.canvas_original.delete(cid)
        self.probe_canvas_ids = []
        self.probe_point = None
        self.current_bacteria_index = -1
        self.reset_measurement_panel()
        if self.current_file:
            self.status_var.set(f"Ready – {self.current_file.name}")

    def update_measurement_panel(self, x, y, val):
        """Update measurement panel with clicked pixel information."""
        pixel_size = self.get_pixel_size()
        
        self.measure_labels["pixel_coord"].config(text=f"Pixel: ({x}, {y})")
        self.measure_labels["pixel_value"].config(text=f"Value: {val}")
        inside = area_px = area_um2 = 0
        if self.current_contours:
            for c in self.current_contours:
                if cv2.pointPolygonTest(c, (x, y), False) >= 0:
                    inside = True
                    area_px = cv2.contourArea(c)
                    area_um2 = area_px * (pixel_size ** 2)
                    break
        self.measure_labels["inside_contour"].config(
            text=f"Inside Contour: {'Yes' if inside else 'No'}",
            foreground="#27ae60" if inside else "#e74c3c")
        self.measure_labels["contour_area_px"].config(
            text=f"Contour Area: {int(area_px)} px²" if inside else "Contour Area: - px²")
        self.measure_labels["contour_area_um"].config(
            text=f"Contour Area: {area_um2:.3f} µm²" if inside else "Contour Area: - µm²")

    def reset_measurement_panel(self):
        """Reset measurement panel to default values."""
        defaults = {
            "pixel_coord": "Pixel: -, -", 
            "pixel_value": "Value: -",
            "inside_contour": "Inside Contour: -", 
            "contour_area_px": "Contour Area: - px²",
            "contour_area_um": "Contour Area: - µm²"
        }
        for k, txt in defaults.items():
            if k in self.measure_labels:
                fg = "#e0e0e0" if self.dark_mode_var.get() else "#2c3e50"
                self.measure_labels[k].config(text=txt, foreground=fg)

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
        lf = ttk.LabelFrame(parent, text=" Labels & Scale ", padding=8)
        lf.pack(fill=tk.X, pady=(0, 5))
        
        cb = ttk.Checkbutton(lf, text="Show Labels", variable=self.params['show_labels'],
                             command=self.update_preview)
        cb.pack(anchor=tk.W, pady=2)
        ToolTip(cb, "Display numbered labels for bacteria")
        
        cb_scale = ttk.Checkbutton(lf, text="Show Scale Bar", variable=self.params['show_scale_bar'],
                                   command=self.update_preview)
        cb_scale.pack(anchor=tk.W, pady=2)
        ToolTip(cb_scale, "Display scale bar with physical units (µm)")
        
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
    # Load image (IMPROVED: metadata handling)
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

        metadata = self.load_metadata_for_image(path)
        if metadata:
            self.current_metadata = metadata
            self.has_metadata = True
            pixel_size = metadata.get('pixel_size_um', 0.1289)
            print(f"  📊 Metadata loaded - Pixel size: {pixel_size:.4f} µm")
        else:
            print(f"  ⚠️  No metadata found - using defaults")
            self.current_metadata = {'pixel_size_um': 0.1289}
            self.has_metadata = False

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
                
                if self.has_metadata and 'channels' in self.current_metadata:
                    self.apply_metadata_display_settings()
                else:
                    print(f"  ℹ️  No channel metadata - keeping manual display settings")
            else:
                print(f"  ⚠️  Fluorescence file exists but couldn't be read")
        else:
            print(f"  ℹ️  No fluorescence file found")

        self.probe_point = None
        self.clear_probe()
        self.bacteria_stats = []
        self.current_bacteria_index = -1

        h, w = bf.shape[:2]
        fmsg = " + fluorescence" if self.fluorescence_image is not None else ""
        meta_status = " (with metadata)" if self.has_metadata else " (no metadata)"
        self.status_var.set(f"Loaded: {path.name}{fmsg}{meta_status} ({w}x{h})")
        self.root.title(f"{path.name} - Bacteria Segmentation Tuner")
        self.update_navigation_buttons()
        self.update_metadata_panel()
        self.update_preview()
        print(f"✅ Image loaded successfully: {w}x{h}{fmsg}{meta_status}\n")

    # --------------------------------------------------------------------- #
    # SEGMENTATION
    # --------------------------------------------------------------------- #
    def segment_bacteria(self, gray_bf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Segment bacteria from bright-field image."""
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
        contours: Sequence[MatLike] = res[-2]

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
        
        pixel_size = self.get_pixel_size()
        pil_img = self.draw_scale_bar(pil_img, pixel_size)
        
        img_rgb_array = np.array(pil_img)
        return cv2.cvtColor(img_rgb_array, cv2.COLOR_RGB2BGR)

    # --------------------------------------------------------------------- #
    # Preview update (OPTIMIZED with throttling)
    # --------------------------------------------------------------------- #
    def update_preview(self):
        """Update all preview tabs (with throttling)."""
        if self._preview_update_pending:
            return
        
        self._preview_update_pending = True
        self.root.after(50, self._do_update_preview)

    def _do_update_preview(self):
        """Actual preview update logic."""
        try:
            if self.original_image is None:
                return
            
            enhanced, thresh, cleaned, bacteria = self.segment_bacteria(self.original_image)

            all_stats = self.calculate_bacteria_statistics(bacteria, self.original_image, self.fluorescence_image)
            min_fpa = self.params['min_fluor_per_area'].get()

            if self.fluorescence_image is not None and min_fpa > 0:
                self.bacteria_stats = [s for s in all_stats if s['fluor_per_area'] >= min_fpa]
            else:
                self.bacteria_stats = all_stats
            self.apply_stats_sort()
            bacteria = [s['contour'] for s in self.bacteria_stats]
            self.current_contours = bacteria

            base = f"Detected {len(bacteria)} bacteria"
            if self.fluorescence_image is not None and min_fpa > 0:
                base = f"Detected {len(self.bacteria_stats)}/{len(all_stats)} (min F/A {min_fpa:.1f})"
            if not self.probe_point and self.current_file:
                fstat = " (with fluorescence)" if self.fluorescence_image is not None else ""
                meta_status = "" if self.has_metadata else " [no metadata]"
                self.status_var.set(f"{base} | {self.current_file.name}{fstat}{meta_status}")

            self.update_statistics_table()
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
            ov_rgb = cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)
            ov_pil = Image.fromarray(ov_rgb)
            ov_pil = self.draw_scale_bar(ov_pil, self.get_pixel_size())
            ov = cv2.cvtColor(np.array(ov_pil), cv2.COLOR_RGB2BGR)
            self.display_image(ov, self.canvas_overlay)

        except Exception as e:
            import traceback
            self.status_var.set(f"Error: {e}")
            traceback.print_exc()
        finally:
            self._preview_update_pending = False

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
            assert self.original_image is not None
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
    def exit_with_confirmation(self):
        """Exit application with confirmation dialog (for Escape key)."""
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            self.exit_application()
    
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
    