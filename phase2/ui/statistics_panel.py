#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Statistics table and histogram panel.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable

# Try importing matplotlib
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class StatisticsPanel:
    """Manages statistics table and histograms."""
    
    def __init__(self, 
                 parent: ttk.Frame,
                 on_row_select: Callable,
                 dark_mode_var: tk.BooleanVar):
        """Initialize statistics panel.
        
        Args:
            parent: Parent frame
            on_row_select: Callback when row is selected
            dark_mode_var: Dark mode boolean variable
        """
        self.parent = parent
        self.on_row_select = on_row_select
        self.dark_mode_var = dark_mode_var
        
        self.stats_data: List[Dict] = []
        self.sort_key: str = "fluor_per_area"
        self.sort_descending: bool = True
        
        self.column_map = {
            "Index": "index",
            "BF Area (px²)": "bf_area_px",
            "BF Area (µm²)": "bf_area_um2",
            "Fluor Mean": "fluor_mean",
            "Fluor Total": "fluor_total",
            "Fluor/Area": "fluor_per_area",
        }
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create statistics panel widgets."""
        main_container = ttk.Frame(self.parent)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Table section
        table_frame = ttk.Frame(main_container)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = tuple(self.column_map.keys())
        self.tree = ttk.Treeview(table_frame, columns=columns, 
                                show='headings', height=12)
        
        # Configure columns
        for col in columns:
            key = self.column_map.get(col, "")
            if key:
                self.tree.heading(
                    col, text=col,
                    command=lambda k=key: self._on_heading_click(k)
                )
            else:
                self.tree.heading(col, text=col)
            
            self.tree.column(col, width=120, anchor=tk.CENTER)
        
        self.tree.column("Index", width=60)
        self.tree.bind('<<TreeviewSelect>>', self._on_tree_select)
        
        # Scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", 
                          command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal",
                          command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Bottom controls
        bottom_frame = ttk.Frame(table_frame)
        bottom_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky='ew')
        
        self.summary_label = ttk.Label(
            bottom_frame, text="No data", font=("Segoe UI", 10, "bold")
        )
        self.summary_label.pack(side=tk.LEFT, padx=(0, 10))
        
        from .widgets import ToolTip
        export_btn = ttk.Button(
            bottom_frame, text="Export to CSV",
            command=self._export_csv
        )
        export_btn.pack(side=tk.RIGHT)
        ToolTip(export_btn, "Export current statistics table to CSV file")
        
        # Histogram section
        if MATPLOTLIB_AVAILABLE:
            self._create_histogram_section(main_container)
        else:
            self._create_no_matplotlib_message(main_container)
        
        self._update_heading_arrows()
    
    def _create_histogram_section(self, parent):
        """Create matplotlib histogram section."""
        histogram_frame = ttk.LabelFrame(
            parent, text=" Distribution Histograms ", padding=10
        )
        histogram_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.hist_fig = Figure(figsize=(12, 3), dpi=80)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=histogram_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_no_matplotlib_message(self, parent):
        """Create message when matplotlib not available."""
        no_hist_frame = ttk.LabelFrame(parent, text=" Histograms ", padding=10)
        no_hist_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(
            no_hist_frame,
            text="⚠️  matplotlib not installed\nInstall with: pip install matplotlib",
            font=("Segoe UI", 9),
            foreground="#e67e22"
        ).pack()
    
    def update_data(self, stats: List[Dict]):
        """Update statistics table with new data.
        
        Args:
            stats: List of statistics dictionaries
        """
        self.stats_data = stats
        self._refresh_table()
        self._update_summary()
        self._update_histograms()
    
    def _refresh_table(self):
        """Refresh table display."""
        # Clear existing
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add data
        for i, s in enumerate(self.stats_data, 1):
            self.tree.insert('', 'end', values=(
                i,
                f"{s['bf_area_px']:.1f}",
                f"{s['bf_area_um2']:.3f}",
                f"{s['fluor_mean']:.2f}",
                f"{s['fluor_total']:.1f}",
                f"{s['fluor_per_area']:.3f}"
            ))
    
    def _update_summary(self):
        """Update summary statistics label."""
        if not self.stats_data:
            self.summary_label.config(text="No bacteria detected")
            return
        
        total = len(self.stats_data)
        avg_area = np.mean([s['bf_area_um2'] for s in self.stats_data])
        avg_fpa = np.mean([s['fluor_per_area'] for s in self.stats_data])
        
        self.summary_label.config(
            text=f"Total: {total} bacteria | "
                 f"Avg Area: {avg_area:.3f} µm² | "
                 f"Avg F/A: {avg_fpa:.3f}"
        )
    
    def _update_histograms(self):
        """Update histogram plots."""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'hist_fig'):
            return
        
        if not self.stats_data:
            self.hist_fig.clear()
            self.hist_canvas.draw()
            return
        
        # Extract data
        areas = [s['bf_area_um2'] for s in self.stats_data]
        fluor_per_area = [s['fluor_per_area'] for s in self.stats_data]
        fluor_total = [s['fluor_total'] for s in self.stats_data]
        
        # Clear and create subplots
        self.hist_fig.clear()
        ax1 = self.hist_fig.add_subplot(131)
        ax2 = self.hist_fig.add_subplot(132)
        ax3 = self.hist_fig.add_subplot(133)
        
        # Theme colors
        is_dark = self.dark_mode_var.get()
        bg_color = '#2b2b2b' if is_dark else 'white'
        fg_color = '#e0e0e0' if is_dark else 'black'
        grid_color = '#4a4a4a' if is_dark else '#e0e0e0'
        
        # Apply theme
        self.hist_fig.set_facecolor(bg_color)
        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor(bg_color)
            ax.tick_params(colors=fg_color)
            for spine in ax.spines.values():
                spine.set_color(fg_color)
            ax.xaxis.label.set_color(fg_color)
            ax.yaxis.label.set_color(fg_color)
            ax.title.set_color(fg_color)
        
        # Plot histograms
        ax1.hist(areas, bins=20, color='#4CAF50', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Area (µm²)', fontsize=9)
        ax1.set_ylabel('Count', fontsize=9)
        ax1.set_title(f'BF Area (n={len(areas)})', fontsize=10)
        ax1.grid(True, alpha=0.3, color=grid_color)
        ax1.tick_params(labelsize=8)
        
        ax2.hist(fluor_per_area, bins=20, color='#FF5722', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Fluor/Area', fontsize=9)
        ax2.set_ylabel('Count', fontsize=9)
        ax2.set_title(f'Fluorescence per Area (n={len(fluor_per_area)})', fontsize=10)
        ax2.grid(True, alpha=0.3, color=grid_color)
        ax2.tick_params(labelsize=8)
        
        ax3.hist(fluor_total, bins=20, color='#2196F3', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Total Fluorescence', fontsize=9)
        ax3.set_ylabel('Count', fontsize=9)
        ax3.set_title(f'Total Fluorescence (n={len(fluor_total)})', fontsize=10)
        ax3.grid(True, alpha=0.3, color=grid_color)
        ax3.tick_params(labelsize=8)
        
        self.hist_fig.tight_layout()
        self.hist_canvas.draw()
    
    def _on_heading_click(self, key: str):
        """Handle column heading click for sorting."""
        if key == self.sort_key:
            self.sort_descending = not self.sort_descending
        else:
            self.sort_key = key
            self.sort_descending = True
        
        self._sort_data()
        self._update_heading_arrows()
        self._refresh_table()
    
    def _sort_data(self):
        """Sort statistics data by current sort key."""
        from ..analysis.statistics import BacteriaStatistics
        self.stats_data = BacteriaStatistics.sort_stats(
            self.stats_data, self.sort_key, self.sort_descending
        )
    
    def _update_heading_arrows(self):
        """Update column headings to show sort arrows."""
        for col_name, col_key in self.column_map.items():
            if col_key == self.sort_key:
                arrow = ' ▼' if self.sort_descending else ' ▲'
            else:
                arrow = ''
            self.tree.heading(col_name, text=f"{col_name}{arrow}")
    
    def _on_tree_select(self, event):
        """Handle tree row selection."""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = selection[0]
        values = self.tree.item(item, 'values')
        if not values:
            return
        
        selected_index = int(values[0]) - 1
        self.on_row_select(selected_index)
    
    def _export_csv(self):
        """Export statistics to CSV."""
        if not self.stats_data:
            messagebox.showinfo("No Data", 
                              "No statistics to export. Load an image first.")
            return
        
        # This will be called from main window with proper context
        # For now, just show a placeholder message
        messagebox.showinfo("Export", 
                          "Export functionality requires main window context")