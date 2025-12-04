#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom UI widgets.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable, Union


class ToolTip:
    """Tooltip widget that appears on hover."""
    
    def __init__(self, widget: tk.Widget, text: str):
        """Initialize tooltip.
        
        Args:
            widget: Widget to attach tooltip to
            text: Tooltip text
        """
        self.widget = widget
        self.text = text
        self.tip_window: Optional[tk.Toplevel] = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)
    
    def show_tip(self, event=None):
        """Show tooltip."""
        if self.tip_window or not self.text:
            return
        
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = ttk.Label(
            tw, text=self.text, justify=tk.LEFT,
            background="#ffffe0", relief=tk.SOLID, borderwidth=1,
            font=("Segoe UI", 9), padding=(5, 3)
        )
        label.pack()
    
    def hide_tip(self, event=None):
        """Hide tooltip."""
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class CollapsibleFrame(ttk.Frame):
    """Frame that can be collapsed/expanded."""
    
    def __init__(self, parent, title: str = "", **kwargs):
        """Initialize collapsible frame.
        
        Args:
            parent: Parent widget
            title: Frame title
            **kwargs: Additional frame options
        """
        super().__init__(parent, **kwargs)
        self.is_collapsed = False
        
        # Header
        self.header = ttk.Frame(self, relief=tk.RAISED, borderwidth=1)
        self.header.pack(fill=tk.X, padx=2, pady=2)
        
        self.toggle_btn = ttk.Button(
            self.header, text="▼", width=3, command=self.toggle
        )
        self.toggle_btn.pack(side=tk.LEFT, padx=2)
        
        self.title_label = ttk.Label(
            self.header, text=title, font=("Segoe UI", 10, "bold")
        )
        self.title_label.pack(side=tk.LEFT, padx=5)
        
        # Content
        self.content = ttk.Frame(self, padding=10)
        self.content.pack(fill=tk.BOTH, expand=True)
    
    def toggle(self):
        """Toggle collapsed/expanded state."""
        if self.is_collapsed:
            self.expand()
        else:
            self.collapse()
    
    def collapse(self):
        """Collapse the frame."""
        self.content.pack_forget()
        self.toggle_btn.config(text="▶")
        self.is_collapsed = True
    
    def expand(self):
        """Expand the frame."""
        self.content.pack(fill=tk.BOTH, expand=True)
        self.toggle_btn.config(text="▼")
        self.is_collapsed = False
    
    def get_content_frame(self) -> ttk.Frame:
        """Get the content frame for adding widgets.
        
        Returns:
            Content frame widget
        """
        return self.content


class ProgressEntry(ttk.Frame):
    """Entry field with visual progress bar."""
    
    def __init__(self, 
                 parent,
                 label_text: str,
                 min_val: float,
                 max_val: float,
                 initial_val: float,
                 on_change: Callable[[Union[int, float]], None],
                 resolution: float = 1.0,
                 is_float: bool = False,
                 tooltip: str = ""):
        """Initialize progress entry.
        
        Args:
            parent: Parent widget
            label_text: Label text
            min_val: Minimum value
            max_val: Maximum value
            initial_val: Initial value
            on_change: Callback when value changes
            resolution: Value resolution/step
            is_float: True if float values
            tooltip: Tooltip text
        """
        super().__init__(parent)
        
        self.min_val = min_val
        self.max_val = max_val
        self.resolution = resolution
        self.is_float = is_float
        self.on_change = on_change
        
        # Label
        self.label = ttk.Label(self, text=label_text, width=20, anchor=tk.W)
        self.label.pack(side=tk.LEFT)
        if tooltip:
            ToolTip(self.label, tooltip)
        
        # Entry
        self.entry = ttk.Entry(self, width=7, justify=tk.RIGHT, 
                              font=("Consolas", 10))
        self.entry.pack(side=tk.LEFT, padx=(2, 4))
        self.entry.insert(0, str(initial_val))
        self.entry.bind('<Return>', self._on_entry_change)
        self.entry.bind('<FocusOut>', self._on_entry_change)
        if tooltip:
            ToolTip(self.entry, tooltip)
        
        # Progress bar
        self.progressbar = ttk.Progressbar(
            self, orient=tk.HORIZONTAL, mode='determinate', length=150
        )
        self.progressbar.pack(side=tk.RIGHT, padx=(0, 2))
        if tooltip:
            ToolTip(self.progressbar, tooltip)
        
        self.update_progress(initial_val)
    
    def get_value(self) -> Union[int, float]:
        """Get current numeric value.
        
        Returns:
            Current value
        """
        try:
            return (float if self.is_float else int)(
                float(self.entry.get().strip())
            )
        except ValueError:
            return self.min_val
    
    def set_value(self, value: Union[int, float]):
        """Set entry value.
        
        Args:
            value: Value to set
        """
        value = max(self.min_val, min(self.max_val, value))
        if self.resolution != 1.0:
            value = round(value / self.resolution) * self.resolution
        
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(value))
        self.update_progress(value)
    
    def update_progress(self, value: Union[int, float]):
        """Update progress bar to reflect value.
        
        Args:
            value: Current value
        """
        percentage = (value - self.min_val) / (self.max_val - self.min_val) * 100
        self.progressbar['value'] = percentage
    
    def _on_entry_change(self, event):
        """Handle entry value change."""
        try:
            value = (float if self.is_float else int)(
                float(self.entry.get().strip())
            )
            value = max(self.min_val, min(self.max_val, value))
            
            if self.resolution != 1.0:
                value = round(value / self.resolution) * self.resolution
            
            self.set_value(value)
            self.on_change(value)
        except ValueError:
            self.set_value(self.min_val)