#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bacteria Analyzer - Main Entry Point
Version 2.0

A comprehensive tool for analyzing bacterial microscopy images with
automated segmentation, fluorescence analysis, and statistical reporting.
"""

import sys
import tkinter as tk
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ui.main_window import MainWindow


def main():
    """Main application entry point."""
    # Create root window
    root = tk.Tk()
    
    # Set application icon (if available)
    # Uncomment and modify path if you have an icon file
    # try:
    #     icon_path = Path(__file__).parent / "resources" / "icon.ico"
    #     root.iconbitmap(str(icon_path))
    # except:
    #     pass
    
    # Create main window
    app = MainWindow(root)
    
    # Start main loop
    root.mainloop()


if __name__ == "__main__":
    main()