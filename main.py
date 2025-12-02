"""
main.py - Application Entry Point
Bacteria Segmentation Tuner
"""

import tkinter as tk
from ui.app import SegmentationViewer


def main():
    """Initialize and run the application."""
    root = tk.Tk()
    app = SegmentationViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()