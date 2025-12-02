# ui/app.py

import tkinter as tk
from tkinter import ttk  # ttk provides more modern-looking widgets

class SegmentationViewer:
    """
    Main application class for the Bacteria Segmentation Tuner UI.
    """
    def __init__(self, root_window):
        # Store the main application window (the Tk instance from main.py)
        self.root = root_window
        self.root.title("Bacteria Segmentation Tuner")
        self.root.geometry("800x600")

        # Set up the main layout
        self.setup_ui()

    def setup_ui(self):
        """
        Build all the widgets (buttons, labels, canvas, etc.) here.
        """
        # Example: A simple label in the center of the window
        # In a real app, you would add image viewers, sliders, buttons, etc.
        welcome_label = ttk.Label(
            self.root,
            text="Welcome to the Segmentation Viewer UI!",
            font=("Helvetica", 16)
        )
        # Pack the label into the window manager
        welcome_label.pack(pady=50)

        # Example: A button that does something
        quit_button = ttk.Button(
            self.root,
            text="Quit Application",
            command=self.root.destroy # command to close the window
        )
        quit_button.pack(pady=10)

        print("UI initialized successfully.")

# You don't need to add `if __name__ == "__main__":` here,
# as the class is imported and run by `main.py`.
