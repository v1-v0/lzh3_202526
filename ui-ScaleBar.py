from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import numpy as np
import tifffile as tiff
from PIL import Image, ImageTk, ImageDraw, ImageFont

from register_dataset import register_all_datasets, ImagePairRecord

# ---------------------------------------------------------------------------
# Pillow resampling compatibility
# ---------------------------------------------------------------------------
try:
    RESAMPLE = Image.Resampling.LANCZOS  # Pillow >= 9.1
except AttributeError:
    # Fallback for older Pillow versions; avoid referencing missing attributes directly
    resampling = (
        getattr(Image, "LANCZOS", None)
        or getattr(Image, "ANTIALIAS", None)
        or getattr(Image, "BICUBIC", None)
        or getattr(Image, "BILINEAR", None)
        or getattr(Image, "NEAREST", None)
    )
    RESAMPLE = resampling if resampling is not None else 0  # 0 ~ NEAREST


class BFFluoViewer(tk.Tk):
    def __init__(self, source_root: Path):
        super().__init__()

        self.title("BF/FLUO Viewer with Scale Bar")
        # Fixed main window size 1920x1080
        self.geometry("1920x1080")

        # Load dataset registry
        self.records = register_all_datasets(source_root)
        if not self.records:
            messagebox.showerror("Error", "No image pairs found.")
            self.destroy()
            return

        # State
        self.current_record: ImagePairRecord | None = None
        self.show_scaled = tk.BooleanVar(value=True)
        self.bar_length_um = tk.DoubleVar(value=20.0)

        # UI layout
        self._build_ui()
        self._populate_listbox()

        # Keep references to Tk images so they are not GC'd
        self._bf_tkimg: ImageTk.PhotoImage | None = None
        self._fluo_tkimg: ImageTk.PhotoImage | None = None

        # Ensure proper cleanup on window close (clicking the [X])
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        # Root horizontal split: left controls, right image region
        root_pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        root_pane.pack(fill=tk.BOTH, expand=True)

        # Left: list of image pairs + controls
        left_frame = ttk.Frame(root_pane, width=350)
        root_pane.add(left_frame, weight=0)

        ttk.Label(left_frame, text="Image pairs").pack(anchor=tk.W, padx=5, pady=(5, 0))

        # Listbox for dataset pairs
        self.listbox = tk.Listbox(left_frame, height=30)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.listbox.bind("<<ListboxSelect>>", self.on_select_pair)

        # Controls
        controls = ttk.Frame(left_frame)
        controls.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(
            controls,
            text="Show scale bar",
            variable=self.show_scaled,
            command=self.update_images,
        ).pack(anchor=tk.W)

        bar_frame = ttk.Frame(controls)
        bar_frame.pack(anchor=tk.W, pady=2)
        ttk.Label(bar_frame, text="Bar length (µm):").pack(side=tk.LEFT)
        ttk.Entry(bar_frame, width=6, textvariable=self.bar_length_um).pack(
            side=tk.LEFT
        )

        ttk.Button(
            controls,
            text="Refresh",
            command=self.update_images,
        ).pack(anchor=tk.W, pady=2)

        ttk.Button(
            controls,
            text="Exit",
            command=self.on_exit,  # unified cleanup path
        ).pack(anchor=tk.W, pady=8)

        # Right: two image panes in a PanedWindow (BF left, FLUO right)
        right_pane = ttk.Panedwindow(root_pane, orient=tk.HORIZONTAL)
        root_pane.add(right_pane, weight=1)

        # Brightfield pane
        bf_frame = ttk.Frame(right_pane)
        right_pane.add(bf_frame, weight=1)

        self.bf_label = ttk.Label(bf_frame, text="Brightfield")
        self.bf_label.pack(anchor=tk.CENTER, pady=(5, 0))

        self.bf_canvas = tk.Label(bf_frame, bg="black")
        self.bf_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Fluorescence pane
        fluo_frame = ttk.Frame(right_pane)
        right_pane.add(fluo_frame, weight=1)

        self.fluo_label = ttk.Label(fluo_frame, text="Fluorescence")
        self.fluo_label.pack(anchor=tk.CENTER, pady=(5, 0))

        self.fluo_canvas = tk.Label(fluo_frame, bg="black")
        self.fluo_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _populate_listbox(self):
        self.listbox.delete(0, tk.END)
        for idx, rec in enumerate(self.records):
            label = f"{rec.dataset} :: {rec.pair_name}"
            self.listbox.insert(tk.END, label)

            # Highlight datasets without metadata (pixel_size_um is None) in red
            if getattr(rec, "pixel_size_um", None) is None:
                self.listbox.itemconfig(idx, foreground="red")
            else:
                self.listbox.itemconfig(idx, foreground="black")

        # Select first by default
        if self.records:
            self.listbox.selection_set(0)
            self.on_select_pair()

    # ------------------------------------------------------------------ Callbacks

    def on_select_pair(self, event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.current_record = self.records[idx]
        self.update_images()

    def update_images(self):
        if self.current_record is None:
            return

        rec = self.current_record

        # Load images
        try:
            bf = tiff.imread(rec.bf_path)
            fluo = tiff.imread(rec.fluo_path)
        except Exception as e:
            messagebox.showerror("Error", f"Error loading images:\n{e}")
            return

        bf_img = self._to_pil(bf)
        fluo_img = self._to_pil(fluo)

        # Optionally add scale bar
        if self.show_scaled.get():
            bf_img = self._add_scale_bar(bf_img, rec)
            fluo_img = self._add_scale_bar(fluo_img, rec)

        # Resize to fit into window: two images side-by-side in 1920px
        # Reserve ~350px for left panel, remaining ~1570px → ~785px per image
        target_width = 780
        bf_img = self._resize_to_width(bf_img, target_width)
        fluo_img = self._resize_to_width(fluo_img, target_width)

        # Convert to PhotoImage
        self._bf_tkimg = ImageTk.PhotoImage(bf_img)
        self._fluo_tkimg = ImageTk.PhotoImage(fluo_img)

        self.bf_canvas.configure(image=self._bf_tkimg)
        self.fluo_canvas.configure(image=self._fluo_tkimg)

        # Update labels
        if rec.pixel_size_um is not None:
            px_text = f" (px: {rec.pixel_size_um:.3f} µm/pixel)"
        else:
            px_text = " (no metadata)"

        self.bf_label.config(text=f"Brightfield - {rec.pair_name}{px_text}")
        self.fluo_label.config(text=f"Fluorescence - {rec.pair_name}{px_text}")

    def on_exit(self):
        """Unified exit handler: release large objects and destroy the window."""
        # Clear references to images so they can be garbage-collected
        self._bf_tkimg = None
        self._fluo_tkimg = None
        # Optionally free record list if it is large
        # self.records = []

        self.destroy()

    # ------------------------------------------------------------------ Image helpers

    def _to_pil(self, arr: np.ndarray) -> Image.Image:
        """Convert a 2D or 3D numpy array to a PIL image, normalizing if needed."""
        # If image is 2D, convert to 8-bit gray
        if arr.ndim == 2:
            arr = arr.astype(np.float32)
            if arr.max() > 0:
                arr = arr / arr.max() * 255.0
            arr8 = arr.astype(np.uint8)
            return Image.fromarray(arr8, mode="L")

        # If image has channels
        if arr.ndim == 3:
            # Assume last axis is channels; scale each channel
            if arr.dtype != np.uint8:
                arr = arr.astype(np.float32)
                if arr.max() > 0:
                    arr = arr / arr.max() * 255.0
                arr = arr.astype(np.uint8)

            # If single channel but included as shape (H, W, 1)
            if arr.shape[2] == 1:
                arr = arr[..., 0]
                return Image.fromarray(arr, mode="L")

            # For > 3 channels, keep first 3 as RGB
            if arr.shape[2] > 3:
                arr = arr[..., :3]

            return Image.fromarray(arr, mode="RGB")

        raise ValueError(f"Unsupported image shape: {arr.shape}")

    def _resize_to_width(self, img: Image.Image, target_w: int) -> Image.Image:
        """Resize image to a given width, preserving aspect ratio."""
        w, h = img.size
        if w <= target_w:
            return img
        scale = target_w / w
        new_size = (target_w, int(h * scale))
        return img.resize(new_size, RESAMPLE)

    def _add_scale_bar(self, img: Image.Image, rec: ImagePairRecord) -> Image.Image:
        """Overlay a scale bar at the bottom-left of the image."""
        if rec.pixel_size_um is None:
            # No valid pixel size -> no scale bar
            return img

        try:
            bar_len_um = float(self.bar_length_um.get())
        except Exception:
            bar_len_um = 20.0

        px_size = rec.pixel_size_um  # µm / pixel
        bar_len_px = int(round(bar_len_um / px_size))

        if bar_len_px <= 0:
            return img

        draw = ImageDraw.Draw(img)
        w, h = img.size

        margin = 20
        bar_height = max(3, h // 200)

        x0 = margin
        y0 = h - margin - bar_height
        x1 = x0 + bar_len_px
        y1 = y0 + bar_height

        # Ensure the bar does not run outside the image
        if x1 > w - margin:
            bar_len_px = max(10, w - 2 * margin)
            bar_len_um = bar_len_px * px_size
            x1 = x0 + bar_len_px

        # Draw bar (white with black outline for visibility)
        draw.rectangle([x0, y0, x1, y1], fill="white")
        draw.rectangle([x0, y0, x1, y1], outline="black")

        # Label
        label = f"{bar_len_um:.0f} µm"
        text_x = x0 + bar_len_px / 2
        text_y = y0 - 4

        draw.text((text_x, text_y), label, fill="white", anchor="ms")  # middle, south

        return img


if __name__ == "__main__":
    project_root = Path.cwd()
    source_root = project_root / "source"

    app = BFFluoViewer(source_root)
    app.mainloop()