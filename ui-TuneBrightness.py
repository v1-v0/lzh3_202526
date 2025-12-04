from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import numpy as np
import tifffile as tiff
from PIL import Image, ImageTk, ImageDraw

# matplotlib for diagnostic histogram
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from register_dataset import register_all_datasets, ImagePairRecord

# ---------------------------------------------------------------------------
# Pillow resampling compatibility
# ---------------------------------------------------------------------------
try:
    RESAMPLE = Image.Resampling.LANCZOS  # Pillow >= 9.1
except AttributeError:
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

        self.title("BF/FLUO Viewer with Scale Bar + CLAHE + Histogram")
        # Fixed main window size 1920x1080
        self.geometry("1920x1080")

        # Load dataset registry
        self.records = register_all_datasets(source_root)
        if not self.records:
            messagebox.showerror("Error", "No image pairs found.")
            self.destroy()
            return

        # --- State --------------------------------------------------------
        self.current_record: ImagePairRecord | None = None

        self.show_scaled = tk.BooleanVar(value=True)
        self.bar_length_um = tk.DoubleVar(value=20.0)

        # Contrast mode:
        #   0 = Raw (no Leica, no auto)
        #   1 = Leica metadata
        #   2 = Auto normalize (percentile)
        #   3 = CLAHE (local contrast)
        self.contrast_mode = tk.IntVar(value=2)  # start with auto-normalize

        # Extra brightness factor applied after contrast mode
        self.brightness_factor = tk.DoubleVar(value=1.0)

        # CLAHE parameters (for BF-focused local contrast)
        self.clahe_clip_limit = tk.DoubleVar(value=2.0)  # typical: 1.0–3.0
        self.clahe_tile_size = tk.IntVar(value=64)       # tile side length in pixels

        # For diagnostic overlay (histogram & levels)
        self._diagnostic_window = None

        # To store last intensity transforms for diagnostics
        # Each is a dict per image: {'mode': str, 'low': float, 'high': float, 'gamma': float or None}
        self._bf_transform_info = None
        self._fluo_transform_info = None
        self._bf_last_array_raw = None
        self._fluo_last_array_raw = None

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
        left_frame = ttk.Frame(root_pane, width=380)
        root_pane.add(left_frame, weight=0)

        ttk.Label(left_frame, text="Image pairs").pack(
            anchor=tk.W, padx=5, pady=(5, 0)
        )

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

        # --- Contrast mode selection ------------------------------------
        contrast_frame = ttk.LabelFrame(controls, text="Contrast mode")
        contrast_frame.pack(anchor=tk.W, fill=tk.X, pady=(8, 4))

        ttk.Radiobutton(
            contrast_frame,
            text="Raw (no Leica, no auto)",
            variable=self.contrast_mode,
            value=0,
            command=self.update_images,
        ).pack(anchor=tk.W)

        ttk.Radiobutton(
            contrast_frame,
            text="Leica metadata",
            variable=self.contrast_mode,
            value=1,
            command=self.update_images,
        ).pack(anchor=tk.W)

        ttk.Radiobutton(
            contrast_frame,
            text="Auto normalize (percentile)",
            variable=self.contrast_mode,
            value=2,
            command=self.update_images,
        ).pack(anchor=tk.W)

        ttk.Radiobutton(
            contrast_frame,
            text="CLAHE (local contrast, BF‑tuned)",
            variable=self.contrast_mode,
            value=3,
            command=self.update_images,
        ).pack(anchor=tk.W)

        # Brightness factor control (applied after chosen contrast)
        bright_frame = ttk.Frame(controls)
        bright_frame.pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(bright_frame, text="Brightness ×").pack(side=tk.LEFT)
        ttk.Entry(bright_frame, width=4, textvariable=self.brightness_factor).pack(
            side=tk.LEFT
        )

        # CLAHE parameter controls
        clahe_frame = ttk.LabelFrame(controls, text="CLAHE settings")
        clahe_frame.pack(anchor=tk.W, fill=tk.X, pady=(6, 4))

        row1 = ttk.Frame(clahe_frame)
        row1.pack(anchor=tk.W)
        ttk.Label(row1, text="Clip limit:").pack(side=tk.LEFT)
        ttk.Entry(row1, width=5, textvariable=self.clahe_clip_limit).pack(side=tk.LEFT)
        ttk.Label(row1, text="  (1.0–3.0 typical)").pack(side=tk.LEFT)

        row2 = ttk.Frame(clahe_frame)
        row2.pack(anchor=tk.W)
        ttk.Label(row2, text="Tile size (px):").pack(side=tk.LEFT)
        ttk.Entry(row2, width=5, textvariable=self.clahe_tile_size).pack(side=tk.LEFT)

        # Buttons
        btn_frame = ttk.Frame(controls)
        btn_frame.pack(anchor=tk.W, pady=(6, 0))

        ttk.Button(
            btn_frame,
            text="Refresh",
            command=self.update_images,
        ).pack(side=tk.LEFT, padx=(0, 4))

        ttk.Button(
            btn_frame,
            text="Show diagnostics",
            command=self.show_diagnostics,
        ).pack(side=tk.LEFT)

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

        # Store raw arrays for diagnostics
        self._bf_last_array_raw = bf.copy()
        self._fluo_last_array_raw = fluo.copy()

        mode = self.contrast_mode.get()

        # Reset transform info
        self._bf_transform_info = {"mode": None, "low": None, "high": None, "gamma": None}
        self._fluo_transform_info = {"mode": None, "low": None, "high": None, "gamma": None}

        if mode == 1:
            # Leica metadata contrast
            bf, bf_info = self._apply_leica_scaling(bf, rec, channel_idx=0)
            fluo, fluo_info = self._apply_leica_scaling(fluo, rec, channel_idx=1)
            self._bf_transform_info = bf_info
            self._fluo_transform_info = fluo_info

        elif mode == 2:
            # Auto normalize (percentile-based)
            bf, bf_info = self._auto_normalize(bf)
            fluo, fluo_info = self._auto_normalize(fluo)
            self._bf_transform_info = bf_info
            self._fluo_transform_info = fluo_info

        elif mode == 3:
            # CLAHE (local contrast)
            bf, bf_info = self._apply_clahe(bf, image_type="bf")
            fluo, fluo_info = self._apply_clahe(fluo, image_type="fluo")
            self._bf_transform_info = bf_info
            self._fluo_transform_info = fluo_info

        else:
            # mode == 0: raw – leave arrays as they are, only display normalization in _to_pil
            self._bf_transform_info = {
                "mode": "raw",
                "low": None,
                "high": None,
                "gamma": None,
            }
            self._fluo_transform_info = {
                "mode": "raw",
                "low": None,
                "high": None,
                "gamma": None,
            }

        # Apply brightness factor *after* contrast
        bf = self._apply_brightness(bf)
        fluo = self._apply_brightness(fluo)

        bf_img = self._to_pil(bf)
        fluo_img = self._to_pil(fluo)

        # Optionally add scale bar
        if self.show_scaled.get():
            bf_img = self._add_scale_bar(bf_img, rec)
            fluo_img = self._add_scale_bar(fluo_img, rec)

        # Resize to fit into window: two images side-by-side in 1920px
        # Reserve ~380px for left panel, remaining ~1540px → ~770px per image
        target_width = 770
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

        cm_txt = {0: "Raw", 1: "Leica", 2: "Auto", 3: "CLAHE"}[self.contrast_mode.get()]
        self.bf_label.config(
            text=f"Brightfield - {rec.pair_name}{px_text} [{cm_txt}]"
        )
        self.fluo_label.config(
            text=f"Fluorescence - {rec.pair_name}{px_text} [{cm_txt}]"
        )

        # If diagnostic window is open, refresh it
        if self._diagnostic_window is not None and tk.Toplevel.winfo_exists(self._diagnostic_window):
            self._update_diagnostics_canvas()

    def on_exit(self):
        """Unified exit handler: release large objects and destroy the window."""
        # Clear references to images so they can be garbage-collected
        self._bf_tkimg = None
        self._fluo_tkimg = None
        self.destroy()

    # ------------------------------------------------------------------ Image helpers

    def _apply_leica_scaling(
        self, arr: np.ndarray, rec: ImagePairRecord, channel_idx: int
    ):
        """
        Apply Leica viewer-style contrast scaling using metadata:
        rec.bit_depth and rec.channel_scaling[channel_idx].
        Returns (float32 array in [0,1], info_dict).
        Safe no-op if metadata is missing.
        """
        cs_list = getattr(rec, "channel_scaling", None)
        bit_depth = getattr(rec, "bit_depth", None)

        info = {"mode": "leica", "low": None, "high": None, "gamma": None}

        if cs_list is None or not cs_list:
            return arr, info
        if bit_depth is None:
            return arr, info
        if channel_idx >= len(cs_list):
            return arr, info

        cs = cs_list[channel_idx]
        try:
            black_norm = float(cs.get("black_norm", 0.0))
            white_norm = float(cs.get("white_norm", 1.0))
            gamma = float(cs.get("gamma", 1.0))
        except (TypeError, ValueError):
            return arr, info

        if white_norm <= black_norm:
            return arr, info

        max_val = float(2**bit_depth - 1)
        black_raw = black_norm * max_val
        white_raw = white_norm * max_val

        arr_f = arr.astype(np.float32)
        arr_f = (arr_f - black_raw) / (white_raw - black_raw)
        arr_f = np.clip(arr_f, 0.0, 1.0)

        # Apply gamma if needed
        if gamma != 1.0:
            # Leica gamma is for display; use 1/gamma as usual
            arr_f = np.power(arr_f, 1.0 / gamma)

        info["low"] = black_raw
        info["high"] = white_raw
        info["gamma"] = gamma

        return arr_f, info

    def _auto_normalize(self, arr: np.ndarray):
        """
        Simple auto-contrast based on percentiles.
        Works on raw integer or float arrays.
        Returns (float32 in [0,1], info_dict).
        """
        arr_f = arr.astype(np.float32)

        finite_mask = np.isfinite(arr_f)
        info = {"mode": "auto", "low": None, "high": None, "gamma": None}

        if not np.any(finite_mask):
            return arr_f, info

        # Percentile thresholds; can tune e.g. 0.5 / 99.5
        low = np.percentile(arr_f[finite_mask], 1.0)
        high = np.percentile(arr_f[finite_mask], 99.0)

        if high <= low:
            # Fallback: simple min‑max
            low = float(arr_f[finite_mask].min())
            high = float(arr_f[finite_mask].max())
            if high <= low:
                return arr_f, info  # constant image

        arr_f = (arr_f - low) / (high - low)
        arr_f = np.clip(arr_f, 0.0, 1.0)

        info["low"] = low
        info["high"] = high

        return arr_f, info

    def _apply_clahe(self, arr: np.ndarray, image_type: str = "bf"):
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Returns (float32 in [0,1], info_dict).
        We record only that CLAHE was applied; 'low'/'high' are not global levels.
        """
        # Always treat as single channel
        if arr.ndim == 3 and arr.shape[2] > 1:
            # take first channel or mean across channels
            arr_mono = arr[..., 0]
        else:
            arr_mono = arr

        arr_mono = arr_mono.astype(np.float32)

        # Normalize to [0,1] based on min/max
        finite_mask = np.isfinite(arr_mono)
        info = {"mode": "clahe", "low": None, "high": None, "gamma": None}

        if not np.any(finite_mask):
            return arr_mono, info

        vmin = float(arr_mono[finite_mask].min())
        vmax = float(arr_mono[finite_mask].max())
        if vmax <= vmin:
            return arr_mono, info

        norm = (arr_mono - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0.0, 1.0)

        # Convert to 8‑bit for CLAHE
        img8 = (norm * 255.0).astype(np.uint8)

        # Get parameters
        try:
            clip = float(self.clahe_clip_limit.get())
        except Exception:
            clip = 2.0
        try:
            tile = int(self.clahe_tile_size.get())
        except Exception:
            tile = 64
        tile = max(8, tile)  # minimum tile size

        # --- Pure numpy CLAHE implementation (simplified) ----------------
        # Split image into tiles and apply local histogram equalization with clipping.
        h, w = img8.shape
        # Determine number of tiles
        n_tiles_y = max(1, h // tile)
        n_tiles_x = max(1, w // tile)

        # Actual tile size (may be slightly larger to cover full image)
        tile_h = int(np.ceil(h / n_tiles_y))
        tile_w = int(np.ceil(w / n_tiles_x))

        # Precompute LUTs for each tile
        out = np.zeros_like(img8, dtype=np.uint8)

        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                y0 = ty * tile_h
                y1 = min((ty + 1) * tile_h, h)
                x0 = tx * tile_w
                x1 = min((tx + 1) * tile_w, w)

                tile_region = img8[y0:y1, x0:x1]
                hist, _ = np.histogram(tile_region, bins=256, range=(0, 255))

                # Clip histogram
                if clip > 0:
                    max_count = clip * hist.mean()
                    excess = np.maximum(hist - max_count, 0)
                    hist = hist - excess
                    redist = int(excess.sum())
                    # Redistribute excess uniformly
                    hist += redist // 256
                    remainder = redist % 256
                    # Distribute remaining counts
                    if remainder > 0:
                        hist[:remainder] += 1

                cdf = hist.cumsum()
                if cdf[-1] == 0:
                    lut = np.arange(256, dtype=np.uint8)
                else:
                    cdf_norm = cdf / cdf[-1]
                    lut = np.floor(255 * cdf_norm).astype(np.uint8)

                out[y0:y1, x0:x1] = lut[tile_region]

        # Convert back to float [0,1]
        out_f = out.astype(np.float32) / 255.0

        # For BF images we want to enhance subtle particles; CLAHE does that locally.
        # We do not define a single global low/high; histogram is effectively "flattened".
        return out_f, info

    def _apply_brightness(self, arr: np.ndarray) -> np.ndarray:
        """
        Apply brightness factor to array.
        Works for both float [0,1] and integer arrays.
        """
        try:
            b = float(self.brightness_factor.get())
        except Exception:
            b = 1.0
        if b == 1.0:
            return arr

        if np.issubdtype(arr.dtype, np.integer):
            max_val = float(np.iinfo(arr.dtype).max)
            arr_f = arr.astype(np.float32) / max_val
            arr_f = np.clip(arr_f * b, 0.0, 1.0)
            arr_f *= max_val
            return arr_f.astype(arr.dtype)
        else:
            arr_f = arr.astype(np.float32)
            arr_f = np.clip(arr_f * b, 0.0, 1.0)
            return arr_f

    def _to_pil(self, arr: np.ndarray) -> Image.Image:
        """Convert a 2D or 3D numpy array to a PIL image, normalizing if needed."""
        # If image is 2D, convert to 8-bit gray
        if arr.ndim == 2:
            arr = arr.astype(np.float32)

            # If values are all <= 1.5, likely already in [0,1]
            if arr.max() <= 1.5:
                arr = arr * 255.0
            else:
                # Raw integer or float range
                maxv = arr.max()
                if maxv > 0:
                    arr = arr / maxv * 255.0

            arr8 = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr8, mode="L")

        # If image has channels
        if arr.ndim == 3:
            arr = arr.astype(np.float32)
            if arr.max() <= 1.5:
                arr = arr * 255.0
            else:
                maxv = arr.max()
                if maxv > 0:
                    arr = arr / maxv * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)

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

    # ------------------------------------------------------------------ Diagnostics (histogram overlay)

    def show_diagnostics(self):
        """Open or focus the diagnostic window (histogram + black/white levels)."""
        if self._diagnostic_window is not None and tk.Toplevel.winfo_exists(self._diagnostic_window):
            # Just bring it to front
            self._diagnostic_window.lift()
            self._update_diagnostics_canvas()
            return

        win = tk.Toplevel(self)
        win.title("Intensity diagnostics (histogram & levels)")
        win.geometry("900x450")
        self._diagnostic_window = win

        # Two subplots: BF (left) and FLUO (right)
        fig = Figure(figsize=(9, 4), dpi=100)
        self._diag_ax_bf = fig.add_subplot(1, 2, 1)
        self._diag_ax_fluo = fig.add_subplot(1, 2, 2)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        self._diag_canvas = canvas
        self._diag_fig = fig

        self._update_diagnostics_canvas()

        def on_close():
            self._diagnostic_window = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    def _update_diagnostics_canvas(self):
        """Plot histograms and overlay chosen black/white levels (if available)."""
        if self._diagnostic_window is None:
            return

        ax_bf = self._diag_ax_bf
        ax_fluo = self._diag_ax_fluo

        ax_bf.clear()
        ax_fluo.clear()

        # Plot BF histogram
        if self._bf_last_array_raw is not None:
            arr = self._bf_last_array_raw.astype(np.float32)
            mask = np.isfinite(arr)
            if np.any(mask):
                vals = arr[mask].ravel()
                ax_bf.hist(vals, bins=256, color="gray", alpha=0.7)
                ax_bf.set_title("BF raw histogram")
                ax_bf.set_xlabel("Intensity")
                ax_bf.set_ylabel("Count")

                # Overlay black/white lines for relevant modes
                info = self._bf_transform_info or {}
                low = info.get("low", None)
                high = info.get("high", None)
                mode = info.get("mode", "unknown")

                if low is not None:
                    ax_bf.axvline(low, color="red", linestyle="--", label=f"low={low:.1f}")
                if high is not None:
                    ax_bf.axvline(high, color="green", linestyle="--", label=f"high={high:.1f}")

                ax_bf.legend(loc="upper right")
                ax_bf.text(
                    0.02,
                    0.98,
                    f"mode={mode}",
                    transform=ax_bf.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
                )

        # Plot FLUO histogram
        if self._fluo_last_array_raw is not None:
            arr = self._fluo_last_array_raw.astype(np.float32)
            mask = np.isfinite(arr)
            if np.any(mask):
                vals = arr[mask].ravel()
                ax_fluo.hist(vals, bins=256, color="gray", alpha=0.7)
                ax_fluo.set_title("FLUO raw histogram")
                ax_fluo.set_xlabel("Intensity")
                ax_fluo.set_ylabel("Count")

                info = self._fluo_transform_info or {}
                low = info.get("low", None)
                high = info.get("high", None)
                mode = info.get("mode", "unknown")

                if low is not None:
                    ax_fluo.axvline(low, color="red", linestyle="--", label=f"low={low:.1f}")
                if high is not None:
                    ax_fluo.axvline(high, color="green", linestyle="--", label=f"high={high:.1f}")

                ax_fluo.legend(loc="upper right")
                ax_fluo.text(
                    0.02,
                    0.98,
                    f"mode={mode}",
                    transform=ax_fluo.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
                )

        self._diag_canvas.draw_idle()


if __name__ == "__main__":
    project_root = Path.cwd()
    source_root = project_root / "source"

    app = BFFluoViewer(source_root)
    app.mainloop()