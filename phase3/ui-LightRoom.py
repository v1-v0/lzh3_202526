from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import tifffile as tiff
from PIL import Image, ImageTk, ImageDraw

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

        self.title("BF/FLUO Viewer with Scale Bar")
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

        # Contrast mode: 'auto', 'leica', 'safe-leica'
        self.contrast_mode = tk.StringVar(value="safe-leica")

        # UI layout
        self._build_ui()
        self._populate_listbox()

        # Tk image refs
        self._bf_tkimg: ImageTk.PhotoImage | None = None
        self._fluo_tkimg: ImageTk.PhotoImage | None = None

        self.protocol("WM_DELETE_WINDOW", self.on_exit)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root_pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        root_pane.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(root_pane, width=350)
        root_pane.add(left_frame, weight=0)

        ttk.Label(left_frame, text="Image pairs").pack(anchor=tk.W, padx=5, pady=(5, 0))

        self.listbox = tk.Listbox(left_frame, height=30)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.listbox.bind("<<ListboxSelect>>", self.on_select_pair)

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

        # Contrast mode radio buttons
        ttk.Label(controls, text="Contrast mode:").pack(anchor=tk.W, pady=(6, 0))
        rb_frame = ttk.Frame(controls)
        rb_frame.pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            rb_frame,
            text="Auto (percentile)",
            value="auto",
            variable=self.contrast_mode,
            command=self.update_images,
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            rb_frame,
            text="Leica (raw metadata)",
            value="leica",
            variable=self.contrast_mode,
            command=self.update_images,
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            rb_frame,
            text="Safe Leica",
            value="safe-leica",
            variable=self.contrast_mode,
            command=self.update_images,
        ).pack(anchor=tk.W)

        ttk.Button(
            controls,
            text="Refresh",
            command=self.update_images,
        ).pack(anchor=tk.W, pady=4)

        ttk.Button(
            controls,
            text="Exit",
            command=self.on_exit,
        ).pack(anchor=tk.W, pady=8)

        # Right: image panes
        right_pane = ttk.Panedwindow(root_pane, orient=tk.HORIZONTAL)
        root_pane.add(right_pane, weight=1)

        bf_frame = ttk.Frame(right_pane)
        right_pane.add(bf_frame, weight=1)

        self.bf_label = ttk.Label(bf_frame, text="Brightfield")
        self.bf_label.pack(anchor=tk.CENTER, pady=(5, 0))

        self.bf_canvas = tk.Label(bf_frame, bg="black")
        self.bf_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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

            if getattr(rec, "pixel_size_um", None) is None:
                self.listbox.itemconfig(idx, foreground="red")
            else:
                self.listbox.itemconfig(idx, foreground="black")

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

        try:
            bf_raw = tiff.imread(rec.bf_path)
            fluo_raw = tiff.imread(rec.fluo_path)
        except Exception as e:
            messagebox.showerror("Error", f"Error loading images:\n{e}")
            return

        mode = self.contrast_mode.get()

        # Apply chosen contrast mode
        if mode == "auto":
            bf = self._auto_normalize(bf_raw)
            fluo = self._auto_normalize(fluo_raw)
        elif mode == "leica":
            bf = self._apply_leica_scaling(bf_raw, rec, channel_idx=0)
            fluo = self._apply_leica_scaling(fluo_raw, rec, channel_idx=1)
        elif mode == "safe-leica":
            bf = self._apply_safe_leica_scaling(bf_raw, rec, channel_idx=0)
            fluo = self._apply_safe_leica_scaling(fluo_raw, rec, channel_idx=1)
        else:
            bf, fluo = bf_raw, fluo_raw

        bf_img = self._to_pil(bf)
        fluo_img = self._to_pil(fluo)

        if self.show_scaled.get():
            bf_img = self._add_scale_bar(bf_img, rec)
            fluo_img = self._add_scale_bar(fluo_img, rec)

        target_width = 780
        bf_img = self._resize_to_width(bf_img, target_width)
        fluo_img = self._resize_to_width(fluo_img, target_width)

        self._bf_tkimg = ImageTk.PhotoImage(bf_img)
        self._fluo_tkimg = ImageTk.PhotoImage(fluo_img)

        self.bf_canvas.configure(image=self._bf_tkimg)
        self.fluo_canvas.configure(image=self._fluo_tkimg)

        if rec.pixel_size_um is not None:
            px_text = f" (px: {rec.pixel_size_um:.3f} µm/pixel)"
        else:
            px_text = " (no metadata)"

        self.bf_label.config(text=f"Brightfield - {rec.pair_name}{px_text}")
        self.fluo_label.config(text=f"Fluorescence - {rec.pair_name}{px_text}")

    def on_exit(self):
        self._bf_tkimg = None
        self._fluo_tkimg = None
        self.destroy()

    # ------------------------------------------------------------------ Contrast helpers

    def _percentiles(self, arr: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0):
        arr_f = arr.astype(np.float32)
        finite = np.isfinite(arr_f)
        if not np.any(finite):
            return 0.0, 1.0
        low = np.percentile(arr_f[finite], low_pct)
        high = np.percentile(arr_f[finite], high_pct)
        if high <= low:
            # Fallback to min/max if percentiles are degenerate
            low = float(arr_f[finite].min())
            high = float(arr_f[finite].max())
        return float(low), float(high)

    def _auto_normalize(self, arr: np.ndarray) -> np.ndarray:
        """Simple percentile-based auto contrast (1/99%)."""
        low, high = self._percentiles(arr, 1.0, 99.0)
        arr_f = arr.astype(np.float32)
        if high <= low:
            return arr_f
        arr_f = (arr_f - low) / (high - low)
        arr_f = np.clip(arr_f, 0.0, 1.0)
        arr_f *= 255.0
        return arr_f.astype(np.uint8)

    def _apply_safe_leica_scaling(
        self, arr: np.ndarray, rec: ImagePairRecord, channel_idx: int
    ) -> np.ndarray:
        """
        'Safe Leica' scaling:
        - use Leica black/white from ViewerScaling if they
          sensibly cover the data;
        - otherwise fall back to auto_percentile.
        """
        cs_list = getattr(rec, "channel_scaling", None)
        bit_depth = getattr(rec, "bit_depth", None)

        if not cs_list or bit_depth is None:
            return self._auto_normalize(arr)
        if channel_idx >= len(cs_list):
            return self._auto_normalize(arr)

        cs = cs_list[channel_idx]
        try:
            black_norm = float(cs.get("black_norm", 0.0))
            white_norm = float(cs.get("white_norm", 1.0))
            gamma = float(cs.get("gamma", 1.0))
        except (TypeError, ValueError):
            return self._auto_normalize(arr)

        if white_norm <= black_norm:
            return self._auto_normalize(arr)

        max_val = float(2**bit_depth - 1)
        leica_low = black_norm * max_val
        leica_high = white_norm * max_val

        # Compare with data percentiles
        data_low, data_high = self._percentiles(arr, 1.0, 99.0)

        # Heuristic: Leica low should not be above data_low,
        # and Leica high should not be below data_high.
        if leica_low > data_low or leica_high < data_high:
            # Metadata range is incompatible with actual histogram → auto
            return self._auto_normalize(arr)

        # If we get here, use Leica scaling
        arr_f = arr.astype(np.float32)
        arr_f = (arr_f - leica_low) / (leica_high - leica_low)
        arr_f = np.clip(arr_f, 0.0, 1.0)

        if gamma != 1.0:
            arr_f = np.power(arr_f, 1.0 / gamma)

        arr_f *= 255.0
        return arr_f.astype(np.uint8)

    def _apply_leica_scaling(
        self, arr: np.ndarray, rec: ImagePairRecord, channel_idx: int
    ) -> np.ndarray:
        """
        Apply Leica viewer-style contrast scaling using metadata:
        rec.bit_depth and rec.channel_scaling[channel_idx].
        No safety checks; this is the raw Leica mode.
        """
        cs_list = getattr(rec, "channel_scaling", None)
        bit_depth = getattr(rec, "bit_depth", None)

        if not cs_list or bit_depth is None:
            return arr
        if channel_idx >= len(cs_list):
            return arr

        cs = cs_list[channel_idx]
        try:
            black_norm = float(cs.get("black_norm", 0.0))
            white_norm = float(cs.get("white_norm", 1.0))
            gamma = float(cs.get("gamma", 1.0))
        except (TypeError, ValueError):
            return arr

        if white_norm <= black_norm:
            return arr

        max_val = float(2**bit_depth - 1)
        black_raw = black_norm * max_val
        white_raw = white_norm * max_val

        arr_f = arr.astype(np.float32)
        arr_f = (arr_f - black_raw) / (white_raw - black_raw)
        arr_f = np.clip(arr_f, 0.0, 1.0)

        if gamma != 1.0:
            arr_f = np.power(arr_f, 1.0 / gamma)

        arr_f *= 255.0
        return arr_f.astype(np.uint8)

    # ------------------------------------------------------------------ Image helpers

    def _to_pil(self, arr: np.ndarray) -> Image.Image:
        """Convert a 2D or 3D numpy array to a PIL image, normalizing to 8-bit."""
        if arr.ndim == 2:
            arr = arr.astype(np.float32)
            if arr.max() > 0:
                arr = arr / arr.max() * 255.0
            arr8 = arr.astype(np.uint8)
            return Image.fromarray(arr8, mode="L")

        if arr.ndim == 3:
            if arr.dtype != np.uint8:
                arr = arr.astype(np.float32)
                if arr.max() > 0:
                    arr = arr / arr.max() * 255.0
                arr = arr.astype(np.uint8)

            if arr.shape[2] == 1:
                arr = arr[..., 0]
                return Image.fromarray(arr, mode="L")

            if arr.shape[2] > 3:
                arr = arr[..., :3]

            return Image.fromarray(arr, mode="RGB")

        raise ValueError(f"Unsupported image shape: {arr.shape}")

    def _resize_to_width(self, img: Image.Image, target_w: int) -> Image.Image:
        w, h = img.size
        if w <= target_w:
            return img
        scale = target_w / w
        new_size = (target_w, int(h * scale))
        return img.resize(new_size, RESAMPLE)

    def _add_scale_bar(self, img: Image.Image, rec: ImagePairRecord) -> Image.Image:
        if rec.pixel_size_um is None:
            return img

        try:
            bar_len_um = float(self.bar_length_um.get())
        except Exception:
            bar_len_um = 20.0

        px_size = rec.pixel_size_um
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

        if x1 > w - margin:
            bar_len_px = max(10, w - 2 * margin)
            bar_len_um = bar_len_px * px_size
            x1 = x0 + bar_len_px

        draw.rectangle([x0, y0, x1, y1], fill="white")
        draw.rectangle([x0, y0, x1, y1], outline="black")

        label = f"{bar_len_um:.0f} µm"
        text_x = x0 + bar_len_px / 2
        text_y = y0 - 4

        draw.text((text_x, text_y), label, fill="white", anchor="ms")

        return img


if __name__ == "__main__":
    project_root = Path.cwd()
    source_root = project_root / "source"

    app = BFFluoViewer(source_root)
    app.mainloop()