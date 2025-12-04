from pathlib import Path
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser

import os
import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image, ImageTk, ImageDraw, ImageFont

from register_dataset import register_all_datasets, ImagePairRecord
from preprocess import (
    preprocess_fluo_for_seg,
    preprocess_bf_for_seg,  # kept for completeness
    segment_particles_from_fluo,
    extract_contours,
    measure_particles,
)

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


CONFIG_FILE = "contours_config.json"


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
        # Default scale bar length set to 5 µm
        self.bar_length_um = tk.DoubleVar(value=5.0)

        self.contrast_mode = tk.StringVar(value="seg-preproc-contours")

        # Tk image refs (four panes)
        self._bf_raw_tkimg: ImageTk.PhotoImage | None = None
        self._fluo_raw_tkimg: ImageTk.PhotoImage | None = None
        self._bf_seg_tkimg: ImageTk.PhotoImage | None = None
        self._fluo_seg_tkimg: ImageTk.PhotoImage | None = None

        # Caches
        self._seg_contour_cache: dict[tuple[str, str], list[np.ndarray]] = {}
        self._measurement_cache: dict[tuple[str, str], pd.DataFrame | None] = {}

        # Contour config
        self.min_area_px_var = tk.IntVar(value=20)
        self.min_contour_len_var = tk.IntVar(value=10)
        self.contour_width_var = tk.IntVar(value=1)
        self.contour_color_hex = tk.StringVar(value="#ff0000")

        # Progress bar state
        self._refresh_thread: threading.Thread | None = None
        self._refresh_running = False

        self._load_contour_config_from_file()

        # UI
        self._build_ui()
        self._populate_listbox()

        self.protocol("WM_DELETE_WINDOW", self.on_exit)

    # ------------------------------------------------------------------ Config I/O

    def _load_contour_config_from_file(self) -> None:
        cfg_path = Path(CONFIG_FILE)
        if not cfg_path.is_file():
            return
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read {CONFIG_FILE}: {e}")
            return

        try:
            if "min_area_px" in data:
                self.min_area_px_var.set(int(data["min_area_px"]))
            if "min_contour_len" in data:
                self.min_contour_len_var.set(int(data["min_contour_len"]))
            if "contour_width" in data:
                self.contour_width_var.set(int(data["contour_width"]))
            if "contour_color" in data:
                self.contour_color_hex.set(str(data["contour_color"]))
        except Exception as e:
            print(f"[WARN] Invalid values in {CONFIG_FILE}: {e}")

    def _save_contour_config_to_file(self) -> None:
        data = {
            "min_area_px": int(self.min_area_px_var.get()),
            "min_contour_len": int(self.min_contour_len_var.get()),
            "contour_width": int(self.contour_width_var.get()),
            "contour_color": self.contour_color_hex.get(),
        }
        try:
            with Path(CONFIG_FILE).open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Contours config", f"Saved to {CONFIG_FILE}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save {CONFIG_FILE}:\n{e}")

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        root_pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        root_pane.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(root_pane, width=380)
        root_pane.add(left_frame, weight=0)

        ttk.Label(left_frame, text="Image pairs").pack(anchor=tk.W, padx=5, pady=(5, 0))

        self.listbox = tk.Listbox(left_frame, height=18)
        self.listbox.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
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

        ttk.Label(controls, text="Mode:").pack(anchor=tk.W, pady=(6, 0))
        rb_frame = ttk.Frame(controls)
        rb_frame.pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            rb_frame,
            text="Raw only",
            value="auto",
            variable=self.contrast_mode,
            command=self.update_images,
        ).pack(anchor=tk.W)

        ttk.Radiobutton(
            rb_frame,
            text="Segmentation preproc + contours",
            value="seg-preproc-contours",
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

        # Progress
        progress_frame = ttk.Frame(left_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        self.progress_label = ttk.Label(progress_frame, text="", foreground="gray")
        self.progress_label.pack(anchor=tk.W)

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode="indeterminate",
            length=200,
        )
        self.progress_bar.pack(fill=tk.X, pady=(2, 0))
        self.progress_bar.stop()
        self.progress_bar.pack_forget()

        # Stats
        self.stats_label = ttk.Label(
            left_frame,
            text="",
            foreground="blue",
        )
        self.stats_label.pack(anchor=tk.W, padx=5, pady=(0, 5))

        # Contour config
        contour_frame = ttk.LabelFrame(left_frame, text="Contours configuration")
        contour_frame.pack(fill=tk.X, padx=5, pady=5)

        row1 = ttk.Frame(contour_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Min particle area (px):").pack(side=tk.LEFT)
        ttk.Spinbox(
            row1,
            from_=1,
            to=10000,
            width=6,
            textvariable=self.min_area_px_var,
            command=self._on_contour_config_changed,
        ).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(contour_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Min contour length:").pack(side=tk.LEFT)
        ttk.Spinbox(
            row2,
            from_=1,
            to=10000,
            width=6,
            textvariable=self.min_contour_len_var,
            command=self._on_contour_config_changed,
        ).pack(side=tk.LEFT, padx=4)

        row3 = ttk.Frame(contour_frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="Line width (px):").pack(side=tk.LEFT)
        ttk.Spinbox(
            row3,
            from_=1,
            to=20,
            width=4,
            textvariable=self.contour_width_var,
            command=self._on_contour_config_changed,
        ).pack(side=tk.LEFT, padx=4)

        row4 = ttk.Frame(contour_frame)
        row4.pack(fill=tk.X, pady=2)
        ttk.Label(row4, text="Color:").pack(side=tk.LEFT)

        self.color_preview = tk.Label(
            row4, width=3, relief=tk.SUNKEN, bg=self.contour_color_hex.get()
        )
        self.color_preview.pack(side=tk.LEFT, padx=4)

        ttk.Button(
            row4,
            text="Pick...",
            command=self._choose_contour_color,
        ).pack(side=tk.LEFT)

        ttk.Button(
            contour_frame,
            text="Save contours config",
            command=self._save_contour_config_to_file,
        ).pack(anchor=tk.W, pady=(6, 2))

        # Right: 2x2 panes
        right_frame = ttk.Frame(root_pane)
        root_pane.add(right_frame, weight=1)

        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)
        right_frame.columnconfigure(1, weight=1)

        # Top-left
        bf_raw_frame = ttk.Frame(right_frame)
        bf_raw_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.bf_label = ttk.Label(bf_raw_frame, text="Brightfield (raw)")
        self.bf_label.pack(anchor=tk.CENTER, pady=(5, 0))

        self.bf_raw_canvas = tk.Label(bf_raw_frame, bg="black")
        self.bf_raw_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Top-right
        fluo_raw_frame = ttk.Frame(right_frame)
        fluo_raw_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.fluo_label = ttk.Label(fluo_raw_frame, text="Fluorescence (raw)")
        self.fluo_label.pack(anchor=tk.CENTER, pady=(5, 0))

        self.fluo_raw_canvas = tk.Label(fluo_raw_frame, bg="black")
        self.fluo_raw_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bottom-left
        bf_seg_frame = ttk.Frame(right_frame)
        bf_seg_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.bf_seg_label = ttk.Label(bf_seg_frame, text="Brightfield + contours")
        self.bf_seg_label.pack(anchor=tk.CENTER, pady=(5, 0))

        self.bf_seg_canvas = tk.Label(bf_seg_frame, bg="black")
        self.bf_seg_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bottom-right
        fluo_seg_frame = ttk.Frame(right_frame)
        fluo_seg_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        self.fluo_seg_label = ttk.Label(fluo_seg_frame, text="Fluorescence + contours")
        self.fluo_seg_label.pack(anchor=tk.CENTER, pady=(5, 0))

        self.fluo_seg_canvas = tk.Label(fluo_seg_frame, bg="black")
        self.fluo_seg_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Redraw images when panes are resized
        for w in (
            self.bf_raw_canvas,
            self.fluo_raw_canvas,
            self.bf_seg_canvas,
            self.fluo_seg_canvas,
        ):
            w.bind("<Configure>", lambda e: self.update_images())

    def _populate_listbox(self) -> None:
        self.listbox.delete(0, tk.END)
        for idx, rec in enumerate(self.records):
            label = f"{rec.dataset} :: {rec.pair_name}"
            self.listbox.insert(tk.END, label)

            if getattr(rec, "pixel_size_um", None) is None:
                self.listbox.itemconfig(idx, foreground="red")
            else:
                self.listbox.itemconfig(idx, foreground="black")

        self.current_record = None

    # ------------------------------------------------------------------ Progress helpers

    def _start_progress(self, text: str = "Refreshing…") -> None:
        if self._refresh_running:
            return
        self._refresh_running = True
        self.progress_label.config(text=text)
        self.progress_bar.pack(fill=tk.X, pady=(2, 0))
        self.progress_bar.start(50)

    def _stop_progress(self) -> None:
        self._refresh_running = False
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.progress_label.config(text="")

    # ------------------------------------------------------------------ Contour config callbacks

    def _on_contour_config_changed(self) -> None:
        self._seg_contour_cache.clear()
        self._measurement_cache.clear()
        self.update_images()

    def _choose_contour_color(self) -> None:
        initial = self.contour_color_hex.get()
        color = colorchooser.askcolor(color=initial, title="Choose contour color")
        if color[1] is not None:
            self.contour_color_hex.set(color[1])
            self.color_preview.configure(bg=color[1])
            self._on_contour_config_changed()

    # ------------------------------------------------------------------ Callbacks

    def _rec_key(self, rec: ImagePairRecord) -> tuple[str, str]:
        return (str(rec.bf_path), str(rec.fluo_path))

    def on_select_pair(self, event=None) -> None:
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.current_record = self.records[idx]
        self.update_images()

    def _overlay_contours_on_gray(
        self,
        img01: np.ndarray,
        contours: list[np.ndarray],
        color=(255, 0, 0),
        line_width: int = 1,
    ) -> Image.Image:
        if img01.dtype != np.uint8:
            base = (np.clip(img01, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            base = img01

        pil = Image.fromarray(base, mode="L").convert("RGB")
        draw = ImageDraw.Draw(pil)

        for c in contours:
            pts = [(float(col), float(row)) for row, col in c]
            if len(pts) > 1:
                draw.line(pts, fill=color, width=line_width)

        return pil

    # ------------------------------------------------------------------ Pane size helper

    def _pane_sizes(self):
        # fallback minimal sizes so we have something meaningful at startup
        def size_of(widget):
            w = max(widget.winfo_width(), 200)
            h = max(widget.winfo_height(), 200)
            return w, h

        return {
            "bf_raw": size_of(self.bf_raw_canvas),
            "fluo_raw": size_of(self.fluo_raw_canvas),
            "bf_seg": size_of(self.bf_seg_canvas),
            "fluo_seg": size_of(self.fluo_seg_canvas),
        }

    # ------------------------------------------------------------------ Main refresh

    def update_images(self) -> None:
        if self.current_record is None:
            return
        if self._refresh_running:
            return

        rec: ImagePairRecord = self.current_record
        self._start_progress("Refreshing images…")

        def worker() -> None:
            try:
                result = self._update_images_worker(rec)
            except Exception as e:
                self.after(
                    0,
                    lambda: messagebox.showerror(
                        "Error", f"Error during refresh:\n{e}"
                    ),
                )
                result = None

            def finish() -> None:
                self._stop_progress()
                if result is not None:
                    (
                        bf_top_img,
                        fluo_top_img,
                        bf_bottom_img,
                        fluo_bottom_img,
                        label_text_bf,
                        label_text_fluo,
                        stats_text,
                    ) = result

                    self._bf_raw_tkimg = ImageTk.PhotoImage(bf_top_img)
                    self._fluo_raw_tkimg = ImageTk.PhotoImage(fluo_top_img)
                    self._bf_seg_tkimg = ImageTk.PhotoImage(bf_bottom_img)
                    self._fluo_seg_tkimg = ImageTk.PhotoImage(fluo_bottom_img)

                    self.bf_raw_canvas.configure(image=self._bf_raw_tkimg)
                    self.fluo_raw_canvas.configure(image=self._fluo_raw_tkimg)
                    self.bf_seg_canvas.configure(image=self._bf_seg_tkimg)
                    self.fluo_seg_canvas.configure(image=self._fluo_seg_tkimg)

                    self.bf_label.config(text=label_text_bf)
                    self.fluo_label.config(text=label_text_fluo)
                    self.stats_label.config(text=stats_text)

            self.after(0, finish)

        self._refresh_thread = threading.Thread(target=worker, daemon=True)
        self._refresh_thread.start()

    def _update_images_worker(
        self, rec: ImagePairRecord
    ) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, str, str, str]:
        bf_raw = tiff.imread(rec.bf_path)
        fluo_raw = tiff.imread(rec.fluo_path)

        mode = self.contrast_mode.get()

        bf_top_arr = bf_raw
        fluo_top_arr = fluo_raw
        bf_bottom_arr = bf_raw.copy()
        fluo_bottom_arr = fluo_raw.copy()

        stats_text = ""

        if mode == "seg-preproc-contours":
            fluo_proc = preprocess_fluo_for_seg(rec)

            key = self._rec_key(rec)
            if key in self._seg_contour_cache:
                contours = self._seg_contour_cache[key]
            else:
                min_area_px = int(self.min_area_px_var.get())
                min_len = int(self.min_contour_len_var.get())
                mask = segment_particles_from_fluo(
                    fluo_proc, min_area_px=min_area_px
                )
                contours = extract_contours(mask, min_length=min_len)
                self._seg_contour_cache[key] = contours

            if key in self._measurement_cache:
                df_meas = self._measurement_cache[key]
            else:
                min_area_px = int(self.min_area_px_var.get())
                df_meas, _ = measure_particles(rec, min_area_px=min_area_px)
                self._measurement_cache[key] = df_meas

            if isinstance(df_meas, pd.DataFrame) and not df_meas.empty:
                n_particles = len(df_meas)
                if "equivalent_diameter_um" in df_meas.columns:
                    med_d = float(df_meas["equivalent_diameter_um"].median())
                    mean_d = float(df_meas["equivalent_diameter_um"].mean())
                    stats_text = (
                        f"{n_particles} particles; "
                        f"median eq. diameter {med_d:.2f} µm; "
                        f"mean {mean_d:.2f} µm"
                    )
                else:
                    stats_text = f"{n_particles} particles (pixel size unknown)"
            else:
                stats_text = "No particles detected"

            bf_raw_pil = self._to_pil(bf_raw)
            fluo_raw_pil = self._to_pil(fluo_raw)

            bf_raw01 = np.array(bf_raw_pil.convert("L"), dtype=np.float32)
            fluo_raw01 = np.array(fluo_raw_pil.convert("L"), dtype=np.float32)

            if bf_raw01.max() > 0:
                bf_raw01 /= bf_raw01.max()
            if fluo_raw01.max() > 0:
                fluo_raw01 /= fluo_raw01.max()

            hex_color = self.contour_color_hex.get()
            try:
                rgb = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
            except Exception:
                rgb = (255, 0, 0)
            line_w = int(self.contour_width_var.get())

            bf_annot_pil = self._overlay_contours_on_gray(
                bf_raw01, contours, color=rgb, line_width=line_w
            )
            fluo_annot_pil = self._overlay_contours_on_gray(
                fluo_raw01, contours, color=rgb, line_width=line_w
            )

            bf_bottom_arr = np.array(bf_annot_pil)
            fluo_bottom_arr = np.array(fluo_annot_pil)
        else:
            stats_text = ""

        # Convert arrays to PIL
        bf_top_img = self._to_pil(bf_top_arr)
        fluo_top_img = self._to_pil(fluo_top_arr)
        bf_bottom_img = self._to_pil(bf_bottom_arr)
        fluo_bottom_img = self._to_pil(fluo_bottom_arr)

        # --- resize each image to fit its pane (current window size) ---
        pane_sizes = self._pane_sizes()

        bf_top_img = self._resize_to_box(bf_top_img, *pane_sizes["bf_raw"])
        fluo_top_img = self._resize_to_box(fluo_top_img, *pane_sizes["fluo_raw"])
        bf_bottom_img = self._resize_to_box(bf_bottom_img, *pane_sizes["bf_seg"])
        fluo_bottom_img = self._resize_to_box(
            fluo_bottom_img, *pane_sizes["fluo_seg"]
        )

        # Draw scale bar AFTER resizing
        if self.show_scaled.get():
            bf_top_img = self._add_scale_bar(bf_top_img, rec)
            fluo_top_img = self._add_scale_bar(fluo_top_img, rec)
            bf_bottom_img = self._add_scale_bar(bf_bottom_img, rec)
            fluo_bottom_img = self._add_scale_bar(fluo_bottom_img, rec)

        # Labels
        if rec.pixel_size_um is not None:
            px_text = f" (px: {rec.pixel_size_um:.3f} µm/pixel)"
        else:
            px_text = " (no metadata)"

        label_text_bf = f"Brightfield (raw) - {rec.pair_name}{px_text}"
        label_text_fluo = f"Fluorescence (raw) - {rec.pair_name}{px_text}"

        return (
            bf_top_img,
            fluo_top_img,
            bf_bottom_img,
            fluo_bottom_img,
            label_text_bf,
            label_text_fluo,
            stats_text,
        )

    def on_exit(self) -> None:
        self._bf_raw_tkimg = None
        self._fluo_raw_tkimg = None
        self._bf_seg_tkimg = None
        self._fluo_seg_tkimg = None
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
            low = float(arr_f[finite].min())
            high = float(arr_f[finite].max())
        return float(low), float(high)

    def _auto_normalize(self, arr: np.ndarray) -> np.ndarray:
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

        data_low, data_high = self._percentiles(arr, 1.0, 99.0)
        if leica_low > data_low or leica_high < data_high:
            return self._auto_normalize(arr)

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
        if arr.ndim == 2:
            if arr.dtype == np.uint8:
                return Image.fromarray(arr, mode="L")
            arr = arr.astype(np.float32)
            max_val = arr.max()
            if max_val > 0:
                arr = arr / max_val * 255.0
            arr8 = arr.astype(np.uint8)
            return Image.fromarray(arr8, mode="L")

        if arr.ndim == 3:
            if arr.dtype != np.uint8:
                arr = arr.astype(np.float32)
                max_val = arr.max()
                if max_val > 0:
                    arr = arr / max_val * 255.0
                arr = arr.astype(np.uint8)

            if arr.shape[2] == 1:
                arr = arr[..., 0]
                return Image.fromarray(arr, mode="L")

            if arr.shape[2] > 3:
                arr = arr[..., :3]

            return Image.fromarray(arr, mode="RGB")

        raise ValueError(f"Unsupported image shape: {arr.shape}")

    def _resize_to_box(self, img: Image.Image, max_w: int, max_h: int) -> Image.Image:
        w, h = img.size
        if w <= max_w and h <= max_h:
            return img
        scale = min(max_w / w, max_h / h)
        new_size = (int(w * scale), int(h * scale))
        return img.resize(new_size, RESAMPLE)

    # -------------------------------------------------------------- font helper
    def _get_scale_font(self, img_height: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """
        Return a font object for the scale bar label.
        Size is scaled with image height.
        Falls back to default bitmap font if TTF not available.
        """
        # font size ~ 2–3% of image height, capped between 10 and 24 px
        size = max(10, min(24, img_height // 35))

        # try some common TrueType fonts that contain "µ"
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",  # Windows
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "/Library/Fonts/Arial.ttf",  # macOS
        ]
        for fp in font_paths:
            if os.path.isfile(fp):
                try:
                    return ImageFont.truetype(fp, size=size)
                except Exception:
                    pass

        # fall back to default bitmap font (type: ImageFont.ImageFont)
        return ImageFont.load_default()

    def _add_scale_bar(self, img: Image.Image, rec: ImagePairRecord) -> Image.Image:
        if rec.pixel_size_um is None:
            return img

        try:
            bar_len_um = float(self.bar_length_um.get())
        except Exception:
            bar_len_um = 5.0

        if bar_len_um <= 0:
            return img

        px_size = rec.pixel_size_um
        bar_len_px = int(round(bar_len_um / px_size))
        if bar_len_px <= 0:
            return img

        draw = ImageDraw.Draw(img)
        w, h = img.size

        margin = max(15, h // 30)
        bar_height = max(3, h // 200)

        x0 = margin
        y0 = h - margin - bar_height
        x1 = x0 + bar_len_px
        y1 = y0 + bar_height

        if x1 > w - margin:
            bar_len_px = max(10, w - 2 * margin)
            bar_len_um = bar_len_px * px_size
            x1 = x0 + bar_len_px

        # white bar with black edge
        draw.rectangle([x0, y0, x1, y1], fill="white", outline="black")

        # ---- text label ----
        label = f"{bar_len_um:.0f} µm"

        font = self._get_scale_font(h)
        # Pillow >=10 removed getsize; use drawing context to get bbox
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        text_x = x0 + bar_len_px / 2
        text_y = y0 - text_h - 4  # above the bar

        # keep label within the image
        if text_x - text_w / 2 < margin:
            text_x = margin + text_w / 2
        if text_x + text_w / 2 > w - margin:
            text_x = w - margin - text_w / 2

        # draw black outline around white text for readability
        for dx, dy in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ]:
            draw.text(
                (text_x + dx, text_y + dy),
                label,
                font=font,
                fill="black",
                anchor="ms",
            )
        draw.text(
            (text_x, text_y),
            label,
            font=font,
            fill="white",
            anchor="ms",
        )

        return img


if __name__ == "__main__":
    project_root = Path.cwd()
    source_root = project_root / "source"

    app = BFFluoViewer(source_root)
    app.mainloop()