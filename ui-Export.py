from pathlib import Path
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser

import numpy as np
import tifffile as tiff
from PIL import Image, ImageTk, ImageDraw, ImageFont

from register_dataset import register_all_datasets, ImagePairRecord
from preprocess import (
    preprocess_fluo_for_seg,
    preprocess_bf_for_seg,
    segment_particles_from_fluo,
    extract_contours,
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

# ---------------------------------------------------------------------------
# Matplotlib for histograms
# ---------------------------------------------------------------------------
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------
from export_utils import export_current_view  # external export

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
        self.bar_length_um = tk.DoubleVar(value=5.0)

        # Mode
        self.contrast_mode = tk.StringVar(value="seg-preproc-contours")

        # Tk image refs
        self._bf_raw_tkimg: ImageTk.PhotoImage | None = None
        self._fluo_raw_tkimg: ImageTk.PhotoImage | None = None
        self._bf_seg_tkimg: ImageTk.PhotoImage | None = None
        self._fluo_seg_tkimg: ImageTk.PhotoImage | None = None

        # PIL images (for export / resizing)
        self._bf_top_pil: Image.Image | None = None
        self._fluo_top_pil: Image.Image | None = None
        self._bf_bottom_pil: Image.Image | None = None
        self._fluo_bottom_pil: Image.Image | None = None

        # Caches
        self._seg_contour_cache: dict[tuple[str, str], list[np.ndarray]] = {}
        self._stats_cache: dict[tuple[str, str], tuple[list[dict], set[int]]] = {}

        # Contour config
        self.min_area_px_var = tk.IntVar(value=20)
        self.min_contour_len_var = tk.IntVar(value=10)
        self.contour_width_var = tk.IntVar(value=1)
        self.contour_color_hex = tk.StringVar(value="#ff0000")

        self.middle_low_pct_var = tk.IntVar(value=30)
        self.middle_high_pct_var = tk.IntVar(value=70)
        self.middle_contour_color_hex = tk.StringVar(value="#ffff00")

        self.arrow_length_var = tk.IntVar(value=25)

        # Progress
        self._refresh_thread: threading.Thread | None = None
        self._refresh_running = False

        # Matplotlib objects
        self.histo_fig: Figure | None = None
        self.ax_pdf_intensity_per_area = None
        self.ax_pdf_total_intensity = None
        self.ax_pdf_area = None
        self.histo_canvas: FigureCanvasTkAgg | None = None

        self._load_contour_config_from_file()
        self._build_ui()
        self._populate_listbox()

        self.bind("<Configure>", self._on_root_configure)
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
            if "middle_low_pct" in data:
                self.middle_low_pct_var.set(int(data["middle_low_pct"]))
            if "middle_high_pct" in data:
                self.middle_high_pct_var.set(int(data["middle_high_pct"]))
            if "middle_contour_color" in data:
                self.middle_contour_color_hex.set(str(data["middle_contour_color"]))
            if "arrow_length_px" in data:
                self.arrow_length_var.set(int(data["arrow_length_px"]))
        except Exception as e:
            print(f"[WARN] Invalid values in {CONFIG_FILE}: {e}")

    def _save_contour_config_to_file(self) -> None:
        data = {
            "min_area_px": int(self.min_area_px_var.get()),
            "min_contour_len": int(self.min_contour_len_var.get()),
            "contour_width": int(self.contour_width_var.get()),
            "contour_color": self.contour_color_hex.get(),
            "middle_low_pct": int(self.middle_low_pct_var.get()),
            "middle_high_pct": int(self.middle_high_pct_var.get()),
            "middle_contour_color": self.middle_contour_color_hex.get(),
            "arrow_length_px": int(self.arrow_length_var.get()),
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

        # Image list
        ttk.Label(left_frame, text="Image pairs").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.listbox = tk.Listbox(left_frame, height=18)
        self.listbox.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        self.listbox.bind("<<ListboxSelect>>", self.on_select_pair)

        # Display options
        display_frame = ttk.LabelFrame(left_frame, text="Display options")
        display_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(
            display_frame,
            text="Show scale bar",
            variable=self.show_scaled,
            command=self.update_images,
        ).pack(anchor=tk.W)

        bar_frame = ttk.Frame(display_frame)
        bar_frame.pack(anchor=tk.W, pady=2)
        ttk.Label(bar_frame, text="Bar length (µm):").pack(side=tk.LEFT)
        ttk.Entry(bar_frame, width=6, textvariable=self.bar_length_um).pack(side=tk.LEFT)

        ttk.Label(display_frame, text="Mode:").pack(anchor=tk.W, pady=(6, 0))
        rb_frame = ttk.Frame(display_frame)
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

        # Actions
        actions_frame = ttk.LabelFrame(left_frame, text="Actions")
        actions_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            actions_frame,
            text="Refresh",
            command=self.update_images,
        ).pack(anchor=tk.W, pady=4)

        # Export uses external helper
        ttk.Button(
            actions_frame,
            text="Export",
            command=lambda: export_current_view(self),
        ).pack(anchor=tk.W, pady=4)

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

        # Contours configuration
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
        ttk.Label(row4, text="Main color:").pack(side=tk.LEFT)
        self.color_preview = tk.Label(
            row4, width=3, relief=tk.SUNKEN, bg=self.contour_color_hex.get()
        )
        self.color_preview.pack(side=tk.LEFT, padx=4)
        ttk.Button(row4, text="Pick...", command=self._choose_contour_color).pack(
            side=tk.LEFT
        )

        row5 = ttk.Frame(contour_frame)
        row5.pack(fill=tk.X, pady=2)
        ttk.Label(row5, text="Middle range from (% of rank):").pack(side=tk.LEFT)
        ttk.Spinbox(
            row5,
            from_=0,
            to=100,
            width=4,
            textvariable=self.middle_low_pct_var,
            command=self._on_middle_range_changed,
        ).pack(side=tk.LEFT, padx=4)

        row6 = ttk.Frame(contour_frame)
        row6.pack(fill=tk.X, pady=2)
        ttk.Label(row6, text="Middle range to   (% of rank):").pack(side=tk.LEFT)
        ttk.Spinbox(
            row6,
            from_=0,
            to=100,
            width=4,
            textvariable=self.middle_high_pct_var,
            command=self._on_middle_range_changed,
        ).pack(side=tk.LEFT, padx=4)

        row7 = ttk.Frame(contour_frame)
        row7.pack(fill=tk.X, pady=2)
        ttk.Label(row7, text="Middle-range color:").pack(side=tk.LEFT)
        self.middle_color_preview = tk.Label(
            row7, width=3, relief=tk.SUNKEN, bg=self.middle_contour_color_hex.get()
        )
        self.middle_color_preview.pack(side=tk.LEFT, padx=4)
        ttk.Button(row7, text="Pick...", command=self._choose_middle_contour_color).pack(
            side=tk.LEFT
        )

        row8 = ttk.Frame(contour_frame)
        row8.pack(fill=tk.X, pady=2)
        ttk.Label(row8, text="Arrow length (px):").pack(side=tk.LEFT)
        ttk.Spinbox(
            row8,
            from_=5,
            to=200,
            width=4,
            textvariable=self.arrow_length_var,
            command=self._on_contour_config_changed,
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            contour_frame,
            text="Save contours config",
            command=self._save_contour_config_to_file,
        ).pack(anchor=tk.W, pady=(6, 2))

        # Exit
        ttk.Button(
            left_frame,
            text="Exit",
            command=self.on_exit,
        ).pack(anchor=tk.W, padx=5, pady=8)

        # Right notebook
        right_notebook = ttk.Notebook(root_pane)
        root_pane.add(right_notebook, weight=1)

        # Images tab
        images_tab = ttk.Frame(right_notebook)
        right_notebook.add(images_tab, text="Images")

        images_tab.rowconfigure(0, weight=1)
        images_tab.rowconfigure(1, weight=1)
        images_tab.columnconfigure(0, weight=1)
        images_tab.columnconfigure(1, weight=1)

        bf_raw_frame = ttk.Frame(images_tab)
        bf_raw_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        bf_raw_frame.rowconfigure(1, weight=1)
        bf_raw_frame.columnconfigure(0, weight=1)

        self.bf_label = ttk.Label(bf_raw_frame, text="Brightfield (raw)")
        self.bf_label.grid(row=0, column=0, pady=(5, 0))

        self.bf_raw_canvas = tk.Label(bf_raw_frame, bg="white")
        self.bf_raw_canvas.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        fluo_raw_frame = ttk.Frame(images_tab)
        fluo_raw_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        fluo_raw_frame.rowconfigure(1, weight=1)
        fluo_raw_frame.columnconfigure(0, weight=1)

        self.fluo_label = ttk.Label(fluo_raw_frame, text="Fluorescence (raw)")
        self.fluo_label.grid(row=0, column=0, pady=(5, 0))

        self.fluo_raw_canvas = tk.Label(fluo_raw_frame, bg="white")
        self.fluo_raw_canvas.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        bf_seg_frame = ttk.Frame(images_tab)
        bf_seg_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        bf_seg_frame.rowconfigure(1, weight=1)
        bf_seg_frame.columnconfigure(0, weight=1)

        self.bf_seg_label = ttk.Label(
            bf_seg_frame, text="Brightfield (enhanced + contours)"
        )
        self.bf_seg_label.grid(row=0, column=0, pady=(5, 0))

        self.bf_seg_canvas = tk.Label(bf_seg_frame, bg="white")
        self.bf_seg_canvas.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        fluo_seg_frame = ttk.Frame(images_tab)
        fluo_seg_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        fluo_seg_frame.rowconfigure(1, weight=1)
        fluo_seg_frame.columnconfigure(0, weight=1)

        self.fluo_seg_label = ttk.Label(
            fluo_seg_frame, text="Fluorescence (enhanced + contours)"
        )
        self.fluo_seg_label.grid(row=0, column=0, pady=(5, 0))

        self.fluo_seg_canvas = tk.Label(fluo_seg_frame, bg="white")
        self.fluo_seg_canvas.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Statistics tab
        stats_tab = ttk.Frame(right_notebook)
        right_notebook.add(stats_tab, text="Statistics")

        stats_tab.rowconfigure(0, weight=1)
        stats_tab.rowconfigure(1, weight=1)
        stats_tab.columnconfigure(0, weight=1)

        table_frame = ttk.Frame(stats_tab)
        table_frame.grid(row=0, column=0, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        columns = (
            "idx",
            "area_px",
            "eq_diam_px",
            "total_intensity",
            "intensity_per_area",
        )
        self.stats_tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", selectmode="browse"
        )
        self.stats_tree.grid(row=0, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.stats_tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.stats_tree.configure(yscrollcommand=vsb.set)

        self.stats_tree.heading("idx", text="# (rank by Intensity / Area)")
        self.stats_tree.heading("area_px", text="Area (px)")
        self.stats_tree.heading("eq_diam_px", text="Eq. diameter (px)")
        self.stats_tree.heading("total_intensity", text="Total Intensity (Fluo)")
        self.stats_tree.heading("intensity_per_area", text="Intensity / Area")

        self.stats_tree.tag_configure("highlight_middle", background="#ffffcc")

        histo_frame = ttk.LabelFrame(stats_tab, text="Distributions (PDF)")
        histo_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        histo_frame.rowconfigure(0, weight=1)
        histo_frame.columnconfigure(0, weight=1)

        self.histo_fig = Figure(figsize=(6, 2.4), dpi=100)
        self.ax_pdf_intensity_per_area = self.histo_fig.add_subplot(1, 3, 1)
        self.ax_pdf_total_intensity = self.histo_fig.add_subplot(1, 3, 2)
        self.ax_pdf_area = self.histo_fig.add_subplot(1, 3, 3)

        self.histo_canvas = FigureCanvasTkAgg(self.histo_fig, master=histo_frame)
        self.histo_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _populate_listbox(self) -> None:
        self.listbox.delete(0, tk.END)
        for idx, rec in enumerate(self.records):
            label = f"{rec.dataset} :: {rec.pair_name}"
            self.listbox.insert(tk.END, label)
            if rec.pixel_size_um is None:
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
        self._stats_cache.clear()
        self.update_images()

    def _on_middle_range_changed(self) -> None:
        try:
            low = int(self.middle_low_pct_var.get())
            high = int(self.middle_high_pct_var.get())
        except Exception:
            low, high = 30, 70

        low = max(0, min(low, 100))
        high = max(0, min(high, 100))
        if low >= high:
            if low == 100:
                low, high = 90, 100
            else:
                low = max(0, high - 10)

        self.middle_low_pct_var.set(low)
        self.middle_high_pct_var.set(high)
        self._on_contour_config_changed()

    def _choose_contour_color(self) -> None:
        initial = self.contour_color_hex.get()
        color = colorchooser.askcolor(color=initial, title="Choose contour color")
        if color[1] is not None:
            self.contour_color_hex.set(color[1])
            self.color_preview.configure(bg=color[1])
            self._on_contour_config_changed()

    def _choose_middle_contour_color(self) -> None:
        initial = self.middle_contour_color_hex.get()
        color = colorchooser.askcolor(
            color=initial, title="Choose middle-range contour color"
        )
        if color[1] is not None:
            self.middle_contour_color_hex.set(color[1])
            self.middle_color_preview.configure(bg=color[1])
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

    def _on_root_configure(self, event=None) -> None:
        if (
            self._bf_top_pil is None
            or self._fluo_top_pil is None
            or self._bf_bottom_pil is None
            or self._fluo_bottom_pil is None
        ):
            return
        self._fit_and_show_all_panes()

    # ------------------------------------------------------------------ Middle-range computation

    def _compute_highlighted_idxs(self, stats_rows: list[dict]) -> set[int]:
        if not stats_rows:
            return set()
        low_pct = int(self.middle_low_pct_var.get())
        high_pct = int(self.middle_high_pct_var.get())
        low_pct = max(0, min(low_pct, 100))
        high_pct = max(0, min(high_pct, 100))
        if high_pct < low_pct:
            low_pct, high_pct = high_pct, low_pct
        n = len(stats_rows)
        start = int(np.floor(n * (low_pct / 100.0)))
        end = int(np.ceil(n * (high_pct / 100.0))) - 1
        start = max(0, min(start, n - 1))
        end = max(start, min(end, n - 1))
        return {stats_rows[i]["idx"] for i in range(start, end + 1)}

    # ------------------------------------------------------------------ Overlay / arrow drawing

    def _overlay_contours_on_gray(
        self,
        img01: np.ndarray,
        contours: list[np.ndarray],
        color=(255, 0, 0),
        middle_color=(255, 255, 0),
        line_width: int = 1,
        label_indices: list[int] | None = None,
        highlight_mask: np.ndarray | None = None,
    ) -> Image.Image:
        if img01.dtype != np.uint8:
            base = (np.clip(img01, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            base = img01

        pil = Image.fromarray(base).convert("RGB")
        draw = ImageDraw.Draw(pil)
        w, h = pil.size

        if highlight_mask is None:
            highlight_mask = np.zeros(len(contours), dtype=bool)
        else:
            highlight_mask = np.array(highlight_mask, dtype=bool)
            if highlight_mask.size != len(contours):
                highlight_mask = np.zeros(len(contours), dtype=bool)

        def _brighten(rgb, factor=1.3):
            r, g, b = rgb
            return (
                min(int(r * factor), 255),
                min(int(g * factor), 255),
                min(int(b * factor), 255),
            )

        main_color = _brighten(color)
        middle_color_b = _brighten(middle_color)
        font = self._get_scaled_font(base_size=12)

        centroids: list[tuple[float | None, float | None]] = []
        for c in contours:
            if c.shape[0] < 3:
                centroids.append((None, None))
                continue
            poly_x = c[:, 1].astype(float)
            poly_y = c[:, 0].astype(float)
            centroids.append((float(poly_x.mean()), float(poly_y.mean())))

        arrow_len = max(5, int(self.arrow_length_var.get()))

        for i, c in enumerate(contours):
            if c.shape[0] < 2:
                continue

            contour_color = middle_color_b if highlight_mask[i] else main_color
            pts = [(float(col), float(row)) for row, col in c]
            if len(pts) > 1:
                draw.line(pts, fill=contour_color, width=line_width)

            if label_indices is None or i >= len(label_indices):
                continue
            if c.shape[0] < 3:
                continue

            cx, cy = centroids[i]
            if cx is None or cy is None:
                continue

            left_count = right_count = up_count = down_count = 0
            for j, (ox, oy) in enumerate(centroids):
                if j == i or ox is None or oy is None:
                    continue
                if ox < cx:
                    left_count += 1
                elif ox > cx:
                    right_count += 1
                if oy < cy:
                    up_count += 1
                elif oy > cy:
                    down_count += 1

            scores: dict[str, int] = {}

            def add_dir(name: str, base_score: int, tx: float, ty: float) -> None:
                penalty = 0
                if tx < 10 or tx > w - 10 or ty < 10 or ty > h - 10:
                    penalty = 1000
                scores[name] = base_score + penalty

            add_dir("up", up_count, cx, cy - arrow_len)
            add_dir("down", down_count, cx, cy + arrow_len)
            add_dir("left", left_count, cx - arrow_len, cy)
            add_dir("right", right_count, cx + arrow_len, cy)

            best_dir = min(scores.keys(), key=lambda k: scores[k])

            if best_dir == "up":
                start = (cx, cy)
                end = (cx, max(cy - arrow_len, 0))
                label_pos = (cx, max(end[1] - 5, 0))
            elif best_dir == "down":
                start = (cx, cy)
                end = (cx, min(cy + arrow_len, h - 1))
                label_pos = (cx, min(end[1] + 10, h - 1))
            elif best_dir == "left":
                start = (cx, cy)
                end = (max(cx - arrow_len, 0), cy)
                label_pos = (max(end[0] - 5, 0), cy)
            else:
                start = (cx, cy)
                end = (min(cx + arrow_len, w - 1), cy)
                label_pos = (min(end[0] + 5, w - 1), cy)

            draw.line([start, end], fill=contour_color, width=1)
            draw.text(label_pos, str(label_indices[i]), fill=contour_color, font=font, anchor="ms")

        return pil

    # ------------------------------------------------------------------ Statistics and histograms

    def _measure_contours(
        self,
        contours: list[np.ndarray],
        bf_enh: np.ndarray,
        fluo_enh: np.ndarray,
    ) -> list[dict]:
        if bf_enh.ndim != 2:
            bf_enh_gray = bf_enh.mean(axis=-1)
        else:
            bf_enh_gray = bf_enh

        if fluo_enh.ndim != 2:
            fluo_enh_gray = fluo_enh.mean(axis=-1)
        else:
            fluo_enh_gray = fluo_enh

        h, w = bf_enh_gray.shape
        yy, xx = np.mgrid[0:h, 0:w]

        rows: list[dict] = []
        tmp_rows: list[dict] = []
        for local_idx, c in enumerate(contours):
            if c.shape[0] < 3:
                continue

            poly_x = c[:, 1].astype(float)
            poly_y = c[:, 0].astype(float)

            min_x = max(int(np.floor(poly_x.min())), 0)
            max_x = min(int(np.ceil(poly_x.max())), w - 1)
            min_y = max(int(np.floor(poly_y.min())), 0)
            max_y = min(int(np.ceil(poly_y.max())), h - 1)

            if min_x >= max_x or min_y >= max_y:
                continue

            bx = xx[min_y:max_y + 1, min_x:max_x + 1]
            by = yy[min_y:max_y + 1, min_x:max_x + 1]

            inside = np.zeros_like(bx, dtype=bool)
            x = bx.astype(float)
            y = by.astype(float)

            x0 = poly_x
            y0 = poly_y
            x1 = np.roll(poly_x, -1)
            y1 = np.roll(poly_y, -1)

            for xj0, yj0, xj1, yj1 in zip(x0, y0, x1, y1):
                cond = ((yj0 <= y) & (yj1 > y)) | ((yj1 <= y) & (yj0 > y))
                if not np.any(cond):
                    continue
                x_int = xj0 + (y - yj0) * (xj1 - xj0) / (yj1 - yj0 + 1e-12)
                inside ^= cond & (x < x_int)

            area = float(inside.sum())
            if area <= 0:
                continue

            eq_diam = float(np.sqrt(4.0 * area / np.pi))
            fluo_patch = fluo_enh_gray[min_y:max_y + 1, min_x:max_x + 1]
            total_intensity = float(fluo_patch[inside].sum()) if inside.any() else 0.0
            intensity_per_area = total_intensity / area if area > 0 else 0.0

            tmp_rows.append(
                dict(
                    contour_local_idx=local_idx,
                    area_px=area,
                    eq_diam_px=eq_diam,
                    total_intensity=total_intensity,
                    intensity_per_area=intensity_per_area,
                )
            )

        tmp_rows.sort(key=lambda r: r["intensity_per_area"], reverse=True)
        for rank, r in enumerate(tmp_rows, start=1):
            rows.append(
                dict(
                    idx=rank,
                    contour_local_idx=r["contour_local_idx"],
                    area_px=r["area_px"],
                    eq_diam_px=r["eq_diam_px"],
                    total_intensity=r["total_intensity"],
                    intensity_per_area=r["intensity_per_area"],
                )
            )
        return rows

    def _update_histograms(self, stats_rows: list[dict]) -> None:
        if (
            self.histo_fig is None
            or self.ax_pdf_intensity_per_area is None
            or self.ax_pdf_total_intensity is None
            or self.ax_pdf_area is None
            or self.histo_canvas is None
        ):
            return

        self.ax_pdf_intensity_per_area.clear()
        self.ax_pdf_total_intensity.clear()
        self.ax_pdf_area.clear()

        if not stats_rows:
            self.ax_pdf_intensity_per_area.set_title("Fluo / Area")
            self.ax_pdf_total_intensity.set_title("Total Fluo")
            self.ax_pdf_area.set_title("Area (px)")
            self.histo_canvas.draw_idle()
            return

        intensity_per_area = np.array(
            [r["intensity_per_area"] for r in stats_rows], dtype=float
        )
        total_intensity = np.array(
            [r["total_intensity"] for r in stats_rows], dtype=float
        )
        area_px = np.array([r["area_px"] for r in stats_rows], dtype=float)

        self.ax_pdf_intensity_per_area.hist(
            intensity_per_area, bins=30, density=True, color="tab:blue", alpha=0.7
        )
        self.ax_pdf_intensity_per_area.set_title("Fluo / Area")
        self.ax_pdf_intensity_per_area.set_xlabel("Intensity / Area")
        self.ax_pdf_intensity_per_area.set_ylabel("PDF")

        self.ax_pdf_total_intensity.hist(
            total_intensity, bins=30, density=True, color="tab:green", alpha=0.7
        )
        self.ax_pdf_total_intensity.set_title("Total Fluo")
        self.ax_pdf_total_intensity.set_xlabel("Total intensity")
        self.ax_pdf_total_intensity.set_ylabel("PDF")

        self.ax_pdf_area.hist(
            area_px, bins=30, density=True, color="tab:purple", alpha=0.7
        )
        self.ax_pdf_area.set_title("Area (px)")
        self.ax_pdf_area.set_xlabel("Area (px)")
        self.ax_pdf_area.set_ylabel("PDF")

        self.histo_fig.tight_layout()
        self.histo_canvas.draw_idle()

    def _update_stats_view(
        self,
        rec: ImagePairRecord,
        stats_rows: list[dict],
        highlighted_idx_set: set[int] | None = None,
    ) -> None:
        if highlighted_idx_set is None:
            highlighted_idx_set = set()

        for iid in self.stats_tree.get_children():
            self.stats_tree.delete(iid)

        for r in stats_rows:
            tags = ()
            if r["idx"] in highlighted_idx_set:
                tags = ("highlight_middle",)

            self.stats_tree.insert(
                "",
                tk.END,
                values=(
                    r["idx"],
                    f"{r['area_px']:.0f}",
                    f"{r['eq_diam_px']:.2f}",
                    f"{r['total_intensity']:.2f}",
                    f"{r['intensity_per_area']:.4f}",
                ),
                tags=tags,
            )

        self._update_histograms(stats_rows)

    # ------------------------------------------------------------------ Main refresh

    def update_images(self) -> None:
        if self.current_record is None:
            return
        if self._refresh_running:
            return

        rec: ImagePairRecord = self.current_record
        self._start_progress("Refreshing…")

        def worker():
            try:
                result = self._update_images_worker(rec)
            except Exception as e:
                err_msg = f"Error during refresh:\n{e}"
                self.after(0, lambda msg=err_msg: messagebox.showerror("Error", msg))
                result = None

            def finish():
                self._stop_progress()
                if result is None:
                    return
                (
                    bf_top_img,
                    fluo_top_img,
                    bf_bottom_img,
                    fluo_bottom_img,
                    label_text_bf,
                    label_text_fluo,
                    stats_rows,
                    highlighted_idx_set,
                ) = result

                self._bf_top_pil = bf_top_img
                self._fluo_top_pil = fluo_top_img
                self._bf_bottom_pil = bf_bottom_img
                self._fluo_bottom_pil = fluo_bottom_img

                self._fit_and_show_all_panes()
                self.bf_label.config(text=label_text_bf)
                self.fluo_label.config(text=label_text_fluo)
                self._update_stats_view(rec, stats_rows, highlighted_idx_set)

            self.after(0, finish)

        self._refresh_thread = threading.Thread(target=worker, daemon=True)
        self._refresh_thread.start()

    def _update_images_worker(self, rec: ImagePairRecord):
        bf_raw = tiff.imread(rec.bf_path)
        fluo_raw = tiff.imread(rec.fluo_path)
        mode = self.contrast_mode.get()

        bf_top_arr = bf_raw
        fluo_top_arr = fluo_raw

        stats_rows: list[dict] = []
        highlighted_idx_set: set[int] = set()

        if mode == "seg-preproc-contours":
            fluo_proc = preprocess_fluo_for_seg(rec)

            key = self._rec_key(rec)
            if key in self._seg_contour_cache:
                contours = self._seg_contour_cache[key]
            else:
                min_area_px = int(self.min_area_px_var.get())
                min_len = int(self.min_contour_len_var.get())
                mask = segment_particles_from_fluo(fluo_proc, min_area_px=min_area_px)
                contours = extract_contours(mask, min_length=min_len)
                self._seg_contour_cache[key] = contours

            bf_enh = preprocess_bf_for_seg(rec)
            fluo_enh = fluo_proc

            bf_enh_f = bf_enh.astype(np.float32)
            fluo_enh_f = fluo_enh.astype(np.float32)
            if bf_enh_f.max() > 0:
                bf_enh_f /= bf_enh_f.max()
            if fluo_enh_f.max() > 0:
                fluo_enh_f /= fluo_enh_f.max()

            stats_key = self._rec_key(rec)
            if stats_key in self._stats_cache:
                stats_rows, highlighted_idx_set = self._stats_cache[stats_key]
            else:
                stats_rows = self._measure_contours(contours, bf_enh, fluo_enh)
                highlighted_idx_set = self._compute_highlighted_idxs(stats_rows)
                self._stats_cache[stats_key] = (stats_rows, highlighted_idx_set)

            local_to_rank = {r["contour_local_idx"]: r["idx"] for r in stats_rows}

            highlight_mask = np.zeros(len(contours), dtype=bool)
            label_indices = [0] * len(contours)
            for local_idx, rank_idx in local_to_rank.items():
                label_indices[local_idx] = rank_idx
                if rank_idx in highlighted_idx_set:
                    highlight_mask[local_idx] = True

            hex_color = self.contour_color_hex.get()
            mid_hex_color = self.middle_contour_color_hex.get()
            try:
                rgb = tuple(int(hex_color[i: i + 2], 16) for i in (1, 3, 5))
            except Exception:
                rgb = (255, 0, 0)
            try:
                mid_rgb = tuple(int(mid_hex_color[i: i + 2], 16) for i in (1, 3, 5))
            except Exception:
                mid_rgb = (255, 255, 0)

            line_w = int(self.contour_width_var.get())

            bf_enh_pil = self._overlay_contours_on_gray(
                bf_enh_f,
                contours,
                color=rgb,
                middle_color=mid_rgb,
                line_width=line_w,
                label_indices=label_indices,
                highlight_mask=highlight_mask,
            )
            fluo_enh_pil = self._overlay_contours_on_gray(
                fluo_enh_f,
                contours,
                color=rgb,
                middle_color=mid_rgb,
                line_width=line_w,
                label_indices=label_indices,
                highlight_mask=highlight_mask,
            )

            bf_bottom_img = bf_enh_pil
            fluo_bottom_img = fluo_enh_pil

        else:
            bf_bottom_img = self._to_pil(bf_raw)
            fluo_bottom_img = self._to_pil(fluo_raw)
            stats_rows = []
            highlighted_idx_set = set()

        bf_top_img = self._to_pil(bf_top_arr)
        fluo_top_img = self._to_pil(fluo_top_arr)

        if self.show_scaled.get():
            bf_top_img = self._add_scale_bar(bf_top_img, rec)
            fluo_top_img = self._add_scale_bar(fluo_top_img, rec)
            bf_bottom_img = self._add_scale_bar(bf_bottom_img, rec)
            fluo_bottom_img = self._add_scale_bar(fluo_bottom_img, rec)

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
            stats_rows,
            highlighted_idx_set,
        )

    def _fit_and_show_all_panes(self) -> None:
        if self._bf_top_pil is not None:
            tw = max(self.bf_raw_canvas.winfo_width(), 1)
            th = max(self.bf_raw_canvas.winfo_height(), 1)
            img_fit = self._resize_to_box(self._bf_top_pil, tw, th)
            self._bf_raw_tkimg = ImageTk.PhotoImage(img_fit)
            self.bf_raw_canvas.configure(image=self._bf_raw_tkimg)

        if self._fluo_top_pil is not None:
            tw = max(self.fluo_raw_canvas.winfo_width(), 1)
            th = max(self.fluo_raw_canvas.winfo_height(), 1)
            img_fit = self._resize_to_box(self._fluo_top_pil, tw, th)
            self._fluo_raw_tkimg = ImageTk.PhotoImage(img_fit)
            self.fluo_raw_canvas.configure(image=self._fluo_raw_tkimg)

        if self._bf_bottom_pil is not None:
            tw = max(self.bf_seg_canvas.winfo_width(), 1)
            th = max(self.bf_seg_canvas.winfo_height(), 1)
            img_fit = self._resize_to_box(self._bf_bottom_pil, tw, th)
            self._bf_seg_tkimg = ImageTk.PhotoImage(img_fit)
            self.bf_seg_canvas.configure(image=self._bf_seg_tkimg)

        if self._fluo_bottom_pil is not None:
            tw = max(self.fluo_seg_canvas.winfo_width(), 1)
            th = max(self.fluo_seg_canvas.winfo_height(), 1)
            img_fit = self._resize_to_box(self._fluo_bottom_pil, tw, th)
            self._fluo_seg_tkimg = ImageTk.PhotoImage(img_fit)
            self.fluo_seg_canvas.configure(image=self._fluo_seg_tkimg)

    # ------------------------------------------------------------------ Image helpers and scale bar

    def _percentiles(
        self, arr: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0
    ) -> tuple[float, float]:
        arr_f = arr.astype(np.float32)
        finite = np.isfinite(arr_f)
        if not np.any(finite):
            return 0.0, 1.0
        low = float(np.percentile(arr_f[finite], low_pct))
        high = float(np.percentile(arr_f[finite], high_pct))
        if high <= low:
            low = float(arr_f[finite].min())
            high = float(arr_f[finite].max())
        return low, high

    def _auto_normalize(self, arr: np.ndarray) -> np.ndarray:
        low, high = self._percentiles(arr, 1.0, 99.0)
        arr_f = arr.astype(np.float32)
        if high <= low:
            return arr_f
        arr_f = (arr_f - low) / (high - low)
        arr_f = np.clip(arr_f, 0.0, 1.0)
        arr_f *= 255.0
        return arr_f.astype(np.uint8)

    def _to_pil(self, arr: np.ndarray) -> Image.Image:
        if arr.ndim == 2:
            arr = arr.astype(np.float32)
            if arr.max() > 0:
                arr = arr / arr.max() * 255.0
            return Image.fromarray(arr.astype(np.uint8), mode="L")

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

    def _resize_to_box(self, img: Image.Image, box_w: int, box_h: int) -> Image.Image:
        if box_w <= 0 or box_h <= 0:
            return img
        w, h = img.size
        scale = min(box_w / w, box_h / h)
        if scale <= 0:
            return img
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return img.resize(new_size, RESAMPLE)

    def _pick_contrasting_colors(self, bg_gray: float) -> str:
        return "black" if bg_gray > 128 else "white"

    def _get_scaled_font(self, base_size: int = 12):
        size = max(1, int(base_size * 2))
        try:
            return ImageFont.truetype("arial.ttf", size=size)
        except Exception:
            return ImageFont.load_default()

    def _add_scale_bar(self, img: Image.Image, rec: ImagePairRecord) -> Image.Image:
        if rec.pixel_size_um is None:
            return img

        try:
            bar_len_um = float(self.bar_length_um.get())
        except Exception:
            bar_len_um = 5.0

        px_size = rec.pixel_size_um
        bar_len_px = int(round(bar_len_um / px_size))
        if bar_len_px <= 0:
            return img

        w, h = img.size
        sample_height = max(5, h // 20)
        y_start = max(0, h - sample_height)
        bottom_band = img.crop((0, y_start, w, h))
        gray_band = bottom_band.convert("L")
        bg_gray = float(np.array(gray_band, dtype=np.uint8).mean())

        bar_color = self._pick_contrasting_colors(bg_gray)
        draw = ImageDraw.Draw(img)
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

        draw.rectangle([x0, y0, x1, y1], fill=bar_color, outline=bar_color)

        label = f"{bar_len_um:.0f} um"
        text_x = x0 + bar_len_px / 2
        text_y = y0 - 8
        font = self._get_scaled_font(base_size=12)
        draw.text((text_x, text_y), label, fill=bar_color, font=font, anchor="ms")

        return img

    # ------------------------------------------------------------------ Exit

    def on_exit(self) -> None:
        self._bf_raw_tkimg = None
        self._fluo_raw_tkimg = None
        self._bf_seg_tkimg = None
        self._fluo_seg_tkimg = None
        self.destroy()


if __name__ == "__main__":
    project_root = Path.cwd()
    source_root = project_root / "source"
    app = BFFluoViewer(source_root)
    app.mainloop()