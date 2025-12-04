from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd

from register_dataset import register_all_datasets, ImagePairRecord
from preprocess import measure_particles


class CounterApp(tk.Tk):
    def __init__(self, source_root: Path):
        super().__init__()

        self.title("Particle Counter / Summary")
        self.geometry("1200x700")

        # Load dataset registry
        self.records = register_all_datasets(source_root)
        if not self.records:
            messagebox.showerror("Error", "No image pairs found.")
            self.destroy()
            return

        # Cache: per pair measurements
        self._measurement_cache: dict[tuple[str, str], pd.DataFrame | None] = {}

        # Threading
        self._worker_thread: threading.Thread | None = None
        self._worker_running = False

        self._build_ui()

        self.protocol("WM_DELETE_WINDOW", self.on_exit)

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(root)
        top_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(top_frame, text="Source root:").pack(side=tk.LEFT)
        ttk.Label(
            top_frame,
            text=str(Path.cwd() / "source"),
            foreground="gray",
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            top_frame, text="Analyze all", command=self.run_analysis_all
        ).pack(side=tk.LEFT, padx=8)
        ttk.Button(
            top_frame, text="Export CSV", command=self.export_csv
        ).pack(side=tk.LEFT, padx=8)

        self.progress_label = ttk.Label(top_frame, text="", foreground="gray")
        self.progress_label.pack(side=tk.LEFT, padx=15)

        # Table
        columns = [
            "dataset",
            "pair_name",
            "n_particles",
            "median_diam_um",
            "mean_diam_um",
            "median_bf_mean",
            "median_fluo_mean",
        ]
        self.tree = ttk.Treeview(root, columns=columns, show="headings", height=20)
        for col in columns:
            self.tree.heading(col, text=col)
        # Set widths
        self.tree.column("dataset", width=120)
        self.tree.column("pair_name", width=160)
        self.tree.column("n_particles", width=100, anchor=tk.E)
        self.tree.column("median_diam_um", width=120, anchor=tk.E)
        self.tree.column("mean_diam_um", width=120, anchor=tk.E)
        self.tree.column("median_bf_mean", width=120, anchor=tk.E)
        self.tree.column("median_fluo_mean", width=120, anchor=tk.E)

        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Status bar
        self.status_var = tk.StringVar(value="")
        status_frame = ttk.Frame(root)
        status_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)

    # ------------------------------------------------------------------ Helpers

    def _rec_key(self, rec: ImagePairRecord) -> tuple[str, str]:
        return (str(rec.bf_path), str(rec.fluo_path))

    def _start_worker(self, text: str) -> None:
        if self._worker_running:
            return
        self._worker_running = True
        self.progress_label.config(text=text)

    def _stop_worker(self) -> None:
        self._worker_running = False
        self.progress_label.config(text="")

    @staticmethod
    def _fmt_float(value: float | None, fmt: str) -> str:
        """
        Safely format a value that is expected to be float-like (or None).
        """
        if value is None:
            return ""
        if np.isnan(value):
            return ""
        return fmt.format(value)

    @staticmethod
    def _safe_median_mean(series: pd.Series) -> tuple[float | None, float | None]:
        """
        Compute median and mean from a Series, returning floats or None.
        """
        if series.empty:
            return None, None
        try:
            med_val = float(series.median())
        except (TypeError, ValueError):
            med_val = None
        try:
            mean_val = float(series.mean())
        except (TypeError, ValueError):
            mean_val = None
        return med_val, mean_val

    # ------------------------------------------------------------------ Main analysis

    def run_analysis_all(self) -> None:
        if self._worker_running:
            return

        self._start_worker("Analyzing all image pairs…")

        def worker() -> None:
            results: list[dict[str, object]] = []
            n_total = len(self.records)
            for i, rec in enumerate(self.records):
                key = self._rec_key(rec)
                try:
                    if key in self._measurement_cache:
                        df = self._measurement_cache[key]
                    else:
                        df, _ = measure_particles(rec, min_area_px=20)
                        self._measurement_cache[key] = df
                except Exception as e:
                    df = None
                    self._measurement_cache[key] = None
                    print(f"[WARN] measure_particles failed for {rec.pair_name}: {e}")

                row = self._summarize_record(rec, df)
                results.append(row)

                # Update status for each record
                self.after(
                    0,
                    lambda i=i, rec=rec, n_total=n_total: self.status_var.set(
                        f"Processed {i+1}/{n_total}: {rec.pair_name}"
                    ),
                )

            def finish() -> None:
                self._stop_worker()
                self._fill_table(results)
                self.status_var.set(
                    f"Done. {len(results)} pairs analyzed."
                )

            self.after(0, finish)

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _summarize_record(
        self, rec: ImagePairRecord, df_meas: pd.DataFrame | None
    ) -> dict[str, object]:
        """
        Build a summary row for a single record.
        """
        dataset: str = rec.dataset
        pair_name: str = rec.pair_name

        n_particles: int = 0
        median_diam_um: float | None = None
        mean_diam_um: float | None = None
        median_bf_mean: float | None = None
        median_fluo_mean: float | None = None

        if isinstance(df_meas, pd.DataFrame) and not df_meas.empty:
            n_particles = len(df_meas)

            if "equivalent_diameter_um" in df_meas.columns:
                med, mean = self._safe_median_mean(df_meas["equivalent_diameter_um"])
                median_diam_um, mean_diam_um = med, mean
            elif "equivalent_diameter_px" in df_meas.columns:
                # if no pixel metadata, keep in px but still report something
                med, mean = self._safe_median_mean(df_meas["equivalent_diameter_px"])
                median_diam_um, mean_diam_um = med, mean

            if "bf_mean_intensity" in df_meas.columns:
                med, _ = self._safe_median_mean(df_meas["bf_mean_intensity"])
                median_bf_mean = med

            if "fluo_mean_intensity" in df_meas.columns:
                med, _ = self._safe_median_mean(df_meas["fluo_mean_intensity"])
                median_fluo_mean = med

        return {
            "dataset": dataset,
            "pair_name": pair_name,
            "n_particles": n_particles,
            "median_diam_um": median_diam_um,
            "mean_diam_um": mean_diam_um,
            "median_bf_mean": median_bf_mean,
            "median_fluo_mean": median_fluo_mean,
        }

    def _fill_table(self, rows: list[dict[str, object]]) -> None:
        # Clear existing rows
        for item in self.tree.get_children():
            self.tree.delete(item)

        for row in rows:
            median_diam = row.get("median_diam_um")
            mean_diam = row.get("mean_diam_um")
            med_bf = row.get("median_bf_mean")
            med_fluo = row.get("median_fluo_mean")

            values = [
                row["dataset"],
                row["pair_name"],
                row["n_particles"],
                self._fmt_float(median_diam if isinstance(median_diam, float) else None, "{:.2f}"),
                self._fmt_float(mean_diam if isinstance(mean_diam, float) else None, "{:.2f}"),
                self._fmt_float(med_bf if isinstance(med_bf, float) else None, "{:.3f}"),
                self._fmt_float(med_fluo if isinstance(med_fluo, float) else None, "{:.3f}"),
            ]
            self.tree.insert("", tk.END, values=values)

    # ------------------------------------------------------------------ Export

    def export_csv(self) -> None:
        """
        Export a long-form CSV with per-particle measurements for all records
        that have already been analyzed.
        """
        if not self._measurement_cache:
            messagebox.showwarning(
                "Export CSV", "No measurements available yet. Run 'Analyze all' first."
            )
            return

        rows: list[pd.DataFrame] = []
        for rec in self.records:
            key = self._rec_key(rec)
            df_meas = self._measurement_cache.get(key)
            if not isinstance(df_meas, pd.DataFrame) or df_meas.empty:
                continue

            df_tmp = df_meas.copy()
            df_tmp["dataset"] = rec.dataset
            df_tmp["pair_name"] = rec.pair_name
            rows.append(df_tmp)

        if not rows:
            messagebox.showwarning(
                "Export CSV", "No non-empty measurement tables to export."
            )
            return

        df_all = pd.concat(rows, ignore_index=True)

        out_path = Path("particle_measurements.csv")
        try:
            df_all.to_csv(out_path, index=False)
            messagebox.showinfo(
                "Export CSV", f"Exported per-particle measurements to {out_path}"
            )
        except Exception as e:
            messagebox.showerror(
                "Export CSV", f"Could not write {out_path}:\n{e}"
            )

    # ------------------------------------------------------------------ Exit

    def on_exit(self) -> None:
        self.destroy()


if __name__ == "__main__":
    project_root = Path.cwd()
    source_root = project_root / "source"

    app = CounterApp(source_root)
    app.mainloop()