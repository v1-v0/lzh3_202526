"""
run_profiler.py
Lightweight run profiler for the microgel fluorescence pipeline.

Outputs per run:
  <output_dir>/run_profile.json   — full structured detail
  <project_root>/runs_history.csv — one appended row per run (cross-run trends)
"""

from __future__ import annotations

import csv
import json
import time as pytime
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class RunProfiler:
    """Accumulates timing, decision, and quality statistics for one pipeline run."""

    def __init__(self, run_id: str, project_root: Path) -> None:
        self.run_id       = run_id
        self.project_root = project_root
        self.run_start    = pytime.monotonic()
        self.wall_start   = datetime.now().isoformat()

        # Ordered phase records  {name: {duration_s, ...metadata}}
        self.phases: dict[str, dict[str, Any]] = {}

        # Ordered event log  [{t_offset_s, event, ...}]
        self.events: list[dict[str, Any]] = []

        # Per-image records
        self.images: list[dict[str, Any]] = []

        # Top-level summary fields written to history CSV
        self.summary: dict[str, Any] = {
            "run_id":          run_id,
            "wall_start":      self.wall_start,
            "mode":            "",          # DEFAULT / DEBUG
            "processing_mode": "",          # multi_scan / single
            "dataset_mode":    "",          # batch / single
            "dataset_id":      "",
            "chosen_config":   "",
            "selection_rule":  "",
            "total_s":         0.0,
        }

    # ── Context-manager phase timing ───────────────────────────────────────

    @contextmanager
    def phase(self, name: str, **meta):
        """Time a named phase.  Nested phases are allowed.

        Usage::

            with profiler.phase("multi_scan_G+", bacteria_configs=3):
                ...
        """
        t0 = pytime.monotonic()
        self._log_event("phase_start", name=name, **meta)
        try:
            yield self
        finally:
            elapsed = round(pytime.monotonic() - t0, 3)
            self.phases[name] = {"duration_s": elapsed, **meta}
            self._log_event("phase_end", name=name, duration_s=elapsed)

    # ── Per-image recording ────────────────────────────────────────────────

    def record_image(
        self,
        *,
        image_name: str,
        group: str,
        bacteria_config: str,
        processing_time_s: float,
        accepted: int,
        rejected: int,
        alignment_method: str,   # "Raw+DoG averaged" | "DoG only" | "Raw only" | "none"
        shift_px: tuple[float, float],
        fluor_threshold_otsu: float,
    ) -> None:
        self.images.append(
            {
                "image_name":           image_name,
                "group":                group,
                "bacteria_config":      bacteria_config,
                "processing_time_s":    round(processing_time_s, 3),
                "accepted":             accepted,
                "rejected":             rejected,
                "rejection_rate":       round(
                    rejected / max(accepted + rejected, 1), 4
                ),
                "alignment_method":     alignment_method,
                "shift_y_px":           round(shift_px[0], 3),
                "shift_x_px":           round(shift_px[1], 3),
                "fluor_threshold_otsu": fluor_threshold_otsu,
            }
        )

    # ── Decision / milestone recording ────────────────────────────────────

    def record_decision(self, decision_type: str, **kwargs) -> None:
        """Record a branching decision point (config selection, classification, etc.)."""
        self._log_event("decision", decision_type=decision_type, **kwargs)

    def record_multi_scan_result(
        self,
        channel: str,           # "G+" or "G-"
        ranked_results: list[dict],
        stat_ambiguous: bool,
    ) -> None:
        self._log_event(
            "multi_scan_result",
            channel=channel,
            top_config=ranked_results[0]["config_key"] if ranked_results else None,
            top_confidence=ranked_results[0]["confidence"] if ranked_results else None,
            ambiguous=stat_ambiguous,
            all_configs=[
                {"key": r["config_key"], "confidence": r["confidence"]}
                for r in ranked_results
            ],
        )

    def record_final_classification(
        self, group: str, final_class: str, gp_class: str, gm_class: str
    ) -> None:
        self._log_event(
            "final_classification",
            group=group,
            final=final_class,
            g_plus=gp_class,
            g_minus=gm_class,
        )

    # ── Computed aggregate statistics ─────────────────────────────────────

    def compute_image_stats(self) -> dict[str, Any]:
        """Return aggregate metrics over all recorded images.

        Gap 1 additions vs original:
            std_processing_time_s    — spread of per-image runtimes
            median_processing_time_s — robust central tendency (less skewed by outliers)
            total_accepted           — overall particle yield (particle-count weighted)
            total_rejected           — overall particle rejection count
            overall_rejection_rate   — total_rejected / (total_accepted + total_rejected);
                                       more meaningful than the mean of per-image rates
                                       when particle counts vary across images
            per_group                — per-group breakdown of the above metrics so
                                       callers can identify which concentration group
                                       drives rejection or runtime cost
        """
        import statistics as _statistics

        if not self.images:
            return {}

        total     = len(self.images)
        times     = [i["processing_time_s"] for i in self.images]
        rej_rates = [i["rejection_rate"]    for i in self.images]
        aligned   = sum(1 for i in self.images if i["alignment_method"] != "none")

        # Particle totals (particle-count weighted, not image-count weighted)
        total_accepted = sum(i["accepted"] for i in self.images)
        total_rejected = sum(i["rejected"] for i in self.images)

        # Per-group breakdown
        groups: dict[str, list[dict[str, Any]]] = {}
        for img_record in self.images:
            g = img_record.get("group", "unknown")
            groups.setdefault(g, []).append(img_record)

        per_group: dict[str, dict[str, Any]] = {}
        for g, img_list in groups.items():
            g_times = [i["processing_time_s"] for i in img_list]
            g_acc   = sum(i["accepted"] for i in img_list)
            g_rej   = sum(i["rejected"] for i in img_list)
            g_n     = len(img_list)
            per_group[g] = {
                "n_images":               g_n,
                "total_accepted":         g_acc,
                "total_rejected":         g_rej,
                "overall_rejection_rate": round(
                    g_rej / max(g_acc + g_rej, 1), 4
                ),
                "mean_processing_time_s": round(
                    sum(g_times) / g_n, 3
                ),
                "std_processing_time_s":  round(
                    _statistics.stdev(g_times), 3
                ) if g_n > 1 else 0.0,
                "alignment_success_rate": round(
                    sum(1 for i in img_list if i["alignment_method"] != "none") / g_n, 4
                ),
            }

        return {
            # ── Per-image timing ───────────────────────────────────────────
            "total_images_processed":   total,
            "mean_processing_time_s":   round(sum(times) / total, 3),
            "std_processing_time_s":    round(
                _statistics.stdev(times), 3
            ) if total > 1 else 0.0,
            "median_processing_time_s": round(_statistics.median(times), 3),
            "max_processing_time_s":    round(max(times), 3),
            "min_processing_time_s":    round(min(times), 3),
            # ── Rejection rates ────────────────────────────────────────────
            "mean_rejection_rate":      round(sum(rej_rates) / total, 4),
            "total_accepted":           total_accepted,
            "total_rejected":           total_rejected,
            "overall_rejection_rate":   round(
                total_rejected / max(total_accepted + total_rejected, 1), 4
            ),
            # ── Alignment ─────────────────────────────────────────────────
            "alignment_success_rate":   round(aligned / total, 4),
            "alignment_method_counts":  {
                m: sum(1 for i in self.images if i["alignment_method"] == m)
                for m in ("Raw+DoG averaged", "DoG only", "Raw only", "none")
            },
            # ── Per-group detail (written to run_profile.json, not history CSV)
            "per_group":                per_group,
        }

    def compute_phase_breakdown(self) -> list[dict[str, Any]]:
        """Return phases sorted by duration descending — useful for spotting bottlenecks."""
        return sorted(
            [
                {"phase": k, "duration_s": v["duration_s"]}
                for k, v in self.phases.items()
            ],
            key=lambda x: x["duration_s"],
            reverse=True,
        )

    # ── Output ────────────────────────────────────────────────────────────

    def finalise(self, output_dir: Optional[Path] = None) -> dict[str, Any]:
        """Compute totals, write output files, return the full profile dict."""
        total_s = round(pytime.monotonic() - self.run_start, 3)
        self.summary["total_s"]  = total_s
        self.summary["wall_end"] = datetime.now().isoformat()

        image_stats  = self.compute_image_stats()
        phase_sorted = self.compute_phase_breakdown()

        profile = {
            "summary":         self.summary,
            "phase_timing":    self.phases,
            "phase_breakdown": phase_sorted,
            "image_stats":     image_stats,
            "events":          self.events,
            "images":          self.images,
        }

        if output_dir is not None and output_dir.exists():
            self._write_json(output_dir / "run_profile.json", profile)

        self._append_history_row(image_stats, total_s)
        self._print_summary(total_s, image_stats, phase_sorted)

        return profile

    # ── Internal helpers ──────────────────────────────────────────────────

    def _log_event(self, event_type: str, **kwargs) -> None:
        self.events.append(
            {
                "t_offset_s": round(pytime.monotonic() - self.run_start, 3),
                "event":      event_type,
                **kwargs,
            }
        )

    def _write_json(self, path: Path, data: dict) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"  ✓ Run profile: {path.name}")
        except Exception as exc:
            print(f"  ⚠ Could not write run profile: {exc}")


    def _append_history_row(
        self, image_stats: dict[str, Any], total_s: float
    ) -> None:
        """Append one row to runs_history.csv.

        Gap 2 fix (revised):
            The original code read the existing header and merged new keys to the
            end of all_fields, but only wrote the header on the very first run.
            On subsequent runs with a wider schema it appended rows with more
            values than the header had columns, silently breaking column alignment
            for every downstream reader.

            Fix strategy:
            1. Read the existing file as raw CSV rows (not DictReader) so that
                overflow values already present beyond the old header width are
                preserved rather than silently dropped.
            2. Compute the merged field list exactly as before.
            3. If the schema has grown (or the file is new), rewrite the entire
                file — header + all existing rows (with overflow values remapped
                to the new named columns) + the current row.  This self-heals a
                previously broken file on the very next pipeline run.
            4. If the schema is unchanged, do a cheap single-row append.

        Gap 1 fix:
            New image-quality metrics from compute_image_stats are included:
            total_accepted, total_rejected, std_img_time_s, median_img_time_s,
            overall_rejection_rate.

        The 'per_group' nested dict is intentionally excluded from the flat CSV
        row — it is available in run_profile.json.
        """
        history_path = self.project_root / "runs_history.csv"

        # ── Build the flat row for this run ───────────────────────────────────
        row: dict[str, Any] = {
            **self.summary,
            # Core image counts
            "total_images":           image_stats.get("total_images_processed", 0),
            "total_accepted":         image_stats.get("total_accepted", ""),
            "total_rejected":         image_stats.get("total_rejected", ""),
            # Timing
            "mean_img_time_s":        image_stats.get("mean_processing_time_s", ""),
            "std_img_time_s":         image_stats.get("std_processing_time_s", ""),
            "median_img_time_s":      image_stats.get("median_processing_time_s", ""),
            # Rejection rates (both metrics kept for backward compatibility)
            "mean_rejection_rate":    image_stats.get("mean_rejection_rate", ""),
            "overall_rejection_rate": image_stats.get("overall_rejection_rate", ""),
            # Alignment
            "alignment_success":      image_stats.get("alignment_success_rate", ""),
            # Per-phase durations (flattened)
            **{
                f"phase_{k.replace(' ', '_')}": v["duration_s"]
                for k, v in self.phases.items()
            },
        }
        # Explicitly exclude nested dict — not CSV-serialisable
        row.pop("per_group", None)

        # ── Read the existing file state ──────────────────────────────────────
        # Use raw csv.reader (not DictReader) so that overflow values already
        # present beyond the old header width are not silently discarded.
        file_exists = history_path.exists()
        existing_fieldnames: list[str] = []
        existing_raw_rows:   list[list[str]] = []

        if file_exists:
            try:
                with open(history_path, "r", newline="", encoding="utf-8") as f:
                    all_raw = list(csv.reader(f))
                if all_raw:
                    existing_fieldnames = all_raw[0]
                    existing_raw_rows   = all_raw[1:]
            except Exception:
                existing_fieldnames = []
                existing_raw_rows   = []

        write_header = not file_exists or not existing_fieldnames

        # ── Compute the merged field list ─────────────────────────────────────
        # Preserve existing column order; append any new keys at the end.
        all_fields: list[str] = list(existing_fieldnames)
        schema_changed = False
        for key in row:
            if key not in all_fields:
                all_fields.append(key)
                schema_changed = True

        # Fill gaps in the current row
        for field in all_fields:
            row.setdefault(field, "")

        # ── Write ─────────────────────────────────────────────────────────────
        try:
            if write_header or schema_changed:
                # New file or schema evolution: rewrite the entire file so the
                # header always reflects the full column set.
                #
                # Overflow values in existing rows (written by the old buggy code
                # beyond the old header width) are remapped here to their correct
                # named columns, in the same order they were appended.
                new_fields = [f for f in all_fields if f not in existing_fieldnames]

                with open(history_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=all_fields, extrasaction="ignore"
                    )
                    writer.writeheader()

                    for raw_row in existing_raw_rows:
                        row_dict: dict[str, str] = {}
                        # Map values that fall within the old header width
                        for i, field in enumerate(existing_fieldnames):
                            row_dict[field] = raw_row[i] if i < len(raw_row) else ""
                        # Remap overflow values to the newly named columns
                        overflow = raw_row[len(existing_fieldnames):]
                        for i, new_field in enumerate(new_fields):
                            row_dict[new_field] = overflow[i] if i < len(overflow) else ""
                        # Pad any remaining gaps
                        for field in all_fields:
                            row_dict.setdefault(field, "")
                        writer.writerow(row_dict)

                    writer.writerow(row)

            else:
                # Schema unchanged — safe single-row append
                with open(history_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=all_fields, extrasaction="ignore"
                    )
                    writer.writerow(row)

            print(f"  ✓ History appended: runs_history.csv")
        except Exception as exc:
            print(f"  ⚠ Could not update runs_history.csv: {exc}")


    @staticmethod
    def repair_history_csv(history_path: Path) -> None:
        """One-time explicit repair for a runs_history.csv with misaligned columns.

        Use this to fix the existing file immediately without running the pipeline.
        Note: the corrected _append_history_row also self-heals the file on the
        next pipeline run, so calling this is optional but recommended if you need
        the CSV to be correct before then.

        What it does:
        - Detects rows that are wider than the header (the Gap 2 / Gap 1 symptom).
        - Assigns the 5 known Gap-1 overflow columns their correct names in the
            header: total_accepted, total_rejected, std_img_time_s,
            median_img_time_s, overall_rejection_rate.
        - Any overflow beyond those 5 receives a placeholder name (_colN) for
            manual inspection.
        - Pads rows that are narrower than the new header with empty strings.
        - Writes a backup to <history_path>.bak before making any changes.

        Usage::

            RunProfiler.repair_history_csv(Path("runs_history.csv"))
        """
        import shutil

        # The 5 columns added by Gap 1, in the order _append_history_row appended
        # them to the row dict (and therefore to the CSV) under the old buggy code.
        GAP1_OVERFLOW_COLUMNS: list[str] = [
            "total_accepted",
            "total_rejected",
            "std_img_time_s",
            "median_img_time_s",
            "overall_rejection_rate",
        ]

        if not history_path.exists():
            print(f"  ⚠ File not found: {history_path}")
            return

        with open(history_path, "r", newline="", encoding="utf-8") as f:
            all_raw = list(csv.reader(f))

        if not all_raw:
            print("  ⚠ File is empty, nothing to repair.")
            return

        header    = all_raw[0]
        data_rows = all_raw[1:]
        header_width  = len(header)
        max_row_width = max((len(r) for r in data_rows), default=header_width)

        if max_row_width <= header_width:
            print(f"  ✓ No misalignment detected in {history_path.name}, no repair needed.")
            return

        overflow_count = max_row_width - header_width
        print(
            f"  ℹ {history_path.name}: header has {header_width} columns, "
            f"widest data row has {max_row_width} — {overflow_count} unnamed column(s) detected."
        )

        # Name the overflow columns: use known Gap-1 names first, then placeholders
        overflow_names: list[str] = []
        for i in range(overflow_count):
            if i < len(GAP1_OVERFLOW_COLUMNS):
                overflow_names.append(GAP1_OVERFLOW_COLUMNS[i])
            else:
                overflow_names.append(f"_col{header_width + i + 1}")

        new_header = header + overflow_names

        # Pad all data rows to the new full width
        repaired_rows = [
            raw_row + [""] * (max_row_width - len(raw_row))
            for raw_row in data_rows
        ]

        # Write backup before touching the original
        backup_path = history_path.with_suffix(".csv.bak")
        try:
            shutil.copy2(history_path, backup_path)
            print(f"  ✓ Backup written: {backup_path.name}")
        except Exception as exc:
            print(f"  ⚠ Could not write backup — aborting repair: {exc}")
            return

        # Write repaired file
        try:
            with open(history_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(new_header)
                writer.writerows(repaired_rows)

            placeholder_names = [n for n in overflow_names if n.startswith("_col")]
            print(
                f"  ✓ Repaired: {history_path.name}  "
                f"({len(repaired_rows)} data rows, {len(new_header)} columns)"
            )
            if placeholder_names:
                print(
                    f"  ⚠ {len(placeholder_names)} column(s) could not be named automatically "
                    f"and received placeholder names: {placeholder_names}\n"
                    f"    Inspect and rename these manually."
                )
        except Exception as exc:
            print(f"  ⚠ Could not write repaired file: {exc}")




    # FIX: dedented from inside _append_history_row to class-method scope
    def _print_summary(
        self,
        total_s: float,
        image_stats: dict[str, Any],
        phase_sorted: list[dict],
    ) -> None:
        print("\n" + "=" * 80)
        print("RUN PROFILE SUMMARY")
        print("=" * 80)
        print(f"  Run ID          : {self.run_id}")
        print(f"  Total runtime   : {total_s:.1f} s  "
              f"({total_s / 60:.1f} min)")
        print(f"  Dataset         : {self.summary['dataset_id']}")
        print(f"  Mode            : {self.summary['processing_mode']} / "
              f"{self.summary['dataset_mode']}")
        print(f"  Chosen config   : {self.summary['chosen_config']}")

        if image_stats:
            n   = image_stats["total_images_processed"]
            avg = image_stats["mean_processing_time_s"]
            rej = image_stats["mean_rejection_rate"]
            aln = image_stats["alignment_success_rate"]
            print(f"\n  Images processed: {n}")
            print(f"  Mean time/image : {avg:.2f} s")
            print(f"  Mean rejection  : {rej * 100:.1f}%")
            print(f"  Alignment found : {aln * 100:.1f}% of images")

        print(f"\n  Top-5 slowest phases:")
        for p in phase_sorted[:5]:
            bar_len = int(
                p["duration_s"] / max(phase_sorted[0]["duration_s"], 1) * 30
            )
            bar = "█" * bar_len
            print(f"    {p['phase']:<35} {p['duration_s']:7.1f} s  {bar}")
        print("=" * 80 + "\n")