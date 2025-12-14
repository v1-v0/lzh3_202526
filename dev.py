import cv2
import numpy as np
import sys
import csv

import atexit
import re
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Optional, Tuple, Any, cast
from tqdm import tqdm
import pandas as pd
from scipy import stats as scipy_stats

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage

# ==================================================
# Logging: tee stdout/stderr to a file
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


_project_root = Path(__file__).resolve().parent
_logs_dir = _project_root / "logs"
_logs_dir.mkdir(exist_ok=True)
_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_script_name = Path(__file__).stem
_log_path = _logs_dir / f"run_{_timestamp}_{_script_name}.txt"
_log_file = open(_log_path, "w", encoding="utf-8")

sys.stdout = Tee(sys.stdout, _log_file)
sys.stderr = Tee(sys.stderr, _log_file)
print(f"Saving output to: {_log_path}")


@atexit.register
def _close_log_file() -> None:
    try:
        _log_file.close()
    except Exception:
        pass


# ==================================================
# Configuration
# ==================================================
SOURCE_DIR = Path("./source")
CONTROL_DIR = SOURCE_DIR / "Control group"

# Segment only brightfield channel
IMAGE_GLOB = "*_ch00.tif"

OUTPUT_DIR = _project_root / "debug"
OUTPUT_DIR.mkdir(exist_ok=True)

# Scale bar parameters
SCALE_BAR_LENGTH_UM = 10
SCALE_BAR_HEIGHT = 4
SCALE_BAR_MARGIN = 15
SCALE_BAR_COLOR = (255, 255, 255)
SCALE_BAR_BG_COLOR = (0, 0, 0)
SCALE_BAR_TEXT_COLOR = (255, 255, 255)
SCALE_BAR_FONT_SCALE = 0.5
SCALE_BAR_FONT_THICKNESS = 1

# --- Segmentation ---
GAUSSIAN_SIGMA = 15
MORPH_KERNEL_SIZE = 3
MORPH_ITERATIONS = 1
DILATE_ITERATIONS = 1
ERODE_ITERATIONS = 1

# Filtering (in micrometers)
MIN_AREA_UM2 = 5.0
MAX_AREA_UM2 = 2000.0
MIN_CIRCULARITY = 0.0
MAX_FRACTION_OF_IMAGE_AREA = 0.25

# Debug options
CLEAR_OUTPUT_DIR_EACH_RUN = True
SEPARATE_OUTPUT_BY_GROUP = True
FALLBACK_UM_PER_PX: Optional[float] = 0.109492


# ==================================================
# Helpers
# ==================================================
def clear_output_dir(folder: Path) -> None:
    for p in folder.glob("*"):
        try:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                for q in p.rglob("*"):
                    try:
                        if q.is_file():
                            q.unlink()
                    except Exception:
                        pass
        except Exception:
            pass


def add_scale_bar(
    img: np.ndarray, pixel_size: float, unit: str = "um", length_um: float = 10
) -> np.ndarray:
    """Add a scale bar to the image"""
    if pixel_size is None or pixel_size <= 0:
        return img

    bar_length_px = int(round(length_um / pixel_size))
    if bar_length_px < 10:
        return img

    h, w = img.shape[:2]
    bar_x = w - bar_length_px - SCALE_BAR_MARGIN
    bar_y = h - SCALE_BAR_HEIGHT - SCALE_BAR_MARGIN

    # Check for overflow
    if bar_x < 0 or bar_y < 0:
        return img

    label = (
        f"{int(length_um)} um"
        if unit in ["µm", "um"]
        else f"{int(length_um)} {unit}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(
        label, font, SCALE_BAR_FONT_SCALE, SCALE_BAR_FONT_THICKNESS
    )

    text_x = bar_x + (bar_length_px - text_w) // 2
    text_y = bar_y - 8

    bg_padding = 5
    bg_x1 = min(bar_x, text_x) - bg_padding
    bg_y1 = text_y - text_h - bg_padding
    bg_x2 = max(bar_x + bar_length_px, text_x + text_w) + bg_padding
    bg_y2 = bar_y + SCALE_BAR_HEIGHT + bg_padding

    img = img.copy()
    overlay = img.copy()
    cv2.rectangle(
        overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), SCALE_BAR_BG_COLOR, -1
    )
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    cv2.rectangle(
        img,
        (bar_x, bar_y),
        (bar_x + bar_length_px, bar_y + SCALE_BAR_HEIGHT),
        SCALE_BAR_COLOR,
        -1,
    )

    cv2.putText(
        img,
        label,
        (text_x, text_y),
        font,
        SCALE_BAR_FONT_SCALE,
        SCALE_BAR_TEXT_COLOR,
        SCALE_BAR_FONT_THICKNESS,
        cv2.LINE_AA,
    )

    return img


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_debug(
    folder: Path,
    name: str,
    img: np.ndarray,
    pixel_size_um: Optional[float] = None,
) -> None:
    """Save debug image with optional scale bar"""
    out = folder / name
    img_to_save = img.copy()

    if pixel_size_um is not None and pixel_size_um > 0:
        img_to_save = add_scale_bar(
            img_to_save, float(pixel_size_um), "um", SCALE_BAR_LENGTH_UM
        )

    cv2.imwrite(str(out), img_to_save)


def list_sample_group_folders(source_dir: Path) -> list[Path]:
    groups: list[Path] = []
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir.resolve()}")

    for p in source_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name == "Control group":
            continue
        if re.fullmatch(r"\d+", p.name):
            groups.append(p)

    groups.sort(key=lambda x: int(x.name))
    return groups


def prompt_user_select_group(groups: list[Path]) -> Optional[Path]:
    """Return a selected group folder, or None if user chooses 'all'."""
    if not groups:
        return None

    print("\nSelect sample group folder to process:")
    print("  [0] ALL numeric groups")
    for i, g in enumerate(groups, 1):
        print(f"  [{i}] {g.name}")

    while True:
        s = input("Enter number (or 'q' to quit): ").strip().lower()
        if s in {"q", "quit", "exit"}:
            raise SystemExit(0)

        if not s.isdigit():
            print("Please enter a valid number.")
            continue

        idx = int(s)
        if idx == 0:
            return None
        if 1 <= idx <= len(groups):
            return groups[idx - 1]

        print("Out of range. Try again.")


def find_metadata_paths(img_path: Path) -> tuple[Optional[Path], Optional[Path]]:
    base = img_path.stem
    if base.endswith("_ch00"):
        base = base[:-5]
    md_dir = img_path.parent / "MetaData"
    xml_main = md_dir / f"{base}.xml"
    xml_props = md_dir / f"{base}_Properties.xml"

    return (
        xml_props if xml_props.exists() else None,
        xml_main if xml_main.exists() else None,
    )


def _require_attr(elem: ET.Element, attr: str, context: str) -> str:
    v = elem.get(attr)
    if v is None:
        raise ValueError(f"Missing attribute '{attr}' in {context}")
    return v


def _parse_float(s: str) -> float:
    return float(s.strip().replace(",", "."))


def get_pixel_size_um(
    xml_props_path: Optional[Path],
    xml_main_path: Optional[Path],
) -> Tuple[float, float]:
    if xml_props_path is not None:
        try:
            tree = ET.parse(xml_props_path)
            root = tree.getroot()

            dims = root.findall(".//ImageDescription/Dimensions/DimensionDescription")
            by_id = {d.get("DimID"): d for d in dims}

            def read_dim(dim_id: str) -> Tuple[float, int, str]:
                d = by_id.get(dim_id)
                if d is None:
                    raise ValueError(
                        f"Missing DimensionDescription with DimID='{dim_id}' in {xml_props_path.name}"
                    )

                length_s = _require_attr(d, "Length", f"{xml_props_path.name} DimID={dim_id}")
                n_s = _require_attr(d, "NumberOfElements", f"{xml_props_path.name} DimID={dim_id}")
                unit = _require_attr(d, "Unit", f"{xml_props_path.name} DimID={dim_id}")

                length = _parse_float(length_s)
                n = int(n_s)
                return length, n, unit

            x_len, x_n, x_unit = read_dim("X")
            y_len, y_n, y_unit = read_dim("Y")

            if x_unit != "µm" or y_unit != "µm":
                raise ValueError(
                    f"Unexpected units in {xml_props_path.name}: X={x_unit}, Y={y_unit}"
                )

            return float(x_len / x_n), float(y_len / y_n)

        except Exception as e:
            print(f"[WARN] Failed to read pixel size from {xml_props_path}: {e}")

    if xml_main_path is not None:
        try:
            tree = ET.parse(xml_main_path)
            root = tree.getroot()

            dims = root.findall(".//ImageDescription/Dimensions/DimensionDescription")
            by_id = {d.get("DimID"): d for d in dims}

            def read_dim(dim_id: str) -> Tuple[float, int, str]:
                d = by_id.get(dim_id)
                if d is None:
                    raise ValueError(
                        f"Missing DimensionDescription with DimID='{dim_id}' in {xml_main_path.name}"
                    )

                length_s = _require_attr(d, "Length", f"{xml_main_path.name} DimID={dim_id}")
                n_s = _require_attr(d, "NumberOfElements", f"{xml_main_path.name} DimID={dim_id}")
                unit = _require_attr(d, "Unit", f"{xml_main_path.name} DimID={dim_id}")

                length = _parse_float(length_s)
                n = int(n_s)
                return length, n, unit

            x_len_m, x_n, x_unit = read_dim("1")
            y_len_m, y_n, y_unit = read_dim("2")

            if x_unit != "m" or y_unit != "m":
                raise ValueError(
                    f"Unexpected units in {xml_main_path.name}: X={x_unit}, Y={y_unit}"
                )

            return float((x_len_m * 1e6) / x_n), float((y_len_m * 1e6) / y_n)

        except Exception as e:
            print(f"[WARN] Failed to read pixel size from {xml_main_path}: {e}")

    raise ValueError("Could not determine pixel size (µm/px). Missing/invalid metadata XML.")


def contour_perimeter_um(contour: np.ndarray, um_per_px_x: float, um_per_px_y: float) -> float:
    pts = contour.reshape(-1, 2).astype(np.float64)
    pts[:, 0] *= float(um_per_px_x)
    pts[:, 1] *= float(um_per_px_y)
    d = np.diff(np.vstack([pts, pts[0]]), axis=0)
    seg = np.sqrt((d[:, 0] ** 2) + (d[:, 1] ** 2))
    return float(seg.sum())


def equivalent_diameter_from_area(area: float) -> float:
    return float(2.0 * np.sqrt(area / np.pi)) if area > 0 else 0.0


def normalize_to_8bit(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        out = np.zeros_like(img, dtype=np.uint8)
        cv2.normalize(img, out, 0, 255, cv2.NORM_MINMAX)
        return out
    img_f = img.astype(np.float32)
    mn, mx = float(np.min(img_f)), float(np.max(img_f))
    if mx <= mn:
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img_f - mn) * (255.0 / (mx - mn))).clip(0, 255).astype(np.uint8)


def draw_object_ids(img_bgr: np.ndarray, contours: list[np.ndarray]) -> np.ndarray:
    out = img_bgr.copy()
    for i, c in enumerate(contours, 1):
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(
            out,
            str(i),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


# ==================================================
# Fluorescence measurement functions
# ==================================================
def measure_fluorescence_intensity(
    fluor_img: np.ndarray,
    contours: list[np.ndarray],
    um_per_px_x: float,
    um_per_px_y: float,
) -> list[dict]:
    """Measure fluorescence intensity for each contour using brightfield masks."""
    measurements: list[dict] = []

    # Force floats for Pylance (and safety)
    um_per_px_x_f = float(um_per_px_x)
    um_per_px_y_f = float(um_per_px_y)
    um2_per_px2 = um_per_px_x_f * um_per_px_y_f

    for i, c in enumerate(contours, 1):
        mask = np.zeros(fluor_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, thickness=-1)

        fluor_values = fluor_img[mask > 0]

        if len(fluor_values) > 0:
            area_px = float(cv2.contourArea(c))
            area_um2 = area_px * um2_per_px2

            measurements.append(
                {
                    "object_id": i,
                    "fluor_area_px": area_px,
                    "fluor_area_um2": area_um2,
                    "fluor_mean": float(np.mean(fluor_values)),
                    "fluor_median": float(np.median(fluor_values)),
                    "fluor_std": float(np.std(fluor_values)),
                    "fluor_min": float(np.min(fluor_values)),
                    "fluor_max": float(np.max(fluor_values)),
                    "fluor_integrated_density": float(np.sum(fluor_values)),
                }
            )
        else:
            measurements.append(
                {
                    "object_id": i,
                    "fluor_area_px": 0.0,
                    "fluor_area_um2": 0.0,
                    "fluor_mean": 0.0,
                    "fluor_median": 0.0,
                    "fluor_std": 0.0,
                    "fluor_min": 0.0,
                    "fluor_max": 0.0,
                    "fluor_integrated_density": 0.0,
                }
            )

    return measurements


def visualize_fluorescence_measurements(
    fluor_img8: np.ndarray, contours: list[np.ndarray], measurements: list[dict]
) -> np.ndarray:
    """Create visualization with contours and intensity labels"""
    vis = cv2.cvtColor(fluor_img8, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)

    for m in measurements:
        c = contours[m["object_id"] - 1]
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            label = f"{m['object_id']}: {m['fluor_mean']:.0f}"
            cv2.putText(
                vis,
                label,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    return vis


# ==================================================
# Segmentation
# ==================================================
def segment_particles_brightfield(img8: np.ndarray, pixel_size_um: float, out_dir: Path) -> np.ndarray:
    """Brightfield segmentation: dark objects on light background"""
    bg = cv2.GaussianBlur(img8, (0, 0), sigmaX=GAUSSIAN_SIGMA, sigmaY=GAUSSIAN_SIGMA)
    enhanced = cv2.subtract(bg, img8)
    enhanced_blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

    save_debug(out_dir, "02_enhanced.png", enhanced, pixel_size_um)
    save_debug(out_dir, "03_enhanced_blur.png", enhanced_blur, pixel_size_um)

    _, thresh = cv2.threshold(enhanced_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_debug(out_dir, "04_thresh_raw.png", thresh, pixel_size_um)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    bw = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)
    bw = cv2.dilate(bw, kernel, iterations=DILATE_ITERATIONS)
    bw = cv2.erode(bw, kernel, iterations=ERODE_ITERATIONS)
    save_debug(out_dir, "05_closed.png", bw, pixel_size_um)

    print(f"Mask white fraction (final): {float((bw > 0).mean()):.4f}")
    return bw


# ==================================================
# Excel consolidation with enhanced statistics
# ==================================================
def consolidate_to_excel(output_dir: Path, group_name: str) -> None:
    """Consolidate all CSVs in a group folder into one Excel workbook with statistics and color coding"""
    csv_files = list(output_dir.glob("*/object_stats.csv"))

    if not csv_files:
        print(f"[WARN] No CSV files found in {output_dir}")
        return

    excel_path = output_dir / f"{group_name}_consolidated.xlsx"

    # Try to delete existing file if it exists
    if excel_path.exists():
        try:
            excel_path.unlink()
        except PermissionError:
            print(f"[ERROR] Cannot overwrite {excel_path} - file is open in another program")
            print("        Please close the file and run again, or delete it manually")
            return

    try:
        from openpyxl.styles import PatternFill, Font, Alignment

        # Colors for percentile ranges
        green_fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
        red_fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")
        yellow_fill = PatternFill(start_color="FFFFE0", end_color="FFFFE0", fill_type="solid")
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")
        center_align = Alignment(horizontal="center", vertical="center")

        all_typical_particles = []

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # README sheet
            readme_df = pd.DataFrame(
                {
                    "Column Name": [
                        "Object_ID",
                        "BF_Area_px",
                        "BF_Area_um2",
                        "Perimeter_px",
                        "Perimeter_um",
                        "EquivDiameter_px",
                        "EquivDiameter_um",
                        "Circularity",
                        "AspectRatio",
                        "CentroidX_px",
                        "CentroidY_px",
                        "CentroidX_um",
                        "CentroidY_um",
                        "BBoxX_px",
                        "BBoxY_px",
                        "BBoxW_px",
                        "BBoxH_px",
                        "BBoxW_um",
                        "BBoxH_um",
                        "Fluor_Area_px",
                        "Fluor_Area_um2",
                        "Fluor_Mean",
                        "Fluor_Median",
                        "Fluor_Std",
                        "Fluor_Min",
                        "Fluor_Max",
                        "Fluor_IntegratedDensity",
                        "Fluor_Density_per_BF_Area",
                        "BF_to_Fluor_Area_Ratio",
                    ],
                    "Description": [
                        "Unique particle identifier (format: GroupID_ImageSequence_ParticleID, e.g., 12_2_5)",
                        "Brightfield particle area (pixels²)",
                        "Brightfield particle area (µm²)",
                        "Particle perimeter (pixels)",
                        "Particle perimeter (µm)",
                        "Diameter of equivalent circle (pixels)",
                        "Diameter of equivalent circle (µm)",
                        "Shape roundness: 4π×Area/Perimeter² (0-1, 1=perfect circle)",
                        "Bounding box width/height ratio",
                        "Particle center X-coordinate (pixels)",
                        "Particle center Y-coordinate (pixels)",
                        "Particle center X-coordinate (µm)",
                        "Particle center Y-coordinate (µm)",
                        "Bounding box top-left X (pixels)",
                        "Bounding box top-left Y (pixels)",
                        "Bounding box width (pixels)",
                        "Bounding box height (pixels)",
                        "Bounding box width (µm)",
                        "Bounding box height (µm)",
                        "Fluorescent region area (pixels²)",
                        "Fluorescent region area (µm²)",
                        "Average fluorescence intensity",
                        "Median fluorescence intensity",
                        "Standard deviation of fluorescence",
                        "Minimum fluorescence value",
                        "Maximum fluorescence value",
                        "Total fluorescence signal (sum of all pixel values)",
                        "Fluorescence density normalized by brightfield area (IntegratedDensity/BF_Area_um2) - PRIMARY METRIC",
                        "Ratio of brightfield area to fluorescence area (BF_Area_um2/Fluor_Area_um2)",
                    ],
                    "Unit": [
                        "String",
                        "pixels²",
                        "µm²",
                        "pixels",
                        "µm",
                        "pixels",
                        "µm",
                        "dimensionless",
                        "dimensionless",
                        "pixels",
                        "pixels",
                        "µm",
                        "µm",
                        "pixels",
                        "pixels",
                        "pixels",
                        "pixels",
                        "µm",
                        "µm",
                        "pixels²",
                        "µm²",
                        "a.u.",
                        "a.u.",
                        "a.u.",
                        "a.u.",
                        "a.u.",
                        "a.u.",
                        "a.u./µm²",
                        "dimensionless",
                    ],
                }
            )

            readme_df.to_excel(writer, sheet_name="README", index=False)
            ws_readme = writer.sheets["README"]

            # Format README
            for cell in ws_readme[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = center_align

            ws_readme.column_dimensions["A"].width = 30
            ws_readme.column_dimensions["B"].width = 80
            ws_readme.column_dimensions["C"].width = 15

            # Process each image
            for csv_file in sorted(csv_files):
                image_name = csv_file.parent.name
                df = pd.read_csv(csv_file)

                # Rename Area_px to BF_Area_px and Area_um2 to BF_Area_um2 (if old CSVs exist)
                df.rename(columns={"Area_px": "BF_Area_px", "Area_um2": "BF_Area_um2"}, inplace=True)

                # Calculate ratios (guard against missing columns)
                if "Fluor_IntegratedDensity" in df.columns and "BF_Area_um2" in df.columns:
                    df["Fluor_Density_per_BF_Area"] = (
                        df["Fluor_IntegratedDensity"].astype(float) / df["BF_Area_um2"].astype(float)
                    )
                else:
                    df["Fluor_Density_per_BF_Area"] = 0.0

                if "Fluor_Area_um2" in df.columns and "BF_Area_um2" in df.columns:
                    df["BF_to_Fluor_Area_Ratio"] = (
                        df["BF_Area_um2"].astype(float) / df["Fluor_Area_um2"].astype(float)
                    )
                else:
                    df["BF_to_Fluor_Area_Ratio"] = 0.0

                # Replace inf and NaN with 0
                df["Fluor_Density_per_BF_Area"] = df["Fluor_Density_per_BF_Area"].replace(
                    [np.inf, -np.inf], 0
                )
                df["BF_to_Fluor_Area_Ratio"] = df["BF_to_Fluor_Area_Ratio"].replace(
                    [np.inf, -np.inf], 0
                )
                df = df.fillna(0)

                # Sort by primary metric descending
                df_sorted = df.sort_values("Fluor_Density_per_BF_Area", ascending=False).reset_index(drop=True)

                # Calculate percentiles for color coding
                n_rows = len(df_sorted)
                top_20_threshold = int(np.ceil(n_rows * 0.2))
                bottom_20_threshold = int(np.floor(n_rows * 0.8))

                # Extract typical particles (middle 60%)
                typical_particles = df_sorted.iloc[top_20_threshold:bottom_20_threshold].copy()
                typical_particles["Source_Image"] = image_name
                all_typical_particles.append(typical_particles)

                # Write main sheet
                sheet_name = image_name[:31]
                df_sorted.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]

                # Format headers
                for cell in ws[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = center_align

                # Enable AutoFilter on first row
                ws.auto_filter.ref = ws.dimensions

                # Format data columns by type
                for col_idx, col_name in enumerate(df_sorted.columns, start=1):
                    if col_name in ["Object_ID", "Source_Image"]:
                        # Text columns - left align
                        for row in range(2, len(df_sorted) + 2):
                            ws.cell(row=row, column=col_idx).alignment = Alignment(horizontal="left")

                    elif (
                        "_px" in col_name
                        or col_name
                        in [
                            "BBoxX_px",
                            "BBoxY_px",
                            "BBoxW_px",
                            "BBoxH_px",
                            "CentroidX_px",
                            "CentroidY_px",
                            "Perimeter_px",
                            "EquivDiameter_px",
                            "Fluor_Area_px",
                            "BF_Area_px",
                        ]
                    ):
                        # Integer pixel values - no decimals
                        for row in range(2, len(df_sorted) + 2):
                            ws.cell(row=row, column=col_idx).number_format = "0"
                            ws.cell(row=row, column=col_idx).alignment = Alignment(horizontal="right")

                    elif "_um" in col_name or col_name.endswith("_um2"):
                        # Micrometer values - 2 decimals
                        for row in range(2, len(df_sorted) + 2):
                            ws.cell(row=row, column=col_idx).number_format = "0.00"
                            ws.cell(row=row, column=col_idx).alignment = Alignment(horizontal="right")

                    elif col_name in ["Circularity", "AspectRatio", "BF_to_Fluor_Area_Ratio"]:
                        # Ratios - 4 decimals
                        for row in range(2, len(df_sorted) + 2):
                            ws.cell(row=row, column=col_idx).number_format = "0.0000"
                            ws.cell(row=row, column=col_idx).alignment = Alignment(horizontal="right")

                    elif col_name in ["Fluor_Mean", "Fluor_Median", "Fluor_Std", "Fluor_Min", "Fluor_Max"]:
                        # Fluorescence intensity - 2 decimals
                        for row in range(2, len(df_sorted) + 2):
                            ws.cell(row=row, column=col_idx).number_format = "0.00"
                            ws.cell(row=row, column=col_idx).alignment = Alignment(horizontal="right")

                    elif col_name in ["Fluor_IntegratedDensity", "Fluor_Density_per_BF_Area"]:
                        # Large values - 2 decimals
                        for row in range(2, len(df_sorted) + 2):
                            ws.cell(row=row, column=col_idx).number_format = "0.00"
                            ws.cell(row=row, column=col_idx).alignment = Alignment(horizontal="right")

                # Auto-adjust column widths
                for column in ws.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    col_name_val = column[0].value
                    col_name = str(col_name_val) if col_name_val is not None else ""

                    for cell in column:
                        try:
                            cell_val = str(cell.value) if cell.value is not None else ""
                            max_length = max(max_length, len(cell_val))
                        except Exception:
                            pass

                    if col_name == "Object_ID":
                        adjusted_width = max(max_length + 2, 15)
                    elif col_name == "Source_Image" or "Image" in col_name:
                        adjusted_width = max(max_length + 2, 25)
                    elif "Density" in col_name or "IntegratedDensity" in col_name:
                        adjusted_width = max(max_length + 2, 20)
                    elif col_name.endswith("_um2") or col_name.endswith("_um"):
                        adjusted_width = max(max_length + 2, 12)
                    elif col_name.endswith("_px"):
                        adjusted_width = max(max_length + 2, 10)
                    elif "Ratio" in col_name or col_name in ["Circularity", "AspectRatio"]:
                        adjusted_width = max(max_length + 2, 12)
                    elif "Centroid" in col_name or "BBox" in col_name:
                        adjusted_width = max(max_length + 2, 11)
                    elif "Fluor_" in col_name:
                        adjusted_width = max(max_length + 2, 14)
                    else:
                        adjusted_width = max_length + 2

                    ws.column_dimensions[column_letter].width = min(adjusted_width, 30)

                # Apply color coding to rows based on Fluor_Density_per_BF_Area
                for row_idx in range(2, n_rows + 2):
                    df_idx = row_idx - 2

                    if df_idx < top_20_threshold:
                        fill = green_fill
                    elif df_idx >= bottom_20_threshold:
                        fill = red_fill
                    else:
                        fill = yellow_fill

                    for col_idx in range(1, len(df_sorted.columns) + 1):
                        ws.cell(row=row_idx, column=col_idx).fill = fill

            # Merge all typical particles into one sheet
            if all_typical_particles:
                merged_typical = pd.concat(all_typical_particles, ignore_index=True)
                merged_typical = merged_typical.sort_values("Fluor_Density_per_BF_Area", ascending=False)

                typical_sheet_name = f"{group_name}_Typical_Particles"
                merged_typical.to_excel(writer, sheet_name=typical_sheet_name, index=False)
                ws_merged = writer.sheets[typical_sheet_name]

                # Format merged sheet headers
                for cell in ws_merged[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = center_align

                ws_merged.auto_filter.ref = ws_merged.dimensions

                # Apply yellow fill to all data rows
                for row in ws_merged.iter_rows(min_row=2, max_row=len(merged_typical) + 1):
                    for cell in row:
                        cell.fill = yellow_fill

                # Auto-adjust column widths
                for column in ws_merged.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            cell_val = str(cell.value) if cell.value is not None else ""
                            max_length = max(max_length, len(cell_val))
                        except Exception:
                            pass
                    ws_merged.column_dimensions[column_letter].width = min(max_length + 2, 30)

            # ============= ERROR BAR SUMMARY SHEET =============
            try:
                error_bar_data = []
                
                for csv_file in sorted(csv_files):
                    image_name = csv_file.parent.name
                    df = pd.read_csv(csv_file)
                    
                    # Get typical particles (middle 60%)
                    df.rename(columns={"Area_px": "BF_Area_px", "Area_um2": "BF_Area_um2"}, inplace=True)
                    
                    if "Fluor_IntegratedDensity" in df.columns and "BF_Area_um2" in df.columns:
                        df["Fluor_Density_per_BF_Area"] = (
                            pd.to_numeric(df["Fluor_IntegratedDensity"], errors='coerce') / 
                            pd.to_numeric(df["BF_Area_um2"], errors='coerce')
                        )
                        df["Fluor_Density_per_BF_Area"] = df["Fluor_Density_per_BF_Area"].replace([np.inf, -np.inf], 0)
                        df = df.fillna(0)
                        
                        df_sorted = df.sort_values("Fluor_Density_per_BF_Area", ascending=False).reset_index(drop=True)
                        
                        n_rows = len(df_sorted)
                        top_20 = int(np.ceil(n_rows * 0.2))
                        bottom_20 = int(np.floor(n_rows * 0.8))
                        
                        typical = df_sorted.iloc[top_20:bottom_20]["Fluor_Density_per_BF_Area"]
                                               
                        mean_val = float(np.asarray(typical.mean()).item())
                        std_val  = float(np.asarray(typical.std()).item())
                        sem_val  = float(np.asarray(typical.sem()).item())


                        error_bar_data.append({
                            "Image": image_name,
                            "n": len(typical),
                            "Mean": mean_val,
                            "SD": std_val,
                            "SEM": sem_val,
                            "95% CI": 1.96 * sem_val
                        })
                
                if error_bar_data:
                    summary_df = pd.DataFrame(error_bar_data)
                    summary_df.to_excel(writer, sheet_name="Error_Bar_Summary", index=False)
                    ws_summary = writer.sheets["Error_Bar_Summary"]
                    
                    # Format headers
                    for cell in ws_summary[1]:
                        cell.fill = header_fill
                        cell.font = header_font
                        cell.alignment = center_align
                    
                    # Format data cells
                    for row_idx in range(2, len(summary_df) + 2):
                        ws_summary.cell(row=row_idx, column=1).alignment = Alignment(horizontal="left")  # Image name
                        ws_summary.cell(row=row_idx, column=2).number_format = "0"  # n (integer)
                        
                        for col_idx in range(3, 7):  # Mean, SD, SEM, 95% CI
                            ws_summary.cell(row=row_idx, column=col_idx).number_format = "0.00"
                            ws_summary.cell(row=row_idx, column=col_idx).alignment = Alignment(horizontal="right")
                    
                    # Auto-adjust column widths
                    ws_summary.column_dimensions["A"].width = 25
                    ws_summary.column_dimensions["B"].width = 10
                    for col in ["C", "D", "E", "F"]:
                        ws_summary.column_dimensions[col].width = 15
                    
                    print("  - Error_Bar_Summary sheet added")

            except Exception as e:
                print(f"[WARN] Could not create error bar summary: {e}")

        print(f"Excel consolidation saved: {excel_path}")
        print("  - Individual image sheets with color coding")
        print(f"  - {group_name}_Typical_Particles merged sheet")
        print("  - README sheet with column descriptions")

    except PermissionError:
        print(f"[ERROR] Cannot write to {excel_path} - file may be open")
        print("        Close Excel and try again")
    except Exception as e:
        print(f"[ERROR] Failed to create Excel file: {e}")
        import traceback

        traceback.print_exc()


# ==================================================
def generate_pdf_report(output_dir: Path, group_name: str):
    """Generate comprehensive A4 PDF report with full processing pipeline visualization"""
    pdf_path = output_dir / f"{group_name}_report.pdf"

    excel_path = output_dir / f"{group_name}_consolidated.xlsx"
    if not excel_path.exists():
        print(f"[WARN] No Excel file found for {group_name}, skipping PDF")
        return

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
    )

    story = []
    styles = getSampleStyleSheet()

    # Find first image folder for representative images
    img_folders = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    if not img_folders:
        print(f"[WARN] No image folders found in {output_dir}")
        return

    first_folder = img_folders[0]

    # ============= LOAD DATA ONCE AT THE START =============
    combined_df: Optional[pd.DataFrame] = None
    total_images = 0
    total_particles = 0

    try:
        xl_file = pd.ExcelFile(excel_path)
        all_data = []
        for sheet_name in xl_file.sheet_names:
            if sheet_name in ["README", f"{group_name}_Typical_Particles"]:
                continue
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            all_data.append(df)

        if not all_data:
            print(f"[ERROR] No valid data sheets found in {excel_path}")
            return

        combined_df = pd.concat(all_data, ignore_index=True)
        total_images = len(all_data)
        total_particles = len(combined_df)
    except Exception as e:
        print(f"[ERROR] Failed to load Excel data: {e}")
        return

    assert combined_df is not None, "combined_df should be initialized"

    # ============= PAGE 1: COVER & SUMMARY =============
    story.append(Paragraph("Particle Analysis Report", styles["Title"]))
    story.append(Paragraph(f"Sample Group: {group_name}", styles["Heading1"]))
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 20))

    # Key Statistics Table
    try:
        summary_data = [
            ["Metric", "Value"],
            ["Total Images Processed", str(total_images)],
            ["Total Particles Detected", str(total_particles)],
            [
                "Acceptance Rate",
                f"{(total_particles / (total_images * 100)):.1f}%" if total_images > 0 else "N/A",
            ],
            [
                "Mean Particle Size (µm²)",
                f"{combined_df['BF_Area_um2'].mean():.2f} ± {combined_df['BF_Area_um2'].std():.2f}",
            ],
            [
                "Mean Fluorescence Intensity",
                f"{combined_df['Fluor_Mean'].mean():.2f} ± {combined_df['Fluor_Mean'].std():.2f}",
            ],
        ]

        t = Table(summary_data, colWidths=[90 * mm, 70 * mm])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#E7E6E6")),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        story.append(t)

    except Exception as e:
        story.append(Paragraph(f"Error creating summary table: {e}", styles["Normal"]))

    story.append(PageBreak())

    # ============= PAGE 2-3: PROCESSING PIPELINE (2×2 GRIDS) =============
    story.append(Paragraph("Processing Pipeline", styles["Heading1"]))
    story.append(Spacer(1, 12))

    pipeline_page1 = [
        ("01_gray_8bit.png", "1. Raw Brightfield"),
        ("02_enhanced.png", "2. Background Subtracted"),
        ("03_enhanced_blur.png", "3. Noise Reduction"),
        ("04_thresh_raw.png", "4. Initial Threshold"),
    ]

    img_size = 70 * mm
    grid_data = []
    for i in range(0, 4, 2):
        row = []
        for j in range(2):
            idx = i + j
            img_name, caption = pipeline_page1[idx]
            img_path = first_folder / img_name
            if img_path.exists():
                img = Image(str(img_path), width=img_size, height=img_size)
                cell_content = [img, Paragraph(f"<font size=8>{caption}</font>", styles["Normal"])]
                row.append(cell_content)
            else:
                row.append(Paragraph(f"Missing: {img_name}", styles["Normal"]))
        grid_data.append(row)

    grid_table = Table(grid_data, colWidths=[img_size, img_size])
    grid_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(grid_table)
    story.append(PageBreak())

    story.append(Paragraph("Segmentation Refinement", styles["Heading1"]))
    story.append(Spacer(1, 12))

    pipeline_page2 = [
        ("05_closed.png", "5. Morphological Cleanup"),
        ("10_contours_all.png", "6. All Detected Contours"),
        ("12_mask_all.png", "7. Complete Segmentation Mask"),
        ("13_mask_accepted.png", "8. Filtered Particles Only"),
    ]

    grid_data = []
    for i in range(0, 4, 2):
        row = []
        for j in range(2):
            idx = i + j
            img_name, caption = pipeline_page2[idx]
            img_path = first_folder / img_name
            if img_path.exists():
                img = Image(str(img_path), width=img_size, height=img_size)
                cell_content = [img, Paragraph(f"<font size=8>{caption}</font>", styles["Normal"])]
                row.append(cell_content)
            else:
                row.append(Paragraph(f"Missing: {img_name}", styles["Normal"]))
        grid_data.append(row)

    grid_table = Table(grid_data, colWidths=[img_size, img_size])
    grid_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(grid_table)
    story.append(PageBreak())

    # ============= PAGE 4: DETECTION RESULTS =============
    story.append(Paragraph("Detection Results", styles["Heading1"]))
    story.append(Spacer(1, 12))

    detection_images = [
        ("10_contours_all.png", "All Detected Particles"),
        ("11_contours_rejected_orange_accepted_red_ids_green.png", "Quality Filtered Results"),
    ]

    large_img_size = 85 * mm
    detection_row = []
    for img_name, caption in detection_images:
        img_path = first_folder / img_name
        if img_path.exists():
            img = Image(str(img_path), width=large_img_size, height=large_img_size)
            detection_row.append([img, Paragraph(f"<b>{caption}</b>", styles["Normal"])])
        else:
            detection_row.append([Paragraph(f"Missing: {img_name}", styles["Normal"])])

    detection_table = Table([detection_row], colWidths=[large_img_size, large_img_size])
    detection_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(detection_table)
    story.append(PageBreak())

    # ============= PAGE 5: FLUORESCENCE ANALYSIS =============
    story.append(Paragraph("Fluorescence Analysis", styles["Heading1"]))
    story.append(Spacer(1, 12))

    fluor_images = [
        ("20_fluorescence_8bit.png", "Raw Fluorescence Channel"),
        ("21_fluorescence_overlay.png", "Intensity Measurements"),
    ]

    fluor_row = []
    for img_name, caption in fluor_images:
        img_path = first_folder / img_name
        if img_path.exists():
            img = Image(str(img_path), width=large_img_size, height=large_img_size)
            fluor_row.append([img, Paragraph(f"<b>{caption}</b>", styles["Normal"])])
        else:
            fluor_row.append([Paragraph(f"Missing: {img_name}", styles["Normal"])])

    fluor_table = Table([fluor_row], colWidths=[large_img_size, large_img_size])
    fluor_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(fluor_table)
    story.append(PageBreak())

    # ============= PAGE 6: DATA SUMMARY =============
    story.append(Paragraph("Data Summary", styles["Heading1"]))
    story.append(Spacer(1, 12))

    try:
        top10 = combined_df.nlargest(10, "BF_Area_um2")[
            ["Object_ID", "BF_Area_um2", "EquivDiameter_um", "Circularity", "Fluor_Mean"]
        ]
        top10_data = [["ID", "Area (µm²)", "Diameter (µm)", "Circularity", "Fluor Mean"]]
        for _, row in top10.iterrows():
            top10_data.append(
                [
                    str(row["Object_ID"]),
                    f"{row['BF_Area_um2']:.2f}",
                    f"{row['EquivDiameter_um']:.2f}",
                    f"{row['Circularity']:.3f}",
                    f"{row['Fluor_Mean']:.2f}",
                ]
            )

        top10_table = Table(top10_data, colWidths=[25 * mm, 25 * mm, 25 * mm, 25 * mm, 30 * mm])
        top10_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ]
            )
        )
        story.append(Paragraph("Top 10 Largest Particles", styles["Heading2"]))
        story.append(Spacer(1, 6))
        story.append(top10_table)
        story.append(Spacer(1, 20))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

        ax1.hist(combined_df["BF_Area_um2"], bins=30, color="steelblue", edgecolor="black")
        ax1.set_xlabel("Area (µm²)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Particle Size Distribution")
        ax1.grid(True, alpha=0.3)

        ax2.hist(combined_df["Fluor_Mean"], bins=30, color="green", edgecolor="black")
        ax2.set_xlabel("Mean Fluorescence Intensity")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Fluorescence Distribution")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        hist_path = output_dir / f"{group_name}_histograms.png"
        plt.savefig(hist_path, dpi=150, bbox_inches="tight")
        plt.close()

        story.append(Paragraph("Distribution Analysis", styles["Heading2"]))
        story.append(Spacer(1, 6))
        story.append(Image(str(hist_path), width=170 * mm, height=60 * mm))

        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<i>Full dataset available in: {excel_path.name}</i>", styles["Normal"]))

    except Exception as e:
        story.append(Paragraph(f"Error generating data summary: {e}", styles["Normal"]))

    try:
        doc.build(story)
        print(f"PDF report saved: {pdf_path}")
    except Exception as e:
        print(f"[ERROR] Failed to generate PDF: {e}")


# ==================================================
def generate_error_bar_comparison(output_dir: Path) -> None:
    """Generate error bar comparison plot with overlaid jitter (strip plot)"""

    excel_files = list(output_dir.rglob("*_consolidated.xlsx"))

    if len(excel_files) < 2:
        print(f"[INFO] Need at least 2 groups for comparison. Found {len(excel_files)}")
        return

    # 1. Collect ALL raw data into a single DataFrame for Seaborn
    all_data_rows = []
    
    # We also keep track of stats for the text file summary
    group_stats = {} 

    for excel_path in sorted(excel_files):
        group_name = excel_path.stem.replace("_consolidated", "")

        try:
            typical_sheet = f"{group_name}_Typical_Particles"
            df = pd.read_excel(excel_path, sheet_name=typical_sheet)

            if "Fluor_Density_per_BF_Area" in df.columns:
                values = df["Fluor_Density_per_BF_Area"].dropna()
                
                # Store raw values for the plot
                for v in values:
                    all_data_rows.append({
                        "Group": group_name,
                        "Fluorescence Density": float(np.asarray(v).item())
                    })

                # Calculate stats for the CSV summary
                mean_val = float(np.asarray(values.mean()).item())
                std_val  = float(np.asarray(values.std()).item())
                sem_val  = float(np.asarray(values.sem()).item())
                
                group_stats[group_name] = {
                    "n": len(values),
                    "mean": mean_val,
                    "std": std_val,
                    "sem": sem_val,
                    "ci_95": 1.96 * sem_val
                }
                
                print(f"Loaded {len(values)} points for group: {group_name}")

        except Exception as e:
            print(f"[WARN] Could not read {group_name}: {e}")
            continue

    if not all_data_rows:
        print("[WARN] No valid data found for comparison")
        return

    # Create the Master DataFrame
    df_all = pd.DataFrame(all_data_rows)

    # 2. Define Error Bar Types to Generate
    plot_configs = [
        ("SD", "sd", "Standard Deviation"),
        ("CI95", 95, "95% Confidence Interval"),
    ]

    # Define colors
    unique_groups = df_all['Group'].unique()
    palette_colors = ['silver', 'violet'] 
    if len(unique_groups) > 2:
        palette_colors = sns.color_palette("husl", len(unique_groups))

    for suffix, ci_param, label in plot_configs:
        plt.figure(figsize=(6, 5))
        sns.set_style("ticks")

        # A. Bar Plot
        try:
            # Newer Seaborn (v0.12+)
            ebar_arg = "sd" if ci_param == "sd" else ("ci", ci_param)
            ax = sns.barplot(
                data=df_all,
                x="Group",
                y="Fluorescence Density",
                hue="Group",              # fixes palette deprecation
                palette=palette_colors,
                legend=False,             # keep same appearance
                errorbar=ebar_arg,
                capsize=0.1,
                edgecolor="black",
                alpha=0.7,
                err_kws={"color": "black", "linewidth": 1.5},  # replaces errcolor/errwidth
            )
        except TypeError:
            # Older Seaborn
            ax = sns.barplot(
                x='Group', 
                y='Fluorescence Density', 
                data=df_all, 
                ci=ci_param,
                capsize=0.1, 
                palette=palette_colors,
                edgecolor="black", 
                errcolor="black",
                errwidth=1.5,
                alpha=0.7
            )

        # B. Jitter Plot
        sns.stripplot(
            x='Group', 
            y='Fluorescence Density', 
            data=df_all, 
            jitter=True, 
            color='cyan',
            edgecolor='black', 
            linewidth=0.5,
            size=6, 
            alpha=0.6
        )

        # C. Styling
        plt.ylabel('Fluorescence Density (a.u./µm²)', fontsize=12, fontweight='bold')
        plt.xlabel('')
        plt.xticks(fontsize=10, fontweight='bold')
        plt.yticks(fontsize=10, fontweight='bold')
        plt.title(f"Comparison (Error Bars: {label})", fontsize=11)

        for axis in ['top','bottom','left','right']:
            plt.gca().spines[axis].set_linewidth(1.5)

        plt.tight_layout()
        
        out_path = output_dir / f"error_bar_jitter_comparison_{suffix}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved plot: {out_path}")

    # 3. Save Statistical Summary CSV
    summary_data = []
    for g_name, stats in group_stats.items():
        summary_data.append({
            "Group": g_name,
            "n": stats["n"],
            "Mean": f"{stats['mean']:.2f}",
            "SD": f"{stats['std']:.2f}",
            "SEM": f"{stats['sem']:.2f}",
            "95% CI": f"±{stats['ci_95']:.2f}"
        })
    
    pd.DataFrame(summary_data).to_csv(output_dir / "comparison_statistics.csv", index=False)

    # 4. Perform T-Test if exactly 2 groups
    if len(unique_groups) == 2:
        g1, g2 = unique_groups[0], unique_groups[1]
        data1 = df_all[df_all['Group'] == g1]['Fluorescence Density']
        data2 = df_all[df_all['Group'] == g2]['Fluorescence Density']
        
        t_stat, p_value = scipy_stats.ttest_ind(data1, data2)
        
        # FIX: Explicitly cast p_value to float to satisfy Pylance
        p_val_float = float(cast(float, p_value))

        with open(output_dir / "statistical_test.txt", "w", encoding="utf-8") as f:
            f.write("Two-Sample t-Test Results\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"Group 1: {g1} (n={len(data1)})\n")
            f.write(f"Group 2: {g2} (n={len(data2)})\n\n")
            f.write(f"t-statistic: {t_stat:.4f}\n")
            f.write(f"p-value: {p_val_float:.4f}\n")
            f.write(f"Significance (alpha=0.05): {'YES' if p_val_float < 0.05 else 'NO'}\n")

def embed_comparison_plots_into_all_excels(
    output_root: Path,
    sd_plot: str = "error_bar_jitter_comparison_SD.png",
    ci_plot: str = "error_bar_jitter_comparison_CI95.png",
) -> None:
    """
    Post-process all consolidated Excel files under output_root:
      - rename 'Error_Bar_Summary' to 'Summary' (or keep 'Summary' if already present)
      - move 'Summary' to the first worksheet
      - embed SD and CI95 comparison plots into 'Summary'

    This is run AFTER generate_error_bar_comparison(output_root) so the images exist.
    """
    sd_img_path = output_root / sd_plot
    ci_img_path = output_root / ci_plot

    if not sd_img_path.exists():
        print(f"[WARN] SD plot not found, embedding skipped: {sd_img_path}")
    if not ci_img_path.exists():
        print(f"[WARN] CI95 plot not found, embedding skipped: {ci_img_path}")

    excel_files = sorted(output_root.rglob("*_consolidated.xlsx"))
    if not excel_files:
        print(f"[WARN] No consolidated Excel files found under {output_root}")
        return

    def add_png(ws, path: Path, anchor_cell: str, width_px: int = 620) -> None:
        if not path.exists():
            return
        img = XLImage(str(path))
        # Resize to a reasonable width while keeping aspect ratio
        if getattr(img, "width", None) and getattr(img, "height", None):
            scale = width_px / float(img.width)
            img.width = int(img.width * scale)
            img.height = int(img.height * scale)
        ws.add_image(img, anchor_cell)

    updated = 0
    for excel_path in excel_files:
        try:
            wb = load_workbook(excel_path)

            # Decide which sheet becomes "Summary"
            if "Summary" in wb.sheetnames:
                ws_summary = wb["Summary"]
            elif "Error_Bar_Summary" in wb.sheetnames:
                ws_summary = wb["Error_Bar_Summary"]
                ws_summary.title = "Summary"
                ws_summary = wb["Summary"]
            else:
                # If neither exists, create a Summary sheet
                ws_summary = wb.create_sheet("Summary")

            modified = False

            # Rename sheet if needed
            if "Summary" in wb.sheetnames:
                ws_summary = wb["Summary"]
            elif "Error_Bar_Summary" in wb.sheetnames:
                ws_summary = wb["Error_Bar_Summary"]
                ws_summary.title = "Summary"
                ws_summary = wb["Summary"]
                modified = True
            else:
                ws_summary = wb.create_sheet("Summary")
                modified = True

            # Move Summary to first position (only if not already first)
            current_idx = wb.sheetnames.index(ws_summary.title)
            if current_idx != 0:
                wb.move_sheet(ws_summary, offset=-current_idx)
                modified = True

            # Avoid duplicates: embed only once
            marker_cell = "A10"
            marker_value = "COMPARISON_PLOTS_EMBEDDED"
            if ws_summary[marker_cell].value != marker_value:
                add_png(ws_summary, sd_img_path, "A12")
                add_png(ws_summary, ci_img_path, "H12")
                ws_summary[marker_cell].value = marker_value
                modified = True
            else:
                print(f"Plots already embedded in {excel_path.name}; skipping embedding.")

            if modified:
                wb.save(excel_path)
                updated += 1
                print(f"Updated workbook: {excel_path}")
            else:
                print(f"No changes needed: {excel_path}")

        except Exception as e:
            print(f"[WARN] Could not embed plots into {excel_path}: {e}")

    print(f"Embedding complete. Updated {updated}/{len(excel_files)} workbooks.")



# ==================================================
# Main processing
# ==================================================
def process_image(img_path: Path, output_root: Path) -> None:
    print("\n" + "=" * 80)
    print(f"Processing: {img_path}")

    xml_props, xml_main = find_metadata_paths(img_path)
    print(f"Metadata (Properties): {xml_props}")
    print(f"Metadata (Main):       {xml_main}")

    try:
        um_per_px_x, um_per_px_y = get_pixel_size_um(xml_props, xml_main)
        um_per_px_x = float(um_per_px_x)
        um_per_px_y = float(um_per_px_y)
    except Exception as e:
        if FALLBACK_UM_PER_PX is None:
            raise
        print(f"[WARN] {e} -> using fallback pixel size {FALLBACK_UM_PER_PX} µm/px")
        um_per_px_x = um_per_px_y = float(FALLBACK_UM_PER_PX)

    um_per_px_avg = (um_per_px_x + um_per_px_y) / 2.0
    print(
        f"Pixel size: X={um_per_px_x:.6f} µm/px, Y={um_per_px_y:.6f} µm/px (avg={um_per_px_avg:.6f})"
    )

    img_out = output_root / img_path.stem
    ensure_dir(img_out)

    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(str(img_path))
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"Loaded: dtype={img.dtype}, shape={img.shape}, range=[{img.min()}-{img.max()}]")

    img8 = normalize_to_8bit(img)
    save_debug(img_out, "01_gray_8bit.png", img8, um_per_px_avg)

    mask = segment_particles_brightfield(img8, float(um_per_px_avg), img_out)

    # OpenCV stubs are messy; avoid tuple slicing typing issues
    _fc = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cast(list[np.ndarray], _fc[-2])

    print(f"Contours found (pre-filter): {len(contours)}")

    vis_all = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis_all, contours, -1, (0, 0, 255), 1)
    save_debug(img_out, "10_contours_all.png", vis_all, um_per_px_avg)

    um2_per_px2 = float(um_per_px_x) * float(um_per_px_y)
    min_area_px = MIN_AREA_UM2 / um2_per_px2
    max_area_px = MAX_AREA_UM2 / um2_per_px2

    H, W = img8.shape[:2]
    img_area_px = float(H * W)
    max_big_area_px = MAX_FRACTION_OF_IMAGE_AREA * img_area_px

    accepted: list[np.ndarray] = []
    rejected: list[np.ndarray] = []

    for c in contours:
        area_px = float(cv2.contourArea(c))
        if area_px <= 0:
            rejected.append(c)
            continue

        if area_px >= max_big_area_px:
            rejected.append(c)
            continue

        perim_px = float(cv2.arcLength(c, True))
        circ = (4 * np.pi * area_px / (perim_px**2)) if perim_px > 0 else 0.0

        ok = (min_area_px <= area_px <= max_area_px) and (circ >= MIN_CIRCULARITY)
        (accepted if ok else rejected).append(c)

    print(f"Accepted: {len(accepted)} | Rejected: {len(rejected)}")
    print(
        f"Filter thresholds: area [{MIN_AREA_UM2}-{MAX_AREA_UM2}] µm² "
        f"(~[{min_area_px:.1f}-{max_area_px:.1f}] px²), circularity >= {MIN_CIRCULARITY}, "
        f"max_single_contour_area <= {MAX_FRACTION_OF_IMAGE_AREA:.0%} of image"
    )

    vis_acc = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis_acc, rejected, -1, (0, 165, 255), 1)
    cv2.drawContours(vis_acc, accepted, -1, (0, 0, 255), 1)
    vis_acc = draw_object_ids(vis_acc, accepted)
    save_debug(img_out, "11_contours_rejected_orange_accepted_red_ids_green.png", vis_acc, um_per_px_avg)

    mask_all = np.zeros_like(mask)
    cv2.drawContours(mask_all, contours, -1, 255, thickness=-1)
    save_debug(img_out, "12_mask_all.png", mask_all)

    mask_acc = np.zeros_like(mask)
    cv2.drawContours(mask_acc, accepted, -1, 255, thickness=-1)
    save_debug(img_out, "13_mask_accepted.png", mask_acc)

    fluor_path = img_path.parent / img_path.name.replace("_ch00", "_ch01")
    fluor_measurements: Optional[list[dict]] = None

    if fluor_path.exists():
        fluor_img = cv2.imread(str(fluor_path), cv2.IMREAD_UNCHANGED)
        if fluor_img is not None:
            if fluor_img.ndim == 3:
                fluor_img = cv2.cvtColor(fluor_img, cv2.COLOR_BGR2GRAY)

            print(f"Fluorescence loaded: dtype={fluor_img.dtype}, range=[{fluor_img.min()}-{fluor_img.max()}]")

            fluor_img8 = normalize_to_8bit(fluor_img)
            save_debug(img_out, "20_fluorescence_8bit.png", fluor_img8, um_per_px_avg)

            fluor_measurements = measure_fluorescence_intensity(
                fluor_img, accepted, float(um_per_px_x), float(um_per_px_y)
            )

            fluor_overlay = visualize_fluorescence_measurements(
                fluor_img8, accepted, fluor_measurements
            )
            save_debug(img_out, "21_fluorescence_overlay.png", fluor_overlay, um_per_px_avg)
        else:
            print(f"[WARN] Could not load fluorescence image: {fluor_path}")
    else:
        print(f"[WARN] Fluorescence channel not found: {fluor_path}")

    parts = img_path.stem.split()
    group_id = parts[0] if parts else "unk"
    sequence_num = parts[-1].split("_")[0] if parts else "0"

    csv_path = img_out / "object_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Object_ID",
                "BF_Area_px",
                "BF_Area_um2",
                "Perimeter_px",
                "Perimeter_um",
                "EquivDiameter_px",
                "EquivDiameter_um",
                "Circularity",
                "AspectRatio",
                "CentroidX_px",
                "CentroidY_px",
                "CentroidX_um",
                "CentroidY_um",
                "BBoxX_px",
                "BBoxY_px",
                "BBoxW_px",
                "BBoxH_px",
                "BBoxW_um",
                "BBoxH_um",
                "Fluor_Area_px",
                "Fluor_Area_um2",
                "Fluor_Mean",
                "Fluor_Median",
                "Fluor_Std",
                "Fluor_Min",
                "Fluor_Max",
                "Fluor_IntegratedDensity",
            ]
        )

        for i, c in enumerate(accepted, 1):
            compound_id = f"{group_id}_{sequence_num}_{i}"

            area_px = float(cv2.contourArea(c))
            area_um2 = area_px * um2_per_px2

            perim_px = float(cv2.arcLength(c, True))
            perim_um = contour_perimeter_um(c, float(um_per_px_x), float(um_per_px_y))

            eqd_px = equivalent_diameter_from_area(area_px)
            eqd_um = equivalent_diameter_from_area(area_um2)

            circ = (4 * np.pi * area_px / (perim_px**2)) if perim_px > 0 else 0.0

            x, y, bw, bh = cv2.boundingRect(c)
            aspect = (float(bw) / float(bh)) if bh > 0 else 0.0

            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                cx, cy = 0.0, 0.0

            cx_um = cx * float(um_per_px_x)
            cy_um = cy * float(um_per_px_y)
            bw_um = float(bw) * float(um_per_px_x)
            bh_um = float(bh) * float(um_per_px_y)

            if fluor_measurements is not None:
                fm = fluor_measurements[i - 1]
            else:
                fm = {
                    "fluor_area_px": 0.0,
                    "fluor_area_um2": 0.0,
                    "fluor_mean": 0.0,
                    "fluor_median": 0.0,
                    "fluor_std": 0.0,
                    "fluor_min": 0.0,
                    "fluor_max": 0.0,
                    "fluor_integrated_density": 0.0,
                }

            w.writerow(
                [
                    compound_id,
                    f"{area_px:.2f}",
                    f"{area_um2:.4f}",
                    f"{perim_px:.2f}",
                    f"{perim_um:.4f}",
                    f"{eqd_px:.2f}",
                    f"{eqd_um:.4f}",
                    f"{circ:.4f}",
                    f"{aspect:.4f}",
                    f"{cx:.2f}",
                    f"{cy:.2f}",
                    f"{cx_um:.4f}",
                    f"{cy_um:.4f}",
                    x,
                    y,
                    bw,
                    bh,
                    f"{bw_um:.4f}",
                    f"{bh_um:.4f}",
                    f"{float(fm['fluor_area_px']):.2f}",
                    f"{float(fm['fluor_area_um2']):.4f}",
                    f"{float(fm['fluor_mean']):.2f}",
                    f"{float(fm['fluor_median']):.2f}",
                    f"{float(fm['fluor_std']):.2f}",
                    f"{float(fm['fluor_min']):.2f}",
                    f"{float(fm['fluor_max']):.2f}",
                    f"{float(fm['fluor_integrated_density']):.2f}",
                ]
            )

    print(f"CSV saved: {csv_path} ({len(accepted)} objects)")
    print("✓ Done")


def main() -> None:
    if CLEAR_OUTPUT_DIR_EACH_RUN:
        clear_output_dir(OUTPUT_DIR)

    print(f"Input dir: {_project_root.resolve()}")
    groups = list_sample_group_folders(SOURCE_DIR)
    selected_group_dir = prompt_user_select_group(groups)

    if selected_group_dir is None:
        dirs_to_process = groups[:]
    else:
        dirs_to_process = [selected_group_dir]

    if CONTROL_DIR.exists():
        dirs_to_process.append(CONTROL_DIR)

    img_paths: list[Path] = []
    for d in dirs_to_process:
        img_paths.extend(sorted(d.rglob(IMAGE_GLOB)))

    print(f"Found {len(img_paths)} brightfield images matching '{IMAGE_GLOB}'")
    if not img_paths:
        raise FileNotFoundError(f"No images found under {SOURCE_DIR} matching {IMAGE_GLOB}")

    total_processed = 0
    total_failed = 0

    for p in tqdm(img_paths, desc="Processing images", unit="img"):
        out_root = (OUTPUT_DIR / p.parent.name) if SEPARATE_OUTPUT_BY_GROUP else OUTPUT_DIR
        ensure_dir(out_root)

        try:
            process_image(p, out_root)
            total_processed += 1
        except Exception as e:
            tqdm.write(f"[ERROR] Failed processing {p}: {e}")
            total_failed += 1

    print(f"\n{'=' * 80}")
    print(f"SUMMARY: {total_processed} succeeded, {total_failed} failed")

    if SEPARATE_OUTPUT_BY_GROUP:
        for group_dir in OUTPUT_DIR.iterdir():
            if group_dir.is_dir():
                consolidate_to_excel(group_dir, group_dir.name)
                generate_pdf_report(group_dir, group_dir.name)

        print(f"\n{'=' * 80}")
        print("Generating error bar comparison plots...")
        generate_error_bar_comparison(OUTPUT_DIR)

        print("Embedding comparison plots into consolidated Excel files...")
        embed_comparison_plots_into_all_excels(OUTPUT_DIR)


if __name__ == "__main__":
    main()