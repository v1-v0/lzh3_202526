import cv2
import numpy as np
import sys
import csv

import atexit
import re
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Optional, Tuple
from tqdm import tqdm
import pandas as pd

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
    PageBreak
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def add_scale_bar(img: np.ndarray, pixel_size: float, unit: str = "um", length_um: float = 10) -> np.ndarray:
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

    label = f"{int(length_um)} um" if unit in ["µm", "um"] else f"{int(length_um)} {unit}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(label, font, SCALE_BAR_FONT_SCALE, SCALE_BAR_FONT_THICKNESS)

    text_x = bar_x + (bar_length_px - text_w) // 2
    text_y = bar_y - 8

    bg_padding = 5
    bg_x1 = min(bar_x, text_x) - bg_padding
    bg_y1 = text_y - text_h - bg_padding
    bg_x2 = max(bar_x + bar_length_px, text_x + text_w) + bg_padding
    bg_y2 = bar_y + SCALE_BAR_HEIGHT + bg_padding

    img = img.copy()
    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), SCALE_BAR_BG_COLOR, -1)
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


def save_debug(folder: Path, name: str, img: np.ndarray, pixel_size_um: Optional[float] = None) -> None:
    """Save debug image with optional scale bar"""
    out = folder / name
    img_to_save = img.copy()

    if pixel_size_um is not None and pixel_size_um > 0:
        img_to_save = add_scale_bar(img_to_save, pixel_size_um, "um", SCALE_BAR_LENGTH_UM)

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
                raise ValueError(f"Unexpected units in {xml_props_path.name}: X={x_unit}, Y={y_unit}")

            return x_len / x_n, y_len / y_n

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
                raise ValueError(f"Unexpected units in {xml_main_path.name}: X={x_unit}, Y={y_unit}")

            return (x_len_m * 1e6) / x_n, (y_len_m * 1e6) / y_n

        except Exception as e:
            print(f"[WARN] Failed to read pixel size from {xml_main_path}: {e}")

    raise ValueError("Could not determine pixel size (µm/px). Missing/invalid metadata XML.")


def contour_perimeter_um(contour: np.ndarray, um_per_px_x: float, um_per_px_y: float) -> float:
    pts = contour.reshape(-1, 2).astype(np.float64)
    pts[:, 0] *= um_per_px_x
    pts[:, 1] *= um_per_px_y
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
    um_per_px_y: float
) -> list[dict]:
    """Measure fluorescence intensity for each contour using brightfield masks."""
    measurements = []
    um2_per_px2 = um_per_px_x * um_per_px_y

    for i, c in enumerate(contours, 1):
        mask = np.zeros(fluor_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, thickness=-1)

        fluor_values = fluor_img[mask > 0]

        if len(fluor_values) > 0:
            area_px = float(cv2.contourArea(c))
            area_um2 = area_px * um2_per_px2

            measurements.append({
                'object_id': i,
                'fluor_area_px': area_px,
                'fluor_area_um2': area_um2,
                'fluor_mean': float(np.mean(fluor_values)),
                'fluor_median': float(np.median(fluor_values)),
                'fluor_std': float(np.std(fluor_values)),
                'fluor_min': float(np.min(fluor_values)),
                'fluor_max': float(np.max(fluor_values)),
                'fluor_integrated_density': float(np.sum(fluor_values)),
            })
        else:
            measurements.append({
                'object_id': i,
                'fluor_area_px': 0.0,
                'fluor_area_um2': 0.0,
                'fluor_mean': 0.0,
                'fluor_median': 0.0,
                'fluor_std': 0.0,
                'fluor_min': 0.0,
                'fluor_max': 0.0,
                'fluor_integrated_density': 0.0,
            })

    return measurements


def visualize_fluorescence_measurements(
    fluor_img8: np.ndarray,
    contours: list[np.ndarray],
    measurements: list[dict]
) -> np.ndarray:
    """Create visualization with contours and intensity labels"""
    vis = cv2.cvtColor(fluor_img8, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)

    for m in measurements:
        c = contours[m['object_id'] - 1]
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            label = f"{m['object_id']}: {m['fluor_mean']:.0f}"
            cv2.putText(
                vis, label, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA
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
# Excel consolidation
# ==================================================
def consolidate_to_excel(output_dir: Path, group_name: str) -> None:
    """Consolidate all CSVs in a group folder into one Excel workbook"""
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
            print(f"        Please close the file and run again, or delete it manually")
            return

    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for csv_file in sorted(csv_files):
                image_name = csv_file.parent.name
                df = pd.read_csv(csv_file)

                sheet_name = image_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Excel consolidation saved: {excel_path}")
    except PermissionError:
        print(f"[ERROR] Cannot write to {excel_path} - file may be open")
        print(f"        Close Excel and try again")
    except Exception as e:
        print(f"[ERROR] Failed to create Excel file: {e}")

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
        topMargin=15*mm,
        bottomMargin=15*mm,
        leftMargin=15*mm,
        rightMargin=15*mm
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
    combined_df = None
    total_images = 0
    total_particles = 0
    
    try:
        xl_file = pd.ExcelFile(excel_path)
        all_data = []
        for sheet_name in xl_file.sheet_names:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        total_images = len(xl_file.sheet_names)
        total_particles = len(combined_df)
    except Exception as e:
        print(f"[ERROR] Failed to load Excel data: {e}")
        return  # Cannot proceed without data
    
    # ============= PAGE 1: COVER & SUMMARY =============
    story.append(Paragraph(f"Particle Analysis Report", styles['Title']))
    story.append(Paragraph(f"Sample Group: {group_name}", styles['Heading1']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
        styles['Normal']
    ))
    story.append(Spacer(1, 20))
    
    # Key Statistics Table
    try:
        summary_data = [
            ["Metric", "Value"],
            ["Total Images Processed", str(total_images)],
            ["Total Particles Detected", str(total_particles)],
            ["Acceptance Rate", f"{(total_particles / (total_images * 100)):.1f}%" if total_images > 0 else "N/A"],
            ["Mean Particle Size (µm²)", f"{combined_df['Area_um2'].mean():.2f} ± {combined_df['Area_um2'].std():.2f}"],
            ["Mean Fluorescence Intensity", f"{combined_df['Fluor_Mean'].mean():.2f} ± {combined_df['Fluor_Mean'].std():.2f}"],
        ]
        
        t = Table(summary_data, colWidths=[90*mm, 70*mm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#E7E6E6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        
    except Exception as e:
        story.append(Paragraph(f"Error creating summary table: {e}", styles['Normal']))
    
    story.append(PageBreak())
    
    # ============= PAGE 2-3: PROCESSING PIPELINE (2×2 GRIDS) =============
    story.append(Paragraph("Processing Pipeline", styles['Heading1']))
    story.append(Spacer(1, 12))
    
    # Page 2: Initial processing
    pipeline_page1 = [
        ("01_gray_8bit.png", "1. Raw Brightfield"),
        ("02_enhanced.png", "2. Background Subtracted"),
        ("03_enhanced_blur.png", "3. Noise Reduction"),
        ("04_thresh_raw.png", "4. Initial Threshold"),
    ]
    
    img_size = 70*mm
    grid_data = []
    for i in range(0, 4, 2):
        row = []
        for j in range(2):
            idx = i + j
            img_name, caption = pipeline_page1[idx]
            img_path = first_folder / img_name
            if img_path.exists():
                img = Image(str(img_path), width=img_size, height=img_size)
                cell_content = [img, Paragraph(f"<font size=8>{caption}</font>", styles['Normal'])]
                row.append(cell_content)
            else:
                row.append(Paragraph(f"Missing: {img_name}", styles['Normal']))
        grid_data.append(row)
    
    grid_table = Table(grid_data, colWidths=[img_size, img_size])
    grid_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(grid_table)
    story.append(PageBreak())
    
    # Page 3: Morphological refinement
    story.append(Paragraph("Segmentation Refinement", styles['Heading1']))
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
                cell_content = [img, Paragraph(f"<font size=8>{caption}</font>", styles['Normal'])]
                row.append(cell_content)
            else:
                row.append(Paragraph(f"Missing: {img_name}", styles['Normal']))
        grid_data.append(row)
    
    grid_table = Table(grid_data, colWidths=[img_size, img_size])
    grid_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(grid_table)
    story.append(PageBreak())
    
    # ============= PAGE 4: DETECTION RESULTS =============
    story.append(Paragraph("Detection Results", styles['Heading1']))
    story.append(Spacer(1, 12))
    
    detection_images = [
        ("10_contours_all.png", "All Detected Particles"),
        ("11_contours_rejected_orange_accepted_red_ids_green.png", "Quality Filtered Results"),
    ]
    
    large_img_size = 85*mm
    detection_row = []
    for img_name, caption in detection_images:
        img_path = first_folder / img_name
        if img_path.exists():
            img = Image(str(img_path), width=large_img_size, height=large_img_size)
            detection_row.append([img, Paragraph(f"<b>{caption}</b>", styles['Normal'])])
        else:
            detection_row.append([Paragraph(f"Missing: {img_name}", styles['Normal'])])
    
    detection_table = Table([detection_row], colWidths=[large_img_size, large_img_size])
    detection_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(detection_table)
    story.append(PageBreak())
    
    # ============= PAGE 5: FLUORESCENCE ANALYSIS =============
    story.append(Paragraph("Fluorescence Analysis", styles['Heading1']))
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
            fluor_row.append([img, Paragraph(f"<b>{caption}</b>", styles['Normal'])])
        else:
            fluor_row.append([Paragraph(f"Missing: {img_name}", styles['Normal'])])
    
    fluor_table = Table([fluor_row], colWidths=[large_img_size, large_img_size])
    fluor_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(fluor_table)
    story.append(PageBreak())
    
    # ============= PAGE 6: DATA SUMMARY =============
    story.append(Paragraph("Data Summary", styles['Heading1']))
    story.append(Spacer(1, 12))
    
    # Top 10 particles table and histograms
    try:
        top10 = combined_df.nlargest(10, 'Area_um2')[['Object_ID', 'Area_um2', 'EquivDiameter_um', 'Circularity', 'Fluor_Mean']]
        top10_data = [["ID", "Area (µm²)", "Diameter (µm)", "Circularity", "Fluor Mean"]]
        for _, row in top10.iterrows():
            top10_data.append([
                str(int(row['Object_ID'])),
                f"{row['Area_um2']:.2f}",
                f"{row['EquivDiameter_um']:.2f}",
                f"{row['Circularity']:.3f}",
                f"{row['Fluor_Mean']:.2f}"
            ])
        
        top10_table = Table(top10_data, colWidths=[15*mm, 30*mm, 30*mm, 25*mm, 30*mm])
        top10_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
        ]))
        story.append(Paragraph("Top 10 Largest Particles", styles['Heading2']))
        story.append(Spacer(1, 6))
        story.append(top10_table)
        story.append(Spacer(1, 20))
        
        # Generate histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        
        # Particle size distribution
        ax1.hist(combined_df['Area_um2'], bins=30, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Area (µm²)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Particle Size Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Fluorescence intensity distribution
        ax2.hist(combined_df['Fluor_Mean'], bins=30, color='green', edgecolor='black')
        ax2.set_xlabel('Mean Fluorescence Intensity')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Fluorescence Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        hist_path = output_dir / f"{group_name}_histograms.png"
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        story.append(Paragraph("Distribution Analysis", styles['Heading2']))
        story.append(Spacer(1, 6))
        story.append(Image(str(hist_path), width=170*mm, height=60*mm))
        
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<i>Full dataset available in: {excel_path.name}</i>", styles['Normal']))
        
    except Exception as e:
        story.append(Paragraph(f"Error generating data summary: {e}", styles['Normal']))
    
    # Build PDF
    try:
        doc.build(story)
        print(f"PDF report saved: {pdf_path}")
    except Exception as e:
        print(f"[ERROR] Failed to generate PDF: {e}")






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
    except Exception as e:
        if FALLBACK_UM_PER_PX is None:
            raise
        print(f"[WARN] {e} -> using fallback pixel size {FALLBACK_UM_PER_PX} µm/px")
        um_per_px_x = um_per_px_y = float(FALLBACK_UM_PER_PX)

    um_per_px_avg = (um_per_px_x + um_per_px_y) / 2.0
    print(f"Pixel size: X={um_per_px_x:.6f} µm/px, Y={um_per_px_y:.6f} µm/px (avg={um_per_px_avg:.6f})")

    img_out = output_root / img_path.stem
    ensure_dir(img_out)

    # Load brightfield
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(str(img_path))
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"Loaded: dtype={img.dtype}, shape={img.shape}, range=[{img.min()}-{img.max()}]")

    img8 = normalize_to_8bit(img)
    save_debug(img_out, "01_gray_8bit.png", img8, um_per_px_avg)

    mask = segment_particles_brightfield(img8, um_per_px_avg, img_out)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    print(f"Contours found (pre-filter): {len(contours)}")

    vis_all = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis_all, contours, -1, (0, 0, 255), 1)
    save_debug(img_out, "10_contours_all.png", vis_all, um_per_px_avg)

    um2_per_px2 = um_per_px_x * um_per_px_y
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
        circ = (4 * np.pi * area_px / (perim_px ** 2)) if perim_px > 0 else 0.0

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

    # Load and process fluorescence channel
    fluor_path = img_path.parent / img_path.name.replace("_ch00", "_ch01")
    fluor_measurements = None

    if fluor_path.exists():
        fluor_img = cv2.imread(str(fluor_path), cv2.IMREAD_UNCHANGED)
        if fluor_img is not None:
            if fluor_img.ndim == 3:
                fluor_img = cv2.cvtColor(fluor_img, cv2.COLOR_BGR2GRAY)

            print(f"Fluorescence loaded: dtype={fluor_img.dtype}, range=[{fluor_img.min()}-{fluor_img.max()}]")

            fluor_img8 = normalize_to_8bit(fluor_img)
            save_debug(img_out, "20_fluorescence_8bit.png", fluor_img8, um_per_px_avg)

            fluor_measurements = measure_fluorescence_intensity(
                fluor_img, accepted, um_per_px_x, um_per_px_y
            )

            fluor_overlay = visualize_fluorescence_measurements(
                fluor_img8, accepted, fluor_measurements
            )
            save_debug(img_out, "21_fluorescence_overlay.png", fluor_overlay, um_per_px_avg)
        else:
            print(f"[WARN] Could not load fluorescence image: {fluor_path}")
    else:
        print(f"[WARN] Fluorescence channel not found: {fluor_path}")

    # Write CSV with both brightfield and fluorescence data
    csv_path = img_out / "object_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Object_ID",
            "Area_px", "Area_um2",
            "Perimeter_px", "Perimeter_um",
            "EquivDiameter_px", "EquivDiameter_um",
            "Circularity", "AspectRatio",
            "CentroidX_px", "CentroidY_px",
            "CentroidX_um", "CentroidY_um",
            "BBoxX_px", "BBoxY_px", "BBoxW_px", "BBoxH_px",
            "BBoxW_um", "BBoxH_um",
            "Fluor_Area_px", "Fluor_Area_um2",
            "Fluor_Mean", "Fluor_Median", "Fluor_Std",
            "Fluor_Min", "Fluor_Max", "Fluor_IntegratedDensity",
        ])

        for i, c in enumerate(accepted, 1):
            area_px = float(cv2.contourArea(c))
            area_um2 = area_px * um2_per_px2

            perim_px = float(cv2.arcLength(c, True))
            perim_um = contour_perimeter_um(c, um_per_px_x, um_per_px_y)

            eqd_px = equivalent_diameter_from_area(area_px)
            eqd_um = equivalent_diameter_from_area(area_um2)

            circ = (4 * np.pi * area_px / (perim_px ** 2)) if perim_px > 0 else 0.0

            x, y, bw, bh = cv2.boundingRect(c)
            aspect = (float(bw) / float(bh)) if bh > 0 else 0.0

            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                cx, cy = 0.0, 0.0

            cx_um = cx * um_per_px_x
            cy_um = cy * um_per_px_y
            bw_um = bw * um_per_px_x
            bh_um = bh * um_per_px_y

            # Get fluorescence data
            if fluor_measurements:
                fm = fluor_measurements[i - 1]
            else:
                fm = {
                    'fluor_area_px': 0.0,
                    'fluor_area_um2': 0.0,
                    'fluor_mean': 0.0,
                    'fluor_median': 0.0,
                    'fluor_std': 0.0,
                    'fluor_min': 0.0,
                    'fluor_max': 0.0,
                    'fluor_integrated_density': 0.0,
                }

            w.writerow([
                i,
                f"{area_px:.2f}", f"{area_um2:.4f}",
                f"{perim_px:.2f}", f"{perim_um:.4f}",
                f"{eqd_px:.2f}", f"{eqd_um:.4f}",
                f"{circ:.4f}", f"{aspect:.4f}",
                f"{cx:.2f}", f"{cy:.2f}",
                f"{cx_um:.4f}", f"{cy_um:.4f}",
                x, y, bw, bh,
                f"{bw_um:.4f}", f"{bh_um:.4f}",
                f"{fm['fluor_area_px']:.2f}", f"{fm['fluor_area_um2']:.4f}",
                f"{fm['fluor_mean']:.2f}", f"{fm['fluor_median']:.2f}",
                f"{fm['fluor_std']:.2f}", f"{fm['fluor_min']:.2f}",
                f"{fm['fluor_max']:.2f}", f"{fm['fluor_integrated_density']:.2f}",
            ])

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

    print(f"\n{'='*80}")
    print(f"SUMMARY: {total_processed} succeeded, {total_failed} failed")

    # Consolidate to Excel AND generate PDF reports
    if SEPARATE_OUTPUT_BY_GROUP:
        for group_dir in OUTPUT_DIR.iterdir():
            if group_dir.is_dir():
                consolidate_to_excel(group_dir, group_dir.name)
                generate_pdf_report(group_dir, group_dir.name)  # ADD THIS LINE
    else:
        consolidate_to_excel(OUTPUT_DIR, "all_groups")
        generate_pdf_report(OUTPUT_DIR, "all_groups")  # ADD THIS LINE


if __name__ == "__main__":
    main()