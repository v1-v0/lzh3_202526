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


# ==================================================
# Logging: tee stdout/stderr to a file
# ==================================================
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

# Scale bar parameters (matching dev.py)
SCALE_BAR_LENGTH_UM = 10
SCALE_BAR_HEIGHT = 4
SCALE_BAR_MARGIN = 15
SCALE_BAR_COLOR = (255, 255, 255)
SCALE_BAR_BG_COLOR = (0, 0, 0)
SCALE_BAR_TEXT_COLOR = (255, 255, 255)
SCALE_BAR_FONT_SCALE = 0.5
SCALE_BAR_FONT_THICKNESS = 1

# --- Segmentation (debug-meta style) ---
# Background estimation blur (px). Dark objects become bright via (bg - img).
GAUSSIAN_SIGMA = 15

# Morphology (match debug-meta)
MORPH_KERNEL_SIZE = 3
MORPH_ITERATIONS = 1   # close
DILATE_ITERATIONS = 1
ERODE_ITERATIONS = 1

# Filtering (in micrometers; converted to px using metadata)
MIN_AREA_UM2 = 5.0
MAX_AREA_UM2 = 2000.0
MIN_CIRCULARITY = 0.0

# Safety filter: reject "giant background contour" if it covers too much of the image
MAX_FRACTION_OF_IMAGE_AREA = 0.25  # 25%

# Debug options
CLEAR_OUTPUT_DIR_EACH_RUN = True

# Output separation (recommended): debug/<group>/<image_stem>/...
SEPARATE_OUTPUT_BY_GROUP = True

# If metadata is missing, fall back to this pixel size instead of erroring.
# (Set to None to keep strict behavior.)
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
    """
    Return a selected group folder, or None if user chooses 'all'.
    """
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
        base = base[:-5]  # strip "_ch00"
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
    # ---- Try Properties.xml first (µm) ----
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

    # ---- Fallback to main xml (meters) ----
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
# Segmentation (Option A: debug-meta style, single pipeline)
# ==================================================
def segment_particles_brightfield(img8: np.ndarray, pixel_size_um: float, out_dir: Path) -> np.ndarray:
    """
    debug-meta style segmentation for brightfield:
      bg = GaussianBlur(img)
      enhanced = bg - img   (dark objects -> bright)
      small blur
      Otsu threshold
      morph close/dilate/erode
    Returns: binary mask (uint8 0/255)
    """
    # Background estimation and enhancement (dark -> bright)
    bg = cv2.GaussianBlur(img8, (0, 0), sigmaX=GAUSSIAN_SIGMA, sigmaY=GAUSSIAN_SIGMA)
    enhanced = cv2.subtract(bg, img8)
    enhanced_blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

    save_debug(out_dir, "02_enhanced.png", enhanced, pixel_size_um)
    save_debug(out_dir, "03_enhanced_blur.png", enhanced_blur, pixel_size_um)

    # Otsu threshold on enhanced image (not raw gray)
    _, thresh = cv2.threshold(enhanced_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_debug(out_dir, "04_thresh_raw.png", thresh, pixel_size_um)

    # Morphology (match debug-meta)
    
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)

    bw = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)
    bw = cv2.dilate(bw, kernel, iterations=DILATE_ITERATIONS)
    bw = cv2.erode(bw, kernel, iterations=ERODE_ITERATIONS)
    save_debug(out_dir, "05_closed.png", bw, pixel_size_um)

    # Optional: solidify via connected components (commented by default)
    # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    # bw = np.where(labels > 0, 255, 0).astype(np.uint8)
    # save_debug(out_dir, "06_solid.png", bw, pixel_size_um)

    print(f"Mask white fraction (final): {float((bw > 0).mean()):.4f}")
    return bw


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

    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(str(img_path))
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"Loaded: dtype={img.dtype}, shape={img.shape}, range=[{img.min()}-{img.max()}]")

    img8 = normalize_to_8bit(img)
    save_debug(img_out, "01_gray_8bit.png", img8, um_per_px_avg)

    mask = segment_particles_brightfield(img8, um_per_px_avg, img_out)

    # Find contours from mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    print(f"Contours found (pre-filter): {len(contours)}")

    vis_all = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis_all, contours, -1, (0, 0, 255), 1)
    save_debug(img_out, "10_contours_all.png", vis_all, um_per_px_avg)

    # Convert µm² thresholds to px² (anisotropic handled)
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

        # Reject giant "background" blob early
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

    # Visualization similar to your expected overlay
    vis_acc = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis_acc, rejected, -1, (0, 165, 255), 1)  # orange
    cv2.drawContours(vis_acc, accepted, -1, (0, 0, 255), 1)  # red outline for accepted
    vis_acc = draw_object_ids(vis_acc, accepted)
    save_debug(img_out, "11_contours_rejected_orange_accepted_red_ids_green.png", vis_acc, um_per_px_avg)

    # Masks
    mask_all = np.zeros_like(mask)
    cv2.drawContours(mask_all, contours, -1, 255, thickness=-1)
    save_debug(img_out, "12_mask_all.png", mask_all)

    mask_acc = np.zeros_like(mask)
    cv2.drawContours(mask_acc, accepted, -1, 255, thickness=-1)
    save_debug(img_out, "13_mask_accepted.png", mask_acc)

    # Stats CSV
    csv_path = img_out / "object_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Object_ID",
                "Area_px",
                "Area_um2",
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
            ]
        )

        for i, c in enumerate(accepted, 1):
            area_px = float(cv2.contourArea(c))
            area_um2 = area_px * (um_per_px_x * um_per_px_y)

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

            w.writerow(
                [
                    i,
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
                ]
            )

    print(f"CSV saved: {csv_path} ({len(accepted)} objects)")
    print("✓ Done")


def main() -> None:
    if CLEAR_OUTPUT_DIR_EACH_RUN:
        clear_output_dir(OUTPUT_DIR)

    print(f"Input dir: {(_project_root).resolve()}")
    groups = list_sample_group_folders(SOURCE_DIR)
    selected_group_dir = prompt_user_select_group(groups)

    # Build list of folders to process
    if selected_group_dir is None:
        dirs_to_process = groups[:]  # all numeric groups
    else:
        dirs_to_process = [selected_group_dir]

    # Always process Control group too
    if CONTROL_DIR.exists():
        dirs_to_process.append(CONTROL_DIR)

    # Collect images
    img_paths: list[Path] = []
    for d in dirs_to_process:
        img_paths.extend(sorted(d.rglob(IMAGE_GLOB)))

    print(f"Found {len(img_paths)} brightfield images matching '{IMAGE_GLOB}'")
    if not img_paths:
        raise FileNotFoundError(f"No images found under {SOURCE_DIR} matching {IMAGE_GLOB}")

    total_processed = 0
    total_failed = 0
    
    # Progress bar with tqdm
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


if __name__ == "__main__":
    main()