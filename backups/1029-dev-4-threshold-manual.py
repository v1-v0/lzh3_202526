#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bacteria Fluorescence Density Pipeline
- RED fluorescence preserved
- FULLY CONFIGURABLE brightfield threshold & segmentation
- Labels on contours
- Brightfield AND fluorescence intensity measurements
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import measure
from typing import List, Tuple

# --------------------------------------------------------------
# 1. CONFIGURATION – SEGMENTATION PARAMETERS
# --------------------------------------------------------------
IMG_DIR = Path("./source/1")
BF_SUFFIX   = "_ch00.tif"
FLUO_SUFFIX = "_ch01.tif"

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

# ------------------- Brightfield Thresholding -------------------
USE_OTSU_THRESHOLD = False          # Set False → use MANUAL_THRESHOLD
MANUAL_THRESHOLD   = 50           # Only used if above is False, Lower (20-40) = only darkest objects; Higher (60-100) = more sensitive but noisy

# ------------------- Contrast Enhancement -------------------
ENABLE_CLAHE       = True
CLAHE_CLIP_LIMIT   = 5.0 # 1.0-10.0 typical 2.0 to 4.0
CLAHE_TILE_SIZE    = (16, 16) # 4,4 to 32,32

# ------------------- Morphology Cleanup -------------------
OPEN_KERNEL_SIZE   = 3 # 1-15 odd number only
CLOSE_KERNEL_SIZE  = 5 # 1-15 odd number only
OPEN_ITERATIONS    = 3 # 1-5
CLOSE_ITERATIONS   = 2 # 1-5

# ------------------- Watershed Separation -------------------
MIN_AREA           = 50 # 20-200 for bacteria
WATERSHED_DILATE   = 15 # % of max distance transform value (1-20)

# ------------------- Label Display Settings -------------------
LABEL_FONT_SCALE   = 0.4
LABEL_THICKNESS    = 1
LABEL_COLOR        = (255, 255, 0)  # Cyan for visibility

# --------------------------------------------------------------
# 2. HELPERS
# --------------------------------------------------------------
def load_image(path: Path, gray: bool = True) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    if gray and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def align_ecc(ref: np.ndarray, mov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ref8 = cv2.convertScaleAbs(ref) if ref.dtype != np.uint8 else ref
    mov8 = cv2.convertScaleAbs(mov) if mov.dtype != np.uint8 else mov
    warp = np.eye(3, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(ref8, mov8, warp, cv2.MOTION_EUCLIDEAN,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-5))
    except cv2.error:
        pass
    aligned = cv2.warpPerspective(mov8, warp, (ref.shape[1], ref.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return aligned, warp


def segment_bacteria(gray_bf: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    if ENABLE_CLAHE:
        bf8 = cv2.convertScaleAbs(gray_bf)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
        enhanced = clahe.apply(bf8)
    else:
        enhanced = cv2.convertScaleAbs(gray_bf)

    if USE_OTSU_THRESHOLD:
        _, thresh = cv2.threshold(enhanced, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(enhanced, MANUAL_THRESHOLD, 255,
                                  cv2.THRESH_BINARY_INV)

    open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE))
    opened  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k, iterations=OPEN_ITERATIONS)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_k, iterations=CLOSE_ITERATIONS)

    distance = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(distance, WATERSHED_DILATE * distance.max() / 100.0, 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, open_k, iterations=1)

    markers: np.ndarray = measure.label(sure_fg, return_num=False)  # type: ignore[assignment]
    markers = markers.astype(np.int32, copy=False)
    markers += 1
    markers[cleaned == 0] = 0

    watershed_input = cv2.cvtColor(gray_bf, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(watershed_input, markers)

    contours, _ = cv2.findContours(
        (markers > 1).astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bacteria = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        mask = np.zeros(gray_bf.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
        bacteria.append((cnt, mask, area))
    return bacteria


def measure_intensities(bf_gray: np.ndarray, fluo_gray: np.ndarray, bacteria) -> List[dict]:
    stats = []
    for cnt, mask, area in bacteria:
        # Fluorescence measurements
        masked_fluo = cv2.bitwise_and(fluo_gray, fluo_gray, mask=mask)
        total_fluo = np.sum(masked_fluo)
        mean_fluo = total_fluo / area
        
        # Brightfield measurements
        masked_bf = cv2.bitwise_and(bf_gray, bf_gray, mask=mask)
        total_bf = np.sum(masked_bf)
        mean_bf = total_bf / area
        
        stats.append({
            "area_px": area,
            "bf_total_intensity": total_bf,
            "bf_mean_intensity": mean_bf,
            "fluo_total_intensity": total_fluo,
            "fluo_mean_intensity": mean_fluo,
            "fluo_density": mean_fluo
        })
    return stats


def save_visualisations(bf_gray, fluo_color, bacteria, base_name):
    bf_bgr = cv2.cvtColor(bf_gray, cv2.COLOR_GRAY2BGR) if len(bf_gray.shape) == 2 else bf_gray.copy()

    # BF + green contours + labels
    bf_vis = bf_bgr.copy()
    for idx, (cnt, _, _) in enumerate(bacteria, 1):
        cv2.drawContours(bf_vis, [cnt], -1, (0, 255, 0), 1)
        # Add label at centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(bf_vis, str(idx), (cx - 10, cy + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE,
                       LABEL_COLOR, LABEL_THICKNESS)
    cv2.imwrite(str(OUT_DIR / f"{base_name}_bf_contours.png"), bf_vis)

    # FLUO (RED) + green contours + labels
    fluo_vis = fluo_color.copy()
    for idx, (cnt, _, _) in enumerate(bacteria, 1):
        cv2.drawContours(fluo_vis, [cnt], -1, (0, 255, 0), 1)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(fluo_vis, str(idx), (cx - 10, cy + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE,
                       LABEL_COLOR, LABEL_THICKNESS)
    cv2.imwrite(str(OUT_DIR / f"{base_name}_fluo_red_contours.png"), fluo_vis)

    # Overlay: BF + RED + labels
    red_channel = fluo_color[:, :, 2]
    red_pseudo = np.zeros_like(fluo_color)
    red_pseudo[:, :, 2] = red_channel
    overlay = cv2.addWeighted(bf_bgr, 0.6, red_pseudo, 0.4, 0)
    for idx, (cnt, _, _) in enumerate(bacteria, 1):
        cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 1)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(overlay, str(idx), (cx - 10, cy + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE,
                       LABEL_COLOR, LABEL_THICKNESS)
    cv2.imwrite(str(OUT_DIR / f"{base_name}_overlay.png"), overlay)


def pair_images(img_dir: Path):
    bf_paths = sorted(img_dir.glob(f"*{BF_SUFFIX}"))
    pairs = []
    for bf in bf_paths:
        prefix = bf.name[:-len(BF_SUFFIX)]
        fluo = img_dir / f"{prefix}{FLUO_SUFFIX}"
        if fluo.exists():
            pairs.append((bf, fluo))
        else:
            cand = list(img_dir.glob(f"{prefix}*_ch01.tif"))
            if cand:
                pairs.append((bf, cand[0]))
            else:
                print(f"Warning: No fluorescence for {bf.name}")
    return pairs


def main():
    pairs = pair_images(IMG_DIR)
    if not pairs:
        raise FileNotFoundError(f"No image pairs in {IMG_DIR}")

    all_results = []
    for bf_path, fluo_path in pairs:
        base = bf_path.stem.replace(BF_SUFFIX, "")
        print(f"Processing: {base}")

        bf_gray   = load_image(bf_path, gray=True)
        fluo_color = load_image(fluo_path, gray=False)
        fluo_gray  = cv2.cvtColor(fluo_color, cv2.COLOR_BGR2GRAY)

        fluo_gray_aligned, warp = align_ecc(bf_gray, fluo_gray)
        fluo_color_aligned = cv2.warpPerspective(
            fluo_color, warp, (bf_gray.shape[1], bf_gray.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )

        bacteria = segment_bacteria(bf_gray)
        stats = measure_intensities(bf_gray, fluo_gray_aligned, bacteria)

        for i, s in enumerate(stats):
            s.update({"image": base, "object_id": i + 1})
        all_results.extend(stats)

        save_visualisations(bf_gray, fluo_color_aligned, bacteria, base)

    df = pd.DataFrame(all_results)
    # Reorder columns for better readability
    column_order = ["image", "object_id", "area_px", 
                    "bf_total_intensity", "bf_mean_intensity",
                    "fluo_total_intensity", "fluo_mean_intensity", "fluo_density"]
    df = df[column_order]
    
    excel_path = OUT_DIR / "bacteria_measurements.xlsx"
    df.to_excel(excel_path, index=False)

    print(f"\nDone! Excel → {excel_path}")
    print(f"   {len(df)} bacteria, {len(pairs)} image pairs")
    print(f"   Visuals (with RED fluorescence + labels) → {OUT_DIR}")
    print(f"\nMeasurements include:")
    print(f"   - Brightfield intensity (total & mean)")
    print(f"   - Fluorescence intensity (total & mean)")


if __name__ == "__main__":
    main()