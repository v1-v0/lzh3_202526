#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bacteria Fluorescence Density Pipeline
--------------------------------------
* Handles filenames with spaces
* 100 % type‑checker clean (mypy / pyright / VS Code)
* Excel + 3 PNGs per image pair
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import measure
from typing import List, Tuple

# --------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------
IMG_DIR = Path("./source/1")          # <-- your folder
BF_SUFFIX   = "_ch00.tif"
FLUO_SUFFIX = "_ch01.tif"

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

MIN_AREA         = 50
WATERSHED_DILATE = 5
OPEN_KERNEL      = 3
CLOSE_KERNEL     = 5

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
        _, warp = cv2.findTransformECC(
            ref8, mov8, warp, cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-5)
        )
    except cv2.error:
        pass
    aligned = cv2.warpPerspective(
        mov8, warp, (ref.shape[1], ref.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )
    return aligned, warp


def segment_bacteria(gray_bf: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    # 1. CLAHE (uint8)
    bf8 = cv2.convertScaleAbs(gray_bf)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bf8)

    # 2. Otsu
    _, thresh = cv2.threshold(enhanced, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Morphology
    open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL, OPEN_KERNEL))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    opened  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k, iterations=2)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_k, iterations=2)

    # 4. Watershed – TYPE‑SAFE (ignore the stub overload)
    distance = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(distance,
                               WATERSHED_DILATE * distance.max() / 100.0,
                               255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, open_k, iterations=1)

    # <-- ONLY LINE WITH # type: ignore -->
    markers: np.ndarray = measure.label(sure_fg, return_num=False)  # type: ignore[assignment]
    markers = markers.astype(np.int32, copy=False)
    markers += 1
    markers[cleaned == 0] = 0

    # OpenCV watershed needs a 3‑channel image
    watershed_input = cv2.cvtColor(gray_bf, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(watershed_input, markers)

    # 5. Contours
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


def measure_fluorescence(fluo: np.ndarray, bacteria) -> List[dict]:
    stats = []
    for cnt, mask, area in bacteria:
        masked = cv2.bitwise_and(fluo, fluo, mask=mask)
        total = np.sum(masked)
        mean = total / area
        stats.append({
            "area_px": area,
            "total_intensity": total,
            "mean_intensity": mean,
            "density": mean
        })
    return stats


def save_visualisations(bf_img, fluo_img, bacteria, base_name):
    def to_bgr(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()

    bf_bgr = to_bgr(bf_img)
    fluo_bgr = to_bgr(fluo_img)

    # BF + contours
    bf_vis = bf_bgr.copy()
    for cnt, _, _ in bacteria:
        cv2.drawContours(bf_vis, [cnt], -1, (0, 255, 0), 1)
    cv2.imwrite(str(OUT_DIR / f"{base_name}_bf_contours.png"), bf_vis)

    # Fluo + contours
    fluo_vis = fluo_bgr.copy()
    for cnt, _, _ in bacteria:
        cv2.drawContours(fluo_vis, [cnt], -1, (0, 255, 0), 1)
    cv2.imwrite(str(OUT_DIR / f"{base_name}_fluo_contours.png"), fluo_vis)

    # Overlay
    overlay = cv2.addWeighted(bf_bgr, 0.6, fluo_bgr, 0.4, 0)
    for cnt, _, _ in bacteria:
        cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 1)
    cv2.imwrite(str(OUT_DIR / f"{base_name}_overlay.png"), overlay)


# --------------------------------------------------------------
# 3. PAIRING (handles spaces)
# --------------------------------------------------------------
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


# --------------------------------------------------------------
# 4. MAIN
# --------------------------------------------------------------
def main():
    pairs = pair_images(IMG_DIR)
    if not pairs:
        raise FileNotFoundError(f"No image pairs in {IMG_DIR}")

    all_results = []
    for bf_path, fluo_path in pairs:
        base = bf_path.stem.replace(BF_SUFFIX, "")
        print(f"Processing: {base}")

        bf_gray   = load_image(bf_path, gray=True)
        fluo_raw  = load_image(fluo_path, gray=True)

        fluo_aligned, _ = align_ecc(bf_gray, fluo_raw)

        bacteria = segment_bacteria(bf_gray)
        stats = measure_fluorescence(fluo_aligned, bacteria)

        for i, s in enumerate(stats):
            s.update({"image": base, "object_id": i + 1})
        all_results.extend(stats)

        save_visualisations(bf_gray, fluo_aligned, bacteria, base)

    df = pd.DataFrame(all_results)
    excel_path = OUT_DIR / "bacteria_measurements.xlsx"
    df.to_excel(excel_path, index=False)

    print(f"\nDone! Excel -> {excel_path}")
    print(f"   {len(df)} bacteria, {len(pairs)} image pairs")
    print(f"   Visuals -> {OUT_DIR}")


if __name__ == "__main__":
    main()