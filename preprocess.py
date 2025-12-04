# preprocess.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import cv2
import tifffile as tiff
import json

from config import (
    BIT_DEPTH,
    GLOBAL_PERCENTILE_MIN,
    GLOBAL_PERCENTILE_MAX,
    OUTPUT_IMAGES_DIR,
    OUTPUT_META_DIR,
)
from metadata import ImageMeta, image_meta_to_dict


# ---------- I/O ----------
def load_single_channel_tiff(img_path: Path, bit_depth: int = BIT_DEPTH) -> np.ndarray:
    """
    Load a TIFF and return a single-channel float32 image in [0,1].

    Handles:
      - 2D grayscale (H, W)
      - 3D with leading singleton (1, H, W)
      - 3-channel RGB (H, W, 3) -> converted to grayscale
    """
    arr = tiff.imread(str(img_path))
    arr = np.asarray(arr)

    # 2D grayscale
    if arr.ndim == 2:
        raw = arr

    # (1, H, W)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        raw = arr[0]

    # (H, W, 3) RGB
    elif arr.ndim == 3 and arr.shape[2] == 3:
        rgb = arr.astype(np.float32)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        # Standard luminance weights
        raw = 0.299 * r + 0.587 * g + 0.114 * b

    else:
        raise ValueError(
            f"Unexpected shape for image {img_path}: {arr.shape}. "
            "Expected (H,W), (1,H,W), or (H,W,3)."
        )

    # Choose normalization max
    if np.issubdtype(raw.dtype, np.integer):
        # Use the string name to satisfy NumPy typing stubs
        max_val = float(np.iinfo(raw.dtype.name).max)
    else:
        max_val = float((1 << bit_depth) - 1)

    img = raw.astype(np.float32) / max_val
    img = np.clip(img, 0.0, 1.0)
    return img


# ---------- Channel-specific preprocessing ----------

def preprocess_brightfield(bf: np.ndarray) -> np.ndarray:
    """
    Brightfield preprocessing:
    - Smooth background estimate and subtraction
    - Contrast enhancement (CLAHE)
    - Mild denoising

    This is analogous in spirit to the background subtraction and intensity
    normalization applied to holographic images before DIH reconstruction
    and YOLO detection on PD fluids [[1]][doc_1][[3]][doc_3][[6]][doc_6].
    """
    # Large Gaussian to estimate slow background
    background = cv2.GaussianBlur(bf, (0, 0), sigmaX=50, sigmaY=50)
    bf_corr = bf - background
    bf_corr = np.clip(bf_corr, 0, None)

    # CLAHE on 8-bit representation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    bf_uint8 = np.clip(bf_corr * 255.0, 0, 255).astype(np.uint8)
    bf_eq = clahe.apply(bf_uint8).astype(np.float32) / 255.0

    # Mild edge-preserving denoising
    bf_denoised = cv2.bilateralFilter(bf_eq, d=5, sigmaColor=0.1, sigmaSpace=5)
    return bf_denoised


def preprocess_fluorescence(fl: np.ndarray) -> np.ndarray:
    """
    Fluorescence preprocessing:
    - Percentile-based background subtraction
    - Mild Gaussian denoising

    Designed to emphasize labeled bacteria/particles while suppressing
    camera noise and uniform background, analogous to DIH intensity
    enhancement before feeding data to a deep learning model [[2]][doc_2][[4]][doc_4][[6]][doc_6].
    """
    # Simple background estimate: low percentile
    bg = np.percentile(fl, 10)
    fl_corr = fl - bg
    fl_corr = np.clip(fl_corr, 0, None)

    # Mild Gaussian blur to reduce shot noise
    fl_denoised = cv2.GaussianBlur(fl_corr, (3, 3), 0.5)

    return fl_denoised


# ---------- Global normalization across all images ----------

def compute_global_percentiles(
    all_bf: List[np.ndarray],
    all_fl: List[np.ndarray],
    p_min: float = GLOBAL_PERCENTILE_MIN,
    p_max: float = GLOBAL_PERCENTILE_MAX,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute global normalization parameters across all preprocessed BF/FL images.

    This is analogous to using a consistent intensity normalization scheme
    across all holograms in the DIH PD-fluid workflow, stabilizing the
    YOLO-based detection across different experimental batches [[3]][doc_3][[5]][doc_5][[6]][doc_6].
    """
    if not all_bf or not all_fl:
        raise RuntimeError(
            "No preprocessed images passed to compute_global_percentiles. "
            "Check that your first pass actually processed some images."
        )

    bf_concat = np.concatenate([im.flatten() for im in all_bf])
    fl_concat = np.concatenate([im.flatten() for im in all_fl])

    bf_lo, bf_hi = np.percentile(bf_concat, [p_min, p_max])
    fl_lo, fl_hi = np.percentile(fl_concat, [p_min, p_max])

    return {
        "brightfield": (float(bf_lo), float(bf_hi)),
        "fluorescence": (float(fl_lo), float(fl_hi)),
    }


def normalize_image(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Normalize image to [0,1] based on fixed lo/hi.

    Anything below lo maps to 0, above hi to 1.
    """
    img_n = (img - lo) / (hi - lo)
    img_n = np.clip(img_n, 0.0, 1.0)
    return img_n


# ---------- Quality control ----------

def focus_measure(image: np.ndarray) -> float:
    """
    Simple focus metric: variance of Laplacian.
    Typically used on brightfield to detect out-of-focus frames.
    """
    lap = cv2.Laplacian(image, cv2.CV_32F)
    return float(lap.var())


def illumination_stats(image: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(image.mean()),
        "median": float(np.median(image)),
        "std": float(image.std()),
    }


# ---------- Channel registration & fusion ----------

def stack_channels(bf: np.ndarray, fl: np.ndarray) -> np.ndarray:
    """
    Stack BF and FL into a 2-channel array (C,H,W).
    Assumes channels are already spatially aligned (same field of view).
    """
    assert bf.shape == fl.shape, "Brightfield and fluorescence shapes must match"
    return np.stack([bf, fl], axis=0)


# ---------- Saving utilities ----------

def save_processed_image(out_path: Path, img_2ch: np.ndarray) -> None:
    """
    Save (C,H,W) float32 image as TIFF.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(out_path), img_2ch.astype(np.float32))


def save_metadata_json(out_path: Path, meta: ImageMeta, extra: Dict[str, Any]) -> None:
    """
    Save JSON with parsed metadata + additional info (QC, normalization, etc.).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = image_meta_to_dict(meta)
    base.update(extra)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=2)