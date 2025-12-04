# preprocess.py
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import cv2
import json
import tifffile as tiff

from config import (
    BRIGHTFIELD_CHANNEL_INDEX,
    FLUORESCENCE_CHANNEL_INDEX,
    GLOBAL_PERCENTILE_MIN,
    GLOBAL_PERCENTILE_MAX,
    OUTPUT_IMAGES_DIR,
    OUTPUT_META_DIR,
)
from metadata import ImageMeta, image_meta_to_dict


def load_dual_channel_tiff(img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a dual-channel TIFF as float32 in [0,1] assuming 12-bit data (0-4095).
    Adjust reading logic if your TIFF layout is different.
    """
    arr = tiff.imread(str(img_path))  # shape could be (C,H,W) or (H,W,C)
    arr = np.asarray(arr)

    # Try to infer channel axis
    if arr.ndim == 3:
        if arr.shape[0] in (2, 3, 4):  # (C,H,W)
            bf_raw = arr[BRIGHTFIELD_CHANNEL_INDEX]
            fl_raw = arr[FLUORESCENCE_CHANNEL_INDEX]
        elif arr.shape[-1] in (2, 3, 4):  # (H,W,C)
            bf_raw = arr[..., BRIGHTFIELD_CHANNEL_INDEX]
            fl_raw = arr[..., FLUORESCENCE_CHANNEL_INDEX]
        else:
            raise ValueError(f"Unexpected image shape: {arr.shape}")
    else:
        raise ValueError(f"Expected 3D array, got {arr.shape}")

    bf = bf_raw.astype(np.float32) / 4095.0
    fl = fl_raw.astype(np.float32) / 4095.0
    return bf, fl


def preprocess_brightfield(bf: np.ndarray) -> np.ndarray:
    """
    Background subtraction + contrast enhancement + mild denoising for brightfield.
    Similar in spirit to background subtraction and intensity normalization used for
    holograms prior to reconstruction in the DIH pipeline [[1]][doc_1][[6]][doc_6].
    """
    # Estimate slow background with large Gaussian blur
    background = cv2.GaussianBlur(bf, (0, 0), sigmaX=50, sigmaY=50)
    bf_corr = bf - background
    bf_corr = np.clip(bf_corr, 0, None)

    # CLAHE (works on 8-bit; convert back/forth)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    bf_uint8 = np.clip(bf_corr * 255.0, 0, 255).astype(np.uint8)
    bf_eq = clahe.apply(bf_uint8).astype(np.float32) / 255.0

    # Mild denoising that preserves edges
    bf_denoised = cv2.bilateralFilter(bf_eq, d=5, sigmaColor=0.1, sigmaSpace=5)
    return bf_denoised


def preprocess_fluorescence(fl: np.ndarray) -> np.ndarray:
    """
    Background subtraction + denoising for fluorescence channel.
    """
    # Percentile-based background estimate
    bg = np.percentile(fl, 10)
    fl_corr = fl - bg
    fl_corr = np.clip(fl_corr, 0, None)

    # Mild Gaussian blur for noise reduction
    fl_denoised = cv2.GaussianBlur(fl_corr, (3, 3), 0.5)

    return fl_denoised


def compute_global_percentiles(
    all_bf: List[np.ndarray],
    all_fl: List[np.ndarray],
    p_min: float = GLOBAL_PERCENTILE_MIN,
    p_max: float = GLOBAL_PERCENTILE_MAX,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute global percentile-based normalization parameters across all images.
    Conceptually analogous to using a consistent intensity normalization across
    all holograms in the DIH workflow to stabilize deep learning performance [[3]][doc_3][[5]][doc_5].
    """
    bf_concat = np.concatenate([im.flatten() for im in all_bf])
    fl_concat = np.concatenate([im.flatten() for im in all_fl])

    bf_lo, bf_hi = np.percentile(bf_concat, [p_min, p_max])
    fl_lo, fl_hi = np.percentile(fl_concat, [p_min, p_max])

    return {
        "brightfield": (float(bf_lo), float(bf_hi)),
        "fluorescence": (float(fl_lo), float(fl_hi)),
    }


def normalize_image(
    img: np.ndarray,
    lo: float,
    hi: float,
) -> np.ndarray:
    """
    Normalize image to [0,1] using fixed lo/hi percentiles.
    """
    img_n = (img - lo) / (hi - lo)
    img_n = np.clip(img_n, 0.0, 1.0)
    return img_n


def focus_measure(image: np.ndarray) -> float:
    """
    Simple focus metric: variance of Laplacian on brightfield.
    Used for blur/out-of-focus quality control, analogous to
    ensuring high-quality holograms before analysis [[4]][doc_4][[5]][doc_5].
    """
    lap = cv2.Laplacian(image, cv2.CV_32F)
    return float(lap.var())


def illumination_stats(image: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(image.mean()),
        "median": float(np.median(image)),
        "std": float(image.std()),
    }


def stack_channels(bf: np.ndarray, fl: np.ndarray) -> np.ndarray:
    """
    Channel registration & fusion: here assumed already co-registered.
    Stack into (2, H, W).
    """
    assert bf.shape == fl.shape, "BF and FL shapes must match"
    return np.stack([bf, fl], axis=0)


def save_processed_image(
    out_path: Path,
    img_2ch: np.ndarray,
) -> None:
    """
    Save 2-channel preprocessed image as float32 TIFF (C,H,W).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(out_path), img_2ch.astype(np.float32))


def save_metadata_json(
    out_path: Path,
    meta: ImageMeta,
    extra: Dict[str, Any],
) -> None:
    """
    Save JSON with parsed metadata + pre-processing parameters + QC measures.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = image_meta_to_dict(meta)
    base.update(extra)
    with open(out_path, "w") as f:
        json.dump(base, f, indent=2)

