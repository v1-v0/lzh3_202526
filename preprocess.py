from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_filter
from skimage import exposure
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, remove_small_objects, disk
from skimage.measure import find_contours

from register_dataset import ImagePairRecord

# ---------------------------------------------------------------------------
# Utility: µm <-> pixel
# ---------------------------------------------------------------------------

def um_to_px(length_um: float, rec: ImagePairRecord) -> float:
    """
    Convert a physical length in micrometers to pixels, using metadata.
    Raises if pixel_size_um is not available.
    """
    if rec.pixel_size_um is None:
        raise ValueError(
            f"No pixel_size_um in metadata for pair '{rec.pair_name}'. "
            "Cannot convert µm to pixels."
        )
    return length_um / rec.pixel_size_um


def um_to_px_safe(length_um: float, rec: ImagePairRecord, fallback_px: float) -> float:
    """
    Convert µm to pixels if pixel_size_um is available; otherwise return
    a fixed fallback in pixels. This is useful for viewing / troubleshooting
    when metadata is missing.
    """
    if rec.pixel_size_um is None:
        return fallback_px
    return length_um / rec.pixel_size_um


# ---------------------------------------------------------------------------
# Fluorescence preprocessing (primary segmentation channel)
# ---------------------------------------------------------------------------

def fluo_to_float01(raw: np.ndarray, rec: ImagePairRecord) -> np.ndarray:
    """
    Convert raw FLUO image to float32 in [0,1], using bit depth and (optionally)
    exposure time to standardize intensities across acquisitions.
    """
    arr = raw.astype(np.float32)

    # Bit depth normalization
    if rec.bit_depth is not None:
        max_val = float(2**rec.bit_depth - 1)
    else:
        max_val = float(arr.max() or 1.0)

    arr /= max_val

    # Optional: exposure normalization
    if rec.exposure_fluo_s is not None and rec.exposure_fluo_s > 0:
        ref_exp = 1.0  # arbitrary reference
        arr *= (ref_exp / rec.exposure_fluo_s)

    arr = np.clip(arr, 0.0, 1.0)
    return arr


def subtract_background_fluo(
    img01: np.ndarray,
    rec: ImagePairRecord,
    sigma_bg_um: float = 30.0,
    fallback_sigma_bg_px: float = 50.0,
) -> np.ndarray:
    """
    Subtract low-frequency background from FLUO image using Gaussian blur.

    sigma_bg_um: physical scale of background variations (e.g. 20–50 µm).
    If rec.pixel_size_um is missing, uses fallback_sigma_bg_px directly.
    """
    sigma_bg_px = um_to_px_safe(sigma_bg_um, rec, fallback_px=fallback_sigma_bg_px)
    bg = gaussian_filter(img01, sigma=sigma_bg_px)
    corrected = img01 - bg
    corrected[corrected < 0] = 0.0
    max_val = float(corrected.max() or 1.0)
    corrected /= max_val
    return corrected


def denoise_fluo(
    img01: np.ndarray,
    rec: ImagePairRecord,
    sigma_um: float = 0.5,
    fallback_sigma_px: float = 1.0,
) -> np.ndarray:
    """
    Mild Gaussian denoising tuned to bacterial size (~1–2 µm).

    If rec.pixel_size_um is missing, uses fallback_sigma_px directly.
    """
    sigma_px = um_to_px_safe(sigma_um, rec, fallback_px=fallback_sigma_px)
    return gaussian_filter(img01, sigma=sigma_px)


def normalize_contrast_percentile(
    img: np.ndarray,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> np.ndarray:
    """
    Percentile-based contrast normalization to [0,1].
    """
    img_f = img.astype(np.float32)
    finite = np.isfinite(img_f)
    if not np.any(finite):
        return np.zeros_like(img_f, dtype=np.float32)

    low = np.percentile(img_f[finite], low_pct)
    high = np.percentile(img_f[finite], high_pct)

    if high <= low:
        low = float(img_f[finite].min())
        high = float(img_f[finite].max())
        if high <= low:
            return np.zeros_like(img_f, dtype=np.float32)

    out = (img_f - low) / (high - low)
    out = np.clip(out, 0.0, 1.0)
    return out


def preprocess_fluo_for_seg(
    rec: ImagePairRecord,
    sigma_bg_um: float = 30.0,
    sigma_denoise_um: float = 0.5,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> np.ndarray:
    """
    Full FLUO preprocessing pipeline for segmentation:

      1. Load raw FLUO image
      2. Normalize to [0,1] using bit depth and exposure
      3. Subtract low-frequency background (sigma_bg_um or fallback in px)
      4. Denoise with Gaussian (sigma_denoise_um or fallback in px)
      5. Percentile-based contrast normalization

    Returns float32 image in [0,1].
    """
    raw = tiff.imread(rec.fluo_path)

    img = fluo_to_float01(raw, rec)
    img = subtract_background_fluo(img, rec, sigma_bg_um=sigma_bg_um)
    img = denoise_fluo(img, rec, sigma_um=sigma_denoise_um)
    img = normalize_contrast_percentile(img, low_pct=low_pct, high_pct=high_pct)

    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Brightfield preprocessing (for morphology/texture)
# ---------------------------------------------------------------------------

def bf_to_float01(raw: np.ndarray, rec: ImagePairRecord) -> np.ndarray:
    """
    Convert raw BF image to float32 in [0,1] using bit depth.
    """
    arr = raw.astype(np.float32)
    if rec.bit_depth is not None:
        max_val = float(2**rec.bit_depth - 1)
    else:
        max_val = float(arr.max() or 1.0)
    arr /= max_val
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def correct_illumination_bf(
    img01: np.ndarray,
    rec: ImagePairRecord,
    sigma_bg_um: float = 50.0,
    fallback_sigma_bg_px: float = 80.0,
) -> np.ndarray:
    """
    Correct non-uniform illumination / vignetting in BF using division by a
    blurred background.

    sigma_bg_um: scale of illumination variations (tens of µm).
    If rec.pixel_size_um is missing, uses fallback_sigma_bg_px directly.
    """
    sigma_bg_px = um_to_px_safe(sigma_bg_um, rec, fallback_px=fallback_sigma_bg_px)
    bg = gaussian_filter(img01, sigma=sigma_bg_px)
    bg[bg == 0] = 1.0
    corrected = img01 / bg
    max_val = float(corrected.max() or 1.0)
    corrected /= max_val
    return corrected


def enhance_bf_clahe(
    img01: np.ndarray,
    clip_limit: float = 0.01,
    nbins: int = 256,
) -> np.ndarray:
    """
    Apply CLAHE (adaptive histogram equalization) to enhance local contrast
    in BF images while limiting noise amplification.
    """
    enhanced = exposure.equalize_adapthist(img01, clip_limit=clip_limit, nbins=nbins)
    return enhanced.astype(np.float32)


def preprocess_bf_for_seg(
    rec: ImagePairRecord,
    sigma_bg_um: float = 50.0,
    clip_limit: float = 0.01,
) -> np.ndarray:
    """
    Full BF preprocessing pipeline for segmentation:

      1. Load raw BF image
      2. Normalize to [0,1] using bit depth
      3. Correct illumination (divide by blurred background, µm or fallback px)
      4. Apply CLAHE for local contrast enhancement

    Returns float32 image in [0,1].
    """
    raw = tiff.imread(rec.bf_path)

    img = bf_to_float01(raw, rec)
    img = correct_illumination_bf(img, rec, sigma_bg_um=sigma_bg_um)
    img = enhance_bf_clahe(img, clip_limit=clip_limit)

    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Simple particulate-matter segmentation and contours
# ---------------------------------------------------------------------------

def segment_particles_from_fluo(
    seg_fluo01: np.ndarray,
    min_area_px: int = 20,
) -> np.ndarray:
    """
    Very simple particulate-matter segmentation from preprocessed FLUO image.

    seg_fluo01 can be 2D (H, W) or 3D (H, W, C).
    If 3D, we convert to a single grayscale channel before thresholding.

    Returns a 2D boolean mask (True = particle).
    """
    img = seg_fluo01.astype(np.float32)

    # Ensure grayscale 2D
    if img.ndim == 3:
        img = img.mean(axis=-1)

    thr = threshold_otsu(img)
    mask = img > thr

    mask = mask.astype(bool)
    mask = binary_opening(mask, footprint=disk(1))
    mask = remove_small_objects(mask, min_size=min_area_px)

    return mask


def extract_contours(
    mask: np.ndarray,
    level: float = 0.5,
    min_length: int = 10,
) -> list[np.ndarray]:
    """
    Extract contours from a boolean mask (True = foreground).
    Each contour is an (N, 2) array of (row, col) points.

    min_length filters out very short contours.
    """
    contours = find_contours(mask.astype(float), level=level)
    contours = [c for c in contours if len(c) >= min_length]
    return contours