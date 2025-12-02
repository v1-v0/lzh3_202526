"""
core/normalization.py - Image Normalization with Metadata
"""
import numpy as np
from typing import Dict


def normalize_image_with_metadata(raw_image_12bit: np.ndarray, 
                                  channel_metadata: Dict) -> np.ndarray:
    """
    Normalize raw microscope image using acquisition metadata.

    Corrects for:
    1. Different intensity settings (0-255 range)
    2. Exposure time variations
    3. Display scaling (black/white points)
    4. Gamma correction
    """

    # Extract metadata with defaults
    intensity_255 = channel_metadata.get('intensity_255', 255)
    exposure_ms = channel_metadata.get('exposure_ms', 100)
    reference_exposure_ms = channel_metadata.get('reference_exposure_ms', 100)
    black_val_norm = channel_metadata.get('black_value_normalized', 0.0)
    white_val_norm = channel_metadata.get('white_value_normalized', 1.0)
    gamma = channel_metadata.get('gamma_value', 1.0)

    # Convert raw image to float for processing
    img_float = raw_image_12bit.astype(np.float32)

    # STEP 1: Normalize by metadata intensity
    img_max = img_float.max()
    if img_max > 0:
        img_normalized = img_float * intensity_255 / img_max
    else:
        img_normalized = img_float

    # STEP 2: Correct for exposure time
    if exposure_ms > 0:
        exposure_factor = reference_exposure_ms / exposure_ms
        img_normalized = img_normalized * exposure_factor

    img_normalized = np.clip(img_normalized, 0, 255)

    # STEP 3: Apply display scaling (black/white points)
    if white_val_norm > black_val_norm:
        black_px = black_val_norm * 255
        white_px = white_val_norm * 255

        img_normalized = (img_normalized - black_px) / (white_px - black_px + 1e-6)
        img_normalized = np.clip(img_normalized, 0, 1)

    # STEP 4: Apply gamma correction
    if gamma != 1.0:
        img_normalized = np.power(np.clip(img_normalized, 0, 1), gamma)

    # Convert to 8-bit
    img_8bit = (img_normalized * 255).astype(np.uint8)

    return img_8bit