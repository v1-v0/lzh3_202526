"""
core/image_loader.py - Image and Metadata Loading
"""
import json
from pathlib import Path
from typing import Optional, Dict
import cv2
import numpy as np


def load_metadata_for_image(image_path: Path) -> Optional[Dict]:
    """Load metadata JSON for current image."""
    metadata_dir = image_path.parent
    metadata_json = metadata_dir / f"{image_path.stem}.json"

    if not metadata_json.exists():
        return None

    try:
        with open(metadata_json, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


def load_image_from_path(image_path: Path) -> Optional[np.ndarray]:
    """Load grayscale image from file."""
    bf = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if bf is None:
        return None

    # Convert to grayscale if needed
    if len(bf.shape) == 3:
        bf = cv2.cvtColor(bf, cv2.COLOR_BGR2GRAY)

    return bf


def get_fluorescence_path(bf_path: Path) -> Optional[Path]:
    """Given BF image (ch00.tif), find matching fluorescence (ch01.tif)."""
    fluor_path = bf_path.parent / bf_path.name.replace("ch00.tif", "ch01.tif")

    if fluor_path.exists() and not fluor_path.name.startswith('.'):
        return fluor_path

    return None


def get_pixel_size(metadata: Dict) -> float:
    """Get pixel size in micrometers from metadata or default."""
    if metadata and 'pixel_size_um' in metadata:
        return float(metadata['pixel_size_um'])
    return 0.1289  # Default


def extract_channel_metadata(metadata: Dict, channel_type: str = 'BF') -> Dict:
    """Extract channel-specific metadata."""
    channel_meta = {
        'intensity_255': 255,
        'exposure_ms': 100,
        'reference_exposure_ms': 100,
        'black_value_normalized': 0.0,
        'white_value_normalized': 1.0,
        'gamma_value': 1.0,
    }

    if 'channels' not in metadata:
        return channel_meta

    channels = metadata['channels']

    for ch_name, ch_data in channels.items():
        method = ch_data.get('contrasting_method')

        if channel_type == 'BF' and method == 'TL-BF':
            channel_meta.update({
                'intensity_255': int(ch_data.get('intensity', 255)),
                'exposure_ms': ch_data.get('exposure_time_s', 0.1) * 1000,
                'black_value_normalized': ch_data.get('black_value_normalized', 0.0),
                'white_value_normalized': ch_data.get('white_value_normalized', 1.0),
                'gamma_value': ch_data.get('gamma_value', 1.0),
            })
            break

        elif channel_type == 'FLUOR' and method == 'FLUO' and 'AutoFocus' not in ch_name:
            channel_meta.update({
                'intensity_255': int(ch_data.get('intensity', 255)),
                'exposure_ms': ch_data.get('exposure_time_s', 0.1) * 1000,
                'black_value_normalized': ch_data.get('black_value_normalized', 0.0),
                'white_value_normalized': ch_data.get('white_value_normalized', 1.0),
                'gamma_value': ch_data.get('gamma_value', 1.0),
            })
            break

    return channel_meta