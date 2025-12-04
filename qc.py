# qc.py
from typing import Dict, Any
import numpy as np

from preprocess import focus_measure, illumination_stats


def quality_control(
    bf_norm: np.ndarray,
    fl_norm: np.ndarray,
    focus_threshold: float = 5.0,  # tune empirically
    min_mean_intensity: float = 0.02,
    max_mean_intensity: float = 0.98,
) -> Dict[str, Any]:
    """
    Perform basic QC on normalized images:
    - focus measure on BF
    - illumination / intensity sanity checks.
    """
    fm = focus_measure(bf_norm)
    bf_stats = illumination_stats(bf_norm)
    fl_stats = illumination_stats(fl_norm)

    flags = {
        "is_out_of_focus": fm < focus_threshold,
        "bf_too_dark": bf_stats["mean"] < min_mean_intensity,
        "bf_too_bright": bf_stats["mean"] > max_mean_intensity,
        "fl_too_dark": fl_stats["mean"] < min_mean_intensity,
        "fl_too_bright": fl_stats["mean"] > max_mean_intensity,
    }

    return {
        "focus_measure": fm,
        "bf_stats": bf_stats,
        "fl_stats": fl_stats,
        "flags": flags,
    }
