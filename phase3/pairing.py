# pairing.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from config import IMAGE_BF_SUFFIX, IMAGE_FL_SUFFIX


def pair_bf_fl_images(batch_dir: Path) -> List[Tuple[Path, Path, str]]:
    """
    Find all brightfield images (*_ch00.tif) in batch_dir and pair each with
    the corresponding fluorescence image (*_ch01.tif).

    Returns list of (bf_path, fl_path, base_id).
    base_id is the filename without the _ch00/_ch01 suffix.
    """
    bf_paths = sorted(batch_dir.glob(f"*{IMAGE_BF_SUFFIX}"))
    pairs: List[Tuple[Path, Path, str]] = []

    for bf_path in bf_paths:
        name = bf_path.name
        if not name.endswith(IMAGE_BF_SUFFIX):
            continue

        base_id = name[:-len(IMAGE_BF_SUFFIX)]
        fl_name = base_id + IMAGE_FL_SUFFIX
        fl_path = batch_dir / fl_name

        if fl_path.exists():
            pairs.append((bf_path, fl_path, base_id))
        else:
            print(f"[WARN] Fluorescence image not found for {bf_path.name} (expected {fl_name})")

    print(f"[INFO] {batch_dir}: found {len(pairs)} BF/FL pairs")
    return pairs