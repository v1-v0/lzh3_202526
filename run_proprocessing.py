# run_preprocessing.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from config import (
    PROJECT_ROOT,
    SOURCE_ROOT,
    BATCH_IDS,
    METADATA_GLOB,
    OUTPUT_IMAGES_DIR,
    OUTPUT_META_DIR,
    OUTPUT_LABELS_DIR,
    OUTPUT_ROOT
)
from metadata import parse_leica_xml
from preprocess import (
    load_single_channel_tiff,
    preprocess_brightfield,
    preprocess_fluorescence,
    compute_global_percentiles,
    normalize_image,
    stack_channels,
    save_processed_image,
    save_metadata_json,
)
from qc import quality_control
from pairing import pair_bf_fl_images


def get_batch_dirs():
    """
    Generator yielding (batch_id, batch_dir, meta_dir) for configured batches.
    """
    for batch_id in BATCH_IDS:
        batch_dir = SOURCE_ROOT / str(batch_id)
        meta_dir = batch_dir / "MetaData"
        yield batch_id, batch_dir, meta_dir


def find_metadata_for_base(meta_dir: Path, base_id: str) -> Path | None:
    """
    Given MetaData directory and a base_id (e.g. 'sample001'),
    find corresponding XML file where xml.stem == base_id.
    """
    xml_paths = list(meta_dir.glob(METADATA_GLOB))
    for xml_path in xml_paths:
        if xml_path.stem == base_id:
            return xml_path
    return None


def main():
    all_bf_pre: List[np.ndarray] = []
    all_fl_pre: List[np.ndarray] = []
    temp_store: Dict[str, Dict] = {}

    # ---------- First pass: per-image preprocessing, no global normalization ----------
    for batch_id, batch_dir, meta_dir in get_batch_dirs():
        if not batch_dir.exists():
            print(f"[WARN] Batch directory does not exist: {batch_dir}")
            continue
        if not meta_dir.exists():
            print(f"[WARN] MetaData directory does not exist: {meta_dir}")
            continue

        # Images are in ./source/<batch>/ as *_ch00.tif/*_ch01.tif
        pairs = pair_bf_fl_images(batch_dir)

        for bf_path, fl_path, base_id in pairs:
            xml_path = find_metadata_for_base(meta_dir, base_id)
            if xml_path is None:
                print(f"[WARN] No metadata XML found for base_id '{base_id}' in {meta_dir}")
                continue

            try:
                img_meta = parse_leica_xml(xml_path, batch_id=batch_id, image_name=base_id)

                bf_raw = load_single_channel_tiff(bf_path)
                fl_raw = load_single_channel_tiff(fl_path)

                bf_pre = preprocess_brightfield(bf_raw)
                fl_pre = preprocess_fluorescence(fl_raw)
            except Exception as e:
                print(f"[ERROR] Failed to preprocess pair {bf_path.name}, {fl_path.name}: {e}")
                continue

            key = f"batch{batch_id}_{base_id}"
            temp_store[key] = {
                "meta": img_meta,
                "bf_pre": bf_pre,
                "fl_pre": fl_pre,
            }

            all_bf_pre.append(bf_pre)
            all_fl_pre.append(fl_pre)

    print(f"[INFO] Total BF preprocessed images: {len(all_bf_pre)}")
    print(f"[INFO] Total FL preprocessed images: {len(all_fl_pre)}")

    if not all_bf_pre or not all_fl_pre:
        raise RuntimeError(
            "No images were successfully preprocessed; cannot compute global "
            "normalization parameters. Check the warnings/errors above."
        )

    # ---------- Global normalization across all batches ----------
    norm_params = compute_global_percentiles(all_bf_pre, all_fl_pre)
    print("[INFO] Global normalization parameters:")
    print(json.dumps(norm_params, indent=2))

    # ---------- Second pass: normalize, QC, fuse, and save ----------
    for key, info in temp_store.items():
        img_meta = info["meta"]
        bf_pre = info["bf_pre"]
        fl_pre = info["fl_pre"]

        bf_lo, bf_hi = norm_params["brightfield"]
        fl_lo, fl_hi = norm_params["fluorescence"]

        bf_norm = normalize_image(bf_pre, bf_lo, bf_hi)
        fl_norm = normalize_image(fl_pre, fl_lo, fl_hi)

        qc_info = quality_control(bf_norm, fl_norm)

        img_2ch = stack_channels(bf_norm, fl_norm)

        out_img_path = OUTPUT_IMAGES_DIR / f"{key}.tif"
        out_meta_path = OUTPUT_META_DIR / f"{key}.json"
        out_label_path = OUTPUT_LABELS_DIR / f"{key}.txt"

        # Save 2-channel image
        save_processed_image(out_img_path, img_2ch)

        # Save metadata + QC + normalization parameters
        extra_meta = {
            "pixel_size_um": img_meta.pixel_size_m * 1e6,
            "normalization_params": {
                "brightfield": {"lo": bf_lo, "hi": bf_hi},
                "fluorescence": {"lo": fl_lo, "hi": fl_hi},
            },
            "qc": qc_info,
        }
        save_metadata_json(out_meta_path, img_meta, extra_meta)

        # Prepare empty YOLO annotation file (ready for manual/automatic labeling)
        out_label_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_label_path.exists():
            out_label_path.write_text("")

    print("[INFO] Pre-processing complete. Outputs in:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()