# run_preprocessing.py
from pathlib import Path
from typing import Dict, List
import json

from config import (
    SOURCE_ROOT,
    BATCH_IDS,
    METADATA_GLOB,
    IMAGE_GLOB,
    OUTPUT_IMAGES_DIR,
    OUTPUT_META_DIR,
    OUTPUT_LABELS_DIR,
)
from metadata import parse_leica_xml
from preprocess import (
    load_dual_channel_tiff,
    preprocess_brightfield,
    preprocess_fluorescence,
    compute_global_percentiles,
    normalize_image,
    stack_channels,
    save_processed_image,
    save_metadata_json,
)
from qc import quality_control


def find_matching_metadata(
    metadata_dir: Path,
    image_stem: str,
) -> Path | None:
    """
    Simple heuristic: choose first XML whose name contains image_stem.
    You may want to enforce a stricter naming mapping.
    """
    candidates = list(metadata_dir.glob(METADATA_GLOB))
    for xml_path in candidates:
        if image_stem in xml_path.name:
            return xml_path
    return None


def main():
    # First pass: gather all preprocessed (but not yet globally normalized) images
    # to compute global percentiles.
    all_bf_pre: List = []
    all_fl_pre: List = []
    temp_store: Dict[str, Dict] = {}  # hold temporary results keyed by unique id

    for batch_id in BATCH_IDS:
        batch_dir = SOURCE_ROOT / str(batch_id)
        meta_dir = batch_dir / "MetaData"
        img_dir = batch_dir / "Images"  # adapt to your actual structure

        image_paths = sorted(img_dir.glob(IMAGE_GLOB))
        for img_path in image_paths:
            stem = img_path.stem
            xml_path = find_matching_metadata(meta_dir, stem)
            if xml_path is None:
                print(f"Warning: no metadata found for {img_path}")
                continue

            # Parse metadata
            img_meta = parse_leica_xml(xml_path, batch_id=batch_id, image_name=img_path.name)

            # Load raw channels
            bf_raw, fl_raw = load_dual_channel_tiff(img_path)

            # Channel-specific pre-processing
            bf_pre = preprocess_brightfield(bf_raw)
            fl_pre = preprocess_fluorescence(fl_raw)

            key = f"batch{batch_id}_{stem}"
            temp_store[key] = {
                "meta": img_meta,
                "img_path": img_path,
                "bf_pre": bf_pre,
                "fl_pre": fl_pre,
            }

            all_bf_pre.append(bf_pre)
            all_fl_pre.append(fl_pre)

    # Compute global percentile-based normalization parameters across all batches
    norm_params = compute_global_percentiles(all_bf_pre, all_fl_pre)
    print("Global normalization parameters:", json.dumps(norm_params, indent=2))

    # Second pass: normalize, QC, stack channels, save to disk
    for key, info in temp_store.items():
        img_meta = info["meta"]
        stem = Path(info["img_path"]).stem
        bf_pre = info["bf_pre"]
        fl_pre = info["fl_pre"]

        bf_lo, bf_hi = norm_params["brightfield"]
        fl_lo, fl_hi = norm_params["fluorescence"]

        bf_norm = normalize_image(bf_pre, bf_lo, bf_hi)
        fl_norm = normalize_image(fl_pre, fl_lo, fl_hi)

        # Quality control
        qc_info = quality_control(bf_norm, fl_norm)

        # Channel registration & fusion
        img_2ch = stack_channels(bf_norm, fl_norm)

        # Build output paths
        out_img_path = OUTPUT_IMAGES_DIR / f"{key}.tif"
        out_meta_path = OUTPUT_META_DIR / f"{key}.json"
        out_label_path = OUTPUT_LABELS_DIR / f"{key}.txt"

        # Save preprocessed image
        save_processed_image(out_img_path, img_2ch)

        # Save metadata + QC + norm params used
        extra_meta = {
            "pixel_size_um": img_meta.pixel_size_m * 1e6,
            "normalization_params": {
                "brightfield": {"lo": bf_lo, "hi": bf_hi},
                "fluorescence": {"lo": fl_lo, "hi": fl_hi},
            },
            "qc": qc_info,
        }
        save_metadata_json(out_meta_path, img_meta, extra_meta)

        # Prepare empty YOLO annotation file (one line per object later)
        out_label_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_label_path.exists():
            out_label_path.write_text("")  # placeholder for manual or automated labels later

    print("Pre-processing complete.")


if __name__ == "__main__":
    main()

