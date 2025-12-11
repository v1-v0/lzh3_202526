# config.py
from pathlib import Path

# Root of the project (this file's directory)
PROJECT_ROOT = Path(__file__).resolve().parent

# Source data root: ./source/<batch>/
SOURCE_ROOT = PROJECT_ROOT / "source"

# Batches 10..19 inclusive
BATCH_IDS = list(range(10, 20))
BATCH_IDS = list(range(10, 12))  # for testing, smaller range

# Output root
OUTPUT_ROOT = PROJECT_ROOT / "processed"
OUTPUT_IMAGES_DIR = OUTPUT_ROOT / "images"
OUTPUT_META_DIR = OUTPUT_ROOT / "meta"
OUTPUT_LABELS_DIR = OUTPUT_ROOT / "labels"  # YOLO txt labels

# Metadata file pattern
METADATA_GLOB = "*.xml"

# Image naming conventions in each batch dir:
#   <base_id>_ch00.tif  -> brightfield
#   <base_id>_ch01.tif  -> fluorescence
IMAGE_BF_SUFFIX = "_ch00.tif"
IMAGE_FL_SUFFIX = "_ch01.tif"

# Bit depth of raw TIFFs (Leica exports often 12 or 16 bits)
BIT_DEPTH = 12  # adjust if needed, e.g. 16

# Global normalization percentiles
GLOBAL_PERCENTILE_MIN = 1
GLOBAL_PERCENTILE_MAX = 99