# config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "source"
BATCH_IDS = list(range(10, 20))  # 10..19 inclusive

# Where to write pre-processed data
OUTPUT_ROOT = PROJECT_ROOT / "processed"
OUTPUT_IMAGES_DIR = OUTPUT_ROOT / "images"
OUTPUT_META_DIR = OUTPUT_ROOT / "meta"
OUTPUT_LABELS_DIR = OUTPUT_ROOT / "labels"  # for YOLO txt files

# File patterns – adapt to your naming scheme
METADATA_GLOB = "*.xml"
IMAGE_GLOB = "*.tif"  # or "*.ome.tif" or whatever your Leica export uses

# Channels: index mapping if needed
BRIGHTFIELD_CHANNEL_INDEX = 0  # in a multi-channel TIFF
FLUORESCENCE_CHANNEL_INDEX = 1

# Normalization parameters (will be computed globally later)
GLOBAL_PERCENTILE_MIN = 1
GLOBAL_PERCENTILE_MAX = 99
