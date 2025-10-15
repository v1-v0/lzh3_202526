from cellpose.models import CellposeModel
from cellpose import io, plot
import os
import numpy as np

# Clear any conflicting environment variables
if 'CELLPOSE_LOCAL_MODELS_PATH' in os.environ:
    del os.environ['CELLPOSE_LOCAL_MODELS_PATH']

# Set up logging for debugging
io.logger_setup()

# Verify image file exists
image_path = './source/1/1 N NO 1_ch00.tif'
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

# Check model directory and ensure 'bacteria' model exists
model_dir = os.path.expanduser('~/.cellpose/models')
print(f"Model directory contents: {os.listdir(model_dir)}")
model_path = os.path.join(model_dir, 'bacteria')
if not os.path.exists(model_path):
    print(f"Model 'bacteria' not found in {model_dir}. Attempting to download...")

# Load model (pre-trained for bacteria)
try:
    model = CellposeModel(pretrained_model=model_path, gpu=False)
    print(f"Loaded model: {model.pretrained_model}")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Load image
img = io.imread(image_path)
if img is None:
    raise ValueError(f"Failed to load image: {image_path}")

# Convert RGB to grayscale if necessary
if len(img.shape) == 3 and img.shape[2] == 3:
    img = np.mean(img.astype(np.float32), axis=2).astype(np.uint8)  # Cast to float32 for np.mean
print(f"Image shape: {img.shape}, dtype: {img.dtype}")

# Segment with flexible return value handling
try:
    result = model.eval(img, diameter=None)  # No channels parameter
    if len(result) == 4:
        masks, flows, styles, diams = result
        print(f"Diameters: {diams}")
    elif len(result) == 3:
        masks, flows, styles = result
        diams = None
        print("Note: model.eval returned 3 values; diams set to None")
    else:
        raise ValueError(f"Unexpected number of return values from model.eval: {len(result)}")
    print(f"Segmentation completed. Detected {len(np.unique(masks))-1} cells.")
except Exception as e:
    raise RuntimeError(f"Segmentation failed: {e}")

# Generate outlines
outlined = plot.outline_view(img, masks)