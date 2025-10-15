from cellpose.models import CellposeModel
from cellpose import io, plot
import os
import numpy as np

# Set up logging for debugging
io.logger_setup()

# Verify image file exists
image_path = './source/1/1 N NO 1_ch00.tif'
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

# Load model (pre-trained for bacteria)
model = CellposeModel(model_type='bacteria', gpu=False)
print(f"Loaded model: bacteria")

# Load image
img = io.imread(image_path)
if img is None:
    raise ValueError(f"Failed to load image: {image_path}")

# Convert RGB to grayscale if necessary
if len(img.shape) == 3 and img.shape[2] == 3:
    img = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
print(f"Image shape: {img.shape}, dtype: {img.dtype}")

# Segment
try:

    # To get diameter more explicitly:
    masks, flows, styles = model.eval(img, diameter=None, channels=[0, 0])

    # flows is a list: [flow_in_XY, cellprob, estimated_diameter]
    diam = flows[2] if len(flows) > 2 else None

    diams = model.eval(
        img, 
        diameter=None,  # Auto-estimate, or specify e.g., 30 for bacteria
        channels=[0, 0]  # Grayscale
    )
    print(f"Estimated diameter: {diams}")
    print(f"Segmentation completed. Detected {masks.max()} cells.")
except Exception as e:
    raise RuntimeError(f"Segmentation failed: {e}")

# Generate and save outlines
outlined = plot.outline_view(img, masks)
os.makedirs('./output', exist_ok=True)
io.imsave('./output/outlined.png', outlined)

# Optionally save masks
io.imsave('./output/masks.tif', masks)