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

# Handle multi-channel images
if len(img.shape) == 3:
    print(f"Original image shape: {img.shape}")
    # If it has more than 3 channels, take first channel
    if img.shape[2] > 3:
        img = img[:, :, 0]
    # If RGB, convert to grayscale
    elif img.shape[2] == 3:
        img = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
    else:
        img = img[:, :, 0]
        
print(f"Processing image shape: {img.shape}, dtype: {img.dtype}")

# Segment
try:
    # model.eval returns (masks, flows, styles) - 3 values
    masks, flows, styles = model.eval(
        img, 
        diameter=None,  # Auto-estimate
        channels=[0, 0]  # Grayscale: [cytoplasm, nucleus]
    )
    
    # Get diameter from flows if available
    if len(flows) > 2 and flows[2] is not None:
        diam = flows[2]
        print(f"Estimated diameter: {diam}")
    
    num_cells = masks.max()
    print(f"Segmentation completed. Detected {num_cells} cells.")
    
except Exception as e:
    raise RuntimeError(f"Segmentation failed: {e}")

# Generate and save outlines
outlined = plot.outline_view(img, masks)
os.makedirs('./output/beta5', exist_ok=True)
io.imsave('./output/beta5/outlined.png', outlined)

# Save masks
io.imsave('./output/beta5/masks.tif', masks)

print(f"Outputs saved to ./output/beta5/")