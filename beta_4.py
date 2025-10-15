from cellpose.models import CellposeModel  # Correct class for model loading
from cellpose import io, plot  # io and plot are top-level modules

import os  # For path checking

# Verify image file exists
image_path = './source/1/1 N NO 1_ch00.tif'
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

# Load model (pre-trained for bacteria)
# Set gpu=True if you have CUDA; otherwise False for CPU
model = CellposeModel(pretrained_model='bacteria', gpu=False)

# Load image
img = io.imread(image_path)

# Segment (channels=[0,0] for grayscale/single-channel)
masks, flows, styles, diams = model.eval(img, diameter=None, channels=[0,0])

# Generate outlines
outlined = plot.outline_view(img, masks)

# Save output
io.imsave('cellpose_outlined.png', outlined)
print("Segmentation complete! Output saved as 'cellpose_outlined.png'.")