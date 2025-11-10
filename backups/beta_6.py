from cellpose.models import CellposeModel
from cellpose import io, plot
import os
import numpy as np
import torch
import platform
import matplotlib.pyplot as plt

# Set up logging for debugging
io.logger_setup()

# Intelligent GPU detection and configuration
def configure_gpu():
    """Auto-detect and configure GPU for Cellpose across different platforms."""
    system = platform.system()
    
    # Apple Silicon (MLX/MPS)
    if system == "Darwin" and platform.processor() == "arm":
        try:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("✓ Apple Silicon detected - using MPS (Metal)")
                return True, "mps"
            else:
                print("⚠ Apple Silicon detected but MPS unavailable - using CPU")
                return False, "cpu"
        except (AttributeError, Exception):
            print("⚠ MPS not supported - using CPU")
            return False, "cpu"
    
    # NVIDIA CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ NVIDIA GPU detected: {gpu_name}")
        
        # Safe version check with hasattr
        if hasattr(torch, 'version') and hasattr(torch.version, 'cuda'): # type: ignore
            cuda_version = torch.version.cuda  # type: ignore
            print(f"  CUDA version: {cuda_version}")
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Available memory: {total_memory:.2f} GB")
        return True, "cuda"
    
    # AMD ROCm
    try:
        if hasattr(torch, 'version') and hasattr(torch.version, 'hip'): # type: ignore
            hip_version = torch.version.hip  # type: ignore
            if hip_version is not None:
                print(f"✓ AMD GPU detected - using ROCm")
                print(f"  ROCm version: {hip_version}")
                return True, "rocm"
    except (AttributeError, Exception):
        pass
    
    # Fallback to CPU
    print("⚠ No GPU detected - using CPU")
    return False, "cpu"

# Verify image file exists
image_path = './source/1/1 N NO 1_ch00.tif'
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

# Configure and load model
use_gpu, device_type = configure_gpu()

try:
    model = CellposeModel(model_type='bacteria', gpu=use_gpu)
    print(f"✓ Loaded model: bacteria (device: {device_type})")
except Exception as e:
    print(f"⚠ GPU initialization failed: {e}")
    print("  Falling back to CPU...")
    use_gpu = False
    device_type = "cpu"
    model = CellposeModel(model_type='bacteria', gpu=False)
    print(f"✓ Loaded model: bacteria (device: {device_type})")

# Load image
img = io.imread(image_path)
if img is None:
    raise ValueError(f"Failed to load image: {image_path}")

# Handle multi-channel images
if len(img.shape) == 3:
    print(f"Original image shape: {img.shape}")
    if img.shape[2] > 3:
        img = img[:, :, 0]
    elif img.shape[2] == 3:
        img = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
    else:
        img = img[:, :, 0]
        
print(f"Processing image shape: {img.shape}, dtype: {img.dtype}")

# Segment
try:
    import time
    start_time = time.time()
    
    masks, flows, styles = model.eval(
        img, 
        diameter=None,
        channels=[0, 0]
    )
    
    elapsed_time = time.time() - start_time
    print(f"Segmentation time: {elapsed_time:.2f}s on {device_type}")
    
    # Extract diameter properly
    if len(flows) > 2 and flows[2] is not None:
        diam = flows[2]
        # Handle both scalar and array cases
        if isinstance(diam, np.ndarray):
            diam_value = float(diam.item()) if diam.size == 1 else float(np.mean(diam))
        else:
            diam_value = float(diam)
        print(f"Estimated diameter: {diam_value:.1f} pixels")
    
    num_cells = masks.max()
    print(f"Detected {num_cells} cells")
    
except Exception as e:
    raise RuntimeError(f"Segmentation failed: {e}")

# Create output directory
os.makedirs('./output/beta6', exist_ok=True)

# Save masks
io.imsave('./output/beta6/masks.tif', masks)

# Method 1: Simple outline overlay (recommended)
from skimage.segmentation import find_boundaries

# Find cell boundaries
boundaries = find_boundaries(masks, mode='outer')

# Create RGB image with outlines
if len(img.shape) == 2:
    img_rgb = np.stack([img, img, img], axis=-1)
else:
    img_rgb = img.copy()

# Overlay red outlines
img_with_outlines = img_rgb.copy()
img_with_outlines[boundaries, 0] = 255  # Red channel
img_with_outlines[boundaries, 1] = 0    # Green channel
img_with_outlines[boundaries, 2] = 0    # Blue channel

io.imsave('./output/beta6/outlined_red.png', img_with_outlines)

# Method 2: Using Cellpose's built-in outline view
outlined = plot.outline_view(img, masks)
io.imsave('./output/beta6/outlined_cellpose.png', outlined)

# Method 3: Matplotlib visualization with numbered cells
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original image
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# Masks with colors
axes[1].imshow(masks, cmap='viridis')
axes[1].set_title(f'Cell Masks ({num_cells} cells)')
axes[1].axis('off')

# Overlay with outlines
axes[2].imshow(img, cmap='gray')
axes[2].imshow(masks, alpha=0.3, cmap='jet')  # Semi-transparent overlay
axes[2].contour(masks, levels=np.unique(masks)[1:], colors='red', linewidths=1)
axes[2].set_title('Overlay with Outlines')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('./output/beta5/visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# Method 4: Individual cell labeling with numbers
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.imshow(img, cmap='gray')

# Draw outlines and label each cell
from skimage.measure import regionprops

for region in regionprops(masks):
    # Get centroid
    y, x = region.centroid
    
    # Draw outline
    minr, minc, maxr, maxc = region.bbox
    rect = plt.Rectangle((minc, minr), maxc-minc, maxr-minr, # type: ignore
                         fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    
    # Add cell number
    ax.text(x, y, str(region.label), 
           color='yellow', fontsize=8, fontweight='bold',
           ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

ax.set_title(f'Detected Cells (n={num_cells})', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.savefig('./output/beta5/labeled_cells.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Outputs saved to ./output/beta6/")
print(f"  - outlined_red.png (red outlines)")
print(f"  - outlined_cellpose.png (Cellpose default)")
print(f"  - visualization.png (3-panel comparison)")
print(f"  - labeled_cells.png (numbered cells)")
print(f"  - masks.tif (segmentation masks)")