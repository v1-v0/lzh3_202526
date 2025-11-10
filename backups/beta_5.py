from cellpose.models import CellposeModel
from cellpose import io, plot
import os
import numpy as np
import torch
import platform

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

# Simple red outlines on original image
from skimage.segmentation import find_boundaries

boundaries = find_boundaries(masks, mode='outer')
img_rgb = np.stack([img, img, img], axis=-1) if len(img.shape) == 2 else img.copy()
img_rgb[boundaries] = [255, 0, 0]  # Red outlines

os.makedirs('./output/beta5', exist_ok=True)
io.imsave('./output/beta5/outlined.png', img_rgb)
io.imsave('./output/beta5/masks.tif', masks)

print(f"✓ Detected {num_cells} cells - outputs saved to ./output/beta5/")