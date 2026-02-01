"""
Particle Scout - Development Script v2
Batch processing for bacteria segmentation with configurable parameters
"""

import os
import subprocess
import sys
import json
import importlib
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage
from skimage import measure, morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import warnings
warnings.filterwarnings('ignore')
from typing import Optional
import subprocess


# Import bacteria configurations
from bacteria_configs import (
    get_config, 
    list_available_configs, 
    SegmentationConfig,
    print_config_comparison,
    bacteria_map,
    bacteria_display_names
)

#==============================================================================
# Helper Functions
#==============================================================================

def setup_output_directory(config: dict) -> Path:
    """Create and setup output directory structure with timestamp naming.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to the main output directory
    """
    # Resolve the repository root from this file
    project_root = Path(__file__).resolve().parent
    output_root = project_root / 'outputs'
    output_root.mkdir(exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if config.get('batch_mode', False):
        # Batch mode: base_name_timestamp_test
        # Example: PD_sample_20260119_012703_test
        base_name = config.get('dataset_id_base', 'dataset')
        # Replace spaces with underscores for filename safety
        base_name = base_name.replace(' ', '_')
        output_dir_name = f"{base_name}_{timestamp}_test"
        
    else:
        # Single mode: base_name_microgel_timestamp_test
        # Example: PD_Negative_20260119_012703_test
        dataset_id = config.get('dataset_id', 'dataset')
        dataset_id = dataset_id.replace(' ', '_')
        microgel_type = config.get('microgel_type', '')
        
        if microgel_type:
            # Capitalize first letter: negative -> Negative, positive -> Positive
            microgel_label = microgel_type.capitalize()
            output_dir_name = f"{dataset_id}_{microgel_label}_{timestamp}_test"
        else:
            output_dir_name = f"{dataset_id}_{timestamp}_test"
    
    # Create the output directory
    output_dir = output_root / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def open_folder(folder_path: Path) -> None:
    """Open folder in file explorer (cross-platform, Unicode-safe)
    
    Args:
        folder_path: Path to folder to open
    """
    try:
        folder_str = str(folder_path.resolve())
        
        if sys.platform == 'win32':
            # Use os.startfile for better Unicode support on Windows
            os.startfile(folder_str)
        elif sys.platform == 'darwin':  # macOS
            subprocess.run(['open', folder_str])
        else:  # Linux
            subprocess.run(['xdg-open', folder_str])
        
        print(f"  ✓ Opened folder: {folder_path.name}")
    except Exception as e:
        print(f"  ⚠ Could not open folder automatically: {e}")
        print(f"  Please open manually: {folder_path.resolve()}")

def config_to_dict(config: SegmentationConfig) -> dict:
    """Convert SegmentationConfig to dictionary format
    
    Args:
        config: SegmentationConfig object
        
    Returns:
        Dictionary with configuration parameters
    """
    return {
        'gaussian_sigma': config.gaussian_sigma,
        'min_area': config.min_area_px,
        'max_area': config.max_area_px,
        'dilate_iterations': config.dilate_iterations,
        'erode_iterations': config.erode_iterations,
        'morph_kernel_size': config.morph_kernel_size,
        'morph_iterations': config.morph_iterations,
        'min_circularity': config.min_circularity,
        'max_circularity': config.max_circularity,
        'min_aspect_ratio': config.min_aspect_ratio,
        'max_aspect_ratio': config.max_aspect_ratio,
        'min_mean_intensity': config.min_mean_intensity,
        'max_mean_intensity': config.max_mean_intensity,
        'max_edge_gradient': config.max_edge_gradient,
        'min_solidity': config.min_solidity,
        'max_fraction_of_image': config.max_fraction_of_image,
        'description': config.description,
    }


def select_and_load_bacteria_config():
    """Let user select bacteria type and load corresponding config"""
    print("\n" + "="*70)
    print("BACTERIA CONFIGURATION SELECTOR")
    print("="*70)
    
    # Get available bacteria types
    print("\nAvailable bacteria types:")
    bacteria_list = list(bacteria_display_names.items())
    for i, (bacteria_type, display_name) in enumerate(bacteria_list, 1):
        print(f"  {i}. {display_name}")
    print(f"  {len(bacteria_list) + 1}. Default (Generic Bacteria)")   
    # Get user choice
    while True:
        try:
            choice = input(f"\nSelect bacteria type (1-{len(bacteria_list) + 1}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(bacteria_list) + 1:
                break
            print(f"Please enter a number between 1 and {len(bacteria_list) + 1}")
        except ValueError:
            print("Please enter a valid number")
    
    # Create mapping from choice number to bacteria type
    bacteria_map = {i+1: bacteria_type for i, (bacteria_type, _) in enumerate(bacteria_list)}
    
    # Load configuration
    if choice_num in bacteria_map:
        bacteria_type = bacteria_map[choice_num]
        config_obj = get_config(bacteria_type)
        config = config_to_dict(config_obj)
        config['bacteria_type'] = bacteria_type
        config['bacteria_display_name'] = bacteria_display_names[bacteria_type]
    else:
        # Use default
        config_obj = get_config('default')
        config = config_to_dict(config_obj)
        config['bacteria_type'] = 'default'
        config['bacteria_display_name'] = 'Default (Generic Bacteria)'
    
    print(f"\nLoaded configuration: {config['bacteria_display_name']}")
    if config_obj.description:
        print(f"Description: {config_obj.description}")
    
    return config

def select_source_directory(max_depth=2) -> Optional[Path]:
    """Lists directories that either have a Control subfolder OR contain G+/G- subdirectories
    
    Args:
        max_depth: Maximum depth to scan (not used currently)
        
    Returns:
        Path to selected source directory, or None if cancelled
    """
    root_dir = Path('source')
    
    if not root_dir.exists():
        print(f"[ERROR] Source directory not found: {root_dir.resolve()}")
        return None
    
    valid_directories = []
    
    # Check immediate subdirectories of source/
    for item in root_dir.iterdir():
        if not item.is_dir():
            continue
            
        # Check if this directory has both G+ and G- subdirectories
        has_gplus = (item / 'G+').is_dir()
        has_gminus = (item / 'G-').is_dir()
        
        if has_gplus and has_gminus:
            # This is a batch directory (like "PD sample" or "Spike sample")
            valid_directories.append({
                'path': item,
                'name': item.name,
                'mode': 'batch'
            })
            continue
        
        # Check if this directory has a Control subfolder (single-mode directory)
        try:
            subdirs = [d for d in item.iterdir() if d.is_dir()]
            has_control = any(d.name.lower().startswith('control') for d in subdirs)
            if has_control:
                valid_directories.append({
                    'path': item,
                    'name': item.name,
                    'mode': 'single'
                })
        except OSError:
            continue
    
    if not valid_directories:
        print("[ERROR] No valid directories found.")
        print("Valid directories must either:")
        print("  1. Contain both 'G+' and 'G-' subfolders (for batch processing)")
        print("  2. Contain a 'Control' subfolder (for single processing)")
        return None
    
    valid_directories.sort(key=lambda x: x['name'])
    
    print("\n" + "="*80)
    print("SELECT SOURCE DIRECTORY")
    print("="*80)
    print("\nAvailable directories:")
    
    for i, dir_info in enumerate(valid_directories, 1):
        mode_label = "[BATCH: G+ and G-]" if dir_info['mode'] == 'batch' else "[SINGLE]"
        print(f"  [{i}] {dir_info['name']} {mode_label}")
    
    while True:
        selected = input("\nEnter the number or folder name (or 'q' to quit): ").strip()
        
        if selected.lower() in {'q', 'quit', 'exit'}:
            return None
        
        # Try as number
        if selected.isdigit():
            num = int(selected)
            if 1 <= num <= len(valid_directories):
                selected_info = valid_directories[num - 1]
                return selected_info['path']
            else:
                print(f"Invalid number. Please enter between 1 and {len(valid_directories)}.")
        
        # Try as name
        else:
            for dir_info in valid_directories:
                if dir_info['name'] == selected:
                    return dir_info['path']
            print("Invalid selection. Please enter a valid number or folder name.")


def get_image_directory() -> Optional[str]:
    """Get the directory containing images to process
    
    Uses the same logic as dev0.py for source directory selection.
    
    Returns:
        str: Path to selected dataset directory, or None if cancelled
    """
    # Get project root (parent of dev2.py)
    project_root = Path(__file__).parent
    
    print(f"\nProject root: {project_root}")
    
    # Use the same selection logic as dev0.py
    selected_dir = select_source_directory()
    
    if selected_dir is None:
        print("\nNo directory selected - exiting")
        return None
    
    # Check if this is batch mode (has G+ and G- folders)
    gplus_path = selected_dir / 'G+'
    gminus_path = selected_dir / 'G-'
    
    has_gplus = gplus_path.is_dir()
    has_gminus = gminus_path.is_dir()
    
    if has_gplus and has_gminus:
        # Batch mode detected - let user choose which one to process
        print(f"\nDetected BATCH MODE directory: {selected_dir.name}")
        print("\nWhich dataset to process?")
        print("  [1] G+ (Gram-positive)")
        print("  [2] G- (Gram-negative)")
        print("  [3] Both (process G+ then G-)")
        
        while True:
            choice = input("\nSelect option (1/2/3): ").strip()
            
            if choice == "1":
                selected_dataset = gplus_path
                print(f"\n✓ Selected: {selected_dir.name}/G+")
                break
            elif choice == "2":
                selected_dataset = gminus_path
                print(f"\n✓ Selected: {selected_dir.name}/G-")
                break
            elif choice == "3":
                print(f"\n✓ Selected: Both datasets in {selected_dir.name}")
                print("   Will process G+ first, then G-")
                # Return a special marker that indicates batch processing
                return f"BATCH:{selected_dir}"
            else:
                print("Invalid choice. Enter 1, 2, or 3.")
    else:
        # Single mode
        selected_dataset = selected_dir
        print(f"\n✓ Selected: {selected_dir.name}")
    
    return str(selected_dataset)


def detect_microgel_type(directory_path: Path) -> str:
    """Auto-detect microgel type from directory path
    
    Args:
        directory_path: Path to dataset directory
        
    Returns:
        str: 'positive' or 'negative'
    """
    path_str = str(directory_path).upper()
    
    if 'G+' in path_str or 'GPLUS' in path_str or 'POSITIVE' in path_str:
        return 'positive'
    elif 'G-' in path_str or 'GMINUS' in path_str or 'NEGATIVE' in path_str:
        return 'negative'
    else:
        # Default or ask user
        return 'unknown'
    """Get the directory containing images to process
    
    Automatically scans project_root/source/ for dataset folders
    
    Returns:
        str: Path to selected dataset directory
    """
    # Get project root (parent of dev2.py)
    project_root = Path(__file__).parent
    source_dir = project_root / "source"
    
    print(f"\nScanning for datasets in: {source_dir}")
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"\n✗ Error: Source directory not found: {source_dir}")
        print("Please create a 'source' folder in the project root and add your dataset folders.")
        
        # Fallback to manual entry
        manual_path = input("\nEnter image directory path manually (or press Enter to exit): ").strip().strip('"')
        if not manual_path or not Path(manual_path).exists():
            print("Exiting...")
            sys.exit(1)
        return manual_path
    
    # Get all subdirectories (potential datasets)
    datasets = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    
    if not datasets:
        print(f"\n✗ Error: No dataset folders found in {source_dir}")
        print("Please add dataset folders containing images to the 'source' directory.")
        
        # Fallback to manual entry
        manual_path = input("\nEnter image directory path manually (or press Enter to exit): ").strip().strip('"')
        if not manual_path or not Path(manual_path).exists():
            print("Exiting...")
            sys.exit(1)
        return manual_path
    
    # Display available datasets
    print("\nAvailable datasets:")
    for i, dataset in enumerate(datasets, 1):
        # Count images in dataset
        image_files = (list(dataset.glob("*.tif")) + 
                      list(dataset.glob("*.tiff")) + 
                      list(dataset.glob("*.png")) + 
                      list(dataset.glob("*.jpg")) +
                      list(dataset.glob("*.jpeg")))
        print(f"  {i}. {dataset.name} ({len(image_files)} images)")
    
    print(f"  {len(datasets) + 1}. Enter path manually")
    
    # Get user choice
    while True:
        try:
            choice = input(f"\nSelect dataset (1-{len(datasets) + 1}): ").strip()
            choice_num = int(choice)
            
            if choice_num == len(datasets) + 1:
                # Manual entry
                manual_path = input("Enter image directory path: ").strip().strip('"')
                if Path(manual_path).exists():
                    return manual_path
                else:
                    print(f"✗ Directory not found: {manual_path}")
                    continue
            
            if 1 <= choice_num <= len(datasets):
                selected_dataset = datasets[choice_num - 1]
                print(f"\n✓ Selected dataset: {selected_dataset.name}")
                return str(selected_dataset)
            
            print(f"Please enter a number between 1 and {len(datasets) + 1}")
        except ValueError:
            print("Please enter a valid number")

def reload_bacteria_config(config: dict) -> dict:
    """Reload configuration from bacteria_configs.py
    
    Useful when you've just tuned parameters in feedback_tuner.py
    and want to reload without restarting dev2.py
    
    Args:
        config: Current config dictionary
        
    Returns:
        Updated config dictionary
    """
    import bacteria_configs
    
    print("\n" + "="*70)
    print("RELOADING CONFIGURATION...")
    print("="*70)
    
    try:
        # Reload the module to get latest changes from file
        importlib.reload(bacteria_configs)
        
        # Get fresh config
        bacteria_type = config['bacteria_type']
        fresh_config_obj = bacteria_configs.get_config(bacteria_type)
        fresh_config = config_to_dict(fresh_config_obj)
        fresh_config['bacteria_type'] = bacteria_type
        fresh_config['bacteria_display_name'] = bacteria_display_names.get(
            bacteria_type, 
            config.get('bacteria_display_name', 'Unknown')
        )
        
        print(f"✓ Reloaded: {fresh_config['bacteria_display_name']}")
        print(f"\nUpdated parameters:")
        print(f"  • Gaussian sigma: {fresh_config['gaussian_sigma']:.2f}")
        print(f"  • Size range: {fresh_config['min_area']:.0f} - {fresh_config['max_area']:.0f} px²")
        print(f"                ({fresh_config['min_area']*0.012:.2f} - {fresh_config['max_area']*0.012:.2f} µm²)")
        print(f"  • Morphology: dilate={fresh_config['dilate_iterations']}, erode={fresh_config['erode_iterations']}")
        print("="*70 + "\n")
        
        return fresh_config
        
    except Exception as e:
        import traceback
        print(f"❌ Failed to reload config: {e}")
        print(traceback.format_exc())
        print(f"→ Keeping current configuration")
        print("="*70 + "\n")
        return config


def show_current_config(config: dict):
    """Display current configuration details
    
    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*70)
    print(f"CURRENT CONFIGURATION: {config['bacteria_display_name']}")
    print("="*70)
    print(f"Description: {config.get('description', 'Standard configuration')}")
    print(f"\nSegmentation Parameters:")
    print(f"  • Gaussian sigma: {config['gaussian_sigma']:.2f}")
    print(f"  • Min area: {config['min_area']:.0f} px² ({config['min_area']*0.012:.2f} µm²)")
    print(f"  • Max area: {config['max_area']:.0f} px² ({config['max_area']*0.012:.2f} µm²)")
    print(f"  • Dilate iterations: {config['dilate_iterations']}")
    print(f"  • Erode iterations: {config['erode_iterations']}")
    print(f"  • Morph kernel size: {config.get('morph_kernel_size', 3)}")
    print(f"  • Min circularity: {config.get('min_circularity', 0.0):.2f}")
    print(f"  • Max circularity: {config.get('max_circularity', 1.0):.2f}")
    print(f"  • Min aspect ratio: {config.get('min_aspect_ratio', 0.2):.2f}")
    print(f"  • Max aspect ratio: {config.get('max_aspect_ratio', 10.0):.2f}")
    print("="*70 + "\n")


def preprocess_image(image, gaussian_sigma=2.0):
    """Preprocess image with Gaussian blur and normalization
    
    Args:
        image: Input grayscale image
        gaussian_sigma: Sigma for Gaussian blur
        
    Returns:
        Preprocessed image
    """
    # Apply Gaussian blur
    blurred = ndimage.gaussian_filter(image.astype(float), sigma=gaussian_sigma)
    
    # Normalize to 0-255 range
    blurred = ((blurred - blurred.min()) / (blurred.max() - blurred.min()) * 255).astype(np.uint8)
    
    return blurred


def apply_morphological_operations(binary_mask, dilate_iterations=0, erode_iterations=0, kernel_size=3):
    """Apply morphological operations to binary mask
    
    Args:
        binary_mask: Binary segmentation mask
        dilate_iterations: Number of dilation iterations
        erode_iterations: Number of erosion iterations
        kernel_size: Size of morphological kernel
        
    Returns:
        Processed binary mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply dilation
    if dilate_iterations > 0:
        binary_mask = cv2.dilate(binary_mask.astype(np.uint8), kernel, iterations=dilate_iterations)
    
    # Apply erosion
    if erode_iterations > 0:
        binary_mask = cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=erode_iterations)
    
    return binary_mask.astype(bool)


def segment_bacteria_adaptive(image, config):
    """Segment bacteria using adaptive thresholding with configurable parameters
    
    Args:
        image: Input grayscale image
        config: Configuration dictionary with segmentation parameters
        
    Returns:
        tuple: (binary_mask, labeled_image, num_bacteria)
    """
    # Preprocess
    preprocessed = preprocess_image(image, gaussian_sigma=config['gaussian_sigma'])
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        preprocessed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )
    
    # Apply morphological operations
    binary = apply_morphological_operations(
        binary,
        dilate_iterations=config['dilate_iterations'],
        erode_iterations=config['erode_iterations'],
        kernel_size=config.get('morph_kernel_size', 3)
    )
    
    # Remove small objects and fill holes
    binary = morphology.remove_small_objects(binary, min_size=config['min_area'])
    binary = ndimage.binary_fill_holes(binary)
    
    # Label connected components
    labeled = measure.label(binary)
    
    # Filter by size and shape
    props = measure.regionprops(labeled)
    filtered_label = np.zeros_like(labeled)
    new_label = 1
    
    for prop in props:
        area = prop.area
        
        # Size filter
        if area < config['min_area'] or area > config['max_area']:
            continue
        
        # Circularity filter
        perimeter = prop.perimeter
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < config.get('min_circularity', 0.0) or circularity > config.get('max_circularity', 1.0):
                continue
        
        # Aspect ratio filter
        if prop.minor_axis_length > 0:
            aspect_ratio = prop.major_axis_length / prop.minor_axis_length
            if aspect_ratio < config.get('min_aspect_ratio', 0.2) or aspect_ratio > config.get('max_aspect_ratio', 10.0):
                continue
        
        # Solidity filter
        if prop.solidity < config.get('min_solidity', 0.3):
            continue
        
        # Keep this object
        filtered_label[labeled == prop.label] = new_label
        new_label += 1
    
    num_bacteria = new_label - 1
    
    return filtered_label > 0, filtered_label, num_bacteria


def calculate_bacteria_properties(labeled_image, intensity_image):
    """Calculate properties for each detected bacterium
    
    Args:
        labeled_image: Labeled segmentation image
        intensity_image: Original intensity image
        
    Returns:
        pandas.DataFrame: Properties for each bacterium
    """
    props = measure.regionprops_table(
        labeled_image,
        intensity_image=intensity_image,
        properties=[
            'label', 'area', 'centroid', 'major_axis_length', 
            'minor_axis_length', 'orientation', 'perimeter',
            'mean_intensity', 'max_intensity', 'min_intensity'
        ]
    )
    
    df = pd.DataFrame(props)
    
    # Calculate additional properties
    df['circularity'] = 4 * np.pi * df['area'] / (df['perimeter'] ** 2)
    df['aspect_ratio'] = df['major_axis_length'] / df['minor_axis_length']
    
    # Convert to microns (assuming 0.109 µm/pixel)
    um_per_pixel = 0.109
    um2_per_pixel2 = um_per_pixel ** 2
    
    df['area_um2'] = df['area'] * um2_per_pixel2
    df['centroid_x_um'] = df['centroid-1'] * um_per_pixel
    df['centroid_y_um'] = df['centroid-0'] * um_per_pixel
    df['major_axis_um'] = df['major_axis_length'] * um_per_pixel
    df['minor_axis_um'] = df['minor_axis_length'] * um_per_pixel
    
    return df


def visualize_segmentation(image, binary_mask, labeled_image, properties_df, save_path=None):
    """Visualize segmentation results
    
    Args:
        image: Original image
        binary_mask: Binary segmentation mask
        labeled_image: Labeled image
        properties_df: DataFrame with bacteria properties
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Binary mask
    axes[0, 1].imshow(binary_mask, cmap='gray')
    axes[0, 1].set_title('Binary Segmentation')
    axes[0, 1].axis('off')
    
    # Labeled image
    axes[1, 0].imshow(labeled_image, cmap='nipy_spectral')
    axes[1, 0].set_title(f'Labeled Bacteria (n={len(properties_df)})')
    axes[1, 0].axis('off')
    
    # Overlay with bounding boxes
    axes[1, 1].imshow(image, cmap='gray')
    
    for idx, row in properties_df.iterrows():
        y0, x0 = row['centroid-0'], row['centroid-1']
        
        # Draw centroid
        axes[1, 1].plot(x0, y0, 'r+', markersize=10, markeredgewidth=2)
        
        # Draw bounding box
        minr = y0 - row['major_axis_length'] / 2
        minc = x0 - row['minor_axis_length'] / 2
        maxr = y0 + row['major_axis_length'] / 2
        maxc = x0 + row['minor_axis_length'] / 2
        
        rect = Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor='red',
            linewidth=1
        )
        axes[1, 1].add_patch(rect)
    
    axes[1, 1].set_title('Detected Bacteria with Bounding Boxes')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def process_image(image_path, config, output_dir=None, visualize=True):
    """Process a single image
    
    Args:
        image_path: Path to input image
        config: Configuration dictionary
        output_dir: Optional output directory for results
        visualize: Whether to show visualization
        
    Returns:
        tuple: (binary_mask, labeled_image, properties_df)
    """
    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"\nProcessing: {Path(image_path).name}")
    print(f"Image size: {image.shape}")
    
    # Segment bacteria
    binary_mask, labeled_image, num_bacteria = segment_bacteria_adaptive(image, config)
    
    print(f"Detected {num_bacteria} bacteria")
    
    # Calculate properties
    properties_df = calculate_bacteria_properties(labeled_image, image)
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = output_dir / f"{Path(image_path).stem}_properties.csv"
        properties_df.to_csv(csv_path, index=False)
        print(f"Saved properties to: {csv_path}")
        
        # Save visualization
        if visualize:
            viz_path = output_dir / f"{Path(image_path).stem}_segmentation.png"
            visualize_segmentation(image, binary_mask, labeled_image, properties_df, save_path=viz_path)
    elif visualize:
        visualize_segmentation(image, binary_mask, labeled_image, properties_df)
    
    return binary_mask, labeled_image, properties_df


def batch_process_images(image_dir, config, output_dir=None, file_pattern="*.tif", visualize_each=False):
    """Batch process multiple images
    
    Args:
        image_dir: Directory containing images
        config: Configuration dictionary
        output_dir: Output directory for results
        file_pattern: Pattern to match image files
        visualize_each: Whether to visualize each image
        
    Returns:
        dict: Summary statistics
    """
    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob(file_pattern))
    
    if not image_files:
        print(f"No images found matching pattern: {file_pattern}")
        return None
    
    print(f"\nFound {len(image_files)} images to process")
    print("="*70)
    
    # Create output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    all_results = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
        
        try:
            binary_mask, labeled_image, properties_df = process_image(
                image_path,
                config,
                output_dir=output_dir,
                visualize=visualize_each
            )
            
            # Add image filename to properties
            properties_df['image'] = image_path.name
            all_results.append(properties_df)
            
            print(f"✓ Detected {len(properties_df)} bacteria")
            
        except Exception as e:
            print(f"✗ Error processing {image_path.name}: {e}")
            continue
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        if output_dir:
            combined_csv = output_dir / "combined_results.csv"
            combined_df.to_csv(combined_csv, index=False)
            print(f"\n✓ Saved combined results to: {combined_csv}")
        
        # Print summary statistics
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        print(f"Total images processed: {len(all_results)}")
        print(f"Total bacteria detected: {len(combined_df)}")
        print(f"\nArea statistics (µm²):")
        print(f"  Mean: {combined_df['area_um2'].mean():.2f}")
        print(f"  Median: {combined_df['area_um2'].median():.2f}")
        print(f"  Std: {combined_df['area_um2'].std():.2f}")
        print(f"  Min: {combined_df['area_um2'].min():.2f}")
        print(f"  Max: {combined_df['area_um2'].max():.2f}")
        print(f"\nCircularity statistics:")
        print(f"  Mean: {combined_df['circularity'].mean():.3f}")
        print(f"  Median: {combined_df['circularity'].median():.3f}")
        print("="*70)
        
        return {
            'num_images': len(all_results),
            'num_bacteria': len(combined_df),
            'combined_df': combined_df
        }
    
    return None


def interactive_mode():
    """Interactive mode for processing images with live feedback"""
    print("\n" + "="*70)
    print("PARTICLE SCOUT - INTERACTIVE MODE")
    print("="*70)
    
    # Select bacteria configuration
    config = select_and_load_bacteria_config()
    
    # Get image directory using the new function
    image_dir_result = get_image_directory()
    
    # Handle cancellation
    if image_dir_result is None:
        print("\nNo directory selected - returning to main menu")
        return
    
    # Handle batch mode selection
    if image_dir_result.startswith("BATCH:"):
        parent_dir_str = image_dir_result.split(":", 1)[1]
        parent_dir = Path(parent_dir_str)
        
        print(f"\nBatch mode not supported in interactive mode")
        print(f"Please select a specific dataset (G+ or G-)")
        return
    
    # Now image_dir_result is guaranteed to be a valid string path
    image_dir = image_dir_result
    
    # Validate directory exists
    if not Path(image_dir).exists():
        print(f"✗ Directory not found: {image_dir}")
        return
    
    # Get file pattern
    file_pattern = input("File pattern (default: *.tif): ").strip() or "*.tif"
    
    # Get output directory
    use_output = input("Save results? (y/n, default: n): ").strip().lower() == 'y'
    output_dir = None
    
    if use_output:
        output_dir = input("Output directory (default: ./output): ").strip() or "./output"
    
    # Find images
    image_files = sorted(Path(image_dir).glob(file_pattern))
    
    if not image_files:
        print(f"✗ No images found matching: {file_pattern}")
        return
    
    print(f"\n✓ Found {len(image_files)} images")
    
    # Process images one by one
    image_idx = 0
    
    while image_idx < len(image_files):
        image_path = image_files[image_idx]
        
        print("\n" + "="*70)
        print(f"IMAGE {image_idx + 1}/{len(image_files)}: {image_path.name}")
        print("="*70)
        
        try:
            # Process image
            binary_mask, labeled_image, properties_df = process_image(
                image_path,
                config,
                output_dir=output_dir,
                visualize=True
            )
            
            # Show options
            print("\n" + "-"*70)
            print("Options:")
            print("  [Enter] - Next image")
            print("  p - Previous image")
            print("  r - Reload config (if you just saved in feedback_tuner)")
            print("  s - Show current config")
            print("  j - Jump to image number")
            print("  q - Quit")
            print("-"*70)
            
            choice = input("Choice: ").strip().lower()
            
            if choice == 'r':
                config = reload_bacteria_config(config)
                # Reprocess current image with new config
                continue
                
            elif choice == 's':
                show_current_config(config)
                continue
                
            elif choice == 'p':
                image_idx = max(0, image_idx - 1)
                continue
                
            elif choice == 'j':
                try:
                    jump_to = int(input(f"Jump to image (1-{len(image_files)}): "))
                    image_idx = max(0, min(len(image_files) - 1, jump_to - 1))
                    continue
                except ValueError:
                    print("Invalid number")
                    continue
                    
            elif choice == 'q':
                print("\nExiting...")
                break
                
            else:
                # Continue to next image
                image_idx += 1
                
        except Exception as e:
            print(f"✗ Error processing image: {e}")
            import traceback
            traceback.print_exc()
            
            retry = input("\nRetry this image? (y/n): ").strip().lower()
            if retry != 'y':
                image_idx += 1
    
    print("\n" + "="*70)
    print("Processing complete!")
    print("="*70)

def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("PARTICLE SCOUT - BACTERIA SEGMENTATION TOOL")
    print("="*70)
    print("\nModes:")
    print("  1. Interactive mode (process images one by one with live feedback)")
    print("  2. Batch mode (process all images automatically)")
    print("  3. Single image mode")
    
    mode = input("\nSelect mode (1/2/3): ").strip()
    
    if mode == '1':
        interactive_mode()
        
    elif mode == '2':
        # Select bacteria configuration
        config = select_and_load_bacteria_config()
        
        # Get directory using dev0.py-style selector
        image_dir_result = get_image_directory()
        
        # Check if user cancelled
        if image_dir_result is None:
            print("No directory selected - exiting")
            return
        
        # Check if batch mode was selected
        if image_dir_result.startswith("BATCH:"):
            # Extract parent directory
            parent_dir_str = image_dir_result.split(":", 1)[1]
            parent_dir = Path(parent_dir_str)
            
            print("\n" + "="*70)
            print("BATCH PROCESSING MODE")
            print("="*70)
            print(f"\nProcessing both G+ and G- datasets in: {parent_dir.name}\n")
            
            # Setup batch configuration
            batch_config = {
                'batch_mode': True,
                'dataset_id_base': parent_dir.name,
                'microgel_type': None  # Both types
            }
            
            # Setup main output directory
            output_root = setup_output_directory(batch_config)
            print(f"\n📁 Output directory: {output_root.relative_to(Path.cwd())}\n")
            
            # Process G+ first
            gplus_dir = parent_dir / "G+"
            output_dir_gplus = output_root / "Positive"
            output_dir_gplus.mkdir(exist_ok=True)
            
            print("\n" + "-"*70)
            print("Processing G+ (Gram-positive)")
            print("-"*70)
            print(f"Output: {output_dir_gplus.relative_to(Path.cwd())}")
            
            visualize_gplus = input("Visualize each image? (y/n, default: n): ").strip().lower() == 'y'
            
            batch_process_images(
                str(gplus_dir),
                config,
                output_dir=str(output_dir_gplus),
                file_pattern="*.tif",
                visualize_each=visualize_gplus
            )
            
            # Process G- second
            gminus_dir = parent_dir / "G-"
            output_dir_gminus = output_root / "Negative"
            output_dir_gminus.mkdir(exist_ok=True)
            
            print("\n" + "-"*70)
            print("Processing G- (Gram-negative)")
            print("-"*70)
            print(f"Output: {output_dir_gminus.relative_to(Path.cwd())}")
            
            visualize_gminus = input("Visualize each image? (y/n, default: n): ").strip().lower() == 'y'
            
            batch_process_images(
                str(gminus_dir),
                config,
                output_dir=str(output_dir_gminus),
                file_pattern="*.tif",
                visualize_each=visualize_gminus
            )
            
            print("\n" + "="*70)
            print("BATCH PROCESSING COMPLETE")
            print("="*70)
            print(f"\n📁 Results location: {output_root.relative_to(Path.cwd())}")
            print(f"  → G+ results: {output_dir_gplus.name}")
            print(f"  → G- results: {output_dir_gminus.name}")
            
            # Open output folder
            print()
            open_folder(output_root)
            
        else:
            # Single dataset selected - image_dir_result is guaranteed to be a string here
            image_dir = Path(image_dir_result)
            
            # Detect microgel type
            microgel_type = detect_microgel_type(image_dir)
            if microgel_type == 'unknown':
                print("\nCould not auto-detect microgel type")
                print("  [1] Positive (G+)")
                print("  [2] Negative (G-)")
                choice = input("Select type (1/2): ").strip()
                microgel_type = 'positive' if choice == '1' else 'negative'
            
            print(f"Microgel type: {microgel_type.upper()}")
            
            # Get dataset name
            dataset_id = input("\nDataset name (Enter=use folder name): ").strip()
            if not dataset_id:
                dataset_id = image_dir.name
            
            # Setup single configuration
            single_config = {
                'batch_mode': False,
                'dataset_id': dataset_id,
                'microgel_type': microgel_type
            }
            
            # Setup output directory
            output_dir = setup_output_directory(single_config)
            print(f"\n📁 Output directory: {output_dir.relative_to(Path.cwd())}")
            
            file_pattern = input("\nFile pattern (default: *.tif): ").strip() or "*.tif"
            visualize = input("Visualize each image? (y/n, default: n): ").strip().lower() == 'y'
            
            # Process
            batch_process_images(
                str(image_dir),
                config,
                output_dir=str(output_dir),
                file_pattern=file_pattern,
                visualize_each=visualize
            )
            
            print("\n" + "="*70)
            print("PROCESSING COMPLETE")
            print("="*70)
            print(f"\n📁 Results location: {output_dir.relative_to(Path.cwd())}")
            
            # Open output folder
            print()
            open_folder(output_dir)
        
    elif mode == '3':
        # Select bacteria configuration
        config = select_and_load_bacteria_config()
        
        # Get parameters
        image_path_str = input("\nImage path: ").strip().strip('"')
        image_path = Path(image_path_str)
        
        if not image_path.exists():
            print(f"✗ Image not found: {image_path}")
            return
        
        # Get dataset name
        dataset_id = input("Dataset name (Enter=use image name): ").strip()
        if not dataset_id:
            dataset_id = image_path.stem
        
        # Detect microgel type
        microgel_type = detect_microgel_type(image_path.parent)
        if microgel_type == 'unknown':
            print("\nSelect microgel type:")
            print("  [1] Positive (G+)")
            print("  [2] Negative (G-)")
            choice = input("Select type (1/2): ").strip()
            microgel_type = 'positive' if choice == '1' else 'negative'
        
        # Setup configuration
        single_config = {
            'batch_mode': False,
            'dataset_id': dataset_id,
            'microgel_type': microgel_type
        }
        
        # Setup output directory
        output_dir = setup_output_directory(single_config)
        print(f"\n📁 Output directory: {output_dir.relative_to(Path.cwd())}")
        
        # Process
        process_image(
            image_path_str,
            config,
            output_dir=str(output_dir),
            visualize=True
        )
        
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        print(f"\n📁 Results location: {output_dir.relative_to(Path.cwd())}")
        
        # Open output folder
        print()
        open_folder(output_dir)
    
    else:
        print("Invalid mode selected")


if __name__ == "__main__":
    main()