"""
Dual Segmentation Pipeline for Particle Analysis
Processes brightfield and fluorescence images to segment and analyze particles.
Includes robust pre-processing and strict type checking.
"""

import numpy as np
from pathlib import Path
from skimage import io, filters, morphology, measure, exposure, color, util
from skimage.segmentation import watershed
from scipy import ndimage
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import xml.etree.ElementTree as ET
from typing import Dict, Any, Tuple, cast
import warnings

# Suppress warnings about low contrast images or precision loss during conversion
warnings.filterwarnings('ignore')


def find_xml_file(base_path: Path) -> Path:
    """Find the XML metadata file."""
    possible_paths = [
        base_path.with_suffix('.xml'),
        base_path.parent / 'MetaData' / f"{base_path.stem}.xml",
        base_path.parent / 'MetaData' / f"{base_path.stem}_Properties.xml",
    ]
    
    for xml_path in possible_paths:
        if xml_path.exists():
            print(f"Found XML metadata: {xml_path}")
            return xml_path
    
    raise FileNotFoundError("Could not find XML metadata file.")


def load_metadata(xml_path: Path) -> Dict[str, Any]:
    """Load and parse XML metadata with strict type checking."""
    print(f"\n{'='*60}")
    print("LOADING METADATA")
    print(f"{'='*60}")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Helper to safely get attributes
    def get_attr(element: ET.Element | None, attr: str, name: str) -> str:
        if element is None:
            raise ValueError(f"XML Error: Element '{name}' not found")
        val = element.get(attr)
        if val is None:
            raise ValueError(f"XML Error: Attribute '{attr}' not found in '{name}'")
        return val

    dimensions = root.find('.//Dimensions')
    if dimensions is None: raise ValueError("No Dimensions in XML")
    
    dim_x = dimensions.find(".//DimensionDescription[@DimID='1']")
    dim_y = dimensions.find(".//DimensionDescription[@DimID='2']")
    
    # Safe extraction using helper
    length_x = float(get_attr(dim_x, 'Length', 'DimensionDescription X'))
    length_y = float(get_attr(dim_y, 'Length', 'DimensionDescription Y'))
    
    num_elements_x = int(get_attr(dim_x, 'NumberOfElements', 'DimensionDescription X'))
    num_elements_y = int(get_attr(dim_y, 'NumberOfElements', 'DimensionDescription Y'))
    
    pixel_size_x = (length_x * 1e6) / num_elements_x
    pixel_size_y = (length_y * 1e6) / num_elements_y
    
    channel = root.find('.//ChannelDescription')
    bit_depth = int(get_attr(channel, 'Resolution', 'ChannelDescription'))
    
    metadata = {
        'pixel_size_x': pixel_size_x,
        'pixel_size_y': pixel_size_y,
        'bit_depth': bit_depth,
        'image_width': num_elements_x,
        'image_height': num_elements_y,
    }
    
    print(f"Dimensions: {num_elements_x} x {num_elements_y}")
    print(f"Pixel size: {pixel_size_x:.4f} x {pixel_size_y:.4f} µm")
    print(f"Bit depth: {bit_depth}-bit")
    
    return metadata


def load_image(image_path: Path) -> np.ndarray:
    """Load image and ensure it is 2D grayscale."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = io.imread(image_path)
    print(f"Loaded {image_path.name}: shape={image.shape}, dtype={image.dtype}")
    
    # Convert RGB/RGBA to grayscale immediately
    if image.ndim == 3:
        if image.shape[-1] == 3:
            print("  Detected RGB, converting to grayscale...")
            image = color.rgb2gray(image)
        elif image.shape[-1] == 4:
            print("  Detected RGBA, converting to grayscale...")
            image = color.rgb2gray(image[..., :3])
            
    return image


def preprocess_image(image: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Standardize image for segmentation.
    """
    print(f"  Preprocessing ({method})...")
    
    # 1. Normalize to float 0-1 if not already
    if image.dtype != np.float64 and image.dtype != np.float32:
        image = util.img_as_float(image)
        
    # 2. Enhance Contrast (Percentile Stretching)
    if method == 'fluorescence':
        # Keep more dynamic range for fluorescence
        p_low, p_high = np.percentile(image, (0.5, 99.5))
    else:
        # Standard contrast stretch for brightfield
        p_low, p_high = np.percentile(image, (2.0, 98.0))
    
    # Explicitly cast to float tuple
    in_range = (float(p_low), float(p_high))
    
    # Cast in_range to Any because Pylance's stubs for skimage incorrectly 
    # expect only a string here, even though a tuple is valid at runtime.
    image_rescaled = exposure.rescale_intensity(image, in_range=cast(Any, in_range))
    
    # 3. Convert to 8-bit unsigned integer (0-255)
    image_8bit = util.img_as_ubyte(image_rescaled)
    
    return image_8bit


def segment_brightfield(bf_image: np.ndarray, min_size: int = 100) -> np.ndarray:
    """Segment particles from brightfield image."""
    print(f"\n{'='*60}")
    print("BRIGHTFIELD SEGMENTATION")
    print(f"{'='*60}")
    
    # Pre-process
    img_8bit = preprocess_image(bf_image, method='standard')
    
    # Apply Gaussian smoothing
    smoothed = filters.gaussian(img_8bit, sigma=2)
    
    # Otsu thresholding
    threshold = filters.threshold_otsu(smoothed)
    binary = smoothed < threshold  # Particles are darker in BF
    
    print(f"Otsu threshold: {threshold:.2f}")
    print(f"Pixels above threshold: {np.sum(binary):,} ({100*np.sum(binary)/binary.size:.1f}%)")
    
    # Morphological operations
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    binary = morphology.remove_small_holes(binary, area_threshold=min_size)
    binary = morphology.binary_closing(binary, morphology.disk(3))
    
    # Watershed
    # Explicitly cast distance to ndarray to fix "Operator - not supported for None"
    distance = cast(np.ndarray, ndimage.distance_transform_edt(binary))
    
    local_max = morphology.local_maxima(distance)
    markers = measure.label(local_max)
    
    # Explicitly cast result to ndarray
    labels = cast(np.ndarray, watershed(-distance, markers, mask=binary))
    
    print(f"Detected {int(np.max(labels))} particles")
    return labels


def segment_fluorescence(fl_image: np.ndarray, min_size: int = 50) -> np.ndarray:
    """Segment fluorescent regions."""
    print(f"\n{'='*60}")
    print("FLUORESCENCE SEGMENTATION")
    print(f"{'='*60}")
    
    # Pre-process (use 'fluorescence' method to preserve faint signals)
    img_8bit = preprocess_image(fl_image, method='fluorescence')
    
    # Apply Gaussian smoothing
    smoothed = filters.gaussian(img_8bit, sigma=1.5)
    
    # Thresholding (Top 5% of intensity)
    threshold = np.percentile(smoothed, 95)
    binary = smoothed > threshold
    
    print(f"Fluorescence threshold: {threshold:.2f}")
    
    # Clean up
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    binary = morphology.binary_closing(binary, morphology.disk(2))
    
    # Explicitly cast to ndarray to fix "No overloads for amax match"
    labels = cast(np.ndarray, measure.label(binary))
    
    print(f"Detected {int(np.max(labels))} fluorescent regions")
    
    return labels


def calculate_overlap(bf_labels: np.ndarray, fl_labels: np.ndarray, 
                     pixel_size: Tuple[float, float]) -> pd.DataFrame:
    """Calculate overlap between brightfield particles and fluorescent regions."""
    print(f"\n{'='*60}")
    print("CALCULATING OVERLAP")
    print(f"{'='*60}")
    
    pixel_size_x, pixel_size_y = pixel_size
    pixel_area = pixel_size_x * pixel_size_y
    
    results = []
    bf_props = measure.regionprops(bf_labels)
    
    for prop in bf_props:
        particle_id = prop.label
        particle_mask = (bf_labels == particle_id)
        
        # Basic geometry
        area_pixels = prop.area
        area_um2 = area_pixels * pixel_area
        diameter_um = 2 * np.sqrt(area_um2 / np.pi)
        
        # Overlap logic
        overlap_mask = particle_mask & (fl_labels > 0)
        overlap_pixels = np.sum(overlap_mask)
        overlap_pct = (overlap_pixels / area_pixels * 100) if area_pixels > 0 else 0
        
        centroid_y, centroid_x = prop.centroid
        
        results.append({
            'particle_id': particle_id,
            'area_um2': area_um2,
            'diameter_um': diameter_um,
            'overlap_percentage': overlap_pct,
            'has_fluorescence': overlap_pct > 5,  # >5% overlap = Positive
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
        })
    
    df = pd.DataFrame(results)
    print(f"Analyzed {len(df)} particles")
    return df


def visualize_results(bf_image: np.ndarray, fl_image: np.ndarray, 
                     bf_labels: np.ndarray, results_df: pd.DataFrame, 
                     output_dir: Path):
    """Generate visual output."""
    print(f"\nGenerating visualization...")
    
    # Normalize images for display
    bf_disp = exposure.rescale_intensity(util.img_as_float(bf_image))
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(bf_disp, cmap='gray', alpha=0.8)
    
    for _, row in results_df.iterrows():
        color = 'red' if row['has_fluorescence'] else 'blue'
        
        # Draw circle
        circle = Circle((row['centroid_x'], row['centroid_y']), 
                       radius=row['diameter_um']/2, 
                       fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(circle)
        
        # Add ID label
        ax.text(row['centroid_x'], row['centroid_y'], 
               str(int(row['particle_id'])), 
               color='yellow', fontsize=8, ha='center', va='center')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markeredgecolor='red', label='Fluorescent'),
        Line2D([0], [0], marker='o', color='w', markeredgecolor='blue', label='Non-fluorescent')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'classification_result.png', dpi=300)
    plt.close()


def main():
    # Setup paths
    bf_path = Path("source/10/10 P 1_ch00.tif")
    fl_path = Path("source/10/10 P 1_ch01.tif")
    
    # Setup output
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Load Data
    xml_path = find_xml_file(bf_path.parent / bf_path.stem.replace('_ch00', ''))
    metadata = load_metadata(xml_path)
    
    bf_img = load_image(bf_path)
    fl_img = load_image(fl_path)
    
    # 2. Segmentation
    bf_labels = segment_brightfield(bf_img)
    fl_labels = segment_fluorescence(fl_img)
    
    # 3. Analysis
    pixel_size = (metadata['pixel_size_x'], metadata['pixel_size_y'])
    results = calculate_overlap(bf_labels, fl_labels, pixel_size)
    
    # 4. Save
    results.to_csv(output_dir / 'results.csv', index=False)
    visualize_results(bf_img, fl_img, bf_labels, results, output_dir)
    
    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()