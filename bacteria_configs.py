"""
Bacteria-specific segmentation parameters
Each configuration is optimized for the unique morphology of each species
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class SegmentationConfig:
    """Parameters for particle segmentation and filtering"""
    
    # Identity
    name: str
    description: str
    
    # Gaussian blur for background subtraction
    gaussian_sigma: float
    
    # Morphological operations
    morph_kernel_size: int
    morph_iterations: int
    dilate_iterations: int
    erode_iterations: int
    
    # Size filtering (in µm²)
    min_area_um2: float
    max_area_um2: float
    
    # Shape filtering
    min_circularity: float
    max_circularity: float
    min_aspect_ratio: float
    max_aspect_ratio: float
    
    # Intensity filtering (0-255)
    min_mean_intensity: float
    max_mean_intensity: float
    
    # Edge characteristics
    max_edge_gradient: float
    
    # Solidity (convex hull ratio)
    min_solidity: float
    
    # Image area limit
    max_fraction_of_image: float
    
    # Fluorescence parameters
    fluor_gaussian_sigma: float
    fluor_morph_kernel_size: int
    fluor_min_area_um2: float
    fluor_match_min_intersection_px: float


# ==================================================
# Proteus mirabilis Configuration
# ==================================================
PROTEUS_MIRABILIS = SegmentationConfig(
    name="Proteus mirabilis",
    description="Rod-shaped, highly motile, forms swarming patterns",
    
    # Morphology: Rod-shaped (1-2 µm × 2-6 µm)
    # Characteristics: High motility, can form long cells, swarming
    
    # Background subtraction - moderate blur for rod shapes
    gaussian_sigma=12.0,
    
    # Morphological operations - preserve elongated shapes
    morph_kernel_size=3,
    morph_iterations=1,
    dilate_iterations=1,
    erode_iterations=1,
    
    # Size range - accommodate elongated rods
    min_area_um2=3.0,      # Small rods
    max_area_um2=80.0,     # Long swarmer cells can be quite large
    
    # Shape - allow elongated particles
    min_circularity=0.08,  # Very elongated rods acceptable
    max_circularity=0.90,  # Reject perfect circles (artifacts)
    min_aspect_ratio=0.3,  # Can be very elongated
    max_aspect_ratio=5.0,  # Long rods up to 5:1 ratio
    
    # Intensity - rods have moderate contrast
    min_mean_intensity=25.0,
    max_mean_intensity=210.0,
    
    # Edges - rods have defined but not crystal-sharp edges
    max_edge_gradient=140.0,
    
    # Solidity - rods are fairly solid
    min_solidity=0.65,
    
    # Image fraction
    max_fraction_of_image=0.20,
    
    # Fluorescence
    fluor_gaussian_sigma=1.5,
    fluor_morph_kernel_size=3,
    fluor_min_area_um2=2.0,
    fluor_match_min_intersection_px=3.0,
)


# ==================================================
# Klebsiella pneumoniae Configuration
# ==================================================
KLEBSIELLA_PNEUMONIAE = SegmentationConfig(
    name="Klebsiella pneumoniae",
    description="Rod-shaped with prominent polysaccharide capsule",
    
    # Morphology: Short rods (0.3-1.0 µm × 0.6-6.0 µm)
    # Characteristics: LARGE CAPSULE (2-3× cell diameter), non-motile
    
    # Background subtraction - stronger blur for diffuse capsule
    gaussian_sigma=18.0,
    
    # Morphological operations - gentler to preserve capsule
    morph_kernel_size=3,
    morph_iterations=1,
    dilate_iterations=2,  # More dilation for capsule
    erode_iterations=1,
    
    # Size range - capsule makes them appear larger
    min_area_um2=5.0,      # Small but visible with capsule
    max_area_um2=60.0,     # Capsule can make them quite large
    
    # Shape - capsule creates more circular appearance
    min_circularity=0.15,  # Some irregularity from capsule
    max_circularity=0.88,  # Reject geometric crystals
    min_aspect_ratio=0.4,  # Fairly round with capsule
    max_aspect_ratio=2.5,  # Short rods with capsule
    
    # Intensity - capsule creates lower contrast
    min_mean_intensity=20.0,   # Capsule can be quite faint
    max_mean_intensity=200.0,  # But reject overly bright crystals
    
    # Edges - capsule creates VERY soft edges
    max_edge_gradient=120.0,  # Lower than others - soft capsule boundary
    
    # Solidity - capsule can create irregular boundaries
    min_solidity=0.60,  # Lower to accommodate capsule irregularity
    
    # Image fraction
    max_fraction_of_image=0.15,
    
    # Fluorescence - capsule affects fluorescence
    fluor_gaussian_sigma=2.0,  # More blur for diffuse capsule
    fluor_morph_kernel_size=3,
    fluor_min_area_um2=4.0,
    fluor_match_min_intersection_px=4.0,
)


# ==================================================
# Streptococcus mitis Configuration
# ==================================================
STREPTOCOCCUS_MITIS = SegmentationConfig(
    name="Streptococcus mitis",
    description="Spherical cocci, often in chains or pairs",
    
    # Morphology: Cocci (0.5-1.0 µm diameter)
    # Characteristics: Pairs/chains, alpha-hemolytic, Gram-positive
    
    # Background subtraction - less blur for small cocci
    gaussian_sigma=10.0,
    
    # Morphological operations - preserve small spheres
    morph_kernel_size=3,
    morph_iterations=1,
    dilate_iterations=1,
    erode_iterations=1,
    
    # Size range - small cocci, chains appear larger
    min_area_um2=0.8,      # Single small coccus
    max_area_um2=40.0,     # Chain of cocci
    
    # Shape - cocci are circular, chains less so
    min_circularity=0.20,  # Chains can be irregular
    max_circularity=0.95,  # Allow near-perfect circles (cocci are spherical)
    min_aspect_ratio=0.4,  # Chains can be elongated
    max_aspect_ratio=3.0,  # Short chains
    
    # Intensity - Gram-positive have good contrast
    min_mean_intensity=30.0,
    max_mean_intensity=220.0,
    
    # Edges - small cocci have sharper edges than rods
    max_edge_gradient=160.0,  # Higher - cocci edges are defined
    
    # Solidity - cocci and chains are fairly solid
    min_solidity=0.70,
    
    # Image fraction
    max_fraction_of_image=0.18,
    
    # Fluorescence
    fluor_gaussian_sigma=1.2,  # Less blur for small particles
    fluor_morph_kernel_size=3,
    fluor_min_area_um2=0.5,   # Smaller minimum for cocci
    fluor_match_min_intersection_px=2.0,  # Lower for small particles
)


# ==================================================
# Default Configuration (General Purpose)
# ==================================================
DEFAULT = SegmentationConfig(
    name="Default",
    description="General-purpose settings for mixed or unknown bacteria",
    
    # Balanced parameters that work reasonably for most bacteria
    
    # Background subtraction - moderate
    gaussian_sigma=15.0,
    
    # Morphological operations - standard
    morph_kernel_size=3,
    morph_iterations=1,
    dilate_iterations=1,
    erode_iterations=1,
    
    # Size range - broad to accommodate various types
    min_area_um2=2.0,
    max_area_um2=100.0,
    
    # Shape - permissive but reject obvious artifacts
    min_circularity=0.10,
    max_circularity=0.92,
    min_aspect_ratio=0.3,
    max_aspect_ratio=4.0,
    
    # Intensity - broad range
    min_mean_intensity=20.0,
    max_mean_intensity=220.0,
    
    # Edges - moderate
    max_edge_gradient=150.0,
    
    # Solidity - moderate
    min_solidity=0.65,
    
    # Image fraction
    max_fraction_of_image=0.20,
    
    # Fluorescence
    fluor_gaussian_sigma=1.5,
    fluor_morph_kernel_size=3,
    fluor_min_area_um2=2.0,
    fluor_match_min_intersection_px=3.0,
)


# ==================================================
# Configuration Registry
# ==================================================
CONFIGS = {
    'proteus': PROTEUS_MIRABILIS,
    'proteus_mirabilis': PROTEUS_MIRABILIS,
    'klebsiella': KLEBSIELLA_PNEUMONIAE,
    'klebsiella_pneumoniae': KLEBSIELLA_PNEUMONIAE,
    'k_pneumoniae': KLEBSIELLA_PNEUMONIAE,
    'streptococcus': STREPTOCOCCUS_MITIS,
    'streptococcus_mitis': STREPTOCOCCUS_MITIS,
    's_mitis': STREPTOCOCCUS_MITIS,
    'default': DEFAULT,
}


def get_config(bacteria_type: str) -> SegmentationConfig:
    """
    Get configuration for specific bacteria type
    
    Args:
        bacteria_type: Name of bacteria (case-insensitive)
        
    Returns:
        SegmentationConfig for the bacteria
        
    Examples:
        >>> config = get_config('klebsiella')
        >>> config = get_config('Proteus mirabilis')
        >>> config = get_config('default')
    """
    key = bacteria_type.lower().replace(' ', '_')
    return CONFIGS.get(key, DEFAULT)


def list_available_configs() -> list[str]:
    """List all available bacteria configurations"""
    unique_configs = set(CONFIGS.values())
    return [cfg.name for cfg in unique_configs]


def print_config_comparison():
    """Print a comparison table of all configurations"""
    configs = [PROTEUS_MIRABILIS, KLEBSIELLA_PNEUMONIAE, STREPTOCOCCUS_MITIS, DEFAULT]
    
    print("\n" + "="*100)
    print("BACTERIA SEGMENTATION PARAMETERS COMPARISON")
    print("="*100 + "\n")
    
    # Header
    print(f"{'Parameter':<30} {'Proteus':<15} {'Klebsiella':<15} {'Streptococcus':<15} {'Default':<15}")
    print("-"*100)
    
    # Key parameters
    params = [
        ('gaussian_sigma', 'Gaussian Blur'),
        ('min_area_um2', 'Min Area (µm²)'),
        ('max_area_um2', 'Max Area (µm²)'),
        ('min_circularity', 'Min Circularity'),
        ('max_circularity', 'Max Circularity'),
        ('min_aspect_ratio', 'Min Aspect Ratio'),
        ('max_aspect_ratio', 'Max Aspect Ratio'),
        ('max_edge_gradient', 'Max Edge Gradient'),
        ('min_solidity', 'Min Solidity'),
    ]
    
    for attr, label in params:
        values = [f"{getattr(cfg, attr):.2f}" for cfg in configs]
        print(f"{label:<30} {values[0]:<15} {values[1]:<15} {values[2]:<15} {values[3]:<15}")
    
    print("\n" + "="*100 + "\n")
    