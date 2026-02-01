"""
Bacteria configuration system with SegmentationConfig class
Supports both built-in configs and JSON parameter overrides from feedback_tuner.py
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import json


@dataclass
class SegmentationConfig:
    """Configuration for bacteria segmentation parameters"""
    
    # Metadata
    name: str
    description: str
    
    # Core segmentation
    gaussian_sigma: float = 15.0
    
    # Size filtering (in µm²)
    min_area_um2: float = 3.0
    max_area_um2: float = 2000.0
    
    # Morphological operations
    dilate_iterations: int = 1
    erode_iterations: int = 1
    morph_kernel_size: int = 3
    morph_iterations: int = 1
    
    # Shape filters
    min_circularity: float = 0.0
    max_circularity: float = 1.0
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 10.0
    
    # Intensity filters
    min_mean_intensity: float = 0
    max_mean_intensity: float = 255
    max_edge_gradient: float = 200
    
    # Other filters
    min_solidity: float = 0.3
    max_fraction_of_image: float = 0.25
    
    # Fluorescence parameters
    fluor_min_area_um2: float = 3.0
    fluor_match_min_intersection_px: float = 5.0

    @property
    def min_area_px(self) -> float:
        """Convert min_area_um2 to pixels²"""
        um2_per_px2 = 0.012  # 0.109 µm/px → 0.011881 µm²/px²
        return self.min_area_um2 / um2_per_px2
    
    @property
    def max_area_px(self) -> float:
        """Convert max_area_um2 to pixels²"""
        um2_per_px2 = 0.012
        return self.max_area_um2 / um2_per_px2


# Built-in bacteria configurations
PROTEUS_MIRABILIS = SegmentationConfig(
    name="Proteus mirabilis",
    description="Highly motile, urease-positive, swarming behavior",
    gaussian_sigma=15.0,
    min_area_um2=3.0,
    max_area_um2=2000.0,
    min_aspect_ratio=0.3,
    max_aspect_ratio=8.0,
)

KLEBSIELLA_PNEUMONIAE = SegmentationConfig(
    name="Klebsiella pneumoniae",
    description="Encapsulated, non-motile, lactose-fermenting",
    gaussian_sigma=4.1,  # From your JSON file!
    min_area_um2=2.88,   # 240px * 0.012 µm²/px
    max_area_um2=87.99,  # 7333px * 0.012 µm²/px
    min_aspect_ratio=0.2,
    max_aspect_ratio=10.0,
)

STREPTOCOCCUS_MITIS = SegmentationConfig(
    name="Streptococcus mitis",
    description="Alpha-hemolytic, cocci in chains",
    gaussian_sigma=12.0,
    min_area_um2=2.0,
    max_area_um2=1500.0,
    min_aspect_ratio=0.4,
    max_aspect_ratio=6.0,
)

DEFAULT = SegmentationConfig(
    name="Default (General Purpose)",
    description="Generic bacteria detection profile",
    gaussian_sigma=15.0,
    min_area_um2=3.0,
    max_area_um2=2000.0,
)

# Configuration registry
_CONFIGS: Dict[str, SegmentationConfig] = {
    'proteus_mirabilis': PROTEUS_MIRABILIS,
    'klebsiella_pneumoniae': KLEBSIELLA_PNEUMONIAE,
    'streptococcus_mitis': STREPTOCOCCUS_MITIS,
    'default': DEFAULT,
}

# User-friendly mappings for CLI
bacteria_map = {
    '1': 'proteus_mirabilis',
    '2': 'klebsiella_pneumoniae',
    '3': 'streptococcus_mitis',
    '4': 'default',
}

bacteria_display_names = {
    'proteus_mirabilis': 'Proteus mirabilis',
    'klebsiella_pneumoniae': 'Klebsiella pneumoniae',
    'streptococcus_mitis': 'Streptococcus mitis',
    'default': 'Default (General Purpose)',
}


def get_config(bacteria_type: str) -> SegmentationConfig:
    """Get configuration for specified bacteria type
    
    Args:
        bacteria_type: Bacteria identifier (e.g., 'klebsiella_pneumoniae')
        
    Returns:
        SegmentationConfig object
        
    Raises:
        KeyError: If bacteria_type not found
    """
    if bacteria_type not in _CONFIGS:
        print(f"[WARN] Unknown bacteria type '{bacteria_type}', using default")
        return DEFAULT
    
    return _CONFIGS[bacteria_type]


def list_available_configs() -> list[str]:
    """Get list of available bacteria configurations
    
    Returns:
        List of bacteria type identifiers
    """
    return list(_CONFIGS.keys())


def print_config_comparison(configs: list[SegmentationConfig]) -> None:
    """Print comparison table of multiple configurations
    
    Args:
        configs: List of SegmentationConfig objects to compare
    """
    if not configs:
        print("No configurations to compare")
        return
    
    print("\n" + "=" * 80)
    print("CONFIGURATION COMPARISON")
    print("=" * 80)
    
    # Header
    print(f"\n{'Parameter':<30}", end='')
    for config in configs:
        print(f"{config.name[:15]:>15}", end='')
    print()
    print("-" * 80)
    
    # Parameters to compare
    params = [
        ('Gaussian σ', 'gaussian_sigma'),
        ('Min Area (µm²)', 'min_area_um2'),
        ('Max Area (µm²)', 'max_area_um2'),
        ('Min Aspect Ratio', 'min_aspect_ratio'),
        ('Max Aspect Ratio', 'max_aspect_ratio'),
        ('Min Circularity', 'min_circularity'),
        ('Dilate Iterations', 'dilate_iterations'),
        ('Erode Iterations', 'erode_iterations'),
    ]
    
    for label, attr in params:
        print(f"{label:<30}", end='')
        for config in configs:
            value = getattr(config, attr)
            if isinstance(value, float):
                print(f"{value:>15.2f}", end='')
            else:
                print(f"{value:>15}", end='')
        print()
    
    print("=" * 80 + "\n")


def load_config_from_json(json_path: Path) -> SegmentationConfig:
    """Load configuration from feedback_tuner JSON export
    
    Args:
        json_path: Path to JSON parameter file
        
    Returns:
        SegmentationConfig object
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON format is invalid
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        params = data.get('parameters', {})
        bacterium = data.get('bacterium', 'Unknown')
        structure = data.get('structure', 'bacteria')
        mode = data.get('mode', 'DARK')
        
        # Pixel size conversion factor (adjust if needed)
        # From your feedback_tuner: ~0.109 µm/px → 0.012 µm²/px²
        um2_per_px2 = 0.012
        
        # Map JSON parameters to SegmentationConfig
        config = SegmentationConfig(
            name=f"{bacterium} ({structure}, {mode}) [JSON]",
            description=f"Parameters loaded from {json_path.name}",
            
            # Core segmentation
            gaussian_sigma=float(params.get('gaussian_sigma', 15.0)),
            
            # Size filtering - convert from pixels² to µm²
            min_area_um2=float(params.get('min_area', 240.0)) * um2_per_px2,
            max_area_um2=float(params.get('max_area', 7333.0)) * um2_per_px2,
            
            # Morphology
            dilate_iterations=int(params.get('dilate_iterations', 1)),
            erode_iterations=int(params.get('erode_iterations', 1)),
            morph_kernel_size=3,
            morph_iterations=1,
            
            # Shape filters - use defaults (not in JSON)
            min_circularity=0.0,
            max_circularity=1.0,
            min_aspect_ratio=0.2,
            max_aspect_ratio=10.0,
            
            # Intensity filters - defaults
            min_mean_intensity=0,
            max_mean_intensity=255,
            max_edge_gradient=200,
            
            # Other defaults
            min_solidity=0.3,
            max_fraction_of_image=0.25,
            
            # Fluorescence
            fluor_min_area_um2=3.0,
            fluor_match_min_intersection_px=5.0,
        )
        
        print(f"  ✓ Loaded JSON parameters:")
        print(f"    • Bacterium: {bacterium}")
        print(f"    • Gaussian σ: {config.gaussian_sigma:.2f}")
        print(f"    • Size range: {config.min_area_um2:.2f} - {config.max_area_um2:.2f} µm²")
        
        return config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load config: {e}")


# Maintain backward compatibility with old bacteria_configs dict
bacteria_configs = {
    "Klebsiella Pneumoniae": {
        "bacteria_segmentation": {
            "invert_image": False,
            "gaussian_sigma": 4.1,
            "brightness_adjust": 70,
            "contrast_adjust": 2.0,
            "threshold_offset": -40,
            "min_area": 240.31,
            "max_area": 7332.94,
            "dilate_iterations": 0,
            "erode_iterations": 0,
        },
        "inclusion_dark_segmentation": {
            "invert_image": False,
            "gaussian_sigma": 2.0,
            "brightness_adjust": 0,
            "contrast_adjust": 1.0,
            "threshold_offset": -10,
            "min_area": 10,
            "max_area": 2000,
            "dilate_iterations": 1,
            "erode_iterations": 1,
        },
        "inclusion_bright_segmentation": {
            "invert_image": True,
            "gaussian_sigma": 2.0,
            "brightness_adjust": 0,
            "contrast_adjust": 1.0,
            "threshold_offset": -10,
            "min_area": 10,
            "max_area": 2000,
            "dilate_iterations": 1,
            "erode_iterations": 1,
        },
    },
}


if __name__ == "__main__":
    # Demo
    print("Available configurations:")
    for bacteria_type in list_available_configs():
        config = get_config(bacteria_type)
        print(f"\n  {config.name}")
        print(f"    {config.description}")
        print(f"    σ={config.gaussian_sigma:.1f}, Size={config.min_area_um2:.1f}-{config.max_area_um2:.1f} µm²")
    
    # Demo comparison
    print("\n")
    configs_to_compare = [
        get_config('proteus_mirabilis'),
        get_config('klebsiella_pneumoniae'),
        get_config('streptococcus_mitis'),
    ]
    print_config_comparison(configs_to_compare)