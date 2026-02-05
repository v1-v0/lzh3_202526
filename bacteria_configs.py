"""
Bacteria configuration system with JSON storage
Compatible with PyInstaller packaging
"""
import json
import shutil
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from cv2 import invert


@dataclass
class SegmentationConfig:
    """Configuration for bacteria segmentation parameters"""
    name: str
    description: str
    gaussian_sigma: float = 15.0
    min_area_um2: float = 3.0
    max_area_um2: float = 2000.0
    dilate_iterations: int = 1
    erode_iterations: int = 1
    morph_kernel_size: int = 3
    morph_iterations: int = 1
    min_circularity: float = 0.0
    max_circularity: float = 1.0
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 10.0
    min_mean_intensity: float = 0
    max_mean_intensity: float = 255
    max_edge_gradient: float = 200
    min_solidity: float = 0.3
    max_fraction_of_image: float = 0.25
    fluor_min_area_um2: float = 3.0
    fluor_max_area_um2: float = 2000.0
    fluor_match_min_intersection_px: float = 5.0
    invert_image: bool = False
    
    # Metadata fields
    last_modified: Optional[str] = None
    pixel_size_um: Optional[float] = None
    tuned_by: Optional[str] = None

    @property
    def min_area_px(self) -> float:
        """Convert min_area_um2 to pixels²"""
        um2_per_px2 = self.pixel_size_um ** 2 if self.pixel_size_um else 0.012
        return self.min_area_um2 / um2_per_px2

    @property
    def max_area_px(self) -> float:
        """Convert max_area_um2 to pixels²"""
        um2_per_px2 = self.pixel_size_um ** 2 if self.pixel_size_um else 0.012
        return self.max_area_um2 / um2_per_px2


class BacteriaConfigManager:
    """Manages bacteria configurations with JSON storage"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize manager
        
        Args:
            config_dir: Directory for config files. If None, uses default location.
        """
        if config_dir is None:
            # Use directory next to script/executable
            if hasattr(Path, 'home'):
                base_dir = Path(__file__).parent if not getattr(sys, 'frozen', False) else Path(sys.executable).parent
            else:
                base_dir = Path.cwd()
            
            self.config_dir = base_dir / "bacteria_configs"
        else:
            self.config_dir = Path(config_dir)
        
        self.config_dir.mkdir(exist_ok=True)
        
        self._configs: Dict[str, SegmentationConfig] = {}
        self._config_files: Dict[str, Path] = {}
        
        self._load_all_configs()
    
    def _get_config_path(self, bacteria_key: str) -> Path:
        """Get path for a bacteria configuration file"""
        return self.config_dir / f"{bacteria_key}.json"
    
    def _load_all_configs(self):
        """Load all JSON config files from directory"""
        json_files = list(self.config_dir.glob("*.json"))
        
        if not json_files:
            print("📁 No configs found, creating defaults...")
            self._create_defaults()
            return
        
        loaded_count = 0
        for json_file in json_files:
            bacteria_key = json_file.stem
            try:
                config = self._load_single_config(json_file)
                self._configs[bacteria_key] = config
                self._config_files[bacteria_key] = json_file
                loaded_count += 1
            except Exception as e:
                print(f"⚠ Error loading {json_file.name}: {e}")
        
        print(f"✓ Loaded {loaded_count} bacteria configuration(s)")
        
        # Ensure default exists
        if 'default' not in self._configs:
            self._create_default_config()
    
    def _load_single_config(self, json_path: Path) -> SegmentationConfig:
        """Load a single configuration from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both flat and nested formats
        config_data = data.get('config', data)
        
        return SegmentationConfig(**config_data)
    
    def _create_defaults(self):
        """Create default built-in configurations"""
        defaults = {
            'proteus_mirabilis': SegmentationConfig(
                name='Proteus mirabilis',
                description='bacteria segmentation - Default settings',
                gaussian_sigma=18.49,
                min_area_um2=5.99,
                max_area_um2=123.48,
                last_modified=datetime.now().isoformat()
            ),
            'klebsiella_pneumoniae': SegmentationConfig(
                name='Klebsiella pneumoniae',
                description='bacteria segmentation - Default settings',
                gaussian_sigma=4.05,
                min_area_um2=4.19,
                max_area_um2=154.00,
                last_modified=datetime.now().isoformat()
            ),
            'streptococcus_mitis': SegmentationConfig(
                name='Streptococcus mitis',
                description='Alpha-hemolytic, gram-positive cocci in chains',
                gaussian_sigma=12.0,
                min_area_um2=0.2,
                max_area_um2=50.0,
                min_aspect_ratio=0.8,
                max_aspect_ratio=6.0,
                min_circularity=0.5,
                max_circularity=1.0,
                min_solidity=0.7,
                last_modified=datetime.now().isoformat()
            ),
            'default': SegmentationConfig(
                name='Default (General Purpose)',
                description='Generic bacteria detection profile',
                gaussian_sigma=15.0,
                min_area_um2=0.3,
                max_area_um2=2000.0,
                last_modified=datetime.now().isoformat()
            )
        }
        
        for bacteria_key, config in defaults.items():
            self._configs[bacteria_key] = config
            self._save_single_config(bacteria_key, config)
        
        print(f"✓ Created {len(defaults)} default configurations")
    
    def _create_default_config(self):
        """Create just the default config"""
        default_config = SegmentationConfig(
            name='Default (General Purpose)',
            description='Generic bacteria detection profile',
            gaussian_sigma=15.0,
            min_area_um2=0.3,
            max_area_um2=2000.0,
            last_modified=datetime.now().isoformat()
        )
        self._configs['default'] = default_config
        self._save_single_config('default', default_config)
    
    def _save_single_config(self, bacteria_key: str, config: SegmentationConfig):
        """Save a single configuration to JSON file"""
        config_path = self._get_config_path(bacteria_key)
        
        # Create backup if file exists
        if config_path.exists():
            backup_path = config_path.with_suffix('.json.bak')
            shutil.copy2(config_path, backup_path)
        
        # Prepare data
        config_dict = asdict(config)
        
        data = {
            'bacteria_key': bacteria_key,
            'config': config_dict,
            'metadata': {
                'version': '2.0',
                'file_created': datetime.now().isoformat(),
                'compatible_with': ['dev2a.py', 'tuner.py']
            }
        }
        
        # Write
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self._config_files[bacteria_key] = config_path
    
    def get_config(self, bacteria_type: str) -> SegmentationConfig:
        """Get configuration for specified bacteria type
        
        Args:
            bacteria_type: Bacteria identifier (e.g., 'klebsiella_pneumoniae')
            
        Returns:
            SegmentationConfig object
        """
        if bacteria_type not in self._configs:
            print(f"[WARN] Unknown bacteria type '{bacteria_type}', using default")
            if 'default' not in self._configs:
                self._create_default_config()
            return self._configs['default']
        
        return self._configs[bacteria_type]
    
    def update_config(self, bacteria_type: str, config: SegmentationConfig, 
                     create_backup: bool = True) -> bool:
        """Update or add a configuration
        
        Args:
            bacteria_type: Bacteria identifier
            config: SegmentationConfig object
            create_backup: Whether to create backup of old config
            
        Returns:
            True if successful
        """
        try:
            # Update timestamp
            config.last_modified = datetime.now().isoformat()
            
            config_file = self._get_config_path(bacteria_type)
            config_file.parent.mkdir(parents=True, exist_ok=True)

            config_dict = {
            'name': config.name,
            'description': config.description,
            'gaussian_sigma': config.gaussian_sigma,
            'min_area_um2': config.min_area_um2,
            'max_area_um2': config.max_area_um2,
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
            'fluor_min_area_um2': config.fluor_min_area_um2,
            'fluor_max_area_um2': config.fluor_max_area_um2,
            'fluor_match_min_intersection_px': config.fluor_match_min_intersection_px,
            
            'invert_image': config.invert_image,
            
            'pixel_size_um': config.pixel_size_um,
            'last_modified': config.last_modified,
            'tuned_by': config.tuned_by
            }

            json_data = {
                'bacteria_type': bacteria_type,
                'config': config_dict,
                'updated': datetime.now().isoformat()
            }

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)

            print(f"✓ Updated configuration: {bacteria_type}")
            print(f"  Saved to: {self._get_config_path(bacteria_type).name}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to update configuration: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def list_available_configs(self) -> List[str]:
        """Get list of available bacteria configurations"""
        return sorted(self._configs.keys())
    
    def get_config_info(self, bacteria_type: str) -> Dict:
        """Get detailed info about a configuration"""
        if bacteria_type not in self._configs:
            return {}
        
        config = self._configs[bacteria_type]
        config_path = self._config_files.get(bacteria_type)
        
        return {
            'name': config.name,
            'description': config.description,
            'last_modified': config.last_modified,
            'file_path': str(config_path) if config_path else None,
            'parameters': {
                'gaussian_sigma': config.gaussian_sigma,
                'min_area_um2': config.min_area_um2,
                'max_area_um2': config.max_area_um2,
                'dilate_iterations': config.dilate_iterations,
                'erode_iterations': config.erode_iterations,
            }
        }
    
    def export_config(self, bacteria_type: str, output_path: Path) -> bool:
        """Export a configuration to external file
        
        Args:
            bacteria_type: Bacteria identifier
            output_path: Path to export file
            
        Returns:
            True if successful
        """
        try:
            config = self.get_config(bacteria_type)
            config_dict = asdict(config)
            
            data = {
                'bacteria_type': bacteria_type,
                'config': config_dict,
                'exported': datetime.now().isoformat()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Exported {bacteria_type} to: {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ Export failed: {e}")
            return False
    
    def import_config(self, json_path: Path, bacteria_type: Optional[str] = None) -> bool:
        """Import a configuration from external file
        
        Args:
            json_path: Path to JSON file
            bacteria_type: Optional override for bacteria type key
            
        Returns:
            True if successful
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Determine bacteria type
            candidate_type = bacteria_type if bacteria_type is not None else data.get('bacteria_type')
            key = (candidate_type or json_path.stem)
            key = str(key).strip().lower().replace(' ', '_').replace('.', '').replace('-', '_')
            if not key:
                raise ValueError("Bacteria type key cannot be empty")
            
            # Load config
            config_data = data.get('config', data)
            config = SegmentationConfig(**config_data)
            
            # Save
            return self.update_config(key, config)
            
        except Exception as e:
            print(f"✗ Import failed: {e}")
            return False
    
    def print_summary(self):
        """Print summary of all configurations"""
        print("\n" + "="*80)
        print("BACTERIA CONFIGURATIONS")
        print("="*80)
        print(f"\nConfig directory: {self.config_dir}")
        print(f"Total configurations: {len(self._configs)}\n")
        
        for bacteria_key in sorted(self._configs.keys()):
            config = self._configs[bacteria_key]
            print(f"  • {config.name}")
            print(f"    Key: {bacteria_key}")
            print(f"    σ={config.gaussian_sigma:.1f}, "
                  f"Size={config.min_area_um2:.1f}-{config.max_area_um2:.1f} µm²")
            if config.last_modified:
                print(f"    Modified: {config.last_modified}")
            print()


# ==================================================
# Global instance and compatibility functions
# ==================================================

import sys

# Create global manager instance
_manager = BacteriaConfigManager()

# Legacy compatibility
bacteria_map = {
    '1': 'proteus_mirabilis',
    '2': 'klebsiella_pneumoniae',
    '3': 'streptococcus_mitis',
    '4': 'default'
}

bacteria_display_names = {
    'proteus_mirabilis': 'Proteus mirabilis',
    'klebsiella_pneumoniae': 'Klebsiella pneumoniae',
    'streptococcus_mitis': 'Streptococcus mitis',
    'default': 'Default (General Purpose)'
}


def get_config(bacteria_type: str) -> SegmentationConfig:
    """Get configuration for bacteria type"""
    return _manager.get_config(bacteria_type)


def list_available_configs() -> List[str]:
    """List all available configurations"""
    return _manager.list_available_configs()


def update_bacteria_config(bacterium: str, config: SegmentationConfig, 
                          backup: bool = True) -> bool:
    """Update bacteria configuration
    
    Args:
        bacterium: Bacteria name or key
        config: SegmentationConfig object
        backup: Whether to create backup (always True for JSON)
        
    Returns:
        True if successful
    """
    # Convert display name to key if needed
    bacteria_key = bacterium.lower().replace(' ', '_').replace('.', '').replace('-', '_')
    return _manager.update_config(bacteria_key, config, create_backup=backup)


def reload_configs():
    """Reload all configurations from disk"""
    global _manager
    _manager = BacteriaConfigManager()
    print("✓ Configurations reloaded")


# ==================================================
# Main (for testing)
# ==================================================

if __name__ == '__main__':
    _manager.print_summary()