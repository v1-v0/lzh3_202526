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


@dataclass
class SegmentationConfig:
    """Particle segmentation parameters loaded from a species JSON config."""

    # ── Identity ──────────────────────────────────────────────────────────────
    name:                str   = "Default"
    description:         str   = ""

    # ── Pre-processing ────────────────────────────────────────────────────────
    gaussian_sigma:      float = 15.0
    invert_image:        bool  = False

    # ── Morphological operations ──────────────────────────────────────────────
    dilate_iterations:   int   = 1
    erode_iterations:    int   = 1
    morph_kernel_size:   int   = 3
    morph_iterations:    int   = 1

    # ── Physical size thresholds (µm²) ───────────────────────────────────────
    min_area_um2:        float = 0.3
    max_area_um2:        float = 2000.0

    # ── Shape filters ─────────────────────────────────────────────────────────
    min_circularity:     float = 0.0
    max_circularity:     float = 1.0
    min_aspect_ratio:    float = 0.2
    max_aspect_ratio:    float = 10.0
    min_solidity:        float = 0.3
    max_fraction_of_image: float = 0.25

    # ── BF intensity filter (Fix 9: renamed from min/max_mean_intensity) ──────
    # In brightfield, bacteria are dark (low pixel value after 8-bit normalise).
    # min_mean_intensity_bf guards against near-zero artefacts (e.g. dead pixels).
    # max_mean_intensity_bf rejects bright debris/noise objects.
    min_mean_intensity_bf: float = 0.0
    max_mean_intensity_bf: float = 255.0

    # ── BF intensity reference (Fix 9: new field) ─────────────────────────────
    # Mean BF pixel intensity of confirmed bacteria for this species/condition.
    # Populated from object_stats.csv BF_Mean_Intensity column after first run.
    # Used for QC / reporting; NOT used as a hard filter.
    object_mean_intensity_bf: Optional[float] = None

    # ── Edge gradient filter ──────────────────────────────────────────────────
    max_edge_gradient:   float = 200.0

    # ── Thresholding ──────────────────────────────────────────────────────────
    threshold_mode:      str   = "otsu"
    manual_threshold:    int   = 127
    use_intensity_threshold: bool  = False
    intensity_threshold: float = 80.0

    # ── Fluorescence ──────────────────────────────────────────────────────────
    fluor_min_area_um2:              float = 3.0
    fluor_max_area_um2:              float = 2000.0
    fluor_match_min_intersection_px: float = 5.0

    # ── Metadata ──────────────────────────────────────────────────────────────
    pixel_size_um:       float         = 0.109492
    last_modified:       Optional[str] = None
    tuned_by:            Optional[str] = None
    expected_particles_per_image:              Optional[list] = None
    expected_test_vs_control_reduction_pct:    Optional[list] = None


# ---------------------------------------------------------------------------
# Internal helper — applied by both _load_single_config and import_config
# to normalise raw JSON dicts before passing to SegmentationConfig(**…).
# ---------------------------------------------------------------------------
def _normalise_config_data(config_data: dict) -> dict:
    """Sanitise a raw JSON dict so it can be passed to SegmentationConfig.

    Two problems are handled:

    1.  Old field-name rename (Fix 9)
        JSON files written before Fix 9 contain ``min_mean_intensity`` /
        ``max_mean_intensity``.  Silently upgrade them to the new names so
        old config files continue to load without manual editing.

    2.  Unknown keys
        Any key in the JSON that is not a declared dataclass field is
        stripped out.  This prevents ``TypeError: __init__() got an
        unexpected keyword argument`` when loading a config that was saved
        by an older or newer version of the code.

    Args:
        config_data: Raw dict from JSON (possibly containing old keys or
                     extra/unknown keys).

    Returns:
        Cleaned dict safe to pass directly to ``SegmentationConfig(**…)``.
    """
    d = dict(config_data)  # shallow copy — do not mutate caller's dict

    # ── Fix 9: rename shim ────────────────────────────────────────────────
    if 'min_mean_intensity' in d and 'min_mean_intensity_bf' not in d:
        d['min_mean_intensity_bf'] = d.pop('min_mean_intensity')
    if 'max_mean_intensity' in d and 'max_mean_intensity_bf' not in d:
        d['max_mean_intensity_bf'] = d.pop('max_mean_intensity')

    # ── Strip unknown keys ────────────────────────────────────────────────
    valid_fields = {f.name for f in fields(SegmentationConfig)}
    unknown = [k for k in d if k not in valid_fields]
    for k in unknown:
        d.pop(k)
        print(f"  [config loader] Ignored unknown field: '{k}'")

    return d


class BacteriaConfigManager:
    """Manages bacteria configurations with JSON storage"""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize manager

        Args:
            config_dir: Directory for config files. If None, uses default location.
        """
        if config_dir is None:
            if hasattr(Path, 'home'):
                base_dir = (
                    Path(__file__).parent
                    if not getattr(sys, 'frozen', False)
                    else Path(sys.executable).parent
                )
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
        """Load a single configuration from JSON file.

        Handles:
        • Flat vs nested {'config': {…}} JSON layout.
        • Fix-9 field rename  (min/max_mean_intensity → …_bf).
        • Unknown / extra keys introduced by other tool versions.
        • Even morph_kernel_size auto-correction.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Support both flat and wrapped formats
        config_data: dict = data.get('config', data)

        # Auto-correct even morph_kernel_size (must happen before normalise)
        if 'morph_kernel_size' in config_data:
            kernel_size = config_data['morph_kernel_size']
            if isinstance(kernel_size, int) and kernel_size % 2 == 0:
                print(
                    f"⚠️  WARNING: {json_path.name} has even "
                    f"morph_kernel_size={kernel_size}, "
                    f"correcting to {kernel_size + 1}"
                )
                config_data['morph_kernel_size'] = kernel_size + 1

        # Normalise: rename old keys + strip unknowns
        config_data = _normalise_config_data(config_data)

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

        config_dict = asdict(config)

        data = {
            'bacteria_type': bacteria_key,
            'config': config_dict,
            'updated': datetime.now().isoformat(),
            
        }

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

    def update_config(
        self,
        bacteria_type: str,
        config: SegmentationConfig,
        create_backup: bool = True,
    ) -> bool:
        """Update or add a configuration.

        Fix 9: ``min_mean_intensity`` / ``max_mean_intensity`` renamed to
        ``min_mean_intensity_bf`` / ``max_mean_intensity_bf`` in the JSON
        output so saved files are consistent with the dataclass field names.

        Args:
            bacteria_type: Bacteria identifier
            config: SegmentationConfig object
            create_backup: Whether to create backup of old config

        Returns:
            True if successful
        """
        try:
            config.last_modified = datetime.now().isoformat()

            config_file = self._get_config_path(bacteria_type)
            config_file.parent.mkdir(parents=True, exist_ok=True)

            config_dict = {
                'name':                          config.name,
                'description':                   config.description,
                'gaussian_sigma':                config.gaussian_sigma,
                'min_area_um2':                  config.min_area_um2,
                'max_area_um2':                  config.max_area_um2,
                'dilate_iterations':             config.dilate_iterations,
                'erode_iterations':              config.erode_iterations,
                'morph_kernel_size':             config.morph_kernel_size,
                'morph_iterations':              config.morph_iterations,
                'min_circularity':               config.min_circularity,
                'max_circularity':               config.max_circularity,
                'min_aspect_ratio':              config.min_aspect_ratio,
                'max_aspect_ratio':              config.max_aspect_ratio,
                # Fix 9: use renamed field names in JSON output
                'min_mean_intensity_bf':         config.min_mean_intensity_bf,
                'max_mean_intensity_bf':         config.max_mean_intensity_bf,
                'object_mean_intensity_bf':      config.object_mean_intensity_bf,
                'max_edge_gradient':             config.max_edge_gradient,
                'min_solidity':                  config.min_solidity,
                'max_fraction_of_image':         config.max_fraction_of_image,
                'fluor_min_area_um2':            config.fluor_min_area_um2,
                'fluor_max_area_um2':            config.fluor_max_area_um2,
                'fluor_match_min_intersection_px': config.fluor_match_min_intersection_px,
                'invert_image':                  config.invert_image,
                'threshold_mode':                config.threshold_mode,
                'manual_threshold':              config.manual_threshold,
                'use_intensity_threshold':       config.use_intensity_threshold,
                'intensity_threshold':           config.intensity_threshold,
                'pixel_size_um':                 config.pixel_size_um,
                'last_modified':                 config.last_modified,
                'tuned_by':                      config.tuned_by,
                'expected_particles_per_image':             config.expected_particles_per_image,
                'expected_test_vs_control_reduction_pct':   config.expected_test_vs_control_reduction_pct,
            }

            json_data = {
                'bacteria_type': bacteria_type,
                'config': config_dict,
                'updated': datetime.now().isoformat()
            }

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)

            # Keep in-memory cache consistent
            self._configs[bacteria_type] = config
            self._config_files[bacteria_type] = config_file

            print(f"✓ Updated configuration: {bacteria_type}")
            print(f"  Saved to: {config_file.name}")
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
            'name':        config.name,
            'description': config.description,
            'last_modified': config.last_modified,
            'file_path':   str(config_path) if config_path else None,
            'parameters': {
                'gaussian_sigma':    config.gaussian_sigma,
                'min_area_um2':      config.min_area_um2,
                'max_area_um2':      config.max_area_um2,
                'dilate_iterations': config.dilate_iterations,
                'erode_iterations':  config.erode_iterations,
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

    def import_config(
        self,
        json_path: Path,
        bacteria_type: Optional[str] = None,
    ) -> bool:
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

            # Determine bacteria type key
            candidate_type = (
                bacteria_type if bacteria_type is not None
                else data.get('bacteria_type')
            )
            key = str(candidate_type or json_path.stem)
            key = key.strip().lower().replace(' ', '_').replace('.', '').replace('-', '_')
            if not key:
                raise ValueError("Bacteria type key cannot be empty")

            # Normalise: rename old keys + strip unknowns
            config_data = _normalise_config_data(data.get('config', data))
            config = SegmentationConfig(**config_data)

            return self.update_config(key, config)

        except Exception as e:
            print(f"✗ Import failed: {e}")
            return False

    def print_summary(self):
        """Print summary of all configurations"""
        print("\n" + "=" * 80)
        print("BACTERIA CONFIGURATIONS")
        print("=" * 80)
        print(f"\nConfig directory: {self.config_dir}")
        print(f"Total configurations: {len(self._configs)}\n")

        for bacteria_key in sorted(self._configs.keys()):
            config = self._configs[bacteria_key]
            print(f"  • {config.name}")
            print(f"    Key: {bacteria_key}")
            print(
                f"    σ={config.gaussian_sigma:.1f}, "
                f"Size={config.min_area_um2:.1f}–{config.max_area_um2:.1f} µm²"
            )
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
    'proteus_mirabilis':    'Proteus mirabilis',
    'klebsiella_pneumoniae': 'Klebsiella pneumoniae',
    'streptococcus_mitis':  'Streptococcus mitis',
    'default':              'Default (General Purpose)'
}


def get_config(bacteria_type: str) -> SegmentationConfig:
    """Get configuration for bacteria type"""
    return _manager.get_config(bacteria_type)


def list_available_configs() -> List[str]:
    """List all available configurations"""
    return _manager.list_available_configs()


def update_bacteria_config(
    bacterium: str,
    config: SegmentationConfig,
    backup: bool = True,
) -> bool:
    """Update bacteria configuration

    Args:
        bacterium: Bacteria name or key
        config: SegmentationConfig object
        backup: Whether to create backup (always True for JSON)

    Returns:
        True if successful
    """
    if config.morph_kernel_size % 2 == 0:
        print("⚠ Warning: morph_kernel_size should be odd for best results")
        config.morph_kernel_size += 1  # Auto-correct to next odd number

    bacteria_key = (
        bacterium.lower()
        .replace(' ', '_')
        .replace('.', '')
        .replace('-', '_')
    )
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