"""
Integrated Pathogen Configuration Manager
Combines main menu, config management, and segmentation tuner
"""

import os
import sys
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider, Button
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, fields
import xml.etree.ElementTree as ET
import ast

try:
    import astor
except ImportError:
    print("⚠️ Warning: 'astor' module not found. Install with: pip install astor")
    astor = None

# ==================================================
# SECTION 1: Configuration Data Classes
# ==================================================

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
    fluor_match_min_intersection_px: float = 5.0

    @property
    def min_area_px(self) -> float:
        """Convert min_area_um2 to pixels²"""
        um2_per_px2 = 0.012
        return self.min_area_um2 / um2_per_px2

    @property
    def max_area_px(self) -> float:
        """Convert max_area_um2 to pixels²"""
        um2_per_px2 = 0.012
        return self.max_area_um2 / um2_per_px2


# Built-in configurations
PROTEUS_MIRABILIS = SegmentationConfig(
    name='Proteus mirabilis',
    description='bacteria segmentation - Tuned 2026-02-02',
    gaussian_sigma=2.17,
    min_area_um2=3.60,
    max_area_um2=72.11
)

KLEBSIELLA_PNEUMONIAE = SegmentationConfig(
    name='Klebsiella pneumoniae',
    description='bacteria segmentation - Tuned 2026-02-03',
    gaussian_sigma=4.05,
    min_area_um2=4.19,
    max_area_um2=154.00
)

STREPTOCOCCUS_MITIS = SegmentationConfig(
    name='Streptococcus mitis',
    description='Alpha-hemolytic, gram-positive cocci in chains',
    gaussian_sigma=12.0,
    min_area_um2=0.2,
    max_area_um2=50.0,
    min_aspect_ratio=0.8,
    max_aspect_ratio=6.0,
    min_circularity=0.5,
    max_circularity=1.0,
    min_solidity=0.7
)

DEFAULT_CONFIG = SegmentationConfig(
    name='Default (General Purpose)',
    description='Generic bacteria detection profile',
    gaussian_sigma=15.0,
    min_area_um2=0.3,
    max_area_um2=2000.0
)

_CONFIGS: Dict[str, SegmentationConfig] = {
    'proteus_mirabilis': PROTEUS_MIRABILIS,
    'klebsiella_pneumoniae': KLEBSIELLA_PNEUMONIAE,
    'streptococcus_mitis': STREPTOCOCCUS_MITIS,
    'default': DEFAULT_CONFIG
}


def get_config(bacteria_type: str) -> SegmentationConfig:
    """Get configuration for specified bacteria type"""
    if bacteria_type not in _CONFIGS:
        print(f"[WARN] Unknown bacteria type '{bacteria_type}', using default")
        return DEFAULT_CONFIG
    return _CONFIGS[bacteria_type]


# ==================================================
# SECTION 2: Config File Manager (AST-based)
# ==================================================

class ConfigFileManager:
    """Manages bacteria_configs.py using AST parsing"""
    
    def __init__(self, config_file: Path):
        self.config_file = config_file
        self.tree: Optional[ast.Module] = None
        self.source: Optional[str] = None
        
    def load(self) -> bool:
        """Load and parse the config file"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.source = f.read()
            
            self.tree = ast.parse(self.source)
            return True
        except Exception as e:
            print(f"❌ Failed to load config file: {e}")
            return False
    
    def find_config_assignment(self, var_name: str) -> Optional[Tuple[int, ast.Assign]]:
        """Find the assignment node for a specific config variable"""
        if self.tree is None:
            return None
        
        for idx, node in enumerate(self.tree.body):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        return idx, node
        
        return None
    
    def create_config_assignment(self, var_name: str, config_data: dict) -> ast.Assign:
        """Create an AST assignment node for a SegmentationConfig"""
        keywords = []
        
        for key, value in config_data.items():
            if isinstance(value, str):
                value_node = ast.Constant(value=value)
            elif isinstance(value, (int, float)):
                value_node = ast.Constant(value=value)
            elif isinstance(value, bool):
                value_node = ast.Constant(value=value)
            else:
                value_node = ast.Constant(value=value)
            
            keywords.append(ast.keyword(arg=key, value=value_node))
        
        config_call = ast.Call(
            func=ast.Name(id='SegmentationConfig', ctx=ast.Load()),
            args=[],
            keywords=keywords
        )
        
        assignment = ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=config_call
        )
        
        return assignment
    
    def update_config(self, var_name: str, config_data: dict) -> bool:
        """Update or add a configuration"""
        if self.tree is None:
            if not self.load():
                return False
        
        assert self.tree is not None
        
        new_node = self.create_config_assignment(var_name, config_data)
        result = self.find_config_assignment(var_name)
        
        if result:
            idx, _ = result
            self.tree.body[idx] = new_node
            print(f"  ✓ Updated existing {var_name} configuration")
        else:
            default_idx = self._find_default_config_index()
            
            if default_idx is not None:
                self.tree.body.insert(default_idx, new_node)
                print(f"  ✓ Inserted new {var_name} configuration before DEFAULT")
            else:
                self.tree.body.append(new_node)
                print(f"  ✓ Appended new {var_name} configuration")
        
        return True
    
    def _find_default_config_index(self) -> Optional[int]:
        """Find the index of DEFAULT config assignment"""
        result = self.find_config_assignment('DEFAULT')
        return result[0] if result else None
    
    def save(self, backup: bool = True) -> bool:
        """Save the modified AST back to file"""
        if self.tree is None:
            print("❌ No AST tree to save")
            return False
        
        if astor is None:
            print("❌ 'astor' module not available. Cannot save.")
            return False
        
        try:
            if backup:
                backup_path = self.config_file.with_suffix('.py.bak')
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    with open(backup_path, 'w', encoding='utf-8') as bf:
                        bf.write(f.read())
                print(f"  ✓ Created backup: {backup_path.name}")
            
            new_source = astor.to_source(self.tree)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                f.write(new_source)
            
            print(f"  ✓ Saved to {self.config_file.name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save: {e}")
            return False
    
    def validate_syntax(self) -> bool:
        """Validate that the generated file has valid Python syntax"""
        if self.tree is None or astor is None:
            return False
        
        try:
            compile(astor.to_source(self.tree), str(self.config_file), 'exec')
            return True
        except SyntaxError as e:
            print(f"❌ Syntax error in generated code: {e}")
            return False


def config_to_dict(config: SegmentationConfig) -> dict:
    """Convert SegmentationConfig to dictionary"""
    return {
        field.name: getattr(config, field.name)
        for field in fields(config)
    }


def update_bacteria_config(bacterium: str, config: SegmentationConfig, backup: bool = True) -> bool:
    """Update bacteria configuration in bacteria_configs.py"""
    config_file = Path(__file__).parent / "bacteria_configs.py"
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return False
    
    manager = ConfigFileManager(config_file)
    
    if not manager.load():
        return False
    
    var_name = bacterium.upper().replace(' ', '_').replace('.', '')
    config_dict = config_to_dict(config)
    
    if not manager.update_config(var_name, config_dict):
        return False
    
    if not manager.validate_syntax():
        print("❌ Generated code has syntax errors - not saving")
        return False
    
    return manager.save(backup=backup)


# ==================================================
# SECTION 3: Unicode-Safe File I/O
# ==================================================

def safe_imread(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    """Read image with Unicode path support on Windows"""
    try:
        with open(path, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, flags)
        
        if img is None:
            print(f"[WARN] cv2.imdecode returned None for {path.name}")
            return None
            
        return img
    except Exception as e:
        print(f"[ERROR] Failed to read image {path.name}: {e}")
        return None


def safe_imwrite(path: Path, img: np.ndarray, params: Optional[list] = None) -> bool:
    """Write image with Unicode path support on Windows"""
    try:
        ext = path.suffix.lower()
        if not ext:
            ext = '.png'
        
        if params is None:
            is_success, buffer = cv2.imencode(ext, img)
        else:
            is_success, buffer = cv2.imencode(ext, img, params)
        
        if not is_success:
            print(f"[WARN] cv2.imencode failed for {path.name}")
            return False
        
        with open(path, 'wb') as f:
            f.write(buffer.tobytes())
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write image {path.name}: {e}")
        return False


def safe_xml_parse(xml_path: Path) -> Optional[ET.ElementTree]:
    """Parse XML file with Unicode path support"""
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        root = ET.fromstring(content)
        tree = ET.ElementTree(root)
        
        return tree
    except FileNotFoundError:
        print(f"[WARN] XML file not found: {xml_path.name}")
        return None
    except ET.ParseError as e:
        print(f"[ERROR] XML parse error in {xml_path.name}: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to read XML {xml_path.name}: {e}")
        return None


def validate_path_encoding(path: Path) -> bool:
    """Check if path can be properly encoded for filesystem"""
    try:
        path_str = str(path.resolve())
        path_str.encode(sys.getfilesystemencoding())
        return True
    except UnicodeEncodeError as e:
        print(f"[WARN] Path encoding issue: {path}")
        print(f"       {e}")
        return False
    except Exception as e:
        print(f"[WARN] Path validation error: {e}")
        return False


# ==================================================
# SECTION 4: Metadata Extraction
# ==================================================

def find_metadata_paths(img_path: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Find associated metadata XML files"""
    base = img_path.stem
    if base.endswith("_ch00"):
        base = base[:-5]
    md_dir = img_path.parent / "MetaData"
    xml_main = md_dir / f"{base}.xml"
    xml_props = md_dir / f"{base}_Properties.xml"

    return (
        xml_props if xml_props.exists() else None,
        xml_main if xml_main.exists() else None,
    )


def _require_attr(elem: ET.Element, attr: str, context: str) -> str:
    """Helper to extract required XML attribute"""
    v = elem.get(attr)
    if v is None:
        raise ValueError(f"Missing attribute '{attr}' in {context}")
    return v


def _parse_float(s: str) -> float:
    """Parse float with comma/period handling"""
    return float(s.strip().replace(",", "."))


def get_pixel_size_um(
    xml_props_path: Optional[Path],
    xml_main_path: Optional[Path],
) -> Tuple[float, float]:
    """Extract pixel size with detailed error reporting"""
    errors = []
    
    if xml_props_path is not None:
        try:
            tree = safe_xml_parse(xml_props_path)
            if tree is None:
                raise ValueError(f"Could not parse {xml_props_path.name}")
            
            root = tree.getroot()
            if root is None:
                raise ValueError(f"Empty XML document: {xml_props_path.name}")

            dims = root.findall(".//ImageDescription/Dimensions/DimensionDescription")
            by_id = {d.get("DimID"): d for d in dims}

            def read_dim(dim_id: str) -> Tuple[float, int, str]:
                d = by_id.get(dim_id)
                if d is None:
                    raise ValueError(f"Missing DimensionDescription with DimID='{dim_id}'")

                length_s = _require_attr(d, "Length", f"{xml_props_path.name} DimID={dim_id}")
                n_s = _require_attr(d, "NumberOfElements", f"{xml_props_path.name} DimID={dim_id}")
                unit = _require_attr(d, "Unit", f"{xml_props_path.name} DimID={dim_id}")

                length = _parse_float(length_s)
                n = int(n_s)
                return length, n, unit

            x_len, x_n, x_unit = read_dim("X")
            y_len, y_n, y_unit = read_dim("Y")

            if x_unit != "µm" or y_unit != "µm":
                raise ValueError(f"Unexpected units: X={x_unit}, Y={y_unit}")

            return float(x_len / x_n), float(y_len / y_n)
        except Exception as e:
            errors.append(f"Properties XML: {e}")

    if xml_main_path is not None:
        try:
            tree = safe_xml_parse(xml_main_path)
            if tree is None:
                raise ValueError(f"Could not parse {xml_main_path.name}")
            
            root = tree.getroot()
            if root is None:
                raise ValueError(f"Empty XML document: {xml_main_path.name}")

            dims = root.findall(".//ImageDescription/Dimensions/DimensionDescription")
            by_id = {d.get("DimID"): d for d in dims}

            def read_dim(dim_id: str) -> Tuple[float, int, str]:
                d = by_id.get(dim_id)
                if d is None:
                    raise ValueError(f"Missing DimensionDescription with DimID='{dim_id}'")

                length_s = _require_attr(d, "Length", f"{xml_main_path.name} DimID={dim_id}")
                n_s = _require_attr(d, "NumberOfElements", f"{xml_main_path.name} DimID={dim_id}")
                unit = _require_attr(d, "Unit", f"{xml_main_path.name} DimID={dim_id}")

                length = _parse_float(length_s)
                n = int(n_s)
                return length, n, unit

            x_len_m, x_n, x_unit = read_dim("1")
            y_len_m, y_n, y_unit = read_dim("2")

            if x_unit != "m" or y_unit != "m":
                raise ValueError(f"Unexpected units: X={x_unit}, Y={y_unit}")

            return float((x_len_m * 1e6) / x_n), float((y_len_m * 1e6) / y_n)
        except Exception as e:
            errors.append(f"Main XML: {e}")
    
    error_summary = "\n  - ".join(errors) if errors else "No XML files provided"
    raise ValueError(f"Could not determine pixel size (µm/px).\nAttempted sources:\n  - {error_summary}")


def normalize_to_8bit(img: np.ndarray) -> np.ndarray:
    """Normalize image to 8-bit"""
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        out = np.zeros_like(img, dtype=np.uint8)
        cv2.normalize(img, out, 0, 255, cv2.NORM_MINMAX)
        return out
    img_f = img.astype(np.float32)
    mn, mx = float(np.min(img_f)), float(np.max(img_f))
    if mx <= mn:
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img_f - mn) * (255.0 / (mx - mn))).clip(0, 255).astype(np.uint8)


# ==================================================
# SECTION 5: Segmentation Tuner
# ==================================================

class SegmentationTuner:
    """Interactive segmentation parameter tuner"""
    
    DEFAULT_PARAMS = {
        "gaussian_sigma": 2.0,
        "brightness_adjust": 0,
        "contrast_adjust": 1.0,
        "threshold_offset": 0,
        "min_area": 20,
        "max_area": 5000,
        "dilate_iterations": 0,
        "erode_iterations": 0,
    }
    
    FALLBACK_UM_PER_PX = 0.109492
    
    COLORS = {
        'bg': '#f0f0f0',
        'header': '#2c3e50',
        'primary': '#3498db',
        'success': '#27ae60',
        'warning': '#e67e22',
        'info': '#16a085',
        'secondary': '#34495e',
        'danger': '#e74c3c',
        'purple': '#9b59b6',
        'gray': '#95a5a6',
    }
    
    def __init__(self, root: tk.Tk, image_path: str, bacterium: str, structure: str, mode: str, return_callback=None):
        """Initialize the segmentation tuner"""
        self.root = root
        self.image_path = Path(image_path)
        self.bacterium = bacterium
        self.structure = structure
        self.mode = mode
        self.return_callback = return_callback
        
        if not validate_path_encoding(self.image_path):
            raise ValueError(f"Path contains problematic characters: {image_path}")
        
        self.original_image = self._load_image(self.image_path)
        self.pixel_size_um, self.has_metadata = self._load_pixel_size()
        
        self._initialize_parameters()
        
        self.processed_image: np.ndarray = np.zeros_like(self.original_image)
        self.binary_mask: np.ndarray = np.zeros_like(self.original_image)
        self.contours: List[np.ndarray] = []
        self.contour_areas: List[float] = []
        self.current_suggestions: Dict[str, Any] = {}
        
        self.sliders: Dict[str, Slider] = {}
        self.param_labels: Dict[str, tk.Label] = {}
        
        self.setup_gui()
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load and validate image"""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = safe_imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = normalize_to_8bit(image)
        
        print(f"✅ Loaded image: {image_path.name}")
        print(f"   Shape: {image.shape}, Dtype: {image.dtype}")
        return image
    
    def _load_pixel_size(self) -> Tuple[float, bool]:
        """Load pixel size from metadata with fallback"""
        try:
            xml_props, xml_main = find_metadata_paths(self.image_path)
            
            if xml_props or xml_main:
                um_per_px_x, um_per_px_y = get_pixel_size_um(xml_props, xml_main)
                um_per_px_avg = (um_per_px_x + um_per_px_y) / 2.0
                
                print(f"✅ Loaded pixel size from metadata: {um_per_px_avg:.6f} µm/px")
                return um_per_px_avg, True
            else:
                print(f"⚠ No metadata found, using fallback: {self.FALLBACK_UM_PER_PX} µm/px")
                return self.FALLBACK_UM_PER_PX, False
                
        except Exception as e:
            print(f"⚠ Error loading metadata, using fallback: {e}")
            return self.FALLBACK_UM_PER_PX, False
    
    def _initialize_parameters(self):
        """Initialize parameters"""
        self.params = self.DEFAULT_PARAMS.copy()
        self.invert_image = False
    
    def setup_gui(self):
        """Setup the GUI"""
        self.root.title(f"Segmentation Tuner - {self.bacterium}")
        self.root.geometry("1920x1080")
        self.root.minsize(1600, 900)
        self.root.configure(bg=self.COLORS['bg'])
        
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self._create_header(main_container)
        content_frame = self._create_content_area(main_container)
        self._create_control_panel(main_container)
        self._create_action_buttons(main_container)
        
        print("✅ GUI Setup Complete")
        self.update_visualization()
    
    def _create_header(self, parent: ttk.Frame):
        """Create header section"""
        header_frame = tk.Frame(parent, bg=self.COLORS['header'], height=45)
        header_frame.pack(fill=tk.X, pady=(0, 5))
        header_frame.pack_propagate(False)
        
        title_text = f"🔬 {self.bacterium} - {self.structure}"
        tk.Label(
            header_frame,
            text=title_text,
            font=("Segoe UI", 14, "bold"),
            bg=self.COLORS['header'],
            fg="white"
        ).pack(side=tk.LEFT, padx=20, pady=6)
        
        mode_text = f"Mode: {self.mode} {'(Inverted)' if self.invert_image else ''}"
        tk.Label(
            header_frame,
            text=mode_text,
            font=("Segoe UI", 9),
            bg=self.COLORS['secondary'],
            fg="white",
            padx=12,
            pady=4,
            relief=tk.RAISED
        ).pack(side=tk.LEFT, pady=6)
        
        pixel_color = self.COLORS['success'] if self.has_metadata else self.COLORS['warning']
        pixel_text = f"Pixel: {self.pixel_size_um:.6f} µm"
        if not self.has_metadata:
            pixel_text += " (fallback)"
        
        tk.Label(
            header_frame,
            text=pixel_text,
            font=("Segoe UI", 9),
            bg=pixel_color,
            fg="white",
            padx=12,
            pady=4,
            relief=tk.RAISED
        ).pack(side=tk.LEFT, padx=10, pady=6)
        
        tk.Button(
            header_frame,
            text="📁 LOAD IMAGE",
            font=("Segoe UI", 9, "bold"),
            bg=self.COLORS['primary'],
            fg="white",
            padx=12,
            pady=4,
            relief=tk.RAISED,
            command=self.load_new_image,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=10, pady=6)
        
        self.contour_count_label = tk.Label(
            header_frame,
            text="Contours: 0",
            font=("Segoe UI", 9),
            bg=self.COLORS['success'],
            fg="white",
            padx=10,
            pady=4,
            relief=tk.RAISED
        )
        self.contour_count_label.pack(side=tk.RIGHT, padx=20, pady=6)
    
    def _create_content_area(self, parent: ttk.Frame) -> ttk.Frame:
        """Create main content area"""
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        self._create_image_panel(content_frame)
        self._create_right_panel(content_frame)
        
        return content_frame
    
    def _create_image_panel(self, parent: ttk.Frame):
        """Create left image panel"""
        left_panel = ttk.Frame(parent, relief=tk.RIDGE, borderwidth=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        header = tk.Frame(left_panel, bg=self.COLORS['secondary'], height=28)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="📷 IMAGE ANALYSIS - Original + Contours",
            font=("Segoe UI", 10, "bold"),
            bg=self.COLORS['secondary'],
            fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=4)
        
        canvas_frame = ttk.Frame(left_panel)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_image = Figure(figsize=(13, 9.5), facecolor='white')
        self.ax_image = self.fig_image.add_subplot(111)
        self.canvas_image = FigureCanvasTkAgg(self.fig_image, canvas_frame)
        self.canvas_image.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_image.mpl_connect("button_press_event", self.on_image_click)
        
        instruction = tk.Frame(left_panel, bg=self.COLORS['primary'], height=25)
        instruction.pack(fill=tk.X)
        instruction.pack_propagate(False)
        
        tk.Label(
            instruction,
            text="💡 Click on a particle to analyze and get parameter suggestions",
            font=("Segoe UI", 8),
            bg=self.COLORS['primary'],
            fg="white"
        ).pack(pady=3)
    
    def _create_right_panel(self, parent: ttk.Frame):
        """Create right control panel"""
        right_panel = ttk.Frame(parent, width=380)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        self._create_parameters_section(right_panel)
        self._create_target_analysis_section(right_panel)
        self._create_histogram_section(right_panel)
    
    def _create_parameters_section(self, parent: ttk.Frame):
        """Create parameters display section"""
        header = tk.Frame(parent, bg=self.COLORS['secondary'], height=28)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="⚙️ PARAMETERS",
            font=("Segoe UI", 10, "bold"),
            bg=self.COLORS['secondary'],
            fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=4)
        
        display = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        display.pack(fill=tk.X, padx=5, pady=(5, 0))
        
        inner = ttk.Frame(display)
        inner.pack(fill=tk.X, padx=8, pady=8)
        
        metadata_status = "✓ From metadata" if self.has_metadata else "⚠ Fallback"
        self._add_param_section(inner, "Basic Information", [
            ("Pathogen:", self.bacterium, self.COLORS['danger']),
            ("Structure:", self.structure, self.COLORS['purple']),
            ("Mode:", f"{self.mode} particles", self.COLORS['primary']),
            ("Pixel size:", f"{self.pixel_size_um:.6f} µm/px", 
             self.COLORS['success'] if self.has_metadata else self.COLORS['warning']),
            ("Metadata:", metadata_status, 
             self.COLORS['success'] if self.has_metadata else self.COLORS['warning']),
        ])
        
        ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        
        self._add_param_section(inner, "Preprocessing", [
            ("Invert:", "ON" if self.invert_image else "OFF",
             self.COLORS['success'] if self.invert_image else self.COLORS['gray']),
            ("Gaussian σ:", f"{self.params['gaussian_sigma']:.1f}", None),
            ("Brightness:", f"{self.params['brightness_adjust']:+.0f}", None),
            ("Contrast:", f"{self.params['contrast_adjust']:.2f}", None),
            ("Threshold Δ:", f"{self.params['threshold_offset']:+.0f}", None),
        ])
        
        ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        
        um2_per_px2 = self.pixel_size_um ** 2
        min_area_um2 = self.params['min_area'] * um2_per_px2
        max_area_um2 = self.params['max_area'] * um2_per_px2
        
        self._add_param_section(inner, "Filtering & Morphology", [
            ("Min area:", f"{self.params['min_area']:.0f} px ({min_area_um2:.2f} µm²)", None),
            ("Max area:", f"{self.params['max_area']:.0f} px ({max_area_um2:.2f} µm²)", None),
            ("Dilate iter:", str(self.params["dilate_iterations"]), None),
            ("Erode iter:", str(self.params["erode_iterations"]), None),
        ])
    
    def _add_param_section(self, parent: ttk.Frame, title: str,
                           params: List[Tuple[str, str, Optional[str]]]):
        """Add a parameter display section"""
        tk.Label(
            parent,
            text=title,
            font=("Segoe UI", 9, "bold"),
            foreground=self.COLORS['header']
        ).pack(anchor="w", pady=(0, 4))
        
        for label_text, value_text, color in params:
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=1)
            
            tk.Label(
                frame,
                text=label_text,
                font=("Segoe UI", 8),
                foreground="gray",
                width=15,
                anchor="w"
            ).pack(side=tk.LEFT)
            
            if color:
                value_label = tk.Label(
                    frame,
                    text=value_text,
                    font=("Segoe UI", 8, "bold"),
                    foreground="white",
                    bg=color,
                    padx=6,
                    pady=1,
                    relief=tk.RAISED
                )
            else:
                value_label = tk.Label(
                    frame,
                    text=value_text,
                    font=("Segoe UI", 8, "bold"),
                    foreground=self.COLORS['header']
                )
            value_label.pack(side=tk.LEFT)
            
            self.param_labels[label_text] = value_label
    
    def _create_target_analysis_section(self, parent: ttk.Frame):
        """Create target analysis section"""
        header = tk.Frame(parent, bg=self.COLORS['warning'], height=26)
        header.pack(fill=tk.X, pady=(8, 0))
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="🎯 TARGET ANALYSIS",
            font=("Segoe UI", 9, "bold"),
            bg=self.COLORS['warning'],
            fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=3)
        
        display = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        display.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        self.target_analysis_label = tk.Label(
            display,
            text="Click on image to analyze a particle",
            font=("Segoe UI", 8),
            foreground="gray",
            justify=tk.LEFT,
            wraplength=360,
            bg="white",
            anchor="w",
            padx=8,
            pady=8
        )
        self.target_analysis_label.pack(fill=tk.BOTH, expand=True)
    
    def _create_histogram_section(self, parent: ttk.Frame):
        """Create histogram section"""
        header = tk.Frame(parent, bg=self.COLORS['info'], height=26)
        header.pack(fill=tk.X, pady=(5, 0))
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="📊 AREA DISTRIBUTION",
            font=("Segoe UI", 9, "bold"),
            bg=self.COLORS['info'],
            fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=3)
        
        canvas_frame = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        
        self.fig_hist = Figure(figsize=(5, 4), facecolor='white')
        self.ax_hist = self.fig_hist.add_subplot(111)
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, canvas_frame)
        self.canvas_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    def _create_control_panel(self, parent: ttk.Frame):
        """Create slider control panel"""
        panel = ttk.Frame(parent)
        panel.pack(fill=tk.X, pady=(5, 0))
        
        header = tk.Frame(panel, bg=self.COLORS['secondary'], height=28)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="🎚️ ADJUST PARAMETERS",
            font=("Segoe UI", 10, "bold"),
            bg=self.COLORS['secondary'],
            fg="white"
        ).pack(side=tk.LEFT, padx=10, pady=4)
        
        slider_frame = ttk.Frame(panel, relief=tk.SUNKEN, borderwidth=1)
        slider_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        self.fig_sliders = Figure(figsize=(18, 2.2), facecolor='#f8f9fa')
        self.canvas_sliders = FigureCanvasTkAgg(self.fig_sliders, slider_frame)
        self.canvas_sliders.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self._create_sliders()
    
    def _create_sliders(self):
        """Create parameter sliders"""
        self.fig_sliders.text(0.195, 0.95, 'PREPROCESSING', ha='center', va='top',
                             fontsize=9, fontweight='bold', color=self.COLORS['header'])
        self.fig_sliders.text(0.695, 0.95, 'FILTERING & MORPHOLOGY', ha='center', va='top',
                             fontsize=9, fontweight='bold', color=self.COLORS['header'])
        
        slider_configs = {
            'left': [
                ("gaussian_sigma", "Gaussian σ", 0.5, 20.0, 0.72, 0.04),
                ("brightness_adjust", "Brightness", -100, 100, 0.51, 0.04),
                ("contrast_adjust", "Contrast", 0.5, 3.0, 0.30, 0.04),
                ("threshold_offset", "Threshold Δ", -50, 50, 0.09, 0.04),
            ],
            'right': [
                ("min_area", "Min Area (px)", 10, 5000, 0.72, 0.51),
                ("max_area", "Max Area (px)", 100, 20000, 0.51, 0.51),
                ("dilate_iterations", "Dilate", 0, 5, 0.30, 0.51),
                ("erode_iterations", "Erode", 0, 5, 0.09, 0.51),
            ]
        }
        
        slider_height = 0.13
        slider_width = 0.38
        
        for param_key, label, vmin, vmax, y_pos, x_start in slider_configs['left']:
            ax = self.fig_sliders.add_axes((x_start, y_pos, slider_width, slider_height))
            valstep = 1 if "Brightness" in label or "Threshold" in label else None
            slider = Slider(ax, label, vmin, vmax, valinit=self.params[param_key],
                          valstep=valstep, color=self.COLORS['primary'])
            slider.on_changed(lambda val, key=param_key: self.update_parameter(key, val))
            self.sliders[param_key] = slider
        
        for param_key, label, vmin, vmax, y_pos, x_start in slider_configs['right']:
            ax = self.fig_sliders.add_axes((x_start, y_pos, slider_width, slider_height))
            valstep = 1 if "iter" in param_key else None
            slider = Slider(ax, label, vmin, vmax, valinit=self.params[param_key],
                          valstep=valstep, color=self.COLORS['info'])
            slider.on_changed(lambda val, key=param_key: self.update_parameter(key, val))
            self.sliders[param_key] = slider
        
        invert_color = self.COLORS['success'] if self.invert_image else self.COLORS['gray']
        invert_text = f'INVERT\n{"ON" if self.invert_image else "OFF"}'
        
        ax_invert = self.fig_sliders.add_axes((0.915, 0.51, 0.07, 0.34))
        self.btn_invert = Button(ax_invert, invert_text, color=invert_color,
                                hovercolor=self.COLORS['success'])
        self.btn_invert.on_clicked(self.toggle_invert)
        
        ax_apply = self.fig_sliders.add_axes((0.915, 0.09, 0.07, 0.34))
        self.btn_apply = Button(ax_apply, "APPLY\nSUGGESTIONS",
                               color=self.COLORS['primary'],
                               hovercolor='#5dade2')
        self.btn_apply.on_clicked(self.apply_suggestions)
    
    def process_image(self):
        """Process image with current parameters"""
        img = self.original_image.copy()
        
        if self.invert_image:
            img = cv2.bitwise_not(img)
        
        img = cv2.convertScaleAbs(
            img,
            alpha=self.params["contrast_adjust"],
            beta=self.params["brightness_adjust"]
        )
        
        if self.params["gaussian_sigma"] > 0:
            ksize = int(2 * np.ceil(2 * self.params["gaussian_sigma"]) + 1)
            if ksize % 2 == 0:
                ksize += 1
            img = cv2.GaussianBlur(img, (ksize, ksize), self.params["gaussian_sigma"])
        
        self.processed_image = img
        
        threshold_value = float(img.mean()) + self.params["threshold_offset"]
        thresh_type = cv2.THRESH_BINARY_INV if self.mode == "DARK" else cv2.THRESH_BINARY
        _, binary = cv2.threshold(img, threshold_value, 255, thresh_type)
        
        if self.params["dilate_iterations"] > 0:
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.dilate(binary, kernel,
                              iterations=int(self.params["dilate_iterations"]))
        
        if self.params["erode_iterations"] > 0:
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.erode(binary, kernel,
                             iterations=int(self.params["erode_iterations"]))
        
        self.binary_mask = binary
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.contours = []
        self.contour_areas = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.params["min_area"] <= area <= self.params["max_area"]:
                self.contours.append(cnt)
                self.contour_areas.append(area)
    
    def update_visualization(self):
        """Update all visualizations"""
        self.process_image()
        self._update_image_display()
        self._update_histogram()
        self._update_param_displays()
        self.contour_count_label.config(text=f"Contours: {len(self.contours)}")
    
    def _update_image_display(self):
        """Update main image display"""
        self.ax_image.clear()
        
        if len(self.original_image.shape) == 2:
            display = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB)
        else:
            display = self.original_image.copy()
        
        cv2.drawContours(display, self.contours, -1, (0, 255, 0), 2)
        
        self.ax_image.imshow(display)
        self.ax_image.set_title("Original + Contours", fontsize=12,
                               fontweight='bold', pad=10)
        self.ax_image.axis("off")
        
        self.canvas_image.draw()
    
    def _update_histogram(self):
        """Update area distribution histogram"""
        self.ax_hist.clear()
        
        if not self.contour_areas:
            self.ax_hist.text(0.5, 0.5, "No contours detected",
                            ha='center', va='center', fontsize=10, color='gray')
            self.ax_hist.set_xlim(0, 1)
            self.ax_hist.set_ylim(0, 1)
        else:
            areas_px = np.array(self.contour_areas)
            um2_per_px2 = self.pixel_size_um ** 2
            areas_um2 = areas_px * um2_per_px2
            
            self.ax_hist.hist(areas_um2, bins=30, color=self.COLORS['primary'],
                            alpha=0.7, edgecolor='black')
            
            median = float(np.median(areas_um2))
            mean = float(np.mean(areas_um2))
            min_area = float(np.min(areas_um2))
            max_area = float(np.max(areas_um2))
            
            self.ax_hist.axvline(median, color='orange', linestyle='--',
                               linewidth=2, label=f'Median: {median:.2f} µm²')
            self.ax_hist.axvline(mean, color='red', linestyle='--',
                               linewidth=2, label=f'Mean: {mean:.2f} µm²')
            self.ax_hist.axvline(min_area, color='green', linestyle=':',
                               linewidth=1.5, label=f'Min: {min_area:.2f} µm²')
            self.ax_hist.axvline(max_area, color='purple', linestyle=':',
                               linewidth=1.5, label=f'Max: {max_area:.2f} µm²')
            
            self.ax_hist.set_xlabel("Area (µm²)", fontsize=9)
            self.ax_hist.set_ylabel("Count", fontsize=9)
            self.ax_hist.set_title(f"Distribution (n={len(areas_um2)})",
                                  fontsize=10, fontweight='bold')
            self.ax_hist.legend(fontsize=7, loc='upper right')
            self.ax_hist.grid(True, alpha=0.3)
        
        self.canvas_hist.draw()
    
    def _update_param_displays(self):
        """Update parameter display values"""
        invert_label = self.param_labels["Invert:"]
        invert_text = "ON" if self.invert_image else "OFF"
        invert_color = self.COLORS['success'] if self.invert_image else self.COLORS['gray']
        invert_label.config(text=invert_text, bg=invert_color)
        
        um2_per_px2 = self.pixel_size_um ** 2
        min_area_um2 = self.params['min_area'] * um2_per_px2
        max_area_um2 = self.params['max_area'] * um2_per_px2
        
        updates = {
            "Gaussian σ:": f"{self.params['gaussian_sigma']:.1f}",
            "Brightness:": f"{self.params['brightness_adjust']:+.0f}",
            "Contrast:": f"{self.params['contrast_adjust']:.2f}",
            "Threshold Δ:": f"{self.params['threshold_offset']:+.0f}",
            "Min area:": f"{self.params['min_area']:.0f} px ({min_area_um2:.2f} µm²)",
            "Max area:": f"{self.params['max_area']:.0f} px ({max_area_um2:.2f} µm²)",
            "Dilate iter:": str(int(self.params["dilate_iterations"])),
            "Erode iter:": str(int(self.params["erode_iterations"])),
        }
        
        for key, value in updates.items():
            if key in self.param_labels:
                self.param_labels[key].config(text=value)
    
    def update_parameter(self, param_name: str, value: float):
        """Update a parameter and refresh visualization"""
        self.params[param_name] = value
        self.update_visualization()
    
    def toggle_invert(self, event):
        """Toggle image inversion"""
        self.invert_image = not self.invert_image
        
        invert_text = f'INVERT\n{"ON" if self.invert_image else "OFF"}'
        invert_color = self.COLORS['success'] if self.invert_image else self.COLORS['gray']
        self.btn_invert.label.set_text(invert_text)
        self.btn_invert.color = invert_color
        
        self.update_visualization()
    
    def on_image_click(self, event):
        """Handle click on image to analyze particle"""
        if event.inaxes != self.ax_image or event.xdata is None or event.ydata is None:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        
        clicked_contour = None
        for cnt in self.contours:
            if cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0:
                clicked_contour = cnt
                break
        
        if clicked_contour is None:
            self.target_analysis_label.config(
                text="No particle found at click location",
                foreground="red"
            )
            return
        
        self._analyze_particle(clicked_contour)
    
    def _analyze_particle(self, contour: np.ndarray):
        """Analyze a specific particle"""
        area_px = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        circularity = 4 * np.pi * area_px / (perimeter ** 2) if perimeter > 0 else 0
        
        um2_per_px2 = self.pixel_size_um ** 2
        area_um2 = area_px * um2_per_px2
        
        suggestions = self._generate_suggestions(area_px, circularity, aspect_ratio)
        self.current_suggestions = suggestions
        
        analysis_text = (
            f"🎯 Target Particle Analysis:\n\n"
            f"Area: {area_px:.1f} px² ({area_um2:.2f} µm²)\n"
            f"Perimeter: {perimeter:.1f} px\n"
            f"Aspect Ratio: {aspect_ratio:.2f}\n"
            f"Circularity: {circularity:.3f}\n\n"
            f"📊 Suggestions:\n"
        )
        
        for param, value in suggestions.items():
            if 'area' in param:
                value_um2 = value * um2_per_px2
                analysis_text += f"• {param}: {value} px ({value_um2:.2f} µm²)\n"
            else:
                analysis_text += f"• {param}: {value}\n"
        
        self.target_analysis_label.config(text=analysis_text, foreground="black")
    
    def _generate_suggestions(self, area: float, circularity: float,
                             aspect_ratio: float) -> Dict[str, Any]:
        """Generate parameter suggestions"""
        suggestions: Dict[str, Any] = {}
        
        suggestions["min_area"] = max(10, int(area * 0.3))
        suggestions["max_area"] = min(20000, int(area * 3.0))
        
        if circularity < 0.6:
            suggestions["dilate_iterations"] = min(3, self.params["dilate_iterations"] + 1)
            suggestions["erode_iterations"] = min(3, self.params["erode_iterations"] + 1)
        
        if aspect_ratio > 2.5 or aspect_ratio < 0.4:
            suggestions["gaussian_sigma"] = min(10.0, self.params["gaussian_sigma"] + 1.0)
        
        return suggestions
    
    def apply_suggestions(self, event):
        """Apply suggested parameters"""
        if not self.current_suggestions:
            print("⚠ No suggestions available. Click on a particle first.")
            return
        
        print("\n✅ Applying suggestions:")
        for param, value in self.current_suggestions.items():
            if param in self.params:
                old_value = self.params[param]
                self.params[param] = value
                if param in self.sliders:
                    self.sliders[param].set_val(value)
                print(f"   {param}: {old_value} → {value}")
        
        self.update_visualization()
    
    def load_new_image(self):
        """Load a new image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            new_path = Path(file_path)
            
            if not validate_path_encoding(new_path):
                messagebox.showerror(
                    "Error",
                    f"Path contains problematic characters:\n{file_path}"
                )
                return
            
            self.original_image = self._load_image(new_path)
            self.image_path = new_path
            self.pixel_size_um, self.has_metadata = self._load_pixel_size()
            
            self.update_visualization()
            
            messagebox.showinfo(
                "Success",
                f"Image loaded!\n\n{new_path.name}\n"
                f"Pixel size: {self.pixel_size_um:.6f} µm/px\n"
                f"Source: {'Metadata' if self.has_metadata else 'Fallback'}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            print(f"❌ Error loading image: {e}")
    
    def save(self, event=None) -> bool:
        """Save parameters to JSON"""
        try:
            config = {
                "bacterium": self.bacterium,
                "structure": self.structure,
                "mode": self.mode,
                "image_path": str(self.image_path),
                "pixel_size_um": self.pixel_size_um,
                "metadata_source": "metadata" if self.has_metadata else "fallback",
                "parameters": {
                    "invert_image": self.invert_image,
                    **self.params
                }
            }
            
            filename = f"segmentation_params_{self.bacterium}_{self.structure}_{self.mode}.json"
            
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"\n💾 Parameters saved to: {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving parameters: {e}")
            return False
    
    def save_and_apply(self, event=None):
        """Save parameters and update bacteria_configs.py"""
        if not self.save():
            messagebox.showerror("Error", "Failed to save parameters")
            return
        
        try:
            um2_per_px2 = self.pixel_size_um ** 2
            
            config = SegmentationConfig(
                name=f"{self.bacterium}",
                description=f"{self.structure} segmentation - Tuned {datetime.now().strftime('%Y-%m-%d')} (Pixel: {self.pixel_size_um:.6f} µm)",
                gaussian_sigma=float(self.params['gaussian_sigma']),
                min_area_um2=float(self.params['min_area']) * um2_per_px2,
                max_area_um2=float(self.params['max_area']) * um2_per_px2,
                dilate_iterations=int(self.params['dilate_iterations']),
                erode_iterations=int(self.params['erode_iterations']),
                morph_kernel_size=3,
                morph_iterations=1,
                min_circularity=0.0,
                max_circularity=1.0,
                min_aspect_ratio=0.2,
                max_aspect_ratio=10.0,
                min_mean_intensity=0,
                max_mean_intensity=255,
                max_edge_gradient=200,
                min_solidity=0.3,
                max_fraction_of_image=0.25,
                fluor_min_area_um2=3.0,
                fluor_match_min_intersection_px=5.0,
            )
            
            success = update_bacteria_config(
                bacterium=self.bacterium,
                config=config,
                backup=True
            )
            
            if success:
                print(f"\n✅ Configuration saved: {self.bacterium}")
                messagebox.showinfo(
                    "Success", 
                    f"Parameters saved and applied!\n\n"
                    f"Configuration updated in bacteria_configs.py\n"
                    f"Backup created: bacteria_configs.py.bak"
                )
            else:
                messagebox.showerror(
                    "Error",
                    "Failed to update bacteria_configs.py\n"
                    "Check console for details"
                )
            
        except Exception as e:
            print(f"❌ Error updating bacteria_configs.py: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to update bacteria_configs.py:\n{e}")
    
    def _create_action_buttons(self, parent: ttk.Frame):
        """Create bottom action buttons"""
        action_frame = tk.Frame(parent, bg=self.COLORS['header'], height=55)
        action_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        action_frame.pack_propagate(False)
        
        button_container = tk.Frame(action_frame, bg=self.COLORS['header'])
        button_container.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        style = ttk.Style()
        style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), padding=10)
        
        buttons = [
            ("⬅ BACK", self.back, 13, None),
            ("💾 SAVE JSON", self.save, 15, None),
            ("✅ SAVE & APPLY", self.save_and_apply, 18, self.COLORS['success']),
            ("❌ QUIT", self.quit, 13, None),
        ]
        
        for text, command, width, highlight in buttons:
            if highlight:
                frame = tk.Frame(button_container, bg=highlight, bd=3, relief=tk.RAISED)
                frame.pack(side=tk.LEFT, padx=6)
                btn = ttk.Button(frame, text=text, command=command, width=width,
                            style="Action.TButton")
                btn.pack(padx=2, pady=2)
            else:
                btn = ttk.Button(button_container, text=text, command=command,
                            width=width, style="Action.TButton")
                btn.pack(side=tk.LEFT, padx=5)
    
    def back(self):
        """Close the tuner and return to main menu"""
        if messagebox.askyesno("Confirm", "Close tuner?\n\nUnsaved changes will be lost."):
            print("🔙 Closing tuner...")
            self.root.destroy()
            
            if self.return_callback:
                try:
                    print("🔄 Returning to main menu...")
                    self.return_callback()
                except Exception as e:
                    print(f"⚠️ Error executing return callback: {e}")
            else:
                print("ℹ️ No return destination specified.")
    
    def quit(self, event=None):
        """Quit application"""
        if messagebox.askyesno("Confirm", "Quit application?"):
            print("\n❌ Exiting application")
            self.root.quit()
            self.root.destroy()


# ==================================================
# SECTION 6: Main Menu (Pathogen Config Manager)
# ==================================================

class PathogenConfigManager:
    """Main menu for managing pathogen configurations"""
    
    PATHOGENS = {
        "Proteus mirabilis": {
            "config_key": "proteus_mirabilis",
            "description": "Rod-shaped, flagellated bacterium",
            "common_in": "Catheter-associated infections"
        },
        "Klebsiella pneumoniae": {
            "config_key": "klebsiella_pneumoniae",
            "description": "Gram-negative, encapsulated bacterium",
            "common_in": "Healthcare-associated infections"
        },
        "Streptococcus mitis": {
            "config_key": "streptococcus_mitis",
            "description": "Gram-positive cocci in chains",
            "common_in": "Touch contamination"
        }
    }
    
    COLORS = {
        'bg': '#1e1e1e',
        'fg': '#ffffff',
        'accent': '#007acc',
        'button': '#2d2d2d',
        'button_hover': '#3e3e3e',
        'success': '#4ec9b0',
        'warning': '#ce9178',
        'error': '#f48771',
        'header': '#569cd6'
    }
    
    def __init__(self, root: tk.Tk):
        """Initialize the main menu"""
        self.root = root
        self.root.title("Peritoneal Dialysis Pathogen Configuration Manager")
        self.root.geometry("900x700")
        self.root.configure(bg=self.COLORS['bg'])
        self.root.resizable(True, True)
        
        self._center_window()
        self._create_ui()
        
    def _center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
    def _create_ui(self):
        """Create the main user interface"""
        main_frame = tk.Frame(self.root, bg=self.COLORS['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self._create_header(main_frame)
        self._create_pathogen_cards(main_frame)
        self._create_footer(main_frame)
        
    def _create_header(self, parent):
        """Create header section"""
        header_frame = tk.Frame(parent, bg=self.COLORS['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 30))
        
        title = tk.Label(
            header_frame,
            text="🦠 Pathogen Configuration Manager",
            font=('Segoe UI', 24, 'bold'),
            bg=self.COLORS['bg'],
            fg=self.COLORS['header']
        )
        title.pack(anchor=tk.W)
        
        subtitle = tk.Label(
            header_frame,
            text="Peritoneal Dialysis - Image Analysis Configuration",
            font=('Segoe UI', 12),
            bg=self.COLORS['bg'],
            fg=self.COLORS['fg']
        )
        subtitle.pack(anchor=tk.W, pady=(5, 0))
        
        separator = tk.Frame(header_frame, height=2, bg=self.COLORS['accent'])
        separator.pack(fill=tk.X, pady=(15, 0))
        
    def _create_pathogen_cards(self, parent):
        """Create cards for each pathogen"""
        cards_frame = tk.Frame(parent, bg=self.COLORS['bg'])
        cards_frame.pack(fill=tk.BOTH, expand=True)
        
        for pathogen_name, info in self.PATHOGENS.items():
            card = self._create_pathogen_card(cards_frame, pathogen_name, info)
            card.pack(fill=tk.X, pady=(0, 15))
            
    def _create_pathogen_card(self, parent, pathogen_name: str, info: dict):
        """Create a card for a single pathogen"""
        card = tk.Frame(
            parent,
            bg=self.COLORS['button'],
            relief=tk.RAISED,
            borderwidth=1
        )
        
        card.bind('<Enter>', lambda e: card.configure(bg=self.COLORS['button_hover']))
        card.bind('<Leave>', lambda e: card.configure(bg=self.COLORS['button']))
        
        content = tk.Frame(card, bg=self.COLORS['button'])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        left_frame = tk.Frame(content, bg=self.COLORS['button'])
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        name_label = tk.Label(
            left_frame,
            text=f"🔬 {pathogen_name}",
            font=('Segoe UI', 16, 'bold'),
            bg=self.COLORS['button'],
            fg=self.COLORS['success'],
            anchor=tk.W
        )
        name_label.pack(anchor=tk.W)
        
        desc_label = tk.Label(
            left_frame,
            text=info['description'],
            font=('Segoe UI', 10),
            bg=self.COLORS['button'],
            fg=self.COLORS['fg'],
            anchor=tk.W
        )
        desc_label.pack(anchor=tk.W, pady=(5, 0))
        
        common_label = tk.Label(
            left_frame,
            text=f"Common in: {info['common_in']}",
            font=('Segoe UI', 9, 'italic'),
            bg=self.COLORS['button'],
            fg=self.COLORS['warning'],
            anchor=tk.W
        )
        common_label.pack(anchor=tk.W, pady=(3, 0))
        
        config_label = tk.Label(
            left_frame,
            text=f"📄 Config: {info['config_key']}",
            font=('Segoe UI', 8),
            bg=self.COLORS['button'],
            fg=self.COLORS['fg'],
            anchor=tk.W
        )
        config_label.pack(anchor=tk.W, pady=(8, 0))
        
        right_frame = tk.Frame(content, bg=self.COLORS['button'])
        right_frame.pack(side=tk.RIGHT, padx=(20, 0))
        
        seg_btn = tk.Button(
            right_frame,
            text="🎨 Segmentation",
            font=('Segoe UI', 10, 'bold'),
            bg=self.COLORS['accent'],
            fg='white',
            activebackground=self.COLORS['header'],
            activeforeground='white',
            relief=tk.FLAT,
            cursor='hand2',
            padx=15,
            pady=8,
            command=lambda p=pathogen_name: self._launch_segmentation_tuner(p)
        )
        seg_btn.pack(pady=(0, 8))
        
        return card
        
    def _create_footer(self, parent):
        """Create footer with utility buttons"""
        footer_frame = tk.Frame(parent, bg=self.COLORS['bg'])
        footer_frame.pack(fill=tk.X, pady=(30, 0))
        
        separator = tk.Frame(footer_frame, height=2, bg=self.COLORS['accent'])
        separator.pack(fill=tk.X, pady=(0, 15))
        
        buttons_frame = tk.Frame(footer_frame, bg=self.COLORS['bg'])
        buttons_frame.pack()
        
        info_btn = tk.Button(
            buttons_frame,
            text="ℹ️ About",
            font=('Segoe UI', 10),
            bg=self.COLORS['button'],
            fg=self.COLORS['fg'],
            activebackground=self.COLORS['button_hover'],
            activeforeground=self.COLORS['fg'],
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=10,
            command=self._show_about
        )
        info_btn.pack(side=tk.LEFT, padx=5)
        
        exit_btn = tk.Button(
            buttons_frame,
            text="❌ Exit",
            font=('Segoe UI', 10),
            bg=self.COLORS['error'],
            fg='white',
            activebackground='#d67060',
            activeforeground='white',
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=10,
            command=self._exit_application
        )
        exit_btn.pack(side=tk.RIGHT, padx=5)
        
    def _launch_segmentation_tuner(self, pathogen_name: str):
        """Launch segmentation tuner with setup dialog"""
        print(f"\n🎨 Launching segmentation tuner for {pathogen_name}...")
        
        # Create setup dialog
        setup_dialog = tk.Toplevel(self.root)
        setup_dialog.title(f"Tuner Setup - {pathogen_name}")
        setup_dialog.geometry("500x350")
        setup_dialog.resizable(False, False)
        setup_dialog.configure(bg='white')
        setup_dialog.transient(self.root)
        setup_dialog.grab_set()
        
        # Variables
        image_path_var = tk.StringVar()
        structure_var = tk.StringVar(value="bacteria")
        mode_var = tk.StringVar(value="DARK")
        
        # Main frame
        main_frame = ttk.Frame(setup_dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(
            main_frame,
            text=f"🔬 {pathogen_name} Tuner Setup",
            font=("Segoe UI", 12, "bold")
        ).pack(pady=(0, 20))
        
        # Image selection
        ttk.Label(
            main_frame,
            text="1. Select Image",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 5))
        
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Entry(image_frame, textvariable=image_path_var, width=40).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        def browse_image():
            filename = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                    ("All files", "*.*")
                ]
            )
            if filename:
                image_path_var.set(filename)
        
        ttk.Button(image_frame, text="Browse...", command=browse_image).pack(side=tk.LEFT)
        
        # Structure selection
        ttk.Label(
            main_frame,
            text="2. Select Structure",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 5))
        
        structure_frame = ttk.Frame(main_frame)
        structure_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Radiobutton(
            structure_frame,
            text="Bacteria",
            variable=structure_var,
            value="bacteria"
        ).pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Radiobutton(
            structure_frame,
            text="Inclusions",
            variable=structure_var,
            value="inclusions"
        ).pack(side=tk.LEFT)
        
        # Mode selection
        ttk.Label(
            main_frame,
            text="3. Select Mode",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 5))
        
        mode_frame = ttk.Frame(main_frame)
        mode_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Radiobutton(
            mode_frame,
            text="DARK particles",
            variable=mode_var,
            value="DARK"
        ).pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Radiobutton(
            mode_frame,
            text="BRIGHT particles",
            variable=mode_var,
            value="BRIGHT"
        ).pack(side=tk.LEFT)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        def start_tuner():
            if not image_path_var.get():
                messagebox.showerror("Error", "Please select an image file", parent=setup_dialog)
                return
            
            setup_dialog.destroy()
            self.root.destroy()
            
            try:
                tuner_root = tk.Tk()
                
                def return_to_menu():
                    main()
                
                tuner = SegmentationTuner(
                    root=tuner_root,
                    image_path=image_path_var.get(),
                    bacterium=pathogen_name,
                    structure=structure_var.get(),
                    mode=mode_var.get(),
                    return_callback=return_to_menu
                )
                tuner_root.mainloop()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start tuner:\n{str(e)}")
                import traceback
                traceback.print_exc()
        
        ttk.Button(
            button_frame,
            text="❌ Cancel",
            command=setup_dialog.destroy
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame,
            text="✅ Start Tuner",
            command=start_tuner
        ).pack(side=tk.LEFT)
        
    def _show_about(self):
        """Show about dialog"""
        about_text = """
Peritoneal Dialysis Pathogen Configuration Manager
Version 2.0

This application manages image analysis configurations 
for three common peritoneal dialysis pathogens:

• Proteus mirabilis
• Klebsiella pneumoniae  
• Streptococcus mitis

Features:
🎨 Segmentation Tuner - Adjust image segmentation parameters
📊 Histogram Analysis - Visualize particle distributions
💾 Config Management - Save and apply configurations

© 2026 Pathogen Analysis Suite
        """
        messagebox.showinfo("About", about_text.strip())
        
    def _exit_application(self):
        """Exit the application"""
        if messagebox.askyesno("Confirm Exit", "Are you sure you want to exit?"):
            print("\n👋 Exiting Pathogen Configuration Manager")
            self.root.quit()
            self.root.destroy()


# ==================================================
# SECTION 7: Main Entry Point
# ==================================================

def main():
    """Main entry point"""
    root = tk.Tk()
    app = PathogenConfigManager(root)
    root.mainloop()


if __name__ == "__main__":
    main()