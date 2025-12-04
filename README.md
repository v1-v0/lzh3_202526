# Interactive Segment Peritoneal Dialysis Bacteria Tool

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-green.svg)](#cross-platform)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Clinical Application](#clinical-application-peritoneal-dialysis)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Metadata Integration](#metadata-integration)
- [Parameter Reference](#parameter-reference)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This is an interactive **Tkinter + OpenCV GUI application** for bacteria segmentation and analysis from dual-channel microscopy images. It processes:

- **Brightfield channel** (`_ch00.tif`): Total bacterial count
- **Fluorescence channel** (`_ch01.tif`, optional): Metabolically active bacteria

**Key capability**: Real-time parameter tuning with 8 preview tabs showing segmentation pipeline stages (enhancement → thresholding → morphology → watershed → contour detection).

Designed specifically for **Peritoneal Dialysis (PD) clinical research**, enabling quantitative assessment of bacterial presence and viability in dialysate samples through interactive image analysis.

---

## Clinical Application: Peritoneal Dialysis

**Peritoneal Dialysis (PD)** is a renal replacement therapy where the peritoneal membrane functions as a natural dialysis filter. Bacterial contamination (peritonitis) is a serious complication requiring rapid microbial analysis.

### Why Dual-Channel Analysis?

| Channel          | Purpose                              | Clinical Insight                    |
| ---------------- | ------------------------------------ | ----------------------------------- |
| **Brightfield**  | Total bacterial count                | Overall bacterial load in dialysate |
| **Fluorescence** | Viable/metabolically active bacteria | Current infection severity          |

The fluorescence channel specifically identifies **live, actively metabolizing bacteria**, providing critical information for:

- Early infection detection
- Treatment efficacy monitoring
- Bacterial viability assessment
- Clinical decision-making

---

## Features

### 📂 **File & Navigation**

- **Folder Loading**: Select folders containing `_ch00.tif` images with automatic subfolder detection
- **Image Navigation**: Arrow keys (←/→) or Previous/Next buttons
- **Multi-image Support**: Process entire image stacks sequentially

### 🎛️ **Parameter Tuning** (with real-time visual feedback)

- **Thresholding**: Otsu automatic or manual (0-255)
- **CLAHE Enhancement**: Clip limit (1-10) and tile size (4-32)
- **Morphology Operations**: Opening/closing kernels (1-15, odd) and iterations (1-5)
- **Watershed Segmentation**: Dilation factor (1-20) for object separation
- **Fluorescence Adjustments**: Brightness (0.5-5x) and gamma correction (0.2-2)
- **Label Customization**: Font size, arrow length, offset positioning
- **Progress Indicators**: Visual progress bars for each parameter

### 📺 **8 Preview Tabs**

| Tab | Content              | Purpose                             |
| --- | -------------------- | ----------------------------------- |
| 1   | Original brightfield | Verify image loading                |
| 2   | Fluorescence channel | Check viable bacteria               |
| 3   | Enhanced image       | Evaluate CLAHE effect               |
| 4   | Binary threshold     | Verify segmentation threshold       |
| 5   | Morphology result    | Check opening/closing effectiveness |
| 6   | Detected contours    | Raw segmentation output             |
| 7   | Overlay with labels  | Final annotated image               |
| 8   | Statistics table     | Quantitative metrics & histograms   |

**Tab navigation**: Comma (,) = previous | Period (.) = next

### 📊 **Interactive Measurement & Analysis**

- **Left-click**: Probe pixel values, display area statistics, check contour membership
- **Right-click**: Clear measurements
- **Ctrl+Click**: Auto-tune parameters based on selected bacterium
- **Row selection**: Highlight bacteria on overlay image by selecting statistics rows

### 📈 **Advanced Statistics**

- **Real-time metrics**: Area, fluorescence (mean/total/per-area), viability indicators
- **Histogram visualization** (matplotlib-enabled): Population distributions
- **CSV export**: Complete statistics table for external analysis
- **Physical unit conversion** (metadata-aware): Display both pixels and µm

### 🏷️ **Smart Label Positioning**

- Automatic label placement avoiding overlaps
- Arrow indicators with customizable length & offset
- Semi-transparent backgrounds for clarity
- Numbered labels (1, 2, 3...) for bacteria identification

### 🎨 **Dark Mode**

- Toggle light/dark themes (shortcut: **D**)
- Reduces eye strain during extended analysis sessions

### ⚡ **Performance Optimizations**

- Debounced preview updates for smooth responsiveness
- Vertical scrolling parameter panel for small screens
- Efficient contour processing with configurable filters

### 🌍 **Cross-Platform Support**

- **macOS**: Fully tested & optimized
- **Windows 10/11**: Full compatibility (font adjustments may apply)
- **Linux/Ubuntu**: Compatible with standard font adjustments

---

## Installation

### Prerequisites

- **Python 3.12** or higher
- **Git** (for cloning repository)
- **~500 MB** disk space for dependencies

### Step 1: Clone Repository

```bash
git clone https://github.com/v1-v0/particle-scout.git
cd pd
```

### Step 2: Create Virtual Environment

```bash
# macOS/Linux
python -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Core dependencies
pip install opencv-python pillow numpy scipy pandas openpyxl

# Optional: Histogram visualization (recommended)
pip install matplotlib

# Tkinter usually bundled with Python
# If missing: sudo apt-get install python3-tk (Linux)
```

### Step 4: Verify Installation

```bash
python -c "import cv2, tkinter, pandas; print('✓ All dependencies installed')"
```

---

## Quick Start

```bash
# Run the application
python app.py

# Load sample images
# 1. Press 'L' or click "Load Folder"
# 2. Navigate to ./source/ or your image directory
# 3. Adjust parameters on left panel
# 4. View results in 8 preview tabs
# 5. Export statistics: Tab 8 → "Export CSV"
```

---

## Usage Guide

### Getting Started (Step-by-Step)

#### Step 1: Load Image Folder

```
Action: Press 'L' or click "Load Folder" button
Input: Select folder containing _ch00.tif files
Optional: Automatically detects subfolders
Result: First image displays in all preview tabs
```

#### Step 2: Explore Preview Tabs

```
Tabs 1-7: View different processing stages
Tab 8: Statistics table & histograms

Navigation: Comma (,) = previous tab | Period (.) = next tab
```

#### Step 3: Tune Parameters

```
Left panel: Adjust threshold, CLAHE, morphology, fluorescence
Real-time updates: Previews refresh automatically
Progress bars: Visual feedback for each parameter
```

#### Step 4: Measure & Analyze

```
Left-click image: Probe pixel values & measure areas
Ctrl+Click: Auto-tune based on selected bacterium
Select table row: Highlight bacterium on overlay
```

#### Step 5: Export Results

```
Tab 8 → "Export CSV": Save statistics with metadata
Histograms: Right-click to export as PNG
```

#### Step 6: Navigate Images

```
Arrows (←/→): Previous/next image
Buttons: Previous/Next in GUI
Keyboard: Fast batch processing
```

### Example Workflow: PD Sample Analysis

**Scenario**: Analyzing peritoneal fluid sample from dialysis patient

```bash
Step 1: Load folder
  └─ Directory: ./samples/pd_patient_001/
  └─ Contains: image_001_ch00.tif, image_001_ch01.tif

Step 2: Enhance contrast (Tab 3)
  └─ Enable CLAHE: Yes
  └─ Clip limit: 3.0 (balance enhancement vs noise)
  └─ Tile size: 16 (typical for bacteria imaging)

Step 3: Set threshold (Tab 4)
  └─ Use Otsu: Initially
  └─ Manual fine-tune: ~110 for this patient's bacterial morphology

Step 4: Refine morphology (Tab 5)
  └─ Open kernel: 3 (remove small noise)
  └─ Open iterations: 1
  └─ Close kernel: 5 (bridge gaps in same bacteria)
  └─ Close iterations: 2

Step 5: Separate touching bacteria (Tab 6)
  └─ Watershed dilate: 12 (balance separation vs over-segmentation)

Step 6: Filter artifacts (Tab 8)
  └─ Min area: 100 px² (remove dust, retain bacteria)
  └─ Min fluor/area: 0.5 (focus on viable bacteria)

Step 7: Analyze results
  └─ Sort by area: Identify bacterial population sizes
  └─ View histograms: Distribution of viable vs total
  └─ Click bacterium: Check fluorescence intensity

Step 8: Export for clinical report
  └─ CSV: Statistics with µm measurements (if metadata)
  └─ PNG: Overlay image with labels
  └─ Histogram: Distribution plots for documentation
```

---

## Metadata Integration

### Purpose

Calibrate pixel measurements to **physical units (µm, mm)** using microscopy metadata.

### Setup

#### File Structure

```
project_folder/
├── image_001_ch00.tif
├── image_001_ch01.tif
└── metadata.json          ← Place here
```

#### Example `metadata.json`

```json
{
  "pixel_size_um": 0.5,
  "description": "Zeiss LSM 880 - 40x objective with 2x zoom",
  "magnification": "40x",
  "camera": "Hamamatsu C13440",
  "wavelength_ex_nm": 488,
  "wavelength_em_nm": 515,
  "notes": "Clinical PD sample imaging protocol"
}
```

#### Metadata Fields

| Field              | Type   | Required    | Purpose                                            |
| ------------------ | ------ | ----------- | -------------------------------------------------- |
| `pixel_size_um`    | float  | ✅ Yes      | Pixel-to-micrometer conversion factor              |
| `description`      | string | ❌ Optional | Imaging setup description (e.g., microscope model) |
| `magnification`    | string | ❌ Optional | Objective magnification (e.g., "40x")              |
| `camera`           | string | ❌ Optional | Camera model name                                  |
| `wavelength_ex_nm` | int    | ❌ Optional | Excitation wavelength for fluorescence             |
| `wavelength_em_nm` | int    | ❌ Optional | Emission wavelength for fluorescence               |
| `notes`            | string | ❌ Optional | Any additional imaging notes                       |

### Behavior

| Condition                                 | Measurement Display                              |
| ----------------------------------------- | ------------------------------------------------ |
| **metadata.json found & valid**           | Both pixels and µm (e.g., "245 px² ≈ 61.25 µm²") |
| **metadata.json missing**                 | Pixels only (e.g., "245 px²")                    |
| **Invalid JSON or missing pixel_size_um** | Fallback to pixels, no error thrown              |

### Output Examples

**With metadata loaded:**

```
Bacterium #5:  Area = 245 px² (≈61.25 µm²)  |  Fluorescence = 1200.5  |  Per-area = 4.9 (per µm²)
Scale bar:     50 µm (≈100 px)
```

**Without metadata:**

```
Bacterium #5:  Area = 245 px²  |  Fluorescence = 1200.5  |  Per-area = 4.9
Scale bar:     100 px
```

### Finding Your Microscope's Pixel Size

For **Zeiss**, **Olympus**, **Nikon**, **Leica** systems:

1. Check microscope software metadata in image EXIF
2. Ask your facility's imaging core
3. Calculate: `objective_magnification / camera_pixel_size_um`
   - Example: (40 × 0.00625 µm) / 1 = 0.25 µm/pixel

---

## Parameter Reference

### Threshold Parameters

#### Use Otsu (Boolean)

- **Enabled**: Automatically calculates threshold via Otsu's method
- **Disabled**: Use manual threshold slider (0-255)
- **Recommendation**: Start with Otsu, fine-tune manually if needed

#### Manual Threshold (0-255)

- **Typical range for bacteria**: 100-120
- **Noise-heavy images**: 120-150
- **High-contrast samples**: 80-100

---

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Purpose**: Enhance local contrast to improve bacteria visibility against background

**Process**:

1. Image divided into small tiles (grid)
2. Histogram calculated per tile
3. Excess pixels clipped at `clip_limit` to prevent over-amplification
4. Clipped excess redistributed uniformly
5. Histogram equalization applied to each tile
6. Interpolation blends tiles smoothly

**Parameters**:

- **Clip Limit** (1-10, default 3.0): Higher = stronger enhancement but more noise
  - _1-2_: Subtle, preserves noise
  - _3-5_: Balanced (recommended for most cases)
  - _6-10_: Aggressive, risk of artifact amplification
- **Tile Size** (4-32, default 16): Grid size for local processing
  - _4-8_: Fine enhancement, longer processing
  - _12-16_: Balanced (recommended)
  - _20-32_: Coarse enhancement, faster but less precise

---

### Morphology Operations

**Purpose**: Remove noise and bridge gaps using structuring elements (kernels)

#### Opening (Remove small objects)

- **Kernel size** (1-15, odd values, default 3): Size of structuring element
  - _1_: No effect
  - _3_: Remove small noise (< 3 px)
  - _5-7_: Typical bacteria noise removal
  - _9+_: May merge small bacteria
- **Iterations** (1-5, default 1): Apply operation N times
  - _1_: Gentle smoothing
  - _2-3_: Standard noise removal
  - _4-5_: Aggressive, may change object size

#### Closing (Fill holes, bridge gaps)

- **Kernel size** (1-15, odd values, default 3): Size of structuring element
- **Iterations** (1-5, default 1): Number of applications
- **Use case**: Fix broken bacteria outlines from thresholding

---

### Watershed & Filtering Parameters

#### Watershed Dilate (1-20, default 15)

**Purpose**: Separate touching/overlapping bacteria using distance transform

**Process**:

1. Distance transform: each pixel = distance to nearest background
2. Morphological dilation: expand markers
3. Watershed algorithm: separate objects at marker boundaries
4. Update original mask

**Behavior on change**:
| Value | Effect | Result |
|-------|--------|--------|
| ↑ Increasing | Smaller, conservative markers | More separation, risk of over-segmentation |
| ↓ Decreasing | Larger, merged markers | Fewer objects, risk of missing isolated bacteria |
| 1-5 | Very conservative | Heavy over-segmentation |
| 10-15 | Balanced | Recommended range |
| 18-20 | Aggressive merging | Combine nearby bacteria |

**Recommendation**: Start at 15, adjust ±3 based on preview

---

#### Min Area (10-500 px², default 100)

**Purpose**: Filter out noise, debris, and artifacts post-segmentation

**Behavior**:
| Action | Effect | Use Case |
|--------|--------|----------|
| ↑ Increase to 200 | Remove more small objects | Noisy images, stringent analysis |
| ↓ Decrease to 50 | Keep more small objects | Sparse samples, sensitive detection |
| Default 100 | Balanced filtering | Most clinical samples |

**Clinical note**: Bacterial size range typically 100-500 µm² (10-50 px² at standard magnification)

---

### Fluorescence Parameters

#### Brightness Multiplier (0.5-5, default 1.0)

- **< 1.0**: Darken fluorescence signal
- **= 1.0**: No change
- **> 1.0**: Amplify weak signals

#### Gamma Correction (0.2-2, default 1.0)

- **< 1.0**: Brighten (enhance shadows)
- **= 1.0**: Linear (no correction)
- **> 1.0**: Darken (enhance highlights)

#### Min Fluor/Area (0-255, default 0.1)

**Purpose**: Filter bacteria by viability (metabolic activity proxy)

**Formula**: `fluorescence_total / area` = fluorescence intensity per unit area

**Behavior**:
| Value | Effect | Result |
|-------|--------|--------|
| 0 | No filtering | Display all bacteria regardless of fluorescence |
| 0.5-1.0 | Light filtering | Remove dimmest bacteria (non-viable) |
| 2-5 | Moderate filtering | Retain active bacteria |
| 10+ | Strict filtering | Only brightest, most metabolically active |

**Clinical interpretation**:

- **High fluorescence per area** = Live, metabolically active bacteria
- **Low fluorescence per area** = Dead or dormant bacteria
- **No fluorescence** = Non-viable cells

---

## Advanced Features

### Histogram Visualization (requires matplotlib)

**Available when**: `matplotlib` installed and fluorescence channel present

#### Histograms Provided

1. **Area Distribution**

   - X-axis: Bacterium size (px²)
   - Y-axis: Frequency (count)
   - Use: Identify population subsets (bimodal = mixed viable/non-viable?)

2. **Total Fluorescence**

   - X-axis: Fluorescence intensity
   - Y-axis: Frequency
   - Use: Overall viability distribution

3. **Fluorescence per Area**
   - X-axis: Intensity/area (normalized viability)
   - Y-axis: Frequency
   - Use: Identify metabolic activity threshold

#### Interactive Features

- **Legend toggle**: Click entries to show/hide histogram
- **Zoom & pan**: Mouse wheel zoom, drag to pan
- **Export**: Right-click histogram → "Save as PNG"

#### Disabling Histograms

For headless environments or minimal GUI:

```bash
# Install without matplotlib
pip install opencv-python pillow numpy scipy pandas openpyxl --no-deps
```

The tool will gracefully disable histograms and use text-only statistics display.

---

## Keyboard Shortcuts

| Shortcut        | Action                                       |
| --------------- | -------------------------------------------- |
| **L**           | Load folder (file browser dialog)            |
| **Esc**         | Exit application (with confirmation)         |
| **←/→**         | Navigate between images                      |
| **,**           | Previous preview tab                         |
| **.**           | Next preview tab                             |
| **D**           | Toggle dark mode                             |
| **Left-click**  | Probe pixel values & measure                 |
| **Right-click** | Clear measurements                           |
| **Ctrl+Click**  | Auto-tune parameters from selected bacterium |

---

## Troubleshooting

### Installation Issues

#### "ModuleNotFoundError: No module named 'cv2'"

```bash
# Fix: Install OpenCV
pip install opencv-python
```

#### "No module named 'tkinter'"

```bash
# Linux
sudo apt-get install python3-tk

# macOS (usually bundled)
# Windows (usually bundled)
```

#### "matplotlib not installed - histograms will be disabled"

```bash
# Optional but recommended for statistical visualization
pip install matplotlib
```

### Runtime Issues

#### Metadata not detected (pixels only, no µm)

**Check**:

1. ✓ `metadata.json` in same folder as `_ch00.tif`?
2. ✓ Valid JSON syntax (check for typos)?
3. ✓ Contains `pixel_size_um` field?
4. ✓ Value is numeric (e.g., `0.5`, not `"0.5"`)?

**Example valid metadata.json**:

```json
{
  "pixel_size_um": 0.5
}
```

**Fix**: Create or fix `metadata.json`, reload images (L key)

#### Slow performance with large images

**Solutions**:

1. Reduce preview resolution (not yet configurable - edit `dev.py` line ~XXX)
2. Process image subset instead of full stack
3. Increase system RAM
4. Disable matplotlib histograms if not needed

#### Labels overlapping or misaligned

**Adjust**:

- **Label Font Size**: Smaller to fit more labels
- **Arrow Length**: Shorter to reduce visual clutter
- **Label Offset**: Increase to space labels further from arrows

#### Bacteria not segmenting correctly

**Troubleshooting steps**:

| Issue                   | Likely Cause         | Solution                                  |
| ----------------------- | -------------------- | ----------------------------------------- |
| Missing bacteria        | Threshold too high   | Lower manual threshold 20-30 points       |
| Too much noise          | Threshold too low    | Raise threshold, increase Min Area filter |
| Bacteria merge together | Watershed too weak   | Increase Watershed Dilate (10 → 15)       |
| Bacteria split apart    | Watershed too strong | Decrease Watershed Dilate (15 → 10)       |
| Dark/noisy background   | Poor contrast        | Enable/increase CLAHE (Clip 1-5)          |

### Platform-Specific Issues

#### macOS: Font rendering issues

- Default fonts work fine
- System fonts usually detected automatically

#### Windows: File dialog hangs

- Restart application
- Try alternative file path input

#### Linux: Display server errors

- Verify X11 forwarding if remote
- Install system dependencies: `sudo apt-get install python3-tk`

---

## Dependencies

| Package          | Version | Purpose                        | Required    | Notes                                      |
| ---------------- | ------- | ------------------------------ | ----------- | ------------------------------------------ |
| **OpenCV**       | 4.5+    | Image processing, segmentation | ✅ Yes      | Install: `pip install opencv-python`       |
| **Pillow (PIL)** | 9.0+    | Draw labels, text rendering    | ✅ Yes      | Install: `pip install pillow`              |
| **NumPy**        | 1.22+   | Array operations               | ✅ Yes      | Install: `pip install numpy`               |
| **SciPy**        | 1.8+    | Distance transform, filters    | ✅ Yes      | Install: `pip install scipy`               |
| **Pandas**       | 1.3+    | Statistics DataFrame           | ✅ Yes      | Install: `pip install pandas`              |
| **openpyxl**     | 3.6+    | Excel export                   | ✅ Yes      | Install: `pip install openpyxl`            |
| **Tkinter**      | 8.6+    | GUI framework                  | ✅ Yes      | Bundled with Python (apt install on Linux) |
| **Matplotlib**   | 3.3+    | Histograms                     | ❌ Optional | Install: `pip install matplotlib`          |

### Verify Installation

```bash
python -c "
import cv2, tkinter, numpy, scipy, pandas, openpyxl, PIL
print('✓ Core dependencies OK')
try:
    import matplotlib
    print('✓ Matplotlib installed (histograms enabled)')
except:
    print('⚠ Matplotlib not found (histograms disabled)')
"
```

---

## Contributing

We welcome contributions! Please follow these guidelines:

### Code Standards

- Follow **PEP 8** style guide
- Use descriptive variable names
- Add docstrings to new functions
- Comment complex logic

### Testing

- Test on **macOS**, **Windows**, **Linux** if possible
- Verify with sample PD images
- Test with/without metadata.json

### Documentation

- Update README for new features
- Add inline comments for complex algorithms
- Include usage examples

### Submitting Changes

1. Fork repository
2. Create feature branch (`git checkout -b feature/new-analysis`)
3. Commit with clear messages
4. Push and create Pull Request
5. Describe changes and testing

---

## License

[Specify license type, e.g., MIT, Apache 2.0, GPL-3.0]

---

## Acknowledgments

Developed for clinical microbiology applications in **Peritoneal Dialysis monitoring** at The Chinese University of Hong Kong (CUHK).

### Credits

- **Team**: Biomedical Engineering Department, CUHK
- **Clinical guidance**: PD clinic staff and nephrologists
- **Technologies**: OpenCV, Tkinter, NumPy, SciPy community

---

## Changelog

### v1.1.0 (December 2025)

- ✨ **Metadata Integration**: Automatic physical unit calibration (µm, mm)
- ✨ **Histogram Visualization**: Distribution analysis in statistics tab (requires matplotlib)
- ✨ **Enhanced UI**: Progress bars for parameter feedback
- ✨ **Smart Tooltips**: Hover help for all parameters
- 🐛 **Fixed**: Default parameter values (Min Area: 50→100, Min Fluor/Area: 10.0→0.1)
- 📝 **Documentation**: Comprehensive metadata and histogram guides
- 🎨 **UI Refinement**: Better spacing and organization

### v1.0.0 (November 2025)

- 🎉 Initial release
- Core bacteria segmentation pipeline
- Real-time parameter tuning with 8 preview tabs
- Statistics export (CSV)
- Smart label positioning
- Dark mode support
- Cross-platform compatibility

---

**For questions or issues, please open a GitHub issue or contact the development team.**
