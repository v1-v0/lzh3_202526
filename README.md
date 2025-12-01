# Interactive Bacteria Segmentation Parameter Tuner

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

This is a graphical user interface (GUI) application built with Tkinter and OpenCV for interactively tuning parameters in bacteria segmentation from microscopy images. It processes bright-field (\_ch00.tif) and optional fluorescence (\_ch01.tif) TIFF images, allowing real-time previews of processing stages like enhancement, thresholding, morphology operations, and contour detection.

The tool is designed for researchers and clinicians working with bacterial microscopy data from **Peritoneal Dialysis (PD) clinical samples**, particularly for analyzing microgel cultures and bacterial contamination in dialysate. It provides features like parameter sliders (via entries), statistics export, smart label positioning, and dark mode support to facilitate the quantitative assessment of bacterial presence and activity in PD-related specimens.

## Clinical Application: Peritoneal Dialysis

This tool was developed specifically for analyzing microscopy images from **Peritoneal Dialysis (PD)** clinical samples. PD is a renal replacement therapy where the peritoneal membrane serves as a natural dialysis filter. Monitoring bacterial contamination and biofilm formation in dialysate samples is critical for:

- **Detecting peritonitis**: Early identification of bacterial infections in PD patients
- **Microgel analysis**: Characterizing positive microgel cultures from dialysate
- **Treatment monitoring**: Assessing bacterial load and viability through fluorescence imaging
- **Quality control**: Evaluating sample preparation protocols (e.g., "water C protocol")

The dual-channel imaging approach (brightfield + fluorescence) enables differentiation between total bacterial count and metabolically active bacteria, providing valuable clinical insights for PD patient management.

## Features

- **Folder Loading & Navigation**: Select folders containing \_ch00.tif images (with optional subfolder picker). Navigate between images using arrow keys or buttons.
- **Parameter Tuning**: Interactive controls for:
  - Thresholding (Otsu or manual)
  - CLAHE enhancement (enhancing local contrast)
  - Morphology operations (opening/closing kernels and iterations)
  - Watershed segmentation
  - Fluorescence adjustments (brightness, gamma)
  - Label display (font size, arrow length, offset)
- **Preview Tabs**: Numbered tabs (1-8) for viewing original BF, fluorescence, enhanced, thresholded, morphology, contours, overlay, and statistics. Switch tabs with comma (previous) / period (next).
- **Measurement on Click**: Click on images to probe pixel values, check if inside contours, and measure areas. Ctrl+Click for auto-tuning parameters based on selected bacterium.
- **Statistics Table**: Displays bacteria stats (area, fluorescence mean/total/per-area) with sorting, row selection for highlighting, and CSV export.
- **Smart Labels**: Automatically positions numbered labels on contours to avoid overlaps, with arrows and semi-transparent backgrounds.
- **Dark Mode**: Toggle between light/dark themes (shortcut: D).
- **Keyboard Shortcuts**: L (load folder), Esc (exit), arrows (image nav), etc.
- **Optimizations**: Debounced updates for smooth performance, vertical scrolling for params panel.
- **Cross-Platform**: Tested on macOS; compatible with Windows/Linux (with font adjustments).

## Installation

### Prerequisites

- Python 3.8 or higher
- Dependencies: Install via pip (requirements.txt provided below)

```bash
git clone https://github.com/v1-v0/pd.git
cd pd
```

### Create virtual environment

```bash
python -m venv .venv
python -m pip install --upgrade pip
```

### Activation

```bash
.\.venv\Scripts\Activate.bat # Windows Command Prompt
.\.venv\Scripts\Activate.ps1 # Windows PowerShell
source .venv/Scripts/activate # Linux
```

### Libraries

```bash
pip install opencv-python pillow numpy scipy
pip install tk  # Usually bundled with Python
pip install pandas openpyxl

```

## Usage

### Getting Started

1. Run the script

```bash
python app.py
```

2. **Load Folder**: Click "Load Folder" (or L key) to select a directory with \_ch00.tif images. If subfolders are present, pick one.

3. **Tune Parameters**: Adjust values in the left panel; previews update automatically.

4. **Navigate Images**: Use ←/→ arrows or Previous/Next buttons.

5. **Probe & Measure**: Left-click on images for details; right-click to clear.

6. **Statistics**: View in tab 8; sort columns, select rows to highlight, export to CSV.

7. **Exit**: Click "Exit" or Esc (with confirmation).

### Example Workflow for PD Samples

- Load a folder from `./source/` (default initial dir) containing PD clinical sample images.
- Enable CLAHE and adjust clip/tile for better contrast in microgel samples.
- Set manual threshold ~110 for initial bacterial segmentation.
- Refine morphology to clean noise from debris or artifacts.
- Overlay fluorescence to identify viable/active bacteria.
- Filter by min Fluor/Area to distinguish metabolically active bacteria from background.
- Export statistics for clinical reporting and patient assessment.

## Parameter Reference

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

For enhancing local contrast:

1. The image is divided into small tiles or blocks
2. A histogram is calculated for each tile
3. If any histogram bin is above the specific cliplimit, those excess pixels are clipped and the contrast of that tile is limited
4. The clipped excess is redistributed uniformly to other bins before applying the histogram equalization
5. Histogram equalization is applied to each tile
6. The final image is obtained by interpolating the equalized tiles

### Morphology Operations

OpenCV operations for shape or form of objects:

- Kernel: Structuring element size
- Iteration: Controls how much noise is removed or how completely holes are filled

### Watershed & Filtering Parameters

#### Watershed Dilate (integer, default: 15, range: 1-20)

Process:

1. Segment objects of interest
2. Convert the mask into an intensity profile using the distance transform
3. Run the watershed algorithm
4. Update the original mask

#### Behavior on change:

- Increasing: Creates smaller, more conservative foreground markers, leading to greater separation of objects. This results in more individual contours (better splits clumped bacteria) but can cause over-segmentation or fragmentation if too high.
- Decreasing: Creates larger markers, reducing separation and potentially merging nearby objects into fewer, larger contours. Useful for under-segmented images but risks failing to separate touching bacteria.

#### Min Area (px²) (integer, default: 50, range: 10-500)

**Description**: Minimum area (in pixels) required for a contour to be considered a valid bacterium; smaller contours are filtered out after detection.

#### Behavior on change:

**Increasing**: Stricter filtering, discarding more small contours (e.g., noise or debris), resulting in fewer detected bacteria but cleaner results. May miss legitimate small bacteria.
**Decreasing**: More inclusive, retaining smaller contours and increasing the number of detected bacteria, but introduces more false positives like artifacts.

### Fluorescence Parameters

#### Min Fluor/Area (float, default: 10.0, range: 0-255, step: 0.1)

**Description**: Minimum fluorescence intensity per unit area (total fluorescence divided by contour area) to retain a bacterium; only applies if a fluorescence image is loaded. This filters contours post-detection based on fluorescence data.

#### Behavior on change:

- **Increasing**: More selective, removing contours with low fluorescence density (e.g., dim or non-fluorescent bacteria), reducing the number of displayed contours to focus on "active" ones.
- **Decreasing**: Less filtering, including contours with weaker or no fluorescence, increasing the number of retained contours but potentially including irrelevant ones.

## Dependencies

- **OpenCV**: For image processing and segmentation
- **Pillow (PIL)**: For drawing labels and text on images
- **NumPy & SciPy**: For array operations and distance transforms
- **Tkinter**: Built-in Python GUI library
- **Pandas & openpyxl**: For statistics export to CSV/Excel

## Keyboard Shortcuts

- **L**: Load folder
- **Esc**: Exit application (with confirmation)
- **←/→**: Navigate between images
- **,/.**: Switch between preview tabs (comma = previous, period = next)
- **D**: Toggle dark mode
- **Left-click**: Probe pixel values and measure
- **Right-click**: Clear measurements
- **Ctrl+Click**: Auto-tune parameters based on selected bacterium

## Acknowledgments

Developed for clinical microbiology applications in Peritoneal Dialysis monitoring
