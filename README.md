List of Configurations/Parameters:
Behavior of Parameter Changes:

CLAHE (contrast limiting adqptive equalization) for enhancing local contrast

1. The image is dicided into small titles or blocks
2. A histogram is calculated for each title
3. If any histogram bin is above the specific cliplimit, those excess pixels are clipped and the contrast of that tile is limited
4. The clipped excess is redistributed uniformly to other bins before applying the histogram equalization
5. Histogran equalization is apply to each tile
6. The final image is obtained by interpolating the equalized tiles

Morphology, opencv for shape or form of object

- Kernel, structuring element
- Iteration, how much noise is removed or how completely holes are filled

Watershed & Filtering: Watershed Dilate (integer, default: 15, range: 1-20)

1. Segment objects of interest
2. Convert the mask into an intensity profile using the distance transform
3. Run the watershed algorithm
4. Update the original mask

Behavior on change:
Increasing: Creates smaller, more conservative foreground markers, leading to greater separation of objects. This results in more individual contours (better splits clumped bacteria) but can cause over-segmentation or fragmentation if too high.
Decreasing: Creates larger markers, reducing separation and potentially merging nearby objects into fewer, larger contours. Useful for under-segmented images but risks failing to separate touching bacteria.

Watershed & Filtering: Min Area (px²) (integer, default: 50, range: 10-500)

Description: Minimum area (in pixels) required for a contour to be considered a valid bacterium; smaller contours are filtered out after detection.
Behavior on change:
Increasing: Stricter filtering, discarding more small contours (e.g., noise or debris), resulting in fewer detected bacteria but cleaner results. May miss legitimate small bacteria.
Decreasing: More inclusive, retaining smaller contours and increasing the number of detected bacteria, but introduces more false positives like artifacts.

Fluorescence: Min Fluor/Area (float, default: 10.0, range: 0-255, step: 0.1)

Description: Minimum fluorescence intensity per unit area (total fluorescence divided by contour area) to retain a bacterium; only applies if a fluorescence image is loaded. This filters contours post-detection based on fluorescence data.
Behavior on change:
Increasing: More selective, removing contours with low fluorescence density (e.g., dim or non-fluorescent bacteria), reducing the number of displayed contours to focus on "active" ones.
Decreasing: Less filtering, including contours with weaker or no fluorescence, increasing the number of retained contours but potentially including irrelevant ones.

# Interactive Bacteria Segmentation Parameter Tuner

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

This is a graphical user interface (GUI) application built with Tkinter and OpenCV for interactively tuning parameters in bacteria segmentation from microscopy images. It processes bright-field (\_ch00.tif) and optional fluorescence (\_ch01.tif) TIFF images, allowing real-time previews of processing stages like enhancement, thresholding, morphology operations, and contour detection.

The tool is designed for researchers and biologists working with bacterial microscopy data, providing features like parameter sliders (via entries), statistics export, smart label positioning, and dark mode support.

## Features

- **Folder Loading & Navigation**: Select folders containing \_ch00.tif images (with optional subfolder picker). Navigate between images using arrow keys or buttons.
- **Parameter Tuning**: Interactive controls for:
  - Thresholding (Otsu or manual)
  - CLAHE enhancement (enchancing local contrast)
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

pip install --upgrade pip

Create a `requirements.txt` file with:

```
opencv-python
pillow
numpy
scipy
tk  # Usually bundled with Python

pandas
openpyxl
```

Install:

```
pip install -r requirements.txt
```

### Clone the Repository

```
git clone https://github.com/yourusername/bacteria-segmentation-tuner.git
cd bacteria-segmentation-tuner
```

## Usage

1. Run the script:

   ```
   python dev.py
   ```

2. **Load Folder**: Click "Load Folder" (or L key) to select a directory with \_ch00.tif images. If subfolders are present, pick one.
3. **Tune Parameters**: Adjust values in the left panel; previews update automatically.
4. **Navigate Images**: Use ←/→ arrows or Previous/Next buttons.
5. **Probe & Measure**: Left-click on images for details; right-click to clear.
6. **Statistics**: View in tab 8; sort columns, select rows to highlight, export to CSV.
7. **Exit**: Click "Exit" or Esc (with confirmation).

**Example Workflow**:

- Load a folder from `./source/` (default initial dir).
- Enable CLAHE and adjust clip/tile for better contrast.
- Set manual threshold ~110 for initial segmentation.
- Refine morphology to clean noise.
- Overlay fluorescence and filter by min Fluor/Area.

## Dependencies

- **OpenCV**: For image processing and segmentation.
- **Pillow (PIL)**: For drawing labels and text on images.
- **NumPy & SciPy**: For array operations and distance transforms.
- **Tkinter**: Built-in Python GUI library.

## Contributing

Pull requests welcome! For major changes, open an issue first.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with inspiration from bio-imaging tools like ImageJ/Fiji.
- Fonts and paths adapted for cross-platform use.

Last updated: November 12, 2025.
