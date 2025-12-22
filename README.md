# Microscopy Image Analysis Pipeline

**Automated analysis of brightfield and fluorescence microscopy images with particle segmentation, fluorescence quantification, and statistical comparison.**

---

## 📋 Overview

This pipeline processes paired brightfield and fluorescence microscopy images to:

1. **Segment particles** from brightfield images using adaptive thresholding
2. **Align fluorescence channels** using phase correlation registration
3. **Quantify fluorescence intensity** within segmented particles
4. **Generate comprehensive Excel reports** with quality metrics
5. **Create statistical comparison plots** between experimental groups

---

## 🔧 Requirements

### System Requirements

- Python 3.8+
- ~4GB RAM (for typical image sets)
- ~500MB disk space per group (for outputs)

### Dependencies

Install all required packages:

```bash
pip install opencv-python numpy pandas scipy matplotlib seaborn tqdm openpyxl scikit-image
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

**Required libraries:**

- `opencv-python` (cv2) - Image processing
- `numpy` - Numerical operations
- `pandas` - Data handling
- `scipy` - Statistical analysis
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `tqdm` - Progress bars
- `openpyxl` - Excel file generation
- `scikit-image` - Image registration

---

## 📁 Required Folder Structure

```
project_root/
│
├── app.py                          # Main analysis script
│
└── source/                         # INPUT FOLDER (required)
    │
    ├── Control group/              # Control samples (required)
    │   ├── Control 001_ch00.tif   # Brightfield images
    │   ├── Control 001_ch01.tif   # Fluorescence images
    │   ├── Control 002_ch00.tif
    │   ├── Control 002_ch01.tif
    │   └── MetaData/               # Metadata folder (required)
    │       ├── Control 001_Properties.xml
    │       ├── Control 001.xml
    │       ├── Control 002_Properties.xml
    │       └── Control 002.xml
    │
    ├── 10/                         # Experimental group 10
    │   ├── 10 001_ch00.tif
    │   ├── 10 001_ch01.tif
    │   ├── 10 002_ch00.tif
    │   ├── 10 002_ch01.tif
    │   └── MetaData/
    │       ├── 10 001_Properties.xml
    │       ├── 10 001.xml
    │       └── ...
    │
    ├── 11/                         # Experimental group 11
    ├── 12/                         # Experimental group 12
    └── ...                         # Additional numeric groups
```

### 📌 Folder Naming Rules

1. **Control group:** Must be named `"Control group"` (exact match)
2. **Experimental groups:** Must be **numeric folders only** (e.g., `10`, `11`, `12`)
3. **MetaData folder:** Must exist in each group folder

### 📌 Image Naming Convention

- **Brightfield:** `{GroupName} {Number}_ch00.tif`
- **Fluorescence:** `{GroupName} {Number}_ch01.tif`

**Examples:**

- ✅ `Control 001_ch00.tif` / `Control 001_ch01.tif`
- ✅ `10 001_ch00.tif` / `10 001_ch01.tif`
- ❌ `sample_001.tif` (missing channel suffix)
- ❌ `10_001_ch00.tif` (incorrect format)

### 📌 Metadata Files (Critical!)

Each image requires metadata XML files for pixel size calibration:

**Preferred (most reliable):**

- `{ImageBaseName}_Properties.xml` - Contains pixel size in µm

**Fallback:**

- `{ImageBaseName}.xml` - Alternative metadata format

**Example for image `10 001_ch00.tif`:**

- Look for: `10 001_Properties.xml` (preferred)
- Or: `10 001.xml` (fallback)

**If metadata is missing:**

- Script uses fallback pixel size: `0.109492 µm/px`
- **⚠️ Warning:** This may produce inaccurate measurements!

---

## 🚀 Usage

### Basic Workflow

1. **Prepare your data:**

   ```bash
   # Organize images into the required structure
   source/
   ├── Control group/
   └── 10/, 11/, 12/, ...
   ```

2. **Run the analysis:**

   ```bash
   python app.py
   ```

3. **Follow interactive prompts:**

   **Step 1: Choose percentile cutoff**

   ```
   Select percentile for top/bottom group analysis:
     [1] 20%
     [2] 25%
     [3] 30% (default)
   Enter number (or press Enter for default):
   ```

   - Removes outliers (e.g., 20% cuts top/bottom 20%)

   **Step 2: Enter dataset label (optional)**

   ```
   Enter dataset identifier for plot titles (e.g., 'PD G-', 'Spike G+'):
     - This will appear at the start of all plot titles
     - Press Enter to skip (no dataset label)
   Dataset label: PD G-
   ```

   **Step 3: Select processing mode**

   ```
   Select run mode:
     [1] ALL numeric groups + Control (percentile only)
     [2] Single group + Control
   Enter number: 1
   ```

   **If choosing Single Group Mode (option 2):**

   ```
   Select ONE sample group folder to process (single-group mode):
     [1] 10
     [2] 11
     [3] 12
   Enter number: 1
   ```

4. **Wait for processing:**
   ```
   Processing images: 100%|████████████| 24/24 [02:15<00:00]
   SUMMARY: 24 succeeded, 0 failed
   ```

---

## 📊 Output Structure

```
outputs/                            # AUTO-GENERATED OUTPUT FOLDER
│
├── Control group/
│   ├── Control 001_ch00/
│   │   ├── 01_gray_8bit.png                  # Processing stages
│   │   ├── 02_enhanced.png
│   │   ├── 11_contours_rejected_orange_accepted_yellow_ids_green.png
│   │   ├── 13_mask_accepted.png
│   │   ├── 13_mask_accepted_ids.png          # Labeled version
│   │   ├── 20_fluorescence_aligned_raw.png   # Aligned fluorescence
│   │   ├── 22_fluorescence_mask_global.png
│   │   ├── 24_bf_fluor_matching_overlay.png
│   │   └── object_stats.csv                  # Raw measurements
│   │
│   ├── Control 002_ch00/
│   │   └── ...
│   │
│   ├── Control group_master.xlsx             # CONSOLIDATED EXCEL REPORT
│   └── error_bar_jitter_comparison_SD_all_groups.png  # ALL-GROUPS PLOT
│
├── 10/
│   ├── 10 001_ch00/
│   ├── 10 002_ch00/
│   ├── 10_master.xlsx                        # Group 10 report
│   └── error_bar_jitter_comparison_SD_vs_Control.png  # vs Control plot
│
├── 11/
│   ├── 11_master.xlsx
│   └── error_bar_jitter_comparison_SD_vs_Control.png
│
└── logs/                           # Runtime logs
    └── run_20231215_143022_app.txt
```

---

## 📈 Excel Report Sheets

Each `{GroupName}_master.xlsx` contains:

### 1. **Summary** (First sheet)

- Per-image statistics
- Embedded comparison plot
- Quick overview of results

### 2. **Ratios** (Second sheet)

- Fluorescence density vs rank plots
- BF/Fluor area ratio charts
- Embedded debug images for QA

### 3. **README**

- Column name definitions
- Measurement descriptions

### 4. **{GroupName}\_Typical_Particles**

- **🟡 Yellow-highlighted** typical particles
- Middle 40-60% after percentile filtering
- Primary dataset for statistical analysis

### 5. **{GroupName}\_All_Valid_Objects**

- All particles passing quality filters
- Before percentile cutoff

### 6. **Excluded_Objects**

- **🔴 Red-highlighted** excluded particles
- Detailed exclusion reasons:
  - Zero fluorescence area
  - Zero integrated density
  - Outside typical percentile range

### 7. **Per-Image Sheets**

- Individual image measurements
- All detected objects (before filtering)

---

## 📊 Key Measurements

| Column Name                   | Description                       | Units                      |
| ----------------------------- | --------------------------------- | -------------------------- |
| **Object_ID**                 | Unique particle identifier        | `{Group}_{Image}_{Number}` |
| **BF_Area_um2**               | Brightfield particle area         | µm²                        |
| **Fluor_Area_um2**            | Fluorescence region area (global) | µm²                        |
| **Fluor_Mean**                | Average fluorescence intensity    | a.u.                       |
| **Fluor_IntegratedDensity**   | Total fluorescence signal         | a.u.                       |
| **Fluor_Density_per_BF_Area** | **Primary metric:** Fluor/BF area | a.u./µm²                   |
| **BF_to_Fluor_Area_Ratio**    | Size mismatch indicator           | ratio                      |

**🎯 Primary Analysis Metric:**

- `Fluor_Density_per_BF_Area` = Total fluorescence signal ÷ Brightfield area

---

## 🔬 Processing Pipeline Details

### Step 1: Brightfield Segmentation

1. **Gaussian blur** (σ=15) for background estimation
2. **Background subtraction** (enhances dark objects)
3. **Otsu thresholding** (adaptive binary threshold)
4. **Morphological operations** (close → dilate → erode)
5. **Size filtering:**
   - Min area: 3 µm²
   - Max area: 2000 µm²
   - Max single object: 25% of image area

### Step 2: Fluorescence Channel Alignment

1. **Phase correlation registration** (sub-pixel accuracy)
2. **Invert brightfield** (dark cells → bright features)
3. **Calculate shift** (Y, X translation)
4. **Warp fluorescence** to match brightfield
5. **Save aligned raw image** for verification

### Step 3: Fluorescence Segmentation

1. **Gaussian blur** (σ=1.5)
2. **Modified Otsu threshold** (0.5× Otsu for faint signals)
3. **Morphological refinement** (open → close)
4. **Size filtering** (min area: 3 µm²)

### Step 4: BF-Fluor Matching

- **Many-to-one matching** allowed
- **Overlap-based assignment** (max intersection)
- **Minimum overlap:** 5 pixels
- **Intensity measured:** within BF contour
- **Area measured:** from global fluor contour

### Step 5: Group-Level Filtering

1. Collect **all valid objects** from all images
2. Sort by `Fluor_Density_per_BF_Area`
3. Apply **percentile cutoff** (e.g., cut top/bottom 20%)
4. Export **Typical_Particles** sheet

---

## 🎨 Debug Images Explained

| Image                              | Description                                  | Color Code                         |
| ---------------------------------- | -------------------------------------------- | ---------------------------------- |
| `01_gray_8bit.png`                 | 8-bit normalized brightfield                 | Grayscale                          |
| `02_enhanced.png`                  | Background-subtracted                        | Grayscale                          |
| `11_contours_*.png`                | Particle contours                            | 🟠 Rejected / 🟡 Accepted / 🟢 IDs |
| `13_mask_accepted.png`             | Binary mask of accepted particles            | White/Black                        |
| `20_fluorescence_aligned_raw.png`  | **Aligned fluorescence** (post-registration) | Grayscale                          |
| `22_fluorescence_mask_global.png`  | Fluorescence binary mask                     | White/Black                        |
| `24_bf_fluor_matching_overlay.png` | BF-Fluor matching visualization              | 🔴 BF / 🟢 Fluor                   |

**Files ending with `_ids.png`:**

- Same as original but with **🟢 green object IDs** labeled

---

## 📊 Statistical Plots

### Error Bar Comparison Plots

**Generated plots:**

1. **All-groups plot** (in Control group folder):

   - `error_bar_jitter_comparison_SD_all_groups.png`

2. **Pairwise plots** (in each group folder):
   - `error_bar_jitter_comparison_SD_vs_Control.png`

**Plot features:**

- **Bar height:** Group mean
- **Error bars:** ±1 Standard Deviation
- **Control error bars:** 2× wider caps for visibility
- **Cyan dots:** Individual data points (jittered)
- **Y-axis:** Fluorescence Density (a.u./µm²)

**Title format:**

```
{Dataset_ID} — Comparison (Error Bars: Standard Deviation) —
Typical = Middle 60% (Cut top/bottom 20%) — {Suffix}
```

---

## ⚙️ Configuration Options

Edit `app.py` to customize:

```python
# === FILE PATHS ===
SOURCE_DIR = Path("./source")              # Input folder
OUTPUT_DIR = Path("./outputs")             # Output folder (was "./debug")

# === SEGMENTATION PARAMETERS ===
GAUSSIAN_SIGMA = 15                        # BF blur radius
MIN_AREA_UM2 = 3.0                         # Min particle size (µm²)
MAX_AREA_UM2 = 2000.0                      # Max particle size (µm²)
MIN_CIRCULARITY = 0.0                      # Shape filter (0-1)
MAX_FRACTION_OF_IMAGE_AREA = 0.25          # Max single object size

# === FLUORESCENCE PARAMETERS ===
FLUOR_GAUSSIAN_SIGMA = 1.5                 # Fluor blur radius
FLUOR_MIN_AREA_UM2 = 3.0                   # Min fluor area
FLUOR_MATCH_MIN_INTERSECTION_PX = 5.0      # Min overlap for matching

# === OUTPUT OPTIONS ===
CLEAR_OUTPUT_DIR_EACH_RUN = True           # Delete outputs before running
SEPARATE_OUTPUT_BY_GROUP = True            # One folder per group
FALLBACK_UM_PER_PX = 0.109492              # If metadata missing

# === SCALE BAR ===
SCALE_BAR_LENGTH_UM = 10                   # Scale bar length (µm)
```

---

## ⚠️ Common Issues & Solutions

### 1️⃣ **No images found**

```
FileNotFoundError: No images found under ./source matching *_ch00.tif
```

**Solution:**

- Check folder structure matches requirements
- Verify image naming: `{Name} {Number}_ch00.tif`
- Ensure `source/` folder exists in project root

---

### 2️⃣ **Missing metadata**

```
[WARN] Could not determine pixel size -> using fallback 0.109492 µm/px
```

**Solution:**

- Add `_Properties.xml` files to `MetaData/` folders
- Or update `FALLBACK_UM_PER_PX` to correct value
- Extract pixel size from microscope software

---

### 3️⃣ **Excel file locked**

```
[ERROR] Cannot overwrite {file}_master.xlsx - file is open in another program
```

**Solution:**

- Close Excel before running analysis
- Delete locked file manually if needed

---

### 4️⃣ **Poor segmentation**

- **Too many particles detected:**

  - Increase `MIN_AREA_UM2`
  - Increase `MIN_CIRCULARITY`

- **Too few particles detected:**

  - Decrease `GAUSSIAN_SIGMA` (less blur)
  - Check debug images: `02_enhanced.png`

- **Fluorescence not matching:**
  - Check `20_fluorescence_aligned_raw.png` for alignment
  - Verify `24_bf_fluor_matching_overlay.png`
  - Adjust `FLUOR_MATCH_MIN_INTERSECTION_PX`

---

## 🔍 Quality Control Checklist

Before trusting results:

- [ ] **Check alignment:** `20_fluorescence_aligned_raw.png` - cells should align
- [ ] **Verify segmentation:** `11_contours_*.png` - particles correctly outlined
- [ ] **Check matching:** `24_bf_fluor_matching_overlay.png` - BF/fluor overlap
- [ ] **Review excluded objects:** Check `Excluded_Objects` sheet for valid exclusions
- [ ] **Inspect typical particles:** Yellow-highlighted cells represent your dataset

---

## 📝 Workflow Tips

### Best Practices

1. **Start with single group:**

   - Run mode [2] on one group first
   - Verify segmentation quality
   - Adjust parameters if needed

2. **Check debug images:**

   - Review all `*_ids.png` images
   - Ensure object IDs match expectations

3. **Validate measurements:**

   - Spot-check values in Excel
   - Compare to manual measurements

4. **Use dataset labels:**
   - Label your experiments (e.g., "PD G-")
   - Makes plots self-documenting

### Reproducibility

- All parameters logged to `logs/run_*.txt`
- Save configuration settings used
- Document percentile choice in notes

---

## 🆘 Getting Help

**Check these first:**

1. Review `logs/run_*.txt` for error messages
2. Examine debug images in output folders
3. Verify folder structure matches requirements

**Common debug steps:**

```bash
# Test single image
python app.py
# Choose: [2] Single group + Control
# Select one group to test

# Check what was found
ls source/*/MetaData/

# Verify output
ls outputs/*/
```

---

## 📄 Output File Reference

### CSV Files (`object_stats.csv`)

- Raw per-object measurements
- One file per image
- All columns documented in Excel README sheet

### Excel Files (`*_master.xlsx`)

- Consolidated group data
- 7+ sheets (see Excel Report Sheets section)
- Embedded plots and images

### PNG Files

- Debug visualizations
- QA images for publication
- Files ending `_ids.png` have labeled objects

### Log Files (`logs/run_*.txt`)

- Complete stdout/stderr capture
- Timestamps for each step
- Error messages and warnings

---

## 🔬 Scientific Use

### Citation Recommendations

When publishing results from this pipeline:

1. **Methods section example:**

   ```
   Brightfield and fluorescence microscopy images were analyzed using
   a custom Python pipeline (Python 3.x, OpenCV, scikit-image).
   Particles were segmented using adaptive Otsu thresholding with
   size filtering (3-2000 µm²). Fluorescence channels were registered
   using phase correlation (sub-pixel accuracy). Fluorescence density
   was calculated as integrated intensity per particle area. Outliers
   were excluded using percentile filtering (top/bottom 20%).
   Statistical comparisons used standard deviation error bars.
   ```

2. **Figure legends:**
   ```
   Error bars represent ±1 SD. Cyan dots: individual particles.
   Analysis included middle 60% of particles after percentile filtering.
   ```

### Data Archiving

For reproducibility, archive:

- [ ] Raw `.tif` images
- [ ] Metadata `.xml` files
- [ ] `app.py` script (exact version used)
- [ ] Configuration parameters
- [ ] `logs/run_*.txt` file
- [ ] Final `*_master.xlsx` files

---

## 🔄 Version History

**Current version features:**

- ✅ Fluorescence channel alignment (phase correlation)
- ✅ Group-level percentile filtering
- ✅ Many-to-one BF-Fluor matching
- ✅ Comprehensive Excel reports
- ✅ Statistical comparison plots
- ✅ Dataset identifier labeling
- ✅ Yellow contour visualization

---

## 📧 Support

For issues or questions:

1. Check this README first
2. Review debug images and logs
3. Verify folder structure and naming
4. Ensure all dependencies installed

---

## ⚖️ License

This pipeline uses open-source dependencies. Check individual library licenses:

- OpenCV: Apache 2.0
- NumPy: BSD
- Pandas: BSD
- Matplotlib: PSF-based
- scikit-image: BSD

---

**Happy analyzing! 🔬📊**
