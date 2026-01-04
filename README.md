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
    ├── Experiment_A/               # Experiment folder (any name)
    │   │
    │   ├── Control group/          # Control samples (required)
    │   │   ├── G+ microgel 1_ch00.tif
    │   │   ├── G+ microgel 1_ch01.tif
    │   │   ├── G+ microgel 2_ch00.tif
    │   │   ├── G+ microgel 2_ch01.tif
    │   │   └── MetaData/           # Metadata folder (required)
    │   │       ├── G+ microgel 1_Properties.xml
    │   │       ├── G+ microgel 1.xml
    │   │       └── ...
    │   │
    │   ├── 1/                      # Experimental group 1
    │   │   ├── 1 N NO 1_ch00.tif
    │   │   ├── 1 N NO 1_ch01.tif
    │   │   ├── 1 N NO 2_ch00.tif
    │   │   ├── 1 N NO 2_ch01.tif
    │   │   └── MetaData/
    │   │       ├── 1 N NO 1_Properties.xml
    │   │       ├── 1 N NO 1.xml
    │   │       └── ...
    │   │
    │   ├── 2/                      # Experimental group 2
    │   ├── 3/                      # Experimental group 3
    │   └── ...                     # Additional numeric groups
    │
    ├── PD sample/                  # Another experiment folder
    │   ├── G+/                     # Sub-folder with Control + groups
    │   │   ├── Control group/
    │   │   ├── 1/
    │   │   ├── 2/
    │   │   └── ...
    │   └── G-/
    │       ├── Control group/
    │       └── ...
    │
    └── Spike sample/               # Yet another experiment
        ├── G+/
        └── G-/
```

### 📌 New Folder Organization Features

**✨ Automatic Source Directory Selection:**

The pipeline now automatically scans up to 2 levels deep in the `source/` folder and displays **only valid experiment directories** (those containing a Control subfolder):

```
Available directories (up to 2 levels deep):
  [1] PD sample\G+
  [2] PD sample\G-
  [3] Spike sample\G+
  [4] Spike sample\G-

Enter the number or full path of the directory (or 'q' to quit): 1
```

**Requirements for valid experiment folders:**

1. Must contain a **Control subfolder** (name starts with "Control" or "control")
2. Must contain **numeric group folders** (e.g., 1, 2, 3)
3. Can be nested up to 2 levels deep in `source/`

### 📌 Folder Naming Rules

1. **Control group:** Must start with `"Control"` (case-insensitive)
   - ✅ `Control group`
   - ✅ `Control`
   - ✅ `control_samples`
2. **Experimental groups:** Must be **numeric folders only** (e.g., `1`, `2`, `3`)
   - ✅ `1`, `2`, `10`, `11`
   - ❌ `Group1`, `G1` (not purely numeric)
3. **MetaData folder:** Must exist in each group folder

### 📌 Image Naming Convention

- **Brightfield:** `{GroupName} {Number}_ch00.tif`
- **Fluorescence:** `{GroupName} {Number}_ch01.tif`

**Examples:**

- ✅ `G+ microgel 1_ch00.tif` / `G+ microgel 1_ch01.tif`
- ✅ `1 N NO 1_ch00.tif` / `1 N NO 1_ch01.tif`
- ✅ `10 P 1_ch00.tif` / `10 P 1_ch01.tif`
- ❌ `sample_001.tif` (missing channel suffix)
- ❌ `1_001_ch00.tif` (incorrect format)

### 📌 Metadata Files (Critical!)

Each image requires metadata XML files for pixel size calibration:

**Preferred (most reliable):**

- `{ImageBaseName}_Properties.xml` - Contains pixel size in µm

**Fallback:**

- `{ImageBaseName}.xml` - Alternative metadata format

**Example for image `1 N NO 1_ch00.tif`:**

- Look for: `1 N NO 1_Properties.xml` (preferred)
- Or: `1 N NO 1.xml` (fallback)

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
   ├── Your_Experiment/
   │   ├── Control group/
   │   └── 1/, 2/, 3/, ...
   ```

2. **Run the analysis:**

   ```bash
   python app.py
   ```

3. **Follow interactive prompts:**

   **Step 1: Select experiment directory**

   ```
   ================================================================================
   SELECT SOURCE DIRECTORY
   ================================================================================

   Available directories (up to 2 levels deep):
     [1] PD sample\G+
     [2] PD sample\G-
     [3] Spike sample\G+
     [4] Spike sample\G-

   Enter the number or full path of the directory (or 'q' to quit): 1
   [USER INPUT] Directory selection: 1

   ✓ Selected: source\PD sample\G+
   ✓ Control directory found: Control group
   ```

   **Step 2: Enter dataset label**

   ```
   Enter dataset identifier for plot titles (e.g., 'PD G-', 'Spike G+'):
     - This will appear at the start of all plot titles
     - Press Enter to use timestamp as label
   Dataset label: PD G plus
     → Using dataset label: 'PD G plus'
       Confirm? (y/n, or press Enter for yes): y
   [USER ACTION] Dataset ID confirmed: 'PD G plus'

   ✓ Output directory: C:\...\outputs\PD_G_plus_20260104_194501_app
   ```

   **Step 3: Select processing mode**

   ```
   Select run mode:
     [1] ALL numeric groups + Control
     [2] Single group + Control
   Enter number: 1
   ```

   **Step 4: Choose percentile cutoff**

   ```
   Select percentile for top/bottom group analysis:
     [1] 20%
     [2] 25%
     [3] 30% (default)
   Enter number (or press Enter for default): (pressed Enter)
   [USER ACTION] Selected percentile: 30% (default)
   ```

   **If choosing Single Group Mode (option 2):**

   ```
   Select ONE sample group folder to process (single-group mode):
     [1] 1
     [2] 2
     [3] 3
   Enter number: 1
   ```

4. **Wait for processing:**
   ```
   Processing images: 100%|████████████| 50/50 [00:20<00:00, 2.46img/s]
   SUMMARY: 50 succeeded, 0 failed
   ```

---

## 📊 Output Structure

**✨ New: Dataset-labeled output folders**

```
outputs/                            # AUTO-GENERATED OUTPUT FOLDER
│
├── PD_G_plus_20260104_194501_app/  # Output named by dataset ID + timestamp
│   │
│   ├── Control group/
│   │   ├── G+ microgel 1_ch00/
│   │   │   ├── 01_gray_8bit.png                  # Processing stages
│   │   │   ├── 02_enhanced.png
│   │   │   ├── 11_contours_rejected_orange_accepted_yellow_ids_green.png
│   │   │   ├── 13_mask_accepted.png
│   │   │   ├── 13_mask_accepted_ids.png          # Labeled version
│   │   │   ├── 20_fluorescence_aligned_raw.png   # NEW: Aligned fluorescence
│   │   │   ├── 22_fluorescence_mask_global.png
│   │   │   ├── 24_bf_fluor_matching_overlay.png
│   │   │   └── object_stats.csv                  # Raw measurements
│   │   │
│   │   ├── G+ microgel 2_ch00/
│   │   │   └── ...
│   │   │
│   │   └── Control group_master.xlsx             # CONSOLIDATED EXCEL REPORT
│   │
│   ├── 1/
│   │   ├── 1 N NO 1_ch00/
│   │   ├── 1 N NO 2_ch00/
│   │   ├── 1_master.xlsx                         # Group 1 report
│   │   └── error_bar_jitter_comparison_SD_vs_Control.png  # vs Control plot
│   │
│   ├── 2/
│   │   ├── 2_master.xlsx
│   │   └── error_bar_jitter_comparison_SD_vs_Control.png
│   │
│   ├── group_statistics_summary.csv              # NEW: All-groups statistics CSV
│   │
│   └── run_20260104_194501_app.txt               # Complete execution log
│
└── logs/                           # Original logs (before copying)
    └── run_20260104_194501_app.txt
```

**📌 Output Folder Naming:**

- **With dataset ID:** `{DatasetID}_{Timestamp}_{ScriptName}`
  - Example: `PD_G_plus_20260104_194501_app`
- **Without dataset ID (timestamp only):** `{Timestamp}_{ScriptName}`
  - Example: `20260104_194501_app`

---

## 📈 Excel Report Sheets

Each `{GroupName}_master.xlsx` contains:

### 1. **Summary** (First sheet)

- Per-image statistics
- **✨ NEW:** Embedded comparison plot (group vs Control)
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
- **✨ NEW:** Group-level percentile filtering (not per-image)
- Middle 40-60% after percentile filtering (default 30%)
- Primary dataset for statistical analysis

### 5. **{GroupName}\_All_Valid_Objects**

- All particles passing quality filters
- Before percentile cutoff
- Sorted by fluorescence density

### 6. **Excluded_Objects**

- **🔴 Red-highlighted** excluded particles
- **✨ NEW:** Detailed exclusion reasons:
  - Zero fluorescence area
  - Zero integrated density
  - Zero fluorescence area with positive integrated density
  - Outside typical particle range (top X%)
  - Outside typical particle range (bottom X%)

### 7. **Per-Image Sheets**

- Individual image measurements
- All detected objects (before filtering)
- Sorted by fluorescence density

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

## 📊 New Statistical Outputs

### **✨ Group Statistics CSV** (`group_statistics_summary.csv`)

Comprehensive statistics for all groups exported to CSV:

```csv
Group,N,Mean,Std_Dev,SEM,Median,Q30,Q70,Min,Max,CV_percent
1,19,3625.97,639.78,146.78,3620.57,3151.73,3993.23,2752.32,4632.64,17.64
2,11,1347.82,236.65,71.35,1274.37,1158.44,1541.78,1017.48,1652.67,17.56
Control,22,3918.21,264.28,56.35,3932.54,3783.36,4068.86,3470.90,4477.71,6.75
```

**Columns:**

- `Group` - Group name
- `N` - Sample size (typical particles only)
- `Mean` - Average fluorescence density
- `Std_Dev` - Standard deviation
- `SEM` - Standard error of the mean
- `Median` - 50th percentile
- `Q30` / `Q70` - 30th and 70th percentiles
- `Min` / `Max` - Range
- `CV_percent` - Coefficient of variation (%)

**Console output includes:**

```
Key Insights:
  • Total groups analyzed: 10
  • Total particles (typical only): 170
  • Mean across all groups: 3060.36 ± 1134.92 a.u./µm²
  • Smallest sample size: Group 2 (N=11)
  • Largest sample size: Group Control (N=22)

  ⚠ Groups with high variability (CV >30%):
     - Group 7: CV=26.0%
     - Group 8: CV=25.6%
```

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

### Step 2: Fluorescence Channel Alignment ✨ NEW

1. **Phase correlation registration** (sub-pixel accuracy)
2. **Invert brightfield** (dark cells → bright features)
3. **Calculate shift** (Y, X translation with upsample_factor=10)
4. **Warp fluorescence** to match brightfield
5. **Save aligned raw image** (`20_fluorescence_aligned_raw.png`) for verification
6. **Log alignment details:**
   ```
   Aligning Fluorescence channel to Brightfield...
     -> Detected shift: Y=0.20px, X=-0.00px
     -> Applied correction: ΔY=0.20px, ΔX=-0.00px
   ```

### Step 3: Fluorescence Segmentation

1. **Gaussian blur** (σ=1.5)
2. **Modified Otsu threshold** (0.5× Otsu for faint signals)
3. **Log threshold values:**
   ```
   Fluorescence threshold: Otsu=22.0, Adjusted=11.0
   ```
4. **Morphological refinement** (open → close)
5. **Size filtering** (min area: 3 µm²)

### Step 4: BF-Fluor Matching

- **Many-to-one matching** allowed
- **Overlap-based assignment** (max intersection)
- **Minimum overlap:** 5 pixels
- **Intensity measured:** within BF contour
- **Area measured:** from global fluor contour
- **Matching report:**
  ```
  Fluorescence matching: 9/11 BF objects matched
  Total fluorescence contours available: 9
    → 2 BF objects have NO fluorescence match (will have Fluor_Area_px=0)
  ```

### Step 5: Group-Level Filtering ✨ UPDATED

**Previous behavior:** Per-image percentile filtering  
**New behavior:** Group-level percentile filtering

1. Collect **all valid objects** from all images in the group
2. Sort by `Fluor_Density_per_BF_Area` (descending)
3. Apply **percentile cutoff** at group level (e.g., cut top/bottom 30%)
4. Export **Typical_Particles** sheet with middle particles only
5. Track excluded particles with detailed reasons

**Example log output:**

```
[INFO] Group-level filtering: 47 total → 19 typical (cut top/bottom 30%)
[INFO] Excluded 9 objects (see 'Excluded_Objects' sheet)
```

---

## 🎨 Debug Images Explained

| Image                                                          | Description                                  | Color Code                         |
| -------------------------------------------------------------- | -------------------------------------------- | ---------------------------------- |
| `01_gray_8bit.png`                                             | 8-bit normalized brightfield                 | Grayscale                          |
| `02_enhanced.png`                                              | Background-subtracted                        | Grayscale                          |
| `11_contours_rejected_orange_accepted_yellow_ids_green.png` ✨ | Particle contours                            | 🟠 Rejected / 🟡 Accepted / 🟢 IDs |
| `13_mask_accepted.png`                                         | Binary mask of accepted particles            | White/Black                        |
| `20_fluorescence_aligned_raw.png` ✨ NEW                       | **Aligned fluorescence** (post-registration) | Grayscale                          |
| `20_fluorescence_8bit.png`                                     | 8-bit normalized fluorescence                | Grayscale                          |
| `22_fluorescence_mask_global.png`                              | Fluorescence binary mask                     | White/Black                        |
| `24_bf_fluor_matching_overlay.png`                             | BF-Fluor matching visualization              | 🔴 BF / 🟢 Fluor                   |

**Files ending with `_ids.png`:**

- Same as original but with **🟢 green object IDs** labeled

**✨ NEW: Accepted contours now shown in YELLOW** for better visibility

---

## 📊 Statistical Plots

### Error Bar Comparison Plots

**Generated plots:**

1. **Pairwise plots** (in each group folder):
   - `error_bar_jitter_comparison_SD_vs_Control.png`
   - Shows: Selected group vs Control only

**Plot features:**

- **Bar height:** Group mean
- **Error bars:** ±1 Standard Deviation
- **Control error bars:** 2× wider caps (capsize=14) for visibility
- **Cyan dots:** Individual data points (jittered)
- **Y-axis:** Fluorescence Density (a.u./µm²)
- **✨ NEW:** Dataset ID appears in plot title

**Title format:**

```
{Dataset_ID} — Comparison (Error Bars: Standard Deviation) —
Typical = Middle 60% (Cut top/bottom 20%) — {Group} vs Control
```

**Example:**

```
PD G plus — Comparison (Error Bars: Standard Deviation) —
Typical = Middle 40% (Cut top/bottom 30%) — 1 vs Control
```

### Plot Embedding in Excel

**✨ All pairwise plots are automatically embedded in:**

- Each group's `Summary` sheet (cell G3)
- Marker cell G1 prevents duplicate embedding on re-run

---

## 📋 Complete Log Recording ✨ NEW

### Input Logging Convention

All user interactions are now fully logged with standardized markers:

```
Enter the number or full path of the directory (or 'q' to quit): 1
[USER INPUT] Directory selection: 1

Dataset label: PD G plus
  → Using dataset label: 'PD G plus'
    Confirm? (y/n, or press Enter for yes): y
[USER ACTION] Dataset ID confirmed: 'PD G plus'

Enter number: 1

Enter number (or press Enter for default): (pressed Enter)
[USER ACTION] Selected percentile: 30% (default)
```

**Log markers:**

- `[USER INPUT]` - What the user typed (or "(pressed Enter)")
- `[USER ACTION]` - System's interpretation/response
- `✓` - Successful completion
- `[INFO]` - Informational message
- `[WARN]` - Warning (non-fatal)
- `[ERROR]` - Error (potentially fatal)

### Log File Locations

1. **During execution:** `logs/run_{timestamp}_{script}.txt`
2. **After completion:** Copied to output folder as `run_{timestamp}_{script}.txt`

**Example log path:**

```
outputs/PD_G_plus_20260104_194501_app/run_20260104_194501_app.txt
```

---

## ⚙️ Configuration Options

Edit `app.py` to customize:

```python
# === FILE PATHS ===
SOURCE_DIR = Path("./source")              # Input folder (auto-selected)
OUTPUT_DIR = Path("./outputs")             # Output folder (auto-named by dataset ID)

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

# === ALIGNMENT (NEW) ===
# Phase correlation upsample_factor = 10 (0.1 px precision)
# Threshold multiplier = 0.5 (Otsu × 0.5 for faint signals)

# === OUTPUT OPTIONS ===
CLEAR_OUTPUT_DIR_EACH_RUN = True           # Delete outputs before running
SEPARATE_OUTPUT_BY_GROUP = True            # One folder per group
FALLBACK_UM_PER_PX = 0.109492              # If metadata missing

# === SCALE BAR ===
SCALE_BAR_LENGTH_UM = 10                   # Scale bar length (µm)
```

---

## ⚠️ Common Issues & Solutions

### 1️⃣ **No valid directories found**

```
[ERROR] No valid directories found with Control subfolders.
```

**Solution:**

- Ensure your experiment folder contains a subfolder starting with "Control"
- Check folder structure:
  ```
  source/
  └── Your_Experiment/
      ├── Control group/    <-- Must exist
      ├── 1/
      └── 2/
  ```
- Verify nesting depth ≤2 levels

---

### 2️⃣ **No images found**

```
FileNotFoundError: No images found under ./source matching *_ch00.tif
```

**Solution:**

- Check folder structure matches requirements
- Verify image naming: `{Name} {Number}_ch00.tif`
- Ensure images are in numeric group folders (not in parent experiment folder)

---

### 3️⃣ **Missing metadata**

```
[WARN] Could not determine pixel size -> using fallback 0.109492 µm/px
```

**Solution:**

- Add `_Properties.xml` files to `MetaData/` folders
- Or update `FALLBACK_UM_PER_PX` to correct value
- Extract pixel size from microscope software

---

### 4️⃣ **Excel file locked**

```
[ERROR] Cannot overwrite {file}_master.xlsx - file is open in another program
```

**Solution:**

- Close Excel before running analysis
- Delete locked file manually if needed

---

### 5️⃣ **Poor alignment** ✨ NEW

```
[WARN] Fluorescence not matching after alignment
```

**Solution:**

- Check `20_fluorescence_aligned_raw.png` - brightfield features should align with fluorescence
- Verify shift values in log:
  ```
  Detected shift: Y=0.20px, X=-0.00px
  ```
- Large shifts (>5px) may indicate:
  - Different field of view between channels
  - Stage drift during acquisition
  - Incorrect channel pairing

---

### 6️⃣ **Poor segmentation**

- **Too many particles detected:**

  - Increase `MIN_AREA_UM2`
  - Increase `MIN_CIRCULARITY`

- **Too few particles detected:**

  - Decrease `GAUSSIAN_SIGMA` (less blur)
  - Check debug images: `02_enhanced.png`

- **Fluorescence not matching:**
  - Check `20_fluorescence_aligned_raw.png` for alignment quality
  - Verify `24_bf_fluor_matching_overlay.png` - red (BF) and green (fluor) should overlap
  - Adjust `FLUOR_MATCH_MIN_INTERSECTION_PX`

---

## 🔍 Quality Control Checklist

Before trusting results:

- [ ] **Verify source selection:** Correct experiment folder chosen
- [ ] **Check alignment:** `20_fluorescence_aligned_raw.png` - cells should align
- [ ] **Verify segmentation:** `11_contours_*.png` - particles correctly outlined (yellow = accepted)
- [ ] **Check matching:** `24_bf_fluor_matching_overlay.png` - BF/fluor overlap
- [ ] **Review excluded objects:** Check `Excluded_Objects` sheet for valid exclusions
- [ ] **Inspect typical particles:** Yellow-highlighted cells in `_Typical_Particles` sheet
- [ ] **Verify group-level filtering:** Check console output for filtering statistics
- [ ] **Review statistics CSV:** Check `group_statistics_summary.csv` for high CV groups

---

## 🆘 Getting Help

**Check these first:**

1. Review `logs/run_*.txt` or output folder log for error messages
2. Examine debug images in output folders
3. Verify folder structure matches requirements
4. Check that Control folder exists and is detected

**Common debug steps:**

```bash
# Test single group
python app.py
# Choose: [2] Single group + Control
# Select one group to test

# Check what was found
ls source/*/Control\ group/
ls source/*/1/

# Verify alignment
# Look at: 20_fluorescence_aligned_raw.png

# Check output
ls outputs/*/
cat outputs/*/run_*.txt
```

---

## 📄 Output File Reference

### CSV Files

- **`object_stats.csv`** - Raw per-object measurements (one file per image)
- **`group_statistics_summary.csv`** ✨ NEW - All-groups statistics summary

### Excel Files (`*_master.xlsx`)

- Consolidated group data
- 7+ sheets (see Excel Report Sheets section)
- Embedded plots and images
- Group-level percentile filtering applied

### PNG Files

- Debug visualizations
- QA images for publication
- Files ending `_ids.png` have labeled objects
- **✨ NEW:** `20_fluorescence_aligned_raw.png` shows alignment quality

### Log Files (`logs/run_*.txt`)

- Complete stdout/stderr capture
- Timestamps for each step
- **✨ NEW:** Full user input recording with `[USER INPUT]` markers
- Error messages and warnings
- **✨ Automatically copied** to output folder for archiving

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
   using phase correlation (sub-pixel accuracy, upsample factor 10).
   Fluorescence segmentation used modified Otsu thresholding (0.5×
   Otsu threshold) to detect faint signals. Fluorescence density
   was calculated as integrated intensity per particle area. Outliers
   were excluded using group-level percentile filtering (top/bottom 30%).
   Statistical comparisons used standard deviation error bars with
   individual data points overlaid.
   ```

2. **Figure legends:**
   ```
   Error bars represent ±1 SD. Cyan dots: individual particles.
   Analysis included middle 40% of particles after group-level
   percentile filtering. Fluorescence channels were aligned using
   phase correlation registration.
   ```

### Data Archiving

For reproducibility, archive:

- [ ] Raw `.tif` images (brightfield + fluorescence)
- [ ] Metadata `.xml` files (Properties and Main)
- [ ] `app.py` script (exact version used)
- [ ] Configuration parameters (from Configuration section)
- [ ] Complete execution log (`run_*.txt` from output folder)
- [ ] `group_statistics_summary.csv`
- [ ] Final `*_master.xlsx` files (all sheets)
- [ ] Comparison plots (`.png` files)
- [ ] Dataset identifier used

---

## 🔄 Version History

**Current version features:**

- ✅ Flexible source directory selection (up to 2 levels deep)
- ✅ Automatic Control folder detection
- ✅ Dataset identifier labeling for outputs
- ✅ Fluorescence channel alignment (phase correlation, sub-pixel)
- ✅ Modified Otsu thresholding for faint fluorescence signals
- ✅ Group-level percentile filtering (not per-image)
- ✅ Many-to-one BF-Fluor matching
- ✅ Comprehensive Excel reports with embedded plots
- ✅ Pairwise statistical comparison plots (group vs Control)
- ✅ Yellow contour visualization for accepted particles
- ✅ Detailed exclusion tracking with reasons
- ✅ Group statistics CSV export
- ✅ Complete input logging with standardized markers
- ✅ Automatic log file archiving to output folder

---

## 📧 Support

For issues or questions:

1. **Check this README first** - especially Common Issues section
2. **Review debug images** - especially alignment (`20_fluorescence_aligned_raw.png`)
3. **Check logs** - look for `[ERROR]` or `[WARN]` markers
4. **Verify folder structure** - ensure Control folder exists
5. **Test alignment quality** - large shifts may indicate acquisition issues

**Log format for reporting issues:**

```
Log markers to look for:
  [USER INPUT]  - What you typed
  [USER ACTION] - What the system did
  [ERROR]       - Fatal errors
  [WARN]        - Warnings (may be important)
```

---

**🎉 Happy analyzing!**
