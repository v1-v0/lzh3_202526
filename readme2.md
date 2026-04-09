_Thinking..._

> The user wants me to update a README.md file for their project. They've provided:
>
> 1. A Python pipeline file (scout.py) - a comprehensive image analysis pipeline
> 2. A detailed research abstract/methods section that describes the project
>
> I need to create a comprehensive README.md that:
>
> 1. Includes the project title
> 2. Covers the background, objective, methods, results, and conclusion
> 3. Describes the technical aspects of the pipeline (based on the code)
> 4. Includes installation/usage instructions based on what I can infer from the code
> 5. Notes the limitations and future work
> 6. Addresses the concerns mentioned in the "TBC" section
> 7. Now I'm examining the pipeline's technical architecture - it uses OpenCV, NumPy, pandas, and scikit-image to process brightfield and fluorescence microscopy images, performing particle segmentation, channel alignment via phase correlation, and fluorescence density calculations, with support for batch processing, multi-configuration scanning, and output generation in Excel, CSV, and PDF formats. Now I'm mapping out the full directory structure and planning the README sections—I need to cover the pipeline architecture, configuration system, and output formats while being careful to frame the clinical claims appropriately rather than claiming superiority, and I'll organize this as a professional academic document with clear installation and usage instructions.
>
> Looking at the actual implementation details, I see the image processing uses brightfield and fluorescence channels with specific naming conventions, pulls pixel calibration from microscope metadata with a fallback value, and organizes results by timestamp and dataset ID—the pipeline itself appears to start with image loading and proceeds through several processing steps that I should document clearly.
>
> The full workflow chains together image normalization, filtering operations to identify particles based on morphological properties, channel alignment using cross-correlation techniques, fluorescence analysis, and finally generates comprehensive output including statistical comparisons and clinical classifications. In multi-scan mode, the system evaluates different bacterial configurations against control samples to determine the best discrimination parameters.
>
> Now I'm writing the README documentation, structuring it with installation requirements including numpy, pandas, scipy, opencv, scikit-image, and the custom bacteria_configs module, while maintaining the academic framing from the project abstract.

```markdown
# Automated OpenCV Pipeline for Quantifying Fluorescent Magnetic Spore-Based Microrobot Fluorescence Quenching in Peritoneal Dialysis Effluent: A Staged Proof-of-Concept Validation Study

> **Version:** 0406  
> **Status:** Proof-of-concept — sample identity was known a priori across all stages  
> **Institution:** The Chinese University of Hong Kong (CUHK)

---

## Table of Contents

1. [Background](#background)
2. [Objective](#objective)
3. [Methods](#methods)
4. [Results](#results)
5. [Conclusion and Limitations](#conclusion-and-limitations)
6. [Repository Structure](#repository-structure)
7. [Requirements and Installation](#requirements-and-installation)
8. [Input Data Layout](#input-data-layout)
9. [Pipeline Architecture](#pipeline-architecture)
10. [Bacteria Configuration System](#bacteria-configuration-system)
11. [Running the Pipeline](#running-the-pipeline)
12. [Output Description](#output-description)
13. [Classification Framework](#classification-framework)
14. [Statistical Methods](#statistical-methods)
15. [Multi-Configuration Scanning](#multi-configuration-scanning)
16. [Known Limitations and Future Work](#known-limitations-and-future-work)
17. [Terminology Note](#terminology-note)
18. [Acknowledgements](#acknowledgements)

---

## Background

Peritoneal dialysis (PD)-associated peritonitis is a major clinical complication requiring rapid identification of causative organisms to enable timely, targeted antimicrobial therapy. Conventional culture-based methods require 24–48 hours, creating a diagnostic window during which patients may receive empirical treatment that is suboptimal.

To address this delay, fluorescent magnetic spore-based microrobots (FMSMs) were engineered at CUHK by coating natural spores with Fe₃O₄ nanoparticles and functionalised carbon dots that emit red fluorescence. These FMSMs can be magnetically actuated to target bacteria within fluid samples, and their fluorescence is selectively quenched upon reaction with specific gram-positive or gram-negative bacterial markers, enabling detection within tens of minutes.

Previously, fluorescence quenching was quantified from microscopy images using manual region-of-interest analysis in ImageJ — a process that is time-consuming, operator-dependent, and not scalable for higher-throughput clinical workflows. This pipeline automates that quantification entirely in Python.

---

## Objective

To develop and validate an automated image analysis pipeline using Python and OpenCV to quantify FMSM fluorescence quenching in PD effluent microscopy images, and to evaluate classification concordance with manual ImageJ-based analysis using a staged validation approach across known specimen groups.

---

## Methods

### Specimen Groups

PD effluent was collected from patients undergoing peritoneal dialysis and processed into four specimen groups:

- Clean PD fluid (negative control baseline)
- Laboratory-prepared spike samples (gram-positive and gram-negative organisms)
- Clinical isolates of _Klebsiella pneumoniae_ (gram-negative)
- Clinical isolates of _Proteus mirabilis_ (gram-negative)

For each sample, five gram-positive FMSM (G+) and five gram-negative FMSM (G−) microscopy image pairs were acquired, each comprising a brightfield channel (ch00) and a paired fluorescence channel (ch01), yielding over 1,600 images across more than 80 samples.

### Staged Validation Approach

**Stage 1 — Baseline optimisation:** Image enhancement and contour-detection parameters were optimised using clean PD fluid specimens to establish baseline FMSM fluorescence profiles.

**Stage 2 — Spike sample validation:** Spike samples containing known gram-positive and gram-negative organisms were analysed to confirm that fluorescence quenching was detectable and that classification thresholds correctly distinguished positive from negative samples.

**Stage 3 — Clinical isolate evaluation:** The pipeline was applied to PD effluent containing _K. pneumoniae_ and _P. mirabilis_ (both gram-negative) to evaluate performance on patient-derived specimens.

### Automated Processing Steps

1. **Particle detection:** Individual FMSM particles were detected via contour extraction from Gaussian-blurred brightfield images. Contour detection used a configurable intensity threshold or Otsu thresholding followed by morphological opening and closing to separate touching particles and suppress noise.
2. **Particle filtering:** Detected contours were filtered by area (µm²), circularity, aspect ratio, and solidity using bacteria-specific configuration parameters. Rejected objects were logged separately for quality control.
3. **Channel alignment:** Fluorescence images were spatially co-registered to their paired brightfield images using dual-method phase cross-correlation (raw image correlation and Difference-of-Gaussians-filtered correlation), with the result selected by peak response score.
4. **Fluorescence measurement:** Detected brightfield contours were overlaid onto the aligned fluorescence image. For each accepted particle, fluorescence integrated density (sum of pixel values within the contour) and mean fluorescence intensity were calculated.
5. **Primary metric:** Fluorescence density was expressed as integrated density divided by brightfield contour area (arbitrary units per µm²), providing a size-normalised quenching measure.
6. **Percentile filtering:** To reduce the influence of outlier particles, only the middle 40% of detected particles — ranked by fluorescence density — were retained for statistical analysis, following the established laboratory protocol. This range corresponds to particles ranked from the 30th to the 70th percentile after sorting all valid particles across all images within a group. The criterion was established by the laboratory based on empirical observation that the extremes of the fluorescence distribution are disproportionately affected by imaging artefacts, debris, and aggregation events unrelated to quenching.
7. **Classification:** A threshold was computed as 5% below the negative control group mean fluorescence density. Samples whose typical-particle mean fell below this threshold were classified as bacteria detected.

### Classification Threshold Rationale

The 5% threshold value was adopted from the established laboratory protocol used in manual ImageJ analysis. It represents a margin judged by the laboratory to be large enough to absorb run-to-run variability while remaining sensitive to biologically meaningful quenching. This parameter is user-configurable (1–20%) and can be adjusted as validation data accumulate.

---

## Results

The automated pipeline processed 200 microscopy images, representing 20 specimens of a single FMSM type, in under five minutes. This substantially reduced processing time compared with manual ImageJ analysis and eliminated operator dependency.

Across specimen groups:

- Clinical gram-negative isolates of _K. pneumoniae_ and _P. mirabilis_ showed fluorescence density consistently below the gram-negative FMSM classification threshold, consistent with selective quenching by gram-negative bacterial markers.
- Clean PD fluid samples remained above the threshold, confirming baseline stability.
- Gram-positive classification was validated using spike samples only; gram-positive clinical isolates were not available during the study period.
- Classification outcomes from the automated pipeline were **concordant** with manual ImageJ analysis across all tested specimen groups.

> **Note on the concordance claim:** At present, concordance is reported qualitatively. A formal concordance table (e.g., number of specimens classified identically by both methods) is recommended for any future publication or examination submission. See [Known Limitations and Future Work](#known-limitations-and-future-work).

---

## Conclusion and Limitations

The automated OpenCV-based pipeline provides a reproducible, operator-independent, and scalable method for quantifying FMSM fluorescence quenching in PD effluent microscopy images, with classification concordant with established manual analysis across all tested specimen groups.

As a proof-of-concept study in which sample identity was known a priori, this work validates the analytical pipeline rather than diagnostic performance. It does not establish sensitivity, specificity, or other formal accuracy metrics.

---

## Repository Structure
```

project*root/
│
├── scout.py # Main pipeline script
├── bacteria_configs/ # JSON configuration files for each bacteria type
│ ├── klebsiella_pneumoniae.json
│ ├── proteus_mirabilis.json
│ └── default.json # General-purpose baseline config
│
├── bacteria_configs.py # SegmentationConfig dataclass definition
├── bacteria_registry.py # Multi-scan whitelist registry (optional)
│
├── source/ # Input images (user-created)
│ └── <dataset_name>/
│ ├── G+/ # Gram-positive FMSM images (batch mode)
│ │ ├── Control/
│ │ ├── 1/
│ │ └── 2/
│ └── G-/ # Gram-negative FMSM images (batch mode)
│ ├── Control/
│ ├── 1/
│ └── 2/
│
├── outputs/ # Auto-generated output (gitignored)
│ └── <timestamp>*<dataset_id>/
│ ├── Positive/
│ ├── Negative/
│ ├── final_clinical_results.xlsx
│ ├── laboratory_report.pdf
│ └── confidence_report.txt
│
└── logs/ # Auto-generated run logs (gitignored)

````

---

## Requirements and Installation

### Python Version

Python 3.10 or higher is recommended (required for `list[str]` and `tuple[...]` type hint syntax used throughout).

### Dependencies

Install all dependencies via pip:

```bash
pip install numpy pandas scipy tqdm opencv-python scikit-image matplotlib seaborn openpyxl
````

| Package         | Purpose                                                                    |
| --------------- | -------------------------------------------------------------------------- |
| `numpy`         | Numerical array operations                                                 |
| `pandas`        | Tabular data processing and Excel export                                   |
| `scipy`         | Statistical tests (Welch's t-test, confidence intervals)                   |
| `tqdm`          | Progress bars during batch image processing                                |
| `opencv-python` | Image I/O, Gaussian blur, morphology, contour detection, phase correlation |
| `scikit-image`  | Phase cross-correlation (legacy alignment path)                            |
| `matplotlib`    | Plot generation and PDF report rendering                                   |
| `seaborn`       | Bar and strip plot styling                                                 |
| `openpyxl`      | Excel workbook creation, chart embedding, image insertion                  |

### Custom Modules

The pipeline also requires two local modules in the project root:

- `bacteria_configs.py` — defines the `SegmentationConfig` dataclass
- `bacteria_registry.py` _(optional)_ — provides the `registry.get_whitelist()` method used in multi-scan mode; if absent, the pipeline falls back to a hardcoded whitelist

---

## Input Data Layout

### Single Mode

Place images in `source/<dataset_name>/<group_name>/`. At least one group folder must be named `Control` (case-insensitive prefix match).

```
source/
└── my_dataset/
    ├── Control/
    │   ├── Control 001_ch00.tif
    │   └── Control 001_ch01.tif
    ├── 1/
    │   ├── Group1 001_ch00.tif
    │   └── Group1 001_ch01.tif
    └── 2/
        ├── Group2 001_ch00.tif
        └── Group2 001_ch01.tif
```

### Batch Mode (G+ and G−)

For simultaneous G+ and G− analysis, create `G+` and `G-` subdirectories inside the dataset folder. Both must be present for batch mode to activate:

```
source/
└── my_dataset/
    ├── G+/
    │   ├── Control/
    │   ├── 1/
    │   └── 2/
    └── G-/
        ├── Control/
        ├── 1/
        └── 2/
```

### Image Naming Convention

- Brightfield (channel 0): `*_ch00.tif`
- Fluorescence (channel 1): `*_ch01.tif`

Both files must share the same base name and reside in the same folder. Leica microscope XML metadata files (`MetaData/*.xml`) are read automatically for pixel size calibration; if absent, the pipeline falls back to `0.109492 µm/px`.

---

## Pipeline Architecture

```
Input images (ch00 brightfield + ch01 fluorescence)
        │
        ▼
┌───────────────────────────────────────────────────┐
│  STAGE A: Brightfield Segmentation                │
│  Gaussian blur → threshold → morphology           │
│  → contour detection → filter by area,            │
│    circularity, aspect ratio, solidity            │
└───────────────────────┬───────────────────────────┘
                        │ accepted contours + rejected log
                        ▼
┌───────────────────────────────────────────────────┐
│  STAGE B: Fluorescence Channel Alignment          │
│  Dual-method phase cross-correlation:             │
│    Method 1 — raw images with Hanning window      │
│    Method 2 — DoG-filtered images                 │
│  Best shift selected by response score            │
└───────────────────────┬───────────────────────────┘
                        │ aligned fluorescence image
                        ▼
┌───────────────────────────────────────────────────┐
│  STAGE C: Fluorescence Measurement                │
│  Otsu threshold → fluor contours → overlap        │
│  matching with BF contours → integrated           │
│  density and mean intensity per particle          │
└───────────────────────┬───────────────────────────┘
                        │ per-particle measurements
                        ▼
┌───────────────────────────────────────────────────┐
│  STAGE D: Consolidation & Statistics              │
│  Per-group Excel export → percentile filter       │
│  (middle 40%) → group statistics → CI,            │
│  Cohen's d, Welch's t-test                        │
└───────────────────────┬───────────────────────────┘
                        │ classified groups
                        ▼
┌───────────────────────────────────────────────────┐
│  STAGE E: Clinical Classification & Reporting     │
│  Threshold comparison → gram classification       │
│  → G+/G− decision matrix → Excel + PDF report    │
└───────────────────────────────────────────────────┘
```

### Channel Alignment Details

The alignment module (`align_fluorescence_channel`) uses `cv2.phaseCorrelate` rather than `skimage.phase_cross_correlation` because the skimage error metric (1 − normalised cross-power peak) approaches 1.0 for cross-modal images, making it impossible to distinguish a good shift from no shift. The OpenCV response metric instead approaches 0.0 when no shift is detectable.

A shift is accepted when the response exceeds `MIN_RESPONSE = 0.02` and the magnitude is at most `MAX_SHIFT_PX = 30` pixels. When both methods return accepted shifts that agree within `AGREEMENT_PX = 2` pixels, they are averaged for sub-pixel accuracy. Diagnostic overlay images (`DIAG_A_no_shift.png`, `DIAG_B_positive_shift.png`) are saved alongside each processed image for quality review.

---

## Bacteria Configuration System

Each bacteria type is described by a JSON file in `bacteria_configs/`. The pipeline uses these files to control all segmentation parameters. An example configuration:

```json
{
  "config": {
    "name": "Klebsiella pneumoniae",
    "description": "Gram-negative rod, validated on PD effluent isolates",
    "gaussian_sigma": 12.0,
    "min_area_um2": 3.0,
    "max_area_um2": 800.0,
    "min_circularity": 0.3,
    "max_circularity": 1.0,
    "min_aspect_ratio": 0.5,
    "max_aspect_ratio": 5.0,
    "min_solidity": 0.5,
    "use_intensity_threshold": false,
    "intensity_threshold": 80.0,
    "invert_image": false,
    "pixel_size_um": 0.109492
  }
}
```

All numeric parameters are loaded with safe type-conversion helpers; `null` values in JSON fall back to hardcoded defaults without crashing. The `SegmentationConfig` dataclass (defined in `bacteria_configs.py`) exposes computed properties `min_area_px` and `max_area_px` that convert µm² values using the configured pixel size.

### Multi-scan Whitelist

Only configurations listed in `MULTI_SCAN_WHITELIST` are tested during unknown-sample scanning. The whitelist is loaded from `bacteria_registry.py` if available; otherwise it falls back to `['klebsiella_pneumoniae', 'proteus_mirabilis']`. Configurations not on the whitelist (e.g., `default`, any under-validated types) are excluded with a console explanation.

---

## Running the Pipeline

```bash
python scout.py
```

The script is fully interactive. It will prompt for:

1. **Processing mode** — Multi-scan (unknown sample) or single-configuration (known bacteria)
2. **Source directory** — Selected from directories present in `source/`
3. **Dataset label** — Defaults to the folder name; used in output naming
4. **Percentile filter** — Default 30% (retains middle 40% of particles)
5. **Clinical threshold** — Default 5% below control mean
6. **Configuration confirmation** — Displays a full summary before processing begins

All console output (prompts, responses, progress, warnings) is simultaneously written to a timestamped log file in `logs/` and copied to the output directory on completion.

### Last-used Defaults

The pipeline caches the last processing mode (`.last_processing_mode`) and last bacteria selection (`.last_bacteria_selection`) in the project root, so repeat runs can press Enter to reuse the previous settings.

---

## Output Description

### Single-mode output

```
outputs/<timestamp>_<dataset_id>/
├── <group_name>/
│   ├── <image_name>/
│   │   ├── 01_gray_8bit.png
│   │   ├── 02_blurred.png
│   │   ├── 05_opened.png  …  (segmentation debug images)
│   │   ├── 10_contours_all.png
│   │   ├── 11_contours_rejected_orange_accepted_yellow_ids_green.png
│   │   ├── 13_mask_accepted.png
│   │   ├── 13_mask_accepted_ids.png
│   │   ├── 20_fluorescence_8bit.png
│   │   ├── 22_fluorescence_mask_global.png
│   │   ├── 24_bf_fluor_matching_overlay.png
│   │   ├── DIAG_A_no_shift.png
│   │   ├── DIAG_B_positive_shift.png
│   │   ├── object_stats.csv        ← accepted particles
│   │   └── rejected_objects.csv    ← rejected particles with reasons
│   └── <group_name>_master.xlsx    ← consolidated group workbook
│
├── comparison_<type>_all_groups.png
├── clinical_classification_<type>.csv
├── clinical_classification_<type>.xlsx
├── group_statistics_summary.csv
├── rejection_analysis_summary.csv
└── run_<timestamp>_<file_executed>.txt     ← copied log
```

### Batch-mode additional output

```
outputs/<timestamp>_<dataset_id>/
├── Positive/                        ← G+ results (same structure as above)
├── Negative/                        ← G− results
├── final_clinical_results.csv
├── final_clinical_results.xlsx      ← combined G+/G− decision matrix
├── laboratory_report.pdf            ← 5-page A4 report
├── comparison_positive_all_groups.png
├── comparison_negative_all_groups.png
├── clinical_classification_positive.xlsx
└── clinical_classification_negative.xlsx
```

### Master Excel Workbook Structure

Each group produces a `<group_name>_master.xlsx` containing:

| Sheet                       | Contents                                                                             |
| --------------------------- | ------------------------------------------------------------------------------------ |
| Summary                     | Per-image totals, mean fluorescence density, particles detected                      |
| Ratios                      | Per-image fluorescence density and area ratio charts                                 |
| README                      | Column dictionary                                                                    |
| `<group>_Typical_Particles` | Middle 40% particles (primary analysis population; highlighted yellow)               |
| `<group>_All_Valid_Objects` | All objects with positive fluorescence area and integrated density                   |
| `<group>_Rejected_Objects`  | Contours rejected by morphological filters, with rejection reasons (highlighted red) |
| Excluded_Objects            | Objects excluded by zero-area or zero-density checks, plus percentile exclusions     |
| Per-image sheets            | Raw measurements for each image file                                                 |

### Laboratory Report PDF

The five-page A4 PDF (`laboratory_report.pdf`) contains:

1. **Summary page** — dataset metadata, final results table, group-by-group interpretation
2. **Comparison charts** — bar plots with SD error bars, jitter overlay, and threshold lines for G+ and G−
3. **Forest plot and decision heatmap** — Cohen's d effect sizes with 95% CI; colour-coded gram classification grid
4. **Statistical tables** — N, mean, SD, 95% CI, Cohen's d, p-value, and classification confidence for each group in both channels
5. **Methodology and approval** — processing parameters, quality-control notes, limitations, and signature fields

---

## Classification Framework

The pipeline implements a four-category classification that requires no prior knowledge of specimen contents:

| G+ FMSM fluorescence | G− FMSM fluorescence | Classification                                                        |
| -------------------- | -------------------- | --------------------------------------------------------------------- |
| Above threshold      | Above threshold      | No bacteria detected                                                  |
| Below threshold      | Above threshold      | Gram-positive bacteria detected                                       |
| Above threshold      | Below threshold      | Gram-negative bacteria detected                                       |
| Below threshold      | Below threshold      | Potentially polymicrobial / atypical — further investigation required |

**Threshold formula:** `threshold = control_mean × (1 − threshold_pct)`

The default `threshold_pct` of 5% (0.05) is user-configurable at runtime. Classification confidence is reported as High, Moderate, or Low based on whether the 95% CI of the group mean falls entirely on one side of the threshold and whether Cohen's d exceeds 0.8.

---

## Statistical Methods

For each test group versus the control group, the pipeline computes:

- **Mean, SD, SEM, median** of fluorescence density across typical particles
- **95% confidence interval** — Student's t-distribution with n − 1 degrees of freedom
- **Cohen's d** — pooled standard deviation formula; 95% CI approximated by the Hedges–Olkin method
- **Welch's t-test** — unequal-variance independent samples t-test (`scipy.stats.ttest_ind(equal_var=False)`)
- **Significance labels** — `***` p < 0.001, `**` p < 0.01, `*` p < 0.05, `ns` p ≥ 0.05
- **Strength of evidence** — Strong (p < 0.01, |d| ≥ 0.8, n ≥ 5), Moderate (p < 0.05, |d| ≥ 0.5, n ≥ 3), Weak, or Insufficient

All statistics are exported to `group_statistics_summary.csv`, the master Excel workbooks, and the PDF report.

---

## Multi-Configuration Scanning

When processing mode 1 (Unknown Sample) is selected, the pipeline tests all whitelisted bacteria configurations and scores each one on how well it separates test images from control images. Scoring is out of 100 points:

| Factor                                    | Max points | Basis                             |
| ----------------------------------------- | ---------- | --------------------------------- |
| Effect size (Cohen's d, test vs control)  | 35         | d ≥ 2.0 → 35 pts; d < 0.2 → 0 pts |
| Statistical significance (Welch's t-test) | 20         | p < 0.001 → 20 pts                |
| Particle count appropriateness            | 15         | 3–30 particles/image → 15 pts     |
| Within-group consistency (CV)             | 15         | CV < 0.30 → 15 pts                |
| Direction and magnitude bonus             | 5          | test < control as expected        |

**Non-specific detection disqualifier (post-scoring):** Configurations that detect more than 100 particles/image at mean fluorescence below 2.0 a.u./µm² are capped at 10 points (overcount artefact, likely debris or crystal noise).

**Pairwise discrimination penalty:** If the rank-1 and rank-2 configurations are not statistically distinguishable from each other (p ≥ 0.05 on per-image fluorescence), the rank-1 score is reduced toward rank-2 by a convergence factor proportional to the p-value, and the result is flagged as ambiguous in the confidence report.

The best-matching configuration is then automatically used for the full clinical analysis run (single-config mode) without requiring additional user input.

---

## Known Limitations and Future Work

### Current limitations

| Limitation                         | Detail                                                                                                                                                   |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sample identity known a priori     | All stages used specimens with known bacterial content; this is a pipeline validation, not a diagnostic accuracy study                                   |
| No gram-positive clinical isolates | Gram-positive classification was validated on spike samples only                                                                                         |
| Concordance reported qualitatively | A formal concordance table comparing automated and manual ImageJ classifications per specimen is needed before any publication or examination submission |
| Single FMSM type evaluated         | All 200 images in the quantitative comparison used one FMSM preparation batch                                                                            |
| Threshold empirically derived      | The 5% threshold was adopted from the laboratory protocol; its statistical basis has not been formally validated against a labelled reference set        |
| No prospective blinded evaluation  | No specimens were processed with identity concealed from the analyst                                                                                     |

### Recommended future work

1. **Formal concordance table:** Report the number and proportion of specimens classified identically by the automated pipeline and manual ImageJ analysis. Consider Cohen's κ if a sufficient number of specimens are available.
2. **Gram-positive clinical isolates:** Extend validation to culture-confirmed gram-positive PD peritonitis isolates.
3. **Prospective blinded analysis:** Process specimens where bacterial identity is unknown at the time of analysis; compare pipeline output to culture results.
4. **Formal accuracy metrics:** Once a labelled clinical dataset is available, compute sensitivity, specificity, positive predictive value, and negative predictive value.
5. **Multi-centre validation:** Evaluate robustness across different microscopes, FMSM preparation batches, and patient cohorts.
6. **Structured user feedback:** Design a standardised laboratory feedback form to capture operator-perceived usability, error rates, and classification consistency.
7. **Threshold optimisation:** Use ROC analysis on a labelled dataset to select the classification threshold that maximises a clinically appropriate objective (e.g., sensitivity-weighted) rather than the current empirical 5% value.

---

## Terminology Note

In this repository, **FMSM (fluorescent magnetic spore-based microrobot)** refers to the engineered spore particles described above. The term **microgel** (seen in variable names such as `microgel_type`) is a legacy label used in the source code and does not imply that these particles are polymer hydrogel microgels in the materials-science sense. If this code is adapted for polymer microgel assays, parameter values and threshold assumptions would require independent validation.

In the code, `microgel_type = "positive"` corresponds to G+ FMSMs (gram-positive-targeting) and `microgel_type = "negative"` corresponds to G− FMSMs (gram-negative-targeting).

---

## Acknowledgements

This pipeline was developed at The Chinese University of Hong Kong. FMSM synthesis and experimental protocols were established by the research group. The image analysis pipeline, statistical framework, and reporting system were implemented in Python as described in this repository.

---

_For questions about the image analysis pipeline, please open an issue in this repository. For questions about the FMSM synthesis or experimental protocol, please contact the corresponding laboratory directly._

```

```
