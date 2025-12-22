# Required Files and Folders for app.py

Based on my analysis of `app.py`, here are the **necessary files and folders** you need to keep:

## 📁 **Required Folder Structure**

```
project_root/
│
├── app.py                          # Main script (KEEP)
│
├── source/                         # INPUT folder (KEEP)
│   ├── Control group/              # Control images (KEEP)
│   │   ├── *_ch00.tif             # Brightfield images
│   │   ├── *_ch01.tif             # Fluorescence images
│   │   └── MetaData/              # Metadata XMLs
│   │       ├── *_Properties.xml
│   │       └── *.xml
│   │
│   ├── 10/                        # Sample group folders (KEEP)
│   ├── 11/
│   ├── 12/
│   └── ... (other numeric folders)
│       ├── *_ch00.tif
│       ├── *_ch01.tif
│       └── MetaData/
│
├── logs/                          # Auto-generated (can delete)
│   └── run_*.txt
│
└── debug/                         # OUTPUT folder (can delete)
    ├── Control group/
    ├── 10/
    ├── 11/
    └── ...
```

---

## ✅ **What to KEEP**

### 1. **Core Script**

- `app.py` - Main analysis script

### 2. **Input Data** (`source/` folder)

- **Control group/** - Control sample images
- **Numeric folders** (10, 11, 12, etc.) - Experimental groups
- **Image files:**
  - `*_ch00.tif` - Brightfield channel (required)
  - `*_ch01.tif` - Fluorescence channel (required)
- **MetaData/** subfolder in each group:
  - `*_Properties.xml` - Pixel size metadata (preferred)
  - `*.xml` - Alternative metadata files

### 3. **Dependencies**

- Python libraries (install via pip):
  ```bash
  pip install opencv-python numpy pandas scipy matplotlib seaborn tqdm openpyxl scikit-image
  ```

---

## 🗑️ **What You Can DELETE**

### 1. **Output Folders** (regenerated on each run)

- `debug/` - All analysis outputs
- `logs/` - Runtime logs

### 2. **Generated Files** (recreated automatically)

- `*_master.xlsx` - Excel reports
- `*.png` - Debug images
- `*.csv` - Statistics files
- `run_*.txt` - Log files

---

## 🔍 **Key Points**

### **Metadata Requirement**

The script **requires** XML metadata files to determine pixel size:

- **Preferred:** `*_Properties.xml` (more reliable)
- **Fallback:** Main `*.xml` file
- **Last resort:** Hardcoded fallback (0.109492 µm/px)

### **Image Naming Convention**

- Brightfield: `*_ch00.tif`
- Fluorescence: `*_ch01.tif`
- Must be paired (same base name)

### **Folder Structure Rules**

1. Control group must be named **"Control group"**
2. Experimental groups must be **numeric** (10, 11, 12, etc.)
3. Each group needs a **MetaData/** subfolder

---

## 📋 **Minimal Working Example**

```
project_root/
├── app.py
└── source/
    ├── Control group/
    │   ├── Control 001_ch00.tif
    │   ├── Control 001_ch01.tif
    │   └── MetaData/
    │       └── Control 001_Properties.xml
    │
    └── 10/
        ├── 10 001_ch00.tif
        ├── 10 001_ch01.tif
        └── MetaData/
            └── 10 001_Properties.xml
```

This is the **absolute minimum** needed to run the script successfully.

---

## ⚠️ **Important Notes**

1. **Never delete `source/`** - This is your original data
2. **Output folders regenerate** - Safe to delete `debug/` and `logs/`
3. **Keep metadata XMLs** - Critical for accurate measurements
4. **Backup before cleanup** - Always keep originals safe

Would you like me to create a cleanup script to safely remove only the regenerated files?
