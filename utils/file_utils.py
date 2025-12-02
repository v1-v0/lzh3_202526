"""
utils/file_utils.py - File Operations and Folder Dialogs
"""
from pathlib import Path
from typing import List, Optional
import tkinter.filedialog as filedialog


def is_valid_image_file(filepath: Path) -> bool:
    """Check if file is a valid brightfield image (ch00.tif)."""
    name = filepath.name
    is_valid = (not name.startswith('.') and 
                not name.startswith('~') and 
                name.endswith('ch00.tif'))
    return is_valid


def scan_source_folder(source_dir: Path) -> List[Path]:
    """Scan folder for all valid ch00.tif files."""
    if not source_dir.exists() or not source_dir.is_dir():
        return []

    try:
        all_files = list(source_dir.iterdir())
        tif_files = [f for f in all_files 
                    if f.is_file() and f.suffix.lower() in ['.tif', '.tiff']]
        ch00_files = [f for f in tif_files if 'ch00' in f.name.lower()]

        valid = [f for f in ch00_files if is_valid_image_file(f)]
        return sorted(valid, key=lambda p: p.name)

    except Exception as e:
        print(f"Error scanning folder: {e}")
        return []


def choose_and_load_folder() -> Optional[Path]:
    """Open folder selection dialog."""
    folder = filedialog.askdirectory(
        title="Select Folder containing ch00.tif files"
    )

    if folder:
        return Path(folder)

    return None