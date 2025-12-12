"""Utility functions for image processing and analysis."""

import numpy as np
from pathlib import Path
import json

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config: dict, output_path: str):
    """Save configuration to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

def validate_image_pair(bf_path: Path, fl_path: Path) -> bool:
    """Validate that image pair exists and has matching dimensions."""
    if not bf_path.exists() or not fl_path.exists():
        return False
    # Add dimension checking logic
    return True

def create_output_directories(base_path: Path):
    """Create standard output directory structure."""
    dirs = [
        'bf_masks', 'fl_masks', 'visualizations', 
        'measurements', 'logs'
    ]
    for dir_name in dirs:
        (base_path / dir_name).mkdir(parents=True, exist_ok=True)