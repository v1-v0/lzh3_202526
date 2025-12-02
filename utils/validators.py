"""
utils/validators.py - Input Validation
"""
from pathlib import Path
from typing import Union


def validate_image_path(path: Path) -> bool:
    """Validate that path is a valid image file."""
    if not path.exists():
        return False

    if not path.is_file():
        return False

    if path.suffix.lower() not in ['.tif', '.tiff']:
        return False

    return True


def validate_parameter(param_name: str, value: Union[int, float], 
                      min_val: Union[int, float], 
                      max_val: Union[int, float]) -> bool:
    """Validate that parameter is within acceptable range."""
    if not isinstance(value, (int, float)):
        print(f"{param_name} must be numeric")
        return False

    if value < min_val or value > max_val:
        print(f"{param_name} must be between {min_val} and {max_val}")
        return False

    return True