"""
Bacteria configuration file
Contains segmentation parameters for different bacteria types
"""

bacteria_configs = {
    "Klebsiella Pneumoniae": {
        "bacteria_segmentation": {
            "invert_image": False,
            "gaussian_sigma": 2.0,
            "brightness_adjust": 0,
            "contrast_adjust": 1.0,
            "threshold_offset": 0,
            "min_area": 20,
            "max_area": 5000,
            "dilate_iterations": 0,
            "erode_iterations": 0,
        },
        "inclusion_dark_segmentation": {
            "invert_image": False,
            "gaussian_sigma": 2.0,
            "brightness_adjust": 0,
            "contrast_adjust": 1.0,
            "threshold_offset": -10,
            "min_area": 10,
            "max_area": 2000,
            "dilate_iterations": 1,
            "erode_iterations": 1,
        },
        "inclusion_bright_segmentation": {
            "invert_image": True,
            "gaussian_sigma": 2.0,
            "brightness_adjust": 0,
            "contrast_adjust": 1.0,
            "threshold_offset": -10,
            "min_area": 10,
            "max_area": 2000,
            "dilate_iterations": 1,
            "erode_iterations": 1,
        },
    },
    
    "Escherichia Coli": {
        "bacteria_segmentation": {
            "invert_image": False,
            "gaussian_sigma": 2.0,
            "brightness_adjust": 0,
            "contrast_adjust": 1.0,
            "threshold_offset": 0,
            "min_area": 15,
            "max_area": 3000,
            "dilate_iterations": 0,
            "erode_iterations": 0,
        },
        "inclusion_dark_segmentation": {
            "invert_image": False,
            "gaussian_sigma": 2.5,
            "brightness_adjust": 0,
            "contrast_adjust": 1.0,
            "threshold_offset": -15,
            "min_area": 8,
            "max_area": 1500,
            "dilate_iterations": 1,
            "erode_iterations": 1,
        },
        "inclusion_bright_segmentation": {
            "invert_image": True,
            "gaussian_sigma": 2.5,
            "brightness_adjust": 0,
            "contrast_adjust": 1.0,
            "threshold_offset": -15,
            "min_area": 8,
            "max_area": 1500,
            "dilate_iterations": 1,
            "erode_iterations": 1,
        },
    },
    
    "Staphylococcus Aureus": {
        "bacteria_segmentation": {
            "invert_image": False,
            "gaussian_sigma": 2.0,
            "brightness_adjust": 0,
            "contrast_adjust": 1.0,
            "threshold_offset": 0,
            "min_area": 25,
            "max_area": 4000,
            "dilate_iterations": 0,
            "erode_iterations": 0,
        },
        "inclusion_dark_segmentation": {
            "invert_image": False,
            "gaussian_sigma": 2.0,
            "brightness_adjust": 0,
            "contrast_adjust": 1.0,
            "threshold_offset": -10,
            "min_area": 12,
            "max_area": 1800,
            "dilate_iterations": 1,
            "erode_iterations": 1,
        },
        "inclusion_bright_segmentation": {
            "invert_image": True,
            "gaussian_sigma": 2.0,
            "brightness_adjust": 0,
            "contrast_adjust": 1.0,
            "threshold_offset": -10,
            "min_area": 12,
            "max_area": 1800,
            "dilate_iterations": 1,
            "erode_iterations": 1,
        },
    },
    
    "Pseudomonas Aeruginosa": {
        "bacteria_segmentation": {
            "invert_image": False,
            "gaussian_sigma": 2.0,
            "brightness_adjust": 0,
            "contrast_adjust": 1.0,
            "threshold_offset": 0,
            "min_area": 18,
            "max_area": 3500,
            "dilate_iterations": 0,
            "erode_iterations": 0,
        },
        "inclusion_dark_segmentation": {
            "invert_image": False,
            "gaussian_sigma": 2.0,
            "brightness_adjust": 0,
            "contrast_adjust": 1.0,
            "threshold_offset": -12,
            "min_area": 10,
            "max_area": 1600,
            "dilate_iterations": 1,
            "erode_iterations": 1,
        },
        "inclusion_bright_segmentation": {
            "invert_image": True,
            "gaussian_sigma": 2.0,
            "brightness_adjust": 0,
            "contrast_adjust": 1.0,
            "threshold_offset": -12,
            "min_area": 10,
            "max_area": 1600,
            "dilate_iterations": 1,
            "erode_iterations": 1,
        },
    },
    
    "Salmonella": {
        "bacteria_segmentation": {
            "invert_image": False,
            "gaussian_sigma": 2.0,
            "brightness_adjust": 0,
            "contrast_adjust": 1.0,
            "threshold_offset": 0,
            "min_area": 20,
            "max_area": 4000,
            "dilate_iterations": 0,
            "erode_iterations": 0,
        },
        "inclusion_dark_segmentation": {
            "invert_image": False,
            "gaussian_sigma": 2.0,
            "brightness_adjust": 0,
            "contrast_adjust": 1.0,
            "threshold_offset": -10,
            "min_area": 10,
            "max_area": 1700,
            "dilate_iterations": 1,
            "erode_iterations": 1,
        },
        "inclusion_bright_segmentation": {
            "invert_image": True,
            "gaussian_sigma": 2.0,
            "brightness_adjust": 0,
            "contrast_adjust": 1.0,
            "threshold_offset": -10,
            "min_area": 10,
            "max_area": 1700,
            "dilate_iterations": 1,
            "erode_iterations": 1,
        },
    },
}


def get_bacteria_config(bacterium: str) -> dict:
    """
    Get configuration for a specific bacterium
    
    Args:
        bacterium: Name of the bacterium
    
    Returns:
        Dictionary containing bacteria configuration
    """
    return bacteria_configs.get(bacterium, {})


def get_segmentation_params(bacterium: str, structure: str, mode: str = "DARK") -> dict:
    """
    Get segmentation parameters for a specific bacterium, structure, and mode
    
    Args:
        bacterium: Name of the bacterium
        structure: Structure type ('bacteria' or 'inclusions')
        mode: Segmentation mode ('DARK' or 'BRIGHT') - only applies to inclusions
    
    Returns:
        Dictionary containing segmentation parameters
    """
    config = get_bacteria_config(bacterium)
    
    if structure == "bacteria":
        params_key = "bacteria_segmentation"
    else:  # inclusions
        params_key = f"inclusion_{mode.lower()}_segmentation"
    
    return config.get(params_key, {})


def list_available_bacteria() -> list:
    """
    Get list of all available bacteria configurations
    
    Returns:
        List of bacterium names
    """
    return list(bacteria_configs.keys())


def add_bacteria_config(bacterium: str, config: dict):
    """
    Add or update configuration for a bacterium
    
    Args:
        bacterium: Name of the bacterium
        config: Configuration dictionary
    """
    bacteria_configs[bacterium] = config
    print(f"✅ Configuration added/updated for: {bacterium}")


def validate_params(params: dict) -> bool:
    """
    Validate segmentation parameters
    
    Args:
        params: Parameter dictionary to validate
    
    Returns:
        True if valid, False otherwise
    """
    required_keys = [
        "invert_image",
        "gaussian_sigma",
        "brightness_adjust",
        "contrast_adjust",
        "threshold_offset",
        "min_area",
        "max_area",
        "dilate_iterations",
        "erode_iterations",
    ]
    
    for key in required_keys:
        if key not in params:
            print(f"❌ Missing required parameter: {key}")
            return False
    
    return True


# Example usage
if __name__ == "__main__":
    print("🦠 Available Bacteria Configurations:")
    for bacterium in list_available_bacteria():
        print(f"  - {bacterium}")
    
    print("\n📋 Example Configuration (Klebsiella Pneumoniae - Bacteria):")
    params = get_segmentation_params("Klebsiella Pneumoniae", "bacteria")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    print("\n📋 Example Configuration (Klebsiella Pneumoniae - Dark Inclusions):")
    params = get_segmentation_params("Klebsiella Pneumoniae", "inclusions", "DARK")
    for key, value in params.items():
        print(f"  {key}: {value}")