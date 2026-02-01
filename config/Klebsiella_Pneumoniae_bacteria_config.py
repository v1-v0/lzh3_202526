
# Klebsiella Pneumoniae - bacteria
# Auto-generated from tuning session
# Generated: 2026-02-01 23:10:40
# Source image: G+ microgel control 1_ch00.tif

KLEBSIELLA_PNEUMONIAE = SegmentationConfig(
    name="Klebsiella Pneumoniae",
    description="Tuned configuration for Klebsiella Pneumoniae - bacteria",
    
    # Preprocessing
    gaussian_sigma=1.8,
    brightness_adjust=0.0,
    contrast_adjust=1.00,
    threshold_offset=-29.0,
    
    # Morphology
    morph_kernel_size=3,
    morph_iterations=1,
    dilate_iterations=0,
    erode_iterations=1,
    
    # Filtering
    min_area_um2=217.6,
    max_area_um2=5013.0,
    min_circularity=0.00,
    max_circularity=1.0,
    min_aspect_ratio=0.1,
    max_aspect_ratio=10.0,
    min_solidity=0.5,
    max_fraction_of_image=0.25,
    min_mean_intensity=0,
    max_mean_intensity=255,
    max_edge_gradient=30.0,
    
    # Fluorescence
    fluor_gaussian_sigma=1.5,
    fluor_morph_kernel_size=3,
    fluor_min_area_um2=3.0,
    fluor_match_min_intersection_px=5.0,
    
    # Inversion
    invert_image=True,
)
