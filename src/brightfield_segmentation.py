"""
Bright-field Segmentation Module
Segments individual particles from bright-field microscopy images
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology, measure, filters, segmentation
from typing import Tuple, Dict, Optional, cast
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray


class BrightFieldSegmentation:
    """
    Handles segmentation of bright-field microscopy images to identify
    individual particles.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the bright-field segmentation module.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for segmentation
        """
        self.config = config if config is not None else {}
        
        # Default parameters (can be overridden by config)
        self.params = {
            'gaussian_sigma': self.config.get('gaussian_sigma', 2.0),
            'threshold_method': self.config.get('threshold_method', 'otsu'),
            'min_particle_size': self.config.get('min_particle_size', 100),
            'max_particle_size': self.config.get('max_particle_size', 10000),
            'circularity_threshold': self.config.get('circularity_threshold', 0.3),
            'border_clearance': self.config.get('border_clearance', True)
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the bright-field image.
        
        Parameters:
        -----------
        image : np.ndarray
            Input grayscale image
            
        Returns:
        --------
        preprocessed : np.ndarray
            Preprocessed image
        """
        # Ensure image is grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        sigma = self.params['gaussian_sigma']
        blurred = filters.gaussian(image, sigma=sigma)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply((blurred * 255).astype(np.uint8))
        
        return enhanced
    
    def create_binary_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create binary mask from preprocessed image.
        
        Parameters:
        -----------
        image : np.ndarray
            Preprocessed image
            
        Returns:
        --------
        binary_mask : np.ndarray
            Binary mask of particles
        """
        # Apply threshold
        if self.params['threshold_method'] == 'otsu':
            _, binary = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        elif self.params['threshold_method'] == 'adaptive':
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
        else:
            # Default to Otsu
            _, binary = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        
        # Convert to boolean
        binary_mask = binary > 0
        
        # Morphological operations to clean up mask
        # Remove small holes
        binary_mask = morphology.remove_small_holes(
            binary_mask, area_threshold=64
        )
        
        # Remove small objects
        binary_mask = morphology.remove_small_objects(
            binary_mask, min_size=self.params['min_particle_size']
        )
        
        # Optional: Clear border objects
        if self.params['border_clearance']:
            binary_mask = segmentation.clear_border(binary_mask)
        
        return binary_mask.astype(np.uint8)
    
    def watershed_segmentation(self, binary_mask: np.ndarray,
                          original_image: np.ndarray) -> np.ndarray:
        """
        Apply watershed segmentation to separate touching particles.
        
        Parameters:
        -----------
        binary_mask : np.ndarray
            Binary mask of particles
        original_image : np.ndarray
            Original grayscale image
            
        Returns:
        --------
        labeled_mask : np.ndarray
            Labeled mask with individual particles
        """
        # Ensure mask is boolean
        mask_bool = binary_mask.astype(bool)
        
        # Distance transform
        distance = ndimage.distance_transform_edt(mask_bool)
        distance = np.asarray(distance, dtype=np.float64)
        
        # Find local maxima as markers
        local_max = morphology.local_maxima(distance)
        local_max = np.asarray(local_max, dtype=bool)
        
        # Label the markers
        markers = measure.label(local_max)
        markers = np.asarray(markers, dtype=np.int32)
        
        # Apply watershed
        labeled_mask = segmentation.watershed(-distance, markers, mask=mask_bool)
        labeled_mask = np.asarray(labeled_mask, dtype=np.int32)
        
        return labeled_mask
    
    
    def measure_particles(self, labeled_mask: np.ndarray,
                         original_image: np.ndarray,
                         pixel_size_um: float = 0.65) -> pd.DataFrame:
        """
        Measure properties of segmented particles.
        
        Parameters:
        -----------
        labeled_mask : np.ndarray
            Labeled mask with individual particles
        original_image : np.ndarray
            Original grayscale image
        pixel_size_um : float
            Pixel size in micrometers (default: 0.65)
            
        Returns:
        --------
        measurements : pd.DataFrame
            DataFrame containing particle measurements
        """
        # Get region properties
        props = measure.regionprops_table(
            labeled_mask,
            intensity_image=original_image,
            properties=[
                'label', 'area', 'perimeter', 'centroid',
                'major_axis_length', 'minor_axis_length',
                'eccentricity', 'solidity',
                'mean_intensity', 'max_intensity', 'min_intensity'
            ]
        )
        
        df = pd.DataFrame(props)
        
        # Convert pixel measurements to micrometers
        df['Area_um2'] = df['area'] * (pixel_size_um ** 2)
        df['Perimeter_um'] = df['perimeter'] * pixel_size_um
        df['Major_Axis_um'] = df['major_axis_length'] * pixel_size_um
        df['Minor_Axis_um'] = df['minor_axis_length'] * pixel_size_um
        
        # Calculate derived measurements
        df['Circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2 + 1e-10)
        df['Aspect_Ratio'] = df['major_axis_length'] / (df['minor_axis_length'] + 1e-10)
        df['Equivalent_Diameter_um'] = 2 * np.sqrt(df['Area_um2'] / np.pi)
        
        # Rename columns for clarity
        df = df.rename(columns={
            'label': 'Particle_ID',
            'centroid-0': 'Centroid_Y',
            'centroid-1': 'Centroid_X',
            'mean_intensity': 'Mean_Intensity',
            'max_intensity': 'Max_Intensity',
            'min_intensity': 'Min_Intensity',
            'eccentricity': 'Eccentricity',
            'solidity': 'Solidity'
        })
        
        return df
    
    def filter_particles(self, measurements: pd.DataFrame,
                        labeled_mask: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Filter particles based on quality criteria.
        
        Parameters:
        -----------
        measurements : pd.DataFrame
            Particle measurements
        labeled_mask : np.ndarray
            Labeled mask
            
        Returns:
        --------
        filtered_mask : np.ndarray
            Filtered labeled mask
        quality_scores : pd.DataFrame
            Quality scores for each particle
        """
        # Define quality criteria
        size_valid = (
            (measurements['Area_um2'] >= self.params['min_particle_size'] * 0.65**2) &
            (measurements['Area_um2'] <= self.params['max_particle_size'] * 0.65**2)
        )
        
        circularity_valid = measurements['Circularity'] >= self.params['circularity_threshold']
        solidity_valid = measurements['Solidity'] >= 0.8
        
        # Combine criteria
        valid_particles = size_valid & circularity_valid & solidity_valid
        
        # Create quality scores DataFrame
        quality_scores = pd.DataFrame({
            'Particle_ID': measurements['Particle_ID'],
            'Size_Valid': size_valid,
            'Circularity_Valid': circularity_valid,
            'Solidity_Valid': solidity_valid,
            'Overall_Valid': valid_particles
        })
        
        # Filter labeled mask - get valid particle IDs as list
        filtered_mask = np.zeros_like(labeled_mask)
        particle_series = cast(pd.Series, measurements.loc[valid_particles, 'Particle_ID'])
        valid_ids = particle_series.to_list()
        
        for particle_id in valid_ids:
            filtered_mask[labeled_mask == particle_id] = particle_id
        
        return filtered_mask, quality_scores
    
    def process_image(self, image: np.ndarray,
                     pixel_size_um: float = 0.65,
                     image_id: str = '',
                     group: str = '') -> Dict:
        """
        Complete processing pipeline for a single bright-field image.
        
        Parameters:
        -----------
        image : np.ndarray
            Input bright-field image
        pixel_size_um : float
            Pixel size in micrometers
        image_id : str
            Image identifier
        group : str
            Experimental group label
            
        Returns:
        --------
        results : dict
            Dictionary containing all processing results
        """
        # Preprocess
        preprocessed = self.preprocess_image(image)
        
        # Create binary mask
        binary_mask = self.create_binary_mask(preprocessed)
        
        # Watershed segmentation
        labeled_mask = self.watershed_segmentation(binary_mask, preprocessed)
        
        # Measure particles
        measurements = self.measure_particles(
            labeled_mask, preprocessed, pixel_size_um
        )
        
        # Filter particles
        filtered_mask, quality_scores = self.filter_particles(
            measurements, labeled_mask
        )
        
        # Add metadata to measurements
        measurements['Image_ID'] = image_id
        measurements['Group'] = group
        measurements['Pixel_Size_um'] = pixel_size_um
        
        # Package results
        results = {
            'preprocessed_image': preprocessed,
            'binary_mask': binary_mask,
            'labeled_mask': filtered_mask,
            'measurements': measurements,
            'quality_scores': quality_scores,
            'num_particles': len(measurements)
        }
        
        return results
    
    def visualize_segmentation(self, original_image: np.ndarray,
                              labeled_mask: np.ndarray,
                              quality_df: Optional[pd.DataFrame] = None,
                              output_path: Optional[str] = None):
        """
        Visualize segmentation results.
        
        Parameters:
        -----------
        original_image : np.ndarray
            Original image
        labeled_mask : np.ndarray
            Labeled segmentation mask
        quality_df : pd.DataFrame, optional
            Quality scores DataFrame
        output_path : str, optional
            Path to save visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Labeled mask
        axes[1].imshow(labeled_mask, cmap='nipy_spectral')
        axes[1].set_title(f'Segmented Particles (n={labeled_mask.max()})')
        axes[1].axis('off')
        
        # Overlay
        overlay = np.stack([original_image] * 3, axis=-1)
        overlay = (overlay / overlay.max() * 255).astype(np.uint8)
        
        # Draw contours
        for region_id in np.unique(labeled_mask)[1:]:
            mask = (labeled_mask == region_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Color based on quality
            if quality_df is not None and region_id in quality_df['Particle_ID'].values:
                is_valid = quality_df[
                    quality_df['Particle_ID'] == region_id
                ]['Overall_Valid'].iloc[0]
                color = (0, 255, 0) if is_valid else (255, 0, 0)
            else:
                color = (0, 255, 0)
            
            cv2.drawContours(overlay, contours, -1, color, 2)
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (Green=Valid, Red=Invalid)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_summary_statistics(self, measurements: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for particle measurements.
        
        Parameters:
        -----------
        measurements : pd.DataFrame
            Particle measurements
            
        Returns:
        --------
        summary : pd.DataFrame
            Summary statistics
        """
        numeric_cols = measurements.select_dtypes(include=[np.number]).columns
        summary = measurements[numeric_cols].describe()
        
        return summary


# Example usage and testing
def main():
    """Main function for testing."""
    import os
    
    # Example configuration
    config = {
        'gaussian_sigma': 2.0,
        'threshold_method': 'otsu',
        'min_particle_size': 100,
        'max_particle_size': 10000,
        'circularity_threshold': 0.3,
        'border_clearance': True
    }
    
    # Initialize segmentation module
    segmenter = BrightFieldSegmentation(config=config)
    
    # Example: Process a test image
    test_image_path = 'test_data/sample_brightfield.tif'
    
    if os.path.exists(test_image_path):
        # Load image
        image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is not None:
            # Process image
            results = segmenter.process_image(
                image=image,
                pixel_size_um=0.65,
                image_id='test_image',
                group='test'
            )
            
            # Visualize results
            segmenter.visualize_segmentation(
                original_image=image,
                labeled_mask=results['labeled_mask'],
                quality_df=results['quality_scores'],
                output_path='test_output/bf_segmentation.png'
            )
            
            # Print summary
            print(f"Number of particles detected: {results['num_particles']}")
            print("\nMeasurements summary:")
            print(results['measurements'].describe())
        else:
            print(f"Error: Could not load image from {test_image_path}")
    else:
        print(f"Test image not found: {test_image_path}")


if __name__ == "__main__":
    main()