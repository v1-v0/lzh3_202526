"""
Fluorescence Segmentation Module
Segments and quantifies fluorescence signals in microscopy images
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology, measure, filters, exposure, segmentation
from typing import Tuple, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt


class FluorescenceSegmentation:
    """
    Handles segmentation and quantification of fluorescence microscopy images.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the fluorescence segmentation module.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for segmentation
        """
        self.config = config or {}
        
        # Default parameters (can be overridden by config)
        self.params = {
            'gaussian_sigma': self.config.get('gaussian_sigma', 1.5),
            'threshold_method': self.config.get('threshold_method', 'li'),
            'min_signal_size': self.config.get('min_signal_size', 50),
            'background_percentile': self.config.get('background_percentile', 5),
            'local_background_radius': self.config.get('local_background_radius', 50)
        }
    
    def preprocess_fluorescence(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Preprocess fluorescence image.
        
        Parameters:
        -----------
        image : np.ndarray
            Input fluorescence image
            
        Returns:
        --------
        preprocessed : np.ndarray
            Preprocessed image
        background : float
            Estimated background level
        """
        # Ensure image is grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Convert to float for processing
        image_float = image.astype(np.float32)
        
        # Apply Gaussian blur to reduce noise
        sigma = self.params['gaussian_sigma']
        blurred = filters.gaussian(image_float, sigma=sigma)
        
        # Estimate background
        background = float(np.percentile(blurred[blurred > 0], self.params['background_percentile']))
        
        # Subtract background
        background_subtracted = np.maximum(blurred - background, 0)
        
        # Normalize to 0-255 range
        if background_subtracted.max() > 0:
            normalized = exposure.rescale_intensity(
                background_subtracted,
                in_range='image',
                out_range='dtype'
            ).astype(np.uint8)
        else:
            normalized = background_subtracted.astype(np.uint8)
        
        return normalized, background
    
    def create_fluorescence_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create binary mask of fluorescence signals.
        
        Parameters:
        -----------
        image : np.ndarray
            Preprocessed fluorescence image
            
        Returns:
        --------
        binary_mask : np.ndarray
            Binary mask of fluorescence signals
        """
        # Apply threshold
        if self.params['threshold_method'] == 'li':
            threshold_value = filters.threshold_li(image)
        elif self.params['threshold_method'] == 'otsu':
            threshold_value = filters.threshold_otsu(image)
        elif self.params['threshold_method'] == 'yen':
            threshold_value = filters.threshold_yen(image)
        else:
            # Default to Li
            threshold_value = filters.threshold_li(image)
        
        binary_mask = image > threshold_value
        
        # Morphological operations to clean up mask
        # Remove small objects
        binary_mask = morphology.remove_small_objects(
            binary_mask, min_size=self.params['min_signal_size']
        )
        
        # Fill small holes
        binary_mask = morphology.remove_small_holes(
            binary_mask, area_threshold=64
        )
        
        return binary_mask.astype(np.uint8)
    
    def segment_signals(self, binary_mask: np.ndarray,
                       fluorescence_image: np.ndarray) -> np.ndarray:
        """
        Segment individual fluorescence signals.
        
        Parameters:
        -----------
        binary_mask : np.ndarray
            Binary mask of fluorescence
        fluorescence_image : np.ndarray
            Original fluorescence image
            
        Returns:
        --------
        labeled_mask : np.ndarray
            Labeled mask with individual signals
        """
        # Distance transform
        distance = ndimage.distance_transform_edt(binary_mask)
        distance = np.asarray(distance, dtype=np.float64)
        
        # Find local maxima as markers
        local_max = morphology.local_maxima(distance)
        local_max = np.asarray(local_max, dtype=bool)
        
        # Label the markers
        markers = measure.label(local_max)
        markers = np.asarray(markers, dtype=np.int32)
        
        # Apply watershed
        labeled_mask = segmentation.watershed(-distance, markers, mask=binary_mask.astype(bool))
        labeled_mask = np.asarray(labeled_mask, dtype=np.int32)
        
        return labeled_mask
    
    def calculate_local_background(self, image: np.ndarray,
                                   labeled_mask: np.ndarray,
                                   region_id: int) -> float:
        """
        Calculate local background around a specific region.
        
        Parameters:
        -----------
        image : np.ndarray
            Original fluorescence image
        labeled_mask : np.ndarray
            Labeled mask
        region_id : int
            Region identifier
            
        Returns:
        --------
        local_bg : float
            Local background value
        """
        # Get region mask
        region_mask = (labeled_mask == region_id)
        
        # Dilate region to get surrounding area
        radius = self.params['local_background_radius']
        dilated = morphology.dilation(region_mask, morphology.disk(radius))
        
        # Background is dilated area minus the region itself
        bg_mask = dilated & ~region_mask
        
        if bg_mask.sum() > 0:
            local_bg = float(np.median(image[bg_mask]))
        else:
            local_bg = 0.0
        
        return local_bg
    
    def measure_fluorescence(self, labeled_mask: np.ndarray,
                           fluorescence_image: np.ndarray,
                           background: float,
                           pixel_size_um: float = 0.65) -> pd.DataFrame:
        """
        Measure fluorescence properties of segmented signals.
        
        Parameters:
        -----------
        labeled_mask : np.ndarray
            Labeled mask with individual signals
        fluorescence_image : np.ndarray
            Original fluorescence image
        background : float
            Global background level
        pixel_size_um : float
            Pixel size in micrometers
            
        Returns:
        --------
        measurements : pd.DataFrame
            DataFrame containing fluorescence measurements
        """
        # Get region properties
        props = measure.regionprops_table(
            labeled_mask,
            intensity_image=fluorescence_image,
            properties=[
                'label', 'area', 'centroid',
                'mean_intensity', 'max_intensity', 'min_intensity'
            ]
        )
        
        df = pd.DataFrame(props)
        
        # Calculate integrated intensity immediately after creating DataFrame
        df['Raw_Integrated_Intensity'] = df['mean_intensity'] * df['area']


        # Convert area to micrometers
        df['Area_um2'] = df['area'] * (pixel_size_um ** 2)
        
        # Calculate background-corrected measurements
        df['Global_Background'] = background
        
        # Calculate local background for each region
        local_backgrounds = []
        for region_id in df['label']:
            local_bg = self.calculate_local_background(
                fluorescence_image, labeled_mask, int(region_id)
            )
            local_backgrounds.append(local_bg)
        
        df['Local_Background'] = local_backgrounds
        
        # Corrected intensities
        df['Corrected_Mean_Intensity'] = df['mean_intensity'] - df['Local_Background']
        df['Corrected_Integrated_Intensity'] = (
            df['Raw_Integrated_Intensity'] - df['Local_Background'] * df['area']
        )
        
        # Signal-to-background ratio
        df['Signal_to_Background'] = df['mean_intensity'] / (df['Local_Background'] + 1e-10)
        
        # Rename columns for clarity
        df = df.rename(columns={
            'label': 'Signal_ID',
            'centroid-0': 'Centroid_Y',
            'centroid-1': 'Centroid_X',
            'mean_intensity': 'Raw_Mean_Intensity',
            'max_intensity': 'Max_Intensity',
            'min_intensity': 'Min_Intensity'
        })
        
        df = df.rename(columns={'integrated_intensity': 'Raw_Integrated_Intensity'})

        return df
    
    def colocalize_with_particles(self, fluorescence_mask: np.ndarray,
                                 particle_mask: np.ndarray,
                                 fluorescence_measurements: pd.DataFrame) -> pd.DataFrame:
        """
        Determine colocalization between fluorescence signals and particles.
        
        Parameters:
        -----------
        fluorescence_mask : np.ndarray
            Labeled fluorescence mask
        particle_mask : np.ndarray
            Labeled particle mask from bright-field
        fluorescence_measurements : pd.DataFrame
            Fluorescence measurements
            
        Returns:
        --------
        colocalization : pd.DataFrame
            DataFrame with colocalization information
        """
        colocalization_data = []
        
        for _, row in fluorescence_measurements.iterrows():
            signal_id = int(row['Signal_ID'])
            
            # Get fluorescence signal mask
            signal_mask = (fluorescence_mask == signal_id)
            
            # Find overlapping particles
            overlapping_particles = particle_mask[signal_mask]
            unique_particles = np.unique(overlapping_particles[overlapping_particles > 0])
            
            # Calculate overlap statistics
            if len(unique_particles) > 0:
                # Find dominant particle (most overlap)
                overlap_counts = [(p, np.sum(overlapping_particles == p)) 
                                for p in unique_particles]
                dominant_particle = max(overlap_counts, key=lambda x: x[1])[0]
                overlap_fraction = max(overlap_counts, key=lambda x: x[1])[1] / signal_mask.sum()
                
                colocalization_data.append({
                    'Signal_ID': signal_id,
                    'Colocalized': True,
                    'Particle_ID': int(dominant_particle),
                    'Overlap_Fraction': float(overlap_fraction),
                    'Num_Overlapping_Particles': len(unique_particles)
                })
            else:
                colocalization_data.append({
                    'Signal_ID': signal_id,
                    'Colocalized': False,
                    'Particle_ID': -1,
                    'Overlap_Fraction': 0.0,
                    'Num_Overlapping_Particles': 0
                })
        
        return pd.DataFrame(colocalization_data)
    
    def process_image(self, image: np.ndarray,
                     particle_mask: Optional[np.ndarray] = None,
                     pixel_size_um: float = 0.65,
                     image_id: str = '',
                     group: str = '') -> Dict:
        """
        Complete processing pipeline for fluorescence image.
        
        Parameters:
        -----------
        image : np.ndarray
            Input fluorescence image
        particle_mask : np.ndarray, optional
            Labeled particle mask from bright-field for colocalization
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
        preprocessed, background = self.preprocess_fluorescence(image)
        
        # Create binary mask
        binary_mask = self.create_fluorescence_mask(preprocessed)
        
        # Segment signals
        labeled_mask = self.segment_signals(binary_mask, preprocessed)
        
        # Measure fluorescence
        measurements = self.measure_fluorescence(
            labeled_mask, preprocessed, background, pixel_size_um
        )
        
        # Add metadata
        measurements['Image_ID'] = image_id
        measurements['Group'] = group
        measurements['Pixel_Size_um'] = pixel_size_um
        
        # Colocalization if particle mask provided
        colocalization = None
        if particle_mask is not None:
            colocalization = self.colocalize_with_particles(
                labeled_mask, particle_mask, measurements
            )
        
        # Package results
        results = {
            'preprocessed_image': preprocessed,
            'binary_mask': binary_mask,
            'labeled_mask': labeled_mask,
            'measurements': measurements,
            'colocalization': colocalization,
            'background': background,
            'num_signals': len(measurements)
        }
        
        return results
    
    def visualize_fluorescence(self, original_image: np.ndarray,
                              labeled_mask: np.ndarray,
                              measurements: pd.DataFrame,
                              output_path: Optional[str] = None):
        """
        Visualize fluorescence segmentation results.
        
        Parameters:
        -----------
        original_image : np.ndarray
            Original fluorescence image
        labeled_mask : np.ndarray
            Labeled segmentation mask
        measurements : pd.DataFrame
            Fluorescence measurements
        output_path : str, optional
            Path to save visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Original image
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original Fluorescence')
        axes[0, 0].axis('off')
        
        # Labeled mask
        axes[0, 1].imshow(labeled_mask, cmap='nipy_spectral')
        axes[0, 1].set_title(f'Segmented Signals (n={labeled_mask.max()})')
        axes[0, 1].axis('off')
        
        # Overlay with intensity labels
        axes[1, 0].imshow(original_image, cmap='gray')
        for _, row in measurements.iterrows():
            y, x = row['Centroid_Y'], row['Centroid_X']
            intensity = row['Corrected_Integrated_Intensity']
            axes[1, 0].plot(x, y, 'r+', markersize=10)
            axes[1, 0].text(x, y, f'{intensity:.0f}', 
                          color='yellow', fontsize=8)
        axes[1, 0].set_title('Signal Locations & Intensities')
        axes[1, 0].axis('off')
        
        # Histogram of corrected integrated intensities
        axes[1, 1].hist(measurements['Corrected_Integrated_Intensity'], 
                       bins=30, edgecolor='black')
        axes[1, 1].set_xlabel('Corrected Integrated Intensity')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Intensity Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# Example usage and testing
def main():
    """Main function for testing."""
    import os
    
    # Example configuration
    config = {
        'gaussian_sigma': 1.5,
        'threshold_method': 'li',
        'min_signal_size': 50,
        'background_percentile': 5,
        'local_background_radius': 50
    }
    
    # Initialize segmentation module
    segmenter = FluorescenceSegmentation(config=config)
    
    # Example: Process a test image
    test_image_path = 'test_data/sample_fluorescence.tif'
    
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
            segmenter.visualize_fluorescence(
                original_image=image,
                labeled_mask=results['labeled_mask'],
                measurements=results['measurements'],
                output_path='test_output/fl_segmentation.png'
            )
            
            # Print summary
            print(f"Number of signals detected: {results['num_signals']}")
            print(f"Background level: {results['background']:.2f}")
            print("\nMeasurements summary:")
            print(results['measurements'].describe())
        else:
            print(f"Error: Could not load image from {test_image_path}")
    else:
        print(f"Test image not found: {test_image_path}")


if __name__ == "__main__":
    main()