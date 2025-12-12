"""
Adapted Batch Segmentation Pipeline for Your Data Structure
Processes images from 'source/' directory with ch00/ch01 naming convention
"""

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from tqdm import tqdm
import xml.etree.ElementTree as ET
import re
import os

class AdaptedSegmentationPipeline:
    """
    Batch processing pipeline adapted for your data structure.
    """
    
    def __init__(self, bf_config: Optional[Dict] = None, fl_config: Optional[Dict] = None,
                 pipeline_config_path: str = 'config/pipeline_config.json'):
        """
        Initialize pipeline with segmentation modules.
        
        Parameters:
        -----------
        bf_config : dict, optional
            Bright-field segmentation configuration
        fl_config : dict, optional
            Fluorescence segmentation configuration
        pipeline_config_path : str
            Path to pipeline configuration file
        """
        from brightfield_segmentation import BrightFieldSegmentation
        from fluorescence_segmentation import FluorescenceSegmentation
        
        self.bf_segmenter = BrightFieldSegmentation(config=bf_config or {})
        self.fl_segmenter = FluorescenceSegmentation(config=fl_config or {})
        
        # Load pipeline configuration
        with open(pipeline_config_path, 'r') as f:
            self.pipeline_config = json.load(f)
        
        self.source_dir = Path(self.pipeline_config['source_directory'])
    
    def parse_metadata(self, xml_path: Path) -> float:
        """
        Extract pixel size from XML metadata.
        
        Parameters:
        -----------
        xml_path : Path
            Path to XML metadata file
            
        Returns:
        --------
        pixel_size_um : float
            Pixel size in micrometers
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Try multiple possible XPath locations for pixel size
            # Adjust based on actual XML structure
            possible_paths = [
                './/PixelSize',
                './/PhysicalSizeX',
                './/Image/Pixels/PhysicalSizeX',
                './/Calibration/PixelWidth'
            ]
            
            for path in possible_paths:
                pixel_size_elem = root.find(path)
                if pixel_size_elem is not None and pixel_size_elem.text is not None:
                    return float(pixel_size_elem.text)
            
            # If not found, try to find in attributes
            for elem in root.iter():
                if 'PhysicalSizeX' in elem.attrib:
                    return float(elem.attrib['PhysicalSizeX'])
            
            print(f"Warning: Pixel size not found in {xml_path}, using default 0.65 μm")
            return 0.65
            
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return 0.65
    
    def find_image_pairs(self, directory: Path) -> List[Tuple[Path, Path, Optional[Path], str]]:
        """
        Find matching bright-field and fluorescence image pairs.
        
        Parameters:
        -----------
        directory : Path
            Directory containing images
            
        Returns:
        --------
        pairs : List[Tuple[Path, Path, Optional[Path], str]]
            List of (bf_path, fl_path, metadata_path, image_id) tuples
        """
        bf_suffix = self.pipeline_config['file_patterns']['brightfield_suffix']
        fl_suffix = self.pipeline_config['file_patterns']['fluorescence_suffix']
        
        # Find all bright-field images
        bf_files = sorted(directory.glob(f'*{bf_suffix}'))
        
        pairs: List[Tuple[Path, Path, Optional[Path], str]] = []
        for bf_path in bf_files:
            # Construct fluorescence path
            fl_path = Path(str(bf_path).replace(bf_suffix, fl_suffix))
            
            if not fl_path.exists():
                print(f"Warning: No matching fluorescence image for {bf_path}")
                continue
            
            # Extract image ID (remove channel suffix)
            image_id = bf_path.stem.replace(bf_suffix.replace('.tif', ''), '')
            
            # Find metadata file
            metadata_dir = directory / 'MetaData'
            metadata_path: Optional[Path] = None
            
            if metadata_dir.exists():
                # Try to find matching XML file
                xml_files = list(metadata_dir.glob(f'{image_id}.xml'))
                if not xml_files:
                    # Try without exact match
                    base_name = image_id.rstrip('_0123456789')
                    xml_files = list(metadata_dir.glob(f'{base_name}*.xml'))
                
                metadata_path = xml_files[0] if xml_files else None
            
            pairs.append((bf_path, fl_path, metadata_path, image_id))
        
        return pairs
    
    def determine_group(self, directory_name: str) -> str:
        """
        Determine experimental group from directory name.
        
        Parameters:
        -----------
        directory_name : str
            Name of the directory
            
        Returns:
        --------
        group : str
            Group label (Control, Positive, or Negative)
        """
        group_mapping = self.pipeline_config['group_mapping']
        
        if directory_name in group_mapping['control']:
            return 'Control'
        elif directory_name in group_mapping['positive']:
            return 'Positive'
        elif directory_name in group_mapping['negative']:
            return 'Negative'
        else:
            return 'Unknown'
    
    def process_image_pair(self, bf_path: Path, fl_path: Path,
                          metadata_path: Optional[Path] = None,
                          image_id: str = '',
                          group: str = '') -> Optional[Dict]:
        """
        Process matched bright-field and fluorescence images.
        
        Parameters:
        -----------
        bf_path : Path
            Path to bright-field image
        fl_path : Path
            Path to fluorescence image
        metadata_path : Path, optional
            Path to metadata XML file
        image_id : str
            Image identifier
        group : str
            Experimental group label
            
        Returns:
        --------
        results : dict or None
            Combined processing results or None if error
        """
        # Load images
        bf_image = cv2.imread(str(bf_path), cv2.IMREAD_GRAYSCALE)
        fl_image = cv2.imread(str(fl_path), cv2.IMREAD_GRAYSCALE)
        
        if bf_image is None or fl_image is None:
            print(f"Error loading images: {bf_path} or {fl_path}")
            return None
        
        # Parse metadata for pixel size
        if metadata_path and metadata_path.exists():
            pixel_size_um = self.parse_metadata(metadata_path)
        else:
            pixel_size_um = 0.65  # Default value
        
        # Process bright-field
        bf_results = self.bf_segmenter.process_image(
            image=bf_image,
            pixel_size_um=pixel_size_um,
            image_id=image_id,
            group=group
        )
        
        # Process fluorescence with particle mask for colocalization
        fl_results = self.fl_segmenter.process_image(
            image=fl_image,
            particle_mask=bf_results['labeled_mask'],
            pixel_size_um=pixel_size_um,
            image_id=image_id,
            group=group
        )
        
        combined_results: Dict = {
            'image_id': image_id,
            'group': group,
            'pixel_size_um': pixel_size_um,
            'bright_field': bf_results,
            'fluorescence': fl_results
        }
        
        return combined_results
    
    def process_directory(self, directory: Path, output_dir: Path,
                         group_label: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Batch process all images in a directory.
        
        Parameters:
        -----------
        directory : Path
            Directory containing images
        output_dir : Path
            Directory for output files
        group_label : str, optional
            Optional group label override
            
        Returns:
        --------
        bf_measurements, fl_measurements : Tuple[pd.DataFrame, pd.DataFrame]
            Measurement data frames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (output_dir / 'bf_masks').mkdir(exist_ok=True)
        (output_dir / 'fl_masks').mkdir(exist_ok=True)
        (output_dir / 'visualizations').mkdir(exist_ok=True)
        (output_dir / 'measurements').mkdir(exist_ok=True)
        (output_dir / 'logs').mkdir(exist_ok=True)
        
        # Determine group
        if group_label is None:
            group_label = self.determine_group(directory.name)
        
        # Find image pairs
        image_pairs = self.find_image_pairs(directory)
        
        if not image_pairs:
            print(f"No image pairs found in {directory}")
            return pd.DataFrame(), pd.DataFrame()
        
        all_bf_measurements: List[pd.DataFrame] = []
        all_fl_measurements: List[pd.DataFrame] = []
        
        print(f"\nProcessing {group_label} group: {directory.name}")
        for bf_path, fl_path, metadata_path, image_id in tqdm(image_pairs):
            
            # Process image pair
            results = self.process_image_pair(
                bf_path=bf_path,
                fl_path=fl_path,
                metadata_path=metadata_path,
                image_id=image_id,
                group=group_label
            )
            
            if results is None:
                continue
            
            # Save masks
            safe_id = image_id.replace(' ', '_').replace('+', 'plus')
            cv2.imwrite(
                str(output_dir / 'bf_masks' / f'{safe_id}_BF_mask.tif'),
                results['bright_field']['labeled_mask'].astype(np.uint16)
            )
            cv2.imwrite(
                str(output_dir / 'fl_masks' / f'{safe_id}_FL_mask.tif'),
                results['fluorescence']['labeled_mask'].astype(np.uint16)
            )
            
            # Load images for visualization
            bf_image = cv2.imread(str(bf_path), cv2.IMREAD_GRAYSCALE)
            fl_image = cv2.imread(str(fl_path), cv2.IMREAD_GRAYSCALE)
            
            # Save visualizations
            if bf_image is not None:
                self.bf_segmenter.visualize_segmentation(
                    original_image=bf_image,
                    labeled_mask=results['bright_field']['labeled_mask'],
                    quality_df=results['bright_field']['quality_scores'],
                    output_path=str(output_dir / 'visualizations' / f'{safe_id}_BF_seg.png')
                )
            
            if fl_image is not None:
                self.fl_segmenter.visualize_fluorescence(
                    original_image=fl_image,
                    labeled_mask=results['fluorescence']['labeled_mask'],
                    measurements=results['fluorescence']['measurements'],
                    output_path=str(output_dir / 'visualizations' / f'{safe_id}_FL_seg.png')
                )
            
            # Collect measurements
            all_bf_measurements.append(results['bright_field']['measurements'])
            all_fl_measurements.append(results['fluorescence']['measurements'])
        
        # Combine measurements
        if all_bf_measurements:
            bf_measurements_df = pd.concat(all_bf_measurements, ignore_index=True)
            fl_measurements_df = pd.concat(all_fl_measurements, ignore_index=True)
            
            # Save measurement tables
            bf_measurements_df.to_csv(
                output_dir / 'measurements' / f'{group_label}_bf_measurements.csv',
                index=False
            )
            fl_measurements_df.to_csv(
                output_dir / 'measurements' / f'{group_label}_fl_measurements.csv',
                index=False
            )
            
            print(f"  Total BF particles: {len(bf_measurements_df)}")
            print(f"  Total FL regions: {len(fl_measurements_df)}")
            
            return bf_measurements_df, fl_measurements_df
        else:
            return pd.DataFrame(), pd.DataFrame()
    
    def process_all_groups(self, output_base_dir: Path = Path('results')):
        """
        Process all groups (control, positive, negative) from source directory.
        
        Parameters:
        -----------
        output_base_dir : Path
            Base directory for all results
        """
        all_results: Dict[str, Dict[str, List[pd.DataFrame]]] = {
            'control': {'bf': [], 'fl': []},
            'positive': {'bf': [], 'fl': []},
            'negative': {'bf': [], 'fl': []}
        }
        
        # Process each subdirectory in source
        for subdir in sorted(self.source_dir.iterdir()):
            if not subdir.is_dir():
                continue
            
            group = self.determine_group(subdir.name)
            
            if group == 'Control':
                output_dir = output_base_dir / 'control'
            elif group == 'Positive':
                output_dir = output_base_dir / 'positive'
            elif group == 'Negative':
                output_dir = output_base_dir / 'negative'
            else:
                print(f"Skipping unknown group: {subdir.name}")
                continue
            
            # Process directory
            bf_df, fl_df = self.process_directory(
                directory=subdir,
                output_dir=output_dir,
                group_label=group
            )
            
            if not bf_df.empty:
                all_results[group.lower()]['bf'].append(bf_df)
                all_results[group.lower()]['fl'].append(fl_df)
        
        # Combine all results
        combined_dir = output_base_dir / 'combined'
        combined_dir.mkdir(parents=True, exist_ok=True)
        
        for group_name, data in all_results.items():
            if data['bf']:
                group_bf = pd.concat(data['bf'], ignore_index=True)
                group_fl = pd.concat(data['fl'], ignore_index=True)
                
                group_bf.to_csv(
                    combined_dir / f'{group_name}_all_bf_measurements.csv',
                    index=False
                )
                group_fl.to_csv(
                    combined_dir / f'{group_name}_all_fl_measurements.csv',
                    index=False
                )
        
        # Create overall combined file
        all_bf: List[pd.DataFrame] = []
        all_fl: List[pd.DataFrame] = []
        for group_data in all_results.values():
            all_bf.extend(group_data['bf'])
            all_fl.extend(group_data['fl'])
        
        if all_bf:
            final_bf = pd.concat(all_bf, ignore_index=True)
            final_fl = pd.concat(all_fl, ignore_index=True)
            
            final_bf.to_csv(combined_dir / 'all_bf_measurements.csv', index=False)
            final_fl.to_csv(combined_dir / 'all_fl_measurements.csv', index=False)
            
            print("\n" + "="*60)
            print("COMPLETE ANALYSIS SUMMARY")
            print("="*60)
            print(f"Total images processed: {final_bf['Image_ID'].nunique()}")
            print(f"Total BF particles: {len(final_bf)}")
            print(f"Total FL regions: {len(final_fl)}")
            print("\nBy Group:")
            print(final_bf.groupby('Group').size())
            print("="*60)


# Main execution script
def main():
    """Main execution function."""
    
    # Load configurations
    from utils import load_config
    
    bf_config = load_config('config/brightfield_config.json')
    fl_config = load_config('config/fluorescence_config.json')
    
    # Initialize pipeline
    pipeline = AdaptedSegmentationPipeline(
        bf_config=bf_config,
        fl_config=fl_config,
        pipeline_config_path='config/pipeline_config.json'
    )
    
    # Process all groups
    pipeline.process_all_groups(output_base_dir=Path('results'))


if __name__ == "__main__":
    main()