import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import argparse
import cv2
import shutil

from brightfield_segmentation import BrightFieldSegmentation
from fluorescence_segmentation import FluorescenceSegmentation


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


class AdaptedSegmentationPipeline:
    """
    Pipeline for processing brightfield and fluorescence microscopy images.
    Handles batch processing of multiple experimental groups.
    """
    
    def __init__(self, bf_config: Dict, fl_config: Dict, pipeline_config: Dict):
        """
        Initialize the pipeline with configurations.
        
        Args:
            bf_config: Brightfield segmentation configuration
            fl_config: Fluorescence segmentation configuration
            pipeline_config: Pipeline configuration including source directory and group mapping
        """
        self.bf_segmenter = BrightFieldSegmentation(bf_config)
        self.fl_segmenter = FluorescenceSegmentation(fl_config)
        
        self.source_dir = Path(pipeline_config['source_directory'])
        self.file_patterns = pipeline_config['file_patterns']
        self.group_mapping = pipeline_config['group_mapping']
        
    def find_image_pairs(self, directory: Path) -> List[Tuple[Path, Path]]:
        """
        Find matching brightfield and fluorescence image pairs in a directory.
        
        Args:
            directory: Directory to search for image pairs
            
        Returns:
            List of (brightfield_path, fluorescence_path) tuples
        """
        bf_suffix = self.file_patterns['brightfield_suffix']
        fl_suffix = self.file_patterns['fluorescence_suffix']
        
        # Find all brightfield images
        bf_images = sorted(directory.glob(f'*{bf_suffix}'))
        
        pairs = []
        for bf_path in bf_images:
            # Construct expected fluorescence filename
            base_name = bf_path.name.replace(bf_suffix, '')
            fl_path = directory / f'{base_name}{fl_suffix}'
            
            if fl_path.exists():
                pairs.append((bf_path, fl_path))
            else:
                print(f"Warning: No matching fluorescence image for {bf_path.name}")
        
        return pairs
    
    def process_image_pair(
        self, 
        bf_path: Path, 
        fl_path: Path,
        pixel_size_um: float = 0.65
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """
        Process a single pair of brightfield and fluorescence images.
        
        Args:
            bf_path: Path to brightfield image
            fl_path: Path to fluorescence image
            pixel_size_um: Pixel size in micrometers
            
        Returns:
            Tuple of (brightfield_results_df, fluorescence_results_df)
        """
        # Load images
        bf_image = cv2.imread(str(bf_path), cv2.IMREAD_GRAYSCALE)
        fl_image = cv2.imread(str(fl_path), cv2.IMREAD_GRAYSCALE)
        
        # Check if images loaded successfully
        if bf_image is None:
            raise ValueError(f"Failed to load brightfield image: {bf_path}")
        if fl_image is None:
            raise ValueError(f"Failed to load fluorescence image: {fl_path}")
        
        image_name = bf_path.stem.replace(self.file_patterns['brightfield_suffix'].replace('.tif', ''), '')
        
        # Process brightfield image
        bf_results = self.bf_segmenter.process_image(
            image=bf_image,
            pixel_size_um=pixel_size_um,
            image_id=image_name
        )
        
        # Process fluorescence image
        fl_results = self.fl_segmenter.process_image(
            image=fl_image,
            pixel_size_um=pixel_size_um,
            image_id=image_name
        )
        
        return bf_results['measurements'], fl_results['measurements'], bf_results, fl_results
    
    def process_directory(
        self,
        directory: Path,
        group_name: str,
        pixel_size_um: float = 0.65,
        output_base_dir: Path = Path('results')
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process all image pairs in a directory.
        
        Args:
            directory: Directory containing image pairs
            group_name: Name of the experimental group
            pixel_size_um: Pixel size in micrometers
            output_base_dir: Base directory for saving results
            
        Returns:
            Tuple of (combined_brightfield_df, combined_fluorescence_df)
        """
        pairs = self.find_image_pairs(directory)
        
        if not pairs:
            print(f"Warning: No image pairs found in {directory}")
            return pd.DataFrame(), pd.DataFrame()
        
        all_bf_results = []
        all_fl_results = []
        
        # Process each image pair
        for bf_path, fl_path in tqdm(pairs, desc=f"Processing {group_name}"):
            # Try to get pixel size from metadata
            metadata_path = directory / 'MetaData' / f'{bf_path.stem.replace(self.file_patterns["brightfield_suffix"].replace(".tif", ""), "")}.xml'
            
            current_pixel_size = pixel_size_um
            if metadata_path.exists():
                parsed_size = self.parse_pixel_size(metadata_path)
                if parsed_size is not None:
                    current_pixel_size = parsed_size
                else:
                    print(f"Warning: Pixel size not found in {metadata_path}, using default 0.65 μm")
            
            try:
                bf_df, fl_df, bf_results, fl_results = self.process_image_pair(bf_path, fl_path, current_pixel_size)
                
                # Save segmentation visualizations
                image_name = bf_path.stem.replace(self.file_patterns['brightfield_suffix'].replace('.tif', ''), '')
                
                # Save brightfield segmentation
                bf_image = cv2.imread(str(bf_path), cv2.IMREAD_GRAYSCALE)
                if bf_image is not None:
                    self.bf_segmenter.visualize_segmentation(
                        original_image=bf_image,
                        labeled_mask=bf_results['labeled_mask'],
                        quality_df=bf_results['quality_scores'],
                        output_path=str(output_base_dir / 'brightfield_images' / f'{group_name}_{image_name}_bf.png')
                    )
                
                # Save fluorescence segmentation
                fl_image = cv2.imread(str(fl_path), cv2.IMREAD_GRAYSCALE)
                if fl_image is not None:
                    self.fl_segmenter.visualize_fluorescence(
                        original_image=fl_image,
                        labeled_mask=fl_results['labeled_mask'],
                        measurements=fl_results['measurements'],
                        output_path=str(output_base_dir / 'fluorescence_images' / f'{group_name}_{image_name}_fl.png')
                    )
                
                all_bf_results.append(bf_df)
                all_fl_results.append(fl_df)
                
            except Exception as e:
                print(f"Error processing {bf_path.name}: {str(e)}")
                continue
        
        # Combine results
        if all_bf_results:
            combined_bf = pd.concat(all_bf_results, ignore_index=True)
            combined_bf['Group'] = group_name
        else:
            combined_bf = pd.DataFrame()
        
        if all_fl_results:
            combined_fl = pd.concat(all_fl_results, ignore_index=True)
            combined_fl['Group'] = group_name
        else:
            combined_fl = pd.DataFrame()
        
        return combined_bf, combined_fl
    
    def parse_pixel_size(self, xml_path: Path) -> Optional[float]:
        """
        Parse pixel size from Leica metadata XML.
        
        Args:
            xml_path: Path to XML metadata file
            
        Returns:
            Pixel size in micrometers, or None if not found
        """
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Method 1: Look for Voxel attribute in DimensionDescription
            for elem in root.iter('DimensionDescription'):
                voxel = elem.get('Voxel')
                if voxel is not None:
                    try:
                        return float(voxel)
                    except ValueError:
                        pass
            
            # Method 2: Calculate from Length and NumberOfElements
            for elem in root.iter('DimensionDescription'):
                dim_id = elem.get('DimID')
                if dim_id in ['1', 'X']:  # X dimension
                    length = elem.get('Length')
                    num_elements = elem.get('NumberOfElements')
                    if length is not None and num_elements is not None:
                        try:
                            # Length is in meters, convert to micrometers
                            length_um = float(length) * 1e6
                            pixels = float(num_elements)
                            return length_um / pixels
                        except ValueError:
                            pass
            
            return None
            
        except Exception as e:
            print(f"Error parsing XML {xml_path}: {e}")
            return None
    
    def process_all_groups(self, output_base_dir: Path):
        """
        Process all experimental groups defined in the configuration.
        
        Args:
            output_base_dir: Base directory for saving results
        """
        # Clean up existing results
        if output_base_dir.exists():
            shutil.rmtree(output_base_dir)
        
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for images
        (output_base_dir / 'brightfield_images').mkdir(exist_ok=True)
        (output_base_dir / 'fluorescence_images').mkdir(exist_ok=True)
        
        all_results = {
            'brightfield': {},
            'fluorescence': {}
        }
        
        # Process each group
        for group_type, directories in self.group_mapping.items():
            for dir_name in directories:
                dir_path = self.source_dir / dir_name
                
                if not dir_path.exists():
                    print(f"Warning: Directory not found: {dir_path}")
                    continue
                
                print(f"\nProcessing {group_type.capitalize()} group: {dir_name}")
                
                bf_df, fl_df = self.process_directory(
                    dir_path, 
                    f"{group_type}_{dir_name}",
                    output_base_dir=output_base_dir
                )
                
                if not bf_df.empty:
                    all_results['brightfield'][f"{group_type}_{dir_name}"] = bf_df
                if not fl_df.empty:
                    all_results['fluorescence'][f"{group_type}_{dir_name}"] = fl_df
        
        # Save results
        self.save_results(all_results, output_base_dir)
        
        # Generate summary statistics
        self.generate_summary(all_results, output_base_dir)
    
    def save_results(self, results: Dict, output_dir: Path):
        """
        Save processing results to CSV files.
        
        Args:
            results: Dictionary containing brightfield and fluorescence results
            output_dir: Directory to save results
        """
        # Save brightfield results
        if results['brightfield']:
            bf_combined = pd.concat(results['brightfield'].values(), ignore_index=True)
            bf_path = output_dir / 'brightfield_results.csv'
            bf_combined.to_csv(bf_path, index=False)
            print(f"\nSaved brightfield results to {bf_path.absolute()}")
        
        # Save fluorescence results
        if results['fluorescence']:
            fl_combined = pd.concat(results['fluorescence'].values(), ignore_index=True)
            fl_path = output_dir / 'fluorescence_results.csv'
            fl_combined.to_csv(fl_path, index=False)
            print(f"Saved fluorescence results to {fl_path.absolute()}")
    
    def generate_summary(self, results: Dict, output_dir: Path):
        """
        Generate summary statistics for each group.
        
        Args:
            results: Dictionary containing brightfield and fluorescence results
            output_dir: Directory to save summary
        """
        summary_data = []
        
        for group_name, bf_df in results['brightfield'].items():
            if group_name in results['fluorescence']:
                fl_df = results['fluorescence'][group_name]
                
                summary_data.append({
                    'Group': group_name,
                    'Total_Particles': len(bf_df),
                    'Total_Signals': len(fl_df),
                    'Mean_Particle_Area_um2': bf_df['Area_um2'].mean(),
                    'Mean_Circularity': bf_df['Circularity'].mean(),
                    'Mean_Signal_Intensity': fl_df['Raw_Mean_Intensity'].mean() if 'Raw_Mean_Intensity' in fl_df.columns else np.nan,
                    'Mean_Corrected_Intensity': fl_df['Corrected_Mean_Intensity'].mean() if 'Corrected_Mean_Intensity' in fl_df.columns else np.nan
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = output_dir / 'summary_statistics.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"Saved summary statistics to {summary_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description='Process specific sample groups')
    parser.add_argument('--sample-group', type=str, help='Sample group directory name')
    parser.add_argument('--control-group', type=str, default='Control group', 
                        help='Control group directory name')
    args = parser.parse_args()
    
    # If no sample group specified, prompt user
    if not args.sample_group:
        print("\nAvailable directories in source folder:")
        source_dir = Path('../source')
        dirs = [d.name for d in source_dir.iterdir() if d.is_dir()]
        for i, d in enumerate(dirs, 1):
            print(f"  {i}. {d}")
        
        choice = input("\nEnter sample group name (or number): ").strip()
        
        # Check if user entered a number
        if choice.isdigit() and 1 <= int(choice) <= len(dirs):
            args.sample_group = dirs[int(choice) - 1]
        else:
            args.sample_group = choice
    
    # If no control group specified, prompt user
    if args.control_group == 'Control group':
        use_default = input(f"\nUse '{args.control_group}' as control? (y/n): ").strip().lower()
        if use_default != 'y':
            args.control_group = input("Enter control group name: ").strip()
    
    print(f"\nProcessing sample group: {args.sample_group}")
    print(f"Control group: {args.control_group}\n")
    
    # Load configurations
    bf_config = load_config('../config/brightfield_config.json')
    fl_config = load_config('../config/fluorescence_config.json')
    
    # Create custom group mapping
    custom_config = {
        "source_directory": "../source",
        "file_patterns": {
            "brightfield_suffix": "_ch00.tif",
            "fluorescence_suffix": "_ch01.tif"
        },
        "group_mapping": {
            "control": [args.control_group],
            "sample": [args.sample_group]
        }
    }
    
    # Initialize and run pipeline
    pipeline = AdaptedSegmentationPipeline(
        bf_config=bf_config,
        fl_config=fl_config,
        pipeline_config=custom_config
    )
    
    pipeline.process_all_groups(output_base_dir=Path('results'))
    
    print("\nProcessing complete!")


if __name__ == '__main__':
    main()