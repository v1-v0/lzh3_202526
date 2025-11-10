import os
import numpy as np
import cv2
from pathlib import Path
from skimage.io import imread
from skimage import filters, morphology, measure
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from scipy import stats
import pandas as pd
import warnings
from openpyxl.utils import get_column_letter
import re
from typing import Optional, Union, Any, cast
from openpyxl.worksheet.worksheet import Worksheet
from openpyxl.cell.cell import Cell, MergedCell

warnings.filterwarnings('ignore')

class BacteriaReviewTool:
    def __init__(self, root_folder="source"):
        """Initialize the review tool with root folder"""
        self.root_folder = root_folder
        self.config = {
            "gray_threshold": 53,
            "fluor_threshold": 5,
            "min_area": 400,
            "min_fluor_area": 400,
            "overlap_threshold": 0.80,
        }
        self.all_results = []
        self.process_all_subfolders()
        self.export_to_excel()

    def process_all_subfolders(self):
        """Process all subfolders under root_folder"""
        root_path = Path(self.root_folder)
        if not root_path.exists():
            print(f"Error: Root folder '{self.root_folder}' does not exist")
            return

        subfolders = [f for f in root_path.iterdir() if f.is_dir()]
        print(f"Found {len(subfolders)} subfolder(s) to process")

        for subfolder in subfolders:
            print(f"\nProcessing subfolder: {subfolder.name}")
            self.image_pairs = []
            self.load_image_pairs(subfolder)
            if self.image_pairs:
                self.generate_detection_reports(subfolder.name)
            else:
                print(f"  No image pairs found in {subfolder.name}")

    def load_image_pairs(self, folder_path):
        """Load all grayscale and fluorescence image pairs from a folder"""
        self.image_pairs = []
        folder_path = Path(folder_path)

        gray_files = sorted([
            f for f in os.listdir(folder_path)
            if f.endswith('_ch00.tif') and os.path.isfile(folder_path / f)
        ])

        for gray_file in gray_files:
            fluor_file = gray_file.replace('_ch00.tif', '_ch01.tif')
            gray_path = folder_path / gray_file
            fluor_path = folder_path / fluor_file

            if fluor_path.exists():
                try:
                    gray_img = imread(gray_path)
                    fluor_img = imread(fluor_path)

                    if len(gray_img.shape) == 3:
                        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)
                    if len(fluor_img.shape) == 3:
                        fluor_img = cv2.cvtColor(fluor_img, cv2.COLOR_RGB2GRAY)

                    self.image_pairs.append({
                        'gray': gray_img,
                        'fluor': fluor_img,
                        'gray_file': gray_file,
                        'fluor_file': fluor_file
                    })
                except Exception as e:
                    print(f"  Error loading {gray_file}: {e}")

        print(f"  Loaded {len(self.image_pairs)} image pair(s)")

    def preprocess_grayscale(self, image):
        """Preprocess grayscale image"""
        median_filtered = filters.median(image, morphology.disk(3))
        if median_filtered.max() > 0:
            img_normalized = (median_filtered / median_filtered.max() * 255).astype(np.uint8)
        else:
            img_normalized = median_filtered.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_normalized)
        return enhanced

    def preprocess_fluorescence(self, image):
        """Preprocess fluorescence image"""
        median_filtered = filters.median(image, morphology.disk(3))
        if median_filtered.max() > 0:
            img_normalized = (median_filtered / median_filtered.max() * 255).astype(np.uint8)
        else:
            img_normalized = median_filtered.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_normalized)
        return enhanced

    def segment_grayscale(self, preprocessed, threshold):
        """Segment grayscale bacteria (dark objects)"""
        binary = np.asarray(preprocessed) <= threshold
        distance = ndi.distance_transform_edt(binary)
        distance = np.asarray(distance, dtype=np.float32)
        local_max = morphology.local_maxima(distance)
        markers = measure.label(local_max)
        labels = watershed(-distance, markers, mask=binary)
        return labels > 0, labels

    def segment_fluorescence(self, preprocessed, threshold):
        """Segment fluorescence bacteria (bright objects)"""
        binary = np.asarray(preprocessed) >= threshold
        distance = ndi.distance_transform_edt(binary)
        distance = np.asarray(distance, dtype=np.float32)
        local_max = morphology.local_maxima(distance)
        markers = measure.label(local_max)
        labels = watershed(-distance, markers, mask=binary)
        return labels > 0, labels

    def postprocess(self, binary_mask):
        """Post-process binary mask"""
        min_size = 50
        cleaned = morphology.remove_small_objects(binary_mask, min_size=min_size)
        filled = ndi.binary_fill_holes(cleaned)
        opened = morphology.binary_opening(filled, morphology.disk(2))
        final = morphology.binary_closing(opened, morphology.disk(2))
        return final

    def detect_bacteria(self, image, is_grayscale=True):
        """Detect bacteria in image with current settings"""
        if is_grayscale:
            preprocessed = self.preprocess_grayscale(image)
            threshold = self.config['gray_threshold']
            min_area = self.config['min_area']
            binary, labels = self.segment_grayscale(preprocessed, threshold)
        else:
            preprocessed = self.preprocess_fluorescence(image)
            threshold = self.config['fluor_threshold']
            min_area = self.config['min_fluor_area']
            binary, labels = self.segment_fluorescence(preprocessed, threshold)

        final = self.postprocess(binary)
        labeled = measure.label(final)
        props = measure.regionprops(labeled)
        large_props = [prop for prop in props if prop.area > min_area]
        return labeled, large_props

    def calculate_mode(self, intensities):
        """Calculate mode of intensity values"""
        if len(intensities) == 0:
            return 0
        mode_result = stats.mode(intensities, keepdims=True)
        return float(mode_result.mode[0])

    def get_contour_intensities(self, image, labeled, props):
        """Get intensity statistics for each contour (using MODE)"""
        contour_data = []
        for i, prop in enumerate(props):
            region_mask = (labeled == prop.label)
            intensities = image[region_mask]
            contour_data.append({
                'id': i + 1,
                'area': prop.area,
                'mode_intensity': self.calculate_mode(intensities),
                'label': prop.label
            })
        return contour_data

    def calculate_colocalization(self, gray_labeled, fluor_labeled, gray_props, fluor_props):
        """Calculate which grayscale and fluorescence bacteria are co-localized"""
        colocalized = []
        overlap_threshold = self.config['overlap_threshold']

        for gray_prop in gray_props:
            gray_mask = (gray_labeled == gray_prop['label'])
            for fluor_prop in fluor_props:
                fluor_mask = (fluor_labeled == fluor_prop['label'])
                intersection = np.logical_and(gray_mask, fluor_mask)
                intersection_area = np.sum(intersection)
                overlap_ratio = intersection_area / gray_prop['area']
                if overlap_ratio >= overlap_threshold:
                    colocalized.append({
                        'gray_id': gray_prop['id'],
                        'fluor_id': fluor_prop['id'],
                        'overlap_ratio': overlap_ratio,
                        'gray_area': gray_prop['area'],
                        'fluor_area': fluor_prop['area']
                    })
        return colocalized

    def generate_detection_reports(self, subfolder_name):
        """Generate detection reports for all image pairs in a subfolder"""
        for pair in self.image_pairs:
            gray_img = pair['gray']
            fluor_img = pair['fluor']
            filename = pair['gray_file']

            print(f"  Processing {filename}")

            # Detect bacteria
            gray_labeled, gray_large = self.detect_bacteria(gray_img, is_grayscale=True)
            fluor_labeled, fluor_large = self.detect_bacteria(fluor_img, is_grayscale=False)

            # Get contour intensity data
            gray_contour_data = self.get_contour_intensities(gray_img, gray_labeled, gray_large)
            fluor_contour_data = self.get_contour_intensities(fluor_img, fluor_labeled, fluor_large)

            # Calculate co-localization
            colocalized = self.calculate_colocalization(gray_labeled, fluor_labeled, 
                                                       gray_contour_data, fluor_contour_data)

            # Calculate statistics
            gray_areas = [d['area'] for d in gray_contour_data]
            fluor_areas = [d['area'] for d in fluor_contour_data]
            gray_intensities = [d['mode_intensity'] for d in gray_contour_data]
            fluor_intensities = [d['mode_intensity'] for d in fluor_contour_data]

            # Store results
            coloc_rate = (len(colocalized) / len(gray_large) * 100) if len(gray_large) > 0 else 0
            self.all_results.append({
                'subfolder': subfolder_name,
                'filename': filename,
                'gray_count': len(gray_large),
                'fluor_count': len(fluor_large),
                'coloc_count': len(colocalized),
                'coloc_rate': coloc_rate,
                'gray_avg_area': np.mean(gray_areas) if gray_areas else 0,
                'gray_std_area': np.std(gray_areas) if gray_areas else 0,
                'gray_avg_mode_intensity': np.mean(gray_intensities) if gray_intensities else 0,
                'fluor_avg_area': np.mean(fluor_areas) if fluor_areas else 0,
                'fluor_std_area': np.std(fluor_areas) if fluor_areas else 0,
                'fluor_avg_mode_intensity': np.mean(fluor_intensities) if fluor_intensities else 0,
            })

    def export_to_excel(self):
        """Export detection reports to a single Excel sheet"""
        print("\nExporting detection reports to Excel...")
        output_file = "bacteria_detection_reports.xlsx"
        writer = pd.ExcelWriter(output_file, engine='openpyxl')

        # Create detection reports data
        report_data = []
        for result in self.all_results:
            report_data.append({
                'Subfolder': result['subfolder'],
                'Filename': result['filename'],
                'Grayscale_Count': result['gray_count'],
                'Fluorescence_Count': result['fluor_count'],
                'Colocalized_Count': result['coloc_count'],
                'Colocalization_Rate_%': round(result['coloc_rate'], 2),
                'Gray_Avg_Area': round(result['gray_avg_area'], 2),
                'Gray_Std_Area': round(result['gray_std_area'], 2),
                'Gray_Avg_Mode_Intensity': round(result['gray_avg_mode_intensity'], 2),
                'Fluor_Avg_Area': round(result['fluor_avg_area'], 2),
                'Fluor_Std_Area': round(result['fluor_std_area'], 2),
                'Fluor_Avg_Mode_Intensity': round(result['fluor_avg_mode_intensity'], 2),
            })

        # Write to Excel
        report_df = pd.DataFrame(report_data)
        report_df.to_excel(writer, sheet_name='Detection_Reports', index=False)

        # Adjust column widths
        ws = writer.book['Detection_Reports']
        for column in ws.columns:
            max_length = 0
            first_cell = column[0]
            column_letter = get_column_letter_from_cell(first_cell, ws) if first_cell is not None else None
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = max_length + 2
            if column_letter:
                ws.column_dimensions[column_letter].width = adjusted_width

        writer.close()
        print(f"✓ Excel file saved: {output_file}")
        print(f"  - Detection reports: {len(report_data)} entries")
        print(f"{'='*70}\n")

def get_column_letter_from_cell(cell: Union[Cell, MergedCell], ws: Optional[Worksheet] = None) -> Optional[str]:
    # If cell is part of a merged range, use the top-left cell of that range
    if ws is not None:
        for rng in ws.merged_cells.ranges:
            if cell.coordinate in rng:
                min_col, min_row, _, _ = rng.bounds
                cell = ws.cell(row=min_row, column=min_col)
                break

    # Prefer parsing the coordinate (e.g. "B12" -> "B")
    m = re.match(r"^([A-Z]+)", cell.coordinate)
    if m:
        return m.group(1)

    # Fallback: use numeric column -> convert to letter
    col = getattr(cell, "column", None)
    if isinstance(col, int):
        return get_column_letter(col)

    column_letter = cast(Any, cell).column_letter  # type: ignore[attr-defined]

    return None

def main():
    """Main function to run the review tool"""
    print("="*70)
    print("BACTERIA DETECTION REPORT GENERATOR")
    print("="*70)
    print("\nFeatures:")
    print("  • Processes all subfolders under 'source'")
    print("  • Generates single Detection Reports sheet in Excel")
    print("  • Includes counts, colocalization rate, and summary statistics")
    print("  • Uses MODE intensity (no MIN)")
    print("\n" + "="*70 + "\n")

    try:
        tool = BacteriaReviewTool(root_folder="source")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()