import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io, filters, morphology, measure, segmentation
from scipy import ndimage
from pathlib import Path
import seaborn as sns

def analyze_bacteria_images(image_path_ch00, image_path_ch01, output_folder='./outputs'):
    """
    Analyzes two channels of bacteria screening images to identify, quantify, label, and measure bacteria.

    Args:
        image_path_ch00 (str): Path to the first channel image (brightfield/phase contrast).
        image_path_ch01 (str): Path to the second channel image (fluorescence).
        output_folder (str): Path to save output files.

    Returns:
        dict: A dictionary containing analysis results, measurements DataFrame, and output paths.
    """
    try:
        # Create output folder if it doesn't exist
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load images
        image_ch00 = io.imread(image_path_ch00)
        image_ch01 = io.imread(image_path_ch01)

        print(f"Loaded {image_path_ch00} with shape {image_ch00.shape} and dtype {image_ch00.dtype}")
        print(f"Loaded {image_path_ch01} with shape {image_ch01.shape} and dtype {image_ch01.dtype}")

        # --- Preprocessing for ch00 (brightfield channel) ---
        if image_ch00.ndim == 3:
            image_ch00_gray = image_ch00[:, :, 0]
        else:
            image_ch00_gray = image_ch00

        # --- Preprocessing for ch01 (fluorescence channel) ---
        if image_ch01.ndim == 3:
            image_ch01_gray = image_ch01[:, :, 0]  # Red channel
        else:
            image_ch01_gray = image_ch01

        # Normalize images
        image_ch00_norm = image_ch00_gray / image_ch00_gray.max() if image_ch00_gray.max() > 0 else image_ch00_gray
        image_ch01_norm = image_ch01_gray / image_ch01_gray.max() if image_ch01_gray.max() > 0 else image_ch01_gray

        # --- Enhanced Segmentation ---
        # Apply Gaussian blur for noise reduction
        blurred_image = filters.gaussian(image_ch01_norm, sigma=1)

        # Apply Otsu's threshold
        thresh = filters.threshold_otsu(blurred_image)
        binary_image = blurred_image > thresh

        # Clean up segmentation
        cleaned_image = morphology.remove_small_objects(binary_image, min_size=30)
        cleaned_image = morphology.remove_small_holes(cleaned_image, area_threshold=50)

        # Watershed segmentation to separate touching bacteria
        distance = ndimage.distance_transform_edt(cleaned_image)
        distance_smooth = filters.gaussian(distance, sigma=1)
        local_max = morphology.local_maxima(distance_smooth)
        markers = measure.label(local_max)
        labeled_bacteria = segmentation.watershed(-distance_smooth, markers, mask=cleaned_image)  # Fixed: use distance_smooth directly

        num_bacteria = int(labeled_bacteria.max())  # Ensure it's an int
        
        print(f"Detected {num_bacteria} bacteria")

        # --- Extract Comprehensive Measurements ---
        measurements_df = extract_measurements(
            labeled_bacteria, 
            image_ch00_gray, 
            image_ch01_gray,
            image_path_ch00
        )

        # Classify bacteria by fluorescence intensity
        measurements_df = classify_bacteria(measurements_df)

        # --- Export Measurements to Tables ---
        # Get base filename
        base_name = Path(image_path_ch00).stem
        
        # Export to Excel
        excel_path = output_path / f'{base_name}_measurements.xlsx'
        export_to_excel(measurements_df, excel_path)
        
        # Export to CSV
        csv_path = output_path / f'{base_name}_measurements.csv'
        measurements_df.to_csv(csv_path, index=False)
        print(f"✓ Exported measurements to CSV: {csv_path}")

        # --- Enhanced Visualization ---
        result_image_path = output_path / f'{base_name}_results.png'
        create_comprehensive_visualization(
            image_ch00_gray,
            image_ch01_gray,
            labeled_bacteria,
            measurements_df,
            num_bacteria,
            result_image_path
        )

        # --- Generate Summary Report ---
        summary_path = output_path / f'{base_name}_summary.txt'
        generate_summary_report(measurements_df, summary_path)

        print(f"\n{'='*60}")
        print(f"Analysis complete for: {Path(image_path_ch00).name}")
        print(f"Total bacteria detected: {num_bacteria}")
        print(f"High fluorescence: {len(measurements_df[measurements_df['fluorescence_class']=='High'])}")
        print(f"Low fluorescence: {len(measurements_df[measurements_df['fluorescence_class']=='Low'])}")
        print(f"Results saved to: {output_folder}")
        print(f"{'='*60}\n")

        return {
            'num_bacteria': num_bacteria,
            'measurements': measurements_df,
            'result_image_path': str(result_image_path),
            'excel_path': str(excel_path),
            'csv_path': str(csv_path),
            'summary_path': str(summary_path)
        }

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def extract_measurements(labeled_bacteria, image_ch00, image_ch01, image_name):
    """Extract comprehensive measurements for each bacterium"""
    
    # Measurements from brightfield channel
    props_ch00 = measure.regionprops_table(
        labeled_bacteria,
        intensity_image=image_ch00,
        properties=['label', 'area', 'perimeter', 'eccentricity',
                   'solidity', 'orientation', 'major_axis_length',
                   'minor_axis_length', 'centroid', 'bbox']
    )
    
    # Measurements from fluorescence channel
    props_ch01 = measure.regionprops_table(
        labeled_bacteria,
        intensity_image=image_ch01,
        properties=['label', 'mean_intensity', 'max_intensity',
                   'min_intensity']
    )
    
    # Create DataFrames
    df_ch00 = pd.DataFrame(props_ch00)
    df_ch01 = pd.DataFrame(props_ch01)
    
    # Merge measurements
    measurements = pd.merge(df_ch00, df_ch01, on='label')
    
    # Add calculated features
    measurements['aspect_ratio'] = (
        measurements['major_axis_length'] / measurements['minor_axis_length']
    )
    
    measurements['circularity'] = (
        4 * np.pi * measurements['area'] / (measurements['perimeter'] ** 2)
    )
    
    measurements['integrated_intensity'] = (
        measurements['mean_intensity'] * measurements['area']
    )
    
    measurements['orientation_degrees'] = np.degrees(measurements['orientation'])
    
    # Calculate bounding box dimensions
    measurements['bbox_width'] = measurements['bbox-3'] - measurements['bbox-1']
    measurements['bbox_height'] = measurements['bbox-2'] - measurements['bbox-0']
    
    # Rename centroid columns for clarity
    measurements.rename(columns={
        'centroid-0': 'centroid_y',
        'centroid-1': 'centroid_x'
    }, inplace=True)
    
    # Add image identifier
    measurements['image_name'] = Path(image_name).stem
    
    # Round numeric columns
    numeric_cols = measurements.select_dtypes(include=[np.number]).columns
    measurements[numeric_cols] = measurements[numeric_cols].round(3)
    
    # Reorder columns for better readability
    column_order = ['label', 'image_name', 'centroid_x', 'centroid_y', 
                   'area', 'perimeter', 'circularity',
                   'major_axis_length', 'minor_axis_length', 'aspect_ratio',
                   'eccentricity', 'solidity', 'orientation_degrees',
                   'mean_intensity', 'max_intensity', 'min_intensity', 
                   'integrated_intensity']
    
    # Add remaining columns
    remaining_cols = [col for col in measurements.columns if col not in column_order]
    measurements = measurements[column_order + remaining_cols]
    
    return measurements


def classify_bacteria(measurements_df, intensity_threshold=None):
    """Classify bacteria based on fluorescence intensity"""
    
    if intensity_threshold is None:
        # Use median as threshold
        intensity_threshold = measurements_df['mean_intensity'].median()
    
    measurements_df['fluorescence_class'] = np.where(
        measurements_df['mean_intensity'] > intensity_threshold,
        'High',
        'Low'
    )
    
    measurements_df['intensity_threshold'] = intensity_threshold
    
    return measurements_df


def create_comprehensive_visualization(image_ch00, image_ch01, labeled_bacteria, 
                                      measurements_df, num_bacteria, save_path):
    """Create comprehensive visualization with multiple panels"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Panel 1: Original brightfield
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_ch00, cmap='gray')
    ax1.set_title('Channel 00 (Original)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Original fluorescence
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(image_ch01, cmap='hot')
    ax2.set_title('Channel 01 (Processed Fluorescence)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Panel 3: Labeled bacteria with colored regions
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(image_ch00, cmap='gray')
    ax3.imshow(labeled_bacteria, cmap='nipy_spectral', alpha=0.5)
    ax3.set_title(f'Detected Bacteria (Count: {num_bacteria})', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Panel 4: Tagged and numbered bacteria
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(image_ch00, cmap='gray', alpha=0.7)
    
    # Add labels and color coding
    for _, row in measurements_df.iterrows():
        color = 'lime' if row['fluorescence_class'] == 'High' else 'cyan'
        marker_size = 12 if row['fluorescence_class'] == 'High' else 8
        
        # Draw circle
        ax4.plot(row['centroid_x'], row['centroid_y'], 'o',
                color=color, markersize=marker_size, 
                markeredgewidth=2, markerfacecolor='none')
        
        # Add label number
        ax4.text(row['centroid_x'] + 10, row['centroid_y'] + 10,
                str(int(row['label'])),
                color='yellow', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    
    ax4.set_title('Tagged & Labeled Bacteria', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Panel 5: Area distribution
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.hist(measurements_df['area'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax5.axvline(measurements_df['area'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {measurements_df["area"].mean():.1f}')
    ax5.set_xlabel('Area (pixels²)', fontsize=10)
    ax5.set_ylabel('Count', fontsize=10)
    ax5.set_title('Size Distribution', fontsize=11, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # Panel 6: Intensity distribution
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.hist(measurements_df['mean_intensity'], bins=20, 
            color='orangered', edgecolor='black', alpha=0.7)
    ax6.axvline(measurements_df['mean_intensity'].median(), color='blue', 
                linestyle='--', linewidth=2, 
                label=f'Median: {measurements_df["mean_intensity"].median():.1f}')
    ax6.set_xlabel('Mean Fluorescence Intensity', fontsize=10)
    ax6.set_ylabel('Count', fontsize=10)
    ax6.set_title('Intensity Distribution', fontsize=11, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # Panel 7: Size vs Intensity scatter
    ax7 = fig.add_subplot(gs[1, 2])
    scatter = ax7.scatter(measurements_df['area'],
                         measurements_df['mean_intensity'],
                         c=measurements_df['mean_intensity'],
                         cmap='hot', s=60, edgecolors='black', 
                         linewidth=0.5, alpha=0.7)
    ax7.set_xlabel('Area (pixels²)', fontsize=10)
    ax7.set_ylabel('Mean Fluorescence Intensity', fontsize=10)
    ax7.set_title('Size vs Intensity', fontsize=11, fontweight='bold')
    ax7.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax7, label='Intensity')
    
    # Panel 8: Morphology analysis
    ax8 = fig.add_subplot(gs[1, 3])
    scatter2 = ax8.scatter(measurements_df['aspect_ratio'],
                          measurements_df['circularity'],
                          c=measurements_df['mean_intensity'],
                          cmap='hot', s=60, edgecolors='black',
                          linewidth=0.5, alpha=0.7)
    ax8.set_xlabel('Aspect Ratio', fontsize=10)
    ax8.set_ylabel('Circularity', fontsize=10)
    ax8.set_title('Morphology Analysis', fontsize=11, fontweight='bold')
    ax8.grid(alpha=0.3)
    plt.colorbar(scatter2, ax=ax8, label='Intensity')
    
    # Panel 9: Classification pie chart
    ax9 = fig.add_subplot(gs[2, 0])
    class_counts = measurements_df['fluorescence_class'].value_counts()
    colors_pie = ['#2ecc71', '#3498db']
    ax9.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
           colors=colors_pie, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax9.set_title('Fluorescence Classification', fontsize=11, fontweight='bold')
    
    # Panel 10: Summary statistics table
    ax10 = fig.add_subplot(gs[2, 1:3])
    ax10.axis('tight')
    ax10.axis('off')
    
    summary_data = {
        'Metric': [
            'Total Bacteria',
            'High Fluorescence',
            'Low Fluorescence',
            'Mean Area (px²)',
            'Mean Intensity',
            'Mean Aspect Ratio',
            'Mean Circularity'
        ],
        'Value': [
            str(num_bacteria),
            str(len(measurements_df[measurements_df['fluorescence_class'] == 'High'])),
            str(len(measurements_df[measurements_df['fluorescence_class'] == 'Low'])),
            f"{measurements_df['area'].mean():.1f}",
            f"{measurements_df['mean_intensity'].mean():.1f}",
            f"{measurements_df['aspect_ratio'].mean():.2f}",
            f"{measurements_df['circularity'].mean():.2f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Fixed: Convert to list of lists for table
    table = ax10.table(cellText=summary_df.values.tolist(),  # Convert to list
                      colLabels=summary_df.columns.tolist(),  # Convert to list
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style table header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    # Panel 11: Top 10 bacteria by intensity
    ax11 = fig.add_subplot(gs[2, 3])
    top10 = measurements_df.nlargest(10, 'mean_intensity')[['label', 'mean_intensity']]
    ax11.barh(top10['label'].astype(str), top10['mean_intensity'], 
             color='orangered', edgecolor='black', alpha=0.7)
    ax11.set_xlabel('Mean Intensity', fontsize=10)
    ax11.set_ylabel('Bacteria Label', fontsize=10)
    ax11.set_title('Top 10 Brightest Bacteria', fontsize=11, fontweight='bold')
    ax11.grid(axis='x', alpha=0.3)
    ax11.invert_yaxis()
    
    plt.suptitle('Comprehensive Bacteria Analysis Results',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Visualization saved: {save_path}")


def export_to_excel(measurements_df, excel_path):
    """Export measurements to formatted Excel file"""
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: All measurements
        measurements_df.to_excel(writer, sheet_name='All Measurements', index=False)
        
        # Sheet 2: Summary statistics
        summary_stats = measurements_df.describe().T
        summary_stats.to_excel(writer, sheet_name='Summary Statistics')
        
        # Sheet 3: Classification breakdown
        classification = measurements_df.groupby('fluorescence_class').agg({
            'label': 'count',
            'area': ['mean', 'std'],
            'mean_intensity': ['mean', 'std'],
            'aspect_ratio': ['mean', 'std']
        }).round(3)
        classification.to_excel(writer, sheet_name='Classification Analysis')
        
        # Sheet 4: Top performers
        top_bacteria = measurements_df.nlargest(20, 'mean_intensity')
        top_bacteria.to_excel(writer, sheet_name='Top 20 by Intensity', index=False)
    
    print(f"✓ Exported measurements to Excel: {excel_path}")


def generate_summary_report(measurements_df, summary_path):
    """Generate text summary report"""
    
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BACTERIA SCREENING ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total Bacteria Detected: {len(measurements_df)}\n\n")
        
        f.write("CLASSIFICATION:\n")
        f.write("-" * 70 + "\n")
        class_counts = measurements_df['fluorescence_class'].value_counts()
        for class_name, count in class_counts.items():
            percentage = (count / len(measurements_df)) * 100
            f.write(f"  {class_name} Fluorescence: {count} ({percentage:.1f}%)\n")
        
        f.write("\nMORPHOLOGY STATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Mean Area: {measurements_df['area'].mean():.2f} ± {measurements_df['area'].std():.2f} pixels²\n")
        f.write(f"  Mean Aspect Ratio: {measurements_df['aspect_ratio'].mean():.2f} ± {measurements_df['aspect_ratio'].std():.2f}\n")
        f.write(f"  Mean Circularity: {measurements_df['circularity'].mean():.2f} ± {measurements_df['circularity'].std():.2f}\n")
        
        f.write("\nFLUORESCENCE STATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Mean Intensity: {measurements_df['mean_intensity'].mean():.2f} ± {measurements_df['mean_intensity'].std():.2f}\n")
        f.write(f"  Median Intensity: {measurements_df['mean_intensity'].median():.2f}\n")
        f.write(f"  Max Intensity: {measurements_df['max_intensity'].max():.2f}\n")
        
        f.write("\nTOP 5 BRIGHTEST BACTERIA:\n")
        f.write("-" * 70 + "\n")
        top5 = measurements_df.nlargest(5, 'mean_intensity')
        for idx, row in top5.iterrows():
            f.write(f"  #{int(row['label'])}: Intensity = {row['mean_intensity']:.2f}, Area = {row['area']:.1f} px²\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✓ Summary report saved: {summary_path}")


# Batch processing function
def batch_analyze_folder(input_folder, output_folder='./outputs'):
    """
    Batch process all image pairs in a folder
    
    Args:
        input_folder (str): Folder containing image pairs (ch00 and ch01)
        output_folder (str): Folder to save all results
    """
    input_path = Path(input_folder)
    
    # Find all ch00 images
    ch00_images = sorted(input_path.glob('*_ch00.tif'))
    
    if not ch00_images:
        print(f"No ch00 images found in {input_folder}")
        return None
    
    print(f"\nFound {len(ch00_images)} image pairs to process\n")
    
    all_results = []
    all_measurements = []
    
    for ch00_path in ch00_images:
        # Find corresponding ch01 image
        ch01_path = ch00_path.parent / ch00_path.name.replace('_ch00', '_ch01')
        
        if not ch01_path.exists():
            print(f"Warning: No ch01 image found for {ch00_path.name}, skipping...")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {ch00_path.name}")
        print(f"{'='*70}")
        
        # Analyze
        results = analyze_bacteria_images(str(ch00_path), str(ch01_path), output_folder)
        
        if 'error' not in results:
            all_results.append(results)
            all_measurements.append(results['measurements'])
    
    # Combine all measurements
    if all_measurements:
        combined_measurements = pd.concat(all_measurements, ignore_index=True)
        
        # Save combined results
        combined_excel = Path(output_folder) / 'combined_all_images.xlsx'
        combined_csv = Path(output_folder) / 'combined_all_images.csv'
        
        export_to_excel(combined_measurements, combined_excel)
        combined_measurements.to_csv(combined_csv, index=False)
        
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Total images processed: {len(all_results)}")
        print(f"Total bacteria detected: {len(combined_measurements)}")
        print(f"Combined results saved to: {output_folder}")
        print(f"{'='*70}\n")
        
        return combined_measurements
    
    return None


if __name__ == '__main__':
    # Single image analysis
    ch00_image = './PD image/1/1 N NO 1_ch00.tif'
    ch01_image = './PD image/1/1 N NO 1_ch01.tif'
    results = analyze_bacteria_images(ch00_image, ch01_image)
    print("\nResults dictionary keys:", results.keys())
    
    # Uncomment to process entire folder
    # batch_analyze_folder('./PD image/1', './outputs')