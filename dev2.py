import matplotlib.pyplot as plt
import numpy as np
from skimage import io, filters, morphology, measure
from scipy import ndimage

def analyze_bacteria_images(image_path_ch00, image_path_ch01):
    """
    Analyzes two channels of bacteria screening images to identify and quantify bacteria.

    Args:
        image_path_ch00 (str): Path to the first channel image (e.g., brightfield/phase contrast).
        image_path_ch01 (str): Path to the second channel image (e.g., fluorescence).

    Returns:
        dict: A dictionary containing analysis results, including the number of bacteria
              and paths to generated visualization images.
    """
    try:
        # Load images
        image_ch00 = io.imread(image_path_ch00)
        image_ch01 = io.imread(image_path_ch01)

        print(f"Loaded {image_path_ch00} with shape {image_ch00.shape} and dtype {image_ch00.dtype}")
        print(f"Loaded {image_path_ch01} with shape {image_ch01.shape} and dtype {image_ch01.dtype}")

        # --- Preprocessing for ch01 (fluorescence channel) ---
        if image_ch01.ndim == 3:
            image_ch01_gray = image_ch01[:, :, 0]  # Assuming red channel is the signal
        else:
            image_ch01_gray = image_ch01

        # Normalize image to 0-1 range
        image_ch01_norm = image_ch01_gray / image_ch01_gray.max()

        # Apply Gaussian blur for noise reduction
        blurred_image = filters.gaussian(image_ch01_norm, sigma=1)

        # Apply Otsu's threshold
        thresh = filters.threshold_otsu(blurred_image)
        binary_image = blurred_image > thresh

        # Clean up segmentation
        cleaned_image = morphology.remove_small_objects(binary_image, min_size=50)
        cleaned_image = morphology.remove_small_holes(cleaned_image, area_threshold=50)

        # Debug: Inspect cleaned_image
        print(f"cleaned_image type: {type(cleaned_image)}, shape: {cleaned_image.shape}, unique values: {np.unique(cleaned_image)}")

        # Label connected components
        result = measure.label(cleaned_image, return_num=True)
        if isinstance(result, tuple) and len(result) == 2:
            labeled_bacteria, num_bacteria = result
        else:
            raise ValueError("Unexpected return value from measure.label")
        print(f"labeled_bacteria type: {type(labeled_bacteria)}, shape: {labeled_bacteria.shape}")
        print(f"num_bacteria type: {type(num_bacteria)}, value: {num_bacteria}")

        # --- Visualization ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        ax = axes.ravel()

        ax[0].imshow(image_ch00, cmap='gray')
        ax[0].set_title('Channel 00 (Original)')
        ax[0].axis('off')

        ax[1].imshow(image_ch01_gray, cmap='hot')
        ax[1].set_title('Channel 01 (Processed Fluorescence)')
        ax[1].axis('off')

        # Overlay detected bacteria
        ax[2].imshow(image_ch00, cmap='gray')
        ax[2].imshow(labeled_bacteria, cmap='nipy_spectral', alpha=0.5)
        ax[2].set_title(f'Detected Bacteria (Count: {num_bacteria})')
        ax[2].axis('off')

        plt.tight_layout()
        output_plot_path = './outputs/results.png'
        plt.savefig(output_plot_path)
        plt.close(fig)

        print(f"Analysis complete. Detected {num_bacteria} bacteria. Results saved to {output_plot_path}")

        return {
            'num_bacteria': num_bacteria,
            'result_image_path': output_plot_path
        }

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return {'error': str(e)}

if __name__ == '__main__':
    ch00_image = './PD image/1/1 N NO 1_ch00.tif'
    ch01_image = './PD image/1/1 N NO 1_ch01.tif'
    results = analyze_bacteria_images(ch00_image, ch01_image)
    print(results)