"""
core/segmentation.py - Bacteria Segmentation Algorithms
"""
import cv2
import numpy as np
from scipy import ndimage
from typing import List, Tuple, Optional, Union, Sequence

# Type alias for contours
Contour = np.ndarray
ContourList = Sequence[Contour]

def segment_bacteria(gray_bf: np.ndarray, 
                     use_otsu: bool = False,
                     manual_threshold: int = 110,
                     enable_clahe: bool = True,
                     clahe_clip: float = 5.0,
                     clahe_tile: int = 32,
                     open_kernel: int = 3,
                     close_kernel: int = 5,
                     open_iter: int = 3,
                     close_iter: int = 2,
                     min_area: int = 100,
                     watershed_dilate: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
    """
    Segment bacteria from brightfield image.

    Returns: enhanced, threshold, cleaned, contours
    """

    # Step 1: CLAHE Enhancement
    if enable_clahe:
        bf8 = cv2.convertScaleAbs(gray_bf)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, 
                                tileGridSize=(clahe_tile, clahe_tile))
        enhanced = clahe.apply(bf8)
    else:
        enhanced = cv2.convertScaleAbs(gray_bf)

    # Step 2: Thresholding
    if use_otsu:
        _, thresh = cv2.threshold(enhanced, 0, 255, 
                                  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(enhanced, manual_threshold, 255, 
                                  cv2.THRESH_BINARY_INV)

    # Step 3: Morphological Operations
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                       (open_kernel, open_kernel))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                        (close_kernel, close_kernel))

    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k, 
                             iterations=open_iter)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_k, 
                              iterations=close_iter)

    # Step 4: Watershed Segmentation
    distance = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(distance, watershed_dilate * distance.max() / 100.0, 
                            255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, open_k, iterations=1)

    markers, num_features = ndimage.label(sure_fg)
    markers = markers.astype(np.int32)
    markers += 1
    markers[cleaned == 0] = 0

    markers = cv2.watershed(cv2.cvtColor(gray_bf, cv2.COLOR_GRAY2BGR), markers)

    contour_mask = (markers != 1).astype(np.uint8) * 255

    # Fixed: Handle both OpenCV 3.x and 4.x
    find_result = cv2.findContours(
        contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Extract contours (works for both OpenCV versions)
    if len(find_result) == 3:
        contours = find_result[1]
    else:
        contours = find_result[0]

    # Filter by area
    bacteria = [c for c in contours if cv2.contourArea(c) >= min_area]

    return enhanced, thresh, cleaned, bacteria