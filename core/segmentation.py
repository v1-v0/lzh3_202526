#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bacteria segmentation algorithms.

This module is aligned with the segmentation logic used in dev.py
(SegmentationViewer.segment_bacteria):

    1. CLAHE (if enabled)
    2. Threshold (Otsu or manual)
    3. Morphological open / close
    4. Distance transform + watershed markers
    5. Watershed
    6. Area-based contour filtering

The GUI in main_window.py uses the *static* API:

    labeled, contours = BacteriaSegmenter.watershed_segmentation(binary, watershed_dilate)
    contours          = BacteriaSegmenter.filter_bacteria(contours, min_area)

The older instance API (watershed_separation / segment with self.params)
is preserved for backward compatibility but is not used by the main app.
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import List, Sequence, Tuple, cast
from cv2.typing import MatLike


class BacteriaSegmenter:
    """Handles bacteria segmentation from bright-field microscopy images."""

    # ------------------------------------------------------------------ #
    # Legacy instance-based API (uses self.params; kept for compatibility)
    # ------------------------------------------------------------------ #
    def __init__(self, params: dict):
        """Initialize segmenter with parameter dict holding tk.Variables.

        This is the interface used in the dev.py tuner, where params is a
        mapping from parameter-name → tk.Variable (IntVar/DoubleVar/BooleanVar).
        The main application (main_window.py) does NOT use this constructor.
        """
        self.params = params

    def watershed_separation(self, cleaned: np.ndarray,
                             gray_bf: np.ndarray) -> List[np.ndarray]:
        """Apply watershed algorithm to separate touching bacteria.

        This mirrors the logic used in SegmentationViewer.segment_bacteria
        from dev.py (instance-based API).
        """
        # Distance transform on cleaned binary image
        distance = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)

        # Threshold for sure foreground (percent of max distance)
        threshold_val = (
            self.params['watershed_dilate'].get() * distance.max() / 100.0
        )
        _, sure_fg = cv2.threshold(distance, threshold_val, 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        # Additional opening to separate markers
        open_k_size = self._ensure_odd(self.params['open_kernel'].get())
        open_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_k_size, open_k_size)
        )
        sure_fg = cv2.morphologyEx(
            sure_fg, cv2.MORPH_OPEN, open_k, iterations=1
        )

        # Label connected components as markers
        markers, _ = cast(Tuple[np.ndarray, int], ndimage.label(sure_fg))
        markers = markers.astype(np.int32)
        markers += 1
        markers[cleaned == 0] = 0

        # Run watershed on the original gray BF image
        watershed_input = cv2.cvtColor(gray_bf, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(watershed_input, markers)

        # Build mask from labeled regions > 1
        contour_mask = (markers > 1).astype(np.uint8) * 255
        res = cv2.findContours(
            contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours: Sequence[MatLike] = res[-2]

        # Filter by minimum area
        min_area = self.params['min_area'].get()
        bacteria = [c for c in contours if cv2.contourArea(c) >= min_area]

        return bacteria

    def segment(self, gray_bf: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                   np.ndarray,
                                                   List[np.ndarray]]:
        """Segment bacteria from a bright-field image (legacy API).

        Returns:
            enhanced: CLAHE / contrast-adjusted image
            thresh:   binary thresholded image
            cleaned:  morphologically cleaned binary
            contours: final bacteria contours (area-filtered)
        """
        enhanced = self._enhance_image(gray_bf)
        thresh = self._threshold_image(enhanced)
        cleaned = self._morphological_cleanup(thresh)
        contours = self.watershed_separation(cleaned, gray_bf)

        return enhanced, thresh, cleaned, contours

    def _enhance_image(self, gray_bf: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement if enabled (dev.py equivalent)."""
        if self.params['enable_clahe'].get():
            bf8 = cv2.convertScaleAbs(gray_bf)
            clahe = cv2.createCLAHE(
                clipLimit=self.params['clahe_clip'].get(),
                tileGridSize=(self.params['clahe_tile'].get(),) * 2
            )
            return clahe.apply(bf8)
        return cv2.convertScaleAbs(gray_bf)

    def _threshold_image(self, enhanced: np.ndarray) -> np.ndarray:
        """Apply thresholding (Otsu or manual) – same logic as dev.py."""
        if self.params['use_otsu'].get():
            _, thresh = cv2.threshold(
                enhanced, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        else:
            _, thresh = cv2.threshold(
                enhanced,
                self.params['manual_threshold'].get(),
                255,
                cv2.THRESH_BINARY_INV
            )
        return thresh

    def _morphological_cleanup(self, thresh: np.ndarray) -> np.ndarray:
        """Apply morphological open/close cleanup (dev.py equivalent)."""
        open_k_size = self._ensure_odd(self.params['open_kernel'].get())
        close_k_size = self._ensure_odd(self.params['close_kernel'].get())

        open_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_k_size, open_k_size)
        )
        close_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_k_size, close_k_size)
        )

        opened = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, open_k,
            iterations=self.params['open_iter'].get()
        )
        cleaned = cv2.morphologyEx(
            opened, cv2.MORPH_CLOSE, close_k,
            iterations=self.params['close_iter'].get()
        )
        return cleaned

    # ------------------------------------------------------------------ #
    # Shared utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_odd(value: int) -> int:
        """Ensure kernel size is odd (required by many morphology ops)."""
        if value % 2 == 0:
            return max(1, value - 1)
        return value

    @staticmethod
    def filter_bacteria(contours: Sequence[MatLike],
                        min_area: float) -> List[np.ndarray]:
        """Filter bacteria contours by minimum area.

        Args:
            contours: List of contours (each Nx1x2 array)
            min_area: Minimum area threshold in pixels

        Returns:
            Filtered list of contours with area >= min_area
        """
        filtered: List[np.ndarray] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                filtered.append(contour)
        return filtered

    # ------------------------------------------------------------------ #
    # New static API used by main_window.py (dev.py–style watershed)
    # ------------------------------------------------------------------ #
    @staticmethod
    def watershed_segmentation(binary: np.ndarray,
                               watershed_dilate: float
                               ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Perform watershed segmentation matching dev.py behavior.
        
        Args:
            binary:
                Binary image (uint8, 0/255) where foreground ≠ 0.
            watershed_dilate:
                Percentage (0-100) of max distance for marker threshold.
                Higher values = more conservative separation (fewer, larger regions).
                Lower values = more aggressive separation (more, smaller regions).
        
        Returns:
            labeled_image: int32 label image (0 = background, 1..N bacteria).
            contours: List of contours (one per labeled object).
        """
        # Ensure proper uint8 binary
        if binary.dtype != np.uint8:
            binary_u8 = binary.astype(np.uint8)
        else:
            binary_u8 = binary.copy()
        
        # Distance transform
        distance = cv2.distanceTransform(binary_u8, cv2.DIST_L2, 5)
        
        # Handle empty images
        max_dist = distance.max()
        if max_dist == 0:
            h, w = binary_u8.shape[:2]
            return np.zeros((h, w), dtype=np.int32), []
        
        # Use watershed_dilate as percentage threshold (matching dev.py)
        threshold_val = (watershed_dilate / 100.0) * max_dist
        _, sure_fg = cv2.threshold(distance, threshold_val, 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        
        # Additional opening to separate touching markers (matching dev.py)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Label connected components as markers
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers.astype(np.int32)
        markers = markers + 1  # Shift labels (background becomes 1)
        markers[binary_u8 == 0] = 0  # True background is 0
        
        # Watershed expects 3-channel image
        ws_input = cv2.cvtColor(binary_u8, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(ws_input, markers)
        
        # Create labeled image: boundaries (-1) → 0
        labeled_image = markers.copy()
        labeled_image[labeled_image == -1] = 0
        
        # Extract contours for each labeled region (labels 2..max)
        contours: List[np.ndarray] = []
        max_label = int(markers.max())
        for label_id in range(2, max_label + 1):
            mask = (markers == label_id).astype(np.uint8) * 255
            res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = res[-2]
            contours.extend(cnts)
        
        return labeled_image, contours