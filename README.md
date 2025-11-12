List of Configurations/Parameters: 
Behavior of Parameter Changes: 

Watershed & Filtering: Watershed Dilate (integer, default: 15, range: 1-20)

Description: Percentage threshold for creating sure foreground markers in the watershed algorithm, which uses distance transform to separate touching or clumped objects into individual contours.
Behavior on change:
Increasing: Creates smaller, more conservative foreground markers, leading to greater separation of objects. This results in more individual contours (better splits clumped bacteria) but can cause over-segmentation or fragmentation if too high.
Decreasing: Creates larger markers, reducing separation and potentially merging nearby objects into fewer, larger contours. Useful for under-segmented images but risks failing to separate touching bacteria.


Watershed & Filtering: Min Area (px²) (integer, default: 50, range: 10-500)

Description: Minimum area (in pixels) required for a contour to be considered a valid bacterium; smaller contours are filtered out after detection.
Behavior on change:
Increasing: Stricter filtering, discarding more small contours (e.g., noise or debris), resulting in fewer detected bacteria but cleaner results. May miss legitimate small bacteria.
Decreasing: More inclusive, retaining smaller contours and increasing the number of detected bacteria, but introduces more false positives like artifacts.


Fluorescence: Min Fluor/Area (float, default: 10.0, range: 0-255, step: 0.1)

Description: Minimum fluorescence intensity per unit area (total fluorescence divided by contour area) to retain a bacterium; only applies if a fluorescence image is loaded. This filters contours post-detection based on fluorescence data.
Behavior on change:
Increasing: More selective, removing contours with low fluorescence density (e.g., dim or non-fluorescent bacteria), reducing the number of displayed contours to focus on "active" ones.
Decreasing: Less filtering, including contours with weaker or no fluorescence, increasing the number of retained contours but potentially including irrelevant ones.

