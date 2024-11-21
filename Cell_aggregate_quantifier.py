import cv2
import numpy as np
from skimage import measure, morphology, filters
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

def analyze_image(image_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Split the channels (assuming BGR format)
    blue_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    red_channel = img[:, :, 2]

    # Detect blue cells
    blue_threshold = filters.threshold_otsu(blue_channel)
    blue_cells_mask = blue_channel > blue_threshold
    blue_cells_mask = morphology.remove_small_objects(blue_cells_mask, min_size=70)
    labeled_blue_cells, num_blue_cells = measure.label(blue_cells_mask, return_num=True)
    blue_cells_props = measure.regionprops(labeled_blue_cells, intensity_image=blue_channel)

    # Define acceptable red intensity range around #bf0006
    min_red_intensity = 100  # Lower bound
    max_red_intensity = 255  # Upper bound for bright red tones

    # Create a mask for red aggregates within the acceptable intensity range
    red_aggregates_mask = (
        (red_channel >= min_red_intensity) &
        (red_channel <= max_red_intensity) &
        (red_channel > green_channel * 1.5) &
        (red_channel > blue_channel * 1.5)
    )

    # Apply morphology to refine the mask
    red_aggregates_mask = morphology.binary_opening(red_aggregates_mask, morphology.disk(1))
    red_aggregates_mask = morphology.binary_closing(red_aggregates_mask, morphology.disk(1))

    # Distance transform for watershed segmentation
    distance = ndi.distance_transform_edt(red_aggregates_mask)
    local_maxi = morphology.local_maxima(distance)
    markers, _ = ndi.label(local_maxi)
    labels = watershed(-distance, markers, mask=red_aggregates_mask)

    # Measure properties of red aggregates after watershed
    red_aggregates_props = measure.regionprops(labels, intensity_image=red_channel)

    # Count remaining red aggregates and measure diameters
    red_aggregates_diameters = [prop.equivalent_diameter for prop in red_aggregates_props]
    num_red_aggregates = len(red_aggregates_diameters)

    # Plot the original image with detections
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Draw circles around each blue cell
    for prop in blue_cells_props:
        y, x = prop.centroid
        diameter = prop.equivalent_diameter / 2
        circle = plt.Circle((x, y), diameter, color='green', fill=False, linewidth=1)
        ax.add_patch(circle)

    # Draw circles around each red aggregate
    for prop in red_aggregates_props:
        y, x = prop.centroid
        diameter = prop.equivalent_diameter / 2
        circle = plt.Circle((x, y), diameter, color='yellow', fill=False, linewidth=1)
        ax.add_patch(circle)

    # Display counts on the plot
    ax.set_title(f"Blue cells: {num_blue_cells}, Red aggregates: {num_red_aggregates}")
    plt.axis('off')
    plt.show()

    # Print results
    print(f"Number of blue cells: {num_blue_cells}")
    print(f"Number of red aggregates: {num_red_aggregates}")
    print("Diameters of red aggregates:", red_aggregates_diameters)

# Run the analysis on your .tif images
analyze_image("E:\\MSc\\FAU\\LESSONS masters\\Master Project\\MN lab training\\2024-10-09 Thy1 no3-CTSB 488_LB509 568_MJFR14 647-CSTB_PBS- 5_c2+4.tif")
analyze_image("E:\\MSc\\FAU\\LESSONS masters\\Master Project\\MN lab training\\2024-10-09 Thy1 no4-CTSB 488_LB509 568_MJFR14 647-CSTB 2_c2+4.tif")
analyze_image("E:\\MSc\\FAU\\LESSONS masters\\Master Project\\MN lab training\\2024-10-09 Thy1 no4-CTSB 488_LB509 568_MJFR14 647-CSTB 2_c2.tif")
analyze_image("E:\\MSc\\FAU\\LESSONS masters\\Master Project\\MN lab training\\2024-10-09 Thy1 no3-CTSB 488_LB509 568_MJFR14 647-CSTB_PBS- 5_c2.tif")
