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
    blue_channel = img[:, :, 0]  # Blue channel
    green_channel = img[:, :, 1]  # Green channel
    red_channel = img[:, :, 2]  # Red channel

    # Detect blue cells
    blue_threshold = filters.threshold_otsu(blue_channel)
    blue_cells_mask = blue_channel > blue_threshold
    blue_cells_mask = morphology.remove_small_objects(blue_cells_mask, min_size=50)
    labeled_blue_cells, num_blue_cells = measure.label(blue_cells_mask, return_num=True)

    # Set minimum red intensity for counting
    min_red_intensity = 191

    # Create a mask for red aggregates based on specific color criteria
    red_aggregates_mask = (
            (red_channel >= min_red_intensity) &  # Red intensity threshold
            (red_channel > green_channel * 1.5) &  # Red significantly stronger than green
            (red_channel > blue_channel * 1.5)  # Red significantly stronger than blue
    )

    # Apply morphology to refine the mask
    red_aggregates_mask = morphology.remove_small_objects(red_aggregates_mask, min_size=5)

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


# Run the analysis on your .tif image
analyze_image("path_to_image.tif")
