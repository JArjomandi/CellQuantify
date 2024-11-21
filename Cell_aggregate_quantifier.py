import cv2
import numpy as np
from skimage import measure, morphology, filters
import matplotlib.pyplot as plt


def analyze_image(image_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Split the channels (assuming BGR format)
    blue_channel = img[:, :, 0]  # Blue channel
    red_channel = img[:, :, 2]  # Red channel

    # Detect blue cells
    blue_threshold = filters.threshold_otsu(blue_channel)
    blue_cells_mask = blue_channel > blue_threshold
    blue_cells_mask = morphology.remove_small_objects(blue_cells_mask, min_size=50)
    labeled_blue_cells, num_blue_cells = measure.label(blue_cells_mask, return_num=True)

    # Detect red aggregates
    red_threshold = filters.threshold_otsu(red_channel) * 1.2  # Slightly higher threshold for bright spots
    red_aggregates_mask = red_channel > red_threshold
    red_aggregates_mask = morphology.remove_small_objects(red_aggregates_mask, min_size=5)

    # Label and measure properties of red aggregates
    labeled_red_aggregates = measure.label(red_aggregates_mask)
    red_aggregates_props = measure.regionprops(labeled_red_aggregates, intensity_image=red_channel)

    # Count red aggregates and measure diameters
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


# Run the analysis on  .tif image
analyze_image("path_to_image.tif")

