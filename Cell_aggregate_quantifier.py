import cv2
import numpy as np
import os
import pandas as pd
from skimage import measure, morphology, filters
from skimage.segmentation import watershed
from scipy import ndimage as ndi


def analyze_image(image_path, pixel_size_um):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_area = img.shape[0] * img.shape[1]  # Total area in pixels

    # Split the channels (assuming BGR format)
    blue_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    red_channel = img[:, :, 2]

    # Detect blue cells (original method)
    blue_threshold = filters.threshold_otsu(blue_channel)
    blue_cells_mask = blue_channel > blue_threshold
    blue_cells_mask = morphology.remove_small_objects(blue_cells_mask, min_size=60)
    labeled_blue_cells, num_blue_cells = measure.label(blue_cells_mask, return_num=True)
    blue_cells_props = measure.regionprops(labeled_blue_cells, intensity_image=blue_channel)

    # Calculate total area of blue cells
    total_blue_area_pixels = np.sum([prop.area for prop in blue_cells_props])
    total_blue_area_percent = (total_blue_area_pixels / image_area) * 100
    total_blue_area_um2 = total_blue_area_pixels * (pixel_size_um ** 2)
    total_blue_area_mm2 = total_blue_area_um2 / 1e6  # Convert to mm^2

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

    # Count remaining red aggregates
    num_red_aggregates = len(red_aggregates_props)

    # Calculate total area of red aggregates
    total_red_area_pixels = np.sum([prop.area for prop in red_aggregates_props])
    total_red_area_percent = (total_red_area_pixels / image_area) * 100
    total_red_area_um2 = total_red_area_pixels * (pixel_size_um ** 2)
    total_red_area_mm2 = total_red_area_um2 / 1e6  # Convert to mm^2

    # Collect the results for this image
    results = {
        "Image Name": os.path.basename(image_path),
        "Number of Blue Cells": num_blue_cells,
        "Total Blue Cell Area (pixels)": total_blue_area_pixels,
        "Total Blue Cell Area (% of image)": total_blue_area_percent,
        "Total Blue Cell Area (um^2)": total_blue_area_um2,
        "Total Blue Cell Area (mm^2)": total_blue_area_mm2,
        "Number of Red Aggregates": num_red_aggregates,
        "Total Red Aggregate Area (pixels)": total_red_area_pixels,
        "Total Red Aggregate Area (% of image)": total_red_area_percent,
        "Total Red Aggregate Area (um^2)": total_red_area_um2,
        "Total Red Aggregate Area (mm^2)": total_red_area_mm2,
    }

    return results


def analyze_images_in_directory(directory_path, output_excel_path, pixel_size_um):
    # List to store the results for each image
    all_results = []

    # Find all .tif files in the given directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".tif"):
            image_path = os.path.join(directory_path, filename)
            # Analyze the image and store the results
            results = analyze_image(image_path, pixel_size_um)
            all_results.append(results)

    # Create a DataFrame from the results
    df = pd.DataFrame(all_results)

    # Save the DataFrame to an Excel file
    df.to_excel(output_excel_path, index=False)

    print(f"Results saved to {output_excel_path}")


if __name__ == "__main__":
    # Define the directory path containing the images and the output Excel file path
    directory_path = "E:\\MSc\\FAU\\LESSONS masters\\Master Project\\MN lab training\\"
    output_excel_path = "E:\\MSc\\FAU\\LESSONS masters\\Master Project\\MN lab training\\analysis_tif_results.xlsx"

    # Define the physical pixel size in micrometers
    pixel_size_um = 0.08  # Each pixel is 0.08 x 0.08 um

    # Analyze all images in the directory and save results to Excel
    analyze_images_in_directory(directory_path, output_excel_path, pixel_size_um)
