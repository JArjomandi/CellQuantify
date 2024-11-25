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

    # Convert image to grayscale for white/gray analysis
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define the white/gray intensity range
    min_white_gray_intensity = 20  # Minimum intensity (#1c1c1c)
    max_white_gray_intensity = 255  # Maximum intensity (#595959)
    white_gray_mask = (gray_image >= min_white_gray_intensity) & (gray_image <= max_white_gray_intensity)

    # Calculate metrics for white/gray areas
    total_white_gray_pixels = np.sum(white_gray_mask)  # Count all pixels in the mask
    total_white_gray_area_um2 = total_white_gray_pixels * (pixel_size_um ** 2)
    total_white_gray_area_mm2 = total_white_gray_area_um2 / 1e6
    total_white_gray_area_percent = (total_white_gray_pixels / image_area) * 100

    # Calculate average intensity of white/gray pixels
    white_gray_intensity_values = gray_image[white_gray_mask]
    avg_white_gray_intensity = np.mean(white_gray_intensity_values) if white_gray_intensity_values.size > 0 else 0

    # Existing analysis for DAPI nuclei blue cells
    blue_channel = img[:, :, 0]
    if np.mean(blue_channel) < 5:  # Skip blue cell analysis if the mean intensity is too low (no signal)
        num_blue_cells = 0
        total_blue_area_pixels = 0
        total_blue_area_percent = 0
        total_blue_area_um2 = 0
        total_blue_area_mm2 = 0
    else:
        blue_threshold = filters.threshold_otsu(blue_channel)
        blue_cells_mask = blue_channel > blue_threshold
        blue_cells_mask = morphology.remove_small_objects(blue_cells_mask, min_size=60)
        labeled_blue_cells, num_blue_cells = measure.label(blue_cells_mask, return_num=True)
        blue_cells_props = measure.regionprops(labeled_blue_cells, intensity_image=blue_channel)

        total_blue_area_pixels = np.sum([prop.area for prop in blue_cells_props])
        total_blue_area_percent = (total_blue_area_pixels / image_area) * 100
        total_blue_area_um2 = total_blue_area_pixels * (pixel_size_um ** 2)
        total_blue_area_mm2 = total_blue_area_um2 / 1e6

    # Existing analysis for red aggregates
    red_channel = img[:, :, 2]
    green_channel = img[:, :, 1]
    min_red_intensity = 100
    max_red_intensity = 255

    red_aggregates_mask = (
        (red_channel >= min_red_intensity) &
        (red_channel <= max_red_intensity) &
        (red_channel > green_channel * 1.5) &
        (red_channel > blue_channel * 1.5)
    )

    red_aggregates_mask = morphology.binary_opening(red_aggregates_mask, morphology.disk(1))
    red_aggregates_mask = morphology.binary_closing(red_aggregates_mask, morphology.disk(1))

    distance = ndi.distance_transform_edt(red_aggregates_mask)
    local_maxi = morphology.local_maxima(distance)
    markers, _ = ndi.label(local_maxi)
    labels = watershed(-distance, markers, mask=red_aggregates_mask)

    red_aggregates_props = measure.regionprops(labels, intensity_image=red_channel)

    num_red_aggregates = len(red_aggregates_props)
    total_red_area_pixels = np.sum([prop.area for prop in red_aggregates_props])
    total_red_area_percent = (total_red_area_pixels / image_area) * 100
    total_red_area_um2 = total_red_area_pixels * (pixel_size_um ** 2)
    total_red_area_mm2 = total_red_area_um2 / 1e6

    # Collect results
    results = {
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
        "Total White/Gray Pixels": total_white_gray_pixels,
        "Total White/Gray Area (um^2)": total_white_gray_area_um2,
        "Total White/Gray Area (mm^2)": total_white_gray_area_mm2,
        "Total White/Gray Area (% of image)": total_white_gray_area_percent,
        "Average White/Gray Intensity": avg_white_gray_intensity,
    }
    return results


def analyze_images_in_nested_folders(base_path, output_excel_path, pixel_size_um):
    all_results = []

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".tif"):
                image_path = os.path.join(root, file)
                # Get the relative path parts
                relative_path = os.path.relpath(image_path, base_path)
                path_parts = relative_path.split(os.sep)[:-1]  # Exclude the file name for folders

                # Analyze the image and gather results
                results = analyze_image(image_path, pixel_size_um)

                # Add folder structure to results
                folder_structure = {f"Folder Level {i+1}": part for i, part in enumerate(path_parts)}
                folder_structure["Image Name"] = file
                results = {**folder_structure, **results}

                all_results.append(results)

    # Convert results to a DataFrame and save to Excel
    df = pd.DataFrame(all_results)
    df.to_excel(output_excel_path, index=False)
    print(f"Results saved to {output_excel_path}")


if __name__ == "__main__":
    base_path = "E:\\MSc\\FAU\\LESSONS masters\\Master Project\\MN lab training\\Denise\\Thy1 d10"
    output_excel_path = "E:\\MSc\\FAU\\LESSONS masters\\Master Project\\MN lab training\\Denise\\analysis_results.xlsx"
    pixel_size_um = 0.08  # Physical pixel size in micrometers

    analyze_images_in_nested_folders(base_path, output_excel_path, pixel_size_um)
