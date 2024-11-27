import cv2
import numpy as np
import os
import pandas as pd
from skimage import measure, morphology, filters, segmentation
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
from scipy import ndimage as ndi


def analyze_image(image_path, pixel_size_um):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_area = img.shape[0] * img.shape[1]  # Total area in pixels

    # Convert image to grayscale for white/gray analysis
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define the white/gray intensity range
    min_white_gray_intensity = 20
    max_white_gray_intensity = 255
    white_gray_mask = (gray_image >= min_white_gray_intensity) & (gray_image <= max_white_gray_intensity)

    # Calculate metrics for white/gray areas
    total_white_gray_pixels = np.sum(white_gray_mask)
    total_white_gray_area_um2 = total_white_gray_pixels * (pixel_size_um ** 2)
    total_white_gray_area_mm2 = total_white_gray_area_um2 / 1e6
    total_white_gray_area_percent = (total_white_gray_pixels / image_area) * 100

    # Calculate average intensity of white/gray pixels
    white_gray_intensity_values = gray_image[white_gray_mask]
    avg_white_gray_intensity = np.mean(white_gray_intensity_values) if white_gray_intensity_values.size > 0 else 0

    # Blue cell detection with stricter criteria to avoid over-segmentation
    blue_channel = img[:, :, 0]  # Extract the blue channel
    green_channel = img[:, :, 1]
    red_channel = img[:, :, 2]

    # Blue cell detection with stricter criteria to avoid over-segmentation
    min_blue_intensity = 70  # Set a stricter minimum intensity for blue cells
    max_blue_intensity = 255  # Keep maximum intensity as 255

    # Create a mask for blue cells within the specified intensity range
    blue_cells_mask = (
            (blue_channel >= min_blue_intensity) &
            (blue_channel <= max_blue_intensity) &
            (blue_channel > green_channel * 1.5) &  # Blue must be 1.5x stronger than green
            (blue_channel > red_channel * 1.5)  # Blue must be 1.5x stronger than red
    )

    # Apply morphological operations to clean up the mask
    blue_cells_mask = morphology.binary_opening(blue_cells_mask, morphology.disk(2))  # Remove noise
    blue_cells_mask = morphology.binary_closing(blue_cells_mask, morphology.disk(2))  # Fill small holes

    # Filter out small objects (non-cell areas)
    blue_cells_mask = morphology.remove_small_objects(blue_cells_mask, min_size=70)

    # Check if there is a valid signal in the blue channel
    background_intensity = np.median(blue_channel)  # Background noise level
    peak_intensity = np.max(blue_channel[blue_cells_mask]) if np.any(blue_cells_mask) else 0

    # Initialize blue_cells_props to avoid UnboundLocalError
    blue_cells_props = []

    if peak_intensity - background_intensity < 60:  # SNR threshold
        # If the signal-to-noise ratio is too low, set results to zero
        num_blue_cells = 0
        total_blue_area_pixels = 0
        total_blue_area_percent = 0
        total_blue_area_um2 = 0
        total_blue_area_mm2 = 0
    else:
        # Perform distance transform
        distance = distance_transform_edt(blue_cells_mask)

        # Identify local maxima in the distance map
        local_maxi = morphology.h_maxima(distance, h=10)

        # Label maxima to use as markers
        markers, _ = measure.label(local_maxi, return_num=True)

        # Apply watershed segmentation to separate connected regions
        labeled_blue_cells = segmentation.watershed(-distance, markers, mask=blue_cells_mask)

        # Measure properties of the segmented blue cells
        blue_cells_props = measure.regionprops(labeled_blue_cells, intensity_image=blue_channel)

        # Count and calculate areas
        num_blue_cells = len(blue_cells_props)
        total_blue_area_pixels = np.sum([prop.area for prop in blue_cells_props])
        total_blue_area_percent = (total_blue_area_pixels / image_area) * 100
        total_blue_area_um2 = total_blue_area_pixels * (pixel_size_um ** 2)
        total_blue_area_mm2 = total_blue_area_um2 / 1e6

    # Red aggregate analysis remains unchanged
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

    # Visualization of the detected regions #######################################
    overlay = img.copy()

    # Apply a yellow semi-transparent mask for detected blue cells
    blue_mask_overlay = np.zeros_like(img, dtype=np.uint8)
    blue_mask_overlay[blue_cells_mask] = [0, 255, 255]  # Yellow mask for blue cells
    overlay = cv2.addWeighted(overlay, 0.8, blue_mask_overlay, 0.5, 0)  # Blend with transparency

    # Draw bright green circles around detected blue cells
    for prop in blue_cells_props:
        y, x = prop.centroid
        radius = prop.equivalent_diameter / 2
        cv2.circle(overlay, (int(x), int(y)), int(radius), (0, 255, 0), 2)  # Bright green circle

    # Draw yellow circles around detected red aggregates
    for prop in red_aggregates_props:
        y, x = prop.centroid
        radius = prop.equivalent_diameter / 2
        cv2.circle(overlay, (int(x), int(y)), int(radius), (0, 255, 255), 2)  # Yellow circle for red aggregates

    # Overlay white mask for detected white/gray intensity areas
    white_mask_overlay = np.zeros_like(img, dtype=np.uint8)
    white_mask_overlay[white_gray_mask] = [255, 255, 255]  # White mask for white/gray regions
    overlay = cv2.addWeighted(overlay, 0.8, white_mask_overlay, 0.2, 0)  # Blend with transparency

    # Resize the visualization for better screen fit
    window_width = 500  # Desired window width
    window_height = 400  # Desired window height
    overlay_resized = cv2.resize(overlay, (window_width, window_height))

    # Show the resized visualization
    cv2.imshow(f"Visualization - {os.path.basename(image_path)}", overlay_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    #base_path = "E:\\MSc\\FAU\\LESSONS masters\\Master Project\\MN lab training\\Denise\\Thy1 d10"
    base_path = "E:\MSc\FAU\LESSONS masters\Master Project\MN lab training"
    #output_excel_path = "E:\\MSc\\FAU\\LESSONS masters\\Master Project\\MN lab training\\Denise\\analysis_results.xlsx"
    output_excel_path = "E:\\MSc\\FAU\\LESSONS masters\\Master Project\\MN lab training\\Denise\\analysis_results_test.xlsx"
    pixel_size_um = 0.08  # Physical pixel size in micrometers

    analyze_images_in_nested_folders(base_path, output_excel_path, pixel_size_um)
