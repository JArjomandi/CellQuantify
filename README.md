# CellQuantify 
## Quantification of Nuclei and Alpha-Synuclein Aggregates Using Computational Image Analysis
A tool for automated analysis of microscope images, extracting quantitative metrics such as cell counts, aggregate areas, intensity values and distributions (e.g staining), and spatial coverage. Designed for bioimage data in microscopy, with support for pixel-to-physical unit conversions and batch processing.

## Image Acquisition and Preparation
Images for the quantification of DAPI-stained nuclei, alpha-Synuclein (α-Syn) aggregates, and white/grey intensity areas were captured from brain sections of mice subjected to immunohistochemical staining. Four distinct channels were included in each set of images:

- **DAPI (blue)**  
- **α-Synuclein staining 1 (red)**  
- **α-Synuclein staining 2 (white/grey)**  
- **Cathepsin B/L or control PBS (green)**  

Additionally, a merged image containing all channels was provided. All images were captured in TIFF format at a fixed pixel resolution of 0.08 μm x 0.08 μm, providing high spatial resolution for quantitative analysis. This level of resolution allows precise segmentation and measurement of subcellular features (Schindelin et al., 2012).

---

## Image Analysis Pipeline
Image analysis was performed using a custom Python-based pipeline leveraging the **OpenCV** and **SciKit-Image** libraries for efficient and reproducible data processing (van der Walt et al., 2014). The pipeline included preprocessing, segmentation, and quantification steps to extract meaningful metrics from the images.

### 1. Preprocessing and Channel-Specific Analysis
- Each TIFF image was loaded as a three-channel RGB image using **OpenCV**.
- Specific channels corresponding to DAPI, red α-Synuclein staining, and white/grey α-Synuclein staining were extracted for further analysis.
- Conversion to grayscale was performed for the white/grey intensity channel to simplify pixel-level intensity calculations (Schindelin et al., 2012).

Preprocessing steps:
- Intensity normalization and filtering to enhance signal-to-noise ratio.
- Morphological operations to remove small specks and fill small holes in detected regions.

---

### 2. Detection of DAPI-Positive Nuclei
- **Blue channel** analysis for DAPI-stained nuclei.
- Intensity thresholds:  
  - \( B \geq 70 \, \& \, B \leq 255 \)  
  - \( B > 1.5 \cdot G \, \& \, B > 1.5 \cdot R \)  

- Morphological operations: Binary opening and closing (disk size = 2).
- Watershed segmentation to separate connected nuclei:
  \[
  D(x, y) = \text{min}\{ \text{Euclidean Distance from } (x, y) \text{ to nearest zero pixel in the mask} \}
  \]
- Segmented regions were labeled, and nuclei count was computed.  
- Total nuclear area:
  \[
  \text{Area in } \mu m^2 = \text{Pixel Count} \times (0.08 \, \mu m)^2
  \]

---

### 3. Detection and Quantification of Red Alpha-Synuclein Aggregates
- **Red channel** analysis with thresholds:  
  - \( R \geq 100 \, \& \, R \leq 255 \)  
  - \( R > 1.5 \cdot G \, \& \, R > 1.5 \cdot B \)  

- Binary opening and closing operations applied.
- Watershed segmentation to separate connected aggregates.  
- Total aggregate area:
  \[
  \text{Area in } \mu m^2 = \text{Pixel Count} \times (0.08 \, \mu m)^2
  \]

---

### 4. Analysis of White/Grey Intensity Regions
- Grayscale analysis for secondary α-Synuclein staining.
- Binary mask created for intensities between 20 and 255.
- Metrics calculated:
  - Total area:
    \[
    \text{Area in } \mu m^2 = \text{Pixel Count} \times (0.08 \, \mu m)^2
    \]
  - Mean intensity:
    \[
    \text{Mean Intensity} = \frac{\sum \text{Pixel Intensities in Mask}}{\text{Number of Pixels in Mask}}
    \]

---

### 5. Recursive Folder Analysis and Results Compilation
- Recursive folder traversal performed using Python’s `os.walk()` function.
- Metadata (e.g., animal ID, treatment group) extracted for each image.
- Quantitative metrics saved to an Excel file using the **pandas** library.

---

## References
1. **Schindelin, J., Arganda-Carreras, I., Frise, E., et al. (2012)**  
   Fiji: An open-source platform for biological-image analysis. *Nature Methods*, 9(7), 676–682.  
   [https://doi.org/10.1038/nmeth.2019](https://doi.org/10.1038/nmeth.2019)

2. **van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., et al. (2014)**  
   scikit-image: Image processing in Python. *PeerJ*, 2, e453.  
   [https://doi.org/10.7717/peerj.453](https://doi.org/10.7717/peerj.453)

3. **Preibisch, S., Saalfeld, S., & Tomancak, P. (2009)**  
   Globally optimal stitching of tiled 3D microscopic image acquisitions. *Bioinformatics*, 25(11), 1463–1465.  
   [https://doi.org/10.1093/bioinformatics/btp184](https://doi.org/10.1093/bioinformatics/btp184)

---

This pipeline enabled reproducible and accurate quantification of nuclei and α-Synuclein aggregates in wildtype and α-Synuclein-overexpressing mice subjected to Cathepsin treatments.
![Sample Image](samples.png)

--------------------------------------------
# 🧪 Image Segmentation Utilities

This repository contains Python utilities for performing image segmentation and analysis using OpenCV, NumPy, SciPy, and scikit-image.

---

## ✅ Requirements

The code depends on the following Python libraries:

- `opencv-python`
- `numpy`
- `pandas`
- `scikit-image`
- `scipy`

These are listed in the `requirements.txt` file provided in this repo.

---

## 🚀 Getting Started (Beginner-Friendly Setup)

This guide is for **complete beginners** who want to run this code using **Python and PyCharm**.

---

### 1️⃣ Install Python 🐍

- Download Python: [https://www.python.org/downloads/](https://www.python.org/downloads/)
- During installation, **make sure to check this box**:  
  ✅ *"Add Python to PATH"*

---

### 2️⃣ Install PyCharm 💡

- Download PyCharm (Community Edition is free):  
  [https://www.jetbrains.com/pycharm/download/](https://www.jetbrains.com/pycharm/download/)

---

### 3️⃣ Open the Project in PyCharm

1. Open PyCharm
2. Click **"Open"**
3. Select the folder where you downloaded or cloned this repository

---

### 4️⃣ Set Up a Virtual Environment in PyCharm

1. Go to  
   **File > Settings > Project: YourProjectName > Python Interpreter**
2. Click the ⚙️ gear icon and choose **Add**
3. Select **Virtualenv Environment**
4. Click **OK** to create a clean environment for the project

---

### 5️⃣ Install Required Python Packages

1. Open the **Terminal** in PyCharm (bottom of the screen)
2. Run this command:

```bash
pip install -r requirements.txt
```

---
## ▶️ Running the Code

- Open the Python script (.py) you want to run
- Set required paths
- Click the green **▶️ Run** button in the top right of PyCharm


