

<img src="icon.png" width="180"/>

# QGeoAI Toolkit

### AI-Powered Remote Sensing & Machine Learning Plugin for QGIS

![QGIS](https://img.shields.io/badge/QGIS-3.38+-green?logo=qgis)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-success)
![OSGeo4W](https://img.shields.io/badge/OSGeo4W-Shell-blue)
![NumPy](https://img.shields.io/badge/NumPy-Required-orange?logo=numpy)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-Required-F7931E?logo=scikitlearn)
![Joblib](https://img.shields.io/badge/Joblib-Required-lightgrey)

---
**Open-source QGIS plugin for GeoAI, Remote Sensing, Earth Observation and Machine Learning.**

---

Developed by **Dimitra Pappa**

</div>

---

# Overview

QGeoAI Toolkit is an open-source QGIS plugin that integrates Artificial Intelligence, Remote Sensing and Machine Learning directly into QGIS.

The plugin enables users to calculate spectral indices, perform supervised image classification and analyze multispectral satellite imagery using modern GeoAI techniques.

Designed for researchers, GIS professionals and students working with Earth Observation data.

---

# Features

## Spectral Indices

| Index | Description |
|--------|-------------|
| NDVI | Normalized Difference Vegetation Index |
| GNDVI | Green Normalized Difference Vegetation Index |
| NDWI | Normalized Difference Water Index |
| SAVI | Soil Adjusted Vegetation Index |
| NDMI | Normalized Difference Moisture Index |
| NBR | Normalized Burn Ratio |

---

## Machine Learning

✔ Random Forest Classification

✔ Support Vector Machine (SVM)

✔ Training Polygon Support

✔ Classification Report

✔ Accuracy Assessment

✔ GeoTIFF Export

---

# Supported Satellite Data

| Dataset | Supported |
|----------|-----------|
| Sentinel-2 | ✅ |
| Landsat | ✅ |
| GeoTIFF | ✅ |
| Virtual Raster (VRT) | ✅ |
| Multiband Raster | ✅ |

---

# Workflow

```text
Satellite Image
        │
        ▼
Load Multiband Raster
        │
        ▼
Calculate Spectral Index
        │
        ▼
Create Training Areas
        │
        ▼
Train Random Forest / SVM
        │
        ▼
Generate Classified Raster
        │
        ▼
Export GeoTIFF
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/demetrapappa14-geospatial/QGeoAI-Toolkit.git
```

Open your QGIS Plugins folder and copy the plugin directory.

Enable the plugin from:

```
Plugins → Manage and Install Plugins
```

---

# Spectral Indices

| Index | Formula | Sentinel-2 Bands |
|--------|---------|------------------|
| NDVI | (NIR - Red)/(NIR + Red) | B08 / B04 |
| GNDVI | (NIR - Green)/(NIR + Green) | B08 / B03 |
| NDWI | (Green - NIR)/(Green + NIR) | B03 / B08 |
| SAVI | Soil Adjusted Vegetation Index | B08 / B04 |
| NDMI | (NIR - SWIR1)/(NIR + SWIR1) | B08 / B11 |
| NBR | (NIR - SWIR2)/(NIR + SWIR2) | B08 / B12 |

---

# Machine Learning

The plugin currently supports:

| Algorithm | Status |
|------------|--------|
| Random Forest | ✅ |
| Support Vector Machine | ✅ |

Future versions will include additional GeoAI methods.

---

# Screenshots

## Main Interface

The QGeoAI Toolkit provides a user-friendly interface for spectral index computation and AI-based image classification directly within QGIS.

<table>
<tr>

<td align="center">

<b>Machine Learning Module</b><br><br>

<img src="interface_1.png" width="450"/>

</td>

<td align="center">

<b>Spectral Indices Module</b><br><br>

<img src="interface_2.png" width="450"/>

</td>

</tr>
</table>

------

# Project Structure

```text
QGeoAI-Toolkit
│
├── __init__.py
├── metadata.txt
├── qgeoai_plugin.py
├── qgeoai_dialog.py
├── ml_engine.py
├── raster_utils.py
├── indices.py
├── icon.png
├── README.md
└── LICENSE
```

---

# Requirements

| Software | Version |
|-----------|---------|
| QGIS | 3.38 or newer |
| Python | 3.10+ |
| NumPy | Latest |
| Rasterio | Latest |
| Scikit-Learn | Latest |
| Joblib | Latest |

---
# Python Libraries

The plugin is built using the following Python libraries:

| Library | Purpose |
|----------|---------|
| NumPy | Numerical array operations |
| Rasterio | Reading and writing raster datasets |
| Scikit-Learn | Machine Learning algorithms |
| Joblib | Model serialization |
| PyQt5 | Graphical User Interface |
| QGIS Python API | GIS functionality and layer management |

---

# Installation 
QGeoAI Toolkit can currently be installed manually from Github repository or Qgis Plugins.The following libraries must be installed before using the **Random Forest** and **Support Vector Machine (SVM)** modules.

| Library | Required | Purpose |
|----------|----------|---------|
| NumPy | ✅ | Numerical computations and raster array operations |
| Scikit-learn | ✅ | Random Forest, SVM and machine learning algorithms |
| Joblib | ✅ | Saving and loading trained machine learning models |

---
## Installing the required libraries 
The libraries must be installed **inside the Qgis python enviroment**.
## Step 1
Close QGIS.

## Step 2
Open **OSGeo4W Shell**.

## Step 3

Upgrade **pip** (optional but recommended).

```bash
python -m pip install --upgrade pip
```

## Step 4

Install the required libraries.

```bash
python -m pip install numpy scikit-learn joblib
```


### Step 5

Restart QGIS.

---

## Verify the installation

Open **OSGeo4W Shell** and execute

```bash
python -c "import numpy, sklearn, joblib; print('Installation completed successfully.')"
```

If no error appears, all required libraries have been installed correctly.

---

> **Important**
>
> If **Scikit-learn** is not installed, the spectral index tools will still work correctly.
>
> However, the **Random Forest** and **Support Vector Machine (SVM)** classifiers will not run and QGIS may display the following error:
>
> ```text
> No module named 'sklearn'
> ```

# Roadmap

- Deep Learning Classification
- Object Detection
- Change Detection
- SAR Processing
- Explainable AI (XAI)
- Automatic Feature Selection
- GPU Acceleration
- Batch Processing

---

# Contributing

Contributions are welcome!

Feel free to:

- Report issues
- Suggest new features
- Submit pull requests
- Improve documentation

---

# Citation

If you use QGeoAI Toolkit in your research, please cite this repository.

---


# License

This project is distributed under the **MIT License**.

---

# Author

**Dimitra Pappa**

Surveying Engineer & Geoinformatics

MSc Artificial Intelligence & Visual Computing

GeoAI • Remote Sensing • GIS • Earth Observation

GitHub:

https://github.com/demetrapappa14-geospatial/QGeoAI-Toolkit

--

<div align="center">

⭐ If you find this project useful, consider giving it a star!

</div>
