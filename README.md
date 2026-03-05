# CNN-Based Obstacle Detection and Mapping from UAV Imagery for RIS Placement

[![Open high_alt In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shahmehul2005/RIS-placement-using-CNN/blob/main/high_alt.ipynb)
[![Open low_alt In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shahmehul2005/RIS-placement-using-CNN/blob/main/low_alt.ipynb)
[![Open mapping In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shahmehul2005/RIS-placement-using-CNN/blob/main/mapping.ipynb)

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Phase 1 — Label Preparation (`low_alt.ipynb`)](#phase-1--label-preparation-low_altipynb)
  - [Phase 2 — Model Training (`low_alt.ipynb`)](#phase-2--model-training-low_altipynb)
  - [Phase 3 — High-Altitude Inference (`high_alt.ipynb`)](#phase-3--high-altitude-inference-high_altipynb)
  - [Phase 4 — Geospatial Mapping (`mapping.ipynb`)](#phase-4--geospatial-mapping-mappingipynb)
- [Model Details](#model-details)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project develops a **CNN-based obstacle detection and mapping system** that processes UAV (drone) imagery to generate accurate 2D geospatial obstacle maps. These maps are designed to assist engineers in the **optimal placement of Reconfigurable Intelligent Surfaces (RIS)** in urban and semi-urban environments.

RIS technology enhances wireless communications by reflecting signals around obstacles. Identifying the type, location, and spatial extent of obstacles (buildings, trees, towers) is critical for determining viable RIS mounting positions and ensuring clear line-of-sight (LOS) paths.

**Key capabilities:**
- Detects and classifies 5 obstacle classes: `building`, `tree`, `low vegetation`, `transmission tower`, `communication tower`
- Handles both **low-altitude** (≤100 m) and **high-altitude** (>100 m) drone imagery with separate optimized models
- Generates an **interactive HTML map** with GPS-accurate obstacle polygons overlaid on satellite imagery
- Uses a **Teacher–Student semi-supervised learning** strategy to extend training beyond the initial labeled dataset

---

## System Architecture

```
UAV Flight
    │
    ▼
Flight Images + GPS/IMU Metadata (CSV)
    │
    ├─── Altitude ≤ 100 m ──► YOLOv8 (Student Model, low_alt)
    │
    └─── Altitude > 100 m ──► YOLOv5 (high_alt)
                │
                ▼
        Per-Frame Detections (Bounding Boxes + Class)
                │
                ▼
    Confidence Grid Voting System (Canvas-based)
    [Pixel accumulation across overlapping frames]
                │
                ▼
    Threshold Filter → Contour Detection → GPS Polygon Conversion
                │
                ▼
    Interactive Folium Map (HTML)
    [Georeferenced obstacle polygons on satellite basemap]
```

---

## Repository Structure

```
RIS-placement-using-CNN/
│
├── high_alt.ipynb          # Inference notebook for high-altitude imagery (YOLOv5)
├── low_alt.ipynb           # Data prep, Teacher-Student training pipeline (YOLOv8)
├── mapping.ipynb           # Geospatial mapping pipeline (dual-model + Folium output)
│
├── models/                 # (Not tracked — store weights in Google Drive)
│   ├── high.pt             # Trained YOLOv5 weights (high altitude)
│   └── best.pt             # Trained YOLOv8 weights (low altitude / student)
│
├── data/                   # (Not tracked — datasets are large)
│   └── flight_metadata.csv # Example metadata schema
│
└── README.md
```

---

## Datasets

| Dataset | Purpose | Source |
|---|---|---|
| **UAVid** | Primary labeled training data (semantic segmentation → YOLO conversion) | [uavid.uk](https://uavid.uk/) |
| **VisDrone 2019** | Unlabeled images for pseudo-label generation (semi-supervised learning) | [Kaggle](https://www.kaggle.com/datasets/kushagrapandya/visdrone-dataset) |
| **DroneDeploy** | High-altitude orthophoto inference testing | [DroneDeploy](https://www.dronedeploy.com/) |
| **Custom UAV images** | Real-world inference and map generation | Collected locally |

### Obstacle Classes

| Class ID | Class Name | Description |
|---|---|---|
| 0 | `building` | Structures, rooftops |
| 1 | `tree` | Individual trees |
| 2 | `low vegetation` | Grass, bushes, shrubs |
| 3 | `transmission tower` | High-voltage power pylons |
| 4 | `communication tower` | Cell towers, antennas |

---

## Pipeline Walkthrough

### Phase 1 — Label Preparation (`low_alt.ipynb`)

The UAVid dataset provides **semantic segmentation masks**, not bounding boxes. This phase converts them into YOLO-format object detection labels.

**Steps:**
1. Load UAVid semantic label images (PNG masks with class-specific RGB colors)
2. For each obstacle class (`Building`, `Tree`, `Low vegetation`), extract a binary mask
3. Apply the **Watershed algorithm** (`cv2.watershed`) to separate touching instances
4. Compute bounding boxes from instance regions and normalize to YOLO format (`class_id cx cy w h`)
5. Save `.txt` label files alongside each image

**Color Map used:**
```python
UAVID_COLOR_MAP = {
    'Building':            (128, 0,   0),
    'Road':                (128, 64,  128),
    'Tree':                (0,   128, 0),
    'Low vegetation':      (128, 128, 0),
    'Moving car':          (64,  0,   128),
    'Static car':          (192, 0,   192),
    'Human':               (64,  64,  0),
    'Background clutter':  (0,   0,   0)
}
```

---

### Phase 2 — Model Training (`low_alt.ipynb`)

A **Teacher–Student semi-supervised learning** strategy is used to maximize the use of both labeled (UAVid) and unlabeled (VisDrone) data.

#### 2a. Teacher Model Training
- Base model: `yolov8n.pt` (pretrained on COCO)
- Trained on the UAVid-derived YOLO dataset (80/20 train-val split)
- Training config: 50 epochs, image size 1280, batch size 8, patience 20
- **Teacher results:** mAP50 = 0.476, Precision = 0.629, Recall = 0.438

#### 2b. Pseudo-Label Generation
- The trained teacher model runs inference on all VisDrone images
- Predictions with confidence ≥ 0.85 are saved as YOLO labels
- These pseudo-labeled images are combined with the original UAVid labels

#### 2c. Zero-Shot Annotation Augmentation
- **OWL-ViT** (`google/owlvit-base-patch32`) is used for zero-shot detection of `transmission tower` and `communication tower` classes, which are absent in UAVid
- New boxes are appended to existing label files, extending the dataset to 5 classes

#### 2d. Dataset Cleaning
- Remove images with empty labels or only `low vegetation` class (class 2)
- Remove images with fewer than 3 annotations
- Cleaned files are backed up rather than deleted permanently

#### 2e. Student Model Training
- Base model: `yolov8s.pt` (small variant, better accuracy)
- Trained on the combined + cleaned dataset
- Training config: 100 epochs, image size 640, batch size 16, patience 20, GPU

---

### Phase 3 — High-Altitude Inference (`high_alt.ipynb`)

A separate **YOLOv5** model (`high.pt`) handles high-altitude orthophoto imagery (e.g., DroneDeploy GeoTIFF files).

**Steps:**
1. Load `.tif` orthophoto with PIL
2. Convert to BGR format for YOLOv5
3. Run inference and render bounding boxes with class labels
4. Save annotated result as `detection_results.jpg`

```python
model = torch.hub.load('ultralytics/yolov5', 'custom', path='high.pt')
results = model(img_cv)
```

---

### Phase 4 — Geospatial Mapping (`mapping.ipynb`)

This is the final output stage. It produces an **interactive HTML map** showing all detected obstacles as GPS-accurate colored polygons.

#### Key Components

**`ObstacleDetector` class**
- Automatically selects the correct model based on drone altitude:
  - `altitude > 100 m` → YOLOv5 (`high.pt`)
  - `altitude ≤ 100 m` → YOLOv8 (`best.pt`)
- Standardizes output of both models to a unified `[{'bbox': [...], 'class_name': '...'}]` format

**Confidence Grid Voting System**
- A virtual canvas (in pixels) is created to cover the entire GPS flight area
- Each detection from each image "votes" on the canvas pixels it occupies
- Only pixels with votes ≥ `DETECTION_CONFIDENCE_THRESHOLD` (default: 2) are retained
- This suppresses false positives from single-frame detections

**Contour-to-GPS Conversion**
- `cv2.findContours` extracts obstacle shapes from the thresholded grid
- `cv2.approxPolyDP` simplifies contours to reduce polygon size
- Each pixel coordinate is converted back to GPS using:

```python
def pixel_to_gps(origin_lat, origin_lon, x_pixels, y_pixels, gsd):
    distance_x = x_pixels * gsd   # East displacement
    distance_y = y_pixels * gsd   # South displacement
    ...
```

**Output**
- Saved as `final_interactive_merged_map.html`
- Uses Google Satellite basemap tiles via Folium
- Color-coded polygons with class labels and a map legend

**Map Color Scheme:**

| Class | Color |
|---|---|
| Building | Red |
| Tree | Dark Green |
| Low Vegetation | Light Green |
| Transmission Tower | Orange |
| Communication Tower | Blue |

---

## Model Details

| Model | Architecture | Use Case | Input Size | mAP50 |
|---|---|---|---|---|
| Teacher | YOLOv8n | Initial training on UAVid | 1280 | 0.476 |
| Student | YOLOv8s | Full 5-class detection (low altitude) | 640 | — |
| High-Alt | YOLOv5 (custom) | High-altitude ortho imagery | — | — |

**Ground Sample Distance (GSD) calculation:**
```python
gsd = (156543.03 * cos(latitude_rad)) / (2 ** zoom_level)
```
Baseline GSD used: **0.262 m/pixel** at zoom level 20.

---

## Setup and Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (strongly recommended for training)
- Google Colab (all notebooks are configured for Colab + Google Drive)

### 1. Clone the Repository

```bash
git clone https://github.com/shahmehul2005/RIS-placement-using-CNN.git
cd RIS-placement-using-CNN
```

### 2. Install Dependencies

```bash
pip install ultralytics torch torchvision opencv-python numpy pandas \
            geopy folium matplotlib scikit-learn transformers timm \
            supervision pillow tqdm pyyaml
```

Or install per-notebook using the `!pip install` cells at the top of each notebook.

### 3. Download Datasets

**UAVid:**
```bash
# Download from https://uavid.uk/ and place in Google Drive
# Expected structure:
# MyDrive/uavid_train/seq1/Images/
# MyDrive/uavid_train/seq1/Labels/
```

**VisDrone:**
```bash
# Via Kaggle API (run inside Colab):
!pip install kaggle
# Upload kaggle.json, then:
!kaggle datasets download kushagrapandya/visdrone-dataset
```

### 4. Download Pretrained Weights

Place the following files in your Google Drive root:
- `high.pt` — YOLOv5 weights for high-altitude inference
- `best.pt` or `teacher_best.pt` — YOLOv8 student model weights

### 5. Prepare Metadata CSV

For the mapping pipeline, provide a CSV file with the following columns:

```
filename, latitude, longitude, altitude
image_001.jpg, 23.0225, 72.5714, 85.0
image_002.jpg, 23.0226, 72.5715, 85.0
...
```

---

## Usage

### Run High-Altitude Inference

Open `high_alt.ipynb` in Colab and execute all cells. Update:
```python
img_path = '/content/drive/MyDrive/<your_image>.tif'
```

### Run the Full Training Pipeline

Open `low_alt.ipynb` in Colab and follow the section headers in order:
1. Data Preparation and YOLO Label Generation
2. Teacher Model Training
3. Pseudo Labeling
4. Zero-Shot Annotation (OWL-ViT)
5. Dataset Cleaning
6. Student Model Training

### Generate the Obstacle Map

Open `mapping.ipynb` in Colab. Update the configuration block:
```python
METADATA_FILE        = 'flight_metadata.csv'
IMAGE_DIRECTORY      = '/content/drive/MyDrive/<your_image_folder>'
LOW_ALT_MODEL_PATH   = '/content/drive/MyDrive/best.pt'
HIGH_ALT_MODEL_PATH  = '/content/drive/MyDrive/high.pt'
```
Run all cells. The output `final_interactive_merged_map.html` can be opened in any browser.

---

## Results

- **Teacher Model** (YOLOv8n, UAVid only): mAP50 = 0.476, Precision = 0.629, Recall = 0.438
- **Student Model** (YOLOv8s, combined dataset): Improved recall due to pseudo-labeled VisDrone data and zero-shot tower annotations
- **Mapping output**: Interactive HTML map with GPS-accurate obstacle polygons across the full flight area

Sample detection output on a high-altitude DroneDeploy orthophoto is saved as `detection_results.jpg` after running `high_alt.ipynb`.

---

## Contributing

Contributions, issues, and feature requests are welcome.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

This project is released for academic and research purposes. Please cite the respective dataset papers (UAVid, VisDrone) if you use this work in your research.

---

> **Project developed as part of a CNN-based obstacle detection study for RIS deployment planning.**
