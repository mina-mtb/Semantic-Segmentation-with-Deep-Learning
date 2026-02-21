# Semantic Segmentation with Deep Learning: Potsdam Dataset implementation

This repository contains a comprehensive implementation of semantic segmentation for aerial imagery using the **ISPRS Potsdam 2D Semantic Labeling dataset**. The project compares a basic Convolutional Neural Network (CNN) with a more advanced **U-Net (Encoder-Decoder)** architecture.

---

## 📂 Project Structure

- **`PROJECT/notebooks/`**: Step-by-step guides for data prep and model training.
- **`PROJECT/scripts/`**: Python scripts for automation and utility functions.
- **`PROJECT/models/`**: Pre-trained model weights (best performing simple CNN).
- **`PROJECT/outputs/`**: Visualization results, training curves, and evaluation metrics.
- **`PROJECT/report/`**: Professional LaTeX report (English) detailing the methodology and results.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install tensorflow rasterio numpy matplotlib scikit-learn
```

### 2. Dataset Setup
Due to its large size (~15GB), the raw dataset is **not** included in this repository. 

**How to download and use:**
1. Download the **"2D Semantic Labeling Potsdam"** GeoTIFF tiles from the [ISPRS Official Website](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx).
2. Create a folder named `data/` or `Potsdam-GeoTif/` inside the `PROJECT/` directory.
3. Extract the downloaded `.zip` file into that folder. The notebooks are configured to automatically find the `.tif` tiles if they are placed in either of these locations.

### 3. Running the Project
The notebooks are numbered sequence-wise:
1. `Step1_Dataset_Preparation.ipynb`: Visualizes the data and performs 5-fold cross-validation splits.
2. `Step2_Simple_Model.ipynb`: Implements and trains a basic CNN backbone.
3. `Step3_Encoder_Decoder_Model.ipynb`: Implements a full U-Net with skip connections.

---

## 📊 Key Results

- **Simple CNN**: Balanced Accuracy ~68%, suitable for coarse mapping.
- **U-Net**: Balanced Accuracy ~78%+, significantly better at preserving building boundaries and fine-grained vegetation details through skip connections.

---

## 📄 Documentation
For a deep dive into the math and implementation details, check out the [English Project Report](PROJECT/report/project_report_EN.tex) or the PDF in the report folder.

---
**Author:** Mina Tahmasebi  
**Course:** Design of AI Systems
