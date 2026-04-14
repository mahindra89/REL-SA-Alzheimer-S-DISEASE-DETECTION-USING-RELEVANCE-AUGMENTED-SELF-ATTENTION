# Alzheimer’s Disease MRI Classification with CAM-Based Explainable AI

This repository contains training and visualization scripts for Alzheimer’s disease stage classification from 2D MRI images using **ResNet-18** and three explainable AI attention methods: **GradCAM**, **ScoreCAM**, and **LayerCAM**.

The project focuses on two goals:
1. **Classify MRI images into 4 dementia stages**
2. **Improve interpretability** by showing which image regions influenced the prediction

The uploaded code implements a baseline model and three CAM-based dual-branch models, along with separate scripts to generate attention heatmaps from trained checkpoints.

---

## Project Overview

The models classify MRI images into the following four classes:
- `NonDemented`
- `VeryMildDemented`
- `MildDemented`
- `ModerateDemented`

All training scripts use:
- **ResNet-18** backbone
- **Input size:** `224 x 224`
- **Epochs:** `80`
- **Batch size:** `16`
- **Optimizer:** `Adam`
- **Learning rate:** `0.001`
- **Scheduler:** `StepLR(step_size=20, gamma=0.1)`
- **Train/Validation split:** `80/20`

The code is written in PyTorch and automatically uses CUDA if available.

---

## Included Files

### Training scripts
- `1_train_baseline_model.py`  
  Trains the baseline ResNet-18 classifier.

- `2_train_gradcam_model.py`  
  Trains a dual-branch ResNet-18 model using GradCAM-based relevance features.

- `3_train_scorecam_model.py`  
  Trains a dual-branch ResNet-18 model using ScoreCAM-based relevance features. This is marked in the code as the **best-performing model**.

- `4_train_layercam_model.py`  
  Trains a dual-branch ResNet-18 model using LayerCAM-based relevance features. This is marked in the code as the **fastest variant**.

### Heatmap generation scripts
- `generate_heatmaps_gradcam.py`  
  Loads a trained GradCAM model and generates visualization heatmaps.

- `generate_heatmaps_scorecam.py`  
  Loads a trained ScoreCAM model and generates visualization heatmaps.

- `generate_heatmaps_layercam.py`  
  Loads a trained LayerCAM model and generates visualization heatmaps.

### Supporting files
- `NN_FinalReport.pdf`  
  Final report for the project.

- `NN PPT_Final.pptx`  
  Presentation slides.

---

## Model Variants

### 1. Baseline ResNet-18
This is the reference model. It uses a pretrained ResNet-18 and replaces the final fully connected layer for 4-class classification.

### 2. Dual-Branch GradCAM Model
This version adds a second branch intended to capture GradCAM-based relevance information. Features from two branches are fused before final classification.

### 3. Dual-Branch ScoreCAM Model
This version uses ScoreCAM-style attention and a stronger fusion strategy. According to the code comments and summaries, this is the strongest model in the project.

### 4. Dual-Branch LayerCAM Model
This version uses LayerCAM and is designed to keep good interpretability with lower overhead. According to the code comments and summaries, this is the fastest model among the enhanced variants.

---

## Expected Dataset Structure

The code expects the dataset folder to be arranged class-wise like this:

```text
AugmentedAlzheimerDataset/
├── NonDemented/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── VeryMildDemented/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── MildDemented/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── ModerateDemented/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

The scripts read files with these extensions:
- `.jpg`
- `.png`
- `.jpeg`

---

## Important Setup Note

The uploaded scripts currently use **hardcoded Windows paths** inside the `CONFIG` dictionary, for example:

```python
'dataset_path': r'D:\cop\AugmentedAlzheimerDataset'
'output_path': r'D:\cop\AugmentedAlzheimerDataset\roi_scorecam_output\evaluation_results'
```

Before running the code on your machine, update these paths to match your local dataset and output folders.

---

## Installation

Create a Python environment and install the required packages.

```bash
pip install torch torchvision numpy matplotlib tqdm pillow opencv-python scipy
```

If you are using a GPU, make sure your PyTorch installation matches your CUDA version.

---

## How to Run

### 1. Train the baseline model
```bash
python 1_train_baseline_model.py
```

### 2. Train the GradCAM model
```bash
python 2_train_gradcam_model.py
```

### 3. Train the ScoreCAM model
```bash
python 3_train_scorecam_model.py
```

### 4. Train the LayerCAM model
```bash
python 4_train_layercam_model.py
```

### 5. Generate GradCAM heatmaps
```bash
python generate_heatmaps_gradcam.py
```

### 6. Generate ScoreCAM heatmaps
```bash
python generate_heatmaps_scorecam.py
```

### 7. Generate LayerCAM heatmaps
```bash
python generate_heatmaps_layercam.py
```

---

## Outputs

Each training script saves results into the configured output directory.

### Training outputs
Typical saved files include:
- `*_best_model.pth` — best checkpoint based on validation accuracy
- `*_training_history.json` — training and validation metrics across epochs
- `*_training_curves.png` — loss and accuracy plots

Examples:
- `baseline_best_model.pth`
- `gradcam_best_model.pth`
- `scorecam_best_model.pth`
- `layercam_best_model.pth`

### Heatmap outputs
Each heatmap script saves visualizations for sample images. The figure usually contains:
- original MRI image
- CAM heatmap
- overlay image

The scripts generate heatmaps for up to **3 images per class**.

---

## Preprocessing Used in Code

The training scripts apply the following transforms:
- resize to `224 x 224`
- random horizontal flip
- random rotation
- color jitter
- tensor conversion
- ImageNet normalization

Validation uses resize and normalization only.

---

## Notes on the Pipeline

- All models use a shared ResNet-18 backbone.
- The enhanced models use **dual-branch feature fusion**.
- Heatmap scripts load saved checkpoints and generate visual explanations of model attention.
- The code is built for **2D MRI slice classification**, not full 3D MRI volumes.
- The presentation and report describe a broader relevance-attention research direction, but the uploaded code files here mainly cover the **baseline**, **CAM-based dual-branch training**, and **heatmap generation** stages.

---

## Performance Summary

Based on the comments and summaries included in the uploaded scripts:

- **ScoreCAM** is presented as the **best-performing model**
- **LayerCAM** is presented as the **fastest model**
- The project aims to balance **accuracy**, **interpretability**, and **computational efficiency**

If you want exact final numbers for a run, check:
- the terminal logs after training
- the saved `*_training_history.json`
- the generated training curve plots

---

## Suggested Repository Structure

If you want to keep the repo clean, a good structure would be:

```text
project-root/
├── 1_train_baseline_model.py
├── 2_train_gradcam_model.py
├── 3_train_scorecam_model.py
├── 4_train_layercam_model.py
├── generate_heatmaps_gradcam.py
├── generate_heatmaps_scorecam.py
├── generate_heatmaps_layercam.py
├── NN_FinalReport.pdf
├── NN PPT_Final.pptx
├── README.md
└── outputs/
```

---

## Possible Improvements

A few practical improvements for future cleanup:
- move all hardcoded paths into command-line arguments or a config file
- add a `requirements.txt`
- add checkpoint loading for resuming training
- sort or shuffle filenames before the 80/20 split for better reproducibility
- save confusion matrices and per-class metrics
- add support for Linux-friendly relative paths

---

## Acknowledgment

This project is inspired by relevance-augmented attention ideas for Alzheimer’s disease detection, but the implementation here uses a lighter **2D ResNet-18 + CAM-based explainability** pipeline for practical training and visualization.

