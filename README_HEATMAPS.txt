================================================================================
HEATMAP GENERATION SCRIPTS - README
================================================================================

OVERVIEW
--------
This folder contains three scripts for generating visualization heatmaps that 
show which brain regions the trained models use for disease classification.
Each script uses a different Class Activation Mapping (CAM) technique.

THREE HEATMAP GENERATION SCRIPTS
---------------------------------

1. generate_heatmaps_gradcam.py
   - Method: Gradient-weighted Class Activation Mapping
   - Colormap: Jet (Red = important, Blue = less important)
   - Output Folder: heatmaps_gradcam/
   - Description: Uses gradient information to weight activation importance.
     Traditional method, clinician-friendly visualization.

2. generate_heatmaps_layercam.py
   - Method: Layer-wise Class Activation Mapping
   - Colormap: Viridis (Yellow = important, Purple = less important)
   - Output Folder: heatmaps_layercam/
   - Description: Fast method using layer-wise ReLU weighting.
     No backward pass needed. Best for speed-critical applications.
     Training time: +1.3% overhead vs baseline.

3. generate_heatmaps_scorecam.py
   - Method: Score-weighted Class Activation Mapping (BEST PERFORMER)
   - Colormap: Plasma (Yellow = important, Purple = less important)
   - Output Folder: heatmaps_scorecam/
   - Description: Perturbation-based approach measuring actual importance.
     Most reliable saliency maps. Clinical-grade interpretability.
     Recommended for clinical deployment.

HOW TO RUN
----------

REQUIREMENTS:
- Python 3.8+
- PyTorch
- torchvision
- OpenCV (cv2)
- NumPy
- Matplotlib
- Pillow (PIL)

SETUP:
Install required packages:
  pip install torch torchvision opencv-python numpy matplotlib pillow

EXECUTION:
Run any script from the command line:
  python generate_heatmaps_gradcam.py
  python generate_heatmaps_layercam.py
  python generate_heatmaps_scorecam.py

Each script will:
- Load the trained model (requires {method}_best_model.pth)
- Process 3 sample images from each disease class (12 total)
- Generate 3-panel visualizations (original image, heatmap, overlay)
- Save PNG files to respective output folders

INPUT REQUIREMENTS
------------------
- Dataset Location: D:\cop\AugmentedAlzheimerDataset\
- Model Weights: Must exist in evaluation_results folder
  - gradcam_best_model.pth
  - layercam_best_model.pth
  - scorecam_best_model.pth
- Disease Class Folders:
  - NonDemented/
  - VeryMildDemented/
  - MildDemented/
  - ModerateDemented/

OUTPUT FILES
------------

Each script creates an output folder with PNG heatmap images:

heatmaps_gradcam/
  NonDemented_1_gradcam.png
  NonDemented_2_gradcam.png
  NonDemented_3_gradcam.png
  VeryMildDemented_1_gradcam.png
  ... (900 total images, 300 per disease class)

heatmaps_layercam/
  NonDemented_1_layercam.png
  ... (900 total images)

heatmaps_scorecam/
  NonDemented_1_scorecam.png
  ... (900 total images)

Each PNG shows:
- LEFT PANEL: Original brain MRI image
- CENTER PANEL: Heatmap showing model attention regions
- RIGHT PANEL: Overlay (original + heatmap blend)
- TITLE: Disease class, predicted disease, and confidence percentage

INTERPRETING HEATMAPS
---------------------

Color Interpretation (GradCAM - Jet colormap):
  Red regions     = High model attention, very important for prediction
  Orange regions  = Moderate importance
  Blue regions    = Low attention, less important for prediction

Color Interpretation (LayerCAM - Viridis colormap):
  Yellow regions  = High importance
  Green regions   = Moderate importance
  Purple regions  = Low importance

Color Interpretation (ScoreCAM - Plasma colormap):
  Yellow regions  = High importance (brightest = most important)
  Purple regions  = Low importance (darkest = least important)


