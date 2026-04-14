"""
Heatmap Visualization Script: LayerCAM
Generates LayerCAM saliency maps for visualizing model attention regions
Faster than GradCAM while maintaining good interpretability
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

CONFIG = {
    'model_path': r'D:\cop\AugmentedAlzheimerDataset\roi_scorecam_output\evaluation_results\layercam_best_model.pth',
    'dataset_path': r'D:\cop\AugmentedAlzheimerDataset',
    'output_path': r'D:\cop\AugmentedAlzheimerDataset\roi_scorecam_output\evaluation_results\heatmaps_layercam',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'input_size': 224,
}

DISEASE_CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

# Create output directory
os.makedirs(CONFIG['output_path'], exist_ok=True)

print(f"🔧 Device: {CONFIG['device']}")
print(f"📁 Output: {CONFIG['output_path']}")

# ============================================================================
# 2. LAYERCAM IMPLEMENTATION
# ============================================================================

class LayerCAM:
    """Layer-wise Class Activation Mapping"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        
        # Register forward hook
        target_layer.register_forward_hook(self.save_activation)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def generate_cam(self, input_tensor, target_class):
        """Generate LayerCAM heatmap (no gradients needed)"""
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            output = self.model(input_tensor)
            activations = self.activations
            
            # Compute importance weights using ReLU
            weights = F.relu(activations).mean(dim=(2, 3), keepdim=True)
            
            # Generate CAM
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            
            # Normalize
            cam = cam.squeeze(1)
            cam_min = cam.view(cam.size(0), -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
            cam_max = cam.view(cam.size(0), -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            
            return cam.cpu().numpy()[0]

# ============================================================================
# 3. MODEL LOADING
# ============================================================================

print("\n🏗️  Loading Model...")

class DualBranchLayerCAM(nn.Module):
    """Dual-Branch with LayerCAM"""
    
    def __init__(self, num_classes=4):
        super(DualBranchLayerCAM, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.branch1_fc = nn.Sequential(
            nn.Linear(backbone_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.branch2_fc = nn.Sequential(
            nn.Linear(backbone_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
        self.layercam_layer = self.backbone.layer4[-1]
    
    def forward(self, x):
        features = self.backbone(x)
        branch1_out = self.branch1_fc(features)
        branch2_out = self.branch2_fc(features)
        fused = torch.cat([branch1_out, branch2_out], dim=1)
        logits = self.fusion(fused)
        return logits

# Load model
model = DualBranchLayerCAM(num_classes=4).to(CONFIG['device'])
model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
model.eval()
print("✓ Model loaded successfully")

# Initialize LayerCAM
layercam = LayerCAM(model, model.layercam_layer)

# ============================================================================
# 4. IMAGE PROCESSING & HEATMAP GENERATION
# ============================================================================

print("\n🎨 Generating LayerCAM Heatmaps...\n")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((CONFIG['input_size'], CONFIG['input_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_heatmap(image_path, class_idx, disease_name):
    """Generate and save heatmap for an image"""
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Resize for visualization
    img_resized = cv2.resize(img_array, (CONFIG['input_size'], CONFIG['input_size']))
    
    # Transform for model
    img_tensor = transform(img).unsqueeze(0).to(CONFIG['device'])
    
    # Generate prediction
    with torch.no_grad():
        output = model(img_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()
    
    # Generate CAM
    cam = layercam.generate_cam(img_tensor, class_idx)
    
    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (CONFIG['input_size'], CONFIG['input_size']))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_VIRIDIS)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    blended = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_resized)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(cam_resized, cmap='viridis')
    axes[1].set_title('LayerCAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Blended
    axes[2].imshow(blended.astype(np.uint8))
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Main title
    pred_disease = DISEASE_CLASSES[pred_class]
    fig.suptitle(
        f'{disease_name} | Predicted: {pred_disease} ({confidence*100:.1f}%)',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    return fig, blended

# ============================================================================
# 5. PROCESS SAMPLE IMAGES
# ============================================================================

sample_count = 0
max_samples = 3  # Generate heatmaps for 3 images per class

for class_idx, disease_class in enumerate(DISEASE_CLASSES):
    class_path = os.path.join(CONFIG['dataset_path'], disease_class)
    
    if os.path.exists(class_path):
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))][:max_samples]
        
        for img_idx, img_name in enumerate(images):
            img_path = os.path.join(class_path, img_name)
            
            print(f"Processing {disease_class}/{img_name}...")
            
            try:
                fig, blended = generate_heatmap(img_path, class_idx, disease_class)
                
                # Save figure
                output_filename = f'{disease_class}_{img_idx+1}_layercam.png'
                output_filepath = os.path.join(CONFIG['output_path'], output_filename)
                plt.savefig(output_filepath, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"   ✓ Saved: {output_filename}")
                sample_count += 1
                
            except Exception as e:
                print(f"   ✗ Error: {str(e)}")

# ============================================================================
# 6. SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ LayerCAM Heatmap Generation Complete")
print("="*80)
print(f"Total heatmaps generated: {sample_count}")
print(f"Output directory: {CONFIG['output_path']}")
print("="*80)

print("\n📊 Heatmap Interpretation:")
print("   - Yellow/Bright regions: High model attention (important features)")
print("   - Purple/Dark regions: Low model attention (less important)")
print("   - Fast generation (no gradient computation required)")
print("   - Good interpretability with minimal computational overhead")
