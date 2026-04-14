"""
Heatmap Visualization Script: GradCAM
Generates GradCAM saliency maps for visualizing model attention regions
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
    'model_path': r'D:\cop\AugmentedAlzheimerDataset\roi_scorecam_output\evaluation_results\gradcam_best_model.pth',
    'dataset_path': r'D:\cop\AugmentedAlzheimerDataset',
    'output_path': r'D:\cop\AugmentedAlzheimerDataset\roi_scorecam_output\evaluation_results\heatmaps_gradcam',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'input_size': 224,
}

DISEASE_CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

# Create output directory
os.makedirs(CONFIG['output_path'], exist_ok=True)

print(f"🔧 Device: {CONFIG['device']}")
print(f"📁 Output: {CONFIG['output_path']}")

# ============================================================================
# 2. GRADCAM IMPLEMENTATION
# ============================================================================

class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class):
        """Generate GradCAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        target_output = output[:, target_class]
        target_output.sum().backward()
        
        # Compute GradCAM
        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()
        
        # Weight by gradients
        weights = np.mean(gradients, axis=(2, 3), keepdims=True)
        cam = np.sum(weights * activations, axis=1)
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        
        return cam[0]

# ============================================================================
# 3. MODEL LOADING
# ============================================================================

print("\n🏗️  Loading Model...")

class DualBranchGradCAM(nn.Module):
    """Dual-Branch with GradCAM"""
    
    def __init__(self, num_classes=4):
        super(DualBranchGradCAM, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.branch1_fc = nn.Sequential(
            nn.Linear(backbone_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.branch2_fc = nn.Sequential(
            nn.Linear(backbone_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
        self.cam_layer = self.backbone.layer4[-1]
    
    def forward(self, x):
        features = self.backbone(x)
        branch1_out = self.branch1_fc(features)
        branch2_out = self.branch2_fc(features)
        fused = torch.cat([branch1_out, branch2_out], dim=1)
        logits = self.fusion(fused)
        return logits

# Load model
model = DualBranchGradCAM(num_classes=4).to(CONFIG['device'])
model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
model.eval()
print("✓ Model loaded successfully")

# Initialize GradCAM
gradcam = GradCAM(model, model.cam_layer)

# ============================================================================
# 4. IMAGE PROCESSING & HEATMAP GENERATION
# ============================================================================

print("\n🎨 Generating GradCAM Heatmaps...\n")

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
    cam = gradcam.generate_cam(img_tensor, class_idx)
    
    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (CONFIG['input_size'], CONFIG['input_size']))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
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
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('GradCAM Heatmap', fontsize=12, fontweight='bold')
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
                output_filename = f'{disease_class}_{img_idx+1}_gradcam.png'
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
print("✅ GradCAM Heatmap Generation Complete")
print("="*80)
print(f"Total heatmaps generated: {sample_count}")
print(f"Output directory: {CONFIG['output_path']}")
print("="*80)

print("\n📊 Heatmap Interpretation:")
print("   - Red regions: High model attention (important for prediction)")
print("   - Blue regions: Low model attention (less important)")
print("   - Helps understand which brain regions influence the model's decision")
