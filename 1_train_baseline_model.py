"""
Training Script: ResNet-18 Baseline Model
Dataset: 3000 images per class (12,000 total) across 4 Alzheimer's stages
Epochs: 80
GPU: RTX 3060 (12GB VRAM)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

CONFIG = {
    'dataset_path': r'D:\cop\AugmentedAlzheimerDataset',
    'output_path': r'D:\cop\AugmentedAlzheimerDataset\roi_scorecam_output\evaluation_results',
    'num_epochs': 80,
    'batch_size': 16,
    'learning_rate': 0.001,
    'num_classes': 4,
    'input_size': 224,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'seed': 42,
}

DISEASE_CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

print(f"🔧 Device: {CONFIG['device']}")
print(f"🔧 CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# 2. CUSTOM DATASET CLASS
# ============================================================================

class AlzheimersDataset(Dataset):
    """Custom dataset for Alzheimer's MRI images"""
    
    def __init__(self, data_dir, disease_classes, transform=None, train=True):
        self.data_dir = data_dir
        self.disease_classes = disease_classes
        self.transform = transform
        self.train = train
        self.images = []
        self.labels = []
        
        # Load image paths and labels
        for class_idx, disease_class in enumerate(disease_classes):
            class_path = os.path.join(data_dir, disease_class)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                
                # Split into train/val (80/20)
                num_train = int(len(images) * 0.8)
                if train:
                    images = images[:num_train]
                else:
                    images = images[num_train:]
                
                for img_name in images:
                    img_path = os.path.join(class_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)
                    
                print(f"✓ Loaded {len(images)} {'train' if train else 'val'} images from {disease_class}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# ============================================================================
# 3. DATA LOADING
# ============================================================================

print("\n📊 Loading Dataset...")

# Data transformations
train_transforms = transforms.Compose([
    transforms.Resize((CONFIG['input_size'], CONFIG['input_size'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((CONFIG['input_size'], CONFIG['input_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = AlzheimersDataset(
    CONFIG['dataset_path'],
    DISEASE_CLASSES,
    transform=train_transforms,
    train=True
)

val_dataset = AlzheimersDataset(
    CONFIG['dataset_path'],
    DISEASE_CLASSES,
    transform=val_transforms,
    train=False
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

print(f"✓ Training samples: {len(train_dataset)}")
print(f"✓ Validation samples: {len(val_dataset)}")

# ============================================================================
# 4. MODEL DEFINITION
# ============================================================================

print("\n🏗️  Building Model...")

class BaselineResNet18(nn.Module):
    """ResNet-18 for Alzheimer's classification (Baseline)"""
    
    def __init__(self, num_classes=4):
        super(BaselineResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        
        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

model = BaselineResNet18(num_classes=CONFIG['num_classes']).to(CONFIG['device'])
print(f"✓ Model loaded to {CONFIG['device']}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✓ Total parameters: {total_params:,}")
print(f"✓ Trainable parameters: {trainable_params:,}")

# ============================================================================
# 5. TRAINING CONFIGURATION
# ============================================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# ============================================================================
# 6. TRAINING LOOP
# ============================================================================

print("\n" + "="*80)
print("🚀 STARTING TRAINING - ResNet-18 Baseline Model")
print("="*80)
print(f"Epochs: {CONFIG['num_epochs']}")
print(f"Batch Size: {CONFIG['batch_size']}")
print(f"Learning Rate: {CONFIG['learning_rate']}")
print(f"Optimizer: Adam")
print("="*80 + "\n")

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'epoch_times': []
}

best_val_acc = 0.0
start_time = datetime.now()

for epoch in range(CONFIG['num_epochs']):
    epoch_start = datetime.now()
    
    # ========== TRAINING PHASE ==========
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1:2d}/{CONFIG["num_epochs"]} [TRAIN]')
    for images, labels in train_bar:
        images = images.to(CONFIG['device'])
        labels = labels.to(CONFIG['device'])
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        train_bar.set_postfix({
            'loss': train_loss / (train_bar.n + 1),
            'acc': 100 * train_correct / train_total
        })
    
    train_loss /= len(train_loader)
    train_acc = 100 * train_correct / train_total
    
    # ========== VALIDATION PHASE ==========
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1:2d}/{CONFIG["num_epochs"]} [VAL] ')
    with torch.no_grad():
        for images, labels in val_bar:
            images = images.to(CONFIG['device'])
            labels = labels.to(CONFIG['device'])
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            val_bar.set_postfix({
                'loss': val_loss / (val_bar.n + 1),
                'acc': 100 * val_correct / val_total
            })
    
    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total
    
    # ========== EPOCH SUMMARY ==========
    epoch_time = (datetime.now() - epoch_start).total_seconds()
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['epoch_times'].append(epoch_time)
    
    scheduler.step()
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 
                   os.path.join(CONFIG['output_path'], 'baseline_best_model.pth'))
        print(f"   ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
    
    print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print(f"   Time: {epoch_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}\n")

# ============================================================================
# 7. TRAINING SUMMARY
# ============================================================================

total_time = (datetime.now() - start_time).total_seconds()
total_time_hours = total_time / 3600

print("\n" + "="*80)
print("✅ TRAINING COMPLETED - ResNet-18 Baseline Model")
print("="*80)
print(f"Total Training Time: {total_time_hours:.2f} hours ({total_time:.0f} seconds)")
print(f"Average Time per Epoch: {np.mean(history['epoch_times']):.2f} seconds")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
print(f"Final Val Accuracy: {history['val_acc'][-1]:.2f}%")
print("="*80 + "\n")

# ============================================================================
# 8. SAVE RESULTS & HISTORY
# ============================================================================

# Save training history
history_path = os.path.join(CONFIG['output_path'], 'baseline_training_history.json')
with open(history_path, 'w') as f:
    json.dump({
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc'],
        'epoch_times': history['epoch_times'],
        'total_time_seconds': total_time,
        'total_time_hours': total_time_hours,
        'best_val_acc': best_val_acc
    }, f, indent=2)

print(f"✓ History saved to {history_path}")

# ============================================================================
# 9. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(history['train_acc'], label='Train Accuracy', linewidth=2, marker='o', markersize=3)
axes[0].plot(history['val_acc'], label='Val Accuracy', linewidth=2, marker='s', markersize=3)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('ResNet-18 Baseline - Accuracy Progression')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=3)
axes[1].plot(history['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=3)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('ResNet-18 Baseline - Loss Progression')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(CONFIG['output_path'], 'baseline_training_curves.png'), dpi=300, bbox_inches='tight')
print(f"✓ Training curves saved")

# Model summary
print("\n📋 Model Summary:")
print(f"   Architecture: ResNet-18 (Pretrained)")
print(f"   Output Classes: {CONFIG['num_classes']}")
print(f"   Total Parameters: {total_params:,}")
print(f"   Final Accuracy: {history['val_acc'][-1]:.2f}%")
print(f"   Best Accuracy: {best_val_acc:.2f}%")
print(f"   Training Duration: {total_time_hours:.2f} hours")
