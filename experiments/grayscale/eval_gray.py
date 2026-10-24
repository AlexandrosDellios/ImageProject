"""Archived, not maintained: score the grayscale shape model and draw Grad-CAM maps.

Reads best_step0_grayscale.pth and uno_multicard_cotraining/val (turned to grayscale) and
writes per-shape F1, error counts and Grad-CAM figures to diagnostics_grayscale/. That
checkpoint has the old head without dropout (fc, 17 x 5): it does not load into
uno_vision.model as is. See experiments/README.md.
"""

import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for uno_vision
from uno_vision.model import UnoCountingResNet

# --- CONFIGURATION ---
LABELS_CSV = "uno_multicard_cotraining/synthetic_val.csv"      
REAL_DATA_DIR = "uno_multicard_cotraining/val"         
PHASE4_WEIGHTS = "best_step0_grayscale.pth"  
OUTPUT_DIR = "diagnostics_grayscale"                

# UPDATED FOR GRAYSCALE MODEL
NUM_CLASSES = 17
MAX_COPIES = 5
BATCH_SIZE = 32

# The 17 Base Geometric Classes (Sorted alphabetically exactly as the folder dataset did)
CLASS_NAMES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
    'circle_token', 'draw_2', 'draw_4', 'reverse', 'skip', 'square_token', 'wild'
]

# Helper to strip colors from the 56-class CSV format
def get_geometric_class(name):
    name = name.lower()
    if 'black_token' in name: return 'square_token'
    if 'yellow_token' in name: return 'circle_token'
    if 'draw_4' in name: return 'draw_4'
    if 'wild' in name and 'draw' not in name: return 'wild'
    
    parts = name.split('_')
    if len(parts) > 1: return '_'.join(parts[1:])
    return name

# THE ASPECT RATIO SAVIOR
class PadToSquare:
    def __init__(self, fill=0):
        self.fill = fill
    def __call__(self, img):
        w, h = img.size
        if w == h: return img
        max_wh = max(w, h)
        pad_left = (max_wh - w) // 2
        pad_top = (max_wh - h) // 2
        pad_right = max_wh - w - pad_left
        pad_bottom = max_wh - h - pad_top
        return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill, padding_mode='constant')

class UnoRealHandDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, class_names, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    def __len__(self): return len(self.data_frame)
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(self.data_frame.iloc[idx]['filename']))
        image = Image.open(img_name).convert('RGB')
        
        cards_str = str(self.data_frame.iloc[idx]['cards'])
        target = torch.zeros(NUM_CLASSES, dtype=torch.long)
        
        if cards_str.strip().upper() != "EMPTY":
            for card in cards_str.split(';'):
                # Translate the CSV label (e.g., 'r_7') to the geometric label (e.g., '7')
                geo_class = get_geometric_class(card.strip())
                if geo_class in self.class_to_idx:
                    target[self.class_to_idx[geo_class]] += 1
                    
        if self.transform: image = self.transform(image)
        
        return image, target, img_name

# THE GRAD-CAM ENGINE
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx, count_idx):
        self.model.zero_grad()
        output = self.model(input_tensor) 
        
        target_logit = output[0, class_idx, count_idx]
        target_logit.backward(retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        return cam.detach().cpu().numpy()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Shape-Model Diagnostics on {device}")

    # Convert to Grayscale (3-channel) so it perfectly matches the training format
    transform = transforms.Compose([
        PadToSquare(fill=0),
        transforms.Resize((336, 336)),
        transforms.Grayscale(num_output_channels=3), # Strip real colors out before evaluation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = UnoRealHandDataset(LABELS_CSV, REAL_DATA_DIR, CLASS_NAMES, transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize model with 17 classes!
    model = UnoCountingResNet(num_classes=NUM_CLASSES, max_copies=MAX_COPIES)
    model.load_state_dict(torch.load(PHASE4_WEIGHTS, map_location=device, weights_only=True)) 
    model.to(device)
    model.eval()

    all_targets = []
    all_preds = []

    # --- PART 1: STANDARD METRICS ---
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc="Scanning Validation Set"):
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.permute(0, 2, 1) 
            preds = torch.argmax(outputs, dim=1) 
            
            all_targets.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    targets = np.vstack(all_targets)
    preds = np.vstack(all_preds)

    tp = np.minimum(targets, preds).sum(axis=0)
    fp = np.maximum(0, preds - targets).sum(axis=0)
    fn = np.maximum(0, targets - preds).sum(axis=0)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)

    # Plot F1 Scores
    plt.figure(figsize=(12, 6))
    sns.barplot(x=CLASS_NAMES, y=f1, palette="viridis")
    plt.xticks(rotation=45)
    plt.title("Per-Class F1 Score (Shape Expert)")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_grayscale_f1_scores.png"))
    plt.close()

    # Plot Hallucinations vs Blindness
    plt.figure(figsize=(12, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    plt.bar(x - width/2, fp, width, label='Hallucinated (FP)', color='red')
    plt.bar(x + width/2, fn, width, label='Missed (FN)', color='blue')
    plt.xticks(x, CLASS_NAMES, rotation=45)
    plt.title("Errors per Shape")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_grayscale_error_types.png"))
    plt.close()

    # --- PART 2: GRAD-CAM HEATMAPS ---
    print("\nGenerating X-Ray Heatmaps (Grad-CAM)...")
    
    try:
        target_layer = model.backbone.layer4[-1] 
    except AttributeError:
        target_layer = list(model.children())[0][-2] 

    cam_engine = GradCAM(model, target_layer)
    
    indices = random.sample(range(len(val_dataset)), 3)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("What is the Grayscale AI looking at?", fontsize=20, fontweight='bold')
    
    for row_idx, data_idx in enumerate(indices):
        tensor_img, target_counts, img_path = val_dataset[data_idx]
        
        orig_pil_img = Image.open(img_path).convert('RGB')
        padded_orig = PadToSquare(fill=0)(orig_pil_img).resize((336, 336))
        
        active_classes = (target_counts > 0).nonzero(as_tuple=True)[0]
        if len(active_classes) == 0: continue
        
        target_class_idx = active_classes[0].item()
        target_count = target_counts[target_class_idx].item()
        class_name = CLASS_NAMES[target_class_idx]
        
        input_tensor = tensor_img.unsqueeze(0).to(device)
        input_tensor.requires_grad = True 
        
        heatmap = cam_engine.generate_heatmap(input_tensor, target_class_idx, target_count)
        
        heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize((336, 336), Image.Resampling.LANCZOS)
        heatmap_colored = plt.cm.jet(np.array(heatmap_resized) / 255.0)[:, :, :3]
        
        axes[row_idx, 0].imshow(padded_orig)
        axes[row_idx, 0].set_title("Padded Input")
        axes[row_idx, 0].axis('off')
        
        axes[row_idx, 1].imshow(heatmap_colored)
        axes[row_idx, 1].set_title(f"Heat Signature for: {class_name}")
        axes[row_idx, 1].axis('off')
        
        overlay = (0.5 * np.array(padded_orig)/255.0) + (0.5 * heatmap_colored)
        overlay = np.clip(overlay, 0, 1)
        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title("The Overlay")
        axes[row_idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_grayscale_gradcam_diagnostics.png"))
    plt.close()

    print(f"Diagnostics Complete! Check '{OUTPUT_DIR}/04_grayscale_gradcam_diagnostics.png' to see the x-rays.")
    print(f"Overall Shape-Only F1 Score: {np.mean(f1):.4f}")

if __name__ == '__main__':
    main()
