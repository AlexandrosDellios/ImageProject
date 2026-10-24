"""Archived, not maintained: the staged training script, one stage per run (STAGE_TO_RUN).

0: grayscale shapes (uno_step0_grayscale, 17 classes). 1: single colour cards
(uno_classification_dataset). 2: synthetic multi-card hands (uno_multicard_step3), from the
stage 1 weights. 3: fine-tuning on the real crops. Stages 0-2 are the experiments in
experiments/README.md; stage 3 is maintained as scripts/finetune.py. Paths are relative to
this file's folder (SCRIPT_DIR), which was the project root.
"""

import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for uno_vision
from uno_vision.model import UnoCountingResNet

# ==========================================
# --- AUTO-LOCATOR ---
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# --- MASTER CONTROLLER ---
# ==========================================
# 0 = Step 0: Grayscale 
# 1 = Step 1: Color 
# 2 = Step 2: Synthetic Multi-Card
# 3 = Step 3: Real Kaggle Fine-Tuning
STAGE_TO_RUN = 3  

# STARTING WEIGHTS (Absolute Paths)
START_WEIGHTS_STEP1 = None 
START_WEIGHTS_STEP2 = os.path.join(SCRIPT_DIR, "best_phase1_resnet.pth") 
START_WEIGHTS_STEP3 = os.path.join(SCRIPT_DIR, "phase1_best_weights.pth") 

# ==========================================
# --- HYPERPARAMETERS ---
# ==========================================
BATCH_SIZE = 64
MAX_COPIES = 6 
NUM_CLASSES_FULL = 56
NUM_CLASSES_STEP0 = 17

LR_STEP0, EPOCHS_STEP0, PATIENCE_STEP0 = 9e-4, 100, 15
LR_STEP1, EPOCHS_STEP1, PATIENCE_STEP1 = 8e-4, 300, 20
LR_STEP2, EPOCHS_STEP2, PATIENCE_STEP2 = 6e-4, 300, 20

LR_STEP3_WARMUP, EPOCHS_STEP3_WARMUP, PATIENCE_STEP3_WARMUP = 4e-4, 100, 20
LR_STEP3_POLISH, EPOCHS_STEP3_POLISH, PATIENCE_STEP3_POLISH = 6e-4, 200, 25

# ==========================================
# --- FILE PATHS  ---
# ==========================================
DIR_DIAGNOSTICS = os.path.join(SCRIPT_DIR, "diagnostics")
os.makedirs(DIR_DIAGNOSTICS, exist_ok=True)

DIR_STEP0 = os.path.join(SCRIPT_DIR, "uno_step0_grayscale")
DIR_STEP1 = os.path.join(SCRIPT_DIR, "uno_classification_dataset")

DIR_STEP2_TRAIN_IMG = os.path.join(SCRIPT_DIR, "uno_multicard_step3/train")
DIR_STEP2_VAL_IMG = os.path.join(SCRIPT_DIR, "uno_multicard_step3/val")
CSV_STEP2_TRAIN = os.path.join(SCRIPT_DIR, "uno_multicard_step3/synthetic_train.csv")
CSV_STEP2_VAL = os.path.join(SCRIPT_DIR, "uno_multicard_step3/synthetic_val.csv")

DIR_STEP3_TRAIN_IMG = os.path.join(SCRIPT_DIR, "data/phase4_train")
DIR_STEP3_VAL_IMG = os.path.join(SCRIPT_DIR, "data/phase4_val")
CSV_STEP3_TRAIN = os.path.join(SCRIPT_DIR, "phase4_train_strict.csv")
CSV_STEP3_VAL = os.path.join(SCRIPT_DIR, "phase4_val_strict.csv")

W_STEP0 = os.path.join(SCRIPT_DIR, "best_step0_grayscale.pth")
W_STEP1 = os.path.join(SCRIPT_DIR, "best_phase1_resnet.pth")
W_STEP2 = os.path.join(SCRIPT_DIR, "best_final.pth")
W_STEP3 = os.path.join(SCRIPT_DIR, "best_test.pth")

CLASS_NAMES_56 = [
    'b_0', 'b_1', 'b_2', 'b_3', 'b_4', 'b_5', 'b_6', 'b_7', 'b_8', 'b_9', 
    'b_draw_2', 'b_reverse', 'b_skip', 'black_token', 'draw_4', 
    'g_0', 'g_1', 'g_2', 'g_3', 'g_4', 'g_5', 'g_6', 'g_7', 'g_8', 'g_9', 
    'g_draw_2', 'g_reverse', 'g_skip', 
    'r_0', 'r_1', 'r_2', 'r_3', 'r_4', 'r_5', 'r_6', 'r_7', 'r_8', 'r_9', 
    'r_draw_2', 'r_reverse', 'r_skip', 'wild', 
    'y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6', 'y_7', 'y_8', 'y_9', 
    'y_draw_2', 'y_reverse', 'y_skip', 'yellow_token',
]

# ==========================================
# --- UTILITIES & LOSS ---
# ==========================================
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

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss) 
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

# ==========================================
# --- PLOTTING  ---
# ==========================================
def save_stage_plots(history, stage_name):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Training Metrics: {stage_name}", fontsize=16, fontweight='bold')
    
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='red', linewidth=2)
    ax1.set_title("Loss Curve"); ax1.set_xlabel("Epochs"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.plot(epochs, history['val_acc'], label='Active Accuracy', color='green', linewidth=2)
    ax2.set_title("Validation Accuracy"); ax2.set_xlabel("Epochs"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.7)
    
    clean_name = stage_name.lower().replace(" ", "_").replace(":", "")
    save_path = os.path.join(DIR_DIAGNOSTICS, f"curves_{clean_name}.jpg")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300); plt.close()
    print(f"Training curves saved to: {save_path}")

# ==========================================
# --- DATASETS  ---
# ==========================================
class UnoFolderDataset(Dataset):
    def __init__(self, root_dir, num_classes, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.image_paths, self.labels = [], []
        
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    target = torch.zeros(num_classes, dtype=torch.long)
                    target[self.class_to_idx[class_name]] = 1 
                    self.labels.append(target)

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, self.labels[idx]

class UnoCSVDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file) # Explicitly assigned
        self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES_56)}

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(self.df.iloc[idx]['filename']))
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception:
            image = Image.new('RGB', (336, 336), (0, 0, 0))
            
        cards_str = str(self.df.iloc[idx]['cards'])
        target = torch.zeros(NUM_CLASSES_FULL, dtype=torch.long)
        
        if cards_str.strip().upper() != "EMPTY":
            for card in cards_str.split(';'):
                card = card.strip()
                if card in self.class_to_idx:
                    target[self.class_to_idx[card]] = min(target[self.class_to_idx[card]] + 1, MAX_COPIES - 1)
                    
        if self.transform: image = self.transform(image)
        return image, target

# ==========================================
# --- TRAINING ---
# ==========================================
def run_training_stage(model, train_loader, val_loader, criterion, optimizer, scheduler, device, stage_name, epochs, patience, save_path):
    print(f"\n{'='*50}\n STARTING {stage_name}\n{'='*50}")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train(); running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).permute(0, 2, 1) 
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval(); val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).permute(0, 2, 1)
                val_loss += criterion(outputs, labels).item()
                preds = torch.argmax(outputs, dim=1) 
                mask = labels > 0
                correct += (preds[mask] == labels[mask]).sum().item()
                total += mask.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct / total * 100) if total > 0 else 0.0
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        
        print(f"Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.2f}%")
        
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(avg_val_loss)
        else: scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0  
            torch.save(model.state_dict(), save_path)
            print("Saved best weights!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping.")
                break 

    save_stage_plots(history, stage_name)
    return history

# ==========================================
# --- MAIN ---
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    transform = transforms.Compose([
        PadToSquare(fill=0), transforms.Resize((336, 336)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    criterion = FocalLoss()

    # --- SELECTING STAGE ---
    if STAGE_TO_RUN == 0:
        model = UnoCountingResNet(num_classes=NUM_CLASSES_STEP0, max_copies=MAX_COPIES).to(device)
        train_loader = DataLoader(UnoFolderDataset(os.path.join(DIR_STEP0, "train"), NUM_CLASSES_STEP0, transform), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(UnoFolderDataset(os.path.join(DIR_STEP0, "val"), NUM_CLASSES_STEP0, transform), batch_size=BATCH_SIZE)
        optimizer = optim.AdamW(model.parameters(), lr=LR_STEP0, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        run_training_stage(model, train_loader, val_loader, criterion, optimizer, scheduler, device, "STEP 0", EPOCHS_STEP0, PATIENCE_STEP0, W_STEP0)

    elif STAGE_TO_RUN == 1:
        model = UnoCountingResNet(num_classes=NUM_CLASSES_FULL, max_copies=MAX_COPIES).to(device)
        if START_WEIGHTS_STEP1 and os.path.exists(START_WEIGHTS_STEP1):
            weights = torch.load(START_WEIGHTS_STEP1, map_location=device, weights_only=True)
            model.load_state_dict({k:v for k,v in weights.items() if 'classifier' not in k}, strict=False)
        train_loader = DataLoader(UnoFolderDataset(os.path.join(DIR_STEP1, "train"), NUM_CLASSES_FULL, transform), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(UnoFolderDataset(os.path.join(DIR_STEP1, "val"), NUM_CLASSES_FULL, transform), batch_size=BATCH_SIZE)
        optimizer = optim.AdamW(model.parameters(), lr=LR_STEP1, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        run_training_stage(model, train_loader, val_loader, criterion, optimizer, scheduler, device, "STEP 1", EPOCHS_STEP1, PATIENCE_STEP1, W_STEP1)

    elif STAGE_TO_RUN == 2:
        model = UnoCountingResNet(num_classes=NUM_CLASSES_FULL, max_copies=MAX_COPIES).to(device)
        if START_WEIGHTS_STEP2 and os.path.exists(START_WEIGHTS_STEP2):
            model.load_state_dict(torch.load(START_WEIGHTS_STEP2, map_location=device, weights_only=True))
        train_loader = DataLoader(UnoCSVDataset(CSV_STEP2_TRAIN, DIR_STEP2_TRAIN_IMG, transform), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(UnoCSVDataset(CSV_STEP2_VAL, DIR_STEP2_VAL_IMG, transform), batch_size=BATCH_SIZE)
        optimizer = optim.AdamW(model.parameters(), lr=LR_STEP2, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        run_training_stage(model, train_loader, val_loader, criterion, optimizer, scheduler, device, "STEP 2", EPOCHS_STEP2, PATIENCE_STEP2, W_STEP2)

    elif STAGE_TO_RUN == 3:
        model = UnoCountingResNet(num_classes=NUM_CLASSES_FULL, max_copies=MAX_COPIES).to(device)
        if START_WEIGHTS_STEP3 and os.path.exists(START_WEIGHTS_STEP3):
            model.load_state_dict(torch.load(START_WEIGHTS_STEP3, map_location=device, weights_only=True))
        
        train_loader = DataLoader(UnoCSVDataset(CSV_STEP3_TRAIN, DIR_STEP3_TRAIN_IMG, transform), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(UnoCSVDataset(CSV_STEP3_VAL, DIR_STEP3_VAL_IMG, transform), batch_size=BATCH_SIZE)

        # 3A: WARM-UP (Frozen)
        for param in model.parameters(): param.requires_grad = False
        for name, param in model.named_parameters():
            if any(k in name for k in ['fc', 'classifier', 'head']): param.requires_grad = True
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_STEP3_WARMUP)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        run_training_stage(model, train_loader, val_loader, criterion, optimizer, scheduler, device, "STEP 3A: WARMUP", EPOCHS_STEP3_WARMUP, PATIENCE_STEP3_WARMUP, W_STEP3)

        # 3B: POLISH (Unfrozen)
        model.load_state_dict(torch.load(W_STEP3, map_location=device, weights_only=True))
        for param in model.parameters(): param.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=LR_STEP3_POLISH)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        run_training_stage(model, train_loader, val_loader, criterion, optimizer, scheduler, device, "STEP 3B: POLISH", EPOCHS_STEP3_POLISH, PATIENCE_STEP3_POLISH, W_STEP3)

if __name__ == '__main__': main()
