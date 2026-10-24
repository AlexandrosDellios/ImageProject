"""Archived, not maintained: train from scratch on the real crops, small hands first.

Runs the four curriculum_*_stage{1-4}.csv stages in turn on data/phase4_{train,val} and
writes best_phase4_stage{1-4}.pth (56 classes x 5 counts). The surviving checkpoints have the
old head without dropout (fc); with uno_vision.model a re-run builds the current head
(dropout 0.1, fc.1). On current torch it stops with a TypeError: ReduceLROnPlateau no longer
accepts verbose=. See experiments/README.md.
"""

import os
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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for uno_vision
from uno_vision.model import UnoCountingResNet

# ==========================================
# --- CONFIGURATION ---
# ==========================================
# Set to a string like "best_phase1_resnet.pth" to load weights, or None to train from scratch
STARTING_WEIGHTS = None 

REAL_IMG_DIR_TRAIN = "data/phase4_train"
REAL_IMG_DIR_VAL = "data/phase4_val"
CURRICULUM_PREFIX = "curriculum"

BATCH_SIZE = 32
MAX_COPIES = 5
NUM_CLASSES_FULL = 56

# Hyperparameters for the curriculum stages
LEARNING_RATE = 1e-4  
PATIENCE = 10         
MAX_EPOCHS_PER_STAGE = 50

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
# --- ARCHITECTURE & DATASETS ---
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
    # Label Smoothing prevents the model from becoming overly confident
    def __init__(self, gamma=2.0, reduction='mean', label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss) 
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

class UnoCSVDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES_56)}

    def __len__(self): return len(self.data_frame)
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(self.data_frame.iloc[idx]['filename']))
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception:
            image = Image.new('RGB', (336, 336), (0, 0, 0))
            
        cards_str = str(self.data_frame.iloc[idx]['cards'])
        target = torch.zeros(NUM_CLASSES_FULL, dtype=torch.long)
        
        if cards_str.strip().upper() != "EMPTY":
            for card in cards_str.split(';'):
                card = card.strip()
                if card in self.class_to_idx:
                    target[self.class_to_idx[card]] += 1
                    
        if self.transform: image = self.transform(image)
        return image, target


# ==========================================
# --- MILD REAL-WORLD TRANSFORMS ---
# ==========================================
val_transform = transforms.Compose([
    PadToSquare(fill=0),             
    transforms.Resize((336, 336)),   
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    PadToSquare(fill=0),             
    transforms.Resize((336, 336)), 
    # 1. "Rotates upside down once" (Flips combine to create 180 degree rotation possibilities)
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # 2. "Changes the size of the cards a tiny bit" (Zooms in or out by 5%, slight 5-degree tilt)
    transforms.RandomAffine(degrees=5, scale=(0.95, 1.05)),
    # 3. Very slight lighting changes
    transforms.ColorJitter(brightness=0.1, contrast=0.1), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==========================================
# --- CURRICULUM TRAINING ENGINE ---
# ==========================================
def run_stage(model, stage_num, train_csv, val_csv, device):
    print(f"\n{'='*50}")
    print(f"STARTING CURRICULUM STAGE {stage_num}")
    print(f"{'='*50}")

    train_dataset = UnoCSVDataset(train_csv, REAL_IMG_DIR_TRAIN, transform=train_transform)
    val_dataset = UnoCSVDataset(val_csv, REAL_IMG_DIR_VAL, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    save_path = f"best_phase4_stage{stage_num}.pth"
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS_PER_STAGE):
        print(f"\n--- Epoch {epoch+1}/{MAX_EPOCHS_PER_STAGE} [Stage {stage_num}] ---")
        
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc="Training")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).permute(0, 2, 1) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss, correct_counts, total_counts = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).permute(0, 2, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1) 
                active_mask = labels > 0
                correct_counts += (preds[active_mask] == labels[active_mask]).sum().item()
                total_counts += active_mask.sum().item() 

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct_counts / total_counts) * 100 if total_counts > 0 else 0.0
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Active Accuracy: {val_accuracy:.2f}%")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0  
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved as '{save_path}'!")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Patience: {epochs_no_improve}/{PATIENCE}")
            if epochs_no_improve >= PATIENCE:
                print(f"\nPatience Reached! Moving to next curriculum stage.")
                break 

    return save_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Curriculum Pipeline Initialized on {device}")
    
    model = UnoCountingResNet(num_classes=NUM_CLASSES_FULL, max_copies=MAX_COPIES).to(device)
    
    # FIX: Safely checks if STARTING_WEIGHTS exists before trying to load it
    if STARTING_WEIGHTS is not None and os.path.exists(STARTING_WEIGHTS):
        print(f"Loading synthetic backbone from {STARTING_WEIGHTS}...")
        model.load_state_dict(torch.load(STARTING_WEIGHTS, map_location=device, weights_only=True))
    else:
        print("Training entirely from scratch (No starting weights provided).")

    # STAGE 1: Easy Hands (1-2 cards)
    train_csv = f"{CURRICULUM_PREFIX}_train_stage1.csv"
    val_csv = f"{CURRICULUM_PREFIX}_val_stage1.csv"
    best_weights = run_stage(model, 1, train_csv, val_csv, device)
    
    # STAGE 2: Easy-Medium Hands (1-3 cards)
    model.load_state_dict(torch.load(best_weights, map_location=device, weights_only=True))
    train_csv = f"{CURRICULUM_PREFIX}_train_stage2.csv"
    val_csv = f"{CURRICULUM_PREFIX}_val_stage2.csv"
    best_weights = run_stage(model, 2, train_csv, val_csv, device)

    # STAGE 3: Medium-Hard Hands (1-4 cards)
    model.load_state_dict(torch.load(best_weights, map_location=device, weights_only=True))
    train_csv = f"{CURRICULUM_PREFIX}_train_stage3.csv"
    val_csv = f"{CURRICULUM_PREFIX}_val_stage3.csv"
    best_weights = run_stage(model, 3, train_csv, val_csv, device)

    # STAGE 4: ALL Hands (1-5 cards)
    model.load_state_dict(torch.load(best_weights, map_location=device, weights_only=True))
    train_csv = f"{CURRICULUM_PREFIX}_train_stage4.csv"
    val_csv = f"{CURRICULUM_PREFIX}_val_stage4.csv"
    best_weights = run_stage(model, 4, train_csv, val_csv, device)

    print(f"\nPure Curriculum Complete! Your final, battle-ready model is saved as: {best_weights}")

if __name__ == '__main__':
    main()
