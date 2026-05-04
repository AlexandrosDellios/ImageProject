import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model import UnoCountingResNet

# --- CONFIGURATION ---
REAL_DATA_DIR = "data/real_hands_cropped"  # Folder containing your 1000 real cropped hands
LABELS_CSV = "data/real_hands_labels.csv"  # CSV mapping images to cards
PHASE1_WEIGHTS = "best_phase1_resnet.pth"

BATCH_SIZE = 32        # Smaller batch size for fine-tuning
EPOCHS = 100           
PATIENCE = 10          
LEARNING_RATE = 1e-4   # 🚨 10x SMALLER than Phase 1 to prevent destroying learned weights
NUM_CLASSES = 56       
MAX_COPIES = 6         

# Define the exact 56 classes to ensure consistent indexing
# (Make sure this list exactly matches the alphabetical folder names from Phase 1!)
CLASS_NAMES = sorted([
    # Add your 54 card names here, e.g., 'b_0', 'b_1', ..., 'wild', 'wild_draw_4'
    # Plus your tokens: 'black_token', 'yellow_token'
]) 
# For safety in this script, we'll generate a dummy list if you haven't filled it yet.
if not CLASS_NAMES:
    print("⚠️ WARNING: CLASS_NAMES list is empty. Please populate it with your exact 56 class strings!")
    CLASS_NAMES = [f"class_{i}" for i in range(56)] # Replace this!

# --- 1. REAL HAND DATASET LOADER ---
class UnoRealHandDataset(Dataset):
    def __init__(self, csv_file, img_dir, class_names, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Assuming CSV has columns: 'filename' and 'cards'
        # Example 'cards' cell: "r_7;b_draw_2;yellow_token"
        img_name = os.path.join(self.img_dir, str(self.data_frame.iloc[idx]['filename']))
        image = Image.open(img_name).convert('RGB')
        
        cards_str = str(self.data_frame.iloc[idx]['cards'])
        
        # 🚨 THE MULTI-COUNT TARGET PARSER 🚨
        target = torch.zeros(NUM_CLASSES, dtype=torch.long)
        
        if cards_str.strip().upper() != "EMPTY":
            card_list = cards_str.split(';')
            for card in card_list:
                card = card.strip()
                if card in self.class_to_idx:
                    # Increment the count for this specific card!
                    # If there are two 'b_skip's, this index becomes 2.
                    target[self.class_to_idx[card]] += 1
                
        if self.transform:
            image = self.transform(image)
            
        return image, target

# --- MAIN EXECUTION BLOCK ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Fine-Tuning on device: {device}")

    # 🚨 LIGHT AUGMENTATIONS ONLY 🚨
    # Because the Kaggle test set is clean, we only apply gentle tweaks 
    # to simulate minor camera positioning and lighting differences.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=10),                 # Slight camera tilt
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Slight lighting changes
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Split your 1000 real images into Train (800) and Val (200) CSVs beforehand, 
    # or just use a standard random split here. For simplicity, assuming one train CSV here.
    full_dataset = UnoRealHandDataset(LABELS_CSV, REAL_DATA_DIR, CLASS_NAMES, transform=train_transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Overwrite the validation dataset's transform to remove random augmentations
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- 3. LOAD MODEL & PHASE 1 WEIGHTS ---
    model = UnoCountingResNet(num_classes=NUM_CLASSES, max_copies=MAX_COPIES)
    
    print(f"🧠 Loading pre-trained weights from {PHASE1_WEIGHTS}...")
    try:
        model.load_state_dict(torch.load(PHASE1_WEIGHTS, map_location=device))
        print("✅ Weights loaded successfully!")
    except FileNotFoundError:
        print(f"❌ FATAL: Could not find {PHASE1_WEIGHTS}. Did Phase 1 finish?")
        return
        
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # --- 4. FINE-TUNING LOOP ---
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Fine-Tuning")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.permute(0, 2, 1) 
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        correct_counts = 0
        total_counts = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                outputs = outputs.permute(0, 2, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1) 
                correct_counts += (preds == labels).sum().item()
                total_counts += labels.numel() 

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct_counts / total_counts) * 100
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
        
        scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0  
            torch.save(model.state_dict(), "best_finetuned_resnet.pth")
            print("💾 New fine-tuned model saved!")
        else:
            epochs_no_improve += 1
            print(f"⚠️ Validation loss did not improve. Patience: {epochs_no_improve}/{PATIENCE}")
            
            if epochs_no_improve >= PATIENCE:
                print(f"\n🛑 Early stopping triggered!")
                break 

    print("\n🎉 Phase 4 Fine-Tuning Complete! Best model is saved as 'best_finetuned_resnet.pth'")

if __name__ == '__main__':
    main()