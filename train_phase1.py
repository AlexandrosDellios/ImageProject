import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model import UnoCountingResNet

# --- CONFIGURATION ---
DATASET_DIR = "uno_classification_dataset"
BATCH_SIZE = 64
EPOCHS = 300           
PATIENCE = 15          
LEARNING_RATE = 1e-3
NUM_CLASSES = 56       
MAX_COPIES = 6         

# --- 1. DATASET LOADER ---
class UnoPhase1Dataset(Dataset):
    def __init__(self, root_dir, class_names, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = class_names
        
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.image_paths = []
        self.labels = []
        
        print(f"Scanning {root_dir}...")
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir): continue
            
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    
                    target = torch.zeros(NUM_CLASSES, dtype=torch.long)
                    target[self.class_to_idx[class_name]] = 1
                    self.labels.append(target)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- MAIN EXECUTION BLOCK (Fixes the Windows Crash) ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")
    
    if device.type == 'cpu':
        print("⚠️ WARNING: Training on CPU. This will be slow! Check your PyTorch CUDA installation.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    if len(class_names) != NUM_CLASSES:
        print(f"⚠️ Warning: Found {len(class_names)} classes in folder, but expected {NUM_CLASSES}.")

    train_dataset = UnoPhase1Dataset(train_dir, class_names, transform)
    val_dataset = UnoPhase1Dataset(val_dir, class_names, transform)

    # Windows multi-processing works correctly when called from inside main()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = UnoCountingResNet(num_classes=NUM_CLASSES, max_copies=MAX_COPIES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training")
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
            torch.save(model.state_dict(), "best_phase1_resnet.pth")
            print("💾 New best model saved!")
        else:
            epochs_no_improve += 1
            print(f"⚠️ Validation loss did not improve. Patience: {epochs_no_improve}/{PATIENCE}")
            
            if epochs_no_improve >= PATIENCE:
                print(f"\n🛑 Early stopping triggered! Model hasn't improved in {PATIENCE} epochs.")
                break 

    print("\n🎉 Phase 1 Pre-training Complete! Best model is saved as 'best_phase1_resnet.pth'")


# This is the crucial block for Windows
if __name__ == '__main__':
    main()