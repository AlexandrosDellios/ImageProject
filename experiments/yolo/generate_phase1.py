"""Archived, not maintained: YOLOv8 training set made from the single card cut-outs.

Pastes one card per 640x640 canvas tiled with table background, augments it with
albumentations and writes YOLO box labels to uno_dataset/{images,labels}/{train,val}.
Old paths, relative to the working directory: data/single_cards, data/backgrounds.
See experiments/README.md.
"""

import cv2
import os
import random
import numpy as np
import albumentations as A
from glob import glob
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_CARDS_DIR = "data/single_cards"
BACKGROUNDS_DIR = "data/backgrounds"
OUTPUT_DATASET = "uno_dataset"
SAMPLES_PER_CARD = 200     # Base number of samples for regular cards
TOKEN_MULTIPLIER = 3       # Generate 3x as many samples for tokens
CANVAS_SIZE = 640          
CARD_MAX_DIM = 250         

# --- 1. SETUP ALBUMENTATIONS PIPELINE (TUNED DOWN) ---
transform = A.Compose([
    # Spatial transforms: Keep cards a reasonable size and closer to center
    A.Affine(
        scale=(0.6, 1.1),                                             
        translate_percent={"x": (-0.20, 0.20), "y": (-0.20, 0.20)},   
        rotate=(-180, 180),                                           
        p=1.0
    ),
    A.Perspective(scale=(0.02, 0.06), p=0.3),                         
    
    # Pixel transforms: Subtle lighting and camera sensor simulation
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02, p=0.6),
    A.GaussNoise(p=0.2),                                              
    A.MotionBlur(blur_limit=3, p=0.1),                                

    # Occlusion: Simulate thumbs/other cards covering parts of the card
    A.CoarseDropout(
        num_holes_range=(1, 3),                                       
        hole_height_range=(10, 25),                                   
        hole_width_range=(10, 25),
        fill_mask=0, 
        p=0.4                                                         
    ),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.4))

# --- 2. LOAD BACKGROUNDS INTO MEMORY ---
bg_paths = glob(os.path.join(BACKGROUNDS_DIR, "*.*"))
if not bg_paths:
    print(f"FATAL: No background patches found in {BACKGROUNDS_DIR}!")
    exit()

REAL_BACKGROUNDS = [cv2.imread(p) for p in bg_paths if cv2.imread(p) is not None]

# --- 3. HELPERS ---
def generate_background():
    """Tiles a real background crop to perfectly fill the 640x640 canvas."""
    bg_patch = random.choice(REAL_BACKGROUNDS)
    patch_h, patch_w = bg_patch.shape[:2]
    
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
    
    for y in range(0, CANVAS_SIZE, patch_h):
        for x in range(0, CANVAS_SIZE, patch_w):
            h = min(patch_h, CANVAS_SIZE - y)
            w = min(patch_w, CANVAS_SIZE - x)
            canvas[y:y+h, x:x+w] = bg_patch[0:h, 0:w]
            
    shift = random.randint(-15, 15)
    canvas = np.clip(canvas.astype(np.int16) + shift, 0, 255).astype(np.uint8)
    return canvas

def create_class_mapping(card_paths):
    """Assigns an integer ID to each card name and saves a YAML snippet."""
    classes = sorted(list(set([os.path.basename(p).split('.')[0] for p in card_paths])))
    mapping = {name: idx for idx, name in enumerate(classes)}
    
    with open(os.path.join(OUTPUT_DATASET, "classes_snippet.txt"), "w") as f:
        f.write("names:\n")
        for name, idx in mapping.items():
            f.write(f"  {idx}: '{name}'\n")
            
    print(f"Mapped {len(classes)} classes. Snippet saved to {OUTPUT_DATASET}/classes_snippet.txt")
    return mapping

# --- MAIN EXECUTION ---
def main():
    train_img_dir = os.path.join(OUTPUT_DATASET, "images", "train")
    train_lbl_dir = os.path.join(OUTPUT_DATASET, "labels", "train")
    val_img_dir = os.path.join(OUTPUT_DATASET, "images", "val")
    val_lbl_dir = os.path.join(OUTPUT_DATASET, "labels", "val")
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    card_paths = glob(os.path.join(SOURCE_CARDS_DIR, "*.*"))
    if not card_paths:
        print(f"FATAL: No cards found in {SOURCE_CARDS_DIR}!")
        return

    class_mapping = create_class_mapping(card_paths)
    global_img_count = 0

    # Wrap the loop in a tqdm progress bar
    pbar = tqdm(card_paths, desc="Generating Phase 1 Dataset")
    
    for card_path in pbar:
        card_name = os.path.basename(card_path).split('.')[0]
        class_id = class_mapping[card_name]
        
        # --- NEW: COMBAT CLASS IMBALANCE FOR TOKENS ---
        # If the file name contains 'token', boost its generation count
        if "token" in card_name.lower():
            target_samples = SAMPLES_PER_CARD * TOKEN_MULTIPLIER
            pbar.set_postfix({"Boosting": f"{card_name} ({target_samples}x)"})
        else:
            target_samples = SAMPLES_PER_CARD
            pbar.set_postfix({"Current": card_name})
        # ----------------------------------------------------
        
        card_img = cv2.imread(card_path, cv2.IMREAD_UNCHANGED)
        if card_img is None: continue
        
        if card_img.shape[2] == 4:
            card_img = cv2.cvtColor(card_img, cv2.COLOR_BGRA2BGR)
            
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            card_img[mask == 0] = [255, 255, 255]
            
        c_h, c_w = card_img.shape[:2]

        scale = CARD_MAX_DIM / max(c_h, c_w)
        if scale < 1.0: 
            new_w, new_h = int(c_w * scale), int(c_h * scale)
            card_img = cv2.resize(card_img, (new_w, new_h))
            c_h, c_w = new_h, new_w

        start_y = (CANVAS_SIZE - c_h) // 2
        start_x = (CANVAS_SIZE - c_w) // 2
        
        center_x, center_y = 0.5, 0.5
        norm_w, norm_h = c_w / CANVAS_SIZE, c_h / CANVAS_SIZE
        original_bbox = [center_x, center_y, norm_w, norm_h]

        generated = 0
        # Loop now uses the dynamic target_samples
        while generated < target_samples:
            bg = generate_background()
            bg[start_y:start_y+c_h, start_x:start_x+c_w] = card_img
            
            try:
                augmented = transform(image=bg, bboxes=[original_bbox], class_labels=[class_id])
            except ValueError:
                continue
            
            if len(augmented['bboxes']) == 0:
                continue
                
            aug_img = augmented['image']
            aug_bbox = augmented['bboxes'][0]
            
            is_val = random.random() < 0.2
            current_img_dir = val_img_dir if is_val else train_img_dir
            current_lbl_dir = val_lbl_dir if is_val else train_lbl_dir
            
            img_filename = f"{card_name}_{global_img_count}.jpg"
            cv2.imwrite(os.path.join(current_img_dir, img_filename), aug_img)
            
            txt_filename = f"{card_name}_{global_img_count}.txt"
            with open(os.path.join(current_lbl_dir, txt_filename), "w") as f:
                f.write(f"{class_id} {aug_bbox[0]:.6f} {aug_bbox[1]:.6f} {aug_bbox[2]:.6f} {aug_bbox[3]:.6f}\n")
            
            generated += 1
            global_img_count += 1

if __name__ == "__main__":
    main()
