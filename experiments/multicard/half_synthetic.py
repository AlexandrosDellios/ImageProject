"""Archived, not maintained: co-training set, half synthetic 2-5 card hands, half real crops.

Writes uno_multicard_cotraining/{train,val}/ and synthetic_{train,val}.csv (5,000 + 1,000
images, 336x336). The real half is drawn from data/phase4_{train,val} and padded to a
square. See experiments/README.md.
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
from tqdm import tqdm

# ==========================================
# --- CONFIGURATION ---
# ==========================================
BASE_CARDS_DIR = "data/single_cards"       
OUTPUT_DIR = "uno_multicard_cotraining"        
BACKGROUNDS_DIR = "data/backgrounds"       

# REAL DATA PATHS (For the 50% Injection)
REAL_TRAIN_DIR = "data/phase4_train"
REAL_TRAIN_CSV = "phase4_train_strict.csv"
REAL_VAL_DIR = "data/phase4_val"
REAL_VAL_CSV = "phase4_val_strict.csv"

IMAGES_TRAIN = 5000                        
IMAGES_VAL = 1000                          
IMAGE_SIZE = 336  
LETTERBOX_PROB = 0.5                       

# ==========================================
# CURRICULUM TUNING KNOBS
# ==========================================
CARDS_PER_IMAGE = (2, 5)         
SCALE_RANGE = (0.35, 0.40)       
X_STRIDE_RATIO = 0.75            
Y_JITTER_MAX = 25                

# ==========================================
# ANTI-OVERFIT & VISION BREAKER TUNING
# ==========================================
CARD_BLUR_RANGE = (0.5, 0.65)  
EDGE_FEATHER_RANGE = (1.0, 1.2) 
NOISE_SCALE_RANGE = (4.2, 5.0) 
BRIGHTNESS_RANGE = (0.94, 1.03) 
PERSPECTIVE_PROB = 0.6
PERSPECTIVE_DISTORTION = (0.05, 0.15) 
CRUNCH_PROB = 0.6
CRUNCH_FACTOR_RANGE = (1.4, 1.7) 
HOLE_PUNCH_PROB = 0.6
HOLE_PUNCH_COUNT = (1, 3) 
HOLE_PUNCH_RADIUS_PCT = (0.15, 0.35) 
# ==========================================

def create_background(size):
    try:
        bg_files = [f for f in os.listdir(BACKGROUNDS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not bg_files: raise FileNotFoundError
        bg_path = os.path.join(BACKGROUNDS_DIR, random.choice(bg_files))
        bg_img = Image.open(bg_path).convert('RGB')
        w, h = bg_img.size
        if w < size or h < size:
            scale = max(size / w, size / h)
            bg_img = bg_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)
            w, h = bg_img.size
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        return bg_img.crop((x, y, x + size, y + size))
    except (FileNotFoundError, Exception):
        return Image.new('RGB', (size, size), (150, 150, 150))

def apply_simulated_letterboxing(img):
    draw = ImageDraw.Draw(img)
    if random.random() < 0.5:
        bar_size = random.randint(20, 80)
        draw.rectangle([0, 0, IMAGE_SIZE, bar_size], fill=(0,0,0))
        draw.rectangle([0, IMAGE_SIZE - bar_size, IMAGE_SIZE, IMAGE_SIZE], fill=(0,0,0))
    else:
        bar_size = random.randint(20, 80)
        draw.rectangle([0, 0, bar_size, IMAGE_SIZE], fill=(0,0,0))
        draw.rectangle([IMAGE_SIZE - bar_size, 0, IMAGE_SIZE, IMAGE_SIZE], fill=(0,0,0))
    return img

def generate_synthetic_hand(base_files):
    canvas = create_background(IMAGE_SIZE)
    num_cards = random.randint(CARDS_PER_IMAGE[0], CARDS_PER_IMAGE[1])
    selected_files = random.choices(base_files, k=num_cards)
    card_labels = [f.replace('.png', '').lower() for f in selected_files]
    
    global_scale = random.uniform(*SCALE_RANGE)
    sample_img = Image.open(os.path.join(BASE_CARDS_DIR, selected_files[0]))
    base_w, base_h = sample_img.size
    card_h = int(IMAGE_SIZE * global_scale)
    card_w = int(card_h * (base_w / base_h))
    
    x_offsets = [0]
    current_x = 0
    for i in range(num_cards - 1):
        is_current_token = 'token' in selected_files[i].lower()
        is_next_token = 'token' in selected_files[i+1].lower()
        
        if is_current_token or is_next_token:
            stride_pixels = int(card_w * 1.1) 
        else:
            stride_pixels = int(card_w * X_STRIDE_RATIO)
            
        current_x += stride_pixels
        x_offsets.append(current_x)
        
    total_hand_width = x_offsets[-1] + card_w
    start_x = (IMAGE_SIZE - total_hand_width) // 2
    start_y = (IMAGE_SIZE - card_h) // 2
    
    for i, filename in enumerate(selected_files):
        card_img = Image.open(os.path.join(BASE_CARDS_DIR, filename)).convert("RGBA")
        card = card_img.resize((card_w, card_h), Image.Resampling.LANCZOS)
        
        angle = random.uniform(-15, 15)
        card = card.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

        if random.random() < PERSPECTIVE_PROB:
            w, h = card.size
            distortion = random.uniform(*PERSPECTIVE_DISTORTION) 
            if random.random() < 0.5:
                coeffs = (1, 0, 0, 0, 1, 0, random.uniform(-distortion, distortion)/w, 0)
            else:
                coeffs = (1, 0, 0, 0, 1, 0, 0, random.uniform(-distortion, distortion)/h)
            card = card.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)

        card = card.filter(ImageFilter.GaussianBlur(radius=random.uniform(*CARD_BLUR_RANGE)))
        r, g, b, a = card.split()
        a = a.filter(ImageFilter.GaussianBlur(radius=random.uniform(*EDGE_FEATHER_RANGE)))
        card = Image.merge('RGBA', (r, g, b, a))
        
        if random.random() < CRUNCH_PROB:
            crunch_factor = random.uniform(*CRUNCH_FACTOR_RANGE)
            crunch_w = max(10, int(card.width / crunch_factor))
            crunch_h = max(10, int(card.height / crunch_factor))
            crunched = card.resize((crunch_w, crunch_h), Image.Resampling.NEAREST)
            card = crunched.resize((card.width, card.height), Image.Resampling.BICUBIC)

        if random.random() < HOLE_PUNCH_PROB:
            draw = ImageDraw.Draw(card)
            for _ in range(random.randint(*HOLE_PUNCH_COUNT)):
                min_rad = int(card.width * HOLE_PUNCH_RADIUS_PCT[0])
                max_rad = int(card.width * HOLE_PUNCH_RADIUS_PCT[1])
                hole_radius = random.randint(min_rad, max_rad)
                cx = card.width // 2 + random.randint(-20, 20)
                cy = card.height // 2 + random.randint(-20, 20)
                draw.ellipse([cx - hole_radius, cy - hole_radius, cx + hole_radius, cy + hole_radius], fill=(0, 0, 0, 0))
        
        offset_x = start_x + x_offsets[i]
        offset_y = start_y + random.randint(-Y_JITTER_MAX, Y_JITTER_MAX)
        canvas.paste(card, (int(offset_x), int(offset_y)), card)

    enhancer = ImageEnhance.Brightness(canvas)
    canvas = enhancer.enhance(random.uniform(*BRIGHTNESS_RANGE))

    if random.random() < LETTERBOX_PROB:
        canvas = apply_simulated_letterboxing(canvas)
        
    img_arr = np.array(canvas, dtype=np.float32)
    noise = np.random.normal(loc=0, scale=random.uniform(*NOISE_SCALE_RANGE), size=img_arr.shape)
    canvas = Image.fromarray(np.clip(img_arr + noise, 0, 255).astype(np.uint8))
    
    label_str = ";".join(card_labels)
    return canvas, label_str

def process_real_hand(df, img_dir):
    """Pulls a real image, perfectly pads it to 336x336, and applies global domain augmentations."""
    row = df.sample(1).iloc[0]
    img_path = os.path.join(img_dir, str(row['filename']))
    labels = str(row['cards'])
    
    img = Image.open(img_path).convert('RGB')
    
    # Safely pad the wide rectangle to a perfect square (This is natural letterboxing!)
    img = ImageOps.pad(img, (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
    
    # Apply global augmentations to match the synthetic domain
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(*BRIGHTNESS_RANGE))
    
    img_arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(loc=0, scale=random.uniform(*NOISE_SCALE_RANGE), size=img_arr.shape)
    img = Image.fromarray(np.clip(img_arr + noise, 0, 255).astype(np.uint8))
    
    return img, labels

def main():
    print("Starting 50/50 Co-Training Generator (Synthetic + Real Mix)...")
    
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "val"), exist_ok=True)
    
    base_files = [f for f in os.listdir(BASE_CARDS_DIR) if f.endswith('.png')]
    
    # Load Real Data CSVs
    print("Loading real data CSVs...")
    real_train_df = pd.read_csv(REAL_TRAIN_CSV)
    real_val_df = pd.read_csv(REAL_VAL_CSV)
    
    train_rows, val_rows = [], []

    for i in tqdm(range(IMAGES_TRAIN), desc="Generating Train (Mixed)"):
        if random.random() < 0.5:
            img, labels = generate_synthetic_hand(base_files)
            src = "synth"
        else:
            img, labels = process_real_hand(real_train_df, REAL_TRAIN_DIR)
            src = "real"
            
        filename = f"mixed_train_{src}_{i}.jpg"
        img.save(os.path.join(OUTPUT_DIR, "train", filename), quality=95)
        train_rows.append({"filename": filename, "cards": labels})

    for i in tqdm(range(IMAGES_VAL), desc="Generating Val (Mixed)"):
        if random.random() < 0.5:
            img, labels = generate_synthetic_hand(base_files)
            src = "synth"
        else:
            img, labels = process_real_hand(real_val_df, REAL_VAL_DIR)
            src = "real"
            
        filename = f"mixed_val_{src}_{i}.jpg"
        img.save(os.path.join(OUTPUT_DIR, "val", filename), quality=95)
        val_rows.append({"filename": filename, "cards": labels})

    pd.DataFrame(train_rows).to_csv(os.path.join(OUTPUT_DIR, "synthetic_train.csv"), index=False)
    pd.DataFrame(val_rows).to_csv(os.path.join(OUTPUT_DIR, "synthetic_val.csv"), index=False)
    
    print("\n50/50 Co-Training Dataset Complete!")

if __name__ == '__main__':
    main()
