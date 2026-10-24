"""Archived, not maintained: synthetic hands of 2-3 cards side by side ("step 2" data).

Writes uno_multicard_step3/{train,val}/ and synthetic_{train,val}.csv (5,000 + 1,000
images, 336x336). Stage 2 of experiments/multistage/train_multistage.py trains on it.
See experiments/README.md.
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from tqdm import tqdm
from torchvision.transforms import ColorJitter # ADDED: The Color Scrambler

# ==========================================
# --- CONFIGURATION ---
# ==========================================
BASE_CARDS_DIR = "data/single_cards"       
OUTPUT_DIR = "uno_multicard_step3"        
BACKGROUNDS_DIR = "data/backgrounds"       

IMAGES_TRAIN = 5000                        
IMAGES_VAL = 1000                          
IMAGE_SIZE = 336  
LETTERBOX_PROB = 0.5                       

# ==========================================
# CURRICULUM TUNING KNOBS
# ==========================================
CARDS_PER_IMAGE = (2, 3)         
SCALE_RANGE = (0.35, 0.50)       
X_STRIDE_RATIO = 0.9           
Y_JITTER_MAX = 25                

# ==========================================
# ANTI-OVERFIT & VISION BREAKER TUNING
# ==========================================
CARD_BLUR_RANGE = (0.9, 1.3)  
EDGE_FEATHER_RANGE = (1.5, 2.0) 
NOISE_SCALE_RANGE = (7.0, 10.0) 
BRIGHTNESS_RANGE = (0.94, 1.1) 

PERSPECTIVE_PROB = 0.8
PERSPECTIVE_DISTORTION = (0.45, 0.5) 

CRUNCH_PROB = 0.6
CRUNCH_FACTOR_RANGE = (1.6, 1.7) 

HOLE_PUNCH_PROB = 0.0
HOLE_PUNCH_COUNT = (1, 3) 
HOLE_PUNCH_RADIUS_PCT = (0.15, 0.35) 

GLARE_PROB = 0.0
GLARE_ALPHA_RANGE = (50, 150) 

# 6. BASE IMAGE COLOR SCRAMBLER (Kills Texture Memorization)
SCRAMBLE_HUE = 0.05        # Shifts colors slightly (e.g., Red becomes slightly orange or pink)
SCRAMBLE_SAT = 0.3         # Washes out or super-saturates the color
SCRAMBLE_BRIGHT = 0.3      # Lightens or darkens the base card
SCRAMBLE_CONTRAST = 0.3    # Flattens or sharpens the internal shadows
# ==========================================

# Initialize the scrambler once
color_scrambler = ColorJitter(
    brightness=SCRAMBLE_BRIGHT, 
    contrast=SCRAMBLE_CONTRAST, 
    saturation=SCRAMBLE_SAT, 
    hue=SCRAMBLE_HUE
)

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
    
    # DYNAMIC STRIDE LOGIC
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
        
        # THE FIX: Scramble the hue, saturation, and brightness BEFORE scaling/rotating
        r, g, b, a = card_img.split()
        rgb_img = Image.merge("RGB", (r, g, b))
        rgb_img = color_scrambler(rgb_img) 
        r, g, b = rgb_img.split()
        card_img = Image.merge("RGBA", (r, g, b, a))
        
        card = card_img.resize((card_w, card_h), Image.Resampling.LANCZOS)
        
        angle = random.uniform(-15, 15)
        card = card.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

        # ANTI-OVERFIT: 3D Perspective Distortion
        if random.random() < PERSPECTIVE_PROB:
            w, h = card.size
            distortion = random.uniform(*PERSPECTIVE_DISTORTION) 
            if random.random() < 0.5:
                coeffs = (1, 0, 0, 0, 1, 0, random.uniform(-distortion, distortion)/w, 0)
            else:
                coeffs = (1, 0, 0, 0, 1, 0, 0, random.uniform(-distortion, distortion)/h)
            card = card.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)

        # ANTI-OVERFIT: Blur & Feathering
        card = card.filter(ImageFilter.GaussianBlur(radius=random.uniform(*CARD_BLUR_RANGE)))
        r, g, b, a = card.split()
        a = a.filter(ImageFilter.GaussianBlur(radius=random.uniform(*EDGE_FEATHER_RANGE)))
        card = Image.merge('RGBA', (r, g, b, a))
        
        # ANTI-OVERFIT: The High-Frequency Killer (Crunching)
        if random.random() < CRUNCH_PROB:
            crunch_factor = random.uniform(*CRUNCH_FACTOR_RANGE)
            crunch_w = max(10, int(card.width / crunch_factor))
            crunch_h = max(10, int(card.height / crunch_factor))
            crunched = card.resize((crunch_w, crunch_h), Image.Resampling.NEAREST)
            card = crunched.resize((card.width, card.height), Image.Resampling.BICUBIC)

        # ANTI-OVERFIT: Targeted Cutout (The Hole Punch)
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

    # ANTI-OVERFIT: Global Brightness Randomness
    enhancer = ImageEnhance.Brightness(canvas)
    canvas = enhancer.enhance(random.uniform(*BRIGHTNESS_RANGE))

    # SIMULATED LETTERBOXING
    if random.random() < LETTERBOX_PROB:
        canvas = apply_simulated_letterboxing(canvas)
        
    # ANTI-OVERFIT: Global static noise
    img_arr = np.array(canvas, dtype=np.float32)
    noise = np.random.normal(loc=0, scale=random.uniform(*NOISE_SCALE_RANGE), size=img_arr.shape)
    canvas = Image.fromarray(np.clip(img_arr + noise, 0, 255).astype(np.uint8))
    
    label_str = ";".join(card_labels)
    return canvas, label_str

def main():
    print("Starting Tunable Phase 2: Curriculum Multi-Card Hands...")
    print(f"Settings: {CARDS_PER_IMAGE[0]}-{CARDS_PER_IMAGE[1]} cards, Standard Overlap: {X_STRIDE_RATIO}, Scale: {SCALE_RANGE}")
    
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "val"), exist_ok=True)
    
    base_files = [f for f in os.listdir(BASE_CARDS_DIR) if f.endswith('.png')]
    train_rows, val_rows = [], []

    for i in tqdm(range(IMAGES_TRAIN), desc="Generating Train"):
        img, labels = generate_synthetic_hand(base_files)
        filename = f"synth_hand_{i}.jpg"
        img.save(os.path.join(OUTPUT_DIR, "train", filename), quality=95)
        train_rows.append({"filename": filename, "cards": labels})

    for i in tqdm(range(IMAGES_VAL), desc="Generating Val"):
        img, labels = generate_synthetic_hand(base_files)
        filename = f"synth_hand_val_{i}.jpg"
        img.save(os.path.join(OUTPUT_DIR, "val", filename), quality=95)
        val_rows.append({"filename": filename, "cards": labels})

    pd.DataFrame(train_rows).to_csv(os.path.join(OUTPUT_DIR, "synthetic_train.csv"), index=False)
    pd.DataFrame(val_rows).to_csv(os.path.join(OUTPUT_DIR, "synthetic_val.csv"), index=False)
    
    print("\nDataset Stage Complete!")

if __name__ == '__main__':
    main()
