"""Archived, not maintained: like generate_single_cards.py, but keeps only the top-left corner.

Each card is cropped to its corner index (CROP_AMOUNTS); tokens stay whole and smaller.
Writes uno_classification_corners/{train,val}/<class>/. See experiments/README.md.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from tqdm import tqdm

# ==========================================
# --- CONFIGURATION ---
# ==========================================
BASE_CARDS_DIR = "data/single_cards"               
OUTPUT_DIR = "uno_classification_corners"  
BACKGROUNDS_DIR = "data/backgrounds" 

IMAGES_PER_CLASS_TRAIN = 250                
IMAGES_PER_CLASS_VAL = 50                   
IMAGE_SIZE = 336  

# THE NEW DEAD-SIMPLE CROP TUNER
# How much of the card do you want to chop off from each side? (Left, Top, Right, Bottom)
# 0.0 = Chop nothing. 0.5 = Chop half the card from that side.
# Example: (0.6, 0.0, 0.0, 0.6) -> Chops 60% off the left and 60% off the bottom. Leaves the Top-Right corner!
CROP_AMOUNTS = (0.06, 0.07, 0.68, 0.75) 

TOKEN_SCALE_MULTIPLIER = 0.6  # Shrinks full tokens so they match the size of cropped corners

ROTATION_RANGE = (0, 360)     
SCALE_RANGE = (0.3, 0.65)      
LETTERBOX_PROB = 0.5          

# ==========================================
# ANTI-OVERFIT & VISION BREAKER TUNING
# ==========================================
CARD_BLUR_RANGE = (0.5, 0.65)  
EDGE_FEATHER_RANGE = (1.0, 1.2) 
NOISE_SCALE_RANGE = (4.2, 5.0) 
BRIGHTNESS_RANGE = (0.94, 1.03) 
PERSPECTIVE_PROB = 0.0
PERSPECTIVE_DISTORTION = (0.05, 0.15) 
CRUNCH_PROB = 0.6
CRUNCH_FACTOR_RANGE = (1.4, 1.7) 
HOLE_PUNCH_PROB = 0.0
HOLE_PUNCH_COUNT = (1, 3) 
HOLE_PUNCH_RADIUS_PCT = (0.15, 0.35) 
GLARE_PROB = 0.0
GLARE_ALPHA_RANGE = (30, 80)
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
        return Image.new('RGB', (size, size), (255, 255, 255))

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

def generate_variation(base_card_img, is_token=False):
    if is_token:
        corner_img = base_card_img
        scale_multiplier = TOKEN_SCALE_MULTIPLIER
    else:
        # THE NEW INTUITIVE CROP MATH
        w, h = base_card_img.size
        crop_l, crop_t, crop_r, crop_b = CROP_AMOUNTS
        
        # Calculate exactly how many pixels to chop away from each side
        left = int(w * crop_l)
        top = int(h * crop_t)
        right = int(w - (w * crop_r))
        bottom = int(h - (h * crop_b))
        
        # Safety net: Ensure the box doesn't invert if you accidentally crop > 100%
        left = min(left, w - 1)
        top = min(top, h - 1)
        right = max(right, left + 1)
        bottom = max(bottom, top + 1)

        corner_img = base_card_img.crop((left, top, right, bottom))
        scale_multiplier = 1.0

    colored_card = corner_img.convert('RGBA')
    canvas = create_background(IMAGE_SIZE)
    
    scale_factor = random.uniform(*SCALE_RANGE) * scale_multiplier
    base_w, base_h = colored_card.size
    new_h = max(30, int(IMAGE_SIZE * scale_factor))
    new_w = max(20, int(new_h * (base_w / base_h)))
    
    card = colored_card.resize((new_w, new_h), Image.Resampling.LANCZOS)
    card = card.rotate(random.uniform(*ROTATION_RANGE), expand=True, resample=Image.Resampling.BICUBIC)

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

    if random.random() < GLARE_PROB:
        glare = Image.new("RGBA", card.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(glare)
        glare_width = random.randint(20, int(card.size[0] * 0.8))
        start_x = random.randint(-glare_width, card.size[0])
        points = [(start_x, 0), (start_x + glare_width, 0), (start_x + glare_width + 50, card.size[1]), (start_x + 50, card.size[1])]
        draw.polygon(points, fill=(255, 255, 255, random.randint(*GLARE_ALPHA_RANGE)))
        card = Image.alpha_composite(card, glare)

    # PLACEMENT LOGIC: Safely randomized fully inside the canvas
    max_x = max(0, IMAGE_SIZE - card.width)
    max_y = max(0, IMAGE_SIZE - card.height)
    paste_x = random.randint(0, max_x)
    paste_y = random.randint(0, max_y)
    
    canvas.paste(card, (paste_x, paste_y), card)
    
    enhancer = ImageEnhance.Brightness(canvas)
    canvas = enhancer.enhance(random.uniform(*BRIGHTNESS_RANGE))

    if random.random() < LETTERBOX_PROB:
        canvas = apply_simulated_letterboxing(canvas)
        
    img_arr = np.array(canvas, dtype=np.float32)
    noise = np.random.normal(loc=0, scale=random.uniform(*NOISE_SCALE_RANGE), size=img_arr.shape)
    canvas = Image.fromarray(np.clip(img_arr + noise, 0, 255).astype(np.uint8))

    return canvas

def main():
    print("Starting Corner-Crop Generation (Intuitive Margin Crop Mode)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_files = [f for f in os.listdir(BASE_CARDS_DIR) if f.endswith('.png')]

    for filename in tqdm(base_files, desc="Processing 56 Classes"):
        class_name = filename.replace('.png', '').lower()
        base_img = Image.open(os.path.join(BASE_CARDS_DIR, filename)).convert("RGBA")
        
        is_token = 'token' in class_name

        class_train_dir = os.path.join(OUTPUT_DIR, "train", class_name)
        class_val_dir = os.path.join(OUTPUT_DIR, "val", class_name)
        os.makedirs(class_train_dir, exist_ok=True)
        os.makedirs(class_val_dir, exist_ok=True)

        for i in range(IMAGES_PER_CLASS_TRAIN):
            img = generate_variation(base_img, is_token=is_token)
            img.save(os.path.join(class_train_dir, f"{class_name}_{random.randint(1000,99999)}.jpg"), quality=95)

        for i in range(IMAGES_PER_CLASS_VAL):
            img = generate_variation(base_img, is_token=is_token)
            img.save(os.path.join(class_val_dir, f"{class_name}_val_{random.randint(1000,99999)}.jpg"), quality=95)

if __name__ == '__main__':
    main()
