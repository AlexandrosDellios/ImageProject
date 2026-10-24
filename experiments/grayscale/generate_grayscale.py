"""Archived, not maintained: grayscale cards sorted into 17 shape classes ("step 0" data).

Colour is removed, so the classes are shapes only: 0-9, draw_2, reverse, skip, wild, draw_4,
and the tokens as circle_token / square_token. Writes uno_step0_grayscale/{train,val}/<shape>/
(250 + 50 images per source card, 336x336). Stage 0 of
experiments/multistage/train_multistage.py trains on it. See experiments/README.md.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

# ==========================================
# --- CONFIGURATION ---
# ==========================================
BASE_CARDS_DIR = "data/single_cards"               
OUTPUT_DIR = "uno_step0_grayscale"   
BACKGROUNDS_DIR = "data/backgrounds" 

IMAGES_PER_CLASS_TRAIN = 250                
IMAGES_PER_CLASS_VAL = 50                   
IMAGE_SIZE = 336  

ROTATION_RANGE = (0, 360)     
SCALE_RANGE = (0.3, 0.65)      
CROP_SEVERITY = 0.4           
OUT_OF_FRAME_PROB = 0.3       
MAX_CROP_PERCENT = 0.67       
LETTERBOX_PROB = 0.5         

# ==========================================
# ANTI-OVERFIT & VISION BREAKER TUNING
# ==========================================
# 1. Internal Softness & Feathering
CARD_BLUR_RANGE = (0.9, 1.6)  
EDGE_FEATHER_RANGE = (1.5, 2.5) 
NOISE_SCALE_RANGE = (7.0, 15.0) 

# 2. Global Brightness (Subtle lighting variations)
BRIGHTNESS_RANGE = (0.94, 1.34) # Adjusted to be much softer!

# 3. 3D Perspective Distortion
PERSPECTIVE_PROB = 0.8
PERSPECTIVE_DISTORTION = (0.25, 0.6) # How aggressively the card tilts away

# 4. The High-Frequency Killer (Crunching)
CRUNCH_PROB = 0.6
# How much to divide the resolution by. (e.g., 2.0 cuts resolution in half before blowing it back up)
CRUNCH_FACTOR_RANGE = (1.6, 2.3) 

# 5. Targeted Cutout (The Hole Punch)
HOLE_PUNCH_PROB = 0.6
HOLE_PUNCH_COUNT = (1, 3) # Min/Max number of holes to punch
# Radius of the hole as a percentage of the card's width (0.20 = 20% of the card is punched out)
HOLE_PUNCH_RADIUS_PCT = (0.15, 0.35) 

# 6. Specular Glare (White light reflections)
GLARE_PROB = 0.4
GLARE_ALPHA_RANGE = (50, 150) # Transparency of the glare (0 = invisible, 255 = solid white)
# ==========================================

def get_geometric_class(filename):
    name = filename.replace('.png', '').lower()
    if 'black_token' in name: return 'square_token'
    if 'yellow_token' in name: return 'circle_token'
    if 'draw_4' in name: return 'draw_4'
    if 'wild' in name and 'draw' not in name: return 'wild'
    
    parts = name.split('_')
    if len(parts) > 1: return '_'.join(parts[1:])
    return name

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

def generate_variation(base_card_img):
    # Convert to grayscale, then back to RGBA so we can paste onto colored backgrounds
    gray_card = base_card_img.convert('L').convert('RGBA')
    canvas = create_background(IMAGE_SIZE)
    
    scale_factor = random.uniform(*SCALE_RANGE)
    base_w, base_h = gray_card.size
    new_h = max(30, int(IMAGE_SIZE * scale_factor)) 
    new_w = max(20, int(new_h * (base_w / base_h)))
    
    card = gray_card.resize((new_w, new_h), Image.Resampling.LANCZOS)
    card = card.rotate(random.uniform(*ROTATION_RANGE), expand=True, resample=Image.Resampling.BICUBIC)

    # ANTI-OVERFIT: 3D Perspective Distortion
    if random.random() < PERSPECTIVE_PROB:
        w, h = card.size
        distortion = random.uniform(*PERSPECTIVE_DISTORTION) 
        if random.random() < 0.5:
            coeffs = (1, 0, 0, 0, 1, 0, random.uniform(-distortion, distortion)/w, 0)
        else:
            coeffs = (1, 0, 0, 0, 1, 0, 0, random.uniform(-distortion, distortion)/h)
        card = card.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
    
    # ANTI-OVERFIT: Tiny blur to the entire card
    card = card.filter(ImageFilter.GaussianBlur(radius=random.uniform(*CARD_BLUR_RANGE)))
    
    # ANTI-OVERFIT: Extra feathering on the edges
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

    # ANTI-OVERFIT: Specular Glare
    if random.random() < GLARE_PROB:
        glare = Image.new("RGBA", card.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(glare)
        glare_width = random.randint(20, int(card.size[0] * 0.8))
        start_x = random.randint(-glare_width, card.size[0])
        points = [(start_x, 0), (start_x + glare_width, 0), (start_x + glare_width + 50, card.size[1]), (start_x + 50, card.size[1])]
        draw.polygon(points, fill=(255, 255, 255, random.randint(*GLARE_ALPHA_RANGE)))
        card = Image.alpha_composite(card, glare)

    center_x = (IMAGE_SIZE - card.width) // 2
    center_y = (IMAGE_SIZE - card.height) // 2
    
    # STRICT OUT OF FRAME LOGIC
    if random.random() < OUT_OF_FRAME_PROB:
        max_hide_x = int(card.width * MAX_CROP_PERCENT)
        max_hide_y = int(card.height * MAX_CROP_PERCENT)
        
        edge = random.randint(0, 3)
        if edge == 0:   
            paste_x = random.randint(-max_hide_x - 10, -10)
            paste_y = center_y + random.randint(-50, 50)
        elif edge == 1: 
            paste_x = random.randint(IMAGE_SIZE - card.width + 10, IMAGE_SIZE - card.width + max_hide_x + 10)
            paste_y = center_y + random.randint(-50, 50)
        elif edge == 2: 
            paste_x = center_x + random.randint(-50, 50)
            paste_y = random.randint(-max_hide_y - 10, -10)
        else:           
            paste_x = center_x + random.randint(-50, 50)
            paste_y = random.randint(IMAGE_SIZE - card.height + 10, IMAGE_SIZE - card.height + max_hide_y + 10)
    else:
        max_offset_x = max(0, int((IMAGE_SIZE - card.width) * CROP_SEVERITY))
        max_offset_y = max(0, int((IMAGE_SIZE - card.height) * CROP_SEVERITY))
        paste_x = center_x + (random.randint(-max_offset_x, max_offset_x) if max_offset_x > 0 else 0)
        paste_y = center_y + (random.randint(-max_offset_y, max_offset_y) if max_offset_y > 0 else 0)
    
    canvas.paste(card, (paste_x, paste_y), card)
    
    # SIMULATED LETTERBOXING
    if random.random() < LETTERBOX_PROB:
        canvas = apply_simulated_letterboxing(canvas)
        
    # ANTI-OVERFIT: Global static noise
    img_arr = np.array(canvas, dtype=np.float32)
    noise = np.random.normal(loc=0, scale=random.uniform(*NOISE_SCALE_RANGE), size=img_arr.shape)
    canvas = Image.fromarray(np.clip(img_arr + noise, 0, 255).astype(np.uint8))

    return canvas

def main():
    print("Starting Step 0 Generation (Tunable Anti-Overfit & Vision Breakers)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_files = [f for f in os.listdir(BASE_CARDS_DIR) if f.endswith('.png')]

    for filename in tqdm(base_files, desc="Processing Classes"):
        geo_class = get_geometric_class(filename)
        base_img = Image.open(os.path.join(BASE_CARDS_DIR, filename)).convert("RGBA")

        class_train_dir = os.path.join(OUTPUT_DIR, "train", geo_class)
        class_val_dir = os.path.join(OUTPUT_DIR, "val", geo_class)
        os.makedirs(class_train_dir, exist_ok=True)
        os.makedirs(class_val_dir, exist_ok=True)

        for i in range(IMAGES_PER_CLASS_TRAIN):
            img = generate_variation(base_img)
            img.save(os.path.join(class_train_dir, f"{geo_class}_train_{random.randint(1000,99999)}.jpg"), quality=95)

        for i in range(IMAGES_PER_CLASS_VAL):
            img = generate_variation(base_img)
            img.save(os.path.join(class_val_dir, f"{geo_class}_val_{random.randint(1000,99999)}.jpg"), quality=95)

if __name__ == '__main__':
    main()
