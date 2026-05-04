import os
import cv2
import pandas as pd
import albumentations as A
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_DIR = "data/train_divided"
INPUT_CSV = "real_hands_labels.csv"
OUTPUT_DIR = "data/train_augmented"
OUTPUT_CSV = "augmented_hands_labels.csv"
AUGMENT_MULTIPLIER = 4  # 1 Original + 4 Augmented = 5 variations per image

# --- THE ALBUMENTATIONS PIPELINE ---
# 🚨 IMPORTANT: We do NOT use Horizontal/Vertical Flips! 
# Flipping an Uno card turns a '7' or a '3' into backwards gibberish.
transform = A.Compose([
    # Spatial: Simulate the camera shifting, tilting, or the player moving their hand
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=45, border_mode=cv2.BORDER_REPLICATE, p=0.8),
    A.Perspective(scale=(0.01, 0.03), p=0.4),
    
    # Pixel: Simulate different room lighting, shadows, and bad camera focus
    A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05, p=0.7),
    A.GaussNoise(var_limit=(2.0, 20.0), p=0.3),
    A.MotionBlur(blur_limit=2, p=0.2),
    A.ImageCompression(quality_lower=20, quality_upper=50, p=0.2), # JPEG artifacts
    
    # Occlusion: Simulate glare or an extra thumb covering part of the hand
    A.CoarseDropout(
        num_holes_range=(1, 3), 
        hole_height_range=(15, 35), 
        hole_width_range=(15, 35), 
        fill_mask=0, 
        p=0.3
    )
])

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"🔄 Reading original labels from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    new_rows = []
    
    pbar = tqdm(df.iterrows(), total=len(df), desc="Augmenting Dataset")
    
    for index, row in pbar:
        img_name = str(row['filename'])
        cards = str(row['cards'])
        
        img_path = os.path.join(INPUT_DIR, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"\n⚠️ Skipping {img_name}: File not found or unreadable.")
            continue
            
        # 1. Save the ORIGINAL image to the new folder
        cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), image)
        new_rows.append({'filename': img_name, 'cards': cards})
        
        # 2. Generate and save the AUGMENTED variations
        base_name, ext = os.path.splitext(img_name)
        for i in range(AUGMENT_MULTIPLIER):
            augmented = transform(image=image)['image']
            aug_filename = f"{base_name}_aug{i}{ext}"
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, aug_filename), augmented)
            new_rows.append({'filename': aug_filename, 'cards': cards})
            
    # Save the expanded CSV
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Augmentation Complete!")
    print(f"📈 Multiplied {len(df)} images into {len(new_df)} images.")
    print(f"💾 Saved to {OUTPUT_DIR} and {OUTPUT_CSV}")

if __name__ == "__main__":
    main()