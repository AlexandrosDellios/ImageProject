import os
import numpy as np
from PIL import Image
from pathlib import Path



def divide_pic(img):
    """Découpe une image en 5 zones : 4 joueurs + centre, ET les redresse !"""
    H, W = img.shape[:2]
    h, w = H // 20, W // 20
    
    # Player 1 (Bottom): Already facing the camera upright
    p1 = img[15*h:, 5*w:16*w]
    
    # Player 2 (Right): Player faces left. Rotate 90° Clockwise
    p2 = img[5*h:17*h, 14*w:]
    p2 = np.rot90(p2, k=-1)
    
    # Player 3 (Top): Player faces down. Rotate 180°
    p3 = img[:6*h, 5*w:15*w]
    p3 = np.rot90(p3, k=2)
    
    # Player 4 (Left): Player faces right. Rotate 90° Counter-Clockwise
    p4 = img[5*h:18*h, :6*w]
    p4 = np.rot90(p4, k=1)
    
    # Center Card: Usually upright relative to the camera
    center = img[8*h:14*h, 8*w:12*w]
    
    return {
        'p1': p1,
        'p2': p2,
        'p3': p3,
        'p4': p4,
        'center': center
    }


def divide_folder(input_dir, output_dir):
    """
    Pour chaque image dans input_dir, génère 5 crops dans output_dir.
    Ex: L24362.jpg -> L24362_p1.png, L24362_p2.png, ..., L24362_center.png
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    valid_ext = ('.png', '.jpg', '.jpeg')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]
    
    print(f"→ {len(files)} images à traiter")
    
    for fname in files:
        # Charger l'image
        img_path = os.path.join(input_dir, fname)
        img = np.array(Image.open(img_path).convert('RGB'))
        
        # Nom de base sans extension (ex: 'L24362' depuis 'L24362.jpg')
        base_name = os.path.splitext(fname)[0]
        
        # Découper en 5 zones
        crops = divide_pic(img)
        
        # Sauvegarder chaque crop
        for zone_name, crop in crops.items():
            out_path = os.path.join(output_dir, f"{base_name}_{zone_name}.png")
            Image.fromarray(crop).save(out_path)
    
    print(f" {len(files) * 5} images créées dans {output_dir}")


# Usage
divide_folder('data/train_images', 'data/train_divided')

#divide_pic()

""" img_uno = np.array(Image.open('./data/train_images/L1000782.jpg'))
crops=divide_pic(img_uno)

for i, s in enumerate(crops):
        if i==4 :
            Image.fromarray(s).save(f"center.png")
        else :
            Image.fromarray(s).save(f"player_{i+1}.png") """