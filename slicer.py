import cv2
import os
from glob import glob

INPUT_DIR = "data/raw_images"
OUTPUT_DIR = "data/to_be_labeled"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ZONES = {
    "center": [0.30, 0.70, 0.30, 0.70],
    "p1":     [0.15, 0.85, 0.65, 1.00],
    "p2":     [0.00, 0.35, 0.15, 0.85],
    "p3":     [0.15, 0.85, 0.00, 0.35],
    "p4":     [0.65, 1.00, 0.15, 0.85]
}

images = glob(os.path.join(INPUT_DIR, "*.jpg"))
for img_path in images:
    img = cv2.imread(img_path)
    if img is None: continue
    h, w = img.shape[:2]
    base_name = os.path.basename(img_path).split('.')[0]
    
    for zone, (x1p, x2p, y1p, y2p) in ZONES.items():
        crop = img[int(y1p*h):int(y2p*h), int(x1p*w):int(x2p*w)]
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_{zone}.jpg"), crop)
print(f"✅ Sliced {len(images)} images into {len(images)*5} crops ready for labeling!")