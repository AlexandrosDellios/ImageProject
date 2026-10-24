"""Archived, not maintained: draw the slicer.py zones on one photo (visualized_zones.jpg).

See experiments/README.md.
"""

import cv2
import os

# --- CONFIGURATION ---
# Pick ONE sample image from your raw data folder to test on
SAMPLE_IMAGE = "data/raw_images/L1000777.jpg"  # Change this to an actual filename!
OUTPUT_IMAGE = "visualized_zones.jpg"

# Your exact same zones from Step 1
ZONES = {
    "center": [0.30, 0.70, 0.30, 0.70],
    "p1":     [0.15, 0.85, 0.65, 1.00],
    "p2":     [0.00, 0.35, 0.15, 0.85],
    "p3":     [0.15, 0.85, 0.00, 0.35],
    "p4":     [0.65, 1.00, 0.15, 0.85]
}

# Colors for the boxes in BGR format (OpenCV uses Blue, Green, Red)
COLORS = {
    "center": (255, 255, 255), # White
    "p1":     (0, 0, 255),     # Red
    "p2":     (0, 255, 0),     # Green
    "p3":     (255, 0, 0),     # Blue
    "p4":     (0, 255, 255)    # Yellow
}

def main():
    if not os.path.exists(SAMPLE_IMAGE):
        print(f"Could not find {SAMPLE_IMAGE}. Please update the filename!")
        return

    # Load the image
    img = cv2.imread(SAMPLE_IMAGE)
    h, w = img.shape[:2]

    # Draw a box for each zone
    for zone_name, (x_min_pct, x_max_pct, y_min_pct, y_max_pct) in ZONES.items():
        # Convert percentages to exact pixels
        x1 = int(x_min_pct * w)
        x2 = int(x_max_pct * w)
        y1 = int(y_min_pct * h)
        y2 = int(y_max_pct * h)

        color = COLORS[zone_name]
        
        # Draw the rectangle (image, top-left, bottom-right, color, thickness)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        
        # Add a text label above the box
        cv2.putText(img, zone_name.upper(), (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # Save the output image
    cv2.imwrite(OUTPUT_IMAGE, img)
    print(f"Zones drawn successfully! Open '{OUTPUT_IMAGE}' to see your layout.")

if __name__ == "__main__":
    main()
