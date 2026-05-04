import pandas as pd
from ultralytics import YOLO

# Load your FULLY TRAINED model (after Phase 2)
model = YOLO('uno_project/phase2_grammar/weights/best.pt')

# This is the dictionary YOLO memorized from your YAML
# e.g., {0: 'b_skip', 1: 'r_7', ... 54: 'token'}
class_names = model.names 

results_data = []

# Loop through all the hidden Kaggle Test images
for test_image_path in kaggle_test_images:
    image_id = get_id(test_image_path) # e.g., 'L1000777'
    
    # 1. Slice the test image into your 5 zones
    zones = slice_image_into_5_zones(test_image_path)
    
    # 2. Ask YOLO what it sees in Player 1's zone
    p1_results = model.predict(zones['p1'])
    
    # 3. Translate YOLO's math into TA's strings
    p1_cards = []
    for box in p1_results[0].boxes:
        class_id = int(box.cls[0])
        card_string = class_names[class_id] # translates 12 to 'r_7'
        p1_cards.append(card_string)
        
    # 4. Format exactly to Kaggle's rules
    if len(p1_cards) == 0:
        p1_csv_string = "EMPTY"
    else:
        p1_csv_string = ";".join(p1_cards) # makes 'r_7;b_4'
        
    # ... Repeat for p2, p3, p4, and center ...
    # ... Use the location of the 'token' class to determine active_player ...
    
    # 5. Save the row
    results_data.append([
        image_id, center_card, active_player, 
        p1_csv_string, p2_csv_string, p3_csv_string, p4_csv_string
    ])

# 6. Save directly to Kaggle's required format
df = pd.DataFrame(results_data, columns=['image_id', 'center_card', 'active_player', 'player_1_cards', 'player_2_cards', 'player_3_cards', 'player_4_cards'])
df.to_csv('submission.csv', index=False)


# Inside your final CSV generation script
active_player = "UNKNOWN"

for zone_name, crop in zones.items():
    results = model.predict(crop)
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        card_name = class_names[cls_id]
        
        # Merge the visual classes back into your game logic
        if card_name == 'token_yellow' or card_name == 'token_black':
            active_player = zone_name