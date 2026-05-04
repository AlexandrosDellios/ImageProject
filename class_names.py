import os

# Point this to your Phase 1 training folder
phase1_dir = "uno_classification_dataset/train"

# Get the folder names, sort them alphabetically
class_names = sorted([d for d in os.listdir(phase1_dir) if os.path.isdir(os.path.join(phase1_dir, d))])

# Print it formatted exactly how the Phase 4 script needs it
print("CLASS_NAMES = [")
for name in class_names:
    print(f"    '{name}',")
print("]")