"""Archived, not maintained: split the real-crop label CSVs by hand size.

Reads phase4_{train,val}_strict.csv and writes curriculum_{train,val}_stage{1-4}.csv:
crops with at most 2, 3 and 4 entries, then all of them. See experiments/README.md.
"""

import pandas as pd
import os

# --- CONFIGURATION ---
TRAIN_CSV = "phase4_train_strict.csv"
VAL_CSV = "phase4_val_strict.csv"
OUTPUT_PREFIX = "curriculum"

def count_cards(label_str):
    """Counts the number of cards in a label string. Handles EMPTY."""
    if pd.isna(label_str) or str(label_str).strip().upper() == "EMPTY":
        return 0
    return len(str(label_str).split(';'))

def build_curriculum(csv_path, is_val=False):
    split_name = "VAL" if is_val else "TRAIN"
    print(f"\nAnalyzing {split_name} Dataset: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Create a new column with the card count
    df['card_count'] = df['cards'].apply(count_cards)
    
    total_images = len(df)
    print(f"Total Images: {total_images}")

    # --- DEFINE THE STAGES (Cumulative) ---
    # Stage 1: Extremely easy (0, 1, or 2 cards)
    stage1_df = df[df['card_count'] <= 2].drop(columns=['card_count'])
    
    # Stage 2: Easy to Medium (0, 1, 2, or 3 cards)
    stage2_df = df[df['card_count'] <= 3].drop(columns=['card_count'])
    
    # Stage 3: Medium to Hard (0, 1, 2, 3, or 4 cards)
    stage3_df = df[df['card_count'] <= 4].drop(columns=['card_count'])
    
    # Stage 4: Everything (The full original dataset)
    stage4_df = df.drop(columns=['card_count'])

    # --- SAVE THE CSVS ---
    prefix = f"{OUTPUT_PREFIX}_val" if is_val else f"{OUTPUT_PREFIX}_train"
    
    stage1_df.to_csv(f"{prefix}_stage1.csv", index=False)
    stage2_df.to_csv(f"{prefix}_stage2.csv", index=False)
    stage3_df.to_csv(f"{prefix}_stage3.csv", index=False)
    stage4_df.to_csv(f"{prefix}_stage4.csv", index=False)

    # --- THE DIAGNOSTIC REPORT ---
    print("\nCURRICULUM DISTRIBUTION:")
    print(f"Stage 1 (<= 2 cards): {len(stage1_df)} images ({len(stage1_df)/total_images*100:.1f}%)")
    print(f"Stage 2 (<= 3 cards): {len(stage2_df)} images ({len(stage2_df)/total_images*100:.1f}%)")
    print(f"Stage 3 (<= 4 cards): {len(stage3_df)} images ({len(stage3_df)/total_images*100:.1f}%)")
    print(f"Stage 4 (All hands) : {len(stage4_df)} images (100.0%)")
    
    # The Overfit Warning
    if not is_val and len(stage1_df) < 200:
        print("\nWARNING: Your Stage 1 dataset is dangerously small!")
        print("Training on this will cause immediate overfitting.")
        print("Recommendation: Skip Stage 1, start at Stage 2, and use RandomErasing augmentations.")

def main():
    build_curriculum(TRAIN_CSV, is_val=False)
    build_curriculum(VAL_CSV, is_val=True)
    print("\nCurriculum CSVs generated successfully!")

if __name__ == '__main__':
    main()
