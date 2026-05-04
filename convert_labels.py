import pandas as pd

def reformat_labels(input_csv, output_csv):
    print(f"🔄 Reading original labels from {input_csv}...")
    df = pd.read_csv(input_csv)
    new_rows = []

    for index, row in df.iterrows():
        img_id = str(row['image_id']).strip()

        # 1. Process the Center Card
        new_rows.append({
            'filename': f"{img_id}_center.png",
            'cards': str(row['center_card']).strip()
        })

        # 2. Process the 4 Player Hands
        players = {
            'p1': row['player_1_cards'],
            'p2': row['player_2_cards'],
            'p3': row['player_3_cards'],
            'p4': row['player_4_cards']
        }

        active_p = str(row['active_player']).strip().lower()

        for p_key, p_cards in players.items():
            cards_str = str(p_cards).strip()
            
            # 🚨 TOKEN INJECTION LOGIC 🚨
            # If this player is the active player, we must add the token to their label.
            if p_key == active_p:
                # Assuming 'yellow_token' is your default. 
                # If your game alternates token colors (e.g., p1/p3 yellow, p2/p4 black), 
                # you can add a simple if/else statement here.
                token_class = "yellow_token" 
                
                if cards_str == "EMPTY":
                    cards_str = token_class
                else:
                    cards_str = f"{cards_str};{token_class}"

            new_rows.append({
                'filename': f"{img_id}_{p_key}.png",
                'cards': cards_str
            })

    # Convert back to a DataFrame and save
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(output_csv, index=False)
    
    print(f"✅ Successfully converted {len(df)} table labels into {len(new_df)} individual crop labels!")
    print(f"💾 Saved strictly formatted data to: {output_csv}")

if __name__ == "__main__":
    reformat_labels('train.csv', 'real_hands_labels.csv')