"""From a table photo to the game state: every hand, the center card, the active player.

    photo -> 5 zone crops -> pad, resize, normalise -> one batch of 5 -> (5, 56, 6) logits
          -> argmax = copies of each class per zone -> table state

Decoding rules (as submitted):
- a hand lists its cards in class order, one entry per copy, or "EMPTY";
- the active player is the player whose zone holds a token (the last one if several,
  p1 if none);
- the center card is the first card counted in the center zone; if none is counted, the
  class with the highest "exactly one copy" logit (never a token: then "r_0"), so the
  output always has exactly one center card.
"""

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .classes import CLASS_NAMES, EMPTY, decode_counts, is_token
from .transforms import eval_transform
from .zones import ZONE_NAMES, crop_zones

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
CSV_COLUMNS = ["image_id", "center_card", "active_player",
               "player_1_cards", "player_2_cards", "player_3_cards", "player_4_cards"]

_transform = eval_transform()


@torch.no_grad()
def zone_logits(model: torch.nn.Module, image: Image.Image, device: torch.device) -> torch.Tensor:
    """The (5, 56, 6) logits for one photo, zones in ZONE_NAMES order, on the CPU."""
    crops = crop_zones(np.array(image.convert("RGB")))
    batch = torch.stack([_transform(Image.fromarray(c)) for c in crops]).to(device)
    return model(batch).cpu()


def table_state(logits: torch.Tensor) -> Dict[str, str]:
    """(5, 56, 6) zone logits -> the CSV row (without image_id)."""
    counts = logits.argmax(dim=2)                    # (5, 56): copies of each class per zone
    hands: List[str] = []
    active = "p1"
    for i in range(4):
        cards, has_token = decode_counts(counts[i])
        if has_token:
            active = f"p{i + 1}"
        hands.append(";".join(cards) if cards else EMPTY)

    center_cards, _ = decode_counts(counts[ZONE_NAMES.index("center")])
    if center_cards:
        center = center_cards[0]
    else:
        center = CLASS_NAMES[int(logits[4, :, 1].argmax())]
        if is_token(center):
            center = "r_0"

    return {"center_card": center, "active_player": active,
            "player_1_cards": hands[0], "player_2_cards": hands[1],
            "player_3_cards": hands[2], "player_4_cards": hands[3]}


def predict_image(model: torch.nn.Module, path: Union[str, Path], device: torch.device) -> Dict[str, str]:
    path = Path(path)
    row = {"image_id": path.stem}
    row.update(table_state(zone_logits(model, Image.open(path), device)))
    return row


def list_images(folder: Union[str, Path]) -> List[Path]:
    """The photos in `folder`, sorted; hidden files (macOS "._" copies) are skipped."""
    return sorted(p for p in Path(folder).iterdir()
                  if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("."))


def predict_folder(model: torch.nn.Module, folder: Union[str, Path], device: torch.device,
                   progress: bool = True) -> pd.DataFrame:
    """One row per photo in `folder`, sorted by file name, in the submission's columns."""
    images = list_images(folder)
    if progress:
        from tqdm import tqdm
        images = tqdm(images, desc="Predicting")
    return pd.DataFrame([predict_image(model, p, device) for p in images], columns=CSV_COLUMNS)
