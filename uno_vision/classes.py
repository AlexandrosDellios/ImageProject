"""The 56 output classes and the count encoding.

The model predicts, for each class, how many copies (0 to 5) a zone holds. The order of
CLASS_NAMES is the order of the model's outputs: do not change it, or the weights no
longer match.
"""

from typing import List, Sequence, Tuple

import torch

CLASS_NAMES: List[str] = [
    "b_0", "b_1", "b_2", "b_3", "b_4", "b_5", "b_6", "b_7", "b_8", "b_9",
    "b_draw_2", "b_reverse", "b_skip", "black_token", "draw_4",
    "g_0", "g_1", "g_2", "g_3", "g_4", "g_5", "g_6", "g_7", "g_8", "g_9",
    "g_draw_2", "g_reverse", "g_skip",
    "r_0", "r_1", "r_2", "r_3", "r_4", "r_5", "r_6", "r_7", "r_8", "r_9",
    "r_draw_2", "r_reverse", "r_skip", "wild",
    "y_0", "y_1", "y_2", "y_3", "y_4", "y_5", "y_6", "y_7", "y_8", "y_9",
    "y_draw_2", "y_reverse", "y_skip", "yellow_token",
]

NUM_CLASSES = len(CLASS_NAMES)   # 54 card faces + the 2 turn tokens
MAX_COPIES = 6                   # count bins: 0, 1, 2, 3, 4, 5 copies
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
EMPTY = "EMPTY"                  # how the CSVs write an empty hand


def is_token(name: str) -> bool:
    """The black and yellow tokens mark whose turn it is; they are not cards."""
    return "token" in name


def encode_cards(cards: str) -> torch.Tensor:
    """"r_1;r_1;b_skip" -> a length-56 tensor of counts (clamped to MAX_COPIES - 1).
    "EMPTY" gives all zeros; unknown names are ignored."""
    target = torch.zeros(NUM_CLASSES, dtype=torch.long)
    if cards.strip().upper() == EMPTY:
        return target
    for card in cards.split(";"):
        idx = CLASS_TO_IDX.get(card.strip())
        if idx is not None:
            target[idx] = min(target[idx] + 1, MAX_COPIES - 1)
    return target


def decode_counts(counts: Sequence) -> Tuple[List[str], bool]:
    """Per-class counts -> (cards, has_token). Cards are repeated once per copy, in
    class order; tokens are reported as a flag instead of a card."""
    cards: List[str] = []
    has_token = False
    for idx, count in enumerate(counts):
        count = int(count)
        if count > 0:
            name = CLASS_NAMES[idx]
            if is_token(name):
                has_token = True
            else:
                cards.extend([name] * count)
    return cards, has_token
