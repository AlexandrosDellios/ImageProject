"""Synthetic training data: card cut-outs composited on table backgrounds.

common.py holds the shared pieces (asset loading, background patches, the card
augmentation chain, letterbox and noise); overlap.py is the two-card recipe used for
pre-training. scripts/generate_synthetic.py writes the dataset to disk.
"""

from .common import CardAugment, augment_card, load_assets, load_backgrounds, load_cards, sample_rngs
from .overlap import OverlapRecipe, compose_overlap

__all__ = [
    "CardAugment", "augment_card", "load_assets", "load_backgrounds", "load_cards", "sample_rngs",
    "OverlapRecipe", "compose_overlap",
]
