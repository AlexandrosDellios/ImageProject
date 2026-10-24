"""The synthetic pre-training set: two overlapping cards on a table background.

Each 336 x 336 image holds exactly two cards (drawn with replacement, so sometimes the
same one twice). The second is pasted 20-60 px to the right of the first and up to 30 px
higher or lower, so it partly covers it, as cards do in a real hand. The model is
pre-trained from scratch on 10,000 of these before it is fine-tuned on real zone crops.

    from uno_vision.synth import load_assets, compose_overlap, sample_rngs
    cards, backgrounds = load_assets()
    rng, np_rng = sample_rngs(0, 0, 0)
    image, names = compose_overlap(cards, backgrounds, rng, np_rng)

Port of the old generate_doubles.py with its parameter values. That script was edited
after the shipped dataset was generated, so these values may differ slightly from the ones
that produced the images the submitted model was pre-trained on. On 500 images from each,
the letterbox rate and bar width, the background mix and the brightness match.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance

from .common import CardAugment, Stages, add_noise, augment_card, letterbox, random_background, record_stage


@dataclass
class OverlapRecipe:
    image_size: int = 336
    n_cards: int = 2
    card_height: Tuple[float, float] = (0.5, 0.6)    # fraction of the image size
    start_margin: int = 30                           # min x / y of the first card, px
    end_margin: int = 80                             # room kept right of / below it, px
    shift_x: Tuple[int, int] = (20, 60)              # offset of each next card, px
    shift_y: Tuple[int, int] = (-30, 30)
    brightness: Tuple[float, float] = (0.94, 1.14)   # whole image
    letterbox_prob: float = 0.5
    noise_sigma: Tuple[float, float] = (7.0, 15.0)
    # The old generator sized every card with the first card's aspect ratio, so a round
    # yellow token drawn second is stretched to card shape, and a card drawn after it is
    # squashed square. False keeps that behaviour; True gives each card its own ratio.
    own_aspect: bool = False
    card: CardAugment = field(default_factory=CardAugment)


def compose_overlap(cards: Dict[str, Image.Image], backgrounds: List[Image.Image],
                    rng: random.Random, np_rng: np.random.Generator,
                    recipe: Optional[OverlapRecipe] = None, stages: Stages = None) -> Tuple[Image.Image, List[str]]:
    """One sample -> (RGB image, the class names it shows). Pass a list as `stages` to
    get a snapshot of every step: the background, each card's augmentations, each paste,
    and the final brightness, letterbox and noise."""
    recipe = recipe or OverlapRecipe()
    size = recipe.image_size
    canvas = random_background(backgrounds, size, rng)
    record_stage(stages, "background", canvas)

    names = rng.choices(sorted(cards), k=recipe.n_cards)
    card_h = int(size * rng.uniform(*recipe.card_height))
    first = cards[names[0]]
    card_w = int(card_h * first.width / first.height)

    max_x = max(recipe.start_margin, size - card_w - recipe.end_margin)
    max_y = max(recipe.start_margin, size - card_h - recipe.end_margin)
    start_x = rng.randint(recipe.start_margin, max_x)
    start_y = rng.randint(recipe.start_margin, max_y)

    for i, name in enumerate(names):
        cut_out = cards[name]
        width = int(card_h * cut_out.width / cut_out.height) if recipe.own_aspect else card_w
        card = augment_card(cut_out, width, card_h, rng, recipe.card, stages, label=f"card {i + 1} ({name})")
        x = start_x + i * rng.randint(*recipe.shift_x)
        y = start_y + i * rng.randint(*recipe.shift_y)
        canvas.paste(card, (x, y), card)
        record_stage(stages, f"paste card {i + 1}", canvas)

    canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(*recipe.brightness))
    record_stage(stages, "brightness", canvas)
    if rng.random() < recipe.letterbox_prob:
        canvas = letterbox(canvas, rng)
        record_stage(stages, "letterbox", canvas)
    canvas = add_noise(canvas, rng.uniform(*recipe.noise_sigma), np_rng)
    record_stage(stages, "noise", canvas)
    return canvas, names
