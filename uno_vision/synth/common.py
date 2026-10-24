"""Building blocks for synthetic training images: card cut-outs pasted on table backgrounds.

The inputs are RGBA cut-outs of every class (assets/cards/<class>.png) and patches of the
two table surfaces (assets/backgrounds/). Before it is pasted, each card goes through
`augment_card`, a chain meant to make a clean cut-out look like a card in a real zone
crop once that crop is squeezed into the 336 px network input:

    colour jitter -> resize -> rotate -> perspective -> blur -> feathered edges
    -> crunch (downscale then upscale) -> hole punch -> glare

All randomness comes from the `rng` (random.Random) and `np_rng` (numpy Generator)
arguments, never from global state, so a seed reproduces a sample exactly.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFilter

from ..paths import ASSETS

IMAGE_EXTS = (".png", ".jpg", ".jpeg")

# Optional (name, image) snapshots of each step, for figures and GIFs of the composition.
Stages = Optional[List[Tuple[str, Image.Image]]]


@dataclass
class CardAugment:
    """Parameters of `augment_card`. The defaults are the overlap generator's values."""

    rotation: Tuple[float, float] = (-360.0, 360.0)      # degrees
    jitter_brightness: float = 0.3                       # ColorJitter-style strengths
    jitter_contrast: float = 0.3
    jitter_saturation: float = 0.3
    jitter_hue: float = 0.05
    perspective_prob: float = 0.8
    perspective_strength: Tuple[float, float] = (0.45, 0.5)
    blur: Tuple[float, float] = (0.9, 1.3)               # Gaussian radius, px
    edge_feather: Tuple[float, float] = (1.5, 2.5)       # Gaussian radius on the alpha, px
    crunch_prob: float = 0.6
    crunch_factor: Tuple[float, float] = (1.6, 1.8)
    hole_punch_prob: float = 0.0                         # off in the final recipe
    hole_punch_count: Tuple[int, int] = (1, 3)
    hole_punch_radius: Tuple[float, float] = (0.15, 0.35)   # fraction of the card width
    glare_prob: float = 0.0                              # off in the final recipe
    glare_alpha: Tuple[int, int] = (50, 150)


def sample_rngs(*key: int) -> Tuple[random.Random, np.random.Generator]:
    """Independent generators for one sample, derived from integers such as
    (seed, split, index): sample i is the same whatever the dataset size."""
    py_seq, np_seq = np.random.SeedSequence(list(key)).spawn(2)
    return random.Random(int(py_seq.generate_state(1, dtype=np.uint64)[0])), np.random.default_rng(np_seq)


def _list_images(folder: Path) -> List[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_cards(folder: Union[str, Path]) -> Dict[str, Image.Image]:
    """{class name: RGBA cut-out}, the name being the lower-cased file stem."""
    cards = {}
    for path in _list_images(folder):
        with Image.open(path) as im:
            cards[path.stem.lower()] = im.convert("RGBA")
    return cards


def load_backgrounds(folder: Union[str, Path]) -> List[Image.Image]:
    backgrounds = []
    for path in _list_images(folder):
        with Image.open(path) as im:
            backgrounds.append(im.convert("RGB"))
    return backgrounds


def load_assets(assets: Union[str, Path] = ASSETS) -> Tuple[Dict[str, Image.Image], List[Image.Image]]:
    """(cards, backgrounds) from <assets>/cards and <assets>/backgrounds."""
    return load_cards(Path(assets) / "cards"), load_backgrounds(Path(assets) / "backgrounds")


def random_background(backgrounds: List[Image.Image], size: int, rng: random.Random) -> Image.Image:
    """A random size x size patch of a random background (plain white if there are none)."""
    if not backgrounds:
        return Image.new("RGB", (size, size), (255, 255, 255))
    bg = rng.choice(backgrounds)
    w, h = bg.size
    if w < size or h < size:
        scale = max(size / w, size / h)
        bg = bg.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.Resampling.BICUBIC)
        w, h = bg.size
    x = rng.randint(0, w - size)
    y = rng.randint(0, h - size)
    return bg.crop((x, y, x + size, y + size))


def jitter_colors(card: Image.Image, rng: random.Random, aug: CardAugment) -> Image.Image:
    """torchvision's ColorJitter on the RGB channels (alpha untouched): same factor ranges
    and random order of the four adjustments, drawn from `rng` instead of torch's RNG."""
    rgb, alpha = card.convert("RGB"), card.getchannel("A")
    ops = []
    if aug.jitter_brightness:
        ops.append((TF.adjust_brightness, rng.uniform(1 - aug.jitter_brightness, 1 + aug.jitter_brightness)))
    if aug.jitter_contrast:
        ops.append((TF.adjust_contrast, rng.uniform(1 - aug.jitter_contrast, 1 + aug.jitter_contrast)))
    if aug.jitter_saturation:
        ops.append((TF.adjust_saturation, rng.uniform(1 - aug.jitter_saturation, 1 + aug.jitter_saturation)))
    if aug.jitter_hue:
        ops.append((TF.adjust_hue, rng.uniform(-aug.jitter_hue, aug.jitter_hue)))
    rng.shuffle(ops)
    for op, factor in ops:
        rgb = op(rgb, factor)
    rgb.putalpha(alpha)
    return rgb


def record_stage(stages: Stages, name: str, image: Image.Image) -> None:
    if stages is not None:
        stages.append((name, image.copy()))


def augment_card(card: Image.Image, width: int, height: int, rng: random.Random,
                 aug: Optional[CardAugment] = None, stages: Stages = None, label: str = "card") -> Image.Image:
    """An RGBA cut-out -> the RGBA card to paste, resized to width x height before rotation
    (rotation expands the canvas, so the result is larger)."""
    aug = aug or CardAugment()
    card = jitter_colors(card, rng, aug)
    record_stage(stages, f"{label}: colour jitter", card)

    card = card.resize((width, height), Image.Resampling.LANCZOS)
    record_stage(stages, f"{label}: resize", card)

    card = card.rotate(rng.uniform(*aug.rotation), expand=True, resample=Image.Resampling.BICUBIC)
    record_stage(stages, f"{label}: rotate", card)

    if rng.random() < aug.perspective_prob:
        # Tilt along x or y only: a camera that is not exactly overhead.
        w, h = card.size
        strength = rng.uniform(*aug.perspective_strength)
        if rng.random() < 0.5:
            coeffs = (1, 0, 0, 0, 1, 0, rng.uniform(-strength, strength) / w, 0)
        else:
            coeffs = (1, 0, 0, 0, 1, 0, 0, rng.uniform(-strength, strength) / h)
        card = card.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
        record_stage(stages, f"{label}: perspective", card)

    card = card.filter(ImageFilter.GaussianBlur(radius=rng.uniform(*aug.blur)))
    # Soft alpha edges, so the paste leaves no sharp cut-out outline for the network to learn.
    alpha = card.getchannel("A").filter(ImageFilter.GaussianBlur(radius=rng.uniform(*aug.edge_feather)))
    card.putalpha(alpha)
    record_stage(stages, f"{label}: blur and feather", card)

    if rng.random() < aug.crunch_prob:
        # Real zone crops are 700-2200 px wide and get resized to 336: mimic that loss of detail.
        factor = rng.uniform(*aug.crunch_factor)
        small = card.resize((max(10, int(card.width / factor)), max(10, int(card.height / factor))),
                            Image.Resampling.NEAREST)
        card = small.resize(card.size, Image.Resampling.BICUBIC)
        record_stage(stages, f"{label}: crunch", card)

    if rng.random() < aug.hole_punch_prob:
        draw = ImageDraw.Draw(card)
        for _ in range(rng.randint(*aug.hole_punch_count)):
            radius = rng.randint(int(card.width * aug.hole_punch_radius[0]), int(card.width * aug.hole_punch_radius[1]))
            cx = card.width // 2 + rng.randint(-20, 20)
            cy = card.height // 2 + rng.randint(-20, 20)
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(0, 0, 0, 0))
        record_stage(stages, f"{label}: hole punch", card)

    if rng.random() < aug.glare_prob:
        glare = Image.new("RGBA", card.size, (255, 255, 255, 0))
        band = rng.randint(20, int(card.width * 0.8))
        x0 = rng.randint(-band, card.width)
        points = [(x0, 0), (x0 + band, 0), (x0 + band + 50, card.height), (x0 + 50, card.height)]
        ImageDraw.Draw(glare).polygon(points, fill=(255, 255, 255, rng.randint(*aug.glare_alpha)))
        card = Image.alpha_composite(card, glare)
        record_stage(stages, f"{label}: glare", card)

    return card


def letterbox(image: Image.Image, rng: random.Random) -> Image.Image:
    """Black bars on two opposite sides, 20-80 px each: what PadToSquare adds to real crops."""
    w, h = image.size
    draw = ImageDraw.Draw(image)
    if rng.random() < 0.5:
        bar = rng.randint(20, 80)
        draw.rectangle([0, 0, w, bar], fill=(0, 0, 0))
        draw.rectangle([0, h - bar, w, h], fill=(0, 0, 0))
    else:
        bar = rng.randint(20, 80)
        draw.rectangle([0, 0, bar, h], fill=(0, 0, 0))
        draw.rectangle([w - bar, 0, w, h], fill=(0, 0, 0))
    return image


def add_noise(image: Image.Image, sigma: float, np_rng: np.random.Generator) -> Image.Image:
    """Gaussian sensor noise of standard deviation `sigma` (in 0-255 units)."""
    arr = np.asarray(image, dtype=np.float32)
    noisy = arr + np_rng.normal(0.0, sigma, size=arr.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
