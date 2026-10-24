"""Figure: where the model still fails. Two fanned hands from test photos, each with the
cards the model read and the cards it missed (compared by eye with the photo: the test set
has no public labels). Cards mostly hidden under others are the typical miss.

    python scripts/figures/failures.py --font GeistMono.ttf
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image, ImageDraw

import style
from uno_vision.classes import decode_counts
from uno_vision.model import load_model
from uno_vision.paths import FIGURES, RAW, WEIGHTS
from uno_vision.predict import zone_logits
from uno_vision.utils import pick_device
from uno_vision.zones import ZONE_NAMES, crop_zones

# (photo, zone, region of the upright zone crop to show (x0, y0, x1, y1), what was missed)
CASES = [
    ("L1000885", "p1", (640, 0, 1718, 667), "MISSED: BLUE 7, HALF HIDDEN"),
    ("L1001000", "p4", (150, 420, 1250, 1180), "MISSED: 2 CARDS BEHIND IT"),
]
PANEL_W, IMG_H, GAP, MARGIN = 760, 470, 40, 30


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=FIGURES)
    parser.add_argument("--font", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    style.set_font(args.font)

    device = pick_device(args.device)
    model = load_model(WEIGHTS, device)
    chip_font = style.font(22)
    canvas = style.paper_canvas(MARGIN * 2 + PANEL_W * 2 + GAP, MARGIN * 2 + IMG_H + 112)
    draw = ImageDraw.Draw(canvas)

    for k, (photo_id, zone, region, missed) in enumerate(CASES):
        photo = Image.open(RAW / "test_images" / f"{photo_id}.jpg").convert("RGB")
        z = ZONE_NAMES.index(zone)
        cards, has_token = decode_counts(zone_logits(model, photo, device).argmax(dim=2)[z])
        crop = Image.fromarray(crop_zones(np.array(photo))[z]).crop(region)
        crop = crop.resize((PANEL_W, round(crop.height * PANEL_W / crop.width)), Image.LANCZOS)
        crop = crop.crop((0, (crop.height - IMG_H) // 2, PANEL_W, (crop.height - IMG_H) // 2 + IMG_H))

        x, y = MARGIN + k * (PANEL_W + GAP), MARGIN
        canvas.paste(crop, (x, y))
        draw.rectangle([x, y, x + PANEL_W - 1, y + IMG_H - 1], outline=style.INK, width=3)

        cy, cx = y + IMG_H + 20, x
        read = [style.card_label(c) for c in cards] + (["TURN"] if has_token else [])
        w, h = style.draw_chip(draw, (cx, cy), "READ: " + " · ".join(read), chip_font)
        style.draw_chip(draw, (cx, cy + h + 14), missed, chip_font, fg=style.ACCENT_INK, border=style.ACCENT)
        print(photo_id, zone, "read:", cards, "token" if has_token else "")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out_dir / "failures.png", optimize=True)
    print(f"-> {args.out_dir / 'failures.png'} {canvas.size}")


if __name__ == "__main__":
    main()
