"""Draw gradcam.png: where the network looks, on three held-out real crops.

Top row: each crop as the network sees it (padded to a square with black bars). Bottom
row: the layer4 Grad-CAM of one class the network counted correctly, blended over the
crop with the magma colormap. The crops are a fixed list (PICKS), chosen by looking at
every held-out crop with two or more cards; the third one is a case where the heat does
not sit on the object it explains, and it is labelled as such.

    python scripts/figures/gradcam_figure.py --font path/to/GeistMono-500.ttf
    python scripts/figures/gradcam_figure.py --out-dir some/folder --device cpu
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw

import style
from uno_vision.classes import CLASS_TO_IDX, encode_cards, is_token
from uno_vision.gradcam import GradCAM, colorize, display_crop, overlay
from uno_vision.model import load_model
from uno_vision.paths import FIGURES, PROCESSED, WEIGHTS
from uno_vision.transforms import eval_transform
from uno_vision.utils import pick_device

# (crop, class explained, note under the heatmap, heat off target?, box to point at in the
# crop's own pixels or None). In L1000784_p2 the token is a dark sliver cut by the crop's
# right edge (x 1584-1595, y 687-933 of 1596 x 1200); a red bracket marks it.
PICKS = [
    ("L1000853_p2.png", "g_skip", ["HEAT ON THE SKIP CARD", "UNDER THE +2"], False, None),
    ("L1000907_p4.png", "b_8", ["HEAT ON THE BLUE 8,", "NOT THE GREEN 8"], False, None),
    ("L1000784_p2.png", "yellow_token", ["HEAT ON THE CARDS, NOT THE", "TOKEN AT THE RIGHT EDGE"], True,
     (1584, 687, 1596, 934)),
]
CMAP = "magma"
ALPHA = 0.5

WIDTH = 1500
MARGIN = 60
GAP = 48
CHIP_FONT = 28
LABEL_FONT = 24
CHIP_PAD = (12, 8)


def text_height(fnt, text="AQ"):
    _, top, _, bottom = fnt.getbbox(text)
    return bottom - top


def draw_text(draw, xy, text, fnt, fill):
    """Text whose ink box starts exactly at xy (top-left)."""
    left, top, _, _ = fnt.getbbox(text)
    draw.text((xy[0] - left, xy[1] - top), text, font=fnt, fill=fill)


def describe(filename: str, cards: str) -> str:
    """"L1000784_p2.png", "y_0;b_7;wild;yellow_token" -> "P2 HAND · 3 CARDS + TOKEN"."""
    zone = filename.rsplit("_", 1)[1].split(".")[0].upper()
    names = [c for c in cards.split(";") if c]
    n_cards = sum(not is_token(c) for c in names)
    text = f"{zone} HAND · {n_cards} CARDS"
    return text + " + TOKEN" if any(is_token(c) for c in names) else text


def compute(model, device, crops: Path, labels: Path, panel: int):
    """For each pick: (top-row label, chip text, note, off-target flag, marked rows or None,
    crop image, overlay image); marked rows are (top, bottom) in panel pixels."""
    frame = pd.read_csv(labels).set_index("filename")
    transform = eval_transform()
    rows = []
    with GradCAM(model, "layer4") as cam:
        for filename, cls, note, off_target, mark in PICKS:
            path = crops / filename
            cards = str(frame.loc[filename, "cards"])
            target = encode_cards(cards)
            idx = CLASS_TO_IDX[cls]
            count = int(target[idx])
            original = Image.open(path).convert("RGB")
            image = transform(original).to(device)
            with torch.no_grad():
                predicted = int(model(image.unsqueeze(0))[0, idx].argmax())
            n_cards = sum(not is_token(c) for c in cards.split(";"))
            # The figure claims the network got these right: check it, do not assume it.
            if count == 0 or predicted != count or n_cards < 2:
                sys.exit(f"error: {filename} {cls}: true {count}, predicted {predicted}, {n_cards} cards")
            heat = cam(image, idx, count)
            crop = display_crop(path, panel)
            blended = overlay(crop, colorize(heat, panel, CMAP), ALPHA)
            marked = None
            if mark:
                # Same geometry as PadToSquare: centred in a square of the longer side.
                w, h = original.size
                side = max(w, h)
                top = (side - h) // 2
                marked = (round((top + mark[1]) * panel / side), round((top + mark[3]) * panel / side))
            rows.append((describe(filename, cards), style.card_label(cls), note, off_target, marked, crop,
                         Image.fromarray(np.uint8(np.round(blended * 255)))))
            peak = tuple(int(i) for i in np.unravel_index(int(np.argmax(heat)), heat.shape))
            print(f"{filename}: {cls} true {count}, predicted {predicted}; heat peak at cell {peak} "
                  f"of {heat.shape[0]} x {heat.shape[1]}")
    return rows


def render(rows, panel: int) -> Image.Image:
    chip_font = style.font(CHIP_FONT)
    label_font = style.font(LABEL_FONT)
    _, chip_h = style.chip_size("A", chip_font, CHIP_PAD)
    label_h = text_height(label_font)
    line_h = round(label_h * 1.6)

    y_label = 44
    y_input = y_label + label_h + 16
    y_chip = y_input + panel + 40
    y_overlay = y_chip + chip_h + 14
    y_note = y_overlay + panel + 18
    y_scale = y_note + 2 * line_h + 40
    scale_h = 16
    total_h = y_scale + max(scale_h, label_h) + 48

    canvas = style.paper_canvas(WIDTH, total_h)
    draw = ImageDraw.Draw(canvas)
    for k, (label, chip, note, off_target, marked, crop, blended) in enumerate(rows):
        x = MARGIN + k * (panel + GAP)
        draw_text(draw, (x, y_label), label, label_font, style.INK_2)
        for image, y in ((crop, y_input), (blended, y_overlay)):
            canvas.paste(image, (x, y))
            draw.rectangle([x - 1, y - 1, x + panel, y + panel], outline=style.INK, width=2)
            if marked:
                # A bracket just outside the panel's right edge, level with the marked object.
                bx, top, bottom = x + panel + 12, y + marked[0], y + marked[1]
                draw.line([bx, top, bx, bottom], fill=style.ACCENT, width=4)
                draw.line([x + panel + 4, top, bx + 1, top], fill=style.ACCENT, width=4)
                draw.line([x + panel + 4, bottom, bx + 1, bottom], fill=style.ACCENT, width=4)
        style.draw_chip(draw, (x, y_chip), chip, chip_font, pad=CHIP_PAD)
        color = style.ACCENT_INK if off_target else style.INK_2
        for i, line in enumerate(note):
            draw_text(draw, (x, y_note + i * line_h), line, label_font, color)

    # Colour scale, centred under the grid.
    name, low, high = "LAYER4 GRAD-CAM", "LOW", "HIGH"
    bar_w = 260
    widths = [label_font.getbbox(t)[2] - label_font.getbbox(t)[0] for t in (name, low, high)]
    total_w = widths[0] + 36 + widths[1] + 14 + bar_w + 14 + widths[2]
    x = (WIDTH - total_w) // 2
    y_text = y_scale + (scale_h - label_h) // 2
    draw_text(draw, (x, y_text), name, label_font, style.INK_2)
    x += widths[0] + 36
    draw_text(draw, (x, y_text), low, label_font, style.INK_2)
    x += widths[1] + 14
    ramp = colormaps[CMAP](np.linspace(0, 1, bar_w))[:, :3]
    bar = Image.fromarray(np.uint8(np.round(np.tile(ramp[None], (scale_h, 1, 1)) * 255)))
    canvas.paste(bar, (x, y_scale))
    draw.rectangle([x - 1, y_scale - 1, x + bar_w, y_scale + scale_h], outline=style.INK, width=1)
    x += bar_w + 14
    draw_text(draw, (x, y_text), high, label_font, style.INK_2)
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--crops", type=Path, default=PROCESSED / "real_val", help="folder of zone crops")
    parser.add_argument("--labels", type=Path, default=PROCESSED / "real_val.csv", help="filename,cards CSV")
    # CPU by default: three crops are fast, and the heatmaps then match on every machine.
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--out-dir", type=Path, default=FIGURES)
    parser.add_argument("--font", default=None, help="a .ttf for the labels (default: DejaVu Sans Mono)")
    args = parser.parse_args(argv)
    for path in (args.weights, args.crops, args.labels):
        if not path.exists():
            sys.exit(f"error: {path} not found")
    style.set_font(args.font)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    model = load_model(args.weights, device)
    panel = (WIDTH - 2 * MARGIN - 2 * GAP) // 3
    rows = compute(model, device, args.crops, args.labels, panel)
    out = args.out_dir / "gradcam.png"
    render(rows, panel).save(out, optimize=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
