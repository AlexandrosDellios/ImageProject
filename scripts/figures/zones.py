"""Draw zones.png: how one table photo becomes the five network inputs.

Top: test photo L1000797 with the five zones of uno_vision.zones.zone_layout (the corners
where two zones overlap are tinted red). Bottom: the five crops exactly as the network
sees them (crop_zones, then PadToSquare with black bars, then resize to 336), before
normalisation.

    python scripts/figures/zones.py --font path/to/GeistMono-500.ttf
    python scripts/figures/zones.py --out-dir some/folder --photo data/raw/test_images/L1000797.jpg
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image, ImageDraw

import style
from uno_vision.paths import FIGURES, RAW
from uno_vision.transforms import INPUT_SIZE, PadToSquare
from uno_vision.zones import ZONE_NAMES, crop_zones, zone_layout

WIDTH = 1800
MARGIN = 72
PHOTO_W = 1000            # displayed width of the photo
TILE_GAP = 44
LINE = 3                  # zone outline width
OVERLAP_ALPHA = 0.42

CHIP_FONT = 30
NOTE_FONT = 26
CHIP_PAD = (12, 8)

CROP_CHIPS = {"p1": "P1 (UPRIGHT)", "p2": "P2 (UPRIGHT)", "p3": "P3 (UPRIGHT)",
              "p4": "P4 (UPRIGHT)", "center": "CENTER"}
ROTATION_NOTES = {0: "NOT ROTATED", -1: "ROTATED 90° CW", 2: "ROTATED 180°", 1: "ROTATED 90° CCW"}
STEPS = "↓  CROP · ROTATE UPRIGHT · PAD TO SQUARE · RESIZE TO {0} × {0}".format(INPUT_SIZE)


def model_view(crop: np.ndarray, size: int = INPUT_SIZE) -> Image.Image:
    """A zone crop as the network sees it, before normalisation."""
    return PadToSquare(fill=0)(Image.fromarray(crop)).resize((size, size), Image.BILINEAR)


def overlaps(zones):
    """Pixel boxes where two zones intersect."""
    boxes = []
    for i in range(len(zones)):
        for j in range(i + 1, len(zones)):
            a, b = zones[i], zones[j]
            x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
            x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
            if x1 > x0 and y1 > y0:
                boxes.append((x0, y0, x1, y1))
    return boxes


def text_width(text, fnt):
    left, _, right, _ = fnt.getbbox(text)
    return right - left


def draw_text(draw, xy, text, fnt, fill):
    """Text whose ink box starts exactly at xy (top-left)."""
    left, top, _, _ = fnt.getbbox(text)
    draw.text((xy[0] - left, xy[1] - top), text, font=fnt, fill=fill)


def render(photo_path: Path) -> Image.Image:
    photo = Image.open(photo_path).convert("RGB")
    full = np.array(photo)
    height, width = full.shape[:2]
    zones = zone_layout(height, width)
    scale = PHOTO_W / width
    photo_h = round(height * scale)

    chip_font = style.font(CHIP_FONT)
    note_font = style.font(NOTE_FONT)
    _, chip_h = style.chip_size("P1", chip_font, CHIP_PAD)

    tile = (WIDTH - 2 * MARGIN - 4 * TILE_GAP) // 5
    row_w = 5 * tile + 4 * TILE_GAP
    row_x = (WIDTH - row_w) // 2

    # Vertical layout, top to bottom.
    y_p3_chip = 40
    y_photo = y_p3_chip + chip_h + 16
    y_p1_chip = y_photo + photo_h + 16
    y_steps = y_p1_chip + chip_h + 44
    steps_h = note_font.getbbox(STEPS)[3] - note_font.getbbox(STEPS)[1]
    y_tile_chips = y_steps + steps_h + 40
    y_tiles = y_tile_chips + chip_h + 14
    y_notes = y_tiles + tile + 16
    note_h = note_font.getbbox("ROTATED 90°")[3] - note_font.getbbox("ROTATED 90°")[1]
    total_h = y_notes + note_h + 48

    canvas = style.paper_canvas(WIDTH, total_h)
    x_photo = (WIDTH - PHOTO_W) // 2

    # Photo with the overlap corners tinted, then the zone outlines.
    small = photo.resize((PHOTO_W, photo_h), Image.LANCZOS).convert("RGBA")
    tint = Image.new("RGBA", small.size, (0, 0, 0, 0))
    tdraw = ImageDraw.Draw(tint)
    for x0, y0, x1, y1 in overlaps(zones):
        tdraw.rectangle([round(x0 * scale), round(y0 * scale), round(x1 * scale) - 1, round(y1 * scale) - 1],
                        fill=style.ACCENT + (round(255 * OVERLAP_ALPHA),))
    small = Image.alpha_composite(small, tint).convert("RGB")
    canvas.paste(small, (x_photo, y_photo))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([x_photo - 1, y_photo - 1, x_photo + PHOTO_W, y_photo + photo_h], outline=style.INK_2, width=1)
    boxes = {}
    for z in zones:
        # PIL draws an outline inside its box, so edges on the photo's border stay visible.
        x0 = x_photo + round(z.x0 * scale)
        y0 = y_photo + round(z.y0 * scale)
        x1 = x_photo + round(z.x1 * scale) - 1
        y1 = y_photo + round(z.y1 * scale) - 1
        boxes[z.name] = (x0, y0, x1, y1)
        draw.rectangle([x0, y0, x1, y1], outline=style.INK, width=LINE)

    # Hand chips sit outside the photo on their player's side; the center chip sits inside.
    def chip_w(text):
        return style.chip_size(text, chip_font, CHIP_PAD)[0]

    x0, y0, x1, y1 = boxes["p1"]
    style.draw_chip(draw, ((x0 + x1 - chip_w("P1")) // 2, y_p1_chip), "P1", chip_font, pad=CHIP_PAD)
    x0, y0, x1, y1 = boxes["p3"]
    style.draw_chip(draw, ((x0 + x1 - chip_w("P3")) // 2, y_p3_chip), "P3", chip_font, pad=CHIP_PAD)
    x0, y0, x1, y1 = boxes["p2"]
    style.draw_chip(draw, (x_photo + PHOTO_W + 18, (y0 + y1 - chip_h) // 2), "P2", chip_font, pad=CHIP_PAD)
    x0, y0, x1, y1 = boxes["p4"]
    style.draw_chip(draw, (x_photo - 18 - chip_w("P4") - 3, (y0 + y1 - chip_h) // 2), "P4", chip_font,
                    pad=CHIP_PAD)
    x0, y0, x1, y1 = boxes["center"]
    style.draw_chip(draw, ((x0 + x1 - chip_w("CENTER")) // 2, y1 - chip_h - 16), "CENTER", chip_font,
                    pad=CHIP_PAD)

    # Legend for the red corners, under the photo's left edge (P1's chip is further right).
    swatch = note_h + 6
    legend = "SEEN BY TWO ZONES"
    lx = x_photo
    ly = y_p1_chip + (chip_h - swatch) // 2
    sw = Image.new("RGB", (swatch, swatch), style.PAPER)
    sw = Image.blend(sw, Image.new("RGB", (swatch, swatch), style.ACCENT), OVERLAP_ALPHA + 0.2)
    canvas.paste(sw, (lx, ly))
    draw.rectangle([lx, ly, lx + swatch - 1, ly + swatch - 1], outline=style.INK, width=2)
    draw_text(draw, (lx + swatch + 12, ly + (swatch - note_h) // 2), legend, note_font, style.INK)

    # The processing steps between the photo and the crops.
    draw_text(draw, ((WIDTH - text_width(STEPS, note_font)) // 2, y_steps), STEPS, note_font, style.INK_2)

    # The five network inputs.
    crops = crop_zones(full)
    for k, (name, crop, z) in enumerate(zip(ZONE_NAMES, crops, zones)):
        x = row_x + k * (tile + TILE_GAP)
        view = model_view(crop).resize((tile, tile), Image.LANCZOS)
        canvas.paste(view, (x, y_tiles))
        draw.rectangle([x - 1, y_tiles - 1, x + tile, y_tiles + tile], outline=style.INK, width=2)
        label = CROP_CHIPS[name]
        style.draw_chip(draw, (x, y_tile_chips), label, chip_font, pad=CHIP_PAD)
        draw_text(draw, (x, y_notes), ROTATION_NOTES[z.rot90_k], note_font, style.INK_2)
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photo", type=Path, default=RAW / "test_images" / "L1000797.jpg")
    parser.add_argument("--out-dir", type=Path, default=FIGURES)
    parser.add_argument("--font", default=None, help="a .ttf for the labels (default: DejaVu Sans Mono)")
    args = parser.parse_args(argv)
    if not args.photo.exists():
        sys.exit(f"error: {args.photo} not found")
    style.set_font(args.font)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "zones.png"
    render(args.photo).save(out, optimize=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
