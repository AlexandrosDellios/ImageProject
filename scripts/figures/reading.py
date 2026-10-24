"""Animation: the model reading one test photo, zone by zone.

The five zones draw in, then each zone's predicted cards appear next to it, the player
holding the token is marked, and a bottom line sums up the table. Cards the model missed
are shown as "MISSED" chips, compared with the cards visible in the photo (read by eye:
the test set has no public labels), listed in VISIBLE below.

    python scripts/figures/reading.py --font GeistMono.ttf
    python scripts/figures/reading.py --site-dir out/   # also an MP4, a WebM and stills

Writes docs/figures/reading.gif (README) and, with --site-dir, reading.mp4, reading.webm,
reading-first.jpg (first frame) and reading-last.png (last frame).
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image, ImageDraw

import style
from uno_vision.classes import decode_counts
from uno_vision.model import load_model
from uno_vision.paths import FIGURES, RAW, WEIGHTS
from uno_vision.predict import table_state, zone_logits
from uno_vision.utils import pick_device
from uno_vision.zones import ZONE_NAMES, zone_layout

# Cards visible in the photo, per zone (read by eye). Used only to mark missed cards.
VISIBLE = {
    "L1000947": {"p1": ["r_8", "draw_4"], "p2": [], "p3": ["b_5", "y_draw_2"], "p4": ["b_3"],
                 "center": ["b_0"]},
}

W, PHOTO_H, BAND_H = 1200, 800, 100
FPS = 24


class Layer:
    """An RGBA overlay that fades in at `start` (seconds) over `fade` seconds."""

    def __init__(self, start: float, fade: float = 0.18):
        self.image = Image.new("RGBA", (W, PHOTO_H + BAND_H), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)
        self.start, self.fade = start, fade
        self.end = None   # optional fade-out start

    def alpha(self, t: float) -> float:
        a = min(1.0, max(0.0, (t - self.start) / self.fade))
        if self.end is not None:
            a *= 1.0 - min(1.0, max(0.0, (t - self.end) / self.fade))
        return a


def zone_reading(logits):
    """Per zone: (predicted cards, has_token)."""
    counts = logits.argmax(dim=2)
    return {name: decode_counts(counts[i]) for i, name in enumerate(ZONE_NAMES)}


def missed_cards(predicted, visible):
    left = Counter(visible) - Counter(predicted)
    return sorted(left.elements())


def build_layers(photo_id: str, reading, state, scale: float):
    big, small = style.font(23), style.font(19)
    layers = []
    boxes = {z.name: [round(v * scale) for v in (z.x0, z.y0, z.x1, z.y1)] for z in zone_layout(*PHOTO_SIZE)}
    order = ["p1", "p2", "p3", "p4", "center"]
    labels = {"p1": "P1", "p2": "P2", "p3": "P3", "p4": "P4", "center": "CENTER"}
    # Where each zone's chip column starts (inside the zone, clear of the cards).
    anchor = {name: (b[0] + 10, b[1] + 10) for name, b in boxes.items()}
    anchor["p1"] = (boxes["p1"][0] + 10, boxes["p1"][1] + 12)
    anchor["center"] = (boxes["center"][0] + 8, boxes["center"][1] + 8)
    visible = VISIBLE.get(photo_id)

    # 1. Zones draw in, one after the other.
    t = 0.6
    for name in order:
        box = Layer(t, 0.25)
        x0, y0, x1, y1 = boxes[name]
        rect = [x0 + 3, y0 + 3, min(x1, W - 1) - 3, min(y1, PHOTO_H - 1) - 3]
        box.draw.rectangle([rect[0] - 2, rect[1] - 2, rect[2] + 2, rect[3] + 2], outline=style.PAPER + (255,), width=8)  # halo: the ink line stays visible on dark cloth
        box.draw.rectangle(rect, outline=style.INK + (255,), width=4)
        style.draw_chip(box.draw, anchor[name], labels[name], big, fg=style.PAPER, bg=style.INK, border=style.INK, shadow=0)
        layers.append(box)
        t += 0.45

    # 2. Each zone is read: a red flash, then its cards.
    t += 0.4
    for name in order:
        flash = Layer(t, 0.12)
        flash.end = t + 0.45
        x0, y0, x1, y1 = boxes[name]
        flash.draw.rectangle([x0 + 2, y0 + 2, min(x1, W - 1) - 2, min(y1, PHOTO_H - 1) - 2], outline=style.ACCENT + (255,), width=7)
        layers.append(flash)

        cards, has_token = reading[name]
        x, y = anchor[name]
        y += 48
        chips = [(style.card_label(c), "card") for c in cards]
        if not cards and name != "center":
            chips.append(("EMPTY", "card"))
        if visible is not None:
            chips += [("MISSED: " + style.card_label(c), "missed") for c in missed_cards(cards, visible[name])]
        if name == "center":
            x, y = boxes[name][0] + 8, boxes[name][3] - 46   # below the center card
        for k, (text, kind) in enumerate(chips):
            chip = Layer(t + 0.25 + 0.18 * k)
            if kind == "missed":
                w, h = style.draw_chip(chip.draw, (x, y), text, small, fg=style.ACCENT_INK, border=style.ACCENT)
            else:
                w, h = style.draw_chip(chip.draw, (x, y), text, small)
            layers.append(chip)
            y += h + 9
        if has_token:
            turn = Layer(t + 0.25 + 0.18 * len(chips))
            style.draw_chip(turn.draw, (x, y), "TURN", small, fg=style.PAPER, bg=style.ACCENT, border=style.INK)
            layers.append(turn)
        t += 0.75

    # 3. The bottom line (until then, a quiet note of what is happening).
    t += 0.2
    note = Layer(0.4, 0.3)
    note.end = t - 0.3
    note.draw.text((18, PHOTO_H + 38), "READING THE TABLE, ZONE BY ZONE", font=small, fill=style.INK_2 + (255,))
    layers.append(note)
    band = Layer(t, 0.3)
    n_cards = sum(len(v) for v in visible.values()) if visible else None
    n_read = sum(len(reading[z][0]) for z in ZONE_NAMES) if visible else None
    parts = [f"CENTER: {style.card_label(state['center_card'])}", f"TURN: {state['active_player'].upper()}"]
    if visible:
        parts.append(f"{n_read} OF {n_cards} CARDS READ")
    x = 18
    for text in parts:
        w, _ = style.draw_chip(band.draw, (x, PHOTO_H + 30), text, big)
        x += w + 16
    note = "UNSEEN TEST PHOTO"
    nw = band.draw.textlength(note, font=small)
    band.draw.text((W - 18 - nw, PHOTO_H + 41), note, font=small, fill=style.INK_2 + (255,))
    layers.append(band)
    return layers, t + 0.3


PHOTO_SIZE = (0, 0)


def render(photo_id: str, device_name, font_path):
    global PHOTO_SIZE
    style.set_font(font_path)
    device = pick_device(device_name)
    model = load_model(WEIGHTS, device)
    photo = Image.open(RAW / "test_images" / f"{photo_id}.jpg").convert("RGB")
    PHOTO_SIZE = (photo.height, photo.width)
    logits = zone_logits(model, photo, device)
    reading, state = zone_reading(logits), table_state(logits)
    scale = W / photo.width

    base = style.paper_canvas(W, PHOTO_H + BAND_H)
    base.paste(photo.resize((W, PHOTO_H), Image.LANCZOS), (0, 0))
    ImageDraw.Draw(base).line([0, PHOTO_H, W, PHOTO_H], fill=style.INK, width=3)
    base = np.asarray(base, dtype=np.float32)

    layers, built = build_layers(photo_id, reading, state, scale)
    hold, fade_out = 2.6, 0.5
    total = built + hold + fade_out
    for layer in layers:   # everything fades out together so the loop restarts cleanly
        layer.end = built + hold if layer.end is None else layer.end
        if layer.end == built + hold:
            layer.fade = min(layer.fade, fade_out)
    arrays = [np.asarray(l.image, dtype=np.float32) for l in layers]

    frames = []
    for i in range(round(total * FPS)):
        t = i / FPS
        frame = base.copy()
        for layer, arr in zip(layers, arrays):
            a = layer.alpha(t)
            if a > 0:
                m = arr[..., 3:4] / 255.0 * a
                frame = frame * (1 - m) + arr[..., :3] * m
        frames.append(Image.fromarray(frame.clip(0, 255).astype(np.uint8)))
    final = frames[round((built + hold / 2) * FPS)]   # mid-hold: everything shown, nothing fading
    return frames, final, state, reading


def encode(frames, out_dir: Path, site_dir, gif_width: int):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("error: ffmpeg is needed to encode the animation")
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            f.save(f"{tmp}/f_{i:04d}.png")
        src = ["-framerate", str(FPS), "-i", f"{tmp}/f_%04d.png"]
        run = lambda *a: subprocess.run([ffmpeg, "-y", "-loglevel", "error", *a], check=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        palette = f"{tmp}/palette.png"
        scale = f"fps=12,scale={gif_width}:-1:flags=lanczos"
        run(*src, "-vf", f"{scale},palettegen=max_colors=160:stats_mode=diff", palette)
        run(*src, "-i", palette, "-lavfi", f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
            str(out_dir / "reading.gif"))
        if site_dir:
            site_dir.mkdir(parents=True, exist_ok=True)
            # 4:2:0 YUV tagged BT.709 throughout (PNG frames carry no colour tags of their own).
            to_yuv = ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
                      "setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709"]
            run(*src, *to_yuv, "-c:v", "libx264", "-crf", "22", "-preset", "slow",
                "-movflags", "+faststart", "-an", str(site_dir / "reading.mp4"))
            run(*src, *to_yuv, "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "38", "-row-mt", "1", "-an",
                str(site_dir / "reading.webm"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photo", default="L1000947", help="test photo id (in data/raw/test_images)")
    parser.add_argument("--out-dir", type=Path, default=FIGURES)
    parser.add_argument("--site-dir", type=Path, default=None, help="also write MP4, WebM and stills here")
    parser.add_argument("--gif-width", type=int, default=720)
    parser.add_argument("--font", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    # Check the inputs before the minute of rendering.
    photo = RAW / "test_images" / f"{args.photo}.jpg"
    if not photo.exists():
        sys.exit(f"error: {photo} not found")
    if not shutil.which("ffmpeg"):
        sys.exit("error: ffmpeg is needed to encode the animation")

    frames, last, state, reading = render(args.photo, args.device, args.font)
    encode(frames, args.out_dir, args.site_dir, args.gif_width)
    if args.site_dir:
        frames[0].save(args.site_dir / "reading-first.jpg", quality=86)
        last.save(args.site_dir / "reading-last.png")
    print(f"{len(frames)} frames ({len(frames) / FPS:.1f} s); prediction: {state}")


if __name__ == "__main__":
    main()
