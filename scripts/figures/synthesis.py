"""Animation of how one synthetic pre-training image is built, then a glimpse of many.

Left, the 336 x 336 sample at every step of uno_vision.synth.compose_overlap: the table
patch, the first card (cut-out, colour jitter, rotation and tilt, blur and soft edges),
its paste, the second card, then brightness, letterbox and noise. Right, the list of
steps with the current one highlighted. It ends on the finished sample and its label,
then a grid of other samples from the same generator, and fades back to the start.

    python scripts/figures/synthesis.py                                # docs/figures/synthesis.gif
    python scripts/figures/synthesis.py --formats mp4 webm poster --out-dir site/synthesis
    python scripts/figures/synthesis.py --font GeistMono-500.ttf --frames-dir /tmp/frames

The sample is training image --index of scripts/generate_synthetic.py with --seed (the
defaults draw a yellow +2 under a red 8 on the leaf tablecloth). Frames are rendered at
1280 x 720, 24 fps; ffmpeg must be on the PATH to encode them.
"""

import argparse
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import style
from uno_vision.paths import FIGURES
from uno_vision.synth import CardAugment, OverlapRecipe, compose_overlap, load_assets, sample_rngs

W, H, FPS = 1280, 720, 24
SIZE = 336                         # the generated image, as the network sees it
VIEW_X, VIEW_Y, VIEW = 80, 80, 560
LIST_X = 716                       # left edge of the step numbers
ROW_PITCH = 52
GHOST = 0.16                       # how much table shows behind a card being prepared
TRAIN_SPLIT = 0                    # split id of "train" in scripts/generate_synthetic.py
PRETRAIN_COUNT = "10,000"          # size of the synthetic pre-training set

STEPS = (
    "TABLE TEXTURE", "CARD CUT-OUT", "COLOUR JITTER", "ROTATE + PERSPECTIVE",
    "BLUR + EDGE FEATHER", "PASTE", "SECOND CARD, OVERLAPPING", "BRIGHTNESS",
    "LETTERBOX", "SENSOR NOISE",
)
(S_TABLE, S_CUT_OUT, S_JITTER, S_ROTATE, S_BLUR, S_PASTE, S_SECOND,
 S_BRIGHTNESS, S_LETTERBOX, S_NOISE) = range(len(STEPS))

FAINT = tuple(int(a * 0.55 + b * 0.45) for a, b in zip(style.INK_2, style.PAPER))   # steps to come
BICUBIC = Image.Resampling.BICUBIC


def ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def same(a: Image.Image, b: Image.Image) -> bool:
    return a.size == b.size and np.array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------- the sample, step by step

class _Recorder(random.Random):
    """A copy of the sample's random.Random that logs every uniform() draw. The values and
    their order are unchanged, so the sample is identical; the log gives the rotation and
    tilt of each card, which the stage snapshots do not store."""

    def __init__(self, state):
        super().__init__()
        self.setstate(state)
        self.draws: List[Tuple[float, float, float]] = []

    def uniform(self, a, b):
        value = super().uniform(a, b)
        self.draws.append((a, b, value))
        return value


@dataclass
class Card:
    name: str
    cut_out: Image.Image                  # the asset at paste size, before any change
    jittered: Image.Image                 # after colour jitter and resize
    angle: Optional[float]                # None if the logged angle did not reproduce the snapshot
    rotated: Image.Image
    tilt: Optional[Tuple[float, ...]]     # perspective coefficients
    tilted: Optional[Image.Image]
    soft: Image.Image                     # after blur and feathered edges
    crunched: Optional[Image.Image]
    pos: Tuple[int, int]                  # where it is pasted (top-left of the final RGBA)

    @property
    def final(self) -> Image.Image:
        return self.crunched if self.crunched is not None else self.soft

    def at_centre(self, image: Image.Image) -> Tuple[int, int]:
        """Top-left that puts `image` on the same centre as the pasted card."""
        cx = self.pos[0] + self.final.width / 2
        cy = self.pos[1] + self.final.height / 2
        return round(cx - image.width / 2), round(cy - image.height / 2)


@dataclass
class Sample:
    names: List[str]
    stages: Dict[str, Image.Image]
    cards: List[Card]
    brightness: float
    noise_sigma: float
    bars: Tuple[str, int]                 # ("top/bottom" or "left/right", width in px), or ("", 0)


def find_tilt(rotated: Image.Image, tilted: Image.Image, draws, aug: CardAugment) -> Optional[Tuple[float, ...]]:
    w, h = rotated.size
    for i, (a, b, strength) in enumerate(draws):
        if (a, b) != tuple(aug.perspective_strength):
            continue
        for a2, b2, value in draws[i + 1:]:
            if (a2, b2) == (-strength, strength):
                for coeffs in ((1, 0, 0, 0, 1, 0, value / w, 0), (1, 0, 0, 0, 1, 0, 0, value / h)):
                    if same(rotated.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, BICUBIC), tilted):
                        return coeffs
                return None
    return None


def find_paste(before: Image.Image, after: Image.Image, card: Image.Image) -> Tuple[int, int]:
    """The (x, y) at which `card` was pasted to turn `before` into `after`."""
    ys, xs = np.nonzero(np.any(np.asarray(before) != np.asarray(after), axis=2))
    ays, axs = np.nonzero(np.asarray(card.getchannel("A")) > 0)
    x0, y0 = int(xs.min() - axs.min()), int(ys.min() - ays.min())
    offsets = sorted(((dx, dy) for dx in range(-8, 9) for dy in range(-8, 9)), key=lambda d: abs(d[0]) + abs(d[1]))
    for dx, dy in offsets:
        test = before.copy()
        test.paste(card, (x0 + dx, y0 + dy), card)
        if same(test, after):
            return x0 + dx, y0 + dy
    raise RuntimeError("could not locate the pasted card")


def measure_bars(image: Image.Image) -> Tuple[str, int]:
    black = np.all(np.asarray(image) == 0, axis=2)
    rows, cols = black.all(axis=1), black.all(axis=0)
    if rows[0] and rows[-1]:
        return "top/bottom", int(np.argmin(rows))
    if cols[0] and cols[-1]:
        return "left/right", int(np.argmin(cols))
    return "", 0


def build_sample(cards, backgrounds, seed: int, index: int) -> Sample:
    rng, np_rng = sample_rngs(seed, TRAIN_SPLIT, index)
    recorder = _Recorder(rng.getstate())
    recipe = OverlapRecipe()
    stages: List[Tuple[str, Image.Image]] = []
    _, names = compose_overlap(cards, backgrounds, recorder, np_rng, recipe, stages)
    named = dict(stages)
    draws = recorder.draws
    aug = recipe.card

    rotations = [i for i, (a, b, _) in enumerate(draws) if (a, b) == tuple(aug.rotation)]
    built = []
    for k, name in enumerate(names):
        prefix = f"card {k + 1} ({name}): "
        jittered, rotated = named[prefix + "resize"], named[prefix + "rotate"]
        angle: Optional[float] = draws[rotations[k]][2]
        if not same(jittered.rotate(angle, expand=True, resample=BICUBIC), rotated):
            angle = None
        segment = draws[rotations[k] + 1:rotations[k + 1] if k + 1 < len(rotations) else None]
        tilted = named.get(prefix + "perspective")
        tilt = find_tilt(rotated, tilted, segment, aug) if tilted is not None else None
        soft, crunched = named[prefix + "blur and feather"], named.get(prefix + "crunch")
        before = named["background"] if k == 0 else named[f"paste card {k}"]
        final = crunched if crunched is not None else soft
        pos = find_paste(before, named[f"paste card {k + 1}"], final)
        cut_out = cards[name].resize(jittered.size, Image.Resampling.LANCZOS)
        built.append(Card(name, cut_out, jittered, angle, rotated, tilt, tilted, soft, crunched, pos))

    def drawn(lo_hi):
        return next(v for a, b, v in draws if (a, b) == tuple(lo_hi))

    bars = measure_bars(named["letterbox"]) if "letterbox" in named else ("", 0)
    return Sample(names, named, built, drawn(recipe.brightness), drawn(recipe.noise_sigma), bars)


# ---------------------------------------------------------------- drawing

class Painter:
    def __init__(self):
        self.step_font = style.font(28)
        self.chip_font = style.font(26)
        self.big_font = style.font(32)
        self.paper_sq = Image.new("RGB", (SIZE, SIZE), style.PAPER)

    # the 336 x 336 picture
    def compose(self, base: Image.Image, ghost: float = 1.0, card: Optional[Image.Image] = None,
                pos: Tuple[int, int] = (0, 0)) -> Image.Image:
        """`base` faded towards paper (ghost < 1), with an RGBA card on top at pos."""
        view = base if ghost >= 1 else Image.blend(self.paper_sq, base, ghost)
        if card is None:
            return view
        layer = Image.new("RGBA", view.size, (0, 0, 0, 0))
        layer.paste(card, pos)
        return Image.alpha_composite(view.convert("RGBA"), layer).convert("RGB")

    # pieces of a frame
    @staticmethod
    def framed(canvas: Image.Image, image: Image.Image, x: int, y: int, size: int, shadow: int) -> None:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([x + shadow, y + shadow, x + size - 1 + shadow, y + size - 1 + shadow], fill=style.INK)
        canvas.paste(image.resize((size, size), Image.Resampling.LANCZOS), (x, y))
        draw.rectangle([x - 1, y - 1, x + size, y + size], outline=style.INK, width=2)

    @staticmethod
    def chip(draw: ImageDraw.ImageDraw, left: int, top: int, text: str, fnt: ImageFont.FreeTypeFont,
             fg, bg, shadow_color, shadow: int = 4, pad: Tuple[int, int] = (12, 10)) -> Tuple[int, int, int, int]:
        """style's chip, sized on the cap height so chips of one font share a height and a
        baseline whatever their letters (a comma would otherwise make one taller)."""
        cap = -fnt.getbbox("H", anchor="ls")[1]
        x1 = left + round(fnt.getlength(text)) + 2 * pad[0]
        y1 = top + cap + 2 * pad[1]
        if shadow:
            draw.rectangle([left + shadow, top + shadow, x1 + shadow, y1 + shadow], fill=shadow_color)
        draw.rectangle([left, top, x1, y1], fill=bg, outline=style.INK, width=2)
        draw.text((left + pad[0], top + pad[1] + cap), text, font=fnt, fill=fg, anchor="ls")
        return left, top, x1, y1

    def chip_height(self, fnt, pad_y: int = 10) -> int:
        return -fnt.getbbox("H", anchor="ls")[1] + 2 * pad_y

    def steps(self, canvas: Image.Image, current: Optional[int]) -> None:
        """The step list; `current` is highlighted, earlier steps are done, later ones faint.
        current=None: all done."""
        draw = ImageDraw.Draw(canvas)
        fnt = self.step_font
        chip_h = self.chip_height(fnt)
        cap = -fnt.getbbox("H", anchor="ls")[1]
        top = VIEW_Y + (VIEW - ((len(STEPS) - 1) * ROW_PITCH + chip_h)) // 2
        label_x = LIST_X + round(fnt.getlength("00")) + 26
        for i, label in enumerate(STEPS):
            base = top + i * ROW_PITCH + 10 + cap
            done = current is None or i < current
            color = style.INK if i == current else (style.INK_2 if done else FAINT)
            draw.text((LIST_X, base), f"{i + 1:02d}", font=fnt, fill=color, anchor="ls")
            if i == current:
                self.chip(draw, label_x - 12, base - cap - 10, label, fnt, style.PAPER, style.INK, style.ACCENT)
            else:
                draw.text((label_x, base), label, font=fnt, fill=color, anchor="ls")

    def labels(self, canvas: Image.Image, names: Sequence[str]) -> None:
        """The sample's label as chips straddling the bottom edge of the view."""
        draw = ImageDraw.Draw(canvas)
        x = VIEW_X + 20
        top = VIEW_Y + VIEW - self.chip_height(self.chip_font) // 2
        for name in names:
            box = self.chip(draw, x, top, style.card_label(name), self.chip_font, style.INK, style.PAPER, style.INK)
            x = box[2] + 16

    def frame(self, view: Image.Image, current: Optional[int], names: Optional[Sequence[str]] = None) -> Image.Image:
        canvas = style.paper_canvas(W, H)
        self.framed(canvas, view, VIEW_X, VIEW_Y, VIEW, 6)
        self.steps(canvas, current)
        if names:
            self.labels(canvas, names)
        return canvas


# ---------------------------------------------------------------- the reel

class Reel:
    """Writes numbered PNG frames; remembers the last one for cross-fades."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.count = 0
        self.last: Optional[Image.Image] = None
        self._last_path: Optional[Path] = None

    def add(self, image: Image.Image, n: int = 1) -> None:
        for _ in range(n):
            path = self.folder / f"{self.count:04d}.png"
            if image is self.last and self._last_path is not None:
                shutil.copyfile(self._last_path, path)
            else:
                image.save(path, compress_level=1)
            self.last, self._last_path = image, path
            self.count += 1

    def fade_to(self, image: Image.Image, n: int) -> None:
        start = self.last
        for k in range(1, n):
            self.add(Image.blend(start, image, ease(k / n)))
        self.add(image)


def card_motion(reel: Reel, painter: Painter, card: Card, base: Image.Image, step: int,
                rotate: int, tilt: int, hold: int) -> None:
    """Rotation, then perspective, played as motion from the upright jittered card."""
    def show(image, n=1):
        reel.add(painter.frame(painter.compose(base, GHOST, image, card.at_centre(image)), step), n)

    if card.angle is not None:
        target = (card.angle + 180) % 360 - 180          # the short way round
        for k in range(1, rotate):
            show(card.jittered.rotate(target * ease(k / rotate), expand=True, resample=BICUBIC))
        show(card.rotated, hold)
    else:
        reel.fade_to(painter.frame(painter.compose(base, GHOST, card.rotated, card.at_centre(card.rotated)), step), rotate)
        reel.add(reel.last, hold)
    if card.tilted is None:
        return
    if card.tilt is not None:
        w, h = card.rotated.size
        for k in range(1, tilt):
            e = ease(k / tilt)
            coeffs = card.tilt[:6] + (card.tilt[6] * e, card.tilt[7] * e)
            show(card.rotated.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, BICUBIC))
        show(card.tilted, hold)
    else:
        reel.fade_to(painter.frame(painter.compose(base, GHOST, card.tilted, card.at_centre(card.tilted)), step), tilt)
        reel.add(reel.last, hold)


def paste_motion(reel: Reel, painter: Painter, card: Card, base: Image.Image, after: Image.Image,
                 step: int, frames: int, hold: int) -> None:
    """The table comes back around the card, then the real pasted snapshot."""
    for k in range(1, frames):
        ghost = GHOST + (1 - GHOST) * ease(k / frames)
        reel.add(painter.frame(painter.compose(base, ghost, card.final, card.pos), step))
    reel.add(painter.frame(after, step), hold)


def letterbox_motion(reel: Reel, painter: Painter, before: Image.Image, after: Image.Image,
                     bars: Tuple[str, int], frames: int) -> None:
    """The black bars slide in from the two edges."""
    side, width = bars
    for k in range(1, frames):
        grow = round(width * ease(k / frames))
        image = before.copy()
        if grow:
            draw = ImageDraw.Draw(image)
            if side == "top/bottom":
                draw.rectangle([0, 0, SIZE - 1, grow - 1], fill=(0, 0, 0))
                draw.rectangle([0, SIZE - grow, SIZE - 1, SIZE - 1], fill=(0, 0, 0))
            else:
                draw.rectangle([0, 0, grow - 1, SIZE - 1], fill=(0, 0, 0))
                draw.rectangle([SIZE - grow, 0, SIZE - 1, SIZE - 1], fill=(0, 0, 0))
        reel.add(painter.frame(image, S_LETTERBOX))
    reel.add(painter.frame(after, S_LETTERBOX))


def grid_cells(n_cols: int, n_rows: int, tile: int, gap: int, chip_h: int, chip_gap: int):
    grid_w = n_cols * tile + (n_cols - 1) * gap
    grid_h = n_rows * tile + (n_rows - 1) * gap
    x0 = (W - grid_w) // 2
    y0 = (H - (grid_h + chip_gap + chip_h)) // 2
    cells = [(x0 + c * (tile + gap), y0 + r * (tile + gap), c + r) for r in range(n_rows) for c in range(n_cols)]
    return cells, y0 + grid_h + chip_gap


def finale(reel: Reel, painter: Painter, hero: Image.Image, others: List[Image.Image],
           shrink: int = 14, stagger: int = 3, fade: int = 6, hold: int = 22) -> None:
    """The finished sample (alone on the paper) shrinks into the first cell of a grid; the
    other samples fade in in a diagonal wave; then the chip with the size of the
    pre-training set."""
    tile, gap, shadow = 172, 20, 4
    fnt = painter.big_font
    chip_h = painter.chip_height(fnt, 11)
    cells, chip_top = grid_cells(6, 2, tile, gap, chip_h, 34)
    others = others[:len(cells) - 1]
    chip_text = f"{PRETRAIN_COUNT} OF THESE"
    chip_w = round(fnt.getlength(chip_text)) + 2 * 14
    chip_left = (W - chip_w) // 2

    start = shrink - 4
    last_start = start + stagger * max(diag for _, _, diag in cells[1:len(others) + 1])
    chip_start = last_start + 2
    total = chip_start + fade + hold

    for f in range(1, total + 1):
        t = ease(f / shrink)
        canvas = style.paper_canvas(W, H)
        x = round(VIEW_X + (cells[0][0] - VIEW_X) * t)
        y = round(VIEW_Y + (cells[0][1] - VIEW_Y) * t)
        size = round(VIEW + (tile - VIEW) * t)
        painter.framed(canvas, hero, x, y, size, round(6 + (shadow - 6) * t))
        for (cx, cy, diag), image in zip(cells[1:], others):
            alpha = ease((f - start - stagger * diag) / fade)
            if alpha <= 0:
                continue
            layer = canvas.copy()
            painter.framed(layer, image, cx, cy, tile, shadow)
            box = (cx - 2, cy - 2, cx + tile + shadow + 2, cy + tile + shadow + 2)
            canvas.paste(Image.blend(canvas.crop(box), layer.crop(box), alpha), box[:2])
        alpha = ease((f - chip_start) / fade)
        if alpha > 0:
            layer = canvas.copy()
            painter.chip(ImageDraw.Draw(layer), chip_left, chip_top, chip_text, fnt, style.PAPER, style.ACCENT,
                         style.INK, pad=(14, 11))
            box = (chip_left - 2, chip_top - 2, chip_left + chip_w + 8, chip_top + chip_h + 8)
            canvas.paste(Image.blend(canvas.crop(box), layer.crop(box), alpha), box[:2])
        reel.add(canvas)


def render(reel: Reel, painter: Painter, sample: Sample, others: List[Image.Image]) -> None:
    st = sample.stages
    first, second = sample.cards[0], sample.cards[1]
    background, pasted_1 = st["background"], st["paste card 1"]

    def card_view(base, image, card, step):
        return painter.frame(painter.compose(base, GHOST, image, card.at_centre(image)), step)

    opening = painter.frame(background, S_TABLE)
    reel.add(opening, 18)

    # first card, prepared over a faint table
    reel.fade_to(card_view(background, first.cut_out, first, S_CUT_OUT), 6)
    reel.add(reel.last, 12)
    reel.fade_to(card_view(background, first.jittered, first, S_JITTER), 6)
    reel.add(reel.last, 9)
    reel.fade_to(card_view(background, first.jittered, first, S_ROTATE), 4)
    card_motion(reel, painter, first, background, S_ROTATE, rotate=12, tilt=9, hold=6)
    reel.fade_to(card_view(background, first.soft, first, S_BLUR), 6)
    reel.add(reel.last, 9)
    if first.crunched is not None:
        reel.fade_to(card_view(background, first.crunched, first, S_BLUR), 5)
        reel.add(reel.last, 6)
    paste_motion(reel, painter, first, background, pasted_1, S_PASTE, frames=8, hold=10)

    # second card: the same chain, faster, over the faint first paste
    reel.fade_to(card_view(pasted_1, second.jittered, second, S_SECOND), 6)
    reel.add(reel.last, 5)
    card_motion(reel, painter, second, pasted_1, S_SECOND, rotate=9, tilt=6, hold=2)
    reel.fade_to(card_view(pasted_1, second.soft, second, S_SECOND), 4)
    reel.add(reel.last, 2)
    if second.crunched is not None:
        reel.fade_to(card_view(pasted_1, second.crunched, second, S_SECOND), 4)
        reel.add(reel.last, 2)
    paste_motion(reel, painter, second, pasted_1, st["paste card 2"], S_SECOND, frames=8, hold=10)

    # whole-image steps
    reel.fade_to(painter.frame(st["brightness"], S_BRIGHTNESS), 6)
    reel.add(reel.last, 8)
    if "letterbox" in st:
        letterbox_motion(reel, painter, st["brightness"], st["letterbox"], sample.bars, 7)
        reel.add(reel.last, 10)
    reel.fade_to(painter.frame(st["noise"], S_NOISE), 5)
    reel.add(reel.last, 12)

    # the finished sample and its label, then many like it
    reel.fade_to(painter.frame(st["noise"], None, sample.names), 5)
    reel.add(reel.last, 18)
    alone = style.paper_canvas(W, H)
    painter.framed(alone, st["noise"], VIEW_X, VIEW_Y, VIEW, 6)
    reel.fade_to(alone, 6)                       # list and label make way
    finale(reel, painter, st["noise"], others)

    # back to the first frame through blank paper, so the loop restarts cleanly
    reel.fade_to(style.paper_canvas(W, H), 5)
    reel.fade_to(opening, 7)


# ---------------------------------------------------------------- encoding

def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


BT709 = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]


def encode(frames: Path, out_dir: Path, formats: Sequence[str], gif_width: int, gif_fps: int) -> List[Path]:
    pattern = str(frames / "%04d.png")
    # setparams tags the frames too: recent ffmpeg takes primaries and transfer from the
    # frames (unset for PNG input) and ignores -color_primaries / -color_trc.
    to_yuv = ("scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
              "setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709")
    written = []
    if "gif" in formats:
        out = out_dir / "synthesis.gif"
        graph = (f"fps={gif_fps},scale={gif_width}:-2:flags=lanczos,split[a][b];"
                 "[a]palettegen=max_colors=256:stats_mode=full[p];"
                 "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        ffmpeg("-framerate", str(FPS), "-i", pattern, "-vf", graph, "-loop", "0", str(out))
        written.append(out)
    if "mp4" in formats:
        out = out_dir / "synthesis.mp4"
        ffmpeg("-framerate", str(FPS), "-i", pattern, "-vf", to_yuv, "-c:v", "libx264", "-preset", "slow",
               "-crf", "23", "-profile:v", "high", *BT709, "-movflags", "+faststart", "-an", str(out))
        written.append(out)
    if "webm" in formats:
        out = out_dir / "synthesis.webm"
        log = str(frames / "vp9pass")
        common = ["-framerate", str(FPS), "-i", pattern, "-vf", to_yuv, "-c:v", "libvpx-vp9", "-b:v", "0",
                  "-crf", "35", "-row-mt", "1", "-deadline", "good", *BT709, "-an", "-passlogfile", log]
        ffmpeg(*common, "-pass", "1", "-cpu-used", "4", "-f", "null", "-")
        ffmpeg(*common, "-pass", "2", "-cpu-used", "1", str(out))
        written.append(out)
    if "poster" in formats:
        out = out_dir / "synthesis-poster.jpg"
        with Image.open(frames / "0000.png") as first:
            first.convert("RGB").save(out, quality=85, optimize=True, progressive=True)
        written.append(out)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=FIGURES)
    parser.add_argument("--font", default=None, help="a .ttf for the labels (default: style.py's fallback)")
    parser.add_argument("--formats", nargs="+", default=["gif"], choices=["gif", "mp4", "webm", "poster"])
    parser.add_argument("--seed", type=int, default=0, help="generate_synthetic.py --seed")
    parser.add_argument("--index", type=int, default=107, help="the training image to build")
    parser.add_argument("--grid", type=int, nargs="+", default=list(range(11)),
                        help="training images shown in the final grid (11 fill it)")
    parser.add_argument("--gif-width", type=int, default=640)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--frames-dir", type=Path, default=None,
                        help="keep the 1280 x 720 PNG frames here (default: a temporary folder)")
    args = parser.parse_args(argv)

    if shutil.which("ffmpeg") is None:
        sys.exit("error: ffmpeg not found on the PATH")
    style.set_font(args.font)
    cards, backgrounds = load_assets()
    sample = build_sample(cards, backgrounds, args.seed, args.index)
    if len(sample.cards) != 2:
        sys.exit("error: the animation expects the two-card recipe")
    others = []
    for i in args.grid:
        rng, np_rng = sample_rngs(args.seed, TRAIN_SPLIT, i)
        others.append(compose_overlap(cards, backgrounds, rng, np_rng)[0])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.frames_dir is None
    frames = Path(tempfile.mkdtemp(prefix="synthesis_")) if temporary else args.frames_dir
    frames.mkdir(parents=True, exist_ok=True)
    for old in frames.glob("[0-9][0-9][0-9][0-9].png"):    # frames of an earlier run only
        old.unlink()
    try:
        reel = Reel(frames)
        render(reel, Painter(), sample, others)
        written = encode(frames, args.out_dir, args.formats, args.gif_width, args.gif_fps)
    finally:
        if temporary:
            shutil.rmtree(frames, ignore_errors=True)

    print(f"sample {args.index} (seed {args.seed}): {' + '.join(style.card_label(n) for n in sample.names)}")
    for card in sample.cards:
        angle = "?" if card.angle is None else f"{card.angle:.1f}"
        tilt = "none" if card.tilt is None else ("x" if card.tilt[6] else "y")
        print(f"  {card.name}: {card.jittered.width}x{card.jittered.height} px, rotate {angle} deg, "
              f"tilt {tilt}, crunch {'yes' if card.crunched is not None else 'no'}, pasted at {card.pos}")
    print(f"  brightness x{sample.brightness:.3f}, letterbox {sample.bars[0] or 'none'} {sample.bars[1]} px, "
          f"noise sigma {sample.noise_sigma:.1f}")
    print(f"{reel.count} frames, {reel.count / FPS:.1f} s at {FPS} fps")
    for path in written:
        print(f"-> {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
