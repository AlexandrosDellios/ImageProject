"""Draw f1_per_class.png: counting F1 of every class present in the held-out real crops,
sorted from best to worst, with the mean as a dashed line. Classes with F1 = 0 are red.

The numbers are computed here (submitted weights on data/processed/real_val), not read
from runs/: with the default weights the mean is 0.7370 over 48 classes and 81 crops,
as scripts/evaluate.py reports.

    python scripts/figures/f1_chart.py --font path/to/GeistMono-500.ttf
    python scripts/figures/f1_chart.py --weights runs/finetune/best.pth --out-dir some/folder
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.path import Path as MplPath
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

import style
from uno_vision.classes import CLASS_NAMES, is_token
from uno_vision.data import UnoCSVDataset
from uno_vision.metrics import counting_scores
from uno_vision.model import load_model
from uno_vision.paths import FIGURES, PROCESSED, WEIGHTS
from uno_vision.transforms import eval_transform
from uno_vision.utils import pick_device

WIDTH, HEIGHT, DPI = 1800, 800, 100
PT = 72 / DPI                  # points per pixel at this DPI: sizes below are given in pixels
TICK_PX = 23
YTICK_PX = 25
CHIP_PX = 26


@torch.no_grad()
def predict_counts(model, crops: Path, labels: Path, device):
    """Targets and argmax counts, both (n_crops, 56)."""
    dataset = UnoCSVDataset(labels, crops, transform=eval_transform())
    targets, preds = [], []
    for images, counts in DataLoader(dataset, batch_size=32, shuffle=False):
        preds.append(model(images.to(device)).argmax(dim=2).cpu())
        targets.append(counts)
    return torch.cat(targets).numpy(), torch.cat(preds).numpy()


def tick_labels(names):
    """style.card_label for each class. card_label writes both tokens as "TOKEN"; their colour
    is added only if both appear, since the real labels call every turn token "yellow_token"
    (the grey block on the white table as well as the yellow disc on the cloth)."""
    both = sum(is_token(n) for n in names) > 1
    return [n.split("_")[0].upper() + " TOKEN" if both and is_token(n) else style.card_label(n) for n in names]


def sorted_classes(targets, preds):
    """(names, f1, mean) for the classes present, best first (ties: more copies first)."""
    scores = counting_scores(targets, preds)
    active = np.flatnonzero(scores["active"])
    support = targets.sum(axis=0)
    order = sorted(active, key=lambda i: (-scores["f1"][i], -support[i], i))
    return [CLASS_NAMES[i] for i in order], scores["f1"][order], float(scores["f1"][active].mean())


def render(names, f1, mean, n_crops) -> Image.Image:
    style.matplotlib_style()
    # A single-weight font (Geist Mono 500) makes matplotlib log a fallback note per text.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    ink, ink2 = style.hex_color(style.INK), style.hex_color(style.INK_2)
    accent, accent_ink = style.hex_color(style.ACCENT), style.hex_color(style.ACCENT_INK)
    paper = style.hex_color(style.PAPER)

    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
    # Axes box in pixels: room on the left for the y ticks, below for the rotated names.
    left, right, top, bottom = 104, 30, 70, 236
    ax = fig.add_axes([left / WIDTH, bottom / HEIGHT, (WIDTH - left - right) / WIDTH,
                       (HEIGHT - top - bottom) / HEIGHT])
    x = np.arange(len(names))
    zero = f1 == 0
    bars = ax.bar(x[~zero], f1[~zero], width=0.62, color=ink, zorder=2)
    # A zero bar has no height: mark it with a thick red stub on the baseline instead.
    for xi in x[zero]:
        ax.plot([xi - 0.31, xi + 0.31], [0.006, 0.006], color=accent, linewidth=7, solid_capstyle="butt",
                zorder=3, clip_on=False)
    if zero.any():
        ax.text(x[zero].mean(), 0.045, "F1 = 0", ha="center", va="bottom", color=accent_ink,
                fontsize=YTICK_PX * PT)

    # The mean: a dashed line, ink on paper and paper where it crosses a bar (the same line
    # drawn twice, the second copy clipped to the bars, so the dashes stay in phase).
    xlim = (-0.7, len(names) - 0.3)
    dashes = dict(linewidth=2.4, linestyle=(0, (5, 3.5)), dash_capstyle="butt")
    ax.plot(xlim, [mean, mean], color=ink, zorder=4, **dashes)
    knockout, = ax.plot(xlim, [mean, mean], color=paper, zorder=5, **dashes)
    bar_paths = [b.get_path().transformed(b.get_patch_transform()) for b in bars]
    knockout.set_clip_path(MplPath.make_compound_path(*bar_paths), ax.transData)

    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels(names), rotation=90, ha="center", va="top")
    for tick, is_zero in zip(ax.get_xticklabels(), zero):
        tick.set_color(accent_ink if is_zero else ink)
    ax.tick_params(axis="x", length=0, pad=8, labelsize=TICK_PX * PT)
    ax.tick_params(axis="y", length=6, width=1.5, pad=6, labelsize=YTICK_PX * PT)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.5)
    # Axis name above the axis, aligned with the tick labels, instead of a rotated label.
    fig.text((left - 12) / WIDTH, 1 - (top - 24) / HEIGHT, "F1", ha="right", va="bottom",
             fontsize=YTICK_PX * PT, color=ink2)

    fig.canvas.draw()
    image = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    # Where the mean line and the plot's right edge land, in image pixels (y down).
    to_px = ax.transData.transform
    _, mean_y = to_px((0, mean))
    mean_y = HEIGHT - mean_y
    plot_right = WIDTH - right
    plt.close(fig)

    # The chip sits just above the mean line, right-aligned, over bars that end below it.
    text = f"MEAN F1 {mean:.3f} · {len(names)} CLASSES · {n_crops} CROPS"
    fnt = style.font(CHIP_PX)
    w, h = style.chip_size(text, fnt, (12, 8))
    draw = ImageDraw.Draw(image)
    style.draw_chip(draw, (int(plot_right - w - 8), int(mean_y - h - 18)), text, fnt, pad=(12, 8))
    return image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--crops", type=Path, default=PROCESSED / "real_val", help="folder of zone crops")
    parser.add_argument("--labels", type=Path, default=PROCESSED / "real_val.csv", help="filename,cards CSV")
    parser.add_argument("--device", default=None, help="cpu, cuda or mps (default: best available)")
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
    targets, preds = predict_counts(model, args.crops, args.labels, device)
    names, f1, mean = sorted_classes(targets, preds)
    zeros = [n for n, v in zip(names, f1) if v == 0]
    print(f"{len(targets)} crops, {len(names)} classes present, mean F1 {mean:.4f}; F1 = 0: {', '.join(zeros) or 'none'}")

    out = args.out_dir / "f1_per_class.png"
    render(names, f1, mean, len(targets)).save(out, optimize=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
