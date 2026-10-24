"""Evaluate a checkpoint on the held-out real crops: counting F1 per class, error types
and a Grad-CAM grid.

    python scripts/evaluate.py                                  # submitted weights on data/processed/real_val
    python scripts/evaluate.py --weights runs/finetune/best.pth --out-dir runs/evaluate_finetune
    python scripts/evaluate.py --gradcam-layer layer3           # the layer behind the report's Grad-CAM figure

Writes into --out-dir:
    metrics.json          mean F1 over the classes present, per-class tp/fp/fn/precision/recall/f1
    f1_per_class.png      F1 of every class present in the set
    errors_per_class.png  extra copies (false positives) and missed copies (false negatives)
    gradcam.png           --n-gradcam crops, picked with --seed, with their Grad-CAM heatmaps

With the default weights the mean F1 is 0.7370 over 48 classes (the report's figure).
"""

import argparse
import hashlib
import json
import os
import random
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader
from tqdm import tqdm

from uno_vision.classes import CLASS_NAMES
from uno_vision.data import UnoCSVDataset
from uno_vision.gradcam import LAYERS, GradCAM, colorize, display_crop, overlay
from uno_vision.metrics import active_accuracy, counting_scores
from uno_vision.model import load_model
from uno_vision.paths import PROCESSED, ROOT, RUNS, WEIGHTS
from uno_vision.transforms import eval_transform
from uno_vision.utils import pick_device

BATCH_SIZE = 32

# Chart colours: two categorical slots that stay apart under colour-vision deficiencies,
# with text and axes in neutral inks so a light bar colour never has to be read as text.
SERIES_1 = "#2a78d6"     # blue
SERIES_2 = "#eb6834"     # orange
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


@torch.no_grad()
def predict_counts(model, loader, device):
    """Targets and argmax counts, both (n_crops, 56), in dataset order."""
    targets, preds = [], []
    for images, labels, _ in tqdm(loader, desc="Evaluating"):
        logits = model(images.to(device))                 # (batch, 56, 6)
        preds.append(logits.argmax(dim=2).cpu())
        targets.append(labels)
    return torch.cat(targets).numpy(), torch.cat(preds).numpy()


def display_path(path: Path) -> str:
    """Repo-relative when possible, so metrics.json does not record local folders."""
    try:
        # abspath folds "..", so "../repo/weights/x.pth" is found inside the repo and "../x.pth"
        # is not; it keeps symlinks, so data/processed/... stays under the repo.
        return str(Path(os.path.abspath(path)).relative_to(ROOT))
    except ValueError:
        return path.name


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_metrics(targets, preds, args):
    scores = counting_scores(targets, preds)
    active = scores["active"]
    support = targets.sum(axis=0)
    # Share of (crop, class) pairs with true copies whose count is exactly right.
    correct, total = active_accuracy(torch.from_numpy(preds), torch.from_numpy(targets))
    per_class = {}
    for i in np.flatnonzero(active):
        per_class[CLASS_NAMES[i]] = {
            "support": int(support[i]),
            "tp": int(scores["tp"][i]), "fp": int(scores["fp"][i]), "fn": int(scores["fn"][i]),
            "precision": round(float(scores["precision"][i]), 6),
            "recall": round(float(scores["recall"][i]), 6),
            "f1": round(float(scores["f1"][i]), 6),
        }
    # Copies predicted for classes that never occur in the set: real errors, but outside
    # the mean F1 (a class without true copies has no recall).
    absent_fp = {CLASS_NAMES[i]: int(scores["fp"][i])
                 for i in np.flatnonzero(~active) if scores["fp"][i] > 0}
    return {
        "weights": display_path(args.weights),
        "weights_md5": md5(args.weights),
        "crops": display_path(args.crops),
        "labels": display_path(args.labels),
        "n_crops": int(len(targets)),
        "n_active_classes": int(active.sum()),
        "mean_active_f1": round(float(scores["f1"][active].mean()), 6),
        "exact_count_accuracy": round(correct / total, 6) if total else None,
        "totals": {"tp": int(scores["tp"][active].sum()), "fp": int(scores["fp"][active].sum()),
                   "fn": int(scores["fn"][active].sum())},
        "per_class": per_class,
        "false_positives_on_absent_classes": absent_fp,
    }


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def class_ticks(ax, metrics):
    names = list(metrics["per_class"])
    labels = [f"{n} ({metrics['per_class'][n]['support']})" for n in names]
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_xlim(-0.7, len(names) - 0.3)
    ax.set_xlabel("class (true copies in the set)", color=INK_2, fontsize=10)


def plot_f1(metrics, path: Path):
    names = list(metrics["per_class"])
    f1 = np.array([metrics["per_class"][n]["f1"] for n in names])
    mean = metrics["mean_active_f1"]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(18, 6), facecolor=SURFACE)
    style_axes(ax)
    ax.bar(x, f1, width=0.6, color=SERIES_1)
    # A zero bar is invisible: say it is a zero, not a missing class.
    for xi in x[f1 == 0]:
        ax.text(xi, 0.012, "0", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.axhline(mean, color=INK_2, linewidth=1)
    # Label the mean line in the right margin, where no bar can cover it.
    ax.text(1.004, mean, f"mean\n{mean:.4f}", transform=ax.get_yaxis_transform(), ha="left",
            va="center", fontsize=9, color=INK_2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 (counting)", color=INK_2, fontsize=10)
    class_ticks(ax, metrics)
    ax.set_title(f"F1 per class on {metrics['n_crops']} held-out real crops: mean {mean:.4f} over "
                 f"the {metrics['n_active_classes']} classes present ({Path(metrics['weights']).name})",
                 color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_errors(metrics, path: Path):
    names = list(metrics["per_class"])
    fp = np.array([metrics["per_class"][n]["fp"] for n in names])
    fn = np.array([metrics["per_class"][n]["fn"] for n in names])
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(18, 6), facecolor=SURFACE)
    style_axes(ax)
    # 0.36-wide bars 0.40 apart leave a thin gap between the two bars of a class.
    ax.bar(x - 0.2, fp, width=0.36, color=SERIES_2, label="false positives (extra copies)")
    ax.bar(x + 0.2, fn, width=0.36, color=SERIES_1, label="false negatives (missed copies)")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("copies", color=INK_2, fontsize=10)
    class_ticks(ax, metrics)
    # Above the plot area, right-aligned, so it can never cover a tall bar.
    legend = ax.legend(frameon=False, fontsize=10, ncol=2, loc="lower right", bbox_to_anchor=(1, 1.0))
    for text in legend.get_texts():
        text.set_color(INK)
    totals = metrics["totals"]
    title = (f"Counting errors per class: {totals['fp']} extra and {totals['fn']} missed copies "
             f"({Path(metrics['weights']).name})")
    absent = metrics["false_positives_on_absent_classes"]
    if absent:
        listed = ", ".join(f"{n} x{c}" for n, c in absent.items())
        title += f"\nnot shown, classes absent from the set but predicted: {listed}"
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def pick_gradcam_crops(targets, n: int, seed: int):
    """n crops holding at least one card, chosen reproducibly."""
    candidates = [i for i in range(len(targets)) if targets[i].sum() > 0]
    return random.Random(seed).sample(candidates, min(n, len(candidates)))


def plot_gradcam(model, dataset, indices, targets, preds, layer: str, seed: int, device, path: Path):
    # Fixed sizes in inches, so titles and captions never collide whatever the number of rows.
    panel, gap, margin = 3.4, 0.15, 0.15
    title_h, caption_h, top, bottom = 0.35, 0.55, 0.5, 0.7
    rows = len(indices)
    row_h = title_h + panel + caption_h
    width = 2 * margin + 3 * panel + 2 * gap
    height = top + rows * row_h + bottom
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)

    with GradCAM(model, layer) as cam:
        for row, idx in enumerate(indices):
            image, _, crop_path = dataset[idx]
            # Explain the first class present at its true count, as the old script did.
            cls = int(np.flatnonzero(targets[idx])[0])
            count = int(targets[idx][cls])
            heat = cam(image.to(device), cls, count)
            crop = display_crop(crop_path)
            colored = colorize(heat)
            name = CLASS_NAMES[cls]
            true_cards = str(dataset.frame.iloc[idx]["cards"]).replace(";", ", ")

            panels = [(crop, Path(crop_path).name),
                      (colored, f"Grad-CAM for {name} = {count}"),
                      (overlay(crop, colored), f"predicted {name} = {int(preds[idx][cls])}")]
            y = height - top - row * row_h - title_h - panel
            for col, (img, title) in enumerate(panels):
                x = margin + col * (panel + gap)
                ax = fig.add_axes([x / width, y / height, panel / width, panel / height])
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_title(title, color=INK, fontsize=10)
                if col == 0:
                    ax.set_xlabel("\n".join(textwrap.wrap("true: " + true_cards, 52)), color=INK_2, fontsize=8.5)

    crops = "crop" if rows == 1 else "crops"
    fig.suptitle(f"Grad-CAM on {layer}, {rows} held-out {crops} (seed {seed})", color=INK, fontsize=12,
                 y=1 - 0.15 / height, va="top")
    # One shared scale: the heatmaps are each normalised to [0, 1].
    cax = fig.add_axes([0.38, 0.3 / height, 0.24, 0.1 / height])
    bar = fig.colorbar(plt.cm.ScalarMappable(cmap="jet"), cax=cax, orientation="horizontal")
    bar.set_ticks([0, 1])
    bar.set_ticklabels(["low", "high"])
    bar.ax.tick_params(labelsize=8, colors=INK_2)
    bar.outline.set_visible(False)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--crops", type=Path, default=PROCESSED / "real_val", help="folder of zone crops")
    parser.add_argument("--labels", type=Path, default=PROCESSED / "real_val.csv", help="filename,cards CSV")
    parser.add_argument("--out-dir", type=Path, default=RUNS / "evaluate")
    parser.add_argument("--gradcam-layer", choices=sorted(LAYERS), default="layer4",
                        help="residual stage to explain (the report's figure used layer3)")
    parser.add_argument("--n-gradcam", type=int, default=3, help="crops in the Grad-CAM grid (0 to skip)")
    parser.add_argument("--seed", type=int, default=0, help="picks the Grad-CAM crops")
    parser.add_argument("--device", default=None, help="cpu, cuda or mps (default: best available)")
    # 0: for 81 crops, starting worker processes (spawned on macOS) costs more than it saves.
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    args = parser.parse_args(argv)

    for path in (args.weights, args.crops, args.labels):
        if not path.exists():
            sys.exit(f"error: {path} not found")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    print(f"device: {device}")
    model = load_model(args.weights, device)
    dataset = UnoCSVDataset(args.labels, args.crops, transform=eval_transform(), return_path=True)
    if len(dataset) == 0:
        sys.exit(f"error: {args.labels} lists no crops")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers)

    targets, preds = predict_counts(model, loader, device)
    metrics = build_metrics(targets, preds, args)
    print(f"mean F1 over the {metrics['n_active_classes']} classes present: {metrics['mean_active_f1']:.4f}")

    plot_f1(metrics, args.out_dir / "f1_per_class.png")
    plot_errors(metrics, args.out_dir / "errors_per_class.png")
    if args.n_gradcam > 0:
        indices = pick_gradcam_crops(targets, args.n_gradcam, args.seed)
        plot_gradcam(model, dataset, indices, targets, preds, args.gradcam_layer, args.seed, device,
                     args.out_dir / "gradcam.png")
        metrics["gradcam"] = {"layer": args.gradcam_layer, "seed": args.seed,
                              "crops": [str(dataset.frame.iloc[i]["filename"]) for i in indices]}

    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print(f"metrics.json and figures -> {args.out_dir}")


if __name__ == "__main__":
    main()
