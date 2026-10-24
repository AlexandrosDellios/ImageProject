"""Turn the labelled course photos into the real-data training and validation sets.

Three steps, run in this order (all of them by default):

    labels  <raw>/train.csv        -> <out>/crop_labels.csv   one row per zone crop
    crops   <raw>/train_images/    -> <out>/crops/<id>_{p1,p2,p3,p4,center}.png
    split   crops + crop_labels    -> <out>/real_val/ + real_val.csv (20 % of the crops, as is)
                                      <out>/real_train/ + real_train.csv (the other 80 %, each
                                      twice: as is, and rotated 180 degrees and rescaled)

    python scripts/prepare_real_data.py
    python scripts/prepare_real_data.py --out /tmp/processed --steps split --seed 1

This replaces the old convert_labels.py, divide_data.py and generate_data2.py and writes
the same files. The split is per crop, not per photo, so different zones of one photo can
end up on both sides. It is kept that way so validation scores stay comparable with the
report's.
"""

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from uno_vision.classes import EMPTY
from uno_vision.paths import PROCESSED, RAW
from uno_vision.zones import ZONE_NAMES, crop_zones

STEPS = ("labels", "crops", "split")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
TOKEN_LABEL = "yellow_token"
VAL_FRACTION = 0.2
SPLIT_SEED = 42                 # the random_state of the original split
SCALE_RANGE = (0.85, 1.15)      # rescale of the augmented copy (camera height)
JPEG_QUALITY = 95


def refuse_symlinks(*paths: Path) -> None:
    # Dataset folders are often symlinks to a shared copy; writing through one would
    # overwrite that copy.
    for path in paths:
        if path.is_symlink():
            sys.exit(f"error: {path} is a symlink; pass another --out to keep what it points to")


def make_crop_labels(train_csv: Path) -> pd.DataFrame:
    """train.csv (one row per photo) -> one `filename,cards` row per zone crop, center first.

    The course labels say who is active but not which token lies on the table, so the
    active player's hand always gets "yellow_token", also on the white-table photos where
    the token is the black one: black_token never appears in the real labels, only in the
    synthetic data. Kept as is, because the submitted model was trained on these labels.
    """
    table = pd.read_csv(train_csv)
    rows = []
    for _, row in table.iterrows():
        image_id = str(row["image_id"]).strip()
        rows.append({"filename": f"{image_id}_center.png", "cards": str(row["center_card"]).strip()})
        active = str(row["active_player"]).strip().lower()
        for k, zone in enumerate(ZONE_NAMES[:4], start=1):
            cards = str(row[f"player_{k}_cards"]).strip()
            if zone == active:
                cards = TOKEN_LABEL if cards == EMPTY else f"{cards};{TOKEN_LABEL}"
            rows.append({"filename": f"{image_id}_{zone}.png", "cards": cards})
    return pd.DataFrame(rows, columns=["filename", "cards"])


def write_crops(images_dir: Path, crops_dir: Path) -> int:
    photos = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    crops_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(photos, desc="crops"):
        with Image.open(path) as im:
            image = np.array(im.convert("RGB"))
        for zone, crop in zip(ZONE_NAMES, crop_zones(image)):
            Image.fromarray(crop).save(crops_dir / f"{path.stem}_{zone}.png")
    return len(photos)


def split_indices(n: int, val_fraction: float = VAL_FRACTION, seed: int = SPLIT_SEED):
    """(train, val) row indices, identical to sklearn's
    train_test_split(test_size=val_fraction, random_state=seed), without needing sklearn."""
    n_val = math.ceil(val_fraction * n)
    order = np.random.RandomState(seed).permutation(n)
    return order[n_val:], order[:n_val]


def open_crop(path: Path):
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except OSError as err:
        print(f"warning: skipping {path.name} ({err})")
        return None


def write_split(crops_dir: Path, labels: pd.DataFrame, out: Path, seed: int):
    train_idx, val_idx = split_indices(len(labels))
    train_dir, val_dir = out / "real_train", out / "real_val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Validation crops are copied untouched (still PNG), so they look like test zones.
    val_rows = []
    for _, row in tqdm(labels.iloc[val_idx].iterrows(), total=len(val_idx), desc="val"):
        image = open_crop(crops_dir / row["filename"])
        if image is not None:
            image.save(val_dir / row["filename"])
            val_rows.append({"filename": row["filename"], "cards": row["cards"]})

    # The old script drew the scales from the unseeded global RNG, so its _aug_1 images
    # cannot be reproduced exactly; --seed makes the new ones repeatable.
    rng = random.Random(seed)
    train_rows = []
    for _, row in tqdm(labels.iloc[train_idx].iterrows(), total=len(train_idx), desc="train"):
        image = open_crop(crops_dir / row["filename"])
        if image is None:
            continue
        stem = Path(row["filename"]).stem
        image.save(train_dir / f"{stem}_aug_0.jpg", quality=JPEG_QUALITY)
        # The same hand seen from the opposite side of the table, from a slightly
        # different camera height.
        flipped = image.rotate(180, expand=True)
        scale = rng.uniform(*SCALE_RANGE)
        size = (max(10, int(flipped.width * scale)), max(10, int(flipped.height * scale)))
        flipped.resize(size, Image.Resampling.LANCZOS).save(train_dir / f"{stem}_aug_1.jpg", quality=JPEG_QUALITY)
        train_rows += [{"filename": f"{stem}_aug_{k}.jpg", "cards": row["cards"]} for k in (0, 1)]

    pd.DataFrame(train_rows, columns=["filename", "cards"]).to_csv(out / "real_train.csv", index=False)
    pd.DataFrame(val_rows, columns=["filename", "cards"]).to_csv(out / "real_val.csv", index=False)
    return len(train_rows), len(val_rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RAW, help="folder with train.csv and train_images/")
    parser.add_argument("--out", type=Path, default=PROCESSED)
    parser.add_argument("--steps", default=",".join(STEPS), help="comma-separated subset of: " + ", ".join(STEPS))
    parser.add_argument("--seed", type=int, default=0, help="seed for the rescale of the augmented copies")
    args = parser.parse_args(argv)

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if not steps:
        sys.exit(f"error: no step given; choose from {', '.join(STEPS)}")
    if unknown:
        sys.exit(f"error: unknown step(s) {', '.join(unknown)}; choose from {', '.join(STEPS)}")

    train_csv, images_dir = args.raw / "train.csv", args.raw / "train_images"
    labels_csv, crops_dir = args.out / "crop_labels.csv", args.out / "crops"
    # Check every input before writing anything; the split reads what the first two steps write.
    needed = {"labels": [train_csv], "crops": [images_dir], "split": []}
    needed["split"] += [p for p, step in ((labels_csv, "labels"), (crops_dir, "crops")) if step not in steps]
    for path in (p for s in steps for p in needed[s]):
        if not path.exists():
            hint = " (run the labels and crops steps first)" if path in (labels_csv, crops_dir) else ""
            sys.exit(f"error: {path} not found{hint}")
    outputs = {"labels": [labels_csv], "crops": [crops_dir],
               "split": [args.out / n for n in ("real_train", "real_val", "real_train.csv", "real_val.csv")]}
    refuse_symlinks(args.out, *(p for s in steps for p in outputs[s]))
    args.out.mkdir(parents=True, exist_ok=True)

    if "labels" in steps:
        labels = make_crop_labels(train_csv)
        labels.to_csv(labels_csv, index=False)
        print(f"labels: {len(labels) // len(ZONE_NAMES)} photos -> {len(labels)} crop labels in {labels_csv}")

    if "crops" in steps:
        n = write_crops(images_dir, crops_dir)
        print(f"crops: {n} photos -> {n * len(ZONE_NAMES)} crops in {crops_dir}")

    if "split" in steps:
        n_train, n_val = write_split(crops_dir, pd.read_csv(labels_csv), args.out, args.seed)
        print(f"split: {n_train} training images (2 per crop) in {args.out / 'real_train'}, "
              f"{n_val} validation crops in {args.out / 'real_val'}")


if __name__ == "__main__":
    main()
