"""Generate the synthetic pre-training set: 336 x 336 images of two overlapping cards.

    python scripts/generate_synthetic.py                     # 10,000 train + 2,000 val -> data/synthetic/overlap
    python scripts/generate_synthetic.py --n-train 24 --n-val 8 --out /tmp/synth_check --seed 1

Writes <out>/train/occlusion_train_<i>.jpg, <out>/val/occlusion_val_<i>.jpg and the labels
<out>/synthetic_train.csv and <out>/synthetic_val.csv (filename,cards), read by
uno_vision.data.UnoCSVDataset. Each image is seeded from (seed, split, index), so image i
does not change with --n-train or --n-val. The recipe and its parameters are in
uno_vision/synth/overlap.py (a port of the old, unseeded generate_doubles.py).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from tqdm import tqdm

from uno_vision.classes import CLASS_TO_IDX
from uno_vision.paths import ASSETS, SYNTHETIC
from uno_vision.synth import compose_overlap, load_assets, sample_rngs

SPLITS = ("train", "val")   # the index is part of each image's seed


def refuse_symlinks(*paths: Path) -> None:
    # Dataset folders are often symlinks to a shared copy; writing through one would
    # overwrite that copy.
    for path in paths:
        if path.is_symlink():
            sys.exit(f"error: {path} is a symlink; pass another --out to keep what it points to")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=SYNTHETIC / "overlap")
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--assets", type=Path, default=ASSETS, help="folder with cards/ and backgrounds/")
    args = parser.parse_args(argv)

    # numpy's SeedSequence, which seeds every sample, only takes non-negative integers.
    if min(args.n_train, args.n_val, args.seed) < 0:
        sys.exit("error: --n-train, --n-val and --seed must be 0 or more")
    for folder in (args.assets / "cards", args.assets / "backgrounds"):
        if not folder.is_dir():
            sys.exit(f"error: {folder} not found")
    cards, backgrounds = load_assets(args.assets)
    if not cards:
        sys.exit(f"error: no card images in {args.assets / 'cards'}")
    unknown = sorted(set(cards) - set(CLASS_TO_IDX))
    if unknown:
        sys.exit(f"error: card files that are not class names: {', '.join(unknown)}")
    if not backgrounds:
        print("warning: no backgrounds found, cards go on plain white")

    refuse_symlinks(args.out, *(args.out / s for s in SPLITS), *(args.out / f"synthetic_{s}.csv" for s in SPLITS))
    print(f"{len(cards)} cards, {len(backgrounds)} backgrounds -> {args.out}")

    for split_id, (split, count) in enumerate(zip(SPLITS, (args.n_train, args.n_val))):
        folder = args.out / split
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        start = time.perf_counter()
        for i in tqdm(range(count), desc=split):
            rng, np_rng = sample_rngs(args.seed, split_id, i)
            image, names = compose_overlap(cards, backgrounds, rng, np_rng)
            filename = f"occlusion_{split}_{i}.jpg"
            image.save(folder / filename, quality=95)
            rows.append({"filename": filename, "cards": ";".join(names)})
        csv_path = args.out / f"synthetic_{split}.csv"
        pd.DataFrame(rows, columns=["filename", "cards"]).to_csv(csv_path, index=False)
        if count:
            per_image = (time.perf_counter() - start) / count * 1000
            print(f"{split}: {count} images -> {csv_path} ({per_image:.1f} ms per image)")


if __name__ == "__main__":
    main()
