"""Read every table photo in a folder and write the game state as a CSV.

    python scripts/predict.py                                  # data/raw/test_images -> submission.csv
    python scripts/predict.py --images my_photos --out my.csv --device cpu

With the default weights on the course test set, the output equals results/submission.csv.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uno_vision.model import load_model
from uno_vision.paths import RAW, ROOT, WEIGHTS
from uno_vision.predict import predict_folder
from uno_vision.utils import pick_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=RAW / "test_images", help="folder of table photos")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--out", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--device", default=None, help="cpu, cuda or mps (default: best available)")
    args = parser.parse_args(argv)

    for path in (args.images, args.weights):
        if not path.exists():
            sys.exit(f"error: {path} not found")
    if not args.out.parent.is_dir():
        sys.exit(f"error: {args.out.parent} not found (the folder for --out)")

    device = pick_device(args.device)
    print(f"device: {device}")
    model = load_model(args.weights, device)
    table = predict_folder(model, args.images, device)
    table.to_csv(args.out, index=False)
    print(f"{len(table)} photos -> {args.out}")


if __name__ == "__main__":
    main()
