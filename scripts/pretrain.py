"""Pre-train the counting network from scratch on the synthetic two-card images.

    python scripts/pretrain.py                                   # data/synthetic/overlap -> runs/pretrain/
    python scripts/pretrain.py --limit 128 --epochs 1 --batch-size 16 --num-workers 0   # smoke test

--data must hold synthetic_train.csv, synthetic_val.csv, train/ and val/ (the synthetic
overlap set). Writes to --out-dir: best.pth (lowest validation loss), curves.png and
history.json. The result is the starting point of scripts/finetune.py (--init).

Recipe (as in the original pre-training script): no pretrained weights, batch 64, AdamW
(lr 1e-3, weight decay 7e-4), focal loss (gamma 2, label smoothing 0.1), learning rate
halved after 5 epochs without improvement, at most 150 epochs, early stop after 20.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.optim as optim

from uno_vision.data import UnoCSVDataset
from uno_vision.engine import make_loader, save_curves, save_history, train_stage
from uno_vision.losses import FocalLoss
from uno_vision.model import UnoCountingResNet
from uno_vision.paths import RUNS, SYNTHETIC
from uno_vision.transforms import eval_transform
from uno_vision.utils import pick_device, seed_everything

BATCH_SIZE = 64
EPOCHS = 150
PATIENCE = 20                 # early stop: epochs without a better validation loss
LR = 1e-3
WEIGHT_DECAY = 7e-4           # strong decay: the synthetic set is easy to memorise
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1
PLATEAU_FACTOR = 0.5          # ReduceLROnPlateau: halve the learning rate...
PLATEAU_PATIENCE = 5          # ...after 5 epochs without improvement


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=SYNTHETIC / "overlap", help="synthetic dataset folder")
    parser.add_argument("--out-dir", type=Path, default=RUNS / "pretrain")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=0, help="use only the first N train and N val images (smoke tests)")
    parser.add_argument("--num-workers", type=int, default=4, help="data-loading processes (0 = main process)")
    parser.add_argument("--device", default=None, help="cpu, cuda or mps (default: best available)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.epochs < 1 or args.limit < 0 or args.batch_size < 1:
        sys.exit("error: --epochs and --batch-size must be >= 1, --limit >= 0")
    files = {"train_csv": args.data / "synthetic_train.csv", "train_dir": args.data / "train",
             "val_csv": args.data / "synthetic_val.csv", "val_dir": args.data / "val"}
    for path in files.values():
        if not path.exists():
            sys.exit(f"error: {path} not found (scripts/generate_synthetic.py writes it)")

    seed_everything(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")

    transform = eval_transform()
    train_set = UnoCSVDataset(files["train_csv"], files["train_dir"], transform, limit=args.limit)
    val_set = UnoCSVDataset(files["val_csv"], files["val_dir"], transform, limit=args.limit)
    if not len(train_set) or not len(val_set):
        sys.exit("error: empty training or validation CSV")
    train_loader = make_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers, device=device)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers, device=device)

    model = UnoCountingResNet().to(device)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=PLATEAU_FACTOR,
                                                     patience=PLATEAU_PATIENCE)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    weights = args.out_dir / "best.pth"
    history = train_stage(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                          epochs=args.epochs, patience=PATIENCE, save_path=weights, name="pretrain")

    config = {"script": "pretrain", "data": args.data, "train_images": len(train_set),
              "val_images": len(val_set), "epochs": args.epochs, "patience": PATIENCE,
              "batch_size": args.batch_size, "lr": LR, "weight_decay": WEIGHT_DECAY,
              "focal_gamma": FOCAL_GAMMA, "label_smoothing": LABEL_SMOOTHING,
              "plateau_factor": PLATEAU_FACTOR, "plateau_patience": PLATEAU_PATIENCE,
              "seed": args.seed, "device": device}
    save_curves(history, args.out_dir / "curves.png", "Synthetic pre-training")
    save_history(args.out_dir / "history.json", config=config, pretrain=history)
    if history["best_epoch"] is None:   # a best.pth already in out-dir is from an earlier run
        sys.exit(f"error: no checkpoint saved; {weights} is not from this run")
    print(f"best epoch {history['best_epoch']} -> {weights}; curves and history in {args.out_dir}")


if __name__ == "__main__":
    main()
