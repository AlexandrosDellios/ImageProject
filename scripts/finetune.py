"""Fine-tune a pre-trained network on the real zone crops, in two stages.

    python scripts/finetune.py --init runs/pretrain/best.pth          # -> runs/finetune/
    python scripts/finetune.py --init weights/uno_resnet18_cbam.pth --epochs-warmup 1 \\
        --epochs-polish 1 --limit 32 --num-workers 0                 # smoke test

--data must hold real_train.csv + real_train/ (training crops) and real_val.csv +
real_val/ (validation crops), as written by the real-data preparation step.

1. Warm-up: a linear probe. Only the head (fc.*) trains, so it first adapts to real
   photos without large early gradients disturbing the pre-trained features.
   lr 4e-4, at most 100 epochs, early stop after 20 -> warmup.pth
2. Polish: every layer trains, starting from the best warm-up weights.
   lr 6e-4, at most 200 epochs, early stop after 25 -> best.pth (the fine-tuned model)

Both stages (as in stage 3 of the original multi-stage script): batch 64, focal loss
(gamma 2, no label smoothing), AdamW with torch's default weight decay (0.01), learning
rate halved after 5 epochs without improvement, no online augmentation (the training set
already holds a rotated and rescaled copy of every crop). The polish stage counts its
early stopping from its own first epoch, so best.pth is its best epoch even when the
warm-up ended on a lower validation loss.

Also writes curves_warmup.png, curves_polish.png and history.json to --out-dir.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim

from uno_vision.data import UnoCSVDataset
from uno_vision.engine import make_loader, save_curves, save_history, set_trainable, train_stage
from uno_vision.losses import FocalLoss
from uno_vision.model import UnoCountingResNet
from uno_vision.paths import PROCESSED, RUNS
from uno_vision.transforms import eval_transform
from uno_vision.utils import pick_device, seed_everything

BATCH_SIZE = 64
FOCAL_GAMMA = 2.0
WEIGHT_DECAY = 1e-2           # torch's AdamW default, which the original script relied on
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 5
WARMUP = {"lr": 4e-4, "epochs": 100, "patience": 20}
POLISH = {"lr": 6e-4, "epochs": 200, "patience": 25}
HEAD_PREFIXES = ("fc.",)


def run_stage(model, loaders, criterion, device, lr, epochs, patience, save_path, name):
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=PLATEAU_FACTOR,
                                                     patience=PLATEAU_PATIENCE)
    return train_stage(model, loaders[0], loaders[1], criterion, optimizer, scheduler, device,
                       epochs=epochs, patience=patience, save_path=save_path, name=name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", type=Path, required=True, help="pre-trained weights (e.g. runs/pretrain/best.pth)")
    parser.add_argument("--data", type=Path, default=PROCESSED, help="folder with real_train(.csv) and real_val(.csv)")
    parser.add_argument("--out-dir", type=Path, default=RUNS / "finetune")
    parser.add_argument("--epochs-warmup", type=int, default=WARMUP["epochs"], help="0 skips the warm-up")
    parser.add_argument("--epochs-polish", type=int, default=POLISH["epochs"], help="0 skips the polish")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=0, help="use only the first N train and N val crops (smoke tests)")
    parser.add_argument("--num-workers", type=int, default=4, help="data-loading processes (0 = main process)")
    parser.add_argument("--device", default=None, help="cpu, cuda or mps (default: best available)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.epochs_warmup < 0 or args.epochs_polish < 0 or args.epochs_warmup + args.epochs_polish == 0:
        sys.exit("error: --epochs-warmup and --epochs-polish must be >= 0, and not both 0")
    if args.limit < 0 or args.batch_size < 1:
        sys.exit("error: --limit must be >= 0 and --batch-size >= 1")
    if not args.init.is_file():
        sys.exit(f"error: {args.init} not found (or not a file)")
    files = {"train_csv": args.data / "real_train.csv", "train_dir": args.data / "real_train",
             "val_csv": args.data / "real_val.csv", "val_dir": args.data / "real_val"}
    for path in files.values():
        if not path.exists():
            sys.exit(f"error: {path} not found")

    seed_everything(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")

    transform = eval_transform()
    train_set = UnoCSVDataset(files["train_csv"], files["train_dir"], transform, limit=args.limit)
    val_set = UnoCSVDataset(files["val_csv"], files["val_dir"], transform, limit=args.limit)
    if not len(train_set) or not len(val_set):
        sys.exit("error: empty training or validation CSV")
    loaders = (make_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers, device=device),
               make_loader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers, device=device))

    model = UnoCountingResNet().to(device)
    model.load_state_dict(torch.load(args.init, map_location=device, weights_only=True))
    print(f"initial weights: {args.init}")
    criterion = FocalLoss(gamma=FOCAL_GAMMA)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    warmup_path, best_path = args.out_dir / "warmup.pth", args.out_dir / "best.pth"
    config = {"script": "finetune", "init": args.init, "data": args.data, "train_images": len(train_set),
              "val_images": len(val_set), "batch_size": args.batch_size, "focal_gamma": FOCAL_GAMMA,
              "weight_decay": WEIGHT_DECAY, "plateau_factor": PLATEAU_FACTOR,
              "plateau_patience": PLATEAU_PATIENCE, "head_prefixes": HEAD_PREFIXES,
              "warmup": dict(WARMUP, epochs=args.epochs_warmup),
              "polish": dict(POLISH, epochs=args.epochs_polish), "seed": args.seed, "device": device}
    stages = {}

    if args.epochs_warmup:
        # The original filter kept every parameter whose name *contains* "fc", which also
        # trained CBAM's channel attention (ca.fc1, ca.fc2). Here only the head trains.
        # BatchNorm running statistics in the backbone still update (train mode), as before.
        trainable, total = set_trainable(model, HEAD_PREFIXES)
        print(f"warm-up: {trainable:,} of {total:,} parameters trainable (head only)")
        stages["warmup"] = run_stage(model, loaders, criterion, device, WARMUP["lr"], args.epochs_warmup,
                                     WARMUP["patience"], warmup_path, "warmup")
        save_curves(stages["warmup"], args.out_dir / "curves_warmup.png", "Fine-tuning 1/2: warm-up (head only)")
        save_history(args.out_dir / "history.json", config=config, **stages)
        if stages["warmup"]["stopped"] == "interrupted":
            sys.exit(f"stopped during the warm-up; outputs so far in {args.out_dir}")
        # Check this run's history, not the file: a warmup.pth from an earlier run may be there.
        if stages["warmup"]["best_epoch"] is None:
            sys.exit("error: the warm-up saved no checkpoint (validation loss never finite)")
        model.load_state_dict(torch.load(warmup_path, map_location=device, weights_only=True))

    if args.epochs_polish:
        trainable, total = set_trainable(model)
        print(f"polish: all {total:,} parameters trainable")
        stages["polish"] = run_stage(model, loaders, criterion, device, POLISH["lr"], args.epochs_polish,
                                     POLISH["patience"], best_path, "polish")
        save_curves(stages["polish"], args.out_dir / "curves_polish.png", "Fine-tuning 2/2: polish (all layers)")
        save_history(args.out_dir / "history.json", config=config, **stages)
        if stages["polish"]["best_epoch"] is None:
            sys.exit(f"error: the polish stage saved no checkpoint; {best_path} is not from this run")
    else:
        shutil.copyfile(warmup_path, best_path)   # warm-up only: its best is the result

    print(f"fine-tuned weights -> {best_path}; curves and history in {args.out_dir}")


if __name__ == "__main__":
    main()
