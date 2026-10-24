"""Training loop shared by pre-training and fine-tuning.

    train_loader = make_loader(train_set, batch_size=64, shuffle=True, num_workers=4, device=device)
    val_loader = make_loader(val_set, batch_size=64, shuffle=False, num_workers=4, device=device)
    history = train_stage(model, train_loader, val_loader, criterion, optimizer, scheduler,
                          device, epochs=150, patience=20, save_path=out / "best.pth")
    save_curves(history, out / "curves.png", "Synthetic pre-training")
    save_history(out / "history.json", config={...}, pretrain=history)

Each epoch is one pass over the training set, then one over the validation set. The
weights with the lowest validation loss are saved; training stops after `patience` epochs
without improvement. Validation also reports the active accuracy (uno_vision.metrics).
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .metrics import active_accuracy

PathLike = Union[str, Path]


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 4,
                device: Optional[torch.device] = None) -> DataLoader:
    """A DataLoader with the settings every training script uses. num_workers=0 loads in
    the main process: slower, but it works everywhere and is easiest to debug."""
    pin = device is not None and torch.device(device).type == "cuda"   # only CUDA benefits
    # Persistent workers start once per run instead of once per epoch: on macOS starting
    # and stopping them costs several seconds each time.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin, persistent_workers=num_workers > 0)


def set_trainable(model: torch.nn.Module, prefixes: Optional[Iterable[str]] = None) -> Tuple[int, int]:
    """Train only the parameters whose name starts with one of `prefixes` (all of them if
    None). Returns (trainable, total) parameter counts."""
    prefixes = tuple(prefixes) if prefixes is not None else None
    trainable = total = 0
    for name, param in model.named_parameters():
        param.requires_grad = prefixes is None or name.startswith(prefixes)
        total += param.numel()
        trainable += param.numel() if param.requires_grad else 0
    return trainable, total


def train_one_epoch(model: torch.nn.Module, loader: DataLoader, criterion, optimizer,
                    device: torch.device, desc: str = "train") -> float:
    """One pass over `loader`; returns the mean of the batch losses."""
    model.train()
    total = 0.0
    bar = tqdm(loader, desc=desc, leave=False)
    for batch in bar:
        images, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        # (B, 56, 6) -> (B, 6, 56): cross-entropy wants the count bins on dim 1.
        loss = criterion(model(images).permute(0, 2, 1), labels)
        loss.backward()
        optimizer.step()
        total += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, criterion, device: torch.device,
             desc: str = "val") -> Tuple[float, float]:
    """(mean batch loss, active accuracy in %) over `loader`."""
    model.eval()
    total_loss, correct, counted = 0.0, 0, 0
    for batch in tqdm(loader, desc=desc, leave=False):
        images, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(images).permute(0, 2, 1)
        total_loss += criterion(outputs, labels).item()
        c, n = active_accuracy(outputs.argmax(dim=1), labels)
        correct += c
        counted += n
    return total_loss / max(len(loader), 1), (100.0 * correct / counted if counted else 0.0)


def save_weights(model: torch.nn.Module, path: PathLike) -> None:
    """Save the state_dict with every tensor on the CPU, so it loads on any machine."""
    state = model.state_dict()
    for key in state:
        state[key] = state[key].detach().cpu()
    torch.save(state, path)


def train_stage(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                criterion, optimizer, scheduler, device: torch.device, epochs: int, patience: int,
                save_path: PathLike, name: str = "train") -> Dict[str, Any]:
    """Train for up to `epochs` epochs, keeping the weights with the lowest validation loss
    in `save_path`, and stop after `patience` epochs without improvement (0 = never).

    `scheduler` may be None, ReduceLROnPlateau (stepped on the validation loss) or any
    per-epoch scheduler. On Ctrl-C the stage ends cleanly and keeps the best checkpoint.

    Returns the history: per-epoch lists train_loss, val_loss, val_acc (active accuracy,
    %), lr and seconds, plus best_epoch, best_val_loss and stopped ("max_epochs",
    "early_stop" or "interrupted"). The model is left with its last-epoch weights.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    history: Dict[str, Any] = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": [], "seconds": [],
                               "best_epoch": None, "best_val_loss": None, "stopped": "max_epochs"}
    plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    best = math.inf
    stale = 0
    stop_rule = f"early stop after {patience} without improvement" if patience else "no early stop"
    print(f"{name}: up to {epochs} epochs, {stop_rule}, "
          f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val images")

    for epoch in range(1, epochs + 1):
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        try:
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                         desc=f"{name} {epoch}/{epochs}")
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        except KeyboardInterrupt:
            print(f"{name}: interrupted at epoch {epoch}; the best checkpoint so far is kept")
            history["stopped"] = "interrupted"
            break

        # ReduceLROnPlateau lost its verbose= argument in recent torch; the change is printed below.
        if plateau:
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        improved = val_loss < best
        if improved:
            best, stale = val_loss, 0
            save_weights(model, save_path)
            history["best_epoch"], history["best_val_loss"] = epoch, val_loss
        else:
            stale += 1

        seconds = time.time() - start
        for key, value in (("train_loss", train_loss), ("val_loss", val_loss), ("val_acc", val_acc),
                           ("lr", lr), ("seconds", seconds)):
            history[key].append(value)
        note = "saved" if improved else f"no improvement {stale}" + (f"/{patience}" if patience else "")
        print(f"{name} {epoch}/{epochs}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
              f"active acc {val_acc:.2f}%  lr {lr:.1e}  {seconds:.1f}s  ({note})")
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != lr:
            print(f"{name}: learning rate {lr:.1e} -> {new_lr:.1e}")

        if patience and stale >= patience:
            history["stopped"] = "early_stop"
            print(f"{name}: early stop, best epoch {history['best_epoch']} (val loss {best:.4f})"
                  if history["best_epoch"] else f"{name}: early stop")
            break

    if history["best_epoch"] is None:
        print(f"{name}: warning: no epoch ended with a finite validation loss, no checkpoint saved")
    return history


def save_curves(history: Dict[str, Any], path: PathLike, title: str, dpi: int = 150) -> None:
    """Loss (train and val) and validation active accuracy per epoch, side by side."""
    # Figure without pyplot: works headless and leaves the caller's backend alone.
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    n = len(history["train_loss"])
    epochs = range(1, n + 1)
    marker = "o" if n < 30 else None   # a line of one or two points is invisible
    fig = Figure(figsize=(12, 4.5))
    ax_loss, ax_acc = fig.subplots(1, 2)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax_loss.plot(epochs, history["train_loss"], label="train", color="tab:blue", lw=2, marker=marker, ms=3)
    ax_loss.plot(epochs, history["val_loss"], label="validation", color="tab:red", lw=2, marker=marker, ms=3)
    ax_loss.set(title="Focal loss", xlabel="epoch", ylabel="loss")
    ax_acc.plot(epochs, history["val_acc"], label="validation", color="tab:green", lw=2, marker=marker, ms=3)
    ax_acc.set(title="Active accuracy (exact count where a card is present)", xlabel="epoch",
               ylabel="accuracy (%)")
    if history.get("best_epoch"):
        for ax in (ax_loss, ax_acc):
            ax.axvline(history["best_epoch"], color="grey", ls=":", lw=1.5,
                       label=f"best (epoch {history['best_epoch']})")
    for ax in (ax_loss, ax_acc):
        # Whole epochs only; min_n_ticks=1 so a 1-epoch run does not fall back to 0.945, 0.96...
        ax.set_xlim(0.5, max(n, 1) + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        ax.grid(True, ls="--", alpha=0.6)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")


def save_history(path: PathLike, config: Optional[Dict[str, Any]] = None, **stages: Dict[str, Any]) -> None:
    """Write {"config": config, "stages": {name: history, ...}} as JSON."""
    with open(path, "w") as f:
        json.dump({"config": config or {}, "stages": stages}, f, indent=2, default=str)
        f.write("\n")
