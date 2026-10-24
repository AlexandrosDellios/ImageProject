"""Repository paths. Everything resolves from the repo root, so scripts run from any folder."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"                # gitignored: course data and everything derived from it
RAW = DATA / "raw"                  # train_images/, test_images/, train.csv (from the course)
PROCESSED = DATA / "processed"      # zone crops and the real-photo train/val split
SYNTHETIC = DATA / "synthetic"      # generated training sets

ASSETS = ROOT / "assets"            # card cut-outs and table backgrounds used for synthesis
RUNS = ROOT / "runs"                # training and evaluation outputs (gitignored)
FIGURES = ROOT / "docs" / "figures"

WEIGHTS = ROOT / "weights" / "uno_resnet18_cbam.pth"   # the submitted model
