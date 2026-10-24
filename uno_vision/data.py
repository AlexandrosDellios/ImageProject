"""Datasets: one image per row of a `filename,cards` CSV.

Every training set in this project (synthetic composites, real zone crops) uses the same
format: `cards` is a `;`-separated list such as `r_1;b_skip;yellow_token`, or `EMPTY`.
"""

from pathlib import Path
from typing import Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .classes import encode_cards
from .transforms import INPUT_SIZE


class UnoCSVDataset(Dataset):
    """Yields (image tensor, counts[56]) or, with return_path, (image, counts, path)."""

    def __init__(self, csv_file: Union[str, Path], img_dir: Union[str, Path], transform=None,
                 return_path: bool = False, limit: int = 0):
        self.frame = pd.read_csv(csv_file)
        if limit:
            self.frame = self.frame.head(limit)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        path = self.img_dir / str(row["filename"])
        try:
            image = Image.open(path).convert("RGB")
        except OSError as err:
            # A missing or broken file must not kill a long training run: train on black.
            print(f"warning: could not read {path} ({err}); using a black image")
            image = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))
        target = encode_cards(str(row["cards"]))
        if self.transform:
            image = self.transform(image)
        return (image, target, str(path)) if self.return_path else (image, target)
