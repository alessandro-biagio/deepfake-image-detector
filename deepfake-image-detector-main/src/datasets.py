"""Dataset PyTorch che legge un CSV di split (colonne: path, label, generator).

Il `path` e' relativo a `data_root`, cosi' lo stesso CSV funziona in locale
(tests/fixtures) e sulla VM (data/). La label e' gia' nel CSV (nature=0, ai=1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.utils import seed_worker


class GenImageDataset(Dataset):
    """Legge le immagini elencate in un CSV di split."""

    def __init__(self, csv_path: str | Path, data_root: str | Path,
                 transform: Callable | None = None) -> None:
        self.data_root = Path(data_root)
        self.df = pd.read_csv(csv_path)
        if not {"path", "label"}.issubset(self.df.columns):
            raise ValueError(f"{csv_path}: servono almeno le colonne 'path' e 'label'.")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        row = self.df.iloc[idx]
        image = Image.open(self.data_root / row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


def build_dataloader(csv_path: str | Path, data_root: str | Path,
                     transform: Callable, batch_size: int, *,
                     shuffle: bool = False, num_workers: int = 0,
                     seed: int = 42) -> DataLoader:
    """DataLoader riproducibile. pin_memory/persistent_workers attivi solo se utili."""
    dataset = GenImageDataset(csv_path, data_root, transform)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )
