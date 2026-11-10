"""Utility condivise: riproducibilita', device, config, logging, checkpoint.

Questo modulo non dipende da nessun altro modulo del progetto, cosi' puo' essere
importato ovunque (datasets, train, evaluate, optuna_search) senza cicli.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# -----------------------------------------------------------------------------
# Riproducibilita'
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Fissa i seed di random/numpy/torch per esperimenti riproducibili.

    Args:
        seed: valore del seed.
        deterministic: se True forza algoritmi cuDNN deterministici (piu' lento,
            ma utile per confronti rigorosi). In training normale lasciare False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True velocizza quando le shape degli input sono costanti
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Inizializza il seed di ogni worker del DataLoader (uso: worker_init_fn)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------

def get_device(prefer_cuda: bool = True) -> torch.device:
    """Ritorna cuda se disponibile (VM con T4), altrimenti cpu (locale)."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Stringa descrittiva del device per il logging."""
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        return f"CUDA:{idx} ({name}, {total:.1f} GB)"
    return "CPU"


# -----------------------------------------------------------------------------
# Config (YAML)
# -----------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    """Carica un file di configurazione YAML in un dizionario."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} non e' un mapping YAML valido.")
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Salva una copia del config nella cartella della run (riproducibilita')."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def setup_logging(log_dir: str | Path | None = None,
                  name: str = "genimage",
                  level: int = logging.INFO) -> logging.Logger:
    """Configura un logger su console e (opzionale) su file `train.log`."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # evita duplicazioni se chiamato piu' volte

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# -----------------------------------------------------------------------------
# Checkpoint
# -----------------------------------------------------------------------------

def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    """Salva un checkpoint (model/optimizer/scaler/epoch/metric)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path,
                    map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Carica un checkpoint salvato con `save_checkpoint`."""
    return torch.load(Path(path), map_location=map_location)


# -----------------------------------------------------------------------------
# Tracker per la media mobile (loss/accuracy durante l'epoca)
# -----------------------------------------------------------------------------

@dataclass
class AverageMeter:
    """Tiene la media corrente di una metrica (es. loss) durante l'epoca."""

    total: float = 0.0
    count: int = 0
    history: list[float] = field(default_factory=list)

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n
        self.history.append(value)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


# -----------------------------------------------------------------------------
# Helpers vari
# -----------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> int:
    """Numero di parametri allenabili del modello."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: str | Path) -> Path:
    """Crea la directory se non esiste e la ritorna come Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
