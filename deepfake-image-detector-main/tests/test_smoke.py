"""Smoke test M0: verifica che le utility di base funzionino su CPU.

Esegui con:  pytest tests/test_smoke.py -v
Non richiede GPU ne' GenImage: usa solo torch su CPU.
"""

from __future__ import annotations

import torch

from src import utils
from src.models import build_model


def test_set_seed_reproducible():
    """Stesso seed -> stessa sequenza di numeri casuali."""
    utils.set_seed(123)
    a = torch.rand(5)
    utils.set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_get_device_returns_valid_device():
    """get_device ritorna cpu in locale (o cuda se presente)."""
    device = utils.get_device()
    assert device.type in {"cpu", "cuda"}
    # describe_device non deve sollevare eccezioni
    assert isinstance(utils.describe_device(device), str)


def test_config_roundtrip(tmp_path):
    """save_config + load_config preservano il contenuto."""
    cfg = {"model": {"name": "cnn_custom"}, "train": {"epochs": 3}}
    path = tmp_path / "cfg.yaml"
    utils.save_config(cfg, path)
    loaded = utils.load_config(path)
    assert loaded == cfg


def test_checkpoint_roundtrip(tmp_path):
    """save_checkpoint + load_checkpoint preservano i tensori."""
    state = {"epoch": 1, "weights": torch.ones(3)}
    path = tmp_path / "ckpt.pt"
    utils.save_checkpoint(state, path)
    loaded = utils.load_checkpoint(path)
    assert loaded["epoch"] == 1
    assert torch.equal(loaded["weights"], torch.ones(3))


def test_average_meter():
    """AverageMeter calcola la media corretta."""
    meter = utils.AverageMeter()
    meter.update(2.0, n=1)
    meter.update(4.0, n=3)  # media = (2 + 12) / 4 = 3.5
    assert meter.avg == 3.5


def test_build_resnet50_without_pretrained_weights():
    """ResNet-50 si costruisce senza scaricare pesi durante lo smoke test."""
    cfg = {
        "model": {
            "name": "resnet50",
            "num_classes": 2,
            "pretrained": False,
            "dropout": 0.2,
            "freeze_backbone": True,
        }
    }
    model = build_model(cfg)
    out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 2)
    assert all(
        (not param.requires_grad) or name.startswith("fc.")
        for name, param in model.named_parameters()
    )
