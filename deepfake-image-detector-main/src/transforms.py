"""Transform di train (augmentation) e di eval, con normalizzazione ImageNet.

Il resize a `image_size` avviene qui a runtime: le immagini su disco restano alla
risoluzione nativa. mean/std sono letti dal config (default ImageNet).
"""

from __future__ import annotations

from typing import Any

from torchvision import transforms

from src.degradations import build_degradation


def _normalize(cfg: dict[str, Any]) -> transforms.Normalize:
    return transforms.Normalize(mean=cfg["data"]["mean"], std=cfg["data"]["std"])


def build_train_transforms(cfg: dict[str, Any]) -> transforms.Compose:
    """Augmentation di training: crop casuale + flip + normalizzazione."""
    size = cfg["data"]["image_size"]
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _normalize(cfg),
    ])


def build_eval_transforms(cfg: dict[str, Any],
                          degradation: str = "clean") -> transforms.Compose:
    """Transform deterministico per val/test, con degradazione opzionale post-crop."""
    size = cfg["data"]["image_size"]
    resize = int(round(size * 256 / 224))  # margine standard prima del center crop
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(size),
        build_degradation(degradation),
        transforms.ToTensor(),
        _normalize(cfg),
    ])
