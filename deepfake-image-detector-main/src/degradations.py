"""Degradazioni deterministiche per la valutazione di robustezza.

Le trasformazioni lavorano su immagini PIL dopo Resize e CenterCrop, ma prima di
ToTensor e normalizzazione. In questo modo ogni immagine viene degradata alla
stessa scala di eval e le intensita' restano confrontabili tra classi e
generatori.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from PIL import Image, ImageFilter


ImageTransform = Callable[[Image.Image], Image.Image]


@dataclass(frozen=True)
class DegradationSpec:
    """Descrizione di una degradazione disponibile."""

    name: str
    description: str
    transform: ImageTransform


class IdentityDegradation:
    """Nessuna degradazione: utile per avere una baseline nella stessa suite."""

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB") if image.mode != "RGB" else image


class JpegCompression:
    """Ricodifica JPEG con qualita' fissata."""

    def __init__(self, quality: int) -> None:
        if not 1 <= quality <= 100:
            raise ValueError("quality deve essere tra 1 e 100.")
        self.quality = quality

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality, subsampling=2)
        buffer.seek(0)
        degraded = Image.open(buffer).convert("RGB")
        return degraded.copy()


class DownUpResize:
    """Riduce la risoluzione e poi riporta l'immagine alla dimensione originale."""

    def __init__(self, scale: float = 0.5) -> None:
        if not 0 < scale < 1:
            raise ValueError("scale deve essere compreso tra 0 e 1.")
        self.scale = scale

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        down_size = (
            max(1, int(round(width * self.scale))),
            max(1, int(round(height * self.scale))),
        )
        small = image.resize(down_size, Image.Resampling.BILINEAR)
        return small.resize((width, height), Image.Resampling.BICUBIC)


class GaussianBlur:
    """Sfocatura gaussiana PIL con raggio fissato."""

    def __init__(self, radius: float = 1.0) -> None:
        if radius < 0:
            raise ValueError("radius deve essere >= 0.")
        self.radius = radius

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        return image.filter(ImageFilter.GaussianBlur(radius=self.radius))


class GaussianNoise:
    """Rumore gaussiano deterministico ricavato dal contenuto dell'immagine."""

    def __init__(self, std: float = 0.03, seed: int = 42) -> None:
        if std < 0:
            raise ValueError("std deve essere >= 0.")
        self.std = std
        self.seed = seed

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        arr = np.asarray(image).astype(np.float32) / 255.0
        digest = hashlib.blake2b(
            image.tobytes() + str(self.seed).encode("utf-8"),
            digest_size=8,
        ).digest()
        rng_seed = int.from_bytes(digest, "little", signed=False)
        rng = np.random.default_rng(rng_seed)
        noisy = np.clip(arr + rng.normal(0.0, self.std, arr.shape), 0.0, 1.0)
        return Image.fromarray((noisy * 255.0).round().astype(np.uint8)).convert("RGB")


class SequentialDegradation:
    """Composizione sequenziale di piu' degradazioni PIL."""

    def __init__(self, transforms: Iterable[ImageTransform]) -> None:
        self.transforms = tuple(transforms)

    def __call__(self, image: Image.Image) -> Image.Image:
        for transform in self.transforms:
            image = transform(image)
        return image


_REGISTRY: dict[str, DegradationSpec] = {
    "clean": DegradationSpec(
        "clean",
        "nessuna degradazione",
        IdentityDegradation(),
    ),
    "jpeg_q70": DegradationSpec(
        "jpeg_q70",
        "ricodifica JPEG quality=70",
        JpegCompression(quality=70),
    ),
    "jpeg_q50": DegradationSpec(
        "jpeg_q50",
        "ricodifica JPEG quality=50",
        JpegCompression(quality=50),
    ),
    "resize_down_up": DegradationSpec(
        "resize_down_up",
        "downscale 0.5x seguito da upscale alla dimensione originale",
        DownUpResize(scale=0.5),
    ),
    "blur": DegradationSpec(
        "blur",
        "Gaussian blur con raggio 1.0",
        GaussianBlur(radius=1.0),
    ),
    "noise": DegradationSpec(
        "noise",
        "rumore gaussiano deterministico std=0.03",
        GaussianNoise(std=0.03, seed=42),
    ),
    "combined_social": DegradationSpec(
        "combined_social",
        "downscale 0.75x, blur lieve e JPEG quality=70",
        SequentialDegradation([
            DownUpResize(scale=0.75),
            GaussianBlur(radius=0.4),
            JpegCompression(quality=70),
        ]),
    ),
}

BASIC_DEGRADATIONS = (
    "clean",
    "jpeg_q70",
    "jpeg_q50",
    "resize_down_up",
    "blur",
    "combined_social",
)

DEGRADATION_SUITES = {
    "basic": BASIC_DEGRADATIONS,
    "all": tuple(_REGISTRY),
}


def available_degradations() -> tuple[str, ...]:
    """Nomi delle degradazioni registrate."""
    return tuple(_REGISTRY)


def available_suites() -> tuple[str, ...]:
    """Nomi delle suite di degradazioni disponibili."""
    return tuple(DEGRADATION_SUITES)


def build_degradation(name: str) -> ImageTransform:
    """Ritorna la trasformazione PIL associata a `name`."""
    try:
        return _REGISTRY[name].transform
    except KeyError as exc:
        valid = ", ".join(available_degradations())
        raise ValueError(f"Degradazione non supportata: {name}. Valide: {valid}") from exc


def degradation_description(name: str) -> str:
    """Descrizione leggibile di una degradazione."""
    try:
        return _REGISTRY[name].description
    except KeyError as exc:
        valid = ", ".join(available_degradations())
        raise ValueError(f"Degradazione non supportata: {name}. Valide: {valid}") from exc


def expand_degradation_names(names: Iterable[str]) -> list[str]:
    """Espande nomi singoli e suite (`basic`, `all`, `suite:<nome>`).

    Mantiene l'ordine e rimuove duplicati, cosi' si possono combinare suite e
    degradazioni puntuali dalla CLI.
    """
    expanded: list[str] = []
    for raw_name in names:
        name = raw_name.removeprefix("suite:")
        if name in DEGRADATION_SUITES:
            candidates = DEGRADATION_SUITES[name]
        else:
            candidates = (raw_name,)

        for candidate in candidates:
            if candidate not in _REGISTRY:
                valid = ", ".join((*available_degradations(), *available_suites()))
                raise ValueError(
                    f"Degradazione o suite non supportata: {candidate}. Valide: {valid}"
                )
            if candidate not in expanded:
                expanded.append(candidate)
    return expanded
