"""Genera un dataset GenImage FINTO e minuscolo per smoke test su CPU in locale.

Riproduce la struttura reale di GenImage (cartelle ai/ e nature/) con poche
immagini PNG casuali, e i CSV di split, cosi' l'intera pipeline (datasets ->
train -> evaluate) puo' girare in locale senza scaricare nulla.

Uso:
    python scripts/make_fixtures.py
    python scripts/make_fixtures.py --root tests/fixtures --per-class 16

Struttura generata sotto <root>:
    <root>/
      imagenet_sdv4/{train,val}/{ai,nature}/*.png   # generatore "in pool"
      imagenet_biggan/val/{ai,nature}/*.png         # generatore held-out (cross-gen)
      splits/train.csv val.csv test_in_distribution.csv test_biggan.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

# Generatori finti: uno "in pool" (train+val) e uno held-out (solo val/test)
POOL_GENERATORS = ["imagenet_sdv4"]
HELDOUT_GENERATORS = ["imagenet_biggan"]
CLASSES = {"nature": 0, "ai": 1}  # label per nome cartella


def _make_image(path: Path, size: int, fake: bool, rng: np.random.Generator) -> None:
    """Crea un'immagine RGB casuale. Le 'fake' hanno una texture leggermente
    diversa cosi' un modello puo' (in teoria) imparare a separarle nello smoke test."""
    if fake:
        # pattern piu' 'liscio' per le fake (blur sintetico via low-freq noise)
        base = rng.integers(0, 256, size=(size // 4, size // 4, 3), dtype=np.uint8)
        img = np.array(Image.fromarray(base).resize((size, size), Image.BILINEAR))
    else:
        img = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def _populate(gen_dir: Path, splits: list[str], per_class: int,
              size: int, rng: np.random.Generator) -> list[tuple[str, int, str]]:
    """Crea le immagini di un generatore e ritorna le righe (path, label, generator)."""
    rows: list[tuple[str, int, str]] = []
    generator = gen_dir.name
    for split in splits:
        for cls, label in CLASSES.items():
            out_dir = gen_dir / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                img_path = out_dir / f"{split}_{cls}_{i:03d}.png"
                _make_image(img_path, size, fake=(label == 1), rng=rng)
                rows.append((str(img_path.as_posix()), label, generator))
    return rows


def _write_csv(path: Path, rows: list[tuple[str, int, str]], root: Path) -> None:
    """Scrive un CSV di split con path RELATIVI a root (come build_splits.py)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "generator"])
        for abs_path, label, generator in rows:
            rel = Path(abs_path).relative_to(root).as_posix()
            writer.writerow([rel, label, generator])


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera fixtures GenImage finte.")
    parser.add_argument("--root", default="tests/fixtures", type=Path,
                        help="cartella radice delle fixtures (default: tests/fixtures)")
    parser.add_argument("--per-class", default=12, type=int,
                        help="immagini per classe per split (default: 12)")
    parser.add_argument("--size", default=64, type=int,
                        help="lato immagine in px (piccolo per velocita', default: 64)")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    root: Path = args.root
    rng = np.random.default_rng(args.seed)
    root.mkdir(parents=True, exist_ok=True)

    pool_rows: list[tuple[str, int, str]] = []
    for gen in POOL_GENERATORS:
        pool_rows += _populate(root / gen, ["train", "val"], args.per_class,
                               args.size, rng)

    heldout_rows: dict[str, list[tuple[str, int, str]]] = {}
    for gen in HELDOUT_GENERATORS:
        heldout_rows[gen] = _populate(root / gen, ["val"], args.per_class,
                                      args.size, rng)

    # Split dal pool: train da train/, val+test da val/
    train_rows = [r for r in pool_rows if "/train/" in r[0]]
    val_pool = [r for r in pool_rows if "/val/" in r[0]]
    mid = len(val_pool) // 2
    val_rows, test_id_rows = val_pool[:mid], val_pool[mid:]

    splits_dir = root / "splits"
    _write_csv(splits_dir / "train.csv", train_rows, root)
    _write_csv(splits_dir / "val.csv", val_rows, root)
    _write_csv(splits_dir / "test_in_distribution.csv", test_id_rows, root)
    for gen, rows in heldout_rows.items():
        name = gen.replace("imagenet_", "")
        _write_csv(splits_dir / f"test_{name}.csv", rows, root)

    total = len(train_rows) + len(val_rows) + len(test_id_rows) + \
        sum(len(r) for r in heldout_rows.values())
    print(f"Fixtures create in {root}/")
    print(f"  train={len(train_rows)} val={len(val_rows)} "
          f"test_id={len(test_id_rows)} "
          f"held-out={ {g: len(r) for g, r in heldout_rows.items()} }")
    print(f"  totale immagini: {total}")
    print(f"  CSV in: {splits_dir}/")


if __name__ == "__main__":
    main()
