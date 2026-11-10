"""Genera i CSV di split (train/val/test) dalla base GenImage scaricata.

NON copia immagini: scrive solo percorsi relativi a --data-root. Si puo' lanciare
piu' volte con parametri diversi per generare dataset diversi a costo ~zero.

Anti-leakage: train esce da train/, mentre val e test escono da val/ (porzioni
disgiunte) -> nessuna immagine di training compare in validation o test.

Modalita':
  mixed      campiona da tutti i generatori (baseline real/fake misto)
  cross-gen  train/val solo dal --pool; un test_<gen>.csv per ogni held-out

Esempi:
  python scripts/build_splits.py --data-root data --name baseline_misto --mode mixed
  python scripts/build_splits.py --data-root data --name crossgen_sdv4 \
      --mode cross-gen --pool sdv4 --heldout adm biggan glide
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

CLASSES = {"nature": 0, "ai": 1}  # label dal nome cartella
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def gen_dirs(data_root: Path) -> dict[str, Path]:
    """Mappa nome-generatore -> cartella (es. 'sdv4' -> data/imagenet_sdv4)."""
    return {
        p.name.replace("imagenet_", ""): p
        for p in sorted(data_root.iterdir())
        if p.is_dir() and p.name.startswith("imagenet_")
    }


def list_images(gen_dir: Path, split: str, cls: str) -> list[Path]:
    d = gen_dir / split / cls
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def collect(gen_dir: Path, split: str, per_class: int,
            rng: random.Random) -> list[tuple[Path, int, str]]:
    """Righe bilanciate (path, label, generator) campionate da uno split."""
    name = gen_dir.name.replace("imagenet_", "")
    rows: list[tuple[Path, int, str]] = []
    for cls, label in CLASSES.items():
        imgs = list_images(gen_dir, split, cls)
        rng.shuffle(imgs)
        rows += [(p, label, name) for p in imgs[:per_class]]
    return rows


def split_val(gen_dir: Path, val_pc: int, test_pc: int, rng: random.Random
              ) -> tuple[list[tuple[Path, int, str]], list[tuple[Path, int, str]]]:
    """Divide val/ in due porzioni disgiunte: validation e test in-distribution."""
    name = gen_dir.name.replace("imagenet_", "")
    val_rows, test_rows = [], []
    for cls, label in CLASSES.items():
        imgs = list_images(gen_dir, "val", cls)
        rng.shuffle(imgs)
        val_rows += [(p, label, name) for p in imgs[:val_pc]]
        test_rows += [(p, label, name) for p in imgs[val_pc:val_pc + test_pc]]
    return val_rows, test_rows


def write_csv(path: Path, rows: list[tuple[Path, int, str]], data_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "generator"])
        for p, label, gen in rows:
            writer.writerow([Path(p).relative_to(data_root).as_posix(), label, gen])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--name", required=True, help="nome dello split (cartella di output)")
    ap.add_argument("--mode", choices=["mixed", "cross-gen"], default="mixed")
    ap.add_argument("--pool", nargs="*", default=None,
                    help="generatori di training (cross-gen); default: tutti per mixed")
    ap.add_argument("--heldout", nargs="*", default=None,
                    help="generatori held-out per il test (default: tutti i non-pool)")
    ap.add_argument("--train-per-class", type=int, default=5000)
    ap.add_argument("--val-per-class", type=int, default=1000)
    ap.add_argument("--test-per-class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    available = gen_dirs(args.data_root)
    if not available:
        raise SystemExit(f"Nessun 'imagenet_*' trovato in {args.data_root}")

    pool = args.pool or list(available)
    if args.mode == "cross-gen" and not args.pool:
        raise SystemExit("--mode cross-gen richiede --pool")
    missing = [g for g in pool if g not in available]
    if missing:
        raise SystemExit(f"Generatori del pool non trovati: {missing}")

    out_dir = args.data_root / "splits" / args.name
    train_rows, val_rows, test_rows = [], [], []
    for g in pool:
        train_rows += collect(available[g], "train", args.train_per_class, rng)
        v, t = split_val(available[g], args.val_per_class, args.test_per_class, rng)
        val_rows += v
        test_rows += t

    write_csv(out_dir / "train.csv", train_rows, args.data_root)
    write_csv(out_dir / "val.csv", val_rows, args.data_root)
    write_csv(out_dir / "test_in_distribution.csv", test_rows, args.data_root)

    summary = [f"train={len(train_rows)}", f"val={len(val_rows)}",
               f"test_id={len(test_rows)}"]

    if args.mode == "cross-gen":
        heldout = args.heldout or [g for g in available if g not in pool]
        for g in heldout:
            if g not in available:
                print(f"  ! held-out '{g}' non trovato, salto")
                continue
            rows = collect(available[g], "val", args.test_per_class, rng)
            write_csv(out_dir / f"test_{g}.csv", rows, args.data_root)
            summary.append(f"test_{g}={len(rows)}")

    print(f"Split '{args.name}' ({args.mode}) -> {out_dir}")
    print("  pool:", pool)
    print("  " + " ".join(summary))


if __name__ == "__main__":
    main()
