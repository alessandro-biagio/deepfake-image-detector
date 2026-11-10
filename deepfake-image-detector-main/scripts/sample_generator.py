"""Campiona un subset bilanciato da UN generatore GenImage estratto e lo copia
nel dataset incrementale persistente sulla VM (risoluzione nativa, nessun resize).

GenImage estratto ha struttura:  <src>/train/{ai,nature}/  <src>/val/{ai,nature}/
Questo script ricopia SOLO train/ e val/ (mirror fedele): la divisione del test
dalla val e la composizione degli esperimenti sono compito di build_splits.py.

Output (copiato) in  <dst>/imagenet_<gen>/ :
    train/{ai,nature}/   --train-per-class immagini per classe
    val/{ai,nature}/     --val-per-class immagini per classe

Il dataset cresce in modo incrementale: si lancia una volta per generatore.
Dopo aver verificato la copia si possono cancellare gli archivi grezzi.

Esempio (sulla VM):
    python scripts/sample_generator.py \
        --src /scratch/imagenet_ai_0508_adm --dst data/genimage_subset \
        --train-per-class 5000 --val-per-class 2000
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from tqdm import tqdm

CLASSES = ("nature", "ai")  # label 0, 1
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def derive_name(src: Path) -> str:
    """Da 'imagenet_ai_0508_adm' ricava il nome breve 'adm'."""
    return src.name.split("_")[-1]


def find_split_root(src: Path) -> Path:
    """Ritorna la cartella che contiene train/ (gestisce un eventuale annidamento)."""
    if (src / "train").is_dir() or (src / "val").is_dir():
        return src
    subdirs = [p for p in src.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and ((subdirs[0] / "train").is_dir() or (subdirs[0] / "val").is_dir()):
        return subdirs[0]
    raise SystemExit(f"{src}: non trovo train/ o val/ (struttura inattesa).")


def list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def copy_many(files: list[Path], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in tqdm(files, desc=f"-> {dst_dir.parent.name}/{dst_dir.name}", leave=False):
        shutil.copy2(f, dst_dir / f.name)


def write_manifest(path: Path, rows: list[tuple[str, str, str, int]]) -> None:
    """Aggiunge righe (generator, split, classe, count) al manifest del subset."""
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["generator", "split", "class", "count"])
        writer.writerows(rows)


def sample_split(src_root: Path, out: Path, split: str, per_class: int,
                 rng: random.Random, name: str) -> list[tuple[str, str, str, int]]:
    """Copia `per_class` immagini per classe da uno split. Ritorna righe di manifest."""
    rows: list[tuple[str, str, str, int]] = []
    for cls in CLASSES:
        imgs = list_images(src_root / split / cls)
        rng.shuffle(imgs)
        sel = imgs[:per_class]
        if len(sel) < per_class:
            print(f"  ! {split}/{cls}: disponibili solo {len(sel)} (richieste {per_class})")
        copy_many(sel, out / split / cls)
        rows.append((name, split, cls, len(sel)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="cartella del generatore estratto (contiene train/ e val/)")
    ap.add_argument("--dst", type=Path, default=Path("data/genimage_subset"),
                    help="radice del dataset incrementale persistente")
    ap.add_argument("--generator", default=None,
                    help="nome breve (es. adm); default: dedotto da --src")
    ap.add_argument("--train-per-class", type=int, default=5000,
                    help="immagini per classe dal train (ai e nature separatamente)")
    ap.add_argument("--val-per-class", type=int, default=2000,
                    help="immagini per classe dalla val (ai e nature separatamente)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true",
                    help="ricopia anche se il generatore e' gia' presente in --dst")
    args = ap.parse_args()

    src_root = find_split_root(args.src)
    name = args.generator or derive_name(args.src)
    out = args.dst / f"imagenet_{name}"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} esiste gia'. Usa --overwrite per rifarlo.")

    rng = random.Random(args.seed)
    manifest: list[tuple[str, str, str, int]] = []
    manifest += sample_split(src_root, out, "train", args.train_per_class, rng, name)
    manifest += sample_split(src_root, out, "val", args.val_per_class, rng, name)

    write_manifest(args.dst / "manifest.csv", manifest)

    counts = {f"{s}/{c}": n for (_, s, c, n) in manifest}
    print(f"Generatore '{name}' campionato -> {out}")
    print("  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"  manifest aggiornato: {args.dst / 'manifest.csv'}")


if __name__ == "__main__":
    main()
