"""Test M1: pipeline dati (build_splits -> Dataset -> DataLoader) sulle fixtures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from src import utils
from src.datasets import build_dataloader
from src.transforms import build_eval_transforms, build_train_transforms

FIX = Path("tests/fixtures")


def test_dataset_and_transforms():
    """Un batch esce con shape [B,3,224,224] e label binarie."""
    cfg = utils.load_config("configs/cnn_custom.yaml")
    loader = build_dataloader(FIX / "splits" / "val.csv", FIX,
                              build_eval_transforms(cfg), batch_size=4)
    images, labels = next(iter(loader))
    assert images.shape[0] == 4
    assert tuple(images.shape[1:]) == (3, 224, 224)
    assert set(labels.tolist()) <= {0, 1}


def test_train_transforms_shape():
    cfg = utils.load_config("configs/cnn_custom.yaml")
    loader = build_dataloader(FIX / "splits" / "train.csv", FIX,
                              build_train_transforms(cfg), batch_size=2, shuffle=True)
    images, _ = next(iter(loader))
    assert tuple(images.shape[1:]) == (3, 224, 224)


def test_build_splits_mixed(tmp_path):
    """build_splits genera CSV bilanciati e senza overlap train/val."""
    res = subprocess.run(
        [sys.executable, "scripts/build_splits.py", "--data-root", str(FIX),
         "--name", "pytest_mixed", "--mode", "mixed",
         "--train-per-class", "4", "--val-per-class", "3", "--test-per-class", "3"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr

    base = FIX / "splits" / "pytest_mixed"
    train = pd.read_csv(base / "train.csv")
    val = pd.read_csv(base / "val.csv")
    assert {"path", "label", "generator"} <= set(train.columns)
    assert set(train["label"]) <= {0, 1}
    # nessuna immagine di train compare in validation
    assert set(train["path"]) & set(val["path"]) == set()


def test_sample_generator(tmp_path):
    """sample_generator copia un subset bilanciato train/val da un generatore finto."""
    # costruisce un finto generatore estratto: imagenet_ai_9999_test/{train,val}/{ai,nature}
    src = tmp_path / "imagenet_ai_9999_demo"
    for split in ("train", "val"):
        for cls in ("ai", "nature"):
            d = src / split / cls
            d.mkdir(parents=True)
            for i in range(6):
                (d / f"{cls}_{i}.png").write_bytes(b"x")

    dst = tmp_path / "subset"
    res = subprocess.run(
        [sys.executable, "scripts/sample_generator.py", "--src", str(src),
         "--dst", str(dst), "--train-per-class", "4", "--val-per-class", "3"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr

    gen = dst / "imagenet_demo"
    assert len(list((gen / "train" / "ai").iterdir())) == 4
    assert len(list((gen / "train" / "nature").iterdir())) == 4
    assert len(list((gen / "val" / "ai").iterdir())) == 3
    assert not (gen / "test").exists()  # il test lo crea build_splits, non sample_generator
    assert (dst / "manifest.csv").exists()


def test_build_splits_cross_gen():
    """In cross-gen viene prodotto un test_<gen>.csv per l'held-out."""
    res = subprocess.run(
        [sys.executable, "scripts/build_splits.py", "--data-root", str(FIX),
         "--name", "pytest_crossgen", "--mode", "cross-gen",
         "--pool", "sdv4", "--heldout", "biggan",
         "--train-per-class", "4", "--val-per-class", "2", "--test-per-class", "2"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    heldout_csv = FIX / "splits" / "pytest_crossgen" / "test_biggan.csv"
    assert heldout_csv.exists()
    df = pd.read_csv(heldout_csv)
    assert (df["generator"] == "biggan").all()
