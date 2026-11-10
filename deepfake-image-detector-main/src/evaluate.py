"""Valutazione di un checkpoint su uno split di test.

Carica un modello allenato, calcola le metriche (accuracy/precision/recall/F1/
ROC-AUC) su un CSV di test e salva metrics.json + confusion matrix PNG.

Esempio:
    python -m src.evaluate --config experiments/cnn_custom_baseline/config.yaml \
        --data-root data/genimage_subset \
        --checkpoint experiments/cnn_custom_baseline/best.pt \
        --split splits/baseline_misto/test_in_distribution.csv

Se --split e --checkpoint non sono dati, usa data.test_csv del config e
experiments/<name>/best.pt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src import utils
from src.degradations import (
    available_degradations,
    available_suites,
    degradation_description,
    expand_degradation_names,
)
from src.datasets import build_dataloader
from src.metrics import (
    compute_metrics,
    confusion,
    format_metrics,
    save_confusion_matrix,
)
from src.models import build_model
from src.train import collect_predictions
from src.transforms import build_eval_transforms


def infer_experiment_dir(config_path: str | Path, cfg: dict[str, Any],
                         out_root: str | Path = "experiments") -> Path:
    """Determina la cartella run associata a un config.

    Se il config e' gia' `experiments/<name>/config.yaml`, usa quella cartella.
    Altrimenti usa `out_root/<experiment.name>`, cioe' la convenzione di train.py.
    """
    path = Path(config_path)
    exp_name = cfg["experiment"]["name"]
    if path.name == "config.yaml" and path.parent.name == exp_name:
        return path.parent
    return Path(out_root) / exp_name


def default_eval_base_dir(exp_dir: Path, split_csv: Path) -> Path:
    """Cartella base coerente con le eval gia' usate nel progetto."""
    split_name = split_csv.stem
    if split_name == "test_in_distribution":
        return exp_dir / "eval_in_distribution"
    if split_name.startswith("test_"):
        return exp_dir / "eval_heldout"
    return exp_dir / f"eval_{split_name}"


def default_eval_out_dir(exp_dir: Path, split_csv: Path, degradation: str,
                         multi_degradation: bool = False) -> Path:
    """Cartella per un risultato di eval.

    Singola eval clean: `eval_in_distribution` o `eval_heldout`, come nel workflow esistente.
    Eval degradate: sottocartelle dentro la stessa base, es. `eval_heldout/jpeg_q50`.
    """
    base_dir = default_eval_base_dir(exp_dir, split_csv)
    if degradation == "clean" and not multi_degradation:
        return base_dir
    return base_dir / degradation


def load_model_from_checkpoint(cfg: dict[str, Any], checkpoint: str | Path,
                               device):
    """Costruisce il modello e carica i pesi dal checkpoint."""
    model = build_model(cfg).to(device)
    ckpt = utils.load_checkpoint(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model


def _save_predictions_csv(preds: dict[str, Any], loader, out_path: Path) -> None:
    """Salva predizioni per immagine mantenendo l'ordine del DataLoader."""
    import pandas as pd

    df = loader.dataset.df.copy()
    df["y_true"] = preds["y_true"]
    df["y_pred"] = preds["y_pred"]
    df["y_prob_ai"] = preds["y_prob"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def evaluate_split(model, cfg: dict[str, Any], data_root: str | Path,
                   split_csv: str | Path, out_dir: str | Path,
                   degradation: str = "clean", num_workers: int | None = None,
                   device=None, checkpoint: str | Path | None = None,
                   save_predictions: bool = False) -> dict[str, Any]:
    """Esegue inferenza, metriche e salvataggio artefatti per uno split."""
    data_root = Path(data_root)
    split_csv = Path(split_csv)
    out_dir = utils.ensure_dir(out_dir)
    if device is None:
        device = utils.get_device()
    workers = (
        num_workers
        if num_workers is not None
        else cfg["data"].get("num_workers", 0)
    )

    loader = build_dataloader(
        split_csv,
        data_root,
        build_eval_transforms(cfg, degradation=degradation),
        cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=workers,
    )

    preds = collect_predictions(model, loader, device)
    metrics = compute_metrics(preds["y_true"], preds["y_pred"], preds["y_prob"])
    cm = confusion(preds["y_true"], preds["y_pred"])

    metadata = {
        "experiment": cfg["experiment"]["name"],
        "model": cfg["model"]["name"],
        "split_csv": split_csv.as_posix(),
        "split_name": split_csv.stem,
        "degradation": degradation,
        "degradation_description": degradation_description(degradation),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "num_samples": len(preds["y_true"]),
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    save_confusion_matrix(
        cm,
        out_dir / "confusion_matrix.png",
        title=f"{cfg['experiment']['name']} | {split_csv.stem} | {degradation}",
    )
    if save_predictions:
        _save_predictions_csv(preds, loader, out_dir / "predictions.csv")

    return {"metrics": metrics, "metadata": metadata, "out_dir": out_dir}


def main() -> None:
    ap = argparse.ArgumentParser(description="Valutazione GenImage detector.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="default: experiments/<name>/best.pt")
    ap.add_argument("--split", default=None,
                    help="CSV di test relativo a --data-root; default: data.test_csv")
    ap.add_argument("--out-dir", default=None,
                    help="dove salvare i risultati; default: <run>/eval")
    ap.add_argument("--out-root", default="experiments",
                    help="root delle run se il config non e' dentro experiments/")
    ap.add_argument("--degradation", default=None,
                    choices=available_degradations(),
                    help="singola degradazione di eval; default: clean")
    ap.add_argument("--degradations", nargs="+", default=None,
                    help=(
                        "lista degradazioni o suite. Valide: "
                        f"{', '.join(available_degradations())}; suite: "
                        f"{', '.join(available_suites())}"
                    ))
    ap.add_argument("--save-predictions", action="store_true",
                    help="salva anche predictions.csv con y_true/y_pred/y_prob_ai")
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    data_root = Path(args.data_root)
    exp_dir = infer_experiment_dir(args.config, cfg, out_root=args.out_root)

    checkpoint = Path(args.checkpoint) if args.checkpoint else exp_dir / "best.pt"
    split_csv = data_root / (args.split or cfg["data"]["test_csv"])
    raw_degradations = args.degradations or [args.degradation or "clean"]
    degradations = expand_degradation_names(raw_degradations)
    multi_degradation = len(degradations) > 1
    workers = args.num_workers if args.num_workers is not None else cfg["data"].get("num_workers", 0)

    device = utils.get_device()
    print(f"Device: {utils.describe_device(device)}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Test split: {split_csv}")

    # --- modello + pesi allenati ---
    model = load_model_from_checkpoint(cfg, checkpoint, device)

    for degradation in degradations:
        if args.out_dir:
            out_dir = Path(args.out_dir) / degradation if multi_degradation else Path(args.out_dir)
        else:
            out_dir = default_eval_out_dir(exp_dir, split_csv, degradation, multi_degradation)

        result = evaluate_split(
            model,
            cfg,
            data_root,
            split_csv,
            out_dir,
            degradation=degradation,
            num_workers=workers,
            device=device,
            checkpoint=checkpoint,
            save_predictions=args.save_predictions,
        )

        print(f"\n[{degradation}] {format_metrics(result['metrics'])}")
        print(f"Risultati salvati in {result['out_dir']}/")


if __name__ == "__main__":
    main()
