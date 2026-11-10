"""Metriche di valutazione real/fake (basate su scikit-learn).

Convenzione: la classe positiva e' `ai` (label 1), quella negativa `nature` (label 0).
- y_true: label vere (0/1)
- y_pred: label predette (0/1), tipicamente argmax dei logit
- y_prob: probabilita' della classe `ai` (per la ROC-AUC), in [0, 1]

Sequence permette di esprimere sequenze di inter che siano tuple, array o liste
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)

CLASS_NAMES = ["nature", "ai"]  # indice = label


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                    y_prob: Sequence[float] | None = None) -> dict[str, float]:
    """Calcola accuracy/precision/recall/F1 (+ ROC-AUC se y_prob e' fornito).

    precision/recall/F1 sono riferiti alla classe positiva `ai` (zero_division=0
    evita warning quando una classe non e' mai predetta).
    """
    y_true = np.asarray(y_true)         # conversione input in array numpy (per sicurezza) 
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_parts = []
    if (tp + fn) > 0:
        balanced_parts.append(recall)
    if (tn + fp) > 0:
        balanced_parts.append(specificity)
    balanced_accuracy = float(np.mean(balanced_parts)) if balanced_parts else float("nan")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy,
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }

    # La ROC-AUC richiede le probabilita' ed entrambe le classi presenti in y_true.
    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, np.asarray(y_prob))
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    """Matrice di confusione 2x2 con ordine di righe/colonne [nature, ai]."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def save_confusion_matrix(cm: np.ndarray, path: str | Path,
                          title: str = "Confusion matrix") -> None:
    """Salva la matrice di confusione come PNG annotato."""
    import os
    import tempfile

    cache_root = Path(tempfile.gettempdir())
    xdg_cache = cache_root / "xdg-cache"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    mpl_cache = cache_root / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib
    matplotlib.use("Agg")  # backend senza display (per la VM / job SLURM)
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Vero")
    ax.set_title(title)
    # annota ogni cella col conteggio
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def format_metrics(metrics: dict[str, float]) -> str:
    """Riga leggibile per log/console."""
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())


if __name__ == "__main__":
    # Mini-demo su dati finti per vedere l'output.
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)
    y_prob = np.clip(y_true * 0.6 + rng.random(100) * 0.5, 0, 1)  # correlate col vero
    y_pred = (y_prob >= 0.5).astype(int)
    print(format_metrics(compute_metrics(y_true, y_pred, y_prob)))
    print("confusion:\n", confusion(y_true, y_pred))
