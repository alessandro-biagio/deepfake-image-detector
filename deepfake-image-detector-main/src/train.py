"""Loop di training config-driven (generico sul modello).

Esempio (locale, CPU, smoke test sulle fixtures):
    python -m src.train --config configs/cnn_custom.yaml --data-root tests/fixtures \
        --epochs 1 --num-workers 0 --batch-size 8

Esempio (VM, GPU):
    python -m src.train --config configs/cnn_custom.yaml --data-root data/genimage_subset

Ogni run scrive in experiments/<experiment.name>/: copia del config, train.log,
metrics.csv (una riga per epoca), best.pt, last.pt e i log TensorBoard.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src import utils
from src.datasets import build_dataloader
from src.metrics import compute_metrics, format_metrics
from src.models import build_model
from src.transforms import build_eval_transforms, build_train_transforms

# Mappa il nome di `train.monitor` (es. val_f1) alla chiave delle metriche + direzione.
MONITOR_MAP = {
    "val_loss": ("loss", "min"),
    "val_acc": ("accuracy", "max"),
    "val_precision": ("precision", "max"),
    "val_recall": ("recall", "max"),
    "val_f1": ("f1", "max"),
    "val_auc": ("roc_auc", "max"),
}


# -----------------------------------------------------------------------------
# Inferenza (riusata anche da evaluate.py)
# -----------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model: nn.Module, loader, device: torch.device,
                        criterion: nn.Module | None = None) -> dict[str, Any]:
    """Esegue il modello su un loader e raccoglie label vere, predette e probabilita'.

    Ritorna dict con y_true, y_pred, y_prob (prob. classe `ai`) e loss media (se
    `criterion` e' fornito). Usato sia in validazione sia in evaluate.py.
    """
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    loss_meter = utils.AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if criterion is not None:
            loss_meter.update(criterion(logits, labels).item(), images.size(0))
        probs = torch.softmax(logits, dim=1)[:, 1]  # prob. della classe `ai`
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_prob.extend(probs.cpu().tolist())

    return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
            "loss": loss_meter.avg}


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    use_amp: bool) -> float:
    """Una passata di training. Ritorna la loss media dell'epoca."""
    model.train()
    loss_meter = utils.AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


# -----------------------------------------------------------------------------
# Costruzione optimizer / scheduler
# -----------------------------------------------------------------------------

def _trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def _finetuning_param_groups(model: nn.Module, lr: float, backbone_lr_mult: float):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("fc.", "head.")):
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr * backbone_lr_mult})
    if head_params:
        groups.append({"params": head_params, "lr": lr})
    return groups or _trainable_params(model)


def build_optimizer(model: nn.Module, train_cfg: dict[str, Any],
                    model_cfg: dict[str, Any] | None = None):
    name = train_cfg.get("optimizer", "adamw").lower()
    lr = train_cfg["lr"]
    wd = train_cfg.get("weight_decay", 0.0)
    model_cfg = model_cfg or {}
    backbone_lr_mult = model_cfg.get("backbone_lr_mult", 1.0)
    if model_cfg.get("name") in {"resnet50", "swin_tiny", "swin_t"} and backbone_lr_mult != 1.0:
        params = _finetuning_param_groups(model, lr, backbone_lr_mult)
    else:
        params = _trainable_params(model)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Optimizer non supportato: {name}")


def build_scheduler(optimizer, train_cfg: dict[str, Any]):
    if train_cfg.get("scheduler", "none").lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"])
    return None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Training GenImage detector.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--resume", default=None, help="checkpoint da cui riprendere")
    ap.add_argument("--out-root", default="experiments")
    # override opzionali (utili per smoke test e dev)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers

    data_root = Path(args.data_root)
    train_cfg, data_cfg = cfg["train"], cfg["data"]
    utils.set_seed(cfg["experiment"].get("seed", 42))

    # --- cartella della run ---
    exp_dir = utils.ensure_dir(Path(args.out_root) / cfg["experiment"]["name"])
    utils.save_config(cfg, exp_dir / "config.yaml")
    logger = utils.setup_logging(exp_dir, name=cfg["experiment"]["name"])
    writer = SummaryWriter(log_dir=str(exp_dir / "tb"))

    device = utils.get_device()
    use_amp = train_cfg.get("amp", True) and device.type == "cuda"
    logger.info(f"Device: {utils.describe_device(device)} | AMP: {use_amp}")

    # --- dati ---
    workers = data_cfg.get("num_workers", 0)
    train_loader = build_dataloader(
        data_root / data_cfg["train_csv"], data_root, build_train_transforms(cfg),
        train_cfg["batch_size"], shuffle=True, num_workers=workers)
    val_loader = build_dataloader(
        data_root / data_cfg["val_csv"], data_root, build_eval_transforms(cfg),
        train_cfg["batch_size"], shuffle=False, num_workers=workers)
    logger.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # --- modello / loss / optimizer ---
    model = build_model(cfg).to(device)
    logger.info(f"Modello '{cfg['model']['name']}' | "
                f"parametri allenabili: {utils.count_parameters(model):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, train_cfg, cfg.get("model", {}))
    scheduler = build_scheduler(optimizer, train_cfg)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    metric_key, mode = MONITOR_MAP[train_cfg.get("monitor", "val_f1")]
    best_value = float("-inf") if mode == "max" else float("inf")
    patience = train_cfg.get("early_stopping_patience", 5)
    epochs_no_improve = 0
    start_epoch = 1

    # --- resume ---
    if args.resume:
        ckpt = utils.load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        if scheduler and ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_value = ckpt.get("best_value", best_value)
        logger.info(f"Resume da epoca {start_epoch} (best {metric_key}={best_value:.4f})")

    # --- CSV delle metriche ---
    csv_path = exp_dir / "metrics.csv"
    csv_fields = ["epoch", "lr", "train_loss", "val_loss", "val_acc",
                  "val_precision", "val_recall", "val_f1", "val_auc"]
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(csv_fields)

    # --- loop ---
    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     scaler, device, use_amp)
        preds = collect_predictions(model, val_loader, device, criterion)
        val_metrics = compute_metrics(preds["y_true"], preds["y_pred"], preds["y_prob"])
        val_loss = preds["loss"]
        lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        logger.info(f"Epoca {epoch}/{train_cfg['epochs']} | lr={lr:.2e} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"{format_metrics(val_metrics)}")

        # log su CSV
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, lr, train_loss, val_loss, val_metrics["accuracy"],
                val_metrics["precision"], val_metrics["recall"],
                val_metrics["f1"], val_metrics["roc_auc"]])
        # log su TensorBoard
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        # checkpoint
        current = val_loss if metric_key == "loss" else val_metrics[metric_key]
        improved = current < best_value if mode == "min" else current > best_value
        state = {
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "best_value": best_value, "config": cfg,
        }
        utils.save_checkpoint(state, exp_dir / "last.pt")
        if improved:
            best_value = current
            state["best_value"] = best_value
            utils.save_checkpoint(state, exp_dir / "best.pt")
            epochs_no_improve = 0
            logger.info(f"  nuovo best ({metric_key}={best_value:.4f}) -> best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping (nessun miglioramento da {patience} epoche).")
                break

    writer.close()
    logger.info(f"Training concluso. Best {metric_key}={best_value:.4f} | output in {exp_dir}")


if __name__ == "__main__":
    main()
