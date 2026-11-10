"""Definizione dei modelli e factory `build_model`.

Implementazione CNN custom (baseline didattica) e modelli pretrained usati per
il confronto sperimentale, senza specializzare il loop di training.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, Swin_T_Weights, resnet50, swin_t


def vgg_block(in_ch: int, out_ch: int, n_convs: int = 2) -> nn.Sequential:
    """Blocco VGG-style: n_convs x (Conv 3x3 -> BatchNorm -> ReLU) + MaxPool.

    Impilare piu' convoluzioni prima del pooling aumenta la capacita' sulle
    texture e sugli artefatti locali senza rendere la baseline troppo complessa.
    """
    layers: list[nn.Module] = []
    for i in range(n_convs):
        layers += [
            nn.Conv2d(
                in_ch if i == 0 else out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
    layers.append(nn.MaxPool2d(kernel_size=2))
    return nn.Sequential(*layers)


class CustomCNN(nn.Module):
    """CNN VGG-style (baseline real/fake).

    Architettura (input 3x224x224):
        blocco 1: 3   -> 32   (x2 conv) | 224 -> 112
        blocco 2: 32  -> 64   (x2 conv) | 112 -> 56
        blocco 3: 64  -> 128  (x2 conv) | 56  -> 28
        blocco 4: 128 -> 256  (x2 conv) | 28  -> 14
        blocco 5: 256 -> 512  (x1 conv) | 14  -> 7

        AdaptiveAvgPool -> 512 -> Dropout -> Linear(512, hidden_dim)
        -> ReLU -> Dropout -> Linear(hidden_dim, num_classes).

    Resta una baseline custom semplice, ma piu' capace della versione con una
    sola convoluzione per blocco.
    """

    BLOCKS: tuple[tuple[int, int], ...] = (
        (32, 2), (64, 2), (128, 2), (256, 2), (512, 1),
    )

    def __init__(self, num_classes: int = 2, dropout: float = 0.3,
                 dropout_head: float = 0.4, hidden_dim: int = 128) -> None:
        super().__init__()

        # --- Estrattore di feature: sequenza di blocchi convoluzionali ---
        blocks = []
        in_ch = 3  # immagine RGB
        for out_ch, n_convs in self.BLOCKS:
            blocks.append(vgg_block(in_ch, out_ch, n_convs))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)      # *blocks: unpack della lista in argomenti posizionali

        # --- Testa di classificazione ---
        self.pool = nn.AdaptiveAvgPool2d(1)          # -> (512, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                            # -> (512)
            nn.Dropout(dropout_head),                # regolarizzazione testa
            nn.Linear(in_ch, hidden_dim),            # 512 -> hidden_dim
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),      # -> (B, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_resnet50(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
) -> nn.Module:
    """ResNet-50 con testa finale adattata alla classificazione real/fake."""
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")

    return model


def build_swin_tiny(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Swin-Tiny con testa finale adattata alla classificazione real/fake."""
    weights = Swin_T_Weights.DEFAULT if pretrained else None
    model = swin_t(weights=weights)

    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("head.")

    return model


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Factory: costruisce il modello a partire dalla sezione `model` del config.

    Il campo `model.name` seleziona l'architettura. Aggiungere qui i nuovi modelli
    (efficientnet_b0) in M3/M4.
    """
    model_cfg = cfg["model"]
    name = model_cfg["name"]
    num_classes = model_cfg.get("num_classes", 2)

    if name == "cnn_custom":
        return CustomCNN(
            num_classes=num_classes,
            dropout=model_cfg.get("dropout", 0.3),
            dropout_head=model_cfg.get("dropout_head", 0.4),
            hidden_dim=model_cfg.get("hidden_dim", 128),
        )
    if name == "resnet50":
        return build_resnet50(
            num_classes=num_classes,
            pretrained=model_cfg.get("pretrained", True),
            dropout=model_cfg.get("dropout", 0.2),
            freeze_backbone=model_cfg.get("freeze_backbone", False),
        )
    if name in {"swin_tiny", "swin_t"}:
        return build_swin_tiny(
            num_classes=num_classes,
            pretrained=model_cfg.get("pretrained", True),
            dropout=model_cfg.get("dropout", 0.2),
            freeze_backbone=model_cfg.get("freeze_backbone", False),
        )

    raise NotImplementedError(
        f"Modello '{name}' non ancora implementato "
        f"Disponibili al momento: ['cnn_custom', 'resnet50', 'swin_tiny']."
    )


if __name__ == "__main__":
    # Ispezione rapida: architettura, n. parametri e shape di output su input fittizio.
    cfg = {"model": {"name": "cnn_custom", "num_classes": 2, "dropout": 0.3}}
    model = build_model(cfg)
    dummy = torch.randn(2, 3, 224, 224)  # batch di 2 immagini RGB 224x224
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"\nParametri allenabili: {n_params:,}")
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}  (logit per [nature, ai])")
