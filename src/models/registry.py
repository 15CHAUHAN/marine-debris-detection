"""Central registry mapping variant names to model builder functions.

All training/evaluation/dashboard entry points resolve architectures through
this single registry so a new variant only needs registration + a checkpoint.
Each builder accepts ``pretrained`` for parity across variants.
"""
from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from src.models.unet import build_unet
from src.models.unet_attention import build_attention_unet
from src.models.unet_cnn import build_cnn_unet
from src.models.unet_effnet_transformer import build_effnet_transformer_unet
from src.models.yolo_vit_lstm import build_yolo_vit_lstm
from src.models.yolo_bilstm import build_yolo_bilstm
from src.models.weight_init import get_seco_encoder, load_seco_checkpoint

Builder = Callable[..., nn.Module]

VARIANTS: dict[str, tuple[Builder, str]] = {
    "baseline": (lambda pretrained=True, weight_init_strategy="physics_split": build_unet(pretrained=pretrained, weight_init_strategy=weight_init_strategy), "U-Net (ResNet-34)"),
    "attention": (lambda pretrained=True, weight_init_strategy="physics_split": build_attention_unet(pretrained=pretrained, weight_init_strategy=weight_init_strategy), "U-Net + scSE attention"),
    "cnn_unet": (lambda pretrained=True, weight_init_strategy="random": build_cnn_unet(), "CNN + U-Net (from-scratch)"),
    "effnet_transformer": (
        lambda pretrained=True, weight_init_strategy="physics_split": build_effnet_transformer_unet(pretrained=pretrained, weight_init_strategy=weight_init_strategy),
        "EfficientNet-B0 + Transformer",
    ),
    "yolo_vit_lstm": (
        lambda pretrained=True, weight_init_strategy="physics_split": build_yolo_vit_lstm(pretrained=pretrained, weight_init_strategy=weight_init_strategy),
        "YOLO + ViT + BiGRU",
    ),
    "yolo_bilstm": (
        lambda pretrained=True, weight_init_strategy="physics_split": build_yolo_bilstm(pretrained=pretrained, weight_init_strategy=weight_init_strategy),
        "YOLO + BiLSTM",
    ),
}

RUNTIME_ALIASES: dict[str, str] = {
    "baseline_gan_aug_run_final": "baseline",
    "attention_gan_augmented": "attention",
}

# SeCo/SSL4EO pretrained encoder names (for future direct support)
SECO_ENCODERS = {"resnet50", "resnet18"}


def resolve_variant(variant: str, pretrained: bool = True, weight_init_strategy: str = "physics_split") -> nn.Module:
    """Instantiate a model for the given variant name.

    Accepts either a registered architecture key or a runtime run-dir name
    (e.g. ``attention_gan_augmented``) that aliases one via RUNTIME_ALIASES.

    Args:
        variant: Architecture key or runtime alias
        pretrained: Whether to use ImageNet pretrained weights
        weight_init_strategy: "physics_split" (default, CLI: imagenet_physics), "imagenet_replicate", "random", or "seco_s2"
    """
    key = RUNTIME_ALIASES.get(variant, variant)
    if key not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; available: {sorted(VARIANTS)}")

    builder = VARIANTS[key][0]

    # Map CLI-friendly names to internal strategy names
    strategy_map = {
        "imagenet_physics": "physics_split",
        "imagenet_replicate": "imagenet_replicate",
        "random": "random",
        "seco_s2": "seco_s2",
        "physics_split": "physics_split",
    }
    internal_strategy = strategy_map.get(weight_init_strategy, "physics_split")

    # Handle SeCo/SSL4EO-S12 backbone request
    if internal_strategy == "seco_s2":
        # Try to load SeCo encoder for the base architecture
        # For now, fall back to physics_split with ImageNet weights
        # TODO: Implement actual SeCo checkpoint loading
        print("SeCo Sentinel-2 weights requested; falling back to physics_split with ImageNet init")
        internal_strategy = "physics_split"

    return builder(pretrained=pretrained, weight_init_strategy=internal_strategy)


def variant_label(variant: str) -> str:
    """Human-readable label for a variant used in plots and the dashboard."""
    return VARIANTS[variant][1]