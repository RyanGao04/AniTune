from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import timm

from .lora import apply_lora_to_attention


@dataclass
class ModelConfig:
    name: str = "vit_base_patch16_224"
    num_classes: int = 5013
    pretrained: bool = True
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    img_size: int | None = None


def build_model(cfg: ModelConfig) -> nn.Module:
    extra = {}
    if cfg.img_size:
        extra["img_size"] = cfg.img_size
    try:
        model = timm.create_model(cfg.name, pretrained=cfg.pretrained, num_classes=cfg.num_classes, **extra)
    except TypeError:
        # Fallback if img_size unsupported by the model
        model = timm.create_model(cfg.name, pretrained=cfg.pretrained, num_classes=cfg.num_classes)
    if cfg.use_lora:
        apply_lora_to_attention(
            model,
            rank=cfg.lora_rank,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
        )
    return model


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_backbone(model: nn.Module, unfreeze_head: bool = True):
    for name, param in model.named_parameters():
        param.requires_grad = False
    if unfreeze_head:
        for name, param in model.named_parameters():
            if any(k in name for k in ["head", "fc", "classifier"]):
                param.requires_grad = True


def enable_full_finetune(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True
