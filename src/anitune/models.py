"""修复后的模型构建模块

提供4种训练模式的正确实现：
1. Head-only: 冻结backbone，仅训练分类头
2. Full fine-tuning: 训练所有参数（无LoRA）
3. LoRA-only: 冻结backbone，仅训练LoRA adapters + 分类头
4. LoRA + Full: LoRA adapters + 全部参数可训练（实验对比用）
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import timm


from .lora import apply_lora_to_attention, LoRALinear


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

    # 新增：训练模式
    train_mode: str = "lora_only"  # Options: head_only, full_ft, lora_only, lora_full


def _create_timm_model(name: str, num_classes: int, pretrained: bool, img_size: int | None):
    """创建 timm 模型"""
    extra = {}
    if img_size:
        extra["img_size"] = img_size
    try:
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, **extra)
    except TypeError:
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def count_parameters(model: nn.Module, verbose: bool = False):
    """统计参数数量

    Args:
        model: 模型
        verbose: 是否打印详细信息

    Returns:
        (total_params, trainable_params, breakdown_dict)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    breakdown = {}

    if verbose:
        # 统计不同部分的参数
        lora_params = 0
        head_params = 0
        backbone_params = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if 'lora_' in name:
                lora_params += param.numel()
            elif name.startswith('head') or name.startswith('fc') or name.startswith('classifier'):
                head_params += param.numel()
            else:
                backbone_params += param.numel()

        breakdown = {
            'lora': lora_params,
            'head': head_params,
            'backbone': backbone_params,
            'trainable': trainable,
            'total': total,
        }

    return total, trainable, breakdown


def freeze_all_except_head(model: nn.Module):
    """冻结除分类头外的所有参数

    用于 head-only 模式
    """
    # 首先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻分类头
    for name, param in model.named_parameters():
        if name.startswith('head') or name.startswith('fc') or name.startswith('classifier'):
            param.requires_grad = True
            print(f"  解冻: {name} ({param.numel():,} params)")


def freeze_backbone_keep_lora_and_head(model: nn.Module):
    """冻结backbone，保持LoRA和分类头可训练

    用于 lora_only 模式（真正的参数高效微调）
    """
    # 首先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻 LoRA 参数
    lora_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # LoRA 的 A 和 B 矩阵
            for param in module.lora_A.parameters():
                param.requires_grad = True
                lora_count += param.numel()
            for param in module.lora_B.parameters():
                param.requires_grad = True
                lora_count += param.numel()

    print(f"  解冻 LoRA 参数: {lora_count:,}")

    # 解冻分类头
    head_count = 0
    for name, param in model.named_parameters():
        if name.startswith('head') or name.startswith('fc') or name.startswith('classifier'):
            param.requires_grad = True
            head_count += param.numel()
            print(f"  解冻分类头: {name} ({param.numel():,} params)")

    print(f"  总可训练参数: {lora_count + head_count:,}")


def enable_full_finetune(model: nn.Module):
    """启用全量微调

    用于 full_ft 模式
    """
    for param in model.parameters():
        param.requires_grad = True


def build_model(cfg: ModelConfig) -> nn.Module:
    """构建模型并设置训练模式

    Args:
        cfg: 模型配置

    Returns:
        配置好的模型
    """
    print(f"\n{'='*60}")
    print(f"构建模型: {cfg.name}")
    print(f"训练模式: {cfg.train_mode}")
    print(f"{'='*60}\n")

    # 1. 创建基础模型
    try:
        model = _create_timm_model(cfg.name, cfg.num_classes, cfg.pretrained, cfg.img_size)
        print(f"✓ 加载预训练模型: {cfg.name}")
    except Exception as exc:
        if cfg.pretrained:
            print(f"⚠️  预训练权重加载失败 ({exc}), 使用随机初始化")
            model = _create_timm_model(cfg.name, cfg.num_classes, pretrained=False, img_size=cfg.img_size)
        else:
            raise

    # 2. 应用 LoRA（如果需要）
    if cfg.train_mode in ['lora_only', 'lora_full']:
        print(f"\n应用 LoRA:")
        print(f"  Rank: {cfg.lora_rank}")
        print(f"  Alpha: {cfg.lora_alpha}")
        print(f"  Dropout: {cfg.lora_dropout}")

        num_replaced = apply_lora_to_attention(
            model,
            rank=cfg.lora_rank,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
        )
        print(f"  ✓ 替换了 {num_replaced} 个 attention 层")

    # 3. 设置参数冻结状态
    print(f"\n设置训练模式: {cfg.train_mode}")

    if cfg.train_mode == "head_only":
        # 模式1: 仅训练分类头
        freeze_all_except_head(model)

    elif cfg.train_mode == "full_ft":
        # 模式2: 全量微调（无LoRA）
        enable_full_finetune(model)
        print(f"  ✓ 所有参数可训练")

    elif cfg.train_mode == "lora_only":
        # 模式3: LoRA-only（冻结backbone）
        freeze_backbone_keep_lora_and_head(model)

    elif cfg.train_mode == "lora_full":
        # 模式4: LoRA + 全量微调（实验对比用）
        enable_full_finetune(model)
        print(f"  ✓ LoRA参数 + backbone 全部可训练")

    else:
        raise ValueError(f"未知训练模式: {cfg.train_mode}")

    # 4. 统计参数
    print(f"\n参数统计:")
    total, trainable, breakdown = count_parameters(model, verbose=True)

    print(f"  总参数: {total:,} ({total/1e6:.2f}M)")
    print(f"  可训练参数: {trainable:,} ({trainable/1e6:.2f}M)")
    print(f"  可训练比例: {trainable/total*100:.2f}%")

    if breakdown:
        print(f"\n  详细分解:")
        if breakdown['lora'] > 0:
            print(f"    LoRA: {breakdown['lora']:,} ({breakdown['lora']/1e6:.2f}M)")
        if breakdown['head'] > 0:
            print(f"    Head: {breakdown['head']:,} ({breakdown['head']/1e6:.2f}M)")
        if breakdown['backbone'] > 0:
            print(f"    Backbone: {breakdown['backbone']:,} ({breakdown['backbone']/1e6:.2f}M)")

    print(f"\n{'='*60}\n")

    return model


# 兼容旧接口
freeze_backbone = freeze_all_except_head


if __name__ == "__main__":
    """测试不同训练模式的参数统计"""

    print("\n" + "="*80)
    print("测试不同训练模式的参数配置")
    print("="*80)

    modes = [
        ("head_only", False),
        ("full_ft", False),
        ("lora_only", True),
        ("lora_full", True),
    ]

    for mode, use_lora in modes:
        cfg = ModelConfig(
            name="vit_base_patch16_224",
            num_classes=5013,
            pretrained=False,  # 测试时不加载预训练权重
            train_mode=mode,
            use_lora=use_lora,
            lora_rank=8,
            lora_alpha=16,
        )

        model = build_model(cfg)

        print("\n" + "-"*80 + "\n")
