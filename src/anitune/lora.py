import torch
from torch import nn
from typing import Iterable


class LoRALinear(nn.Module):
    """Wrap an existing Linear layer with a low-rank adapter.

    Base weights are frozen; only the low-rank matrices train. Scaling follows
    the LoRA paper convention (alpha / rank).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: int = 1, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.rank = rank
        self.scale = alpha / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.dropout(self.lora_A(x))) * self.scale
        return base_out + lora_out

    @classmethod
    def from_linear(cls, layer: nn.Linear, rank: int, alpha: int = 1, dropout: float = 0.0):
        return cls(layer, rank=rank, alpha=alpha, dropout=dropout)


def apply_lora_to_attention(
    model: nn.Module,
    target_modules: Iterable[str] = ("qkv", "proj"),
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> int:
    """Replace attention linear layers with LoRA-wrapped counterparts.

    Args:
        model: Vision Transformer model (e.g., timm ViT).
        target_modules: module names inside attention blocks to wrap.
        rank: LoRA rank.
        alpha: LoRA scaling parameter.
        dropout: optional dropout before low-rank projection.

    Returns:
        Number of modules wrapped.
    """

    replaced = 0
    for module in model.modules():
        if module.__class__.__name__.lower() in {"attention", "multiheadattention"}:
            for name in target_modules:
                if hasattr(module, name):
                    attn_linear = getattr(module, name)
                    if isinstance(attn_linear, nn.Linear):
                        setattr(
                            module,
                            name,
                            LoRALinear.from_linear(attn_linear, rank=rank, alpha=alpha, dropout=dropout),
                        )
                        replaced += 1
    if replaced == 0:
        raise RuntimeError("No attention linear layers were wrapped; check model architecture.")
    return replaced
