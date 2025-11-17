import torch
from torch import nn

from anitune.lora import LoRALinear


def test_lora_linear_adds_residual():
    base = nn.Linear(8, 4)
    layer = LoRALinear.from_linear(base, rank=2, alpha=4)
    x = torch.randn(3, 8)
    out = layer(x)
    assert out.shape == (3, 4)
    # ensure gradients only flow through LoRA params
    for name, param in layer.named_parameters():
        if name.startswith("lora_"):
            assert param.requires_grad
        else:
            assert not param.requires_grad
