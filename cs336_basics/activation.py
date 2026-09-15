import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    """逐元素应用 SiLU 激活函数。"""
    # SiLU(x) = x * sigmoid(x)。
    return x * torch.sigmoid(x)
