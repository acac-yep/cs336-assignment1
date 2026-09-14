import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    """Apply the SiLU activation elementwise."""
    return x * torch.sigmoid(x)
