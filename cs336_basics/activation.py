import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    """Apply the SiLU activation elementwise."""
    # SiLU(x) = x * sigmoid(x).
    return x * torch.sigmoid(x)
