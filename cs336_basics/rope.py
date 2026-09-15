import torch


def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    x: torch.Tensor,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embeddings to the final dimension of ``x``."""
    if d_k % 2 != 0:
        raise ValueError("RoPE requires an even embedding dimension")
    if x.shape[-1] != d_k:
        raise ValueError(f"Expected the final dimension of x to be {d_k}, got {x.shape[-1]}")
    if token_positions.shape[-1] != x.shape[-2]:
        raise ValueError("token_positions and x must have the same sequence length")
    if token_positions.numel() > 0 and (
        token_positions.min() < 0 or token_positions.max() >= max_seq_len
    ):
        raise ValueError("token_positions must be in [0, max_seq_len)")

    original_dtype = x.dtype
    x_float = x.float().reshape(*x.shape[:-1], d_k // 2, 2)

    frequencies = theta ** (
        -torch.arange(0, d_k, 2, device=x.device, dtype=torch.float32) / d_k
    )
    angles = token_positions.to(device=x.device, dtype=torch.float32).unsqueeze(-1) * frequencies
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    even = x_float[..., 0]
    odd = x_float[..., 1]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.reshape_as(x).to(original_dtype)
