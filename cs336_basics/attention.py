import math

import torch


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention over the final two dimensions."""
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    attention_weights = torch.softmax(scores, dim=-1)
    return attention_weights @ V
