import math

import torch

from .rope import rope


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


def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute causal multi-head self-attention without positional embeddings."""
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if x.shape[-1] != d_model:
        raise ValueError(f"Expected the final dimension of x to be {d_model}, got {x.shape[-1]}")

    d_head = d_model // num_heads
    seq_len = x.shape[-2]

    Q = x @ q_proj_weight.T
    K = x @ k_proj_weight.T
    V = x @ v_proj_weight.T

    Q = Q.reshape(*Q.shape[:-1], num_heads, d_head).transpose(-3, -2)
    K = K.reshape(*K.shape[:-1], num_heads, d_head).transpose(-3, -2)
    V = V.reshape(*V.shape[:-1], num_heads, d_head).transpose(-3, -2)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
    attention_output = scaled_dot_product_attention(Q=Q, K=K, V=V, mask=causal_mask)

    attention_output = attention_output.transpose(-3, -2).contiguous()
    attention_output = attention_output.reshape(*attention_output.shape[:-2], d_model)
    return attention_output @ o_proj_weight.T


def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    x: torch.Tensor,
    token_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute causal multi-head self-attention with rotary positional embeddings."""
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if x.shape[-1] != d_model:
        raise ValueError(f"Expected the final dimension of x to be {d_model}, got {x.shape[-1]}")

    d_head = d_model // num_heads
    seq_len = x.shape[-2]
    if token_positions is None:
        token_positions = torch.arange(seq_len, device=x.device)

    Q = (x @ q_proj_weight.T).reshape(*x.shape[:-1], num_heads, d_head).transpose(-3, -2)
    K = (x @ k_proj_weight.T).reshape(*x.shape[:-1], num_heads, d_head).transpose(-3, -2)
    V = (x @ v_proj_weight.T).reshape(*x.shape[:-1], num_heads, d_head).transpose(-3, -2)

    Q = rope(d_k=d_head, theta=theta, max_seq_len=max_seq_len, x=Q, token_positions=token_positions)
    K = rope(d_k=d_head, theta=theta, max_seq_len=max_seq_len, x=K, token_positions=token_positions)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
    attention_output = scaled_dot_product_attention(Q=Q, K=K, V=V, mask=causal_mask)
    attention_output = attention_output.transpose(-3, -2).contiguous()
    attention_output = attention_output.reshape(*attention_output.shape[:-2], d_model)
    return attention_output @ o_proj_weight.T
