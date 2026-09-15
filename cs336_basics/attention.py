import math

import torch

from .rope import rope


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """在最后两个维度上计算缩放点积注意力。"""
    d_k = Q.shape[-1]
    # 计算 Query 和 Key 的相似度，结果形状为 (..., queries, keys)。
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        # mask 为 False 的位置不能被关注。
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    # 沿 Key 维度归一化，再对 Value 做加权求和。
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
    """计算不带位置编码的因果多头自注意力。"""
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if x.shape[-1] != d_model:
        raise ValueError(f"Expected the final dimension of x to be {d_model}, got {x.shape[-1]}")

    d_head = d_model // num_heads
    seq_len = x.shape[-2]

    # 用三个矩阵乘法把所有 token 投影成 Q、K、V。
    Q = x @ q_proj_weight.T
    K = x @ k_proj_weight.T
    V = x @ v_proj_weight.T

    # 形状从 (..., seq, d_model) 变为 (..., heads, seq, d_head)。
    Q = Q.reshape(*Q.shape[:-1], num_heads, d_head).transpose(-3, -2)
    K = K.reshape(*K.shape[:-1], num_heads, d_head).transpose(-3, -2)
    V = V.reshape(*V.shape[:-1], num_heads, d_head).transpose(-3, -2)

    # 因果注意力：每个 token 只能看到自己和之前的 token。
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
    attention_output = scaled_dot_product_attention(Q=Q, K=K, V=V, mask=causal_mask)

    # 形状从 (..., heads, seq, d_head) 变回 (..., seq, d_model)，再混合各个 head。
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
    """计算带旋转位置编码 RoPE 的因果多头自注意力。"""
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if x.shape[-1] != d_model:
        raise ValueError(f"Expected the final dimension of x to be {d_model}, got {x.shape[-1]}")

    d_head = d_model // num_heads
    seq_len = x.shape[-2]
    if token_positions is None:
        # 默认使用常规位置 0, 1, ..., seq_len - 1。
        token_positions = torch.arange(seq_len, device=x.device)

    # 先投影，再拆分成多个 head。
    Q = (x @ q_proj_weight.T).reshape(*x.shape[:-1], num_heads, d_head).transpose(-3, -2)
    K = (x @ k_proj_weight.T).reshape(*x.shape[:-1], num_heads, d_head).transpose(-3, -2)
    V = (x @ v_proj_weight.T).reshape(*x.shape[:-1], num_heads, d_head).transpose(-3, -2)

    # RoPE 只改变 Q、K 的位置相关表示，不处理 V。
    Q = rope(d_k=d_head, theta=theta, max_seq_len=max_seq_len, x=Q, token_positions=token_positions)
    K = rope(d_k=d_head, theta=theta, max_seq_len=max_seq_len, x=K, token_positions=token_positions)

    # 应用因果注意力，合并各个 head，最后进行输出投影。
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
    attention_output = scaled_dot_product_attention(Q=Q, K=K, V=V, mask=causal_mask)
    attention_output = attention_output.transpose(-3, -2).contiguous()
    attention_output = attention_output.reshape(*attention_output.shape[:-2], d_model)
    return attention_output @ o_proj_weight.T
