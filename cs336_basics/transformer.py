import torch

from .RMSNorm import RMSNorm
from .attention import multihead_self_attention_with_rope
from .feedforward import SwiGLU


def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    """计算一个带 RoPE 的 Pre-Norm Transformer Block。"""
    norm1 = RMSNorm(d_model=d_model, eps=1e-5, device=x.device, dtype=x.dtype)
    norm2 = RMSNorm(d_model=d_model, eps=1e-5, device=x.device, dtype=x.dtype)
    ffn = SwiGLU(d_model=d_model, d_ff=d_ff).to(device=x.device, dtype=x.dtype)

    with torch.no_grad():
        norm1.weight.copy_(weights["ln1.weight"])
        norm2.weight.copy_(weights["ln2.weight"])
        ffn.w1.weight.copy_(weights["ffn.w1.weight"])
        ffn.w2.weight.copy_(weights["ffn.w2.weight"])
        ffn.w3.weight.copy_(weights["ffn.w3.weight"])

    # 先归一化，再计算带 RoPE 的因果多头注意力。
    attention_input = norm1(x)
    attention_output = multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        x=attention_input,
    )
    x = x + attention_output

    # 第二个子层同样采用 Pre-Norm，并通过残差连接回原表示。
    return x + ffn(norm2(x))


def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Run a decoder-only Transformer language model from a state dict."""
    if input_ids.ndim < 1:
        raise ValueError("input_ids must have at least one dimension")
    if input_ids.shape[-1] > context_length:
        raise ValueError("input sequence length cannot exceed context_length")
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    token_embeddings = weights["token_embeddings.weight"]
    if token_embeddings.shape != (vocab_size, d_model):
        raise ValueError("token_embeddings.weight has an unexpected shape")
    if input_ids.numel() > 0 and (
        input_ids.min() < 0 or input_ids.max() >= vocab_size
    ):
        raise ValueError("input_ids contain an out-of-range token id")

    x = token_embeddings[input_ids]
    for layer_idx in range(num_layers):
        layer_prefix = f"layers.{layer_idx}."
        layer_weights = {
            key.removeprefix(layer_prefix): value
            for key, value in weights.items()
            if key.startswith(layer_prefix)
        }
        x = transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            x=x,
        )

    final_norm = RMSNorm(
        d_model=d_model,
        eps=1e-5,
        device=x.device,
        dtype=x.dtype,
    )
    with torch.no_grad():
        final_norm.weight.copy_(weights["ln_final.weight"])
    x = final_norm(x)

    lm_head_weight = weights["lm_head.weight"]
    if lm_head_weight.shape != (vocab_size, d_model):
        raise ValueError("lm_head.weight has an unexpected shape")
    return x @ lm_head_weight.T
