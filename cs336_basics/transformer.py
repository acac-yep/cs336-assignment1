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
