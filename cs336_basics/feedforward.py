import torch
import torch.nn as nn

from .activation import silu
from .linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU 门控逐元素控制 value 分支。
        gate = silu(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)
