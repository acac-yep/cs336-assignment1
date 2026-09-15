import math
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self,  in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 权重形状为 (out_features, in_features)，与 nn.Linear 一致。
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        with torch.no_grad():
            # 使用类似 Xavier 的截断正态分布初始化。
            sigma = math.sqrt(2 / (in_features + out_features))
            nn.init.trunc_normal_(
                self.weight,
                mean=0.0,
                std=sigma,
                a=-3 * sigma,
                b=3 * sigma
            )

    def forward(self, x):
        # 支持任意前导维度的线性变换。
        return x @ self.weight.T
