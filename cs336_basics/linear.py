import math
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self,  in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Store weights as (out_features, in_features), like nn.Linear.
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        with torch.no_grad():
            # Xavier-style truncated normal initialization.
            sigma = math.sqrt(2 / (in_features + out_features))
            nn.init.trunc_normal_(
                self.weight,
                mean=0.0,
                std=sigma,
                a=-3 * sigma,
                b=3 * sigma
            )

    def forward(self, x):
        # Apply the linear map while supporting arbitrary leading dimensions.
        return x @ self.weight.T
