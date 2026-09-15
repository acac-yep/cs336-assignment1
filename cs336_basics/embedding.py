import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # 第 i 行是 token i 对应的可学习向量。
        self.weight = nn.Parameter(torch.empty((vocab_size, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        # 高级索引会保留 token_ids 的所有维度。
        return self.weight[token_ids]
