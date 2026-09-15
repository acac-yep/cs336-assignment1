import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Row i is the learned vector for token i.
        self.weight = nn.Parameter(torch.empty((vocab_size, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        # Advanced indexing preserves all token_ids dimensions.
        return self.weight[token_ids]
