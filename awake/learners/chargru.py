"""Byte-level GRU language model. forward() takes a chunk of byte ids and the carried
hidden state, returns logits for every position, the final hidden state, and the hidden
states over the chunk (as the 'activations' continual backprop scores)."""
from __future__ import annotations
import math
import mlx.core as mx
import mlx.nn as nn


class CharGRU(nn.Module):
    def __init__(self, vocab: int = 256, embed: int = 64, hidden: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed)
        self.gru = nn.GRU(embed, hidden)
        self.out = nn.Linear(hidden, vocab)
        self.hidden = hidden

    def forward(self, idx, h0=None):
        hs = self.gru(self.emb(idx), hidden=h0)      # (L, H) or (B, L, H)
        return self.out(hs), hs[..., -1, :], [hs]

    def unit_specs(self):
        H = self.hidden
        return [{
            "n": H,
            "act": 0,
            # gate rows for unit i are i, i+H, i+2H in Wx / Wh / b; bhn has one row per unit
            "in_params": [("gru.Wx", 0, (H, 3)), ("gru.Wh", 0, (H, 3)), ("gru.b", 0, (H, 3)), ("gru.bhn", 0, None)],
            "out_params": [("gru.Wh", 1, None), ("out.weight", 1, None)],
            "init_scale": 1.0 / math.sqrt(H),
        }]
