"""Model components: Time2Vec and Encoder.

This module is purposely lightweight; it only depends on the project-wide
configuration in `config.py`.  No PyTorch-Lightning code lives here so the
model can be imported and unit-tested without the heavy trainer stack.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Note: we keep the global *cfg* import for a default but allow callers to
# inject their own configuration for easier testing and hyper-parameter
# sweeps.

from utils.config import cfg as _global_cfg, FEAT_DIM, MINUTE_IDX

# ---------------------------------------------------------------------------
# 1. Time2Vec positional encoder
# ---------------------------------------------------------------------------

class Time2Vec(nn.Module):
    """Kazemi et al., 2019 — learnable periodic representation for scalar t."""

    def __init__(self, out_dim: int = 8):
        super().__init__()
        if out_dim < 2:
            raise ValueError("Time2Vec out_dim must be >=2 so that we have at least one periodic term")
        self.out_dim = out_dim

        # Linear component
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))

        # Periodic components
        self.W = nn.Parameter(torch.randn(out_dim - 1))
        self.phi = nn.Parameter(torch.randn(out_dim - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t (...,)
        lin = self.w0 * t + self.b0                       # (...,)
        per = torch.sin(t.unsqueeze(-1) * self.W + self.phi)  # (..., out_dim-1)
        return torch.cat([lin.unsqueeze(-1), per], dim=-1)

# ---------------------------------------------------------------------------
# 2. Main Encoder used by the trading model
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Temporal + cross-sectional encoder that can operate under any compatible
    configuration object.

    Parameters
    ----------
    cfg : Config, optional
        If not supplied, falls back to the project's default global `cfg`.
    """

    def __init__(self, *, cfg=_global_cfg):
        super().__init__()

        self.cfg = cfg  # store local reference

        self.t2v = Time2Vec(out_dim=8)
        t2v_dim = self.t2v.out_dim

        H = self.cfg.hidden
        self.proj = nn.Linear(FEAT_DIM + t2v_dim, H)

        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(H, self.cfg.heads, dim_feedforward=H,
                                       dropout=self.cfg.dropout, batch_first=True),
            num_layers=2,
        )
        self.lstm = nn.LSTM(H, H // 2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.time_attn = nn.Linear(self.cfg.hidden, 1)

        self.stock_emb = nn.Embedding(num_embeddings=self.cfg.num_stocks,
                                       embedding_dim=self.cfg.hidden)
        self.cross_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.cfg.hidden, self.cfg.heads,
                                       dim_feedforward=self.cfg.hidden,
                                       dropout=self.cfg.dropout, batch_first=True),
            num_layers=1,
        )
        self.score_head = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(), nn.Dropout(0.1), nn.Linear(H, 1)
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, S, T, D)
        B, S, T, _ = x.shape

        minute_scalar = x[..., MINUTE_IDX]  # (B, S, T)
        t2v_emb = self.t2v(minute_scalar)   # (B, S, T, t2v_dim)

        x_cat = torch.cat([x, t2v_emb], dim=-1)  # (B, S, T, D+t2v)
        h_seq = self.proj(x_cat.view(B * S, T, -1))
        h_seq = self.tr(h_seq)
        h_seq, _ = self.lstm(h_seq)

        # Temporal attention pooling -----------------------------------
        attn_w = torch.softmax(self.time_attn(h_seq).squeeze(-1), dim=1)
        h_pool = (attn_w.unsqueeze(-1) * h_seq).sum(dim=1)
        h_pool = self.dropout(h_pool)

        stock_ids = torch.arange(S, device=x.device).repeat(B)
        h_tok = h_pool + self.stock_emb(stock_ids)

        h_tok = h_tok.view(B, S, -1)
        h_tok = self.cross_attn(h_tok)
        h_tok = h_tok.view(B * S, -1)

        scores = self.score_head(h_tok).view(B, S)
        return scores 