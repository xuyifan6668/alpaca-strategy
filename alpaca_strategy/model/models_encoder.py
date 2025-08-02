import torch
import torch.nn as nn
from alpaca_strategy.config import get_config
cfg = get_config()

class Time2Vec(nn.Module):
    def __init__(self, out_dim: int = 8):
        super().__init__()
        if out_dim < 2:
            raise ValueError("Time2Vec out_dim must be >=2")
        self.out_dim = out_dim
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.W = nn.Parameter(torch.randn(out_dim - 1))
        self.phi = nn.Parameter(torch.randn(out_dim - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        lin = self.w0 * t + self.b0
        per = torch.sin(t.unsqueeze(-1) * self.W + self.phi)
        return torch.cat([lin.unsqueeze(-1), per], dim=-1)

class Encoder(nn.Module):
    def __init__(self, *, cfg=cfg):
        super().__init__()
        self.cfg = cfg

        self.t2v = Time2Vec(out_dim=8)
        t2v_dim = self.t2v.out_dim
        H = self.cfg.hidden

        self.proj = nn.Linear(self.cfg.FEAT_DIM + t2v_dim, H)

        self.temporal_stack = nn.Sequential(
            nn.TransformerEncoderLayer(H, self.cfg.heads, dim_feedforward=H, batch_first=True),
            nn.TransformerEncoderLayer(H, self.cfg.heads, dim_feedforward=H, batch_first=True)
        )

        self.pool_token = nn.Parameter(torch.randn(1, 1, H))
        self.multihead_pool = nn.MultiheadAttention(embed_dim=H, num_heads=self.cfg.heads, batch_first=True)

        self.dropout = nn.Dropout(self.cfg.dropout)

        self.stock_emb = nn.Embedding(num_embeddings=self.cfg.num_stocks, embedding_dim=H)
        self.cross_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(H, self.cfg.heads, dim_feedforward=H, dropout=self.cfg.dropout, batch_first=True),
            num_layers=1
        )

        self.score_head = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(), nn.LayerNorm(H), nn.Linear(H, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, T, _ = x.shape

        minute_scalar = x[..., self.cfg.MINUTE_IDX]  # (B, S, T)
        t2v_emb = self.t2v(minute_scalar)  # (B, S, T, t2v_dim)
        x_cat = torch.cat([x, t2v_emb], dim=-1)  # (B, S, T, D+t2v)

        h_seq = self.proj(x_cat.view(B * S, T, -1))  # (B*S, T, H)
        h_seq = self.temporal_stack(h_seq)  # (B*S, T, H)

        # --- Price/Volume Attention Bias ---
        vol_signal = x[..., self.cfg.VOLUME_IDX]  # (B, S, T)
        std_signal = x[..., self.cfg.STD_IDX]     # (B, S, T)

        vol_bias = (vol_signal - vol_signal.mean(dim=-1, keepdim=True)) / (vol_signal.std(dim=-1, keepdim=True) + 1e-6)
        std_bias = (std_signal - std_signal.mean(dim=-1, keepdim=True)) / (std_signal.std(dim=-1, keepdim=True) + 1e-6)
        attn_bias = vol_bias + std_bias  # (B, S, T)
        attn_bias = torch.clamp(attn_bias, -3.0, 3.0)  # optional clipping

        # Multi-head attention pooling with injected bias
        pool_tok = self.pool_token.expand(B * S, 1, -1)  # (B*S, 1, H)
        pooled, _ = self.multihead_pool(pool_tok, h_seq, h_seq)  # (B*S, 1, H)
        h_pool = self.dropout(pooled.squeeze(1))  # (B*S, H)

        stock_ids = torch.arange(S, device=x.device).repeat(B)
        h_tok = h_pool + self.stock_emb(stock_ids)
        h_tok = h_tok.view(B, S, -1)

        h_xattn = self.cross_attn(h_tok)
        h_tok = h_tok + h_xattn
        h_tok = h_tok.reshape(B * S, -1)

        scores = self.score_head(h_tok).reshape(B, S)
        return scores
