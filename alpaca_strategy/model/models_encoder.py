import torch
import torch.nn as nn
import math
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

        H = self.cfg.hidden

        self.t2v = Time2Vec(out_dim=8)
        t2v_dim = self.t2v.out_dim

        self.proj = nn.Linear(self.cfg.FEAT_DIM + t2v_dim, H)

        self.temporal_stack = nn.Sequential(
            nn.TransformerEncoderLayer(
                d_model=H, nhead=cfg.heads, dim_feedforward=4*H,
                dropout=cfg.dropout, activation="gelu",
                batch_first=True, norm_first=True
            ),
            nn.TransformerEncoderLayer(
                d_model=H, nhead=cfg.heads, dim_feedforward=4*H,
                dropout=cfg.dropout, activation="gelu",
                batch_first=True, norm_first=True
            ),
        )

        
        self.pool_token = nn.Parameter(torch.randn(1, 1, H))
        # Simple Q/K/V projections for explicit biased attention
        self.q_proj = nn.Linear(H, H)
        self.k_proj = nn.Linear(H, H)
        self.v_proj = nn.Linear(H, H)
        self._raw_attn_bias_scale = nn.Parameter(torch.tensor(0.0))  # trainable
        self.max_attn_bias = 5.0 
        self.dropout = nn.Dropout(cfg.dropout)

        # EWMA coefficient for recency pooling
        self.register_buffer("ewma_decay", torch.tensor(0.03))

        # Pooling projection layer
        self.pool_proj = nn.Linear(3 * H, H)

        self.stock_emb = nn.Embedding(num_embeddings=cfg.num_stocks, embedding_dim=H)

        self.xsec = nn.Sequential(
            nn.TransformerEncoderLayer(
                d_model=H, nhead=cfg.heads, dim_feedforward=4*H,
                dropout=cfg.dropout, activation="gelu",
                batch_first=True, norm_first=True
            ),
            nn.TransformerEncoderLayer(
                d_model=H, nhead=cfg.heads, dim_feedforward=4*H,
                dropout=cfg.dropout, activation="gelu",
                batch_first=True, norm_first=True
            ),
        )

        # ---- Score head: LN -> 2H GELU -> Dropout -> 1 ----
        self.score_head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, 2*H),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2*H, 1),
        )
    @staticmethod
    def _ewma_pool(h_seq: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
        """
        h_seq: [B*S, T, H]; decay: scalar tensor
        Returns: [B*S, H]
        """
        BxS, T, H = h_seq.shape
        device = h_seq.device
        # Older timesteps get larger lag -> weight = exp(-decay * (T-1 - t))
        lags = torch.arange(T, device=device).float()
        # 0 at past, T-1 at latest; reverse lags so recent gets highest weight
        rev = (T - 1) - lags
        w = torch.exp(-decay * rev)                         # [T]
        w = w / (w.sum() + 1e-8)
        pooled = torch.einsum("t, bth -> bh", w, h_seq)     # [B*S, H]
        return pooled

    def forward(self, x: torch.Tensor, symbol_ids: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, S, T, D]
        symbol_ids: optional [B, S] long; real symbol indices for embedding.
                    If None, falls back to arange(S).
        """
        B, S, T, D = x.shape
        device = x.device
        H = self.cfg.hidden

        # ---- Time2Vec concat ----
        minute_scalar = x[..., self.cfg.MINUTE_IDX]               # [B,S,T]
        t2v_emb = self.t2v(minute_scalar)                         # [B,S,T,t2v]
        x_cat  = torch.cat([x, t2v_emb], dim=-1)                  # [B,S,T,D+t2v]

        # ---- Temporal encoding ----
        h_seq = self.proj(x_cat.view(B * S, T, -1))               # [B*S, T, H]
        h_seq = self.temporal_stack(h_seq)                        # [B*S, T, H]

        # ---- Build attention bias from volume/std (per-time additive bias) ----
        vol = x[..., self.cfg.VOLUME_IDX]                         # [B,S,T]
        std = x[..., self.cfg.STD_IDX]                            # [B,S,T]

        def zscore_lastdim(t):
            m = t.mean(dim=-1, keepdim=True)
            s = t.std(dim=-1, keepdim=True)
            return torch.where(s > 1e-8, (t - m) / s, torch.zeros_like(t))

        vol_bias = zscore_lastdim(vol)
        std_bias = zscore_lastdim(std)
        attn_bias = torch.clamp(vol_bias + std_bias, -3.0, 3.0)   # [B,S,T]
        attn_bias = attn_bias.view(B * S, T)                      # [B*S, T]

        # ---- Multi-summary time pooling ----
        # 1) Last state (strong recency signal)
        last = h_seq[:, -1, :]                                    # [B*S, H]

        # 2) EWMA pooled state (smooth recency)
        ewma = self._ewma_pool(h_seq, self.ewma_decay)            # [B*S, H]

        # 3) Biased attention pooling over time
        pool_tok = self.pool_token.expand(B * S, 1, H)            # [B*S, 1, H]
        q = self.q_proj(pool_tok)                                  # [B*S, 1, H]
        k = self.k_proj(h_seq)                                     # [B*S, T, H]
        v = self.v_proj(h_seq)                                     # [B*S, T, H]

        # scaled dot-product attention with additive bias on logits
        logits = torch.einsum("bqh,bkh->bqk", q, k) / math.sqrt(H) # [B*S, 1, T]
        attn_bias_scale = self.max_attn_bias * torch.sigmoid(self._raw_attn_bias_scale)

        logits = logits + attn_bias_scale * attn_bias.unsqueeze(1)  # broadcast
        attn = torch.softmax(logits, dim=-1)                       # [B*S, 1, T]
        attn_pooled = torch.einsum("bqk,bkh->bqh", attn, v).squeeze(1)  # [B*S, H]
        attn_pooled = self.dropout(attn_pooled)

        # concat summaries -> project back to H
        h_pool = torch.cat([last, ewma, attn_pooled], dim=-1)      # [B*S, 3H]
        h_pool = self.dropout(self.pool_proj(h_pool))              # [B*S, H]

        # ---- Cross-sectional encoding ----
        if symbol_ids is None:
            # fallback: assumes fixed order S across batches
            symbol_ids = torch.arange(S, device=device).view(1, S).expand(B, S)
        emb = self.stock_emb(symbol_ids.reshape(-1))               # [B*S, H]

        h_tok = h_pool + emb                                       # [B*S, H]
        h_tok = h_tok.view(B, S, H)
        h_tok = h_tok + self.xsec(h_tok)                           # SAB x2 residual
        h_tok = h_tok.reshape(B * S, H)

        # ---- Scores (center per bar for rank losses) ----
        scores = self.score_head(h_tok).reshape(B, S)              # [B, S]
        scores = scores - scores.mean(dim=1, keepdim=True)         # bar-centering

        return scores
