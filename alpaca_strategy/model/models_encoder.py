# =============================================================================
# NEURAL NETWORK ENCODER ARCHITECTURE
# =============================================================================
# This module implements the core neural network architecture for the
# alpaca-strategy model. It includes a Time2Vec embedding layer and a
# transformer-based encoder for processing multi-stock time series data.
# =============================================================================

import torch
import torch.nn as nn
import math
from alpaca_strategy.config import get_config
cfg = get_config()

# =============================================================================
# TIME2VEC EMBEDDING LAYER
# =============================================================================
# Time2Vec is a learnable time embedding that captures both linear and
# periodic temporal patterns. This is essential for modeling time-dependent
# market behaviors and intraday patterns.
# =============================================================================

class Time2Vec(nn.Module):
    """
    Time2Vec embedding layer for temporal feature representation.
    
    Time2Vec learns both linear and periodic representations of time,
    which is crucial for capturing market patterns like:
    - Intraday seasonality (lunch hours, market open/close effects)
    - Day-of-week patterns
    - Time-based market microstructure effects
    
    The embedding combines:
    - Linear component: Captures monotonic time trends
    - Periodic components: Capture recurring temporal patterns
    
    Args:
        out_dim: Output dimension of the time embedding (must be >= 2)
    """
    
    def __init__(self, out_dim: int = 8):
        super().__init__()
        if out_dim < 2:
            raise ValueError("Time2Vec out_dim must be >=2")
        
        self.out_dim = out_dim
        
        # Linear component parameters (1 output)
        self.w0 = nn.Parameter(torch.randn(1))  # Linear weight
        self.b0 = nn.Parameter(torch.randn(1))  # Linear bias
        
        # Periodic component parameters (out_dim - 1 outputs)
        self.W = nn.Parameter(torch.randn(out_dim - 1))  # Frequency weights
        self.phi = nn.Parameter(torch.randn(out_dim - 1))  # Phase shifts

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute Time2Vec embedding for input time values.
        
        Args:
            t: Input time tensor of any shape
            
        Returns:
            Time2Vec embedding tensor with shape [..., out_dim]
            where the first dimension is linear and the rest are periodic
        """
        # Linear component: w0 * t + b0
        lin = self.w0 * t + self.b0
        
        # Periodic components: sin(w * t + phi)
        per = torch.sin(t.unsqueeze(-1) * self.W + self.phi)
        
        # Concatenate linear and periodic components
        return torch.cat([lin.unsqueeze(-1), per], dim=-1)

# =============================================================================
# MAIN ENCODER ARCHITECTURE
# =============================================================================
# The Encoder class implements the complete neural network for processing
# multi-stock time series data and generating cross-sectional predictions.
# It combines temporal encoding, cross-sectional modeling, and attention mechanisms.
# =============================================================================

class Encoder(nn.Module):
    """
    Neural network encoder for multi-stock time series prediction.
    
    This encoder processes time series data for multiple stocks simultaneously
    and generates cross-sectional predictions. The architecture includes:
    
    1. Time2Vec embedding for temporal features
    2. Temporal transformer layers for sequence modeling
    3. Multi-strategy time pooling (last, EWMA, attention)
    4. Cross-sectional transformer layers for stock interactions
    5. Stock embeddings for symbol-specific features
    6. Final prediction head for generating scores
    
    The model is designed to capture:
    - Temporal dependencies within each stock's time series
    - Cross-sectional relationships between stocks
    - Market microstructure patterns
    - Volume and volatility-based attention mechanisms
    
    Args:
        cfg: Configuration object containing model hyperparameters
    """
    
    def __init__(self, *, cfg=cfg):
        super().__init__()
        self.cfg = cfg
        H = self.cfg.hidden  # Hidden dimension

        # =============================================================================
        # TIME2VEC EMBEDDING
        # =============================================================================
        # Learnable time embedding for capturing temporal patterns
        self.t2v = Time2Vec(out_dim=8)
        t2v_dim = self.t2v.out_dim

        # =============================================================================
        # FEATURE PROJECTION
        # =============================================================================
        # Project concatenated features (original + time2vec) to hidden dimension
        self.proj = nn.Linear(self.cfg.FEAT_DIM + t2v_dim, H)

        # =============================================================================
        # TEMPORAL TRANSFORMER STACK
        # =============================================================================
        # Stack of transformer layers for modeling temporal dependencies
        # Each layer processes the sequence dimension (T) for each stock
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

        # =============================================================================
        # ATTENTION-BASED TIME POOLING
        # =============================================================================
        # Pooling token and attention mechanism for biased time aggregation
        self.pool_token = nn.Parameter(torch.randn(1, 1, H))
        
        # Q/K/V projections for explicit biased attention
        self.q_proj = nn.Linear(H, H)
        self.k_proj = nn.Linear(H, H)
        self.v_proj = nn.Linear(H, H)
        
        # Trainable scale for attention bias
        self._raw_attn_bias_scale = nn.Parameter(torch.tensor(0.0))
        self.max_attn_bias = 5.0  # Maximum attention bias magnitude
        self.dropout = nn.Dropout(cfg.dropout)

        # =============================================================================
        # EWMA POOLING
        # =============================================================================
        # Exponentially weighted moving average for smooth temporal aggregation
        # This captures recent trends while maintaining some historical context
        self.register_buffer("ewma_decay", torch.tensor(0.03))

        # =============================================================================
        # POOLING PROJECTION
        # =============================================================================
        # Project concatenated pooling strategies back to hidden dimension
        self.pool_proj = nn.Linear(3 * H, H)

        # =============================================================================
        # STOCK EMBEDDINGS
        # =============================================================================
        # Learnable embeddings for each stock symbol
        # This captures stock-specific characteristics and behaviors
        self.stock_emb = nn.Embedding(num_embeddings=cfg.num_stocks, embedding_dim=H)

        # =============================================================================
        # CROSS-SECTIONAL TRANSFORMER
        # =============================================================================
        # Transformer layers for modeling relationships between stocks
        # This enables the model to learn cross-sectional patterns and dependencies
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

        # =============================================================================
        # PREDICTION HEAD
        # =============================================================================
        # Final layers for generating stock-specific prediction scores
        # Architecture: LayerNorm -> 2H GELU -> Dropout -> 1
        self.score_head = nn.Sequential(
            nn.LayerNorm(H),           # Normalize features
            nn.Linear(H, 2*H),         # Expand to 2H dimensions
            nn.GELU(),                 # Non-linear activation
            nn.Dropout(cfg.dropout),   # Regularization
            nn.Linear(2*H, 1),         # Final prediction (1 score per stock)
        )

    @staticmethod
    def _ewma_pool(h_seq: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
        """
        Compute exponentially weighted moving average pooling.
        
        This function implements EWMA pooling where more recent timesteps
        receive higher weights. The decay parameter controls the weighting
        schedule.
        
        Args:
            h_seq: Hidden states tensor of shape [B*S, T, H]
            decay: Decay parameter controlling the weighting schedule
            
        Returns:
            EWMA pooled tensor of shape [B*S, H]
        """
        BxS, T, H = h_seq.shape
        device = h_seq.device
        
        # Create lag indices: 0 at past, T-1 at latest
        lags = torch.arange(T, device=device).float()
        
        # Reverse lags so recent timesteps get highest weight
        rev = (T - 1) - lags
        
        # Compute exponential weights: exp(-decay * rev)
        w = torch.exp(-decay * rev)                         # [T]
        w = w / (w.sum() + 1e-8)  # Normalize weights
        
        # Apply weights to sequence: weighted sum across time dimension
        pooled = torch.einsum("t, bth -> bh", w, h_seq)     # [B*S, H]
        return pooled

    def forward(self, x: torch.Tensor, symbol_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass through the encoder architecture.
        
        This function processes multi-stock time series data and generates
        cross-sectional prediction scores. The processing pipeline includes:
        
        1. Time2Vec embedding for temporal features
        2. Temporal transformer encoding
        3. Multi-strategy time pooling (last, EWMA, attention)
        4. Cross-sectional transformer encoding
        5. Final prediction generation
        
        Args:
            x: Input tensor of shape [B, S, T, D] where:
               B = batch size, S = number of stocks, T = time steps, D = features
            symbol_ids: Optional tensor of shape [B, S] containing stock symbol indices.
                       If None, assumes fixed order across batches.
        
        Returns:
            Prediction scores tensor of shape [B, S] with one score per stock
        """
        B, S, T, D = x.shape
        device = x.device
        H = self.cfg.hidden

        # =============================================================================
        # TIME2VEC EMBEDDING
        # =============================================================================
        # Extract minute normalization feature and compute Time2Vec embedding
        minute_scalar = x[..., self.cfg.MINUTE_IDX]               # [B,S,T]
        t2v_emb = self.t2v(minute_scalar)                         # [B,S,T,t2v]
        x_cat = torch.cat([x, t2v_emb], dim=-1)                   # [B,S,T,D+t2v]

        # =============================================================================
        # TEMPORAL ENCODING
        # =============================================================================
        # Reshape for temporal processing and apply transformer layers
        h_seq = self.proj(x_cat.view(B * S, T, -1))               # [B*S, T, H]
        h_seq = self.temporal_stack(h_seq)                        # [B*S, T, H]

        # =============================================================================
        # ATTENTION BIAS COMPUTATION
        # =============================================================================
        # Build attention bias from volume and volatility features
        # This allows the model to focus on high-volume or volatile periods
        vol = x[..., self.cfg.VOLUME_IDX]                         # [B,S,T]
        std = x[..., self.cfg.STD_IDX]                            # [B,S,T]

        def zscore_lastdim(t):
            """Compute z-score normalization across the last dimension."""
            m = t.mean(dim=-1, keepdim=True)
            s = t.std(dim=-1, keepdim=True)
            return torch.where(s > 1e-8, (t - m) / s, torch.zeros_like(t))

        # Compute z-scored volume and volatility biases
        vol_bias = zscore_lastdim(vol)
        std_bias = zscore_lastdim(std)
        
        # Combine biases and clip to reasonable range
        attn_bias = torch.clamp(vol_bias + std_bias, -3.0, 3.0)   # [B,S,T]
        attn_bias = attn_bias.view(B * S, T)                      # [B*S, T]

        # =============================================================================
        # MULTI-STRATEGY TIME POOLING
        # =============================================================================
        # Combine three different pooling strategies for robust temporal aggregation
        
        # 1) Last state (strong recency signal)
        last = h_seq[:, -1, :]                                    # [B*S, H]

        # 2) EWMA pooled state (smooth recency)
        ewma = self._ewma_pool(h_seq, self.ewma_decay)            # [B*S, H]

        # 3) Biased attention pooling over time
        pool_tok = self.pool_token.expand(B * S, 1, H)            # [B*S, 1, H]
        q = self.q_proj(pool_tok)                                  # [B*S, 1, H]
        k = self.k_proj(h_seq)                                     # [B*S, T, H]
        v = self.v_proj(h_seq)                                     # [B*S, T, H]

        # Scaled dot-product attention with additive bias on logits
        logits = torch.einsum("bqh,bkh->bqk", q, k) / math.sqrt(H) # [B*S, 1, T]
        
        # Apply trainable attention bias scaling
        attn_bias_scale = self.max_attn_bias * torch.sigmoid(self._raw_attn_bias_scale)
        logits = logits + attn_bias_scale * attn_bias.unsqueeze(1)  # broadcast
        
        # Compute attention weights and apply to values
        attn = torch.softmax(logits, dim=-1)                       # [B*S, 1, T]
        attn_pooled = torch.einsum("bqk,bkh->bqh", attn, v).squeeze(1)  # [B*S, H]
        attn_pooled = self.dropout(attn_pooled)

        # Concatenate all pooling strategies and project back to H
        h_pool = torch.cat([last, ewma, attn_pooled], dim=-1)      # [B*S, 3H]
        h_pool = self.dropout(self.pool_proj(h_pool))              # [B*S, H]

        # =============================================================================
        # CROSS-SECTIONAL ENCODING
        # =============================================================================
        # Add stock embeddings and apply cross-sectional transformer
        
        # Handle symbol IDs (use default ordering if not provided)
        if symbol_ids is None:
            # Fallback: assumes fixed order S across batches
            symbol_ids = torch.arange(S, device=device).view(1, S).expand(B, S)
        
        # Get stock embeddings and add to pooled features
        emb = self.stock_emb(symbol_ids.reshape(-1))               # [B*S, H]
        h_tok = h_pool + emb                                       # [B*S, H]
        
        # Reshape for cross-sectional processing
        h_tok = h_tok.view(B, S, H)
        
        # Apply cross-sectional transformer layers
        h_tok = h_tok + self.xsec(h_tok)                           # SAB x2 residual
        h_tok = h_tok.reshape(B * S, H)

        # =============================================================================
        # FINAL PREDICTION
        # =============================================================================
        # Generate final prediction scores for each stock
        scores = self.score_head(h_tok).reshape(B, S)              # [B, S]
        
        return scores
