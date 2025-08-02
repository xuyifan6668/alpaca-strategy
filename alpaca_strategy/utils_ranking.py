"""Differentiable ranking utilities (Spearman loss, soft ranks).

This file provides a fallback sigmoid-based soft-rank approximation for cases
where the optional *torchsort* package cannot be installed (e.g. Windows setups
without a proper CUDA toolchain).  If torchsort *is* available, we delegate to
its highly accurate implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Optional dependency ---------------------------------------------------------
try:
    import torchsort  # type: ignore
    HAS_TORCHSORT = True
except ImportError:  # pragma: no cover – we handle both paths
    torchsort = None  # type: ignore
    HAS_TORCHSORT = False


# ---------------------------------------------------------------------------
# Soft-rank implementation
# ---------------------------------------------------------------------------

def _soft_rank(x: torch.Tensor, reg: float = 1e-6, temperature: float = 1.0) -> torch.Tensor:
    """Return differentiable ranks of the last dimension of *x*.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor whose last dimension will be ranked.
    reg : float, default 1e-6
        Regularisation used by *torchsort* when available.
    temperature : float, default 1.0
        Temperature for the fallback pairwise-sigmoid approximation.
    """
    if HAS_TORCHSORT:
        return torchsort.soft_rank(x, regularization_strength=reg)  # type: ignore[attr-defined]

    # Fallback: pair-wise sigmoid approximation -----------------------------
    x1 = x.unsqueeze(-1)  # (..., n, 1)
    x2 = x.unsqueeze(-2)  # (..., 1, n)
    pairwise = torch.sigmoid((x1 - x2) / temperature)  # (..., n, n)
    return pairwise.sum(dim=-1)  # (..., n)


# ---------------------------------------------------------------------------
# Spearman rank loss
# ---------------------------------------------------------------------------

def spearman_loss(preds: torch.Tensor, targets: torch.Tensor, *, reg: float = 1e-6, temperature: float = 1.0) -> torch.Tensor:
    """Differentiable Spearman-ρ loss.

    Computes the mean-squared error between *soft* ranks of *preds* and
    *targets*, normalised per-sample (z-score) to make the loss scale-invariant.
    """
    preds_rank = _soft_rank(preds, reg=reg, temperature=temperature)
    targets_rank = _soft_rank(targets, reg=reg, temperature=temperature)

    # Standardise within each sample for numerical stability
    preds_rank = (preds_rank - preds_rank.mean(dim=1, keepdim=True)) / (preds_rank.std(dim=1, keepdim=True) + 1e-6)
    targets_rank = (targets_rank - targets_rank.mean(dim=1, keepdim=True)) / (targets_rank.std(dim=1, keepdim=True) + 1e-6)

    return F.mse_loss(preds_rank, targets_rank) 