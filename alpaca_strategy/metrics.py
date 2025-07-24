"""Computation of evaluation metrics used during training.

All functions operate purely on in-memory tensors/arrays, so the module has no
external dependencies beyond PyTorch and NumPy.
"""

from __future__ import annotations

import torch
from typing import Dict


@torch.inference_mode()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, *, topk_ratio: float = 0.1) -> Dict[str, torch.Tensor]:
    """Return a dict of metrics (top-k accuracy, bottom-k, long-short, IC, spread).

    Parameters
    ----------
    logits : Tensor (B, S)
        Raw model scores (higher ⇒ more bullish).
    targets : Tensor (B, S)
        Z-scored ground-truth future returns.
    topk_ratio : float, default 0.1
        Fraction of stocks considered "top" (and bottom) per sample.
    """
    S = targets.size(1)
    k = max(1, int(S * topk_ratio))

    pred_rank = logits.argsort(dim=1, descending=True)
    true_rank = targets.argsort(dim=1, descending=True)

    pred_topk = pred_rank[:, :k]
    true_topk = true_rank[:, :k]
    pred_bottomk = pred_rank[:, -k:]
    true_bottomk = true_rank[:, -k:]

    topk_hits = (true_topk.unsqueeze(2) == pred_topk.unsqueeze(1)).any(2).float().mean()
    bottomk_hits = (true_bottomk.unsqueeze(2) == pred_bottomk.unsqueeze(1)).any(2).float().mean()
    longshort_acc = 0.5 * (topk_hits + bottomk_hits)

    # ――― Information Coefficient (per-sample Pearson r) ―――
    # Compute in a fully-vectorised manner to avoid ops that are unsupported for
    # XLA tensors inside `torch.inference_mode()`.
    logits_centered = logits - logits.mean(dim=1, keepdim=True)
    targets_centered = targets - targets.mean(dim=1, keepdim=True)

    cov = (logits_centered * targets_centered).mean(dim=1)  # (B,)
    std_logits = torch.sqrt((logits_centered.square()).mean(dim=1) + 1e-12)
    std_targets = torch.sqrt((targets_centered.square()).mean(dim=1) + 1e-12)

    valid = std_targets > 1e-8
    ic_vals = cov / (std_logits * std_targets + 1e-8)
    ic = ic_vals[valid].mean() if valid.any() else logits.new_zeros(1)[0]

    spread = targets.gather(1, pred_topk).mean() - targets.gather(1, pred_bottomk).mean()

    return {
        "topk_acc":     topk_hits,
        "bottomk_acc":  bottomk_hits,
        "longshort_acc": longshort_acc,
        "ic":           ic,
        "spread":       spread,
    } 