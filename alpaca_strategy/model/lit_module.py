"""PyTorch-Lightning module wrapping the Encoder and training logic."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import pytorch_lightning as pl


from alpaca_strategy.model.models_encoder import Encoder
from alpaca_strategy.config import get_config
cfg = get_config()

# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
"""PyTorch-Lightning module wrapping the Encoder and training logic (simplified)."""



class Lit(pl.LightningModule):
    def __init__(
        self,
        *,
        encoder: Encoder | None = None,
        cfg=cfg,
        base_lr: float = 1e-4,  # Reduced from 5e-4
        scalers=None,
        tau_start: float = 0.6,
        tau_end: float = 0.1,
        div_weight: float = 0.01,
    ):
        """
        Minimal LightningModule for cross-sectional rank training.

        - Loss: Spearman (soft ranks) + tiny diversity
        - No magnitude calibration (rank-only strategy)
        """
        super().__init__()
        self.cfg = cfg
        self.base_lr = base_lr
        self.net = encoder if encoder is not None else Encoder(cfg=cfg)

        # training knobs
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.div_weight = float(div_weight)

        # optional: stash scalers into checkpoint for inference
        self.scalers = scalers

        # keep checkpoints lightweight
        self.save_hyperparameters({"base_lr": base_lr, "tau_start": tau_start, "tau_end": tau_end, "div_weight": div_weight})

        # internal
        self._est_total_steps = None  # filled on fit start

    # ----------------- Optimizer & LR schedule ------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.base_lr)
        # OneCycle needs total steps; we set it exactly in on_fit_start.
        total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 1)
        self._est_total_steps = total_steps

        sch = OneCycleLR(
            opt,
            max_lr=self.base_lr * 2,  # Reduced from 3x to 2x
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            final_div_factor=100,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

    # ----------------- Utilities ------------------------------
    @staticmethod
    def rowwise_z(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        m = x.mean(dim=1, keepdim=True)
        s = x.std(dim=1, keepdim=True)
        return (x - m) / (s + eps)

    @staticmethod
    def soft_rank(x: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
        """Differentiable soft ranks; O(S^2). x: [B,S] -> [B,S] in ~[1..S]."""
        x_i = x.unsqueeze(-1)                         # [B,S,1]
        x_j = x.unsqueeze(-2)                         # [B,1,S]
        P = torch.sigmoid((x_j - x_i) / tau)          # P[j < i]
        return 1.0 + P.sum(dim=-1)                    # expected rank

    @staticmethod
    def hard_rank(y: torch.Tensor) -> torch.Tensor:
        """Non-diff ranks for targets. y: [B,S] -> [B,S] in 1..S."""
        return y.argsort(dim=1).argsort(dim=1).float() + 1.0

    def _spearman_loss(self, pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
        """Rowwise (per-bar) Spearman via Pearson(soft_rank(pred), rank(target))."""
        # skip degenerate rows
        valid = target.std(dim=1) > 1e-6
        if not valid.any():
            # return a tiny 0-like tensor that still backprops
            return pred.sum() * 0.0

        pred = pred[valid]
        target = target[valid]

        sr = self.soft_rank(pred, tau=tau)           # [B',S]
        r  = self.hard_rank(target)                  # [B',S]

        pz, yz = self.rowwise_z(sr), self.rowwise_z(r)
        corr = (pz * yz).mean(dim=1).mean()          # mean over bars
        return -corr                                  # maximize correlation

    # ----------------- Shared step ------------------------------
    def _step(self, batch, stage: str):
        X, y = batch                                  # X: [B,S,T,D], y: [B,S]
        pred = self.net(X)                            # [B,S]; bar-centered by encoder

        # tau anneal by *training progress* (step-based)
        if self._est_total_steps is None or self._est_total_steps <= 1:
            progress = 0.0
        else:
            progress = min(1.0, float(self.global_step) / float(self._est_total_steps - 1))
        tau = self.tau_start + (self.tau_end - self.tau_start) * progress

        # losses
        corr_loss = self._spearman_loss(pred, y, tau=tau)
        diversity = torch.exp(-pred.std(dim=1).mean().clamp_min(1e-8))
        loss = corr_loss + self.div_weight * diversity

        bs = X.size(0)
        show_bar = stage in ("train", "val")
        on_step = stage == "train"  # step-wise for train only
        on_epoch = True

        self.log(f"{stage}_loss", loss, on_step=on_step, on_epoch=on_epoch, prog_bar=show_bar, batch_size=bs)
        self.log(f"{stage}_corr_loss", corr_loss, on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=bs)
        self.log(f"{stage}_diversity", diversity, on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=bs)
        self.log(f"{stage}_tau", tau, on_step=on_step, on_epoch=on_epoch, prog_bar=show_bar, batch_size=bs)

        # cross-sectional monitors
        with torch.no_grad():
            valid = y.std(dim=1) > 1e-6
            if valid.any():
                sr = self.soft_rank(pred[valid], tau=self.tau_end)  # fixed tau for monitor
                r = self.hard_rank(y[valid])
                cs_ic = (self.rowwise_z(sr) * self.rowwise_z(r)).mean(dim=1).mean()
                cs_pred_std = pred[valid].std(dim=1).mean()
                self.log(f"{stage}_cs_ic", cs_ic, on_step=on_step, on_epoch=on_epoch, prog_bar=show_bar, batch_size=bs)
                self.log(f"{stage}_cs_pred_std", cs_pred_std, on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=bs)

        return loss
    # ----------------- Lightning hooks ------------------------------
    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        
        # Monitor gradients
        if self.global_step % 100 == 0:
            total_norm = 0.0
            param_count = 0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                self.log("train_grad_norm", total_norm)
        
        return loss

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, _ = batch
        pred = self.net(X)
        return pred.detach().cpu().numpy()

    # ----------------- Checkpoint I/O for scalers ------------------------------
    def on_save_checkpoint(self, checkpoint):
        if self.scalers is not None:
            checkpoint["scalers"] = self.scalers

    def on_load_checkpoint(self, checkpoint):
        if "scalers" in checkpoint:
            self.scalers = checkpoint["scalers"]

    @property
    def scalers(self):
        """Property to access scalers."""
        return getattr(self, '_scalers', None)
    
    @scalers.setter
    def scalers(self, value):
        """Property setter for scalers."""
        self._scalers = value 