"""PyTorch-Lightning module wrapping the Encoder and training logic."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import pytorch_lightning as pl

from utils.utils_ranking import spearman_loss
from utils.metrics import compute_metrics
from model.models_encoder import Encoder
from utils.config import cfg

# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class Lit(pl.LightningModule):
    def __init__(self, *, encoder: Encoder | None = None, cfg=cfg, base_lr: float = 1e-3):
        """Lightning module wrapping an **Encoder**.

        Parameters
        ----------
        encoder : Encoder, optional
            If supplied we will use this instance directly.  Otherwise a new
            `Encoder(cfg=cfg)` is created.
        cfg : Config, optional
            Configuration to pass to the encoder and for any training logic.
            Defaults to the project-wide global.
        base_lr : float, default 1e-3
            Base learning rate for the AdamW optimiser.
        """
        super().__init__()

        self.cfg = cfg
        self.base_lr = base_lr

        # Save only primitive hyper-parameters so checkpoints stay lightweight
        self.save_hyperparameters({"base_lr": base_lr})

        self.net = encoder if encoder is not None else Encoder(cfg=cfg)

    # ----------------- Optimiser & Scheduler ------------------------------
    def configure_optimizers(self):
        # AdamW optimiser followed by One-Cycle learning-rate policy
        opt = torch.optim.AdamW(self.parameters(), lr=self.base_lr)

        # Lightning sets `self.trainer` before this hook is called, so we
        # can safely derive the total number of optimisation steps.
        total_steps = int(self.trainer.estimated_stepping_batches)  # type: ignore[attr-defined]

        oc_scheduler = OneCycleLR(
            opt,
            max_lr=self.base_lr * 3,  # peak LR = 3× base
            total_steps=total_steps,
            pct_start=0.05,
            anneal_strategy="cos",
            final_div_factor=1e4,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": oc_scheduler,
                "interval": "step",  # update learning-rate every batch
            },
        }

    # ----------------------- shared step ----------------------------------
    def _step(self, batch, stage: str):
        X, y_raw = batch  # y_raw: (B,S)
        y_std = y_raw.std(dim=1, keepdim=True)
        y_mean = y_raw.mean(dim=1, keepdim=True)
        y_raw = (y_raw - y_mean) / (y_std + 1e-6)

        logits = self.net(X)

        loss = spearman_loss(logits, y_raw) + 0.1 * F.mse_loss(logits, y_raw)

        # Metrics ----------------------------------------------------------
        metrics = compute_metrics(logits, y_raw)
        loss = loss + 0.05 * torch.relu(-metrics["ic"])

        # Logging ----------------------------------------------------------
        self.log(f"{stage}_loss", loss, prog_bar=(stage != "train"))
        for k, v in metrics.items():
            self.log(f"{stage}_{k}", v, prog_bar=(k != "spread"))
        return loss

    # ---------------- Lightning hooks -------------------------------------
    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, _ = batch
        return torch.sigmoid(self.net(X)).cpu().numpy() 