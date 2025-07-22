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
    def __init__(self, *, encoder: Encoder | None = None, cfg=cfg, base_lr: float = 5e-4, scalers=None):
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
        scalers : dict, optional
            Dictionary of scalers for each symbol. Will be saved with the model.
        """
        super().__init__()

        self.cfg = cfg
        self.base_lr = base_lr
        self.scalers = scalers

        # Save only primitive hyper-parameters so checkpoints stay lightweight
        self.save_hyperparameters({"base_lr": base_lr})

        self.net = encoder if encoder is not None else Encoder(cfg=cfg)
        
        # Store scalers for checkpoint saving
        if scalers is not None:
            self._scalers = scalers  # Store actual scalers

    # ----------------- Optimiser & Scheduler ------------------------------
    def configure_optimizers(self):
        # AdamW optimiser followed by One-Cycle learning-rate policy
        opt = torch.optim.AdamW(self.parameters(), lr=self.base_lr)


        total_steps = int(self.trainer.estimated_stepping_batches)  # type: ignore[attr-defined]

        scheduler = OneCycleLR(
            opt,
            max_lr=self.base_lr * 3,  # peak LR = 3× base
            total_steps=total_steps,
            pct_start=0.1,  # ramp up for 10% of steps
            anneal_strategy="cos",
            final_div_factor=100,  # final LR = 1.5e-5
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", 
            },
        }

    # ----------------------- shared step ----------------------------------
    def _step(self, batch, stage: str):
        X, y_raw = batch  # y_raw: (B, S)
        logits = self.net(X)  # logits: (B, S)

        loss = F.binary_cross_entropy_with_logits(logits, y_raw)

        # Top-k accuracy calculation
        k = getattr(self.trainer.datamodule.train_ds.dataset, 'top_k', 3) if hasattr(self.trainer, 'datamodule') else 3
        with torch.no_grad():
            pred_topk = torch.topk(logits, k=k, dim=1).indices  # (B, k)
            true_topk = torch.topk(y_raw, k=k, dim=1).indices   # (B, k)
            # For each sample, count how many predicted top k are in true top k
            match = (pred_topk.unsqueeze(2) == true_topk.unsqueeze(1)).any(2).float().mean()
            self.log(f"{stage}_top{k}_acc", match, prog_bar=True)

        self.log(f"{stage}_loss", loss, prog_bar=(stage != "train"))
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
    
    def on_save_checkpoint(self, checkpoint):
        """Save scalers in the checkpoint."""
        if hasattr(self, '_scalers') and self._scalers is not None:
            checkpoint['scalers'] = self._scalers
            print(f"Saved {len(self._scalers)} scalers in checkpoint")
    
    def on_load_checkpoint(self, checkpoint):
        """Load scalers from the checkpoint."""
        if 'scalers' in checkpoint:
            self._scalers = checkpoint['scalers']
            self.scalers = self._scalers  # For backward compatibility
            print(f"Loaded {len(self._scalers)} scalers from checkpoint")
        else:
            print("No scalers found in checkpoint")
    
    @property
    def scalers(self):
        """Property to access scalers."""
        return getattr(self, '_scalers', None)
    
    @scalers.setter
    def scalers(self, value):
        """Property setter for scalers."""
        self._scalers = value 