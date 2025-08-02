"""PyTorch-Lightning module wrapping the Encoder and training logic."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import pytorch_lightning as pl

from alpaca_strategy.utils_ranking import spearman_loss
from alpaca_strategy.metrics import compute_metrics
from alpaca_strategy.model.models_encoder import Encoder
from alpaca_strategy.config import get_config
cfg = get_config()

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

    # ----------------- Forward Method for Model Summary ------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for model summary and inference."""
        return self.net(x)

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
        X, y_raw = batch                    # y_raw: [B, S] future returns
        pred = self.net(X)                 # logits → predicted returns

        loss = F.mse_loss(pred, y_raw)

        # Optional IC logging (correlation between predicted and actual returns)
        ic_vals = []
        for i in range(y_raw.size(0)):
            if y_raw[i].std() > 1e-8:
                ic_vals.append(torch.corrcoef(torch.stack([pred[i], y_raw[i]]))[0, 1])
        ic = torch.stack(ic_vals).mean() if ic_vals else torch.tensor(0.0, device=self.device)

        # Enhanced logging with more metrics
        self.log(f"{stage}_loss", loss, prog_bar=False, sync_dist=True)
        self.log(f"{stage}_ic", ic, prog_bar=False, sync_dist=True)
        self.log(f"{stage}_pred_mean", pred.mean(), prog_bar=False)
        self.log(f"{stage}_pred_std", pred.std(), prog_bar=False)
        self.log(f"{stage}_target_mean", y_raw.mean(), prog_bar=False)
        self.log(f"{stage}_target_std", y_raw.std(), prog_bar=False)
        self.log(f"{stage}_mse", F.mse_loss(pred, y_raw), prog_bar=False)
        self.log(f"{stage}_mae", F.l1_loss(pred, y_raw), prog_bar=False)
        
        # Print progress every 100 steps since progress bar is disabled
        if self.global_step % 100 == 0:
            print(f"Step {self.global_step}: {stage}_loss={loss.item():.4f}, {stage}_ic={ic.item():.4f}")
        
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
        return self.net(X).cpu().numpy()  # skip sigmoid; keep raw return predictions
    
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