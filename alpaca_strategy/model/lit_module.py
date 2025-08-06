# =============================================================================
# PYTORCH LIGHTNING MODULE FOR ALPACA STRATEGY
# =============================================================================
# This module implements the PyTorch Lightning wrapper for the alpaca-strategy
# model. It provides the complete training pipeline including loss computation,
# optimization, and evaluation metrics for cross-sectional ranking prediction.
# =============================================================================

"""PyTorch-Lightning module wrapping the Encoder and training logic."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import pytorch_lightning as pl

from alpaca_strategy.model.models_encoder import Encoder
from alpaca_strategy.config import get_config
cfg = get_config()

# =============================================================================
# LIGHTNING MODULE IMPLEMENTATION
# =============================================================================
# The Lit class implements the complete training logic for the alpaca-strategy
# model, including loss computation, optimization, and evaluation metrics.
# =============================================================================

class Lit(pl.LightningModule):
    """
    PyTorch Lightning module for cross-sectional ranking prediction.
    
    This module implements a complete training pipeline for predicting stock
    rankings based on time series data. It uses:
    - Spearman correlation loss for ranking prediction
    - Diversity regularization to prevent collapse
    - Temperature annealing for soft ranking
    - Cross-sectional evaluation metrics
    
    Key Features:
    - Cross-sectional ranking prediction across multiple stocks
    - Soft ranking with temperature annealing
    - Diversity regularization to maintain prediction spread
    - Comprehensive logging and monitoring
    - Automatic checkpoint management with scalers
    """
    
    def __init__(
        self,
        *,
        encoder: Encoder | None = None,
        cfg=cfg,
        base_lr: float = 1e-4,
        scalers=None,
        tau_start: float = 0.6,  # Initial temperature for soft ranking
        tau_end: float = 0.1,    # Final temperature for soft ranking
        div_weight: float = 0.01,  # Weight for diversity regularization
    ):
        """
        Initialize the Lightning module for cross-sectional ranking training.
        
        Args:
            encoder: Pre-configured encoder model (optional)
            cfg: Configuration object containing model parameters
            base_lr: Base learning rate for optimization
            scalers: Data scalers for normalization (saved in checkpoints)
            tau_start: Initial temperature for soft ranking computation
            tau_end: Final temperature for soft ranking computation
            div_weight: Weight for diversity regularization loss
        """
        super().__init__()
        self.cfg = cfg
        self.base_lr = base_lr
        
        # Initialize the encoder model (either provided or created from config)
        self.net = encoder if encoder is not None else Encoder(cfg=cfg)

        # =============================================================================
        # TRAINING HYPERPARAMETERS
        # =============================================================================
        # These parameters control the training behavior and loss computation
        self.tau_start = float(tau_start)    # Initial soft ranking temperature
        self.tau_end = float(tau_end)        # Final soft ranking temperature
        self.div_weight = float(div_weight)  # Diversity regularization weight

        # Store scalers for checkpoint saving (needed for inference)
        self.scalers = scalers

        # Save hyperparameters for checkpoint management
        # This ensures reproducibility and proper model loading
        self.save_hyperparameters({
            "base_lr": base_lr, 
            "tau_start": tau_start, 
            "tau_end": tau_end, 
            "div_weight": div_weight
        })

        # Internal tracking for total training steps (set during training)
        self._est_total_steps = None

    # =============================================================================
    # OPTIMIZER AND LEARNING RATE SCHEDULING
    # =============================================================================
    # Configure the optimizer and learning rate scheduler for training.
    # Uses AdamW optimizer with OneCycleLR scheduler for optimal convergence.
    # =============================================================================

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary containing optimizer and scheduler configuration
            for PyTorch Lightning training.
        """
        # Use AdamW optimizer with weight decay for regularization
        opt = torch.optim.AdamW(self.parameters(), lr=self.base_lr)
        
        # OneCycleLR needs total steps; we set it exactly in on_fit_start
        # This ensures proper learning rate scheduling throughout training
        total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 1)
        self._est_total_steps = total_steps

        # Configure OneCycleLR scheduler for optimal convergence
        sch = OneCycleLR(
            opt,
            max_lr=self.base_lr * 2,  # Peak learning rate (2x base)
            total_steps=total_steps,   # Total training steps
            pct_start=0.1,            # Warm-up phase (10% of training)
            anneal_strategy="cos",     # Cosine annealing for smooth decay
            final_div_factor=100,     # Final LR = base_lr / 100
        )
        
        return {
            "optimizer": opt, 
            "lr_scheduler": {"scheduler": sch, "interval": "step"}
        }

    # =============================================================================
    # UTILITY FUNCTIONS
    # =============================================================================
    # Helper functions for data preprocessing and ranking computation.
    # These functions support the main training pipeline.
    # =============================================================================

    @staticmethod
    def rowwise_z(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute row-wise z-score normalization.
        
        Args:
            x: Input tensor of shape [B, S] or similar
            eps: Small epsilon to prevent division by zero
            
        Returns:
            Z-score normalized tensor with zero mean and unit variance per row
        """
        m = x.mean(dim=1, keepdim=True)  # Row-wise mean
        s = x.std(dim=1, keepdim=True)   # Row-wise standard deviation
        return (x - m) / (s + eps)       # Z-score normalization

    @staticmethod
    def soft_rank(x: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
        """
        Compute differentiable soft ranks using sigmoid approximation.
        
        This function implements a differentiable approximation of ranking
        that can be used in gradient-based optimization. The temperature
        parameter tau controls the sharpness of the ranking.
        
        Args:
            x: Input tensor of shape [B, S] (batch_size, num_symbols)
            tau: Temperature parameter controlling ranking sharpness
            
        Returns:
            Soft ranks tensor of shape [B, S] with values approximately in [1, S]
            
        Complexity: O(S^2) where S is the number of symbols
        """
        x_i = x.unsqueeze(-1)                         # [B,S,1] - expand for comparison
        x_j = x.unsqueeze(-2)                         # [B,1,S] - expand for comparison
        P = torch.sigmoid((x_j - x_i) / tau)          # P[j < i] - probability that j < i
        return 1.0 + P.sum(dim=-1)                    # Expected rank (1-based)

    @staticmethod
    def hard_rank(y: torch.Tensor) -> torch.Tensor:
        """
        Compute non-differentiable hard ranks for target values.
        
        This function computes the actual ranks of target values for
        comparison with predicted soft ranks.
        
        Args:
            y: Target tensor of shape [B, S]
            
        Returns:
            Hard ranks tensor of shape [B, S] with values in [1, S]
        """
        return y.argsort(dim=1).argsort(dim=1).float() + 1.0

    def _spearman_loss(self, pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Compute Spearman correlation loss via Pearson correlation of soft ranks.
        
        This function implements the main ranking loss by computing the
        correlation between predicted soft ranks and target hard ranks.
        It handles degenerate cases where targets have no variance.
        
        Args:
            pred: Predicted scores of shape [B, S]
            target: Target values of shape [B, S]
            tau: Temperature for soft ranking computation
            
        Returns:
            Negative Spearman correlation (loss to minimize)
        """
        # Skip degenerate rows (those with no variance in targets)
        valid = target.std(dim=1) > 1e-6
        if not valid.any():
            # Return a tiny 0-like tensor that still backprops
            return pred.sum() * 0.0

        # Filter to valid rows only
        pred = pred[valid]
        target = target[valid]

        # Compute soft ranks for predictions and hard ranks for targets
        sr = self.soft_rank(pred, tau=tau)           # [B',S] - soft ranks
        r  = self.hard_rank(target)                  # [B',S] - hard ranks

        # Compute Pearson correlation between z-scored ranks
        pz, yz = self.rowwise_z(sr), self.rowwise_z(r)  # Z-score normalization
        corr = (pz * yz).mean(dim=1).mean()          # Mean correlation across batch
        return -corr                                  # Negative for minimization

    # =============================================================================
    # SHARED TRAINING STEP
    # =============================================================================
    # Core training logic shared between training, validation, and test steps.
    # This function computes loss, metrics, and handles logging.
    # =============================================================================

    def _step(self, batch, stage: str):
        """
        Shared step function for training, validation, and testing.
        
        This function implements the core forward pass, loss computation,
        and metric logging for all training stages.
        
        Args:
            batch: Input batch containing (X, y) where X is features and y is targets
            stage: Current stage ("train", "val", or "test")
            
        Returns:
            Computed loss value
        """
        # Unpack batch data
        X, y = batch                                  # X: [B,S,T,D], y: [B,S]
        pred = self.net(X)                            # [B,S] - model predictions

        # =============================================================================
        # TEMPERATURE ANNEALING
        # =============================================================================
        # Anneal temperature based on training progress for better convergence
        if self._est_total_steps is None or self._est_total_steps <= 1:
            progress = 0.0
        else:
            progress = min(1.0, float(self.global_step) / float(self._est_total_steps - 1))
        
        # Linear interpolation between start and end temperature
        tau = self.tau_start + (self.tau_end - self.tau_start) * progress

        # =============================================================================
        # LOSS COMPUTATION
        # =============================================================================
        # Compute correlation loss and diversity regularization
        corr_loss = self._spearman_loss(pred, y, tau=tau)  # Main ranking loss
        diversity = torch.exp(-pred.std(dim=1).mean().clamp_min(1e-8))  # Diversity penalty
        loss = corr_loss + self.div_weight * diversity  # Combined loss

        # =============================================================================
        # METRIC LOGGING
        # =============================================================================
        # Configure logging parameters
        bs = X.size(0)  # Batch size
        show_bar = stage in ("train", "val")  # Show in progress bar for train/val
        on_step = stage == "train"  # Log per-step only for training
        on_epoch = True  # Always log per-epoch

        # Log main metrics
        self.log(f"{stage}_loss", loss, on_step=on_step, on_epoch=on_epoch, prog_bar=show_bar, batch_size=bs)
        self.log(f"{stage}_corr_loss", corr_loss, on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=bs)
        self.log(f"{stage}_diversity", diversity, on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=bs)
        self.log(f"{stage}_tau", tau, on_step=on_step, on_epoch=on_epoch, prog_bar=show_bar, batch_size=bs)

        # =============================================================================
        # CROSS-SECTIONAL MONITORING
        # =============================================================================
        # Compute additional cross-sectional metrics for monitoring
        with torch.no_grad():
            valid = y.std(dim=1) > 1e-6
            if valid.any():
                # Use fixed tau for consistent monitoring
                sr = self.soft_rank(pred[valid], tau=self.tau_end)
                r = self.hard_rank(y[valid])
                
                # Cross-sectional information coefficient
                cs_ic = (self.rowwise_z(sr) * self.rowwise_z(r)).mean(dim=1).mean()
                
                # Cross-sectional prediction standard deviation
                cs_pred_std = pred[valid].std(dim=1).mean()
                
                # Log cross-sectional metrics
                self.log(f"{stage}_cs_ic", cs_ic, on_step=on_step, on_epoch=on_epoch, prog_bar=show_bar, batch_size=bs)
                self.log(f"{stage}_cs_pred_std", cs_pred_std, on_step=on_step, on_epoch=on_epoch, prog_bar=False, batch_size=bs)

        return loss

    # =============================================================================
    # LIGHTNING TRAINING HOOKS
    # =============================================================================
    # PyTorch Lightning hooks for training, validation, and testing.
    # These methods are called automatically during training.
    # =============================================================================

    def training_step(self, batch, batch_idx):
        """
        Training step with additional gradient monitoring.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        loss = self._step(batch, "train")
        
        # Monitor gradients every 100 steps for debugging
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
        """
        Validation step for model evaluation.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
        """
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        """
        Test step for final model evaluation.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
        """
        self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step for inference.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            dataloader_idx: DataLoader index
            
        Returns:
            Model predictions as numpy array
        """
        X, _ = batch
        pred = self.net(X)
        return pred.detach().cpu().numpy()

    # =============================================================================
    # CHECKPOINT MANAGEMENT
    # =============================================================================
    # Methods for saving and loading scalers with model checkpoints.
    # This ensures data normalization consistency during inference.
    # =============================================================================

    def on_save_checkpoint(self, checkpoint):
        """
        Save scalers with model checkpoint for inference consistency.
        
        Args:
            checkpoint: PyTorch Lightning checkpoint dictionary
        """
        if self.scalers is not None:
            checkpoint["scalers"] = self.scalers

    def on_load_checkpoint(self, checkpoint):
        """
        Load scalers from model checkpoint.
        
        Args:
            checkpoint: PyTorch Lightning checkpoint dictionary
        """
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