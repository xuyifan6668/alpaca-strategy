# =============================================================================
# TRAINING CALLBACKS SYSTEM
# =============================================================================
# This module provides factory functions to create standard PyTorch Lightning
# callbacks for the alpaca-strategy project. Callbacks are essential components
# that monitor training progress and perform actions like model checkpointing
# and early stopping.
# =============================================================================

"""Factory functions to build standard Lightning callbacks for the project."""

from __future__ import annotations

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from alpaca_strategy.config import Config, get_config

# =============================================================================
# CALLBACK FACTORY FUNCTION
# =============================================================================
# This function creates a standardized set of callbacks that work together
# to monitor training progress and save the best models automatically.
# =============================================================================

def default_callbacks(conf: Config | None = None):
    """
    Return a list of callbacks with sensible defaults derived from *conf*.
    
    This function creates the essential callbacks needed for robust model training:
    1. ModelCheckpoint: Saves the best model based on validation loss
    2. EarlyStopping: Prevents overfitting by stopping training when validation
       loss stops improving
    
    Args:
        conf: Configuration object containing training parameters. If None,
              uses the default configuration from get_config().
    
    Returns:
        List of PyTorch Lightning callbacks configured for the project.
    
    Callback Details:
        - ModelCheckpoint: Monitors validation loss and saves the best model
        - EarlyStopping: Stops training when validation loss plateaus
    """
    if conf is None:
        conf = get_config()
    
    # =============================================================================
    # MODEL CHECKPOINT CALLBACK
    # =============================================================================
    # This callback automatically saves the best model during training based on
    # validation loss. It ensures we don't lose the best performing model and
    # provides recovery points in case of training interruption.
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",                    # Metric to monitor for best model
        mode="min",                            # Lower validation loss is better
        filename="best-epoch{epoch}-val{val_loss:.4f}",  # Filename pattern with epoch and loss
        save_last=True,                        # Always save the last model state
        save_top_k=1,                          # Save only the best 1 model (most recent best)
        save_on_train_epoch_end=False,         # Save at validation epoch end, not training end
    )
    
    # =============================================================================
    # EARLY STOPPING CALLBACK
    # =============================================================================
    # This callback prevents overfitting by monitoring validation loss and
    # stopping training when it stops improving for a specified number of epochs.
    # This helps prevent wasting computational resources on overfitting.
    es_cb = EarlyStopping(
        monitor="val_loss",                    # Metric to monitor for early stopping
        patience=conf.patience,                # Number of epochs to wait before stopping
        mode="min"                             # Lower validation loss is better
    )
    
    # Return the configured callbacks as a list
    return [ckpt_cb, es_cb] 