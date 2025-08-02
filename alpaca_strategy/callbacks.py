"""Factory functions to build standard Lightning callbacks for the project."""

from __future__ import annotations

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from alpaca_strategy.config import Config, get_config


def default_callbacks(conf: Config | None = None):
    """Return a list of callbacks with sensible defaults derived from *conf*."""
    if conf is None:
        conf = get_config()
    # Main checkpoint callback - saves best models based on validation loss
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="best-epoch{epoch}-val{val_loss:.4f}",
        save_last=True,
        save_top_k=1,  # Save top 3 models based on val_loss
        save_on_train_epoch_end=False,  # Save at the end of each training epoch
    )
    
    
    es_cb = EarlyStopping(monitor="val_loss", patience=conf.patience, mode="min")
    return [ckpt_cb, es_cb] 