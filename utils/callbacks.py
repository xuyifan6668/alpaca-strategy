"""Factory functions to build standard Lightning callbacks for the project."""

from __future__ import annotations

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from utils.config import cfg


def default_callbacks(conf: cfg.__class__ = cfg):
    """Return a list of callbacks with sensible defaults derived from *conf*."""
    # Main checkpoint callback - saves best models based on validation loss
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="best-epoch{epoch}-val{val_loss:.4f}",
        save_last=True,
        save_top_k=3,  # Save top 3 models based on val_loss
        every_n_epochs=1,  # Save every epoch
        save_on_train_epoch_end=True,  # Save at the end of each training epoch
    )
    
    # Additional callback to save every epoch with simple naming
    every_epoch_cb = ModelCheckpoint(
        filename="epoch-{epoch}",
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_top_k=-1,  # Save all epochs
    )
    
    es_cb = EarlyStopping(monitor="val_loss", patience=conf.patience, mode="min")
    return [ckpt_cb, every_epoch_cb, es_cb] 