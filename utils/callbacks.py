"""Factory functions to build standard Lightning callbacks for the project."""

from __future__ import annotations

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from utils.config import cfg


def default_callbacks(conf: cfg.__class__ = cfg):
    """Return a list of callbacks with sensible defaults derived from *conf*."""
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="epoch{epoch}-val{val_loss:.4f}",
        save_last=True,
    )
    es_cb = EarlyStopping(monitor="val_loss", patience=conf.patience, mode="min")
    return [ckpt_cb, es_cb] 