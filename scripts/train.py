"""Training / prediction entry point.

Replaces the old monolithic pipeline CLI with a thin wrapper that wires
together the DataModule, Lit module, callbacks, and logger.
"""

from __future__ import annotations

import pathlib
import sys
import os

# Add the project root to Python path so we can import alpaca_strategy
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch

from alpaca_strategy.config import get_config
cfg = get_config()
from alpaca_strategy.data.data_module import AllSymbolsDataModule
from alpaca_strategy.callbacks import default_callbacks
from alpaca_strategy.model.lit_module import Lit



def main():
    # Enable Tensor Core optimization for better performance on NVIDIA GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    mode = "train"
    ckpt_path = pathlib.Path("last.ckpt")
    out_path = "preds.npy"

    dm = AllSymbolsDataModule()
    dm.prepare_data()
    dm.setup("fit")

    if mode == "train":
        scalers = getattr(dm, 'scalers', None)
        model = Lit(scalers=scalers)
        if cfg.log_wandb:
            logger = WandbLogger(
                project="micro-graph-v2",
                log_model=True,
                save_dir="results"
            )
        else:
            logger = None
        trainer = pl.Trainer(
            max_epochs=cfg.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=torch.cuda.device_count() or 1,
            precision="16-mixed" if torch.cuda.is_available() else 32,
            callbacks=default_callbacks(cfg),
            gradient_clip_val=1.0,
            logger=logger,
            log_every_n_steps=20,
            enable_progress_bar=True,
            enable_model_summary=True,
        )
        trainer.fit(model, dm)
        trainer.test(model, dm)
        print("Training completed. Model and scalers saved automatically by PyTorch Lightning.")
        print("Check wandb for the complete checkpoint with integrated scalers.")
    else:
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        model = Lit.load_from_checkpoint(str(ckpt_path))
        trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                             devices=torch.cuda.device_count() or 1,
                             enable_progress_bar=False,
                             enable_model_summary=False)
        import numpy as np
        raw_preds = trainer.predict(model, dm.test_dataloader())
        if raw_preds is None:
            raise RuntimeError("Predict returned None")
        preds_list = [p for p in raw_preds if p is not None]
        preds = np.concatenate(preds_list, axis=0).astype(np.float32)
        np.save(out_path, preds)


if __name__ == "__main__":
    main() 