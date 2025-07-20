"""Training / prediction entry point.

Replaces the old monolithic pipeline CLI with a thin wrapper that wires
together the DataModule, Lit module, callbacks, and logger.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.config import cfg
from utils.data_module import AllSymbolsDataModule
from utils.callbacks import default_callbacks
from model.lit_module import Lit


def parse_args():
    p = argparse.ArgumentParser(description="Micro-graph training / predict")
    p.add_argument("mode", choices=["train", "predict"], help="run mode")
    p.add_argument("--ckpt", default="last.ckpt", help="checkpoint path")
    p.add_argument("--out", default="preds.npy", help="prediction output file")
    return p.parse_args()


def main():
    pl.seed_everything(42, workers=True)

    args = parse_args()

    dm = AllSymbolsDataModule()
    dm.prepare_data()
    dm.setup("fit")

    if args.mode == "train":
        model = Lit()
        logger = WandbLogger(project="micro-graph-v0") if cfg.log_wandb else None
        trainer = pl.Trainer(
            max_epochs=cfg.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=torch.cuda.device_count() or 1,
            precision="16-mixed" if torch.cuda.is_available() else 32,
            callbacks=default_callbacks(cfg),
            gradient_clip_val=1.0,
            logger=logger,
            log_every_n_steps=20,
            enable_progress_bar=True,  # keep console output compact
            enable_model_summary=True,
        )
        trainer.fit(model, dm)
        trainer.test(model, dm)
        # Attach the scaler to the checkpoint dict directly, not as a model attribute
        scaler = getattr(dm, 'global_scaler', None)
        
        # Save to results folder
        import os
        os.makedirs('results', exist_ok=True)
        checkpoint_path = os.path.join('results', args.ckpt)
        torch.save({
            'state_dict': model.state_dict(),
            'scaler': scaler
        }, checkpoint_path)
        print(f"Model saved to: {checkpoint_path}")
    else:
        ckpt_path = pathlib.Path(args.ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        model = Lit.load_from_checkpoint(str(ckpt_path))
        trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                             devices=torch.cuda.device_count() or 1,
                             enable_progress_bar=False,
                             enable_model_summary=False)
        # trainer.predict returns a list of numpy arrays (via `predict_step`)
        import numpy as np
        raw_preds = trainer.predict(model, dm.test_dataloader())
        if raw_preds is None:
            raise RuntimeError("Predict returned None")

        preds_list = [p for p in raw_preds if p is not None]  # ensure no None values
        preds = np.concatenate(preds_list, axis=0).astype(np.float32)
        # now `preds` is a single ndarray that can be saved directly
        np.save(args.out, preds)


if __name__ == "__main__":
    main() 