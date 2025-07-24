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

from utils.config import get_config
cfg = get_config()
from utils.data_module import AllSymbolsDataModule
from utils.callbacks import default_callbacks
from model.lit_module import Lit


def parse_args():
    p = argparse.ArgumentParser(description="Micro-graph training / predict")
    p.add_argument("mode", choices=["train", "predict", "list"], help="run mode")
    p.add_argument("--ckpt", default="last.ckpt", help="checkpoint path")
    p.add_argument("--out", default="preds.npy", help="prediction output file")
    return p.parse_args()


def train_model(checkpoint_path=None, max_epochs=None, log_wandb=None, start_time=None, end_time=None, split_ratio=(0.8, 0.1, 0.1)):
    """
    Train the model using all available data (including today's data), or a specified interval.
    Args:
        checkpoint_path (str): Path to save the model checkpoint (default: as in cfg)
        max_epochs (int): Number of epochs to train (default: as in cfg)
        log_wandb (bool): Whether to use wandb logging (default: as in cfg)
        start_time (str or datetime): Start of data interval (inclusive)
        end_time (str or datetime): End of data interval (inclusive)
        split_ratio (tuple): (train, val, test) split fractions, e.g. (0.8, 0.1, 0.1)
    """
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger
    from utils.config import get_config
    from utils.data_module import AllSymbolsDataModule
    from utils.callbacks import default_callbacks
    from model.lit_module import Lit
    import torch

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    dm = AllSymbolsDataModule(start_time=start_time, end_time=end_time, split_ratio=split_ratio)
    dm.prepare_data()
    dm.setup("fit")

    scalers = getattr(dm, 'scalers', None)
    model = Lit(scalers=scalers)

    save_ckpt = checkpoint_path or cfg.checkpoint_path or "last.ckpt"
    epochs = max_epochs or cfg.epochs
    use_wandb = log_wandb if log_wandb is not None else cfg.log_wandb

    if use_wandb:
        logger = WandbLogger(
            project="micro-graph-v2",
            log_model=True,
            save_dir="results"
        )
    else:
        logger = None
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() or 1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=default_callbacks(cfg),
        gradient_clip_val=1.0,
        logger=logger,
        log_every_n_steps=20,
        enable_progress_bar=True,
        enable_model_summary=True,
        default_root_dir=str(pathlib.Path(save_ckpt).parent)
    )
    trainer.fit(model, dm)
    print(f"Training completed. Model and scalers saved to {save_ckpt} (and wandb if enabled).")


def main():
    # Enable Tensor Core optimization for better performance on NVIDIA GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    args = parse_args()

    dm = AllSymbolsDataModule()
    dm.prepare_data()
    dm.setup("fit")

    if args.mode == "train":
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
        ckpt_path = pathlib.Path(args.ckpt)
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
        np.save(args.out, preds)


if __name__ == "__main__":
    main() 