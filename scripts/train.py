# =============================================================================
# MODEL TRAINING PIPELINE
# =============================================================================
# This script implements the main training pipeline for the alpaca-strategy model.
# It orchestrates the complete training process including data loading, model
# initialization, training execution, and evaluation.
# =============================================================================

"""Training / prediction entry point.

Replaces the old monolithic pipeline CLI with a thin wrapper that wires
together the DataModule, Lit module, callbacks, and logger.
"""

from __future__ import annotations

import pathlib
import sys
import os
import multiprocessing as mp
# Add the project root to Python path so we can import alpaca_strategy
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from alpaca_strategy.config import get_config
cfg = get_config()
from alpaca_strategy.data.data_module import AllSymbolsDataModule
from alpaca_strategy.callbacks import default_callbacks
from alpaca_strategy.model.lit_module import Lit

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
# This function orchestrates the complete training pipeline including:
# - Data preparation and loading
# - Model initialization and configuration
# - Training execution with callbacks and logging
# - Model evaluation and testing
# =============================================================================

def main():
    """
    Main training function that orchestrates the complete model training pipeline.
    
    This function:
    1. Sets up hardware optimization and reproducibility
    2. Prepares and loads training data
    3. Initializes the model with proper configuration
    4. Configures training callbacks and logging
    5. Executes training and evaluation
    6. Saves the trained model and scalers
    """
    
    # Enable Tensor Core optimization for better performance on NVIDIA GPUs
    # This trades precision for performance on modern GPUs with Tensor Cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    
    # Set random seed for reproducibility across all components
    pl.seed_everything(42, workers=True)

    # Training mode configuration
    mode = "train"  # Can be extended to support "predict" mode
    ckpt_path = pathlib.Path("last.ckpt")  # Checkpoint path for model loading
    out_path = "preds.npy"  # Output path for predictions

    # =============================================================================
    # DATA PREPARATION
    # =============================================================================
    # Initialize and prepare the data module for training
    # This loads all stock data, applies preprocessing, and creates train/val/test splits
    dm = AllSymbolsDataModule()
    dm.prepare_data()  # Load and filter data from parquet files
    dm.setup("fit")    # Create train/validation/test datasets

    if mode == "train":
        # =============================================================================
        # MODEL INITIALIZATION
        # =============================================================================
        # Get scalers from data module for model initialization
        # Scalers are needed for proper data normalization during inference
        scalers = getattr(dm, 'scalers', None)
        
        # Initialize the PyTorch Lightning model with scalers
        # The model includes the encoder architecture and training logic
        model = Lit(scalers=scalers)
        
        # =============================================================================
        # LOGGING CONFIGURATION
        # =============================================================================
        # Set up TensorBoard logging if enabled in config
        # This provides training metrics visualization and monitoring
        if cfg.log_tensorboard:
            logger = TensorBoardLogger(
                save_dir="results",
                name="tensorboard_logs",
                log_graph=False  # Disable graph logging to save space
            )
        else:
            logger = None
        
        # =============================================================================
        # TRAINER CONFIGURATION
        # =============================================================================
        # Configure the PyTorch Lightning trainer with all necessary settings
        # This includes hardware acceleration, training parameters, and callbacks
        trainer = pl.Trainer(
            max_epochs=cfg.epochs,                    # Maximum training epochs
            accelerator="gpu" if torch.cuda.is_available() else "cpu",  # Hardware acceleration
            devices=torch.cuda.device_count() or 1,   # Number of GPUs to use
            precision="16-mixed" if torch.cuda.is_available() else 32,  # Mixed precision for speed
            callbacks=default_callbacks(cfg),         # Training callbacks (checkpointing, early stopping)
            gradient_clip_val=1.0,                    # Gradient clipping to prevent exploding gradients
            logger=logger,                            # TensorBoard logging
            log_every_n_steps=50,                     # Logging frequency
            enable_progress_bar=True,                 # Show training progress
            enable_model_summary=True,                # Display model architecture summary
        )
        
        # =============================================================================
        # TRAINING EXECUTION
        # =============================================================================
        # Execute the training process
        # This will train the model and automatically save checkpoints
        trainer.fit(model, dm)
        
        # =============================================================================
        # MODEL EVALUATION
        # =============================================================================
        # Evaluate the trained model on the test set
        # This provides final performance metrics
        trainer.test(model, dm)
        
        # Training completion messages
        print("Training completed. Model and scalers saved automatically by PyTorch Lightning.")
        print("Check results/tensorboard_logs for training logs and TensorBoard visualization.")
        
    else:
        # =============================================================================
        # PREDICTION MODE (Alternative execution path)
        # =============================================================================
        # This path is for loading a trained model and making predictions
        # Currently not the primary use case but available for future extension
        
        # Verify checkpoint exists
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        
        # Load trained model from checkpoint
        model = Lit.load_from_checkpoint(str(ckpt_path))
        
        # Configure trainer for prediction (minimal configuration)
        trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                             devices=torch.cuda.device_count() or 1,
                             enable_progress_bar=True,
                             enable_model_summary=True)
        
        # Generate predictions on test data
        import numpy as np
        raw_preds = trainer.predict(model, dm.test_dataloader())
        if raw_preds is None:
            raise RuntimeError("Predict returned None")
        
        # Process and save predictions
        preds_list = [p for p in raw_preds if p is not None]
        preds = np.concatenate(preds_list, axis=0).astype(np.float32)
        np.save(out_path, preds)

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
# Main execution block with proper multiprocessing setup
# =============================================================================

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for Windows compatibility
    # This ensures proper process isolation and prevents issues with CUDA
    mp.set_start_method("spawn", force=True)   
    main() 