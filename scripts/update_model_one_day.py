# =============================================================================
# MODEL UPDATE SYSTEM FOR INCREMENTAL LEARNING
# =============================================================================
# This file implements a system for updating trained ML models with new data
# for incremental learning. It allows the model to adapt to changing market
# conditions by fine-tuning on recent data.
# =============================================================================

import os
import sys
from datetime import datetime, timedelta
from typing import Union, Optional

# Add the project root to Python path so we can import alpaca_strategy
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pytorch_lightning as pl
import torch
from alpaca_strategy.config import get_config
from alpaca_strategy.data.data_module import AllSymbolsDataModule
from alpaca_strategy.model.lit_module import Lit
from alpaca_strategy.callbacks import default_callbacks

# Optimize for Tensor Cores if using CUDA
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')

# Get global configuration
cfg = get_config()

# =============================================================================
# MODEL UPDATE FUNCTIONS
# =============================================================================
# These functions handle updating trained models with new data for
# incremental learning and adaptation to market changes.
# =============================================================================

def update_model_one_day(
    target_date: Union[str, datetime],
    checkpoint_path: Optional[str] = None,
    output_dir: str = "results/updated_model",
    epochs: int = 10,
    learning_rate: float = 1e-5,
    batch_size: Optional[int] = None,
    use_gpu: bool = True
) -> str:
    """
    Update a trained model with one day's worth of data for incremental learning.
    
    This function implements incremental learning by:
    1. Loading an existing trained model checkpoint
    2. Preparing new data for the specified date
    3. Fine-tuning the model on the new data
    4. Saving the updated model
    
    Args:
        target_date: Date to use for model update (string or datetime)
        checkpoint_path: Path to existing model checkpoint (None for new model)
        output_dir: Directory to save updated model
        epochs: Number of training epochs for fine-tuning
        learning_rate: Learning rate for fine-tuning (typically smaller than initial training)
        batch_size: Batch size for training (None to use config default)
        use_gpu: Whether to use GPU for training
        
    Returns:
        Path to the updated model checkpoint
        
    Raises:
        ValueError: If no training data is available
        RuntimeError: If model loading or training fails
    """
    
    # Convert target_date to string if it's a datetime object
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d")
    
    print(f"Starting model update for date: {target_date}")
    
    # Set up device for training
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for the target date
    # Use a 5-day window ending on target_date for more stable training
    start_time = target_date - timedelta(days=5)
    end_time = target_date
    
    print(f"Loading data from {start_time} to {end_time}")
    
    # Create data module with the specified date range
    dm = AllSymbolsDataModule(
        start_time=start_time,
        end_time=end_time,
        split_ratio=(1.0, 0.0, 0.0),  # Use all data for training (no validation/test split)
        batch_size=batch_size
    )
    
    try:
        dm.prepare_data()
        dm.setup("fit")
        print(f"Data prepared successfully. Train samples: {len(dm.train_ds)}")
    except Exception as e:
        print(f"Error preparing data: {e}")
        raise
    
    # Load existing model or create new one
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading existing model from: {checkpoint_path}")
        model = Lit.load_from_checkpoint(checkpoint_path)
    else:
        print("Creating new model (no checkpoint provided or found)")
        model = Lit()
    
    # Move model to device
    model = model.to(device)

    if learning_rate is not None:
        model.base_lr = learning_rate
    
    # Configure trainer for fine-tuning
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto" if use_gpu else "cpu",
        devices=1,
        logger=None,  # No logging for updates
        enable_checkpointing=False,  # Don't create checkpoint folders
        enable_model_summary=False,  # No model summary
        enable_progress_bar=True,  # Show progress bar
        default_root_dir=output_dir
    )
    
    # Train the model on new data
    print(f"Starting fine-tuning for {epochs} epochs...")
    trainer.fit(model, dm)
    
    # Save the updated model
    checkpoint_path_str = os.path.join(output_dir, f"updated_model_{target_date.strftime('%Y%m%d')}.ckpt")
    trainer.save_checkpoint(checkpoint_path_str)
    print(f"Updated model saved to: {checkpoint_path_str}")
    
    return checkpoint_path_str

# =============================================================================
# MAIN EXECUTION
# =============================================================================
# Example usage and main execution block for testing the update functions.
# =============================================================================

def main():
    """Example usage of the update functions."""
    
    try:
        checkpoint_path = update_model_one_day(
            target_date="2025-08-06",
            checkpoint_path="results/updated_model/last.ckpt",
            epochs=5,
            learning_rate=1e-5
        )
        print(f"Model updated successfully: {checkpoint_path}")
    except Exception as e:
        print(f"Error in single day update: {e}")


if __name__ == "__main__":
    main() 