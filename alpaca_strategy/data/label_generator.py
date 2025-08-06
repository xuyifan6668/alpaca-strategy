# =============================================================================
# LABEL GENERATION SYSTEM
# =============================================================================
# This module provides a pluggable interface for generating training labels
# from time series data. Different label generators can be used to create
# various types of targets for the machine learning model, such as:
# - Future returns (regression)
# - Binary classification (up/down)
# - Multi-class ranking (top-k selection)
# - Cross-sectional ranking
# =============================================================================

"""Label generation utilities.

A pluggable interface so datasets can swap their target construction logic
without altering the dataset internals.  The interface is deliberately simple:
callable close_prices -> numpy[window_count].
"""

from __future__ import annotations

import numpy as np
from typing import Protocol

# =============================================================================
# LABEL GENERATOR PROTOCOL
# =============================================================================
# Defines the interface that all label generators must implement.
# This allows for easy swapping of different label generation strategies.
# =============================================================================

class LabelGenerator(Protocol):
    """
    Protocol defining the interface for label generators.
    
    All label generators must implement this callable interface that converts
    a sequence of close prices into training labels. The interface is designed
    to be simple and flexible, allowing different strategies for target creation.
    
    The callable should handle both single time series and multi-stock matrices,
    returning appropriate labels for the given input format.
    """
    
    def __call__(self, close: np.ndarray, seq_len: int, horizon: int) -> np.ndarray: ...

# =============================================================================
# LOG RETURN LABEL GENERATOR
# =============================================================================
# The default label generator that creates future log returns as regression targets.
# This is the primary label generator used for the alpaca-strategy model.
# =============================================================================

class LogReturnLabelGenerator:
    """
    Default label generator that computes future log returns over a specified horizon.
    
    This generator creates regression targets by computing the log return between
    the current price and a future price at the specified horizon. It includes
    noise filtering to reduce the impact of micro-movements in the market.
    
    The log return is calculated as: log(future_price / current_price)
    
    Parameters
    ----------
    threshold : float, default 5e-4
        If abs(log-return) < threshold we set the label to 0 to reduce
        noise around micro-moves. This helps focus the model on meaningful
        price movements rather than market microstructure noise.
    """

    def __init__(self, threshold: float = 5e-4):
        """
        Initialize the log return label generator.
        
        Args:
            threshold: Minimum absolute log return to consider as a valid signal.
                      Returns below this threshold are set to 0 to reduce noise.
        """
        self.threshold = threshold

    def __call__(self, close: np.ndarray, seq_len: int, horizon: int) -> np.ndarray:
        """
        Generate log return labels from close price data.
        
        This function handles both single time series and multi-stock matrices,
        computing future log returns for each valid window position.
        
        Args:
            close: Close price data. Can be:
                  - 1D array: Single time series [T]
                  - 2D array: Multi-stock matrix [S, T] where S=stocks, T=time
            seq_len: Length of the input sequence window
            horizon: Number of time steps into the future to predict
            
        Returns:
            Labels array with shape:
            - For single time series: [nw] where nw = T - seq_len - horizon
            - For multi-stock: [nw, S] where nw = T - seq_len - horizon
            
        The function computes log returns for each valid window position and
        applies noise filtering based on the threshold parameter.
        """
        # Handle both single time series and multi-stock matrix
        if close.ndim == 1:
            # =============================================================================
            # SINGLE TIME SERIES CASE
            # =============================================================================
            # Process a single stock's time series
            close = np.squeeze(close)
            nw = len(close) - seq_len - horizon  # Number of valid windows
            out = np.zeros(nw, dtype=np.float32)
            
            # Compute log returns for each window position
            for i in range(nw):
                cur_idx = i + seq_len - 1      # Current price index (end of sequence)
                fut_idx = cur_idx + horizon    # Future price index
                
                if fut_idx < len(close):
                    # Compute log return: log(future_price / current_price)
                    ret = np.log(close[fut_idx]) - np.log(close[cur_idx])
                    
                    # Apply noise filtering: set small returns to 0
                    out[i] = 0.0 if abs(ret) < self.threshold else ret
            return out
        else:
            # =============================================================================
            # MULTI-STOCK MATRIX CASE
            # =============================================================================
            # Process multiple stocks simultaneously
            S, T = close.shape  # S = number of stocks, T = time steps
            nw = T - seq_len - horizon  # Number of valid windows
            out = np.zeros((nw, S), dtype=np.float32)
            
            # Compute log returns for each stock and window position
            for i in range(nw):
                for s in range(S):
                    cur_idx = i + seq_len - 1      # Current price index
                    fut_idx = cur_idx + horizon    # Future price index
                    
                    if fut_idx < T:
                        # Compute log return for this stock
                        ret = np.log(close[s, fut_idx]) - np.log(close[s, cur_idx])
                        
                        # Apply noise filtering
                        out[i, s] = 0.0 if abs(ret) < self.threshold else ret
            return out

# =============================================================================
# CLASSIFICATION LABEL GENERATORS
# =============================================================================
# Alternative label generators for classification tasks.
# These can be used for different types of prediction objectives.
# =============================================================================

class TopKClassLabelGenerator:
    """
    Label generator that converts returns into class indices by ranking.
    
    This generator creates classification targets by ranking stocks based on
    their future returns. For each sample, it assigns class index 0 to the
    highest return, 1 to the next highest, etc.
    
    This is useful for creating ranking-based classification targets where
    the model learns to predict the relative performance of stocks.
    
    Parameters
    ----------
    k : int
        Number of top classes to consider
    """
    
    def __init__(self, k: int = 3):
        """
        Initialize the top-k classification label generator.
        
        Args:
            k: Number of top classes to consider for ranking
        """
        self.k = k

    def __call__(self, close: np.ndarray, seq_len: int, horizon: int) -> np.ndarray:
        """
        Generate classification labels from close price data.
        
        This implementation provides a simple binary classification example.
        For multi-class ranking, the function would rank stocks by their
        future returns and assign class indices accordingly.
        
        Args:
            close: Close price data (single time series)
            seq_len: Length of the input sequence window
            horizon: Number of time steps into the future to predict
            
        Returns:
            Classification labels array of shape [nw] with class indices
        """
        nw = len(close) - seq_len - horizon
        out = np.zeros(nw, dtype=np.int64)
        
        for i in range(nw):
            cur_idx = i + seq_len - 1
            fut_idx = cur_idx + horizon
            
            if fut_idx < len(close):
                # Compute future return
                ret = np.log(close[fut_idx]) - np.log(close[cur_idx])
                
                # Simple binary classification: 0 for positive return, 1 for negative
                # For multi-class ranking, this would rank stocks and assign class indices
                out[i] = 0 if ret > 0 else 1
        return out

# To use: pass TopKClassLabelGenerator() as label_generator to WindowDataset

class TopKBinaryLabelGenerator:
    """
    Binary label generator for top-k stock selection.
    
    For each window, this generator returns a binary vector where 1 indicates
    the stock is in the top k for future return, and 0 otherwise. This is
    particularly useful for portfolio construction tasks where you want to
    identify the best performing stocks.
    
    This generator is intended to be used across all stocks in the dataset
    to create cross-sectional ranking targets.
    
    Parameters
    ----------
    k : int
        Number of top stocks to label as 1
    """
    
    def __init__(self, k: int = 3):
        """
        Initialize the top-k binary label generator.
        
        Args:
            k: Number of top stocks to label as 1 (winners)
        """
        self.k = k

    def __call__(self, close_matrix: np.ndarray, seq_len: int, horizon: int) -> np.ndarray:
        """
        Generate binary labels for top-k stock selection.
        
        This function computes future returns for all stocks and creates
        binary labels indicating which stocks are in the top k performers.
        
        Args:
            close_matrix: Close price matrix of shape (num_stocks, num_timesteps)
            seq_len: Length of the input sequence window
            horizon: Number of time steps into the future to predict
            
        Returns:
            Binary labels array of shape (nw, num_stocks) where:
            - 1.0 indicates the stock is in the top k for that window
            - 0.0 indicates the stock is not in the top k
        """
        # close_matrix: shape (num_stocks, num_timesteps)
        S, T = close_matrix.shape
        nw = T - seq_len - horizon
        labels = np.zeros((nw, S), dtype=np.float32)
        
        # Process each window
        for i in range(nw):
            # Compute future returns for all stocks in this window
            future_returns = np.zeros(S, dtype=np.float32)
            
            for s in range(S):
                cur_idx = i + seq_len - 1
                fut_idx = cur_idx + horizon
                
                if fut_idx < T:
                    # Compute log return for this stock
                    ret = np.log(close_matrix[s, fut_idx]) - np.log(close_matrix[s, cur_idx])
                    future_returns[s] = ret
            
            # Find indices of top k stocks by return
            topk_idx = future_returns.argsort()[-self.k:]
            
            # Label top k stocks as 1, others as 0
            labels[i, topk_idx] = 1.0
            
        return labels

# =============================================================================
# DEFAULT LABEL GENERATOR
# =============================================================================
# The default label generator used throughout the system.
# This can be overridden by passing a different generator to the dataset.
# =============================================================================

DEFAULT_LABEL_GEN = LogReturnLabelGenerator() 