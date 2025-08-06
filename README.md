# Alpaca Strategy - Machine Learning Trading System

A sophisticated machine learning-based trading system that uses transformer architecture to predict stock price movements and execute automated trades through the Alpaca Markets API.

## Table of Contents

- [Overview](#overview)
- [Trading Strategy](#trading-strategy)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Risk Management](#risk-management)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Overview

This system combines:
- **Machine Learning**: Transformer-based model trained on historical market data
- **Real-time Trading**: Live market data processing and automated execution
- **Risk Management**: Smart position sizing and portfolio controls
- **Backtesting**: Comprehensive historical performance validation

## Trading Strategy

### Core Strategy Implementation

The trading strategy is implemented in `scripts/trade_realtime_ws.py` and follows this exact process:

#### 1. Data Collection & Processing
```python
# Real-time trade data → 1-minute bars → feature engineering
BAR_BUFFERS: Dict[str, deque] = {sym: deque(maxlen=SEQ_LEN) for sym in SYMBOLS}
TRADE_BUFFERS: Dict[str, List[Dict]] = {sym: [] for sym in SYMBOLS}
```

#### 2. Model Prediction
```python
# Get model predictions for all symbols
preds = model.predict_batch(win)
raw_predictions: Dict[str, float] = {
    s: info["pred"] for s, info in preds.items()
}

# Rank symbols by prediction score (highest first)
ranked = sorted(
    ((s, p) for s, p in raw_predictions.items()),
    key=lambda kv: kv[1],
    reverse=True,
)
top_syms = [s for s, _ in ranked[:TOP_K]]  # Select top 3
```

#### 3. Technical Filtering
```python
def vectorized_candidate_filter(bar_buffers):
    """Filter trading candidates using technical indicators."""
    symbols, close_matrix = get_close_matrix(bar_buffers, lookback=15)
    ma5 = np.mean(close_matrix[:, -5:], axis=1)      # 5-period MA
    ma15 = np.mean(close_matrix, axis=1)             # 15-period MA
    momentum = (close_matrix[:, -1] - close_matrix[:, -6]) / (close_matrix[:, -6] + 1e-6)
    rsi14 = compute_rsi(close_matrix, period=14)
    
    # Filter: price > MA5 > MA15, positive momentum, RSI > 45
    mask = (close_matrix[:, -1] > ma5) & (ma5 > MA15) & (momentum > 0) & (rsi14 > 45)
    candidates = [s for s, m in zip(symbols, mask) if m]
    return candidates
```

#### 4. Position Management
```python
# Smart position management with equal-weight allocation
target_per_position = target_total_exposure / max_positions  # $50,000 / 3 = $16,667 per position

# Execute orders based on candidates vs current positions
run_async_orders(smart_position_management(
    candidates=candidates,
    trading_client=trading_client,
    target_total_exposure=TARGET_TOTAL_EXPOSURE,  # $50,000
    max_positions=MAX_POSITIONS,                  # 3 positions
    adjustment_threshold=ADJUSTMENT_THRESHOLD,    # $500
    hold_minutes=HOLD_MINUTES                     # 10 minutes
))
```

### Key Strategy Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TOP_K` | 6 | Number of top stocks to select |
| `MAX_POSITIONS` | 6 | Maximum concurrent positions |
| `TARGET_TOTAL_EXPOSURE` | $50,000 | Total portfolio exposure |
| `DECISION_INTERVAL_MINUTES` | 10 | Trading decision frequency |

| `ADJUSTMENT_THRESHOLD` | $500 | Minimum position adjustment amount |

### Decision Flow

1. **Every 10 minutes**: Check if new data is available
2. **ML Prediction**: Get probability scores for all 30+ stocks
3. **Rank Selection**: Select top 3 stocks by prediction score
4. **Technical Filter**: Apply moving average, momentum, and RSI filters
5. **Position Management**: Adjust positions to match candidates
6. **Order Execution**: Execute market orders via Alpaca API

## Model Architecture

### Neural Network Design

The model processes multi-stock time series data through:

```
Input Data (30 stocks × 240 minutes × 30 features)
         ↓
   Time2Vec Embedding (8D time features)
         ↓
   Feature Concatenation (38D total)
         ↓
   Temporal Transformer (2 layers, 8 heads)
         ↓
   Multi-Strategy Pooling (Last + EWMA + Attention)
         ↓
   Stock Embeddings + Cross-Sectional Transformer
         ↓
   Prediction Head (MLP)
         ↓
   Output Scores (30 probabilities)
```

### Key Components

#### 1. Time2Vec Encoder
- **Purpose**: Learnable temporal representations
- **Output**: 8-dimensional time embeddings
- **Benefits**: Captures intraday patterns and market cycles

#### 2. Feature Engineering (30 Features)
- **Price Data**: OHLC, VWAP, volume, trade count
- **Microstructure**: Order flow imbalance, trade size distributions
- **Technical**: Volatility proxy, buy/sell ratios, momentum
- **Time Features**: Minute-of-day normalization

#### 3. Temporal Encoder
- **Architecture**: 2-layer transformer with 8 attention heads
- **Hidden Size**: 256 dimensions
- **Pooling**: Multi-strategy (last state + EWMA + attention)

#### 4. Cross-Sectional Encoder
- **Stock Embeddings**: Learnable representations per stock
- **Cross-Attention**: 2-layer transformer for stock interactions

### Training Details
- **Loss Function**: Spearman correlation + diversity regularization
- **Optimizer**: AdamW with OneCycleLR scheduler
- **Batch Size**: 16 sequences
- **Training**: 300 epochs with early stopping

## Installation

### Prerequisites
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 10GB free storage space
- NVIDIA GPU with CUDA support (optional but recommended)

### Quick Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/alpaca-strategy.git
   cd alpaca-strategy
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package:**
   ```bash
   # Install in development mode (recommended)
   pip install -e .
   
   # Or install dependencies only
   pip install -r requirements.txt
   ```

4. **For GPU support:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Environment Configuration

Set up Alpaca API credentials in `alpaca_strategy/env.py`:
```python
DATA_KEY = "your_data_key"
DATA_SECRET = "your_data_secret"
TRADE_KEY = "your_trade_key"
TRADE_SECRET = "your_trade_secret"
DEBUG = True
```

## Usage

### Quick Start (Ready to Trade)

**If you have existing data and model files**, you can start live trading immediately:

```bash
# Install the package
pip install -e .

# Start live trading (PAPER TRADING by default)
alpaca-trade
# or
python scripts/trade_realtime_ws.py
```

The system will:
- Load the pre-trained model from `results/updated_model/updated_model_20250806.ckpt`
- Use existing market data from `data/` directory
- Start real-time trading with the 30+ stocks in the portfolio

### Complete Setup (If Starting Fresh)

#### 1. Data Collection

Collect historical market data for training:
```bash
alpaca-fetch-data
# or
python scripts/fetch_trade_data.py
```

This fetches 1-minute bar data for 30+ stocks and saves as parquet files.

#### 2. Model Training

Train the machine learning model:
```bash
alpaca-train
# or
python scripts/train.py
```

Monitor training progress:
```bash
tensorboard --logdir results/tensorboard_logs
```

#### 3. Backtesting

Test the strategy on historical data:
```bash
alpaca-backtest
# or
python scripts/backtest_with_model.py
```

Generates comprehensive performance report in `results/backtest_report.html`.

#### 4. Live Trading

Start real-time trading (PAPER TRADING by default):
```bash
alpaca-trade
# or
python scripts/trade_realtime_ws.py
```

Monitor live trading:
- Press `Ctrl+\` for status updates
- Check `results/logs/` for detailed logs

#### 5. Model Updates (Optional)

Update the model with new data:
```bash
alpaca-update-model
# or
python scripts/update_model_one_day.py
```

## Configuration

### Model Parameters (`alpaca_strategy/config.py`)

```python
# Data parameters
seq_len = 240          # Input sequence length (minutes)
horizon = 10           # Prediction horizon (minutes)
batch_size = 16        # Training batch size

# Model architecture
hidden = 256           # Hidden dimension
heads = 8              # Attention heads
dropout = 0.1          # Dropout rate

# Training parameters
epochs = 50           # Maximum training epochs
lr = 1e-5              # Learning rate
patience = 5          # Early stopping patience
```

### Trading Parameters (`scripts/trade_realtime_ws.py`)

```python
TOP_K = 6                    # Number of top stocks to select
MAX_POSITIONS = 6            # Maximum concurrent positions
TARGET_TOTAL_EXPOSURE = 50000  # Total portfolio exposure ($50,000)
DECISION_INTERVAL_MINUTES = 10  # Trading decision frequency (minutes)

ADJUSTMENT_THRESHOLD = 500   # Minimum position adjustment amount
```

## Project Structure

```
alpaca-strategy/
├── alpaca_strategy/           # Main package
│   ├── config.py             # Configuration management
│   ├── callbacks.py          # Training callbacks
│   ├── env.py               # Environment variables
│   ├── data/                # Data processing
│   │   ├── data_module.py   # PyTorch Lightning DataModule
│   │   ├── data_utils.py    # Data utilities and preprocessing
│   │   └── label_generator.py # Label generation for training
│   ├── model/               # Neural network models
│   │   ├── lit_module.py    # PyTorch Lightning wrapper
│   │   └── models_encoder.py # Transformer encoder architecture
│   └── trading/             # Trading logic
│       ├── trading.py       # Core trading functions
│       └── monitor.py       # System monitoring
├── scripts/                 # Execution scripts
│   ├── train.py            # Model training
│   ├── backtest_with_model.py # Historical backtesting
│   ├── trade_realtime_ws.py   # Live trading
│   ├── fetch_trade_data.py    # Data collection
│   └── update_model_one_day.py # Incremental learning
├── data/                   # Market data storage
├── results/               # Output and results
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
└── README.md             # This file
```

## Risk Management

### Built-in Safety Features

- **Paper Trading**: Default mode for testing
- **Position Limits**: Maximum 3 concurrent positions
- **Exposure Control**: Maximum $50,000 total portfolio exposure
- **Decision Interval**: 10-minute trading decision frequency
- **Adjustment Threshold**: Only adjust positions if difference > $500
- **Technical Filters**: Moving averages, momentum, and RSI validation

### Risk Monitoring

- Real-time position tracking
- Portfolio value monitoring
- Drawdown alerts
- API rate limit monitoring
- System health checks

## Monitoring

### Training Metrics
- **Loss Curves**: Training and validation loss
- **Learning Rate**: OneCycleLR scheduler visualization
- **Metrics**: Spearman correlation, diversity, IC
- **Gradients**: Gradient norm monitoring

### Trading Metrics
- **Portfolio Value**: Real-time equity curve
- **Position Tracking**: Current positions and P&L
- **Trade Logs**: Detailed trade history
- **Risk Metrics**: Drawdown, Sharpe ratio, win rate

### System Monitoring
- **Stream Health**: Data feed status and latency
- **API Usage**: Alpaca API rate limits and errors
- **Memory Usage**: System resource monitoring
- **Error Logs**: Exception handling and recovery

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Memory Issues:**
   - Reduce batch size in `alpaca_strategy/config.py`
   - Use smaller sequence length or fewer stocks

3. **Alpaca API Issues:**
   - Verify API keys are correct
   - Check account status and permissions

4. **Data Loading Issues:**
   ```bash
   ls data/
   python -c "import pandas as pd; pd.read_parquet('data/AAPL_1min.parquet')"
   ```

### Development Setup

For development work:
```bash
pip install -e .[dev]
black alpaca_strategy/ scripts/
flake8 alpaca_strategy/ scripts/
mypy alpaca_strategy/
```

## Important Notes

### Disclaimer
This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results.

### Paper Trading
The system defaults to paper trading mode. Only switch to live trading after thorough testing and understanding of the risks involved.

### API Limits
Alpaca API has rate limits. The system includes built-in rate limiting and error handling, but monitor usage to avoid hitting limits.

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in the code comments
- Review the troubleshooting section in this README

## License

This project is licensed under the MIT License - see the LICENSE file for details. 