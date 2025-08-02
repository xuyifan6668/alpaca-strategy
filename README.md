# Alpaca Strategy

A machine learning-based trading strategy using PyTorch Lightning and Alpaca Markets API.

## Features

- Multi-stock time series prediction using transformer architecture
- Real-time trading with Alpaca Markets API
- Comprehensive backtesting framework
- Log return and top-k binary classification label generation
- **TensorBoard logging for training visualization**

## Training

To train the model:

```bash
python scripts/train.py
```

The model will be saved in the `results/` directory with checkpoints.

## TensorBoard Logging

Training logs are automatically saved to `results/tensorboard_logs/`. To view the training progress:

```bash
tensorboard --logdir results/tensorboard_logs
```

Then open your browser to `http://localhost:6006` to view:
- Training and validation loss curves
- Learning rate schedules
- Model graphs
- Other training metrics

## Configuration

Edit `alpaca_strategy/config.py` to modify:
- Model hyperparameters
- Data settings
- Training parameters
- Logging options

## Overview

This system combines:
- **ML Model**: Trained on historical market data to predict price movements
- **Real-time Trading**: Live market data processing and automated order execution
- **Backtesting**: Historical performance validation
- **Position Management**: Smart portfolio allocation and risk management

## Architecture

```
├── model/                 # ML model components
│   ├── lit_module.py     # PyTorch Lightning model
│   └── models_encoder.py # Model architecture
├── scripts/              # Main execution scripts
│   ├── train.py         # Model training
│   ├── backtest_with_model.py # Historical backtesting
│   ├── trade_realtime_ws.py   # Live trading
│   └── fetch_trade_data.py    # Data collection
├── utils/                # Utility modules
│   ├── config.py        # Configuration
│   ├── data_module.py   # Data loading
│   ├── trading.py       # Trading logic
│   ├── monitor.py       # System monitoring
│   └── ...
```

## Setup

### Prerequisites

```bash
pip install torch pytorch-lightning pandas numpy backtrader alpaca-py websockets
```

### Environment Variables

Create a `.env` file with your Alpaca credentials:

```bash
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_DATA_KEY=your_data_key
ALPACA_DATA_SECRET=your_data_secret
```

## Usage

### 1. Data Collection

Collect historical market data for training:

```bash
# Run from project root directory
python scripts/fetch_trade_data.py
```

### 2. Model Training

Train the ML model:

```bash
# Run from project root directory
python scripts/train.py train
```

This creates a `last.ckpt` file containing the trained model and scaler.

### 3. Backtesting

Test the strategy on historical data:

```bash
# Run from project root directory
python scripts/backtest_with_model.py
```

#### Example Backtest Results

The system generates comprehensive performance reports including:

**Performance Metrics:**
- **Final Portfolio Value**: $XXX,XXX (from $100,000 initial)
- **Total Return**: XX.X%
- **Sharpe Ratio**: X.XX (risk-adjusted returns)
- **Maximum Drawdown**: XX.X% (peak-to-trough decline)
- **Total Trades**: XXX
- **Win Rate**: XX.X%

**Strategy Configuration:**
- **Top-K Selection**: 3 stocks per decision
- **Minimum Probability**: 0.6 threshold
- **Hold Period**: 10 minutes
- **Decision Interval**: 10 minutes
- **Max Positions**: 6 concurrent positions

**Risk Management:**
- **Position Sizing**: Equal weight allocation
- **Technical Filters**: RSI, moving averages, momentum
- **Cooldown Periods**: 15 minutes after liquidations
- **Stop Loss**: Automatic position closure after hold period

**Detailed Reports:**
- Interactive HTML report: `results/backtest_report.html`
- Trade-by-trade analysis
- Equity curve visualization
- Risk metrics breakdown
- Performance attribution

### 4. Live Trading

Start real-time trading:

```bash
# Run from project root directory
python scripts/trade_realtime_ws.py
```

## Configuration

Key parameters in `utils/config.py`:

- `seq_len`: Input sequence length for the model
- `epochs`: Training epochs
- `batch_size`: Training batch size

Trading parameters in `scripts/trade_realtime_ws.py`:

- `TOP_K`: Number of top symbols to consider
- `MAX_POSITIONS`: Maximum concurrent positions
- `TARGET_TOTAL_EXPOSURE`: Total portfolio exposure target
- `HOLD_MINUTES`: Minimum position hold time

## Model Architecture

The system uses a sophisticated neural network architecture designed for multi-stock temporal prediction:

### Core Components

**1. Time2Vec Encoder**
- Learnable periodic representation for time features
- Captures cyclical patterns in market data (daily, weekly cycles)
- Outputs 8-dimensional time embeddings

**2. Feature Processing**
- **Input Features** (30 total):
  - Price data: OHLC, VWAP, volume, trade count
  - Microstructure: Order flow imbalance, trade size distributions
  - Technical indicators: Volatility proxy, buy/sell ratios
  - Time features: Minute-of-day normalization
- **Sequence Length**: 240 minutes (4 hours of market data)
- **Feature Dimension**: 30 features per time step

**3. Temporal Encoder**
- **Transformer Encoder**: 2 layers with 8 attention heads
- **Bidirectional LSTM**: Captures long-term temporal dependencies
- **Temporal Attention**: Weighted pooling across time steps
- **Hidden Dimension**: 256 units

**4. Cross-Sectional Encoder**
- **Stock Embeddings**: Learnable representations for each stock
- **Cross-Attention**: 1-layer transformer for stock interactions
- **Multi-stock Processing**: Handles 30 stocks simultaneously

**5. Prediction Head**
- **Score Network**: 2-layer MLP with ReLU activation
- **Output**: Single probability score per stock
- **Activation**: Sigmoid for 0-1 probability range

### Training Details

**Loss Function**:
- **Spearman Loss**: Rank correlation optimization
- **MSE Loss**: Direct regression (10% weight)
- **IC Penalty**: Information coefficient regularization

**Optimization**:
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: OneCycleLR scheduler (peak 3x base LR)
- **Batch Size**: 16 sequences
- **Training**: 300 epochs with early stopping

**Data Flow**:
```
Raw Trades → Minute Bars → Feature Engineering → Scaler → Model → Predictions
     ↓              ↓              ↓              ↓        ↓         ↓
  Real-time    Aggregation    Technical      Normalize  Neural    Probability
   Stream       (OHLCV)       Features       (Z-score)  Network    Scores
```

### Real-time Processing

The model maintains consistency between training and inference:
- **Same Scaler**: Uses identical normalization from training
- **Feature Pipeline**: Identical preprocessing for live data
- **Batch Processing**: Handles multiple stocks simultaneously
- **Memory Efficient**: Processes streaming data in real-time

## Trading Strategy

1. **Data Processing**: Real-time trade data → minute bars → feature engineering
2. **Prediction**: ML model predicts probability of positive returns
3. **Selection**: Top-K symbols by probability score
4. **Technical Filter**: RSI, moving averages, momentum checks
5. **Position Management**: Smart allocation and risk management
6. **Execution**: Market orders via Alpaca API

## Monitoring

The system includes:
- Real-time status monitoring (`Ctrl+\`)
- Trade activity logging
- Position tracking
- Performance metrics

## Safety Features

- Paper trading mode (default)
- Position size limits
- Cooldown periods after liquidations
- Maximum position limits
- Error handling and recovery

## File Structure

- `data/`: Historical market data
  - `*_1min.parquet`: Processed minute-level data for each symbol
  - `*_symbols_*.txt`: Failed symbol lists
- `results/`: All outputs and results
  - `last.ckpt`: Trained model checkpoint
  - `backtest_report.html`: Comprehensive backtest results and analysis
  - `logs/`: Trading logs and position history

### Backtest Report Contents

The `results/backtest_report.html` file contains:

**Performance Analysis:**
- Equity curve and drawdown charts
- Monthly and annual returns
- Risk metrics (Sharpe, Sortino, Calmar ratios)
- Rolling statistics and volatility analysis

**Trade Analysis:**
- Trade distribution and statistics
- Win/loss analysis by trade size
- Trade duration analysis
- Entry/exit timing analysis

**Risk Metrics:**
- Value at Risk (VaR) calculations
- Maximum drawdown periods
- Underwater periods analysis
- Risk-adjusted return metrics

**Portfolio Analysis:**
- Asset allocation over time
- Correlation analysis
- Sector exposure tracking
- Performance attribution

## Disclaimer

This is for educational purposes. Trading involves risk. Always test thoroughly before using real money. 