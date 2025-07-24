# Alpaca-Strategy: ML-Driven Trading System

A production-grade, modular machine learning trading system for real-time and historical market data.

---

## Project Structure

```
alpaca-strategy/
│
├── alpaca_strategy/         # Main package: all core code
│   ├── config.py            # Central config (ALL_COLS, tickers, etc.)
│   ├── data/
│   │   ├── data_module.py
│   │   ├── data_utils.py
│   │   └── label_generator.py
│   ├── model/
│   │   ├── lit_module.py
│   │   └── models_encoder.py
│   ├── trading/
│   │   ├── trading.py
│   │   └── monitor.py
│   ├── callbacks.py
│   ├── metrics.py
│   └── utils_ranking.py
│
├── scripts/                 # CLI entry points
│   ├── train.py
│   ├── trade_realtime_ws.py
│   ├── fetch_trade_data.py
│   ├── backtest_with_model.py
│   └── ...
│
├── data/                    # Raw and processed data
├── results/                 # Model checkpoints, reports, logs
├── logs/                    # Log files
├── tests/                   # (optional) Unit/integration tests
├── README.md
├── requirements.txt
├── .gitignore
└── setup.py                 # (optional) for pip installable package
```

---

## Setup

### Prerequisites

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file with your Alpaca credentials:

```bash
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_DATA_KEY=your_data_key
ALPACA_DATA_SECRET=your_data_secret
```

---

## Usage

### 1. Data Collection

```bash
python scripts/fetch_trade_data.py
```

### 2. Model Training

```bash
python scripts/train.py train
```

### 3. Backtesting

```bash
python scripts/backtest_with_model.py
```

### 4. Live Trading

```bash
python scripts/trade_realtime_ws.py
```

---

## Configuration

- All core config is in `alpaca_strategy/config.py` as a dataclass.
- Access config in any script/module via:
  ```python
  from alpaca_strategy.config import get_config
  cfg = get_config()
  # Use cfg.ALL_COLS, cfg.tickers, etc.
  ```
- Key fields: `cfg.ALL_COLS`, `cfg.tickers`, `cfg.seq_len`, `cfg.epochs`, etc.

---

## Model & Data Pipeline

- **Data:** Minute-level parquet files in `data/`
- **Features:** See `cfg.ALL_COLS` in config
- **Model:** Transformer + LSTM + attention (see `alpaca_strategy/model/`)
- **Training:** PyTorch Lightning, configurable splits and intervals
- **Inference:** Real-time, batch, or backtest

---

## Trading Logic

- **Session management:** Robust handling of market open/close, holidays, and waiting
- **Position management:** Smart allocation, risk controls, cooldowns
- **Execution:** Alpaca API, paper trading by default

---

## Monitoring & Safety

- Real-time status, logging, and error handling
- Position and trade logs in `results/` and `logs/`
- Paper trading mode, position size limits, cooldowns, and more

---

## File/Folder Details

- `alpaca_strategy/`: All core code, organized by domain
- `scripts/`: Entry points only, no business logic
- `data/`, `results/`, `logs/`: Outputs, not code
- `.gitignore`: Ignores all data, logs, checkpoints, and OS artifacts

---

## Disclaimer

This project is for educational and research purposes only. Trading is risky—use at your own risk and always test thoroughly before deploying with real capital. 