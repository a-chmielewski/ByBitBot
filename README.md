# ByBit Trading Bot

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-blue.svg)](https://github.com/yourusername/bybit-trading-bot)

A sophisticated, modular cryptocurrency trading bot designed for automated trading on ByBit's perpetual futures market. Features advanced risk management, dynamic strategy selection, and comprehensive performance tracking.

## 🚀 Features

### Core Functionality
- **Automated Trading**: Execute trades automatically based on technical analysis strategies
- **Multi-Strategy Support**: 15+ built-in strategies for different market conditions
- **Dynamic Strategy Selection**: Automatic strategy and timeframe selection based on market conditions
- **Real-time Market Analysis**: Continuous market condition monitoring and adaptation
- **Advanced Risk Management**: Position sizing, stop-loss, take-profit, and trailing stop management

### Technical Features
- **Modular Architecture**: Clean, extensible codebase following best practices
- **Comprehensive Logging**: Detailed logging for debugging and performance analysis
- **Performance Tracking**: Track win rates, PnL, drawdown, and other key metrics
- **Session Management**: Persistent session tracking and data export capabilities
- **Error Handling**: Robust error handling with retry mechanisms and recovery procedures

### Market Analysis
- **Strategy Matrix**: 5x5 matrix for optimal strategy selection based on market conditions
- **Multi-Timeframe Analysis**: Support for 1-minute and 5-minute timeframes
- **Market Regime Detection**: Automatic detection of trending, ranging, and transitional markets
- **Volatility Analysis**: Real-time volatility assessment and strategy adjustment

## 📋 Prerequisites

- **Python 3.10 or higher**
- **ByBit API credentials** (API key and secret)
- **Windows** (development) or **Linux/Ubuntu** (production)
- **Internet connection** for real-time data feeds

### System Requirements
- **Windows**: Microsoft C++ Build Tools (for TA-Lib installation)
- **Linux**: `build-essential` package (for TA-Lib compilation)
- **Memory**: Minimum 2GB RAM (4GB+ recommended)
- **Storage**: 1GB free space for logs and data

## 🛠️ Installation

### Quick Start (Automated Setup)

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/bybit-trading-bot.git
   cd bybit-trading-bot
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```

3. **Configure your API credentials**
   - Edit `config/config.json`
   - Add your ByBit API key and secret
   - Set `testnet: true` for testing, `false` for live trading

4. **Activate virtual environment and run**
   ```bash
   # Windows
   venv\Scripts\activate
   python bot.py
   
   # Linux/Mac
   source venv/bin/activate
   python bot.py
   ```

### Manual Installation

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install TA-Lib** (if automatic installation fails)
   
   **Windows:**
   - Download wheel from: https://github.com/cgohlke/talib-build/releases
   - Install: `pip install <downloaded-wheel-file>`
   
   **Linux:**
   ```bash
   sudo apt-get install libta-lib-dev
   pip install ta-lib
   ```

4. **Create configuration**
   ```bash
   cp config/config.json.example config/config.json
   # Edit config/config.json with your API credentials
   ```

## ⚙️ Configuration

### API Configuration (`config/config.json`)

```json
{
  "bybit": {
    "api_key": "your_api_key_here",
    "api_secret": "your_api_secret_here",
    "testnet": true,
    "demo": true
  },
  "default": {
    "coin_pair": "BTC/USDT",
    "leverage": 10,
    "timeframe": "1m",
    "retry_attempts": 3,
    "retry_delay_seconds": 5,
    "max_position_size": 0.01,
    "risk_per_trade": 0.02
  }
}
```

### Environment Variables (`.env`)

```bash
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
BYBIT_TESTNET=true
DEFAULT_COIN_PAIR=BTC/USDT
DEFAULT_LEVERAGE=10
LOG_LEVEL=INFO
```

## 🎯 Usage

### Starting the Bot

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Run the bot
python bot.py
```

### Interactive Setup

1. **Select Trading Strategy**: Choose from available strategies or let the bot auto-select
2. **Choose Trading Pair**: Select from analyzed symbols (BTC/USDT, ETH/USDT, etc.)
3. **Set Leverage**: Configure leverage (1-100x)
4. **Review Configuration**: Confirm settings before starting
5. **Start Trading**: Bot begins monitoring and executing trades

### Available Strategies

- **EMA Trend Rider**: Trend-following with ADX filter
- **Bollinger Mean Reversion**: Range-bound market strategy
- **ATR Momentum Breakout**: Volatility-based breakout strategy
- **RSI Range Scalping**: Short-term scalping strategy
- **Volatility Squeeze Breakout**: High-volatility breakout strategy
- **And 10+ more strategies...**

## 📊 Strategy Matrix

The bot uses a sophisticated 5x5 Strategy Matrix for automatic strategy selection:

| 5m Market | 1m Market | Strategy | Timeframe |
|-----------|-----------|----------|-----------|
| TRENDING | TRENDING | EMA Trend Rider | 5m |
| TRENDING | RANGING | Adaptive Transitional | 1m |
| RANGING | TRENDING | Breakout & Retest | 1m |
| HIGH_VOL | LOW_VOL | Volatility Reversal | 1m |
| LOW_VOL | HIGH_VOL | Micro Range Scalping | 1m |

## 🔧 Project Structure

```
ByBitBot/
├── bot.py                      # Main application entry point
├── config/                     # Configuration files
├── modules/                    # Core modules
│   ├── exchange.py            # ByBit API wrapper
│   ├── data_fetcher.py        # Real-time data management
│   ├── order_manager.py       # Order execution and management
│   ├── performance_tracker.py # Performance metrics
│   ├── market_analyzer.py     # Market condition analysis
│   ├── strategy_matrix.py     # Strategy selection logic
│   └── ...                    # Additional modules
├── strategies/                 # Trading strategies
│   ├── strategy_template.py   # Base strategy class
│   ├── ema_adx_strategy.py    # EMA trend strategy
│   ├── bollinger_mean_reversion_strategy.py
│   └── ...                    # Additional strategies
├── tests/                     # Unit tests
├── logs/                      # Log files
├── performance/               # Performance data
├── sessions/                  # Session tracking
└── requirements.txt           # Python dependencies
```

## 📈 Performance Tracking

The bot automatically tracks:

- **Win Rate**: Percentage of profitable trades
- **Profit/Loss**: Per trade and cumulative
- **Drawdown**: Maximum and current drawdown
- **Sharpe Ratio**: Risk-adjusted returns
- **Trade Duration**: Average holding periods
- **Risk Metrics**: Position sizing and exposure

Performance data is exported to CSV and JSON formats for analysis.

## 🛡️ Risk Management

### Built-in Safety Features

- **Position Sizing**: Automatic calculation based on account equity and risk
- **Stop Loss**: ATR-based dynamic stop losses
- **Take Profit**: Multi-level profit targets
- **Trailing Stops**: Dynamic trailing stop management
- **Maximum Drawdown**: Automatic shutdown on excessive losses
- **Order Validation**: Pre-trade validation and condition checking

### Risk Parameters

- **Risk per Trade**: Configurable percentage of account (default: 2%)
- **Maximum Position Size**: Limits on position size
- **Leverage Limits**: Configurable maximum leverage
- **Daily Loss Limits**: Automatic shutdown on daily loss thresholds

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_ema_adx_strategy.py

# Run with coverage
pytest --cov=strategies tests/
```

### Test Coverage

- **Strategy Tests**: Individual strategy logic testing
- **Module Tests**: Core module functionality testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Risk management and performance tracking

## 📝 Logging

### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General operational information
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical errors

### Log Files

- **Bot Logs**: `logs/bot_YYYY-MM-DD.log`
- **Strategy Logs**: `logs/strategy_<name>_YYYY-MM-DD.log`
- **Performance Logs**: `logs/performance_YYYY-MM-DD.log`
- **Error Logs**: `logs/errors_YYYY-MM-DD.log`

## 🔄 Session Management

### Session Features

- **Persistent Sessions**: Sessions survive bot restarts
- **Data Export**: Export session data in JSON/CSV formats
- **Session Recovery**: Automatic recovery of interrupted sessions
- **Performance Analysis**: Session-specific performance metrics

### Session Commands

```bash
# Export session data
python -c "from modules.session_manager import SessionManager; sm = SessionManager(); sm.export_session_data(['session_id'])"

# View active sessions
python -c "from modules.session_manager import SessionManager; sm = SessionManager(); print(sm.get_active_sessions())"
```

## 🚨 Important Warnings

### Security
- **Never commit API keys** to version control
- **Use testnet first** before live trading
- **Start with small amounts** to test strategies
- **Monitor the bot** regularly during operation

### Risk Disclaimer
- **Cryptocurrency trading is highly risky**
- **Past performance does not guarantee future results**
- **Only trade with money you can afford to lose**
- **This bot is for educational purposes**

### Legal Compliance
- **Check local regulations** before automated trading
- **Ensure compliance** with your jurisdiction's laws
- **Tax implications** may apply to trading profits

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-strategy`
3. **Make your changes** following the coding standards
4. **Add tests** for new functionality
5. **Submit a pull request**

### Coding Standards

- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use type annotations
- **Docstrings**: Document all public functions
- **Error Handling**: Implement robust error handling
- **Testing**: Maintain high test coverage

### Strategy Development

To add a new strategy:

1. **Create strategy file** in `strategies/` directory
2. **Inherit from StrategyTemplate**
3. **Implement required methods**:
   - `init_indicators()`
   - `_check_entry_conditions()`
   - `check_exit()`
4. **Add tests** in `tests/` directory
5. **Update strategy matrix** if needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help

- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the inline code documentation
- **Examples**: Review the strategy implementations

### Common Issues

1. **TA-Lib Installation**: See installation section for platform-specific instructions
2. **API Connection**: Verify API credentials and network connectivity
3. **Strategy Performance**: Review market conditions and strategy parameters
4. **Memory Usage**: Monitor system resources during operation

## 🙏 Acknowledgments

- **ByBit API**: For providing the trading platform
- **CCXT**: For exchange abstraction library
- **TA-Lib**: For technical analysis functions
- **Pandas**: For data manipulation
- **Open Source Community**: For various supporting libraries

## 📊 Performance Disclaimer

This trading bot is provided as-is for educational and research purposes. Trading cryptocurrencies involves substantial risk of loss and is not suitable for all investors. The high degree of leverage can work against you as well as for you. Before deciding to trade cryptocurrencies, you should carefully consider your investment objectives, level of experience, and risk appetite. The possibility exists that you could sustain a loss of some or all of your initial investment and therefore you should not invest money that you cannot afford to lose.

---

**Happy Trading! 🚀**

*Remember: The best strategy is the one you understand and can stick to consistently.*
