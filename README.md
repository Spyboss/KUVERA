# Kuvera Grid Trading Bot v1.1 🚀

**AI-Enhanced Cryptocurrency Trading Bot with Advanced Machine Learning**

A sophisticated trading bot that combines traditional technical analysis with cutting-edge AI and machine learning for optimized cryptocurrency trading strategies.

## 🌟 Key Features

### 🤖 AI-Powered Trading
- **OpenRouter AI Integration**: Advanced sentiment analysis and strategy optimization with 15-minute frequency
- **Real-time AI Sentiment Scoring**: Dynamic position sizing based on market sentiment
- **AI-Enhanced Stop-Loss/Take-Profit**: Adaptive risk management using AI insights
- **Autonomous Trading Capabilities**: Optional fully automated trading with AI oversight
- **Diagnostic Mode**: Built-in OpenRouter API connectivity testing and validation
- **Startup Test Trading**: Automated test trade execution at startup for workflow verification

### 🧠 Advanced Machine Learning
- **XGBoost Integration**: Optimized for 8GB RAM with memory-efficient configurations
- **scikit-learn 1.3.0**: Hyperparameter optimization using GridSearchCV
- **Cross-Validation**: 3-fold CV for robust model selection
- **Performance Metrics**: Comprehensive evaluation with RMSE, R², and directional accuracy
- **Feature Engineering**: 11+ technical indicators for ML model training

### 📊 Enhanced Analytics
- **Real-time Performance Tracking**: Live P&L, win rates, and trade statistics
- **AI Trade Logging**: Separate logging for AI-influenced trades
- **Model Performance Monitoring**: Track ML model accuracy and predictions
- **Sentiment-Based UI**: Visual indicators for AI sentiment in the interface

### 🎮 Modern Interface
- **Rich Terminal UI**: Beautiful, real-time command-line interface
- **Gamification System**: Achievements, streaks, and milestone tracking
- **Live Market Data**: WebSocket integration for real-time price updates
- **Interactive Controls**: Keyboard shortcuts for trading control

### ⚡ Performance Optimizations
- **Memory Efficient**: Optimized for 8GB RAM systems
- **Async Operations**: Non-blocking AI analysis and trading operations
- **Caching System**: Smart balance and data caching for reduced API calls
- **Error Handling**: Robust fallback mechanisms for all components

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 8GB RAM (recommended)
- Binance API account
- OpenRouter API key (for AI features)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd kuvera-grid
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Setup environment variables:**
```bash
cp .env.template .env
# Edit .env with your API keys
```

4. **Configure trading parameters:**
```bash
# Edit config/config.yaml for your preferences
```

5. **Run the bot:**
```bash
python bot.py
```

6. **Run backtesting:**
```bash
python backtest.py
```

### 🔧 Advanced Features

#### Diagnostic Mode
Enable diagnostic mode to test OpenRouter API connectivity at startup:

1. Edit `config/config.yaml`:
```yaml
ai:
  diagnostic_mode: true  # Enable OpenRouter connectivity testing
```

2. The bot will automatically test API connectivity when started
3. Check logs for detailed API call status and responses

#### Startup Test Trading
Enable automatic test trades at startup to verify end-to-end functionality:

1. Edit `config/config.yaml`:
```yaml
trading:
  startup_test_enabled: true  # Enable startup test trades
```

2. Bot will execute a small test buy/sell cycle at startup (testnet only)
3. Test results are logged and displayed in the UI
4. Normal trading resumes after test completion

## 🔧 Troubleshooting

### Common Issues and Fixes

#### 1. UnicodeEncodeError in Backtest
**Problem:** `UnicodeEncodeError: 'charmap' codec can't encode character '🔍'`

**Solution:** The backtest module now uses UTF-8 encoding for all log files and replaces Unicode emojis with plain text alternatives:
- Logs are written with explicit UTF-8 encoding
- Unicode characters are replaced with ASCII equivalents (e.g., '🔍' → '[OPT]')
- Safe logging method handles encoding fallbacks automatically

#### 2. scikit-learn Feature Name Warnings
**Problem:** `X does not have valid feature names, but StandardScaler was fitted with feature names`

**Solution:** Feature names are now consistently maintained throughout the ML pipeline:
- StandardScaler is fitted with proper DataFrame column names
- Feature columns are stored as instance variables for consistency
- ML signal generation uses DataFrames with proper column names
- All transformations preserve feature name integrity

#### 3. Memory Issues on 8GB RAM
**Problem:** Out of memory errors during ML training or backtesting

**Solution:** Memory optimizations implemented:
- Reduced GridSearchCV cross-validation folds from 3 to 2
- Limited XGBoost parameters: n_estimators=50, max_depth=4
- Single-threaded processing (n_jobs=1)
- Efficient tree method ('hist') for XGBoost
- Progress bars to show processing status

#### 4. Backtest Interruption
**Problem:** KeyboardInterrupt or premature termination

**Solution:** Enhanced user experience:
- Progress bars show real-time processing status
- Reduced processing time through optimized parameters
- Clear status indicators for each processing step
- Graceful error handling and logging

### Performance Tips

1. **For 8GB RAM systems:**
   - Use default ML parameters (already optimized)
   - Close other applications during backtesting
   - Consider shorter date ranges for initial testing

2. **For faster backtesting:**
   - Use higher timeframes (1h instead of 5m)
   - Reduce date range for testing
   - Disable AI features temporarily if needed

3. **Log file management:**
   - Logs are saved to `logs/backtest_YYYYMMDD.log`
   - Results are saved to `backtest_results/`
   - Charts are automatically generated and saved

## 🔧 Configuration

### Environment Variables (.env)
```env
# Binance API (Required)
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
BINANCE_TESTNET_API_KEY=your_testnet_api_key
BINANCE_TESTNET_SECRET_KEY=your_testnet_secret_key

# OpenRouter AI (Optional but recommended)
OPENROUTER_API_KEY=your_openrouter_api_key

# Trading Overrides (Optional)
TRADING_SYMBOL=BTCUSDT
AUTONOMOUS_TRADING=false
```

### AI Configuration (config.yaml)
```yaml
ai:
  openrouter: true      # Enable/disable AI features
  frequency: 15m        # AI analysis frequency

trading:
  symbol: "BTCUSDT"
  risk_per_trade: 0.01  # 1% risk per trade
  max_position_size: 0.30  # Max $0.30 per trade

strategy:
  type: "mean_reversion"
  entry_threshold: 0.02  # 2% below SMA for entry
  exit_threshold: 0.015  # 1.5% above SMA for exit
  stop_loss: 0.03       # 3% stop loss
```

## 🎯 Trading Strategies

### Mean Reversion Enhanced
- **Technical Indicators**: SMA, RSI, Bollinger Bands, MACD
- **ML Predictions**: XGBoost price direction forecasting
- **AI Sentiment**: Market sentiment analysis for position sizing
- **Risk Management**: Dynamic stop-loss based on AI confidence

### Entry Conditions
- Price below SMA threshold (configurable)
- RSI oversold (< 30)
- Price near Bollinger Band lower
- Positive ML signal
- Bullish AI sentiment (optional)

### Exit Conditions
- Take profit above SMA threshold
- RSI overbought (> 70)
- Price near Bollinger Band upper
- Negative ML signal
- AI-driven exit signal
- Stop-loss triggered

## 🤖 AI Features

### OpenRouter Integration
- **Strategy Optimization**: AI-powered parameter tuning
- **Sentiment Analysis**: Real-time market sentiment scoring
- **Exit Strategy Analysis**: AI-enhanced sell decision making
- **Risk Assessment**: Dynamic risk adjustment based on market conditions

### Machine Learning Pipeline
1. **Data Collection**: Historical price and volume data
2. **Feature Engineering**: Technical indicators and price patterns
3. **Model Training**: XGBoost with GridSearchCV optimization
4. **Prediction**: Real-time price direction forecasting
5. **Integration**: ML signals combined with traditional TA

## 📊 Performance Monitoring

### Real-time Metrics
- **Trade Statistics**: Win rate, profit/loss, trade count
- **AI Sentiment**: Current market sentiment score (0-1)
- **ML Performance**: Model accuracy and prediction confidence
- **Risk Metrics**: Daily loss tracking and position sizing

### Logging
- **Main Log**: `logs/trading_YYYYMMDD.log`
- **AI Trades**: `logs/ai_trades.log`
- **Backtest Results**: `logs/backtest_YYYYMMDD.log`

## 🧪 Backtesting

Run comprehensive backtests with AI and ML integration:

```bash
python backtest.py
```

### Backtest Features
- **Historical Data**: Binance API integration
- **ML Model Training**: Automated hyperparameter optimization
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate
- **Visual Reports**: Equity curves and trade analysis

## 🎮 Controls

### Keyboard Shortcuts
- **`s`**: Start/Stop trading
- **`m`**: Toggle display mode
- **`q`**: Quit application
- **`Ctrl+C`**: Emergency stop

### Safety Features
- **Testnet Mode**: Safe testing environment
- **Daily Limits**: Maximum trades and loss limits
- **Emergency Stop**: Immediate position closure
- **API Rate Limiting**: Prevents API abuse

## 🔒 Security

- **API Key Protection**: Environment variable storage
- **Testnet First**: Always test before live trading
- **Risk Limits**: Built-in position and loss limits
- **Error Handling**: Graceful failure recovery

## 📈 Performance Targets

- **Weekly Target**: $1-2 profit per week
- **Risk per Trade**: 1% of capital (max $0.30)
- **Win Rate Target**: 60%+ with proper risk management
- **Max Drawdown**: <5% of capital

## 🛠️ Development

### Project Structure
```
kuvera-grid/
├── bot.py                 # Main trading bot
├── backtest.py           # Backtesting engine
├── config/
│   └── config.yaml       # Trading configuration
├── ai/
│   ├── ai_strategy_optimizer.py  # OpenRouter AI integration
│   ├── auto_trader.py            # Autonomous trading
│   └── emergency_stop.py         # Safety systems
├── strategy/
│   └── mean_reversion.py         # Trading strategies
├── utils/
│   └── data_manager.py           # Data utilities
└── tests/
    └── quick_test.py             # Test suite
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## ⚠️ Disclaimer

**This software is for educational purposes only. Cryptocurrency trading involves significant risk of loss. Never trade with money you cannot afford to lose. The developers are not responsible for any financial losses incurred through the use of this software.**

## 📞 Support

- **Author**: Uminda
- **Email**: Uminda.h.aberathne@gmail.com
- **Version**: 1.1
- **License**: MIT

---

**Happy Trading! 🚀📈**