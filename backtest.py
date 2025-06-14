#!/usr/bin/env python3
"""
Kuvera Grid Trading Bot - Backtesting Module v1.1 🚀
Comprehensive backtesting for trading strategies with performance metrics
Author: Uminda
Email: Uminda.h.aberathne@gmail.com
"""

import asyncio
import json
import logging
import os
import time
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich import box
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Technical analysis imports
try:
    import talib
except ImportError:
    talib = None

# ML imports
try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError:
    xgb = None
    StandardScaler = None
    train_test_split = None
    GridSearchCV = None
    cross_val_score = None
    mean_squared_error = None
    r2_score = None

class BacktestEngine:
    """Enhanced backtesting engine with comprehensive metrics"""
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the backtesting engine"""
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        # Setup logging
        self.setup_logging()
        
        # Initialize console for rich output
        self.console = Console()
        
        # Trading parameters
        self.symbol = self.config['trading']['symbol']
        self.initial_capital = 1000.0  # Starting with $1000 for backtesting
        self.risk_per_trade = self.config['trading']['risk_per_trade']
        self.max_position_size = self.config['trading']['max_position_size']
        
        # Strategy parameters
        self.strategy_config = self.config['strategy']
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
        # ML components
        self.ml_model = None
        self.scaler = None
        self.setup_ml_components()
        
    def setup_logging(self):
        """Setup logging configuration with UTF-8 encoding"""
        os.makedirs('logs', exist_ok=True)
        
        # Create file handler with UTF-8 encoding
        file_handler = logging.FileHandler(
            f'logs/backtest_{datetime.now().strftime("%Y%m%d")}.log',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Configure logger
        self.logger = logging.getLogger('BacktestEngine')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
        
    def safe_log(self, message, level="info"):
        """Safe logging method that handles Unicode encoding issues"""
        try:
            # Try to log the original message
            if level == "info":
                self.logger.info(message)
            elif level == "error":
                self.logger.error(message)
            elif level == "warning":
                self.logger.warning(message)
            elif level == "debug":
                self.logger.debug(message)
        except UnicodeEncodeError:
            # Fallback: replace problematic Unicode characters
            safe_message = message.encode('ascii', 'replace').decode('ascii')
            if level == "info":
                self.logger.info(safe_message)
            elif level == "error":
                self.logger.error(safe_message)
            elif level == "warning":
                self.logger.warning(safe_message)
            elif level == "debug":
                self.logger.debug(safe_message)
        
    def setup_ml_components(self):
        """Setup enhanced machine learning components with hyperparameter optimization"""
        if xgb and StandardScaler and GridSearchCV:
            # Base XGBoost model with memory optimization for 8GB RAM
            self.ml_model = xgb.XGBRegressor(
                n_estimators=50,  # Reduced for memory efficiency
                max_depth=4,      # Reduced depth to prevent overfitting
                learning_rate=0.1,
                subsample=0.8,    # Use 80% of data for each tree
                colsample_bytree=0.8,  # Use 80% of features
                random_state=42,
                n_jobs=1,         # Single thread for memory efficiency
                tree_method='hist'  # Memory efficient tree method
            )
            
            # Hyperparameter grid for optimization
            self.param_grid = {
                'n_estimators': [30, 50, 100],
                'max_depth': [3, 4, 6],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9]
            }
            
            self.scaler = StandardScaler()
            self.best_model = None
            self.model_performance = {}
            
            self.logger.info("Enhanced ML components with GridSearchCV initialized for backtesting")
        else:
            self.logger.warning("ML libraries not available, using basic strategy only")
            
    def optimize_model_parameters(self, X_train, y_train):
        """Optimize XGBoost hyperparameters using GridSearchCV"""
        if not (xgb and GridSearchCV and cross_val_score):
            self.logger.warning("GridSearchCV not available, using default parameters")
            return self.ml_model
            
        try:
            # Use safe logging to avoid Unicode encoding issues
            self.safe_log("[OPT] Starting hyperparameter optimization...")
            
            # Use smaller parameter grid for memory efficiency (8GB RAM optimization)
            reduced_param_grid = {
                'n_estimators': [30, 50],  # Reduced options
                'max_depth': [3, 4],       # Reduced depth
                'learning_rate': [0.1, 0.2]  # Reduced options
            }
            
            # GridSearchCV with reduced CV folds for memory efficiency
            grid_search = GridSearchCV(
                estimator=self.ml_model,
                param_grid=reduced_param_grid,
                cv=2,  # Reduced from 3 to 2 for better memory efficiency
                scoring='neg_mean_squared_error',
                n_jobs=1,  # Single job for memory efficiency
                verbose=0
            )
            
            # Fit the grid search
            grid_search.fit(X_train, y_train)
            
            # Store best model and performance metrics
            self.best_model = grid_search.best_estimator_
            self.model_performance = {
                'best_params': grid_search.best_params_,
                'best_score': -grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            self.logger.info(f"✅ Best parameters: {grid_search.best_params_}")
            self.logger.info(f"✅ Best CV score: {-grid_search.best_score_:.4f}")
            
            return self.best_model
            
        except Exception as e:
            self.logger.error(f"❌ Error in hyperparameter optimization: {e}")
            return self.ml_model
            
    def evaluate_model_performance(self, X_test, y_test):
        """Evaluate model performance with comprehensive metrics"""
        if not (self.best_model and mean_squared_error and r2_score):
            return {}
            
        try:
            # Make predictions
            y_pred = self.best_model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate directional accuracy
            direction_actual = np.sign(y_test)
            direction_pred = np.sign(y_pred)
            directional_accuracy = np.mean(direction_actual == direction_pred)
            
            performance_metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2,
                'directional_accuracy': directional_accuracy,
                'mean_prediction': np.mean(y_pred),
                'std_prediction': np.std(y_pred)
            }
            
            self.logger.info(f"📊 Model Performance - RMSE: {rmse:.4f}, R²: {r2:.4f}, Dir. Acc: {directional_accuracy:.2%}")
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating model performance: {e}")
            return {}
            
    def fetch_historical_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data from Binance public API"""
        self.console.print(f"[cyan]📊 Fetching historical data for {symbol}...[/cyan]")
        
        # Convert dates to timestamps
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
        
        # Binance public API endpoint
        url = "https://api.binance.com/api/v3/klines"
        
        all_data = []
        current_start = start_ts
        
        # Fetch data in chunks (max 1000 candles per request)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            total_duration = end_ts - start_ts
            task = progress.add_task("Downloading data...", total=100)
            
            while current_start < end_ts:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'endTime': min(current_start + (1000 * self._interval_to_ms(interval)), end_ts),
                    'limit': 1000
                }
                
                try:
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    if not data:
                        break
                        
                    all_data.extend(data)
                    current_start = data[-1][6] + 1  # Next start time
                    
                    # Update progress
                    progress_pct = ((current_start - start_ts) / total_duration) * 100
                    progress.update(task, completed=min(progress_pct, 100))
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except requests.exceptions.RequestException as e:
                    self.logger.error(f"Error fetching data: {e}")
                    break
                    
        # Convert to DataFrame
        if all_data:
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            self.console.print(f"[green]✅ Downloaded {len(df)} candles[/green]")
            return df
        else:
            raise ValueError("No data retrieved")
            
    def _interval_to_ms(self, interval: str) -> int:
        """Convert interval string to milliseconds"""
        interval_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        return interval_map.get(interval, 5 * 60 * 1000)
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataset"""
        self.console.print("[cyan]🔧 Calculating technical indicators...[/cyan]")
        
        if talib is None:
            self.logger.warning("TA-Lib not available, using simple indicators")
            # Simple moving average
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['rsi'] = 50  # Placeholder
            df['bb_upper'] = df['close'] * 1.02
            df['bb_lower'] = df['close'] * 0.98
            return df
            
        try:
            # Simple Moving Average
            df['sma_20'] = talib.SMA(df['close'].values, timeperiod=20)
            
            # RSI
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2
            )
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # Volume indicators
            df['volume_sma'] = talib.SMA(df['volume'].values, timeperiod=20)
            
            self.console.print("[green]✅ Technical indicators calculated[/green]")
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            
        return df
        
    def prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning model"""
        if self.ml_model is None:
            return df
            
        self.console.print("[cyan]🤖 Preparing ML features...[/cyan]")
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        
        # Volatility features
        df['volatility'] = df['price_change'].rolling(window=20).std()
        
        # Relative position features
        df['close_to_sma'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['close_to_bb_upper'] = (df['close'] - df['bb_upper']) / df['bb_upper']
        df['close_to_bb_lower'] = (df['close'] - df['bb_lower']) / df['bb_lower']
        
        # Enhanced Volume features for better ML performance
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_change'] = df['volume'].pct_change()
        df['volume_momentum'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        df['price_volume_trend'] = (df['close'].pct_change() * df['volume_ratio']).rolling(5).mean()
        df['volume_volatility'] = df['volume'].rolling(10).std() / df['volume'].rolling(10).mean()
        
        # Additional technical features for improved R²
        df['rsi_momentum'] = df['rsi'].diff()
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['price_acceleration'] = df['close'].pct_change().diff()
        
        # Target variable (future return)
        df['future_return'] = df['close'].shift(-5).pct_change(5)
        
        return df
        
    def train_ml_model(self, df: pd.DataFrame):
        """Train the machine learning model with progress tracking"""
        if self.ml_model is None or train_test_split is None:
            return
            
        self.console.print("[cyan]🎯 Training ML model...[/cyan]")
        
        # Enhanced feature selection with volume and technical indicators for improved R²
        self.feature_columns = [
            'price_change', 'price_change_5', 'price_change_10',
            'volatility', 'rsi', 'close_to_sma', 'close_to_bb_upper',
            'close_to_bb_lower', 'volume_ratio', 'macd', 'macd_hist',
            # NEW: Enhanced volume features
            'volume_change', 'volume_momentum', 'price_volume_trend', 'volume_volatility',
            # NEW: Additional technical features
            'rsi_momentum', 'bb_position', 'price_acceleration'
        ]
        
        # Prepare data
        df_clean = df.dropna()
        X = df_clean[self.feature_columns]
        y = df_clean['future_return']
        
        if len(X) < 100:
            self.safe_log("Insufficient data for ML training", "warning")
            return
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features with proper feature names
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store feature names for consistent usage
        self.scaler.feature_names_in_ = self.feature_columns
        
        # Optimize hyperparameters using GridSearchCV
        optimized_model = self.optimize_model_parameters(X_train_scaled, y_train)
        
        # Use optimized model if available, otherwise use default
        if optimized_model:
            self.ml_model = optimized_model
        else:
            # Fallback to default training
            self.ml_model.fit(X_train_scaled, y_train)
        
        # Comprehensive model evaluation
        performance_metrics = self.evaluate_model_performance(X_test_scaled, y_test)
        
        # Basic evaluation for compatibility
        train_score = self.ml_model.score(X_train_scaled, y_train)
        test_score = self.ml_model.score(X_test_scaled, y_test)
        
        # Enhanced logging with performance metrics
        if performance_metrics:
            self.console.print(
                f"[green]✅ Enhanced ML model trained - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}[/green]\n"
                f"[cyan]📊 RMSE: {performance_metrics.get('rmse', 0):.4f}, "
                f"Directional Accuracy: {performance_metrics.get('directional_accuracy', 0):.2%}[/cyan]"
            )
        else:
            self.console.print(f"[green]✅ ML model trained - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}[/green]")
        
        # Log best parameters if optimization was successful
        if hasattr(self, 'model_performance') and self.model_performance:
            self.safe_log(f"[TARGET] Optimized parameters: {self.model_performance.get('best_params', {})}")
        
    def get_ml_signal(self, features: Dict[str, float]) -> float:
        """Get ML prediction signal using optimized model with consistent feature names"""
        # Use best_model if available, otherwise fall back to ml_model
        model_to_use = self.best_model if self.best_model is not None else self.ml_model
        
        if model_to_use is None or self.scaler is None:
            return 0.0
            
        try:
            # Use stored feature columns for consistency
            if not hasattr(self, 'feature_columns'):
                self.feature_columns = [
                    'price_change', 'price_change_5', 'price_change_10',
                    'volatility', 'rsi', 'close_to_sma', 'close_to_bb_upper',
                    'close_to_bb_lower', 'volume_ratio', 'macd', 'macd_hist',
                    # Enhanced volume features
                    'volume_change', 'volume_momentum', 'price_volume_trend', 'volume_volatility',
                    # Additional technical features
                    'rsi_momentum', 'bb_position', 'price_acceleration'
                ]
            
            # Create feature array with consistent ordering
            feature_values = [features.get(col, 0) for col in self.feature_columns]
            feature_array = np.array(feature_values).reshape(1, -1)
            
            # Create DataFrame with proper column names to avoid feature name warnings
            import pandas as pd
            feature_df = pd.DataFrame(feature_array, columns=self.feature_columns)
            feature_scaled = self.scaler.transform(feature_df)
            
            prediction = model_to_use.predict(feature_scaled)[0]
            return prediction
            
        except Exception as e:
            self.safe_log(f"Error getting ML signal: {e}", "error")
            return 0.0
            
    def should_buy_enhanced(self, row: pd.Series, ml_signal: float) -> bool:
        """Enhanced buy logic with optimized RSI thresholds and Bollinger Band sensitivity"""
        if pd.isna(row['sma_20']):
            return False
            
        conditions = []
        
        # Price below SMA threshold
        entry_threshold = self.strategy_config['entry_threshold']
        buy_threshold = row['sma_20'] * (1 - entry_threshold)
        conditions.append(row['close'] <= buy_threshold)
        
        # RSI oversold - OPTIMIZED: Changed from 30 to 40 for better entry timing
        if not pd.isna(row['rsi']):
            conditions.append(row['rsi'] < 40)
            
        # Price near Bollinger Band lower - OPTIMIZED: Increased sensitivity from 1.01 to 1.02
        if not pd.isna(row['bb_lower']):
            conditions.append(row['close'] <= row['bb_lower'] * 1.02)
            
        # ML signal positive - Enhanced threshold
        if ml_signal > 0.002:  # Slightly higher threshold for better signal quality
            conditions.append(True)
            
        # Volume confirmation - NEW: Add volume-based condition
        if not pd.isna(row.get('volume_ratio', np.nan)):
            conditions.append(row['volume_ratio'] > 1.1)  # Above average volume
            
        # Require at least 2 conditions
        return sum(conditions) >= 2
        
    def should_sell_enhanced(self, row: pd.Series, entry_price: float, ml_signal: float) -> Tuple[bool, str]:
        """Enhanced sell logic with optimized RSI thresholds and Bollinger Band sensitivity"""
        if pd.isna(row['sma_20']):
            return False, ""
            
        # Take profit condition
        exit_threshold = self.strategy_config['exit_threshold']
        sell_threshold = row['sma_20'] * (1 + exit_threshold)
        
        if row['close'] >= sell_threshold:
            return True, "TAKE_PROFIT"
            
        # RSI overbought - OPTIMIZED: Changed from 70 to 60 for better exit timing
        if not pd.isna(row['rsi']) and row['rsi'] > 60:
            return True, "RSI_OVERBOUGHT"
            
        # Price near Bollinger Band upper - OPTIMIZED: Increased sensitivity from 0.99 to 0.98
        if not pd.isna(row['bb_upper']) and row['close'] >= row['bb_upper'] * 0.98:
            return True, "BB_UPPER"
            
        # ML signal negative - Enhanced threshold
        if ml_signal < -0.002:  # Slightly lower threshold for better signal quality
            return True, "ML_SIGNAL"
            
        # Volume-based exit - NEW: High volume reversal signal
        if not pd.isna(row.get('volume_ratio', np.nan)) and row['volume_ratio'] > 2.0:
            # High volume with price near resistance could indicate reversal
            if not pd.isna(row['rsi']) and row['rsi'] > 55:
                return True, "VOLUME_REVERSAL"
            
        # Stop loss condition
        stop_loss_pct = self.strategy_config['stop_loss']
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        
        if row['close'] <= stop_loss_price:
            return True, "STOP_LOSS"
            
        return False, ""
        
    def calculate_position_size(self, current_price: float, capital: float) -> float:
        """Calculate position size based on risk management"""
        risk_amount = min(
            capital * self.risk_per_trade,
            self.max_position_size
        )
        
        stop_loss_pct = self.strategy_config['stop_loss']
        stop_loss_price = current_price * (1 - stop_loss_pct)
        risk_per_unit = current_price - stop_loss_price
        
        if risk_per_unit > 0:
            position_size = risk_amount / risk_per_unit
            max_units = self.max_position_size / current_price
            position_size = min(position_size, max_units)
        else:
            position_size = 0
            
        return position_size
        
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Run the backtest simulation"""
        self.console.print("[cyan]🚀 Running backtest simulation...[/cyan]")
        
        # Initialize variables
        capital = self.initial_capital
        position = None
        entry_price = None
        entry_time = None
        
        trades = []
        equity_curve = [capital]
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Backtesting...", total=len(df))
            
            for i, (timestamp, row) in enumerate(df.iterrows()):
                # Skip if insufficient data
                if pd.isna(row['sma_20']):
                    progress.update(task, advance=1)
                    continue
                    
                # Get ML signal
                ml_signal = self.get_ml_signal(row.to_dict())
                
                # Check buy signal
                if position is None and self.should_buy_enhanced(row, ml_signal):
                    position_size = self.calculate_position_size(row['close'], capital)
                    
                    if position_size > 0:
                        position = {
                            'size': position_size,
                            'value': position_size * row['close']
                        }
                        entry_price = row['close']
                        entry_time = timestamp
                        capital -= position['value']
                        
                # Check sell signal
                elif position is not None:
                    should_sell, reason = self.should_sell_enhanced(row, entry_price, ml_signal)
                    
                    if should_sell:
                        # Execute sell
                        sell_value = position['size'] * row['close']
                        profit = sell_value - position['value']
                        profit_pct = (row['close'] - entry_price) / entry_price
                        
                        # Record trade
                        trade = {
                            'entry_time': entry_time,
                            'exit_time': timestamp,
                            'entry_price': entry_price,
                            'exit_price': row['close'],
                            'size': position['size'],
                            'profit': profit,
                            'profit_pct': profit_pct,
                            'reason': reason,
                            'duration': (timestamp - entry_time).total_seconds() / 3600  # hours
                        }
                        trades.append(trade)
                        
                        # Update capital
                        capital += sell_value
                        
                        # Reset position
                        position = None
                        entry_price = None
                        entry_time = None
                        
                # Update equity curve
                current_equity = capital
                if position is not None:
                    current_equity += position['size'] * row['close']
                equity_curve.append(current_equity)
                
                progress.update(task, advance=1)
                
        # Final equity if still in position
        if position is not None:
            final_value = position['size'] * df.iloc[-1]['close']
            capital += final_value
            
        self.trades = trades
        self.equity_curve = equity_curve
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'final_capital': capital
        }
        
    def calculate_performance_metrics(self, results: Dict) -> Dict:
        """Calculate comprehensive performance metrics"""
        trades = results['trades']
        equity_curve = results['equity_curve']
        final_capital = results['final_capital']
        
        if not trades:
            return {
                'total_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }
            
        # Basic metrics
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['profit'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        profits = [t['profit'] for t in trades]
        avg_profit = np.mean(profits)
        avg_win = np.mean([p for p in profits if p > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([p for p in profits if p < 0]) if (total_trades - winning_trades) > 0 else 0
        
        # Profit factor
        gross_profit = sum([p for p in profits if p > 0])
        gross_loss = abs(sum([p for p in profits if p < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown calculation
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # Sharpe ratio (simplified)
        returns = equity_series.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
            
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
        
    def display_results(self, metrics: Dict):
        """Display backtest results in a formatted table"""
        # Performance Summary Table
        summary_table = Table(title="📊 Backtest Performance Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Return", f"{metrics['total_return']:.2%}")
        summary_table.add_row("Total Trades", str(metrics['total_trades']))
        summary_table.add_row("Winning Trades", str(metrics['winning_trades']))
        summary_table.add_row("Win Rate", f"{metrics['win_rate']:.1%}")
        summary_table.add_row("Average Profit", f"${metrics['avg_profit']:.2f}")
        summary_table.add_row("Average Win", f"${metrics['avg_win']:.2f}")
        summary_table.add_row("Average Loss", f"${metrics['avg_loss']:.2f}")
        summary_table.add_row("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        summary_table.add_row("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        summary_table.add_row("Profit Factor", f"{metrics['profit_factor']:.2f}")
        
        self.console.print(summary_table)
        
        # Trade Analysis Table
        if self.trades:
            trade_table = Table(title="📈 Recent Trades (Last 10)", box=box.ROUNDED)
            trade_table.add_column("Entry Time", style="cyan")
            trade_table.add_column("Exit Time", style="cyan")
            trade_table.add_column("Entry Price", style="yellow")
            trade_table.add_column("Exit Price", style="yellow")
            trade_table.add_column("Profit", style="green")
            trade_table.add_column("Profit %", style="green")
            trade_table.add_column("Reason", style="blue")
            
            for trade in self.trades[-10:]:  # Last 10 trades
                profit_color = "green" if trade['profit'] > 0 else "red"
                trade_table.add_row(
                    trade['entry_time'].strftime("%m-%d %H:%M"),
                    trade['exit_time'].strftime("%m-%d %H:%M"),
                    f"${trade['entry_price']:.2f}",
                    f"${trade['exit_price']:.2f}",
                    f"[{profit_color}]${trade['profit']:.2f}[/{profit_color}]",
                    f"[{profit_color}]{trade['profit_pct']:.2%}[/{profit_color}]",
                    trade['reason']
                )
                
            self.console.print(trade_table)
            
    def save_results(self, results: Dict, metrics: Dict):
        """Save backtest results to files"""
        os.makedirs('backtest_results', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trades to CSV
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(f'backtest_results/trades_{timestamp}.csv', index=False)
            
        # Save equity curve
        equity_df = pd.DataFrame({
            'equity': self.equity_curve
        })
        equity_df.to_csv(f'backtest_results/equity_curve_{timestamp}.csv', index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f'backtest_results/metrics_{timestamp}.csv', index=False)
        
        self.console.print(f"[green]💾 Results saved to backtest_results/ folder[/green]")
        
    def plot_results(self, df: pd.DataFrame):
        """Create and save performance plots"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Kuvera Grid Backtest Results', fontsize=16)
            
            # Price and signals
            ax1.plot(df.index, df['close'], label='Price', alpha=0.7)
            ax1.plot(df.index, df['sma_20'], label='SMA 20', alpha=0.7)
            ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.2, label='Bollinger Bands')
            
            # Mark trades
            for trade in self.trades[-50:]:  # Last 50 trades
                color = 'green' if trade['profit'] > 0 else 'red'
                ax1.scatter(trade['entry_time'], trade['entry_price'], color='blue', marker='^', s=50)
                ax1.scatter(trade['exit_time'], trade['exit_price'], color=color, marker='v', s=50)
                
            ax1.set_title('Price Chart with Signals')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Equity curve
            ax2.plot(self.equity_curve, label='Equity Curve')
            ax2.axhline(y=self.initial_capital, color='red', linestyle='--', label='Initial Capital')
            ax2.set_title('Equity Curve')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # RSI
            ax3.plot(df.index, df['rsi'], label='RSI')
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7)
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7)
            ax3.set_title('RSI Indicator')
            ax3.set_ylim(0, 100)
            ax3.grid(True, alpha=0.3)
            
            # Trade distribution
            if self.trades:
                profits = [t['profit'] for t in self.trades]
                ax4.hist(profits, bins=20, alpha=0.7, edgecolor='black')
                ax4.axvline(x=0, color='red', linestyle='--')
                ax4.set_title('Trade Profit Distribution')
                ax4.set_xlabel('Profit ($)')
                ax4.set_ylabel('Frequency')
                ax4.grid(True, alpha=0.3)
                
            plt.tight_layout()
            
            # Save plot
            os.makedirs('backtest_results', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'backtest_results/backtest_chart_{timestamp}.png', dpi=300, bbox_inches='tight')
            
            self.console.print(f"[green][CHART] Charts saved to backtest_results/backtest_chart_{timestamp}.png[/green]")
            
        except Exception as e:
            self.logger.error(f"Error creating plots: {e}")
            
async def main():
    """Main backtesting function"""
    console = Console()
    
    console.print(Panel.fit(
        "[bold cyan]Kuvera Grid Backtesting Engine v1.1 [ROCKET][/bold cyan]\n"
        "[dim]Comprehensive strategy testing with performance analytics[/dim]",
        box=box.DOUBLE
    ))
    
    try:
        # Initialize backtest engine
        engine = BacktestEngine()
        
        # Configuration
        symbol = "BTCUSDT"
        interval = "5m"
        start_date = "2024-01-01"
        end_date = "2024-12-31"
        
        console.print(f"\n[bold][CONFIG] Backtest Configuration:[/bold]")
        config_table = Table(box=box.ROUNDED)
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")
        
        config_table.add_row("Symbol", symbol)
        config_table.add_row("Interval", interval)
        config_table.add_row("Start Date", start_date)
        config_table.add_row("End Date", end_date)
        config_table.add_row("Initial Capital", f"${engine.initial_capital:,.2f}")
        config_table.add_row("Risk per Trade", f"{engine.risk_per_trade:.1%}")
        config_table.add_row("Strategy", "Enhanced Mean Reversion")
        
        console.print(config_table)
        
        # Execute backtest with progress tracking
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # Step 1: Fetch historical data
            data_task = progress.add_task("[cyan]Fetching historical data...", total=100)
            df = engine.fetch_historical_data(symbol, interval, start_date, end_date)
            progress.update(data_task, completed=100)
            
            # Step 2: Calculate technical indicators
            indicators_task = progress.add_task("[yellow]Calculating technical indicators...", total=100)
            df = engine.calculate_technical_indicators(df)
            progress.update(indicators_task, completed=100)
            
            # Step 3: Prepare ML features
            features_task = progress.add_task("[blue]Preparing ML features...", total=100)
            df = engine.prepare_ml_features(df)
            progress.update(features_task, completed=100)
            
            # Step 4: Train ML model
            training_task = progress.add_task("[magenta]Training ML model...", total=100)
            engine.train_ml_model(df)
            progress.update(training_task, completed=100)
            
            # Step 5: Run backtest (this has its own progress bar)
            console.print("\n[bold]Starting backtest execution...[/bold]")
            
        results = engine.run_backtest(df)
        
        # Calculate performance metrics
        metrics = engine.calculate_performance_metrics(results)
        
        # Display results
        console.print("\n[bold green][SUCCESS] Backtest Complete![/bold green]")
        engine.display_results(metrics)
        
        # Save results
        engine.save_results(results, metrics)
        
        # Create plots
        engine.plot_results(df)
        
        # Summary
        console.print(f"\n[bold][SUMMARY] Summary:[/bold]")
        console.print(f"Initial Capital: [green]${engine.initial_capital:,.2f}[/green]")
        console.print(f"Final Capital: [green]${results['final_capital']:,.2f}[/green]")
        console.print(f"Total Return: [green]{metrics['total_return']:.2%}[/green]")
        console.print(f"Total Trades: [cyan]{metrics['total_trades']}[/cyan]")
        console.print(f"Win Rate: [green]{metrics['win_rate']:.1%}[/green]")
        console.print(f"Sharpe Ratio: [green]{metrics['sharpe_ratio']:.2f}[/green]")
        
        if metrics['total_return'] > 0:
            console.print("\n[bold green][SUCCESS] Strategy shows positive returns![/bold green]")
        else:
            console.print("\n[bold red][WARNING] Strategy shows negative returns. Consider optimization.[/bold red]")
            
    except Exception as e:
        console.print(f"\n[bold red][ERROR] Backtest failed: {e}[/bold red]")
        logging.error(f"Backtest error: {e}", exc_info=True)
        
if __name__ == "__main__":
    asyncio.run(main())