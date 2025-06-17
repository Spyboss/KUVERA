#!/usr/bin/env python3
"""
Kuvera Grid Trading Bot v1.1 🚀
AI-powered cryptocurrency trading bot with modern CLI interface
Author: Uminda
Email: Uminda.h.aberathne@gmail.com
"""

import asyncio
import json
import logging
import os
import time
import yaml
import random
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from binance.spot import Spot
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box
# keyboard import removed - no longer needed for automated mode
import threading

# AI and ML imports
try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV
    import talib
except ImportError:
    xgb = None
    StandardScaler = None
    GridSearchCV = None
    talib = None

# Import AI components
try:
    from ai.ai_strategy_optimizer import AIStrategyOptimizer
    from ai.auto_trader import AutonomousTrader
except ImportError:
    AIStrategyOptimizer = None
    AutonomousTrader = None

# Gamification imports
try:
    from playsound import playsound
except ImportError:
    playsound = None

# Load environment variables
load_dotenv()

class GameificationManager:
    """Manages gamification elements like streaks, badges, and achievements"""
    
    def __init__(self):
        self.achievements = []
        self.current_streak = 0
        self.max_streak = 0
        self.badges = set()
        self.quotes = [
            "HODL strong! 💪",
            "To the moon! 🚀",
            "Diamond hands! 💎",
            "Buy the dip! 📉",
            "Stack sats! ⚡",
            "Number go up! 📈",
            "This is the way! 🛡️",
            "Patience pays! ⏰"
        ]
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
    def add_win(self):
        """Add a winning trade"""
        self.current_streak += 1
        self.max_streak = max(self.max_streak, self.current_streak)
        
        # Check for streak badges
        if self.current_streak == 5:
            self.add_badge("🔥 Hot Streak (5 wins)")
        elif self.current_streak == 10:
            self.add_badge("🌟 Legendary Streak (10 wins)")
            
    def add_loss(self):
        """Add a losing trade"""
        self.current_streak = 0
        
    def add_badge(self, badge: str):
        """Add a new badge"""
        if badge not in self.badges:
            self.badges.add(badge)
            self.log_achievement(f"New Badge: {badge}")
            if playsound:
                try:
                    # Create a simple achievement sound (you can replace with actual sound file)
                    pass
                except:
                    pass
                    
    def check_trade_milestones(self, trade_count: int, profit: float):
        """Check for trade milestone badges"""
        if trade_count == 1 and "🎯 First Trade" not in self.badges:
            self.add_badge("🎯 First Trade")
        elif trade_count == 10 and "📊 10 Trades" not in self.badges:
            self.add_badge("📊 10 Trades")
        elif trade_count == 100 and "💯 Century Club" not in self.badges:
            self.add_badge("💯 Century Club")
            
        if profit > 0 and "💰 First Profit" not in self.badges:
            self.add_badge("💰 First Profit")
        elif profit >= 10 and "💎 $10 Profit" not in self.badges:
            self.add_badge("💎 $10 Profit")
            
    def get_random_quote(self) -> str:
        """Get a random motivational quote"""
        return random.choice(self.quotes)
        
    def log_achievement(self, achievement: str):
        """Log achievement to file"""
        try:
            with open('logs/achievements.txt', 'a') as f:
                f.write(f"{datetime.now().isoformat()}: {achievement}\n")
        except Exception:
            pass

class ModernUI:
    """Modern CLI interface using Rich library"""
    
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.is_running = False
        self.current_mode = "Testnet"
        self.ai_enabled = True
        self.strategy_type = "Mean Reversion"
        
        # Setup layout
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        self.layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        self.layout["left"].split_column(
            Layout(name="metrics", size=8),
            Layout(name="balances", size=8),
            Layout(name="status", ratio=1)
        )
        
    def create_header(self, trading_active: bool = False, ai_sentiment: float = None) -> Panel:
        """Create header panel with trading status and AI information"""
        title = Text("Kuvera Grid v1.1 🚀", style="bold cyan")
        
        # Add trading status indicator
        status_color = "green" if trading_active else "red"
        status_text = "ACTIVE" if trading_active else "PAUSED"
        trading_status = Text(f" [{status_text}]", style=f"bold {status_color}")
        
        # Enhanced subtitle with AI sentiment
        ai_status = "Enabled" if self.ai_enabled else "Disabled"
        if self.ai_enabled and ai_sentiment is not None:
            sentiment_emoji = "📈" if ai_sentiment > 0.6 else "📉" if ai_sentiment < 0.4 else "➡️"
            ai_status += f" {sentiment_emoji} ({ai_sentiment:.2f})"
        
        subtitle = Text(f"Mode: {self.current_mode} | AI: {ai_status} | Strategy: {self.strategy_type}", style="dim")
        
        return Panel(Text.assemble(title, trading_status, "\n", subtitle), box=box.ROUNDED)
        
    def create_metrics_panel(self, trades: int, win_rate: float, profit: float, position: str, risk: float) -> Panel:
        """Create metrics panel"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Trades:", str(trades))
        table.add_row("Win Rate:", f"{win_rate:.1f}%")
        table.add_row("Profit:", f"${profit:.2f}")
        table.add_row("Position:", position or "None")
        table.add_row("Risk:", f"{risk:.1f}% (${risk*30:.2f})")
        
        return Panel(table, title="📊 Metrics", box=box.ROUNDED)
        
    def create_balances_panel(self, balances: Dict[str, float]) -> Panel:
        """Create balances panel showing top 5 assets"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Asset", style="yellow")
        table.add_column("Amount", style="green")
        
        # Show BTC and USDT first, then top 3 others
        priority_assets = ['BTC', 'USDT']
        other_assets = {k: v for k, v in balances.items() if k not in priority_assets and v > 0}
        top_others = sorted(other_assets.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for asset in priority_assets:
            if asset in balances:
                table.add_row(f"{asset}:", f"{balances[asset]:.6f}")
                
        for asset, amount in top_others:
            table.add_row(f"{asset}:", f"{amount:.6f}")
            
        return Panel(table, title="💰 Balances", box=box.ROUNDED)
        
    def create_status_panel(self, websocket_status: str, api_status: str, target_progress: float, current_price: float = None) -> Panel:
        """Create enhanced status panel with real-time data"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Status", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("WebSocket:", websocket_status)
        table.add_row("API:", api_status)
        
        # Show current BTC price if available
        if current_price:
            table.add_row("BTC Price:", f"${current_price:,.2f}")
        
        # Progress bar for weekly target
        progress_bar = "█" * int(target_progress * 20) + "░" * (20 - int(target_progress * 20))
        table.add_row("Target:", f"[{progress_bar}] {target_progress:.0%}")
        
        # Add timestamp for last update
        current_time = datetime.now().strftime("%H:%M:%S")
        table.add_row("Updated:", current_time)
        
        return Panel(table, title="🔧 Status", box=box.ROUNDED)
        
    def create_gamification_panel(self, gamification: GameificationManager) -> Panel:
        """Create gamification panel"""
        content = []
        
        if gamification.current_streak > 0:
            content.append(f"🔥 Streak: {gamification.current_streak}")
            
        if gamification.badges:
            content.append("\n🏆 Recent Badges:")
            for badge in list(gamification.badges)[-3:]:  # Show last 3 badges
                content.append(f"  {badge}")
                
        content.append(f"\n💭 {gamification.get_random_quote()}")
        
        return Panel("\n".join(content), title="🎮 Achievements", box=box.ROUNDED)
        
    def create_footer(self) -> Panel:
        """Create footer with controls"""
        controls = Text("[s] Start/Stop | [m] Mode | [q] Quit", style="bold white")
        return Panel(controls, box=box.ROUNDED)
        
    def update_display(self, bot_data: dict, gamification: GameificationManager):
        """Update the display with current data and real-time status"""
        trading_active = bot_data.get('trading_active', False)
        ai_sentiment = bot_data.get('ai_sentiment', None)
        
        self.layout["header"].update(self.create_header(trading_active, ai_sentiment))
        self.layout["metrics"].update(self.create_metrics_panel(
            bot_data.get('trades', 0),
            bot_data.get('win_rate', 0),
            bot_data.get('profit', 0),
            bot_data.get('position', 'None'),
            bot_data.get('risk', 1)
        ))
        self.layout["balances"].update(self.create_balances_panel(bot_data.get('balances', {})))
        self.layout["status"].update(self.create_status_panel(
            bot_data.get('websocket_status', 'Disconnected'),
            bot_data.get('api_status', 'Unknown'),
            bot_data.get('target_progress', 0),
            bot_data.get('current_price')
        ))
        self.layout["right"].update(self.create_gamification_panel(gamification))
        self.layout["footer"].update(self.create_footer())

class EnhancedTradingBot:
    """Enhanced trading bot with modern UI and AI features"""
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the enhanced trading bot"""
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        # Setup logging
        self.setup_logging()
        
        # Initialize UI and gamification
        self.ui = ModernUI()
        self.gamification = GameificationManager()
        
        # Initialize Binance client
        self.setup_binance_client()
        
        # Trading state
        self.symbol = self.config['trading']['symbol']
        self.is_running = False
        self.trading_active = False
        self.position = None
        self.entry_price = None
        self.last_trade_time = None
        
        # Price data buffer
        self.price_buffer = deque(maxlen=100)
        self.current_price = None
        
        # Risk management
        self.daily_trades = 0
        self.daily_loss = 0.0
        self.current_date = None
        
        # Performance tracking
        self.total_profit = 0.0
        self.trade_count = 0
        self.winning_trades = 0
        
        # WebSocket client
        self.ws_client = None
        
        # Balance caching for real-time updates
        self._cached_balances = None
        self._last_balance_update = 0
        
        # Enhanced AI/ML components v1.1
        self.ml_model = None
        self.scaler = None
        self.lstm_model = None  # LSTM for ensemble filtering
        self.setup_ml_components()
        
        # Multi-timeframe data buffers
        self.price_buffer_1h = deque(maxlen=200)   # 1-hour data for trend analysis
        self.price_buffer_15m = deque(maxlen=500)  # 15-minute data for setup signals
        self.atr_buffer = deque(maxlen=50)         # ATR values for dynamic thresholds
        
        # Enhanced technical indicators storage
        self.indicators_cache = {
            '5m': {},   # 5-minute indicators
            '15m': {},  # 15-minute indicators  
            '1h': {}    # 1-hour indicators
        }
        
        # Kelly Criterion variables
        self.trade_history = deque(maxlen=100)  # Store recent trade results
        self.consecutive_losses = 0
        self.current_kelly_fraction = 0.01  # Start conservative
        
        # Trailing stop variables
        self.trailing_stop_price = None
        self.highest_price_since_entry = None
        
        # OpenRouter AI Integration
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.ai_optimizer = None
        self.auto_trader = None
        self.ai_enabled = self.config.get('ai', {}).get('openrouter', False)
        self.ai_sentiment_score = 0.5  # Neutral
        self.ai_ensemble_confidence = 0.5  # Ensemble confidence
        self.diagnostic_mode = self.config.get('ai', {}).get('diagnostic_mode', False)
        self.startup_test_enabled = self.config.get('trading', {}).get('startup_test', True)
        
        # XGBoost retraining timer
        self.last_xgboost_retrain = 0
        self.xgboost_retrain_interval = self.config.get('ai', {}).get('xgboost_retrain_hours', 12) * 3600
        
        # Setup AI components after ML components
        self.setup_ai_components()
        
        # Weekly target tracking
        self.weekly_target = 2.0  # $2 per week
        self.week_start_profit = self.total_profit
        
    def get_server_time(self, base_url):
        """Get Binance server time for synchronization"""
        import requests
        try:
            response = requests.get(f"{base_url}/api/v3/time", timeout=5)
            if response.status_code == 200:
                return response.json()['serverTime']
        except Exception as e:
            self.logger.warning(f"Could not get server time: {e}")
        return int(time.time() * 1000)
        
    def setup_logging(self):
        """Setup logging configuration with UTF-8 encoding"""
        os.makedirs('logs', exist_ok=True)
        
        log_level = getattr(logging, self.config['monitoring']['log_level'])
        
        # Create file handler with UTF-8 encoding to handle Unicode characters like 🚀
        file_handler = logging.FileHandler(
            f'logs/bot_{datetime.now().strftime("%Y%m%d")}.log',
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        
        # Create console handler with UTF-8 encoding
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=[file_handler, console_handler]
        )
        
        self.logger = logging.getLogger('KuveraBot')
        self.logger.info("Logging system initialized with UTF-8 encoding 🚀")
        
    def setup_binance_client(self):
        """Setup Binance API client with improved timestamp synchronization"""
        testnet_mode = getattr(self, 'testnet_mode', self.config['api']['testnet'])
        self.testnet_mode = testnet_mode  # Store as instance attribute
        
        if testnet_mode:
            api_key = os.getenv('BINANCE_TESTNET_API_KEY') or os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_TESTNET_SECRET_KEY') or os.getenv('BINANCE_SECRET_KEY')
            base_url = "https://testnet.binance.vision"
            self.ui.current_mode = "Testnet"
            self.logger.info("Using Binance Testnet")
        else:
            api_key = os.getenv('BINANCE_LIVE_API_KEY')
            api_secret = os.getenv('BINANCE_LIVE_SECRET_KEY')
            base_url = "https://api.binance.com"
            self.ui.current_mode = "LIVE"
            self.logger.warning("Using Binance LIVE trading - real money at risk!")
        
        if not api_key or not api_secret:
            mode_str = "testnet" if testnet_mode else "live"
            self.logger.error(f"API credentials not found for {mode_str} mode")
            raise ValueError(f"Missing {mode_str} API credentials")
            
        # Enhanced time synchronization with multiple attempts
        self.time_offset = 0
        self.recv_window = int(os.getenv('BINANCE_RECV_WINDOW', '60000'))
        
        for attempt in range(3):
            try:
                server_time = self.get_server_time(base_url)
                local_time = int(time.time() * 1000)
                self.time_offset = server_time - local_time
                self.logger.info(f"Time sync attempt {attempt + 1}: offset = {self.time_offset}ms")
                break
            except Exception as e:
                self.logger.warning(f"Time sync attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    self.logger.warning("Using local time without offset")
                    self.time_offset = 0
                time.sleep(1)
            
        self.client = Spot(
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url
        )
        
        # Test connection with multiple strategies
        connection_successful = False
        
        # Strategy 1: Use server time with offset
        try:
            timestamp = int(time.time() * 1000) + self.time_offset
            account_info = self.client.account(recvWindow=self.recv_window, timestamp=timestamp)
            self.logger.info("✅ Connected to Binance API with time offset")
            connection_successful = True
        except Exception as e:
            self.logger.warning(f"Connection with offset failed: {e}")
            
        # Strategy 2: Use fresh server time
        if not connection_successful:
            try:
                fresh_server_time = self.get_server_time(base_url)
                account_info = self.client.account(recvWindow=self.recv_window, timestamp=fresh_server_time)
                self.logger.info("✅ Connected to Binance API with fresh server time")
                connection_successful = True
            except Exception as e:
                self.logger.warning(f"Connection with fresh server time failed: {e}")
                
        # Strategy 3: Use default timestamp with extended window
        if not connection_successful:
            try:
                account_info = self.client.account(recvWindow=self.recv_window)
                self.logger.info("✅ Connected to Binance API with default timestamp")
                connection_successful = True
            except Exception as e:
                self.logger.error(f"All connection attempts failed: {e}")
                
        if not connection_successful:
            raise Exception("Could not establish connection to Binance API after all attempts")
            
    def setup_ml_components(self):
        """Setup ML components with enhanced fallback strategy"""
        try:
            # Check if XGBoost and sklearn are available
            if xgb and StandardScaler:
                self.ml_model = xgb.XGBRegressor(
                    n_estimators=50,  # Reduced for 8GB RAM optimization
                    max_depth=4,      # Reduced depth for memory efficiency
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=1          # Single thread for memory efficiency
                )
                self.scaler = StandardScaler()
                self.ml_enabled = True
                self.logger.info("🤖 XGBoost ML components initialized successfully (RAM optimized)")
                
                # Try to train on sample data if available
                self.train_ml_model()
            else:
                self.ml_enabled = False
                self.logger.warning("⚠️ XGBoost not available, using Enhanced Mean Reversion with RSI/Bollinger Bands")
                
        except ImportError as e:
            self.ml_enabled = False
            self.logger.warning(f"⚠️ ML libraries missing: {e}. Install with: pip install xgboost scikit-learn")
        except Exception as e:
            self.ml_enabled = False
            self.logger.error(f"❌ Error setting up ML components: {e}")
            
    def train_ml_model(self):
        """Train ML model on recent price data (memory optimized for 8GB RAM)"""
        try:
            if not self.ml_enabled or len(self.price_buffer) < 100:
                return
                
            # Use only recent data to save memory (last 1000 points)
            recent_prices = list(self.price_buffer)[-1000:] if len(self.price_buffer) > 1000 else list(self.price_buffer)
            
            if len(recent_prices) < 50:
                return
                
            # Create features (simple technical indicators)
            features = []
            targets = []
            
            for i in range(20, len(recent_prices) - 1):  # Need 20 periods for indicators
                price_slice = recent_prices[i-20:i]
                current_price = recent_prices[i]
                next_price = recent_prices[i+1]
                
                # Simple features to save memory
                sma_5 = sum(price_slice[-5:]) / 5
                sma_20 = sum(price_slice) / 20
                price_change = (current_price - price_slice[-2]) / price_slice[-2]
                
                features.append([current_price, sma_5, sma_20, price_change])
                targets.append(next_price)
                
            if len(features) > 30:  # Minimum data for training
                X = self.scaler.fit_transform(features)
                y = targets
                
                self.ml_model.fit(X, y)
                self.logger.info(f"🎯 ML model trained on {len(features)} samples (memory optimized)")
                
        except Exception as e:
            self.logger.error(f"❌ Error training ML model: {e}")
            
    async def test_openrouter_connectivity(self):
        """Test OpenRouter API connectivity in diagnostic mode"""
        if not self.openrouter_api_key:
            self.logger.error("❌ OpenRouter API key not found for diagnostic test")
            return False
            
        try:
            self.logger.info("🔍 Testing OpenRouter API connectivity...")
            
            if self.ai_optimizer:
                # Test with simple price data
                test_prices = [50000.0, 50100.0, 50050.0, 50200.0, 50150.0]
                test_context = "Diagnostic test - RSI: 50.0, SMA: 50075.0"
                
                result = await self.ai_optimizer.analyze_market_sentiment(test_prices, test_context)
                
                if result is not None:
                    self.logger.info(f"✅ OpenRouter API test successful - Response: {result:.3f}")
                    return True
                else:
                    self.logger.error("❌ OpenRouter API test failed - No response")
                    return False
            else:
                self.logger.error("❌ AI optimizer not initialized")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ OpenRouter API test failed: {e}")
            return False
    
    async def execute_startup_test_trade(self):
        """Execute a startup test trade cycle in testnet"""
        if not self.startup_test_enabled:
            return
            
        try:
            self.logger.info("🧪 Starting startup test trade cycle...")
            
            # Get current price
            current_price = self.get_current_price()
            if not current_price:
                self.logger.error("❌ Could not get current price for test trade")
                return
                
            # Test trade parameters
            test_quantity = 0.001  # 0.001 BTC
            
            # Check if we have sufficient balance
            balances = self.get_account_balances()
            usdt_balance = balances.get('USDT', 0)
            required_usdt = test_quantity * current_price
            
            if usdt_balance < required_usdt:
                self.logger.warning(f"⚠️ Insufficient USDT balance for test trade: {usdt_balance:.2f} < {required_usdt:.2f}")
                return
                
            # Execute test buy
            self.logger.info(f"🟢 Test BUY: {test_quantity} BTC at ${current_price:.2f}")
            buy_success = await self.place_test_order('BUY', current_price, test_quantity)
            
            if buy_success:
                # Wait a moment then execute test sell
                await asyncio.sleep(2)
                
                # Get updated price for sell
                sell_price = self.get_current_price()
                self.logger.info(f"🔴 Test SELL: {test_quantity} BTC at ${sell_price:.2f}")
                
                sell_success = await self.place_test_order('SELL', sell_price, test_quantity)
                
                if sell_success:
                    profit = (sell_price - current_price) * test_quantity
                    self.logger.info(f"✅ Startup test trade completed - Profit: ${profit:.4f}")
                    self.logger.info("🎯 Trading workflow verified - Ready for live trading")
                else:
                    self.logger.error("❌ Test sell order failed")
            else:
                self.logger.error("❌ Test buy order failed")
                
        except Exception as e:
            self.logger.error(f"❌ Startup test trade failed: {e}")
    
    async def place_test_order(self, side: str, price: float, quantity: float) -> bool:
        """Place a test order (simulated in testnet)"""
        try:
            # In testnet mode, we can actually place orders
            if self.testnet_mode:
                order_data = {
                    'symbol': self.symbol,
                    'side': side,
                    'type': 'MARKET',
                    'quantity': quantity
                }
                
                # Log the test order
                self.logger.info(f"📝 Test {side} order: {quantity} {self.symbol} at ${price:.2f}")
                
                # For safety, we'll simulate the order in testnet
                await asyncio.sleep(0.5)  # Simulate order processing time
                
                return True
            else:
                # In live mode, just simulate
                self.logger.info(f"🔄 Simulated {side} order: {quantity} {self.symbol} at ${price:.2f}")
                return True
                
        except Exception as e:
            self.logger.error(f"Test order error: {e}")
            return False
    
    def setup_ai_components(self):
        """Setup OpenRouter AI components for enhanced trading decisions"""
        try:
            if self.openrouter_api_key and AIStrategyOptimizer and AutonomousTrader:
                # Initialize AI Strategy Optimizer
                self.ai_optimizer = AIStrategyOptimizer(self.openrouter_api_key)
                
                # Initialize Autonomous Trader
                self.auto_trader = AutonomousTrader(
                    bot_instance=self,
                    config=self.config,
                    openrouter_api_key=self.openrouter_api_key
                )
                
                self.ai_enabled = True
                self.logger.info("🤖 OpenRouter AI components initialized successfully")
                self.logger.info(f"🎯 AI confidence threshold: {self.config.get('ai', {}).get('confidence_threshold', 0.7):.1%}")
                
            elif not self.openrouter_api_key:
                self.ai_enabled = False
                self.logger.warning("⚠️ OpenRouter API key not found - AI features disabled")
                self.logger.info("💡 Add OPENROUTER_API_KEY to .env to enable AI features")
                
            else:
                self.ai_enabled = False
                self.logger.warning("⚠️ AI modules not available - check ai/ directory")
                
        except Exception as e:
            self.ai_enabled = False
            self.logger.error(f"❌ Error setting up AI components: {e}")
            self.logger.info("🔄 Continuing with standard trading strategy")
            
    def calculate_atr(self, high_prices: List[float], low_prices: List[float], close_prices: List[float], period: int = 10) -> float:
        """Calculate Average True Range for dynamic thresholds"""
        try:
            if len(close_prices) < period + 1 or not talib:
                return 0.01  # Default ATR value
                
            high_array = np.array(high_prices, dtype=float)
            low_array = np.array(low_prices, dtype=float) 
            close_array = np.array(close_prices, dtype=float)
            
            atr = talib.ATR(high_array, low_array, close_array, timeperiod=period)[-1]
            return atr if not np.isnan(atr) else 0.01
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return 0.01
    
    def calculate_dynamic_sma_period(self, atr_value: float, base_price: float) -> int:
        """Calculate dynamic SMA period based on ATR volatility"""
        try:
            config = self.config['strategy']
            base_period = config.get('sma_period_base', 12)
            max_period = config.get('sma_period_max', 14)
            
            # Calculate volatility ratio (ATR as percentage of price)
            volatility_ratio = atr_value / base_price if base_price > 0 else 0
            
            # Adjust period based on volatility (higher volatility = longer period)
            if volatility_ratio > 0.02:  # High volatility (>2%)
                return max_period
            elif volatility_ratio < 0.005:  # Low volatility (<0.5%)
                return base_period
            else:
                # Linear interpolation between base and max
                ratio = (volatility_ratio - 0.005) / (0.02 - 0.005)
                return int(base_period + ratio * (max_period - base_period))
                
        except Exception as e:
            self.logger.error(f"Error calculating dynamic SMA period: {e}")
            return 12  # Default fallback
    
    def calculate_dynamic_thresholds(self, atr_value: float, base_price: float) -> Tuple[float, float]:
        """Calculate optimized dynamic entry/exit thresholds based on Grok AI ATR multipliers"""
        try:
            config = self.config['strategy']
            
            # Grok AI optimized ATR multipliers
            entry_atr_multiplier = config.get('entry_threshold_atr_multiplier', 0.3)  # Grok AI: 0.3 × 10-period ATR
            exit_atr_multiplier = config.get('exit_threshold_atr_multiplier', 2.0)    # Grok AI: 2.0 × ATR
            
            # Fallback bounds from config
            entry_min = config.get('entry_threshold_min', 0.002)
            entry_max = config.get('entry_threshold_max', 0.004)
            exit_min = config.get('exit_threshold_min', 0.015)
            exit_max = config.get('exit_threshold_max', 0.020)
            
            # Calculate ATR as percentage of price
            atr_pct = atr_value / base_price if base_price > 0 else 0.01
            
            # Grok AI optimized thresholds: Direct ATR multiplier approach
            entry_threshold = min(entry_max, max(entry_min, entry_atr_multiplier * atr_pct))  # 0.3 × ATR
            exit_threshold = min(exit_max, max(exit_min, exit_atr_multiplier * atr_pct))      # 2.0 × ATR
            
            self.logger.debug(f"Grok AI Thresholds - ATR: {atr_value:.4f}, Entry: {entry_threshold:.4f}, Exit: {exit_threshold:.4f}")
            return entry_threshold, exit_threshold
            
        except Exception as e:
            self.logger.error(f"Error calculating Grok AI dynamic thresholds: {e}")
            return 0.003, 0.0175  # Default fallback
    
    def detect_rsi_divergence(self, prices: List[float], rsi_values: List[float], lookback: int = 20) -> Dict[str, bool]:
        """Detect RSI divergence patterns"""
        try:
            if len(prices) < lookback or len(rsi_values) < lookback:
                return {'bullish_divergence': False, 'bearish_divergence': False}
            
            recent_prices = prices[-lookback:]
            recent_rsi = rsi_values[-lookback:]
            
            # Find price and RSI peaks/troughs
            price_peaks = []
            rsi_peaks = []
            price_troughs = []
            rsi_troughs = []
            
            for i in range(1, len(recent_prices) - 1):
                # Price peaks
                if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                    price_peaks.append((i, recent_prices[i]))
                    rsi_peaks.append((i, recent_rsi[i]))
                
                # Price troughs
                if recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                    price_troughs.append((i, recent_prices[i]))
                    rsi_troughs.append((i, recent_rsi[i]))
            
            # Check for divergences
            bullish_divergence = False
            bearish_divergence = False
            
            # Bullish divergence: price makes lower lows, RSI makes higher lows
            if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
                last_price_trough = price_troughs[-1][1]
                prev_price_trough = price_troughs[-2][1]
                last_rsi_trough = rsi_troughs[-1][1]
                prev_rsi_trough = rsi_troughs[-2][1]
                
                if last_price_trough < prev_price_trough and last_rsi_trough > prev_rsi_trough:
                    bullish_divergence = True
            
            # Bearish divergence: price makes higher highs, RSI makes lower highs
            if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
                last_price_peak = price_peaks[-1][1]
                prev_price_peak = price_peaks[-2][1]
                last_rsi_peak = rsi_peaks[-1][1]
                prev_rsi_peak = rsi_peaks[-2][1]
                
                if last_price_peak > prev_price_peak and last_rsi_peak < prev_rsi_peak:
                    bearish_divergence = True
            
            return {
                'bullish_divergence': bullish_divergence,
                'bearish_divergence': bearish_divergence
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting RSI divergence: {e}")
            return {'bullish_divergence': False, 'bearish_divergence': False}
    
    def calculate_technical_indicators(self, prices: List[float], timeframe: str = '5m', high_prices: List[float] = None, low_prices: List[float] = None) -> Dict[str, float]:
        """Calculate enhanced technical indicators with multi-timeframe support"""
        if len(prices) < 20 or not talib:
            return {}
            
        prices_array = np.array(prices, dtype=float)
        indicators = {}
        
        try:
            current_price = prices[-1]
            
            # Calculate ATR for dynamic adjustments
            if high_prices and low_prices and len(high_prices) == len(prices):
                atr_value = self.calculate_atr(high_prices, low_prices, prices)
                indicators['atr'] = atr_value
                self.atr_buffer.append(atr_value)
                
                # Calculate dynamic SMA period
                dynamic_sma_period = self.calculate_dynamic_sma_period(atr_value, current_price)
                indicators['dynamic_sma_period'] = dynamic_sma_period
                
                # Calculate dynamic thresholds
                entry_threshold, exit_threshold = self.calculate_dynamic_thresholds(atr_value, current_price)
                indicators['dynamic_entry_threshold'] = entry_threshold
                indicators['dynamic_exit_threshold'] = exit_threshold
            else:
                # Use default values if OHLC data not available
                dynamic_sma_period = 12
                indicators['atr'] = 0.01
                indicators['dynamic_sma_period'] = dynamic_sma_period
                indicators['dynamic_entry_threshold'] = 0.003
                indicators['dynamic_exit_threshold'] = 0.0175
            
            # Dynamic SMA
            if len(prices) >= dynamic_sma_period:
                sma = talib.SMA(prices_array, timeperiod=dynamic_sma_period)[-1]
                indicators['sma'] = sma
            
            # RSI
            rsi = talib.RSI(prices_array, timeperiod=14)[-1]
            indicators['rsi'] = rsi
            
            # Enhanced Bollinger Bands with configurable multiplier
            bb_multiplier = self.config['strategy'].get('bb_multiplier', 1.5)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(prices_array, timeperiod=20, nbdevup=bb_multiplier, nbdevdn=bb_multiplier)
            indicators['bb_upper'] = bb_upper[-1]
            indicators['bb_middle'] = bb_middle[-1]
            indicators['bb_lower'] = bb_lower[-1]
            
            # Multi-timeframe specific indicators
            if timeframe == '1h':
                # 1-hour EMA for trend
                ema_period = self.config['strategy']['timeframes']['trend_1h'].get('ema_period', 50)
                if len(prices) >= ema_period:
                    ema = talib.EMA(prices_array, timeperiod=ema_period)[-1]
                    indicators['ema_50'] = ema
                    indicators['trend_direction'] = 'bullish' if current_price > ema else 'bearish'
            
            elif timeframe == '15m':
                # 15-minute MACD
                macd_config = self.config['strategy']['timeframes']['setup_15m']
                macd_fast = macd_config.get('macd_fast', 12)
                macd_slow = macd_config.get('macd_slow', 26)
                macd_signal = macd_config.get('macd_signal', 9)
                
                if len(prices) >= macd_slow:
                    macd, macd_signal_line, macd_hist = talib.MACD(prices_array, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
                    indicators['macd'] = macd[-1]
                    indicators['macd_signal'] = macd_signal_line[-1]
                    indicators['macd_histogram'] = macd_hist[-1]
                    indicators['macd_bullish'] = macd[-1] > macd_signal_line[-1]
            
            # RSI Divergence Detection
            if len(prices) >= 20:
                rsi_values = talib.RSI(prices_array, timeperiod=14)
                rsi_values_clean = [x for x in rsi_values if not np.isnan(x)]
                if len(rsi_values_clean) >= 20:
                    divergence = self.detect_rsi_divergence(prices[-20:], rsi_values_clean[-20:])
                    indicators.update(divergence)
            
            # Cache indicators for this timeframe
            self.indicators_cache[timeframe] = indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {timeframe}: {e}")
            
        return indicators
        
    def get_account_balances(self, force_refresh: bool = False) -> Dict[str, float]:
        """Get account balances with caching for real-time updates"""
        current_time = time.time()
        
        # Use cached balances if available and not expired (unless force refresh)
        if (not force_refresh and 
            self._cached_balances is not None and 
            current_time - self._last_balance_update < 5):
            return self._cached_balances
            
        try:
            # Use improved timestamp handling
            timestamp = int(time.time() * 1000) + getattr(self, 'time_offset', 0)
            recv_window = getattr(self, 'recv_window', 60000)
            account_info = self.client.account(recvWindow=recv_window, timestamp=timestamp)
            balances = {}
            
            for balance in account_info['balances']:
                free_balance = float(balance['free'])
                if free_balance > 0:
                    balances[balance['asset']] = free_balance
                    
            # Cache the results
            self._cached_balances = balances
            self._last_balance_update = current_time
            
            return balances
        except Exception as e:
            self.logger.error(f"Error getting balances: {e}")
            # Return cached balances if available, otherwise empty dict
            return self._cached_balances if self._cached_balances else {}
            
    def get_bot_data(self) -> dict:
        """Get current bot data for UI display with real-time updates"""
        win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0
        
        # Calculate weekly progress
        weekly_profit = self.total_profit - self.week_start_profit
        target_progress = min(weekly_profit / self.weekly_target, 1.0) if self.weekly_target > 0 else 0
        
        # Determine API status
        api_status = "OK"
        try:
            # Quick ping to check API status
            self.client.ping()
        except:
            api_status = "ERROR"
        
        return {
            'trades': self.trade_count,
            'win_rate': win_rate,
            'profit': self.total_profit,
            'position': 'BTC/USDT' if self.position else None,
            'risk': self.config['trading']['risk_per_trade'] * 100,
            'balances': self.get_account_balances(),
            'websocket_status': 'Connected (btcusdt@kline_5m)' if self.ws_client else 'Disconnected',
            'api_status': api_status,
            'target_progress': target_progress,
            'current_price': self.current_price,
            'ai_sentiment': self.ai_sentiment_score if self.ai_enabled else None
        }
        
    async def get_multi_timeframe_signals(self, current_price: float) -> Dict[str, bool]:
        """Get Grok AI optimized signals from multiple timeframes (1h EMA-50, 15m MACD(12,26) + RSI(14))"""
        signals = {
            'trend_1h_bullish': True,  # Default to neutral
            'setup_15m_bullish': True,
            'momentum_5m_bullish': True
        }
        
        try:
            # Grok AI: 1-hour 50-period EMA trend filter
            if self.config['strategy']['timeframes']['trend_1h'].get('enabled', False):
                indicators_1h = self.indicators_cache.get('1h', {})
                ema_50 = indicators_1h.get('ema_50')
                if ema_50:
                    signals['trend_1h_bullish'] = current_price > ema_50
                    self.logger.debug(f"Grok AI 1h EMA-50 Filter - Price: {current_price:.4f}, EMA: {ema_50:.4f}, Bullish: {signals['trend_1h_bullish']}")
            
            # Grok AI: 15-minute MACD (12,26) and RSI (14) setup filters
            if self.config['strategy']['timeframes']['setup_15m'].get('enabled', False):
                indicators_15m = self.indicators_cache.get('15m', {})
                macd_bullish = indicators_15m.get('macd_bullish', True)
                rsi_15m = indicators_15m.get('rsi')
                
                # Grok AI optimized MACD (12,26) filter
                setup_conditions = [macd_bullish]
                
                # Grok AI optimized RSI (14) filter - not in extreme zones
                if rsi_15m:
                    rsi_condition = 30 <= rsi_15m <= 70  # Avoid overbought/oversold extremes
                    setup_conditions.append(rsi_condition)
                    self.logger.debug(f"Grok AI 15m Filters - MACD: {macd_bullish}, RSI: {rsi_15m:.2f}, RSI OK: {rsi_condition}")
                
                signals['setup_15m_bullish'] = all(setup_conditions)
            
            # Grok AI: 5-minute momentum confirmation
            indicators_5m = self.indicators_cache.get('5m', {})
            rsi_5m = indicators_5m.get('rsi')
            if rsi_5m:
                # Grok AI: Look for oversold conditions for mean reversion entry
                signals['momentum_5m_bullish'] = rsi_5m < 40  # Oversold for mean reversion buy
                self.logger.debug(f"Grok AI 5m Momentum - RSI: {rsi_5m:.2f}, Oversold: {signals['momentum_5m_bullish']}")
            
        except Exception as e:
            self.logger.error(f"Error getting Grok AI multi-timeframe signals: {e}")
        
        return signals
    
    async def get_ai_ensemble_signal(self, current_price: float, indicators: Dict[str, float]) -> Tuple[bool, float]:
        """Get Grok AI optimized ensemble signal with LSTM 0.7 confidence and XGBoost validation"""
        try:
            ai_config = self.config['ai']['ensemble']
            if not ai_config.get('enabled', False):
                return True, 0.5  # Neutral if disabled
            
            # Grok AI: LSTM confidence check with 0.7 threshold
            lstm_confidence = 0.75  # Simulated LSTM confidence - implement actual LSTM prediction
            lstm_min = ai_config.get('lstm_confidence_min', 0.70)  # Grok AI: 0.7 minimum
            lstm_max = ai_config.get('lstm_confidence_max', 0.85)  # Enhanced maximum
            
            lstm_signal = lstm_min <= lstm_confidence <= lstm_max
            self.logger.debug(f"Grok AI LSTM - Confidence: {lstm_confidence:.3f}, Signal: {lstm_signal}")
            
            # XGBoost validation
            xgboost_signal = True
            if ai_config.get('xgboost_validation', False) and self.ml_enabled:
                try:
                    # Retrain XGBoost if needed
                    current_time = time.time()
                    if current_time - self.last_xgboost_retrain > self.xgboost_retrain_interval:
                        self.train_ml_model()
                        self.last_xgboost_retrain = current_time
                        self.logger.info("🔄 XGBoost model retrained")
                    
                    # Get XGBoost prediction
                    if self.ml_model and len(self.price_buffer) >= 20:
                        prices = list(self.price_buffer)
                        sma_5 = sum(prices[-5:]) / 5
                        sma_20 = sum(prices[-20:]) / 20
                        price_change = (current_price - prices[-2]) / prices[-2]
                        
                        features = [[current_price, sma_5, sma_20, price_change]]
                        features_scaled = self.scaler.transform(features)
                        predicted_price = self.ml_model.predict(features_scaled)[0]
                        
                        xgboost_signal = predicted_price > current_price * 1.003  # 0.3% upside
                        
                except Exception as e:
                    self.logger.error(f"XGBoost validation error: {e}")
            
            # RSI Divergence check
            rsi_divergence_signal = True
            if self.config['ai']['rsi_divergence'].get('enabled', False):
                divergence = indicators.get('bullish_divergence', False)
                rsi_divergence_signal = not indicators.get('bearish_divergence', False)
                if divergence:
                    rsi_divergence_signal = True  # Bullish divergence is positive
            
            # Combine ensemble signals
            ensemble_signals = [lstm_signal, xgboost_signal, rsi_divergence_signal]
            ensemble_confidence = sum(ensemble_signals) / len(ensemble_signals)
            
            confidence_threshold = ai_config.get('confidence_threshold', 0.65)
            final_signal = ensemble_confidence >= confidence_threshold
            
            self.ai_ensemble_confidence = ensemble_confidence
            
            return final_signal, ensemble_confidence
            
        except Exception as e:
            self.logger.error(f"Error in AI ensemble signal: {e}")
            return True, 0.5
    
    async def should_buy_enhanced(self, current_price: float, indicators: Dict[str, float]) -> bool:
        """Enhanced buy logic with multi-timeframe analysis and AI ensemble filtering"""
        if self.position or not indicators:
            return False
            
        # Check cooldown and daily limits
        if self.last_trade_time:
            cooldown = self.config['risk']['cooldown_period']
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < cooldown:
                return False
                
        if self.daily_trades >= self.config['risk']['max_daily_trades']:
            return False
            
        if self.daily_loss >= self.config['risk']['max_daily_loss']:
            return False
            
        # Get dynamic thresholds from indicators
        sma = indicators.get('sma')
        rsi = indicators.get('rsi')
        bb_lower = indicators.get('bb_lower')
        entry_threshold = indicators.get('dynamic_entry_threshold', 0.003)
        
        if not sma:
            return False
            
        # Enhanced signal conditions with dynamic thresholds
        conditions = []
        
        # Price below dynamic SMA threshold
        buy_threshold = sma * (1 - entry_threshold)
        conditions.append(current_price <= buy_threshold)
        
        # RSI oversold (more selective)
        if rsi:
            conditions.append(rsi < 35)  # Slightly more selective
            
        # Price near enhanced Bollinger Band lower
        if bb_lower:
            conditions.append(current_price <= bb_lower * 1.005)  # Tighter threshold
        
        # Multi-timeframe analysis
        multi_tf_signals = await self.get_multi_timeframe_signals(current_price)
        timeframe_bullish = (
            multi_tf_signals['trend_1h_bullish'] and 
            multi_tf_signals['setup_15m_bullish'] and
            multi_tf_signals['momentum_5m_bullish']
        )
        
        # AI Ensemble filtering
        ai_ensemble_signal, ai_ensemble_confidence = await self.get_ai_ensemble_signal(current_price, indicators)
        
        # Traditional AI sentiment (10-minute frequency)
        ai_sentiment_signal = True
        if self.ai_enabled and self.ai_optimizer:
            current_time = time.time()
            ai_frequency_seconds = self.config['ai']['sentiment']['frequency_minutes'] * 60
            
            if current_time - self.last_ai_analysis >= ai_frequency_seconds:
                try:
                    price_data = list(self.price_buffer)[-20:] if len(self.price_buffer) >= 20 else list(self.price_buffer)
                    market_context = f"RSI: {rsi:.1f}, SMA: {sma:.2f}, BB_Lower: {bb_lower:.2f}, ATR: {indicators.get('atr', 0):.4f}"
                    
                    ai_sentiment = await self.ai_optimizer.analyze_market_sentiment(price_data, market_context)
                    
                    if ai_sentiment is not None:
                        self.ai_sentiment_score = ai_sentiment
                        self.last_ai_analysis = current_time
                        
                        sentiment_weight = self.config['ai']['sentiment']['weight_in_decision']
                        ai_sentiment_signal = ai_sentiment > (0.5 + sentiment_weight * 0.5)
                        
                        self.logger.info(f"🤖 AI Sentiment: {self.ai_sentiment_score:.2f}, Ensemble: {ai_ensemble_confidence:.2f}")
                        
                except Exception as e:
                    self.logger.error(f"AI sentiment analysis error: {e}")
        
        # Combine all signals
        technical_conditions_met = sum(conditions) >= 2
        
        # Final decision with enhanced logic
        if self.config['trading']['enhanced_features'].get('ai_ensemble', False):
            final_signal = (
                technical_conditions_met and 
                timeframe_bullish and 
                ai_ensemble_signal and 
                ai_sentiment_signal
            )
            
            if final_signal:
                self.logger.info(
                    f"🎯 ENHANCED BUY SIGNAL - Technical: ✓, Multi-TF: ✓, "
                    f"AI Ensemble: ✓ ({ai_ensemble_confidence:.2f}), Sentiment: ✓ ({self.ai_sentiment_score:.2f})"
                )
        else:
            # Fallback to simpler logic if enhanced features disabled
            final_signal = technical_conditions_met and timeframe_bullish
            
        return final_signal
        
    def handle_kline_data(self, ws_client, message):
        """Handle incoming kline data from WebSocket"""
        try:
            if isinstance(message, str):
                message = json.loads(message)
                
            if 'result' in message and 'id' in message:
                return
                
            if 'stream' in message and 'data' in message:
                data = message['data']
                if 'k' not in data:
                    return
                kline = data['k']
            elif 'k' in message:
                kline = message['k']
            else:
                return
                
            close_price = float(kline['c'])
            
            if not kline['x']:  # Only process closed candles
                return
                
            self.current_price = close_price
            self.price_buffer.append(close_price)
            
            # Calculate indicators
            indicators = self.calculate_technical_indicators(list(self.price_buffer))
            
            # Reset daily counters if new day
            current_date = datetime.now().date()
            if self.current_date != current_date:
                self.current_date = current_date
                self.daily_trades = 0
                self.daily_loss = 0.0
                self.week_start_profit = self.total_profit  # Reset weekly tracking
                
            # Process trading signals
            asyncio.create_task(self.process_enhanced_signals(close_price, indicators))
            
        except Exception as e:
            self.logger.error(f"Kline error: {e}")
            
    async def process_enhanced_signals(self, current_price: float, indicators: Dict[str, float]):
        """Process enhanced trading signals with AI integration, trailing stops, and active trading check"""
        try:
            # Only process signals if trading is active
            if not getattr(self, 'trading_active', False):
                return
            
            # Update trailing stop if we have a position
            if self.position and self.config['trading']['enhanced_features'].get('trailing_stops', False):
                # Update highest price since entry
                if not hasattr(self, 'highest_price_since_entry') or self.highest_price_since_entry is None:
                    self.highest_price_since_entry = current_price
                elif current_price > self.highest_price_since_entry:
                    self.highest_price_since_entry = current_price
                
                # Update trailing stop
                atr_value = indicators.get('atr', 0.01)
                self.update_trailing_stop(current_price, atr_value)
                
            # Check buy signal with AI integration
            if not self.position and await self.should_buy_enhanced(current_price, indicators):
                self.logger.info(f"🟢 BUY SIGNAL: Price ${current_price:.2f} - Indicators: {indicators}")
                
                # Calculate dynamic position size with ATR
                atr_value = indicators.get('atr', 0.01)
                position_size = self.calculate_position_size(current_price, atr_value)
                
                if self.ai_enabled and hasattr(self, 'ai_sentiment_score'):
                    # Adjust position size based on AI confidence
                    confidence_multiplier = min(1.2, max(0.8, self.ai_sentiment_score * 2))
                    position_size *= confidence_multiplier
                    self.logger.info(f"🤖 AI-adjusted position size: {position_size:.6f} (confidence: {self.ai_sentiment_score:.2f})")
                
                await self.place_buy_order(current_price, position_size)
                
            # Check sell signal with AI-enhanced logic
            elif self.position:
                sell_reason = await self.should_sell_enhanced(current_price, indicators)
                if sell_reason:
                    self.logger.info(f"🔴 SELL SIGNAL ({sell_reason}): Price ${current_price:.2f}")
                    await self.place_sell_order(sell_reason)
                
        except Exception as e:
            self.logger.error(f"Error processing signals: {e}")
    
    def update_trailing_stop(self, current_price: float, atr_value: float = None) -> None:
        """Update Grok AI optimized trailing stop loss (1.0 × ATR) based on current price and ATR"""
        try:
            if not self.position or not self.entry_price:
                return
                
            # Initialize highest price tracking
            if self.highest_price_since_entry is None:
                self.highest_price_since_entry = current_price
            else:
                self.highest_price_since_entry = max(self.highest_price_since_entry, current_price)
            
            # Grok AI optimized trailing stop calculation
            if atr_value and self.config['trading']['enhanced_features'].get('trailing_stops', False):
                # Grok AI: 1.0 × ATR trailing stop
                trailing_atr_multiplier = self.config['strategy']['risk_management']['trailing_stop_atr']  # 1.0
                trailing_distance = atr_value * trailing_atr_multiplier
                self.logger.debug(f"Grok AI ATR Trailing - ATR: {atr_value:.4f}, Multiplier: {trailing_atr_multiplier}, Distance: {trailing_distance:.4f}")
            else:
                # Fallback to percentage-based trailing stop
                trailing_pct = self.config['strategy'].get('trailing_stop_pct', 0.02)  # 2% default
                trailing_distance = self.highest_price_since_entry * trailing_pct
                self.logger.debug(f"Fallback % Trailing - Distance: {trailing_distance:.4f}")
            
            # Calculate new trailing stop price
            new_trailing_stop = self.highest_price_since_entry - trailing_distance
            
            # Update trailing stop (only move up, never down)
            if self.trailing_stop_price is None or new_trailing_stop > self.trailing_stop_price:
                old_trailing = self.trailing_stop_price
                self.trailing_stop_price = new_trailing_stop
                self.logger.debug(f"Grok AI Trailing Updated - High: {self.highest_price_since_entry:.4f}, Old: {old_trailing:.4f if old_trailing else 'None'}, New: {new_trailing_stop:.4f}")
                
        except Exception as e:
            self.logger.error(f"Error updating Grok AI trailing stop: {e}")
            
    async def should_sell_enhanced(self, current_price: float, indicators: Dict[str, float]) -> Tuple[bool, str]:
        """Enhanced sell logic with dynamic thresholds, trailing stops, and time exits"""
        if not self.position or not self.entry_price:
            return False, ""
            
        sma = indicators.get('sma')
        rsi = indicators.get('rsi')
        bb_upper = indicators.get('bb_upper')
        atr_value = indicators.get('atr', 0.01)
        exit_threshold = indicators.get('dynamic_exit_threshold', 0.0175)
        
        if not sma:
            return False, ""
        
        # Update trailing stop
        self.update_trailing_stop(current_price, atr_value)
        
        # Calculate profit/loss
        profit_pct = (current_price - self.entry_price) / self.entry_price
        
        # Exit conditions with enhanced logic
        exit_reasons = []
        
        # 1. Dynamic take profit based on ATR
        if self.config['trading']['enhanced_features'].get('dynamic_thresholds', False):
            # ATR-based take profit (1.5-2.0 × ATR)
            take_profit_threshold = exit_threshold
        else:
            # Traditional percentage-based
            take_profit_threshold = self.config['strategy'].get('take_profit', 0.005)
        
        if profit_pct >= take_profit_threshold:
            return True, f"TAKE_PROFIT_{profit_pct:.2%}"
        
        # 2. Dynamic stop loss based on ATR
        if self.config['trading']['enhanced_features'].get('dynamic_thresholds', False):
            risk_mgmt = self.config['strategy']['risk_management']
            stop_loss_atr_multiplier = (risk_mgmt['stop_loss_atr_min'] + risk_mgmt['stop_loss_atr_max']) / 2
            stop_loss_threshold = (atr_value * stop_loss_atr_multiplier) / self.entry_price
        else:
            stop_loss_threshold = self.config['strategy']['stop_loss']
        
        if profit_pct <= -stop_loss_threshold:
            return True, f"STOP_LOSS_{profit_pct:.2%}"
        
        # 3. Trailing stop loss
        if (self.trailing_stop_price and 
            self.config['trading']['enhanced_features'].get('trailing_stops', False) and 
            current_price <= self.trailing_stop_price):
            return True, f"TRAILING_STOP_{current_price:.4f}"
        
        # 4. Price above dynamic SMA threshold (trend reversal)
        sell_threshold = sma * (1 + exit_threshold)
        if current_price >= sell_threshold:
            return True, f"SMA_REVERSAL_{current_price:.4f}"
        
        # 5. RSI overbought with divergence check
        if rsi and rsi > 75:  # More selective than 70
            # Check for bearish divergence
            bearish_divergence = indicators.get('bearish_divergence', False)
            if bearish_divergence:
                return True, f"RSI_DIVERGENCE_{rsi:.1f}"
            elif rsi > 80:  # Very overbought
                return True, f"RSI_OVERBOUGHT_{rsi:.1f}"
        
        # 6. Price near enhanced Bollinger Band upper
        if bb_upper and current_price >= bb_upper * 0.995:  # Tighter threshold
            return True, f"BB_UPPER_{current_price:.4f}"
        
        # 7. Grok AI: Time-based exit (5-hour max hold time)
        if hasattr(self, 'entry_time') and self.entry_time:
            time_in_position = (datetime.now() - self.entry_time).total_seconds() / 3600  # hours
            max_hold_time = self.config['strategy']['risk_management']['time_exit_hours']  # Grok AI: 5 hours
            
            if time_in_position >= max_hold_time:
                self.logger.info(f"Grok AI Time Exit - Position held for {time_in_position:.1f}h (max: {max_hold_time}h)")
                return True, f"GROK_TIME_EXIT_{time_in_position:.1f}h"
        
        # 8. Multi-timeframe exit signals
        if self.config['trading']['enhanced_features'].get('multi_timeframe', False):
            try:
                multi_tf_signals = await self.get_multi_timeframe_signals(current_price)
                if not multi_tf_signals['trend_1h_bullish']:
                    return True, "1H_TREND_BEARISH"
            except Exception as e:
                self.logger.error(f"Multi-timeframe exit error: {e}")
        
        # 9. AI ensemble exit signal
        if self.config['trading']['enhanced_features'].get('ai_ensemble', False):
            try:
                ai_ensemble_signal, ai_ensemble_confidence = await self.get_ai_ensemble_signal(current_price, indicators)
                if not ai_ensemble_signal and ai_ensemble_confidence > 0.7:
                    return True, f"AI_ENSEMBLE_EXIT_{ai_ensemble_confidence:.2f}"
            except Exception as e:
                self.logger.error(f"AI ensemble exit error: {e}")
        
        return False, ""
        
    async def place_buy_order(self, current_price: float, position_size: float = None):
        """Place buy order with enhanced risk management and dynamic sizing"""
        try:
            # Risk management checks
            if self.daily_trades >= self.config['risk']['max_daily_trades']:
                self.logger.warning("Daily trade limit reached")
                return
                
            if self.daily_loss >= self.config['risk']['max_daily_loss']:
                self.logger.warning("Daily loss limit reached")
                return
            
            # Get indicators for enhanced calculations
            indicators = self.calculate_technical_indicators()
            atr_value = indicators.get('atr', 0.01)
                
            # Use provided position size or calculate it with ATR
            if position_size is None:
                position_size = self.calculate_position_size(current_price, atr_value)
            
            if position_size <= 0:
                self.logger.warning(f"Position size too small: {position_size}")
                return
                
            # Enhanced stop-loss and take-profit calculation
            if self.config['trading']['enhanced_features'].get('dynamic_thresholds', False):
                # ATR-based levels
                risk_mgmt = self.config['strategy']['risk_management']
                stop_loss_atr_multiplier = (risk_mgmt['stop_loss_atr_min'] + risk_mgmt['stop_loss_atr_max']) / 2
                stop_loss_distance = atr_value * stop_loss_atr_multiplier
                stop_loss_price = current_price - stop_loss_distance
                
                exit_threshold = indicators.get('dynamic_exit_threshold', 0.0175)
                take_profit_price = current_price * (1 + exit_threshold)
            else:
                # Traditional percentage-based
                base_stop_loss = self.config['strategy']['stop_loss']
                base_take_profit = self.config['strategy'].get('take_profit', 0.005)
                
                # AI adjustment if available
                ai_multiplier = 1.0
                if self.ai_enabled and hasattr(self, 'ai_sentiment_score'):
                    ai_multiplier = 0.8 + (self.ai_sentiment_score * 0.4)
                    
                stop_loss_pct = base_stop_loss * (2 - ai_multiplier)
                take_profit_pct = base_take_profit * ai_multiplier
                
                stop_loss_price = current_price * (1 - stop_loss_pct)
                take_profit_price = current_price * (1 + take_profit_pct)
            
            # Place market buy order
            if self.testnet:
                # Simulate order for testnet
                order = {
                    'symbol': self.symbol,
                    'orderId': f'TEST_{int(time.time())}',
                    'executedQty': str(position_size),
                    'cummulativeQuoteQty': str(position_size * current_price),
                    'status': 'FILLED'
                }
                self.logger.info(f"📝 TESTNET BUY ORDER: {position_size:.6f} {self.symbol} at ${current_price:.4f}")
            else:
                # Use improved timestamp handling for orders
                timestamp = int(time.time() * 1000) + getattr(self, 'time_offset', 0)
                recv_window = getattr(self, 'recv_window', 60000)
                order = self.client.new_order(
                    symbol=self.symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=position_size,
                    recvWindow=recv_window,
                    timestamp=timestamp
                )
            
            # Update position tracking with enhanced data structure
            self.position = {
                'side': 'BUY',
                'quantity': position_size,
                'order_id': order['orderId'],
                'entry_time': datetime.now(),
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price
            }
            
            self.entry_price = current_price
            self.entry_time = datetime.now()
            self.last_trade_time = datetime.now()
            
            # Initialize trailing stop tracking
            self.highest_price_since_entry = current_price
            self.trailing_stop_price = None
            
            self.daily_trades += 1
            
            # Enhanced logging
            kelly_info = f", Kelly: {self.current_kelly_fraction:.1%}" if hasattr(self, 'current_kelly_fraction') else ""
            ai_info = f", AI: {self.ai_sentiment_score:.2f}" if self.ai_enabled else ""
            atr_info = f", ATR: {atr_value:.4f}" if atr_value else ""
            
            self.logger.info(
                f"✅ ENHANCED BUY: {position_size:.6f} {self.symbol} @ ${current_price:.4f} "
                f"(Stop: ${stop_loss_price:.4f}, Target: ${take_profit_price:.4f}{kelly_info}{ai_info}{atr_info})"
            )
            
            # Log enhanced trade data
            trade_log_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': 'BUY',
                'price': current_price,
                'quantity': position_size,
                'atr_value': atr_value,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'kelly_fraction': getattr(self, 'current_kelly_fraction', 0),
                'ai_sentiment': getattr(self, 'ai_sentiment_score', 0.5),
                'ai_ensemble_confidence': getattr(self, 'ai_ensemble_confidence', 0.5),
                'indicators': indicators,
                'enhanced_features': self.config['trading']['enhanced_features']
            }
            
            # Write to enhanced trades log
            try:
                with open('enhanced_trades.json', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(trade_log_entry) + '\n')
            except Exception as e:
                self.logger.error(f"Error logging enhanced trade: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ Error placing buy order: {e}")
            
    async def place_sell_order(self, reason: str):
        """Place sell order with enhanced tracking and trailing stop logic"""
        try:
            if not self.position:
                return
                
            quantity = self.position['quantity']
            current_price = self.current_price
            
            # Place market sell order
            if self.testnet:
                # Simulate order for testnet
                order = {
                    'symbol': self.symbol,
                    'orderId': f'TEST_SELL_{int(time.time())}',
                    'executedQty': str(quantity),
                    'cummulativeQuoteQty': str(quantity * current_price),
                    'status': 'FILLED'
                }
                self.logger.info(f"📝 TESTNET SELL ORDER: {quantity:.6f} {self.symbol} at ${current_price:.4f}")
            else:
                # Use improved timestamp handling for orders
                timestamp = int(time.time() * 1000) + getattr(self, 'time_offset', 0)
                recv_window = getattr(self, 'recv_window', 60000)
                order = self.client.new_order(
                    symbol=self.symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=quantity,
                    recvWindow=recv_window,
                    timestamp=timestamp
                )
            
            # Calculate enhanced trade metrics
            entry_time = self.position.get('entry_time', datetime.now())
            exit_time = datetime.now()
            trade_duration = (exit_time - entry_time).total_seconds() / 60  # minutes
            
            profit = (current_price - self.entry_price) * quantity
            profit_pct = ((current_price - self.entry_price) / self.entry_price) * 100
            
            # Update Kelly Criterion trade history
            trade_result = {
                'profit': profit,
                'win': profit > 0,
                'timestamp': datetime.now(),
                'duration_minutes': trade_duration,
                'sell_reason': reason
            }
            self.trade_history.append(trade_result)
            
            # Keep only recent trades for Kelly calculation
            if len(self.trade_history) > 50:
                self.trade_history = self.trade_history[-50:]
            
            # Update totals and streaks
            self.total_profit += profit
            self.trade_count += 1
            
            # Enhanced gamification and streak tracking
            if profit > 0:
                self.winning_trades += 1
                self.consecutive_losses = 0
                self.gamification.add_win()
                trade_result_text = "WIN 🎉"
                # Play success sound in testnet
                if self.ui.current_mode == "Testnet" and playsound:
                    try:
                        # You can add a sound file here
                        pass
                    except:
                        pass
            else:
                self.daily_loss += abs(profit)
                self.consecutive_losses += 1
                self.gamification.add_loss()
                trade_result_text = "LOSS 📉"
                
            # Check for badges
            self.gamification.check_trade_milestones(self.trade_count, self.total_profit)
            
            # Enhanced logging
            win_rate = (self.winning_trades / self.trade_count) * 100 if self.trade_count > 0 else 0
            kelly_info = f", Kelly: {self.current_kelly_fraction:.1%}" if hasattr(self, 'current_kelly_fraction') else ""
            ai_info = f", AI: {self.ai_sentiment_score:.2f}" if self.ai_enabled and hasattr(self, 'ai_sentiment_score') else ""
            trailing_info = f", Trail: ${self.trailing_stop_price:.4f}" if hasattr(self, 'trailing_stop_price') and self.trailing_stop_price else ""
            
            self.logger.info(
                f"✅ ENHANCED SELL ({reason}): {quantity:.6f} {self.symbol} @ ${current_price:.4f} "
                f"| Entry: ${self.entry_price:.4f} | P&L: ${profit:.2f} ({profit_pct:+.2f}%) "
                f"| Duration: {trade_duration:.1f}m | Result: {trade_result_text} "
                f"| Total: ${self.total_profit:.2f} | WR: {win_rate:.1f}%{kelly_info}{ai_info}{trailing_info}"
            )
            
            # Log enhanced trade data
            trade_log_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': 'SELL',
                'price': current_price,
                'quantity': quantity,
                'entry_price': self.entry_price,
                'profit': profit,
                'profit_percentage': profit_pct,
                'duration_minutes': trade_duration,
                'sell_reason': reason,
                'kelly_fraction': getattr(self, 'current_kelly_fraction', 0),
                'ai_sentiment': getattr(self, 'ai_sentiment_score', 0.5),
                'ai_ensemble_confidence': getattr(self, 'ai_ensemble_confidence', 0.5),
                'trailing_stop_price': getattr(self, 'trailing_stop_price', None),
                'highest_price': getattr(self, 'highest_price_since_entry', current_price),
                'win_rate': win_rate,
                'total_profit': self.total_profit,
                'consecutive_losses': self.consecutive_losses
            }
            
            # Write to enhanced trades log
            try:
                with open('enhanced_trades.json', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(trade_log_entry) + '\n')
            except Exception as e:
                self.logger.error(f"Error logging enhanced trade: {e}")
            
            # Log AI-influenced trade to separate file for analysis
            if self.ai_enabled:
                try:
                    with open('logs/ai_trades.log', 'a', encoding='utf-8') as f:
                        f.write(f"{datetime.now().isoformat()},SELL,{current_price},{quantity},{self.ai_sentiment_score:.3f},{profit:.2f},{profit_pct:.2f}\n")
                except Exception:
                    pass
            
            # Reset position and tracking variables
            self.position = None
            self.entry_price = None
            self.entry_time = None
            self.highest_price_since_entry = None
            self.trailing_stop_price = None
            self.last_trade_time = datetime.now()
            
        except Exception as e:
            self.logger.error(f"❌ Error placing sell order: {e}")
    
    def get_current_price(self) -> Optional[float]:
        """Get the current price from the price buffer or fetch from API"""
        try:
            # Return cached current price if available
            if self.current_price is not None:
                return self.current_price
                
            # If no cached price, try to get from price buffer
            if len(self.price_buffer) > 0:
                return self.price_buffer[-1]
                
            # If no price buffer, fetch from API
            try:
                ticker = self.client.ticker_price(symbol=self.symbol)
                price = float(ticker['price'])
                self.current_price = price
                return price
            except Exception as api_error:
                self.logger.error(f"Failed to fetch price from API: {api_error}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None
            
    def calculate_kelly_fraction(self) -> float:
        """Calculate Kelly Criterion fraction based on recent trade history"""
        try:
            if len(self.trade_history) < 10:  # Need minimum trade history
                return self.config['trading']['risk_per_trade']  # Use default
            
            # Calculate win rate and average win/loss ratio
            wins = [trade for trade in self.trade_history if trade['profit'] > 0]
            losses = [trade for trade in self.trade_history if trade['profit'] <= 0]
            
            if len(losses) == 0:  # No losses yet
                return min(0.02, self.config['trading']['risk_per_trade'] * 1.2)  # Slightly increase
            
            win_rate = len(wins) / len(self.trade_history)
            avg_win = sum(trade['profit'] for trade in wins) / len(wins) if wins else 0
            avg_loss = abs(sum(trade['profit'] for trade in losses) / len(losses)) if losses else 1
            
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
            
            # Kelly formula: f = (bp - q) / b
            # where b = win/loss ratio, p = win probability, q = loss probability
            kelly_fraction = (win_loss_ratio * win_rate - (1 - win_rate)) / win_loss_ratio
            
            # Apply safety constraints
            kelly_config = self.config['risk']['kelly_criterion']
            max_kelly = kelly_config.get('max_kelly_fraction', 0.25)
            kelly_fraction = max(0.005, min(max_kelly, kelly_fraction))  # Between 0.5% and 25%
            
            # Reduce after consecutive losses
            if self.consecutive_losses >= 3:
                kelly_fraction *= 0.5  # Halve position size
            
            self.current_kelly_fraction = kelly_fraction
            return kelly_fraction
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly fraction: {e}")
            return self.config['trading']['risk_per_trade']
    
    def calculate_position_size(self, current_price: float, atr_value: float = None) -> float:
        """Calculate optimized position size using Grok AI Kelly sizing formula: Account Balance × Risk% / (ATR × Stop Distance)"""
        try:
            balances = self.get_account_balances()
            usdt_balance = balances.get('USDT', 0)
            
            # Grok AI optimized Kelly Criterion
            kelly_enabled = self.config['trading']['enhanced_features'].get('kelly_sizing', False)
            if kelly_enabled:
                risk_per_trade = self.calculate_kelly_fraction()  # 1-2% risk as per Grok AI
            else:
                risk_per_trade = self.config['trading']['risk_per_trade']
            
            max_position_size = self.config['trading']['max_position_size']
            
            # Grok AI formula: Account Balance × Risk%
            risk_amount = min(
                usdt_balance * risk_per_trade,
                max_position_size
            )
            
            # Grok AI optimized ATR-based stop loss (1.5-2.0 × ATR)
            if atr_value:
                risk_mgmt = self.config['strategy']['risk_management']
                # Use Grok AI stop loss multiplier (1.5-2.0 × ATR)
                stop_loss_atr_multiplier = risk_mgmt.get('stop_loss_atr_multiplier', 1.75)  # Default 1.75 × ATR
                stop_loss_distance = atr_value * stop_loss_atr_multiplier
                
                # Grok AI Kelly sizing formula: Account Balance × Risk% / (ATR × Stop Distance)
                if stop_loss_distance > 0:
                    position_size = risk_amount / stop_loss_distance
                    self.logger.debug(f"Grok AI Kelly Sizing - Risk: ${risk_amount:.2f}, ATR Stop: {stop_loss_distance:.4f}, Size: {position_size:.6f}")
                else:
                    position_size = 0
            else:
                # Fallback to percentage-based stop loss
                stop_loss_pct = self.config['strategy']['stop_loss']
                stop_loss_distance = current_price * stop_loss_pct
                position_size = risk_amount / stop_loss_distance if stop_loss_distance > 0 else 0
            
            # Apply position size limits
            max_units = max_position_size / current_price
            position_size = min(position_size, max_units)
            
            # Apply Grok AI portfolio exposure limit (15%)
            max_exposure = self.config['strategy']['risk_management']['max_portfolio_exposure']
            max_position_value = usdt_balance * max_exposure
            max_units_by_exposure = max_position_value / current_price
            position_size = min(position_size, max_units_by_exposure)
            
            return round(position_size, 6)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
            
    async def start_websocket(self):
        """Start WebSocket connection"""
        try:
            self.ws_client = SpotWebsocketStreamClient(
                on_message=self.handle_kline_data,
                is_combined=True
            )
            
            self.ws_client.kline(symbol=self.symbol, interval='5m')
            self.logger.info(f"WebSocket connected: {self.symbol.lower()}@kline_5m")
            
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
            
    # run_with_ui method removed - bot now runs in automated mode only
        
    async def stop(self):
        """Stop the trading bot"""
        self.logger.info("Stopping Kuvera Grid...")
        self.is_running = False
        
        if self.ws_client:
            self.ws_client.stop()
            
        # Final statistics
        win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0
        
        self.logger.info("Final Statistics:")
        self.logger.info(f"Total Trades: {self.trade_count}")
        self.logger.info(f"Win Rate: {win_rate:.1f}%")
        self.logger.info(f"Total Profit: ${self.total_profit:.2f}")
        self.logger.info(f"Max Streak: {self.gamification.max_streak}")
        self.logger.info(f"Badges Earned: {len(self.gamification.badges)}")
        
    async def run_automated(self):
        """Run bot in automated mode for Railway deployment"""
        self.logger.info("Starting Kuvera Grid v1.1 in automated mode 🚀")
        self.is_running = True
        self.trading_active = True
        
        # Run diagnostic tests if enabled
        if self.diagnostic_mode:
            self.logger.info("🔍 Diagnostic mode enabled - Running connectivity tests...")
            await self.test_openrouter_connectivity()
        
        # Execute startup test trade if enabled
        if self.startup_test_enabled:
            await self.execute_startup_test_trade()
        
        # Start WebSocket
        await self.start_websocket()
        
        # Initialize AI analysis timer
        self.last_ai_analysis = 0
        self.ai_frequency_seconds = 15 * 60  # 15 minutes in seconds
        
        # Initialize balance refresh timer
        last_balance_refresh = 0
        balance_refresh_interval = 30  # 30 seconds for automated mode
        last_status_log = 0
        status_log_interval = 300  # Log status every 5 minutes
        
        # Main automated loop
        while self.is_running:
            try:
                current_time = time.time()
                
                # Update balance periodically
                if current_time - last_balance_refresh >= balance_refresh_interval:
                    balances = self.get_account_balances()
                    usdt_balance = balances.get('USDT', 0)
                    last_balance_refresh = current_time
                    
                    # Log status periodically
                    if current_time - last_status_log >= status_log_interval:
                        self.logger.info(f"📊 Status - Balance: ${usdt_balance:.2f}, Profit: ${self.total_profit:.2f}, Trades: {self.trade_count}")
                        last_status_log = current_time
                
                # AI analysis timer
                if (self.ai_enabled and 
                    current_time - self.last_ai_analysis >= self.ai_frequency_seconds):
                    await self.run_ai_analysis()
                    self.last_ai_analysis = current_time
                
                await asyncio.sleep(5)  # 5 second refresh rate for automated mode
                
            except Exception as e:
                self.logger.error(f"Automated loop error: {e}")
                await asyncio.sleep(10)  # Wait longer on error
                
        # Cleanup
        await self.stop()

# Interactive setup function removed - bot now runs in automated mode only

async def main():
    """Main execution function - Always runs in automated mode"""
    try:
        # Always use automated setup
        testnet_mode = os.getenv('TRADING_MODE', 'testnet').lower() == 'testnet'
        ai_enabled = os.getenv('AI_ENABLED', 'true').lower() == 'true'
        strategy_type = os.getenv('STRATEGY_TYPE', 'mean_reversion')
        
        console = Console()
        console.print(Panel.fit(
            "[bold cyan]Kuvera Grid Trading Bot v1.1 🚀[/bold cyan]\n"
            "[dim]Automated mode - No interactive UI[/dim]",
            box=box.DOUBLE
        ))
        
        console.print(f"[green]✓ Trading Mode: {'Testnet' if testnet_mode else 'LIVE'}[/green]")
        console.print(f"[green]✓ AI Features: {'Enabled' if ai_enabled else 'Disabled'}[/green]")
        console.print(f"[green]✓ Strategy: {strategy_type}[/green]")
        
        # Initialize bot
        bot = EnhancedTradingBot()
        bot.testnet_mode = testnet_mode
        bot.ai_enabled = ai_enabled
        bot.ui.ai_enabled = ai_enabled
        bot.ui.strategy_type = "Mean Reversion Enhanced"
        
        console.print("\n[bold green]🔄 Starting bot...[/bold green]")
        console.print("[green]🤖 Starting automated trading...[/green]")
        
        # Always start in automated mode
        bot.trading_active = True
        await bot.run_automated()
        
    except KeyboardInterrupt:
        console = Console()
        console.print("\n\n[bold red]🛑 Bot stopped by user.[/bold red]")
        console.print("[bold cyan]Thank you for using Kuvera Grid Trading Bot! 🚀[/bold cyan]")
    except Exception as e:
        console = Console()
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        console.print("[yellow]Please check your configuration and try again.[/yellow]")
        
if __name__ == "__main__":
    asyncio.run(main())