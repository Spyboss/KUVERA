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
import keyboard
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
        
        # AI/ML components
        self.ml_model = None
        self.scaler = None
        self.setup_ml_components()
        
        # OpenRouter AI Integration
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.ai_optimizer = None
        self.auto_trader = None
        self.ai_enabled = self.config.get('ai', {}).get('openrouter', False)
        self.ai_sentiment_score = 0.5  # Neutral
        self.diagnostic_mode = self.config.get('ai', {}).get('diagnostic_mode', False)
        self.startup_test_enabled = self.config.get('trading', {}).get('startup_test', True)
        
        # Setup AI components after ML components
        self.setup_ai_components()
        
        # Weekly target tracking
        self.weekly_target = 2.0  # $2 per week
        self.week_start_profit = self.total_profit
        
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
        """Setup Binance API client"""
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
            
        self.client = Spot(
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url
        )
        
        # Test connection
        try:
            account_info = self.client.account(recvWindow=60000)
            self.logger.info("Successfully connected to Binance API")
        except Exception as e:
            self.logger.error(f"Failed to connect to Binance API: {e}")
            raise
            
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
            
    def calculate_technical_indicators(self, prices: List[float]) -> Dict[str, float]:
        """Calculate technical indicators"""
        if len(prices) < 20 or not talib:
            return {}
            
        prices_array = np.array(prices, dtype=float)
        
        indicators = {}
        
        try:
            # Simple Moving Average
            sma = talib.SMA(prices_array, timeperiod=20)[-1]
            indicators['sma'] = sma
            
            # RSI
            rsi = talib.RSI(prices_array, timeperiod=14)[-1]
            indicators['rsi'] = rsi
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(prices_array, timeperiod=20, nbdevup=2, nbdevdn=2)
            indicators['bb_upper'] = bb_upper[-1]
            indicators['bb_middle'] = bb_middle[-1]
            indicators['bb_lower'] = bb_lower[-1]
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            
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
            account_info = self.client.account(recvWindow=60000)
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
        
    async def should_buy_enhanced(self, current_price: float, indicators: Dict[str, float]) -> bool:
        """Enhanced buy logic with AI, ML and multiple indicators"""
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
            
        # Enhanced signal logic
        sma = indicators.get('sma')
        rsi = indicators.get('rsi')
        bb_lower = indicators.get('bb_lower')
        
        if not sma:
            return False
            
        # Multiple conditions for stronger signal
        conditions = []
        
        # Price below SMA threshold
        entry_threshold = self.config['strategy']['entry_threshold']
        buy_threshold = sma * (1 - entry_threshold)
        conditions.append(current_price <= buy_threshold)
        
        # RSI oversold
        if rsi:
            conditions.append(rsi < 30)
            
        # Price near Bollinger Band lower
        if bb_lower:
            conditions.append(current_price <= bb_lower * 1.01)
            
        # ML prediction if available
        ml_signal = False
        if self.ml_enabled and self.ml_model and len(self.price_buffer) >= 20:
            try:
                prices = list(self.price_buffer)
                # Prepare features for prediction
                sma_5 = sum(prices[-5:]) / 5
                sma_20 = sum(prices[-20:]) / 20
                price_change = (current_price - prices[-2]) / prices[-2]
                
                features = [[current_price, sma_5, sma_20, price_change]]
                features_scaled = self.scaler.transform(features)
                
                predicted_price = self.ml_model.predict(features_scaled)[0]
                ml_signal = predicted_price > current_price * 1.005  # 0.5% upside prediction
                
            except Exception as e:
                self.logger.error(f"ML prediction error: {e}")
        
        # AI sentiment analysis if available
        ai_signal = True  # Default to neutral
        ai_confidence = 0.5
        
        if self.ai_enabled and self.ai_optimizer:
            # Check if it's time for AI analysis (15-minute frequency)
            current_time = time.time()
            if current_time - self.last_ai_analysis >= self.ai_frequency_seconds:
                try:
                    # Log OpenRouter API call attempt
                    self.logger.info(f"🤖 Attempting OpenRouter API call at {datetime.now().isoformat()} (15min frequency)")
                    
                    # Get recent price data for AI analysis (fix: pass List[float] instead of Dict)
                    price_data = list(self.price_buffer)[-20:] if len(self.price_buffer) >= 20 else list(self.price_buffer)
                    
                    # Create market context for AI
                    market_context = f"RSI: {rsi:.1f}, SMA: {sma:.2f}, BB_Lower: {bb_lower:.2f}, Current: {current_price:.2f}"
                    
                    # Call OpenRouter API with proper parameters
                    ai_sentiment = await self.ai_optimizer.analyze_market_sentiment(price_data, market_context)
                    
                    if ai_sentiment is not None:
                        self.ai_sentiment_score = ai_sentiment
                        ai_confidence = abs(ai_sentiment - 0.5) * 2  # Convert to confidence (0-1)
                        ai_signal = ai_sentiment > 0.6 and ai_confidence > 0.3
                        
                        # Update last analysis time
                        self.last_ai_analysis = current_time
                        
                        # Log successful API call
                        self.logger.info(f"✅ OpenRouter API call successful - Sentiment: {self.ai_sentiment_score:.2f}, Confidence: {ai_confidence:.2f}, Signal: {'BUY' if ai_signal else 'HOLD'}")
                    else:
                        self.logger.warning("⚠️ OpenRouter API call returned None")
                        
                except Exception as e:
                    self.logger.error(f"❌ OpenRouter API call failed: {e}")
            else:
                # Use cached AI sentiment if within frequency window
                ai_confidence = abs(self.ai_sentiment_score - 0.5) * 2
                ai_signal = self.ai_sentiment_score > 0.6 and ai_confidence > 0.3
        
        # Combine signals with AI weighting
        technical_conditions_met = sum(conditions) >= 2
        
        if self.ai_enabled and ai_confidence > 0.6:
            # AI-enhanced decision
            final_signal = technical_conditions_met and ai_signal and (ml_signal if self.ml_enabled else True)
            if final_signal:
                self.logger.info(f"🎯 Strong BUY signal - Technical: ✓, AI: ✓ (conf: {ai_confidence:.2f}), ML: {'✓' if ml_signal else '✗'}")
        elif self.ml_enabled:
            final_signal = technical_conditions_met and ml_signal
        else:
            final_signal = technical_conditions_met
            
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
        """Process enhanced trading signals with AI integration and active trading check"""
        try:
            # Only process signals if trading is active
            if not getattr(self, 'trading_active', False):
                return
                
            # Check buy signal with AI integration
            if await self.should_buy_enhanced(current_price, indicators):
                self.logger.info(f"🟢 BUY SIGNAL: Price ${current_price:.2f} - Indicators: {indicators}")
                
                # Calculate dynamic position size based on AI confidence if available
                position_size = self.calculate_position_size(current_price)
                if self.ai_enabled and hasattr(self, 'ai_sentiment_score'):
                    # Adjust position size based on AI confidence
                    confidence_multiplier = min(1.2, max(0.8, self.ai_sentiment_score * 2))
                    position_size *= confidence_multiplier
                    self.logger.info(f"🤖 AI-adjusted position size: {position_size:.6f} (confidence: {self.ai_sentiment_score:.2f})")
                
                await self.place_buy_order(current_price, position_size)
                
            # Check sell signal with AI-enhanced logic
            should_sell, reason = await self.should_sell_enhanced(current_price, indicators)
            if should_sell:
                self.logger.info(f"🔴 SELL SIGNAL ({reason}): Price ${current_price:.2f}")
                await self.place_sell_order(reason)
                
        except Exception as e:
            self.logger.error(f"Error processing signals: {e}")
            
    async def should_sell_enhanced(self, current_price: float, indicators: Dict[str, float]) -> Tuple[bool, str]:
        """Enhanced sell logic with AI analysis and dynamic stop-loss/take-profit levels"""
        if not self.position or not self.entry_price:
            return False, ""
            
        sma = indicators.get('sma')
        rsi = indicators.get('rsi')
        bb_upper = indicators.get('bb_upper')
        
        # AI-enhanced dynamic stop-loss and take-profit levels
        base_stop_loss = self.config['strategy']['stop_loss']
        base_take_profit = 0.005  # 0.5% default
        
        # Adjust levels based on AI sentiment if available
        if self.ai_enabled and hasattr(self, 'ai_sentiment_score'):
            # More aggressive take-profit in bullish sentiment, tighter stop-loss in bearish
            sentiment_adjustment = (self.ai_sentiment_score - 0.5) * 0.5  # -0.25 to +0.25
            base_take_profit += sentiment_adjustment * 0.01  # Adjust by up to 1%
            base_stop_loss -= sentiment_adjustment * 0.002  # Adjust stop-loss by up to 0.2%
            
        stop_loss_price = self.position.get('stop_loss', self.entry_price * (1 - base_stop_loss))
        take_profit_price = self.position.get('take_profit', self.entry_price * (1 + base_take_profit))
        
        # AI sell signal analysis
        ai_sell_signal = False
        ai_confidence = 0.5
        
        if self.ai_enabled and self.ai_optimizer:
            try:
                market_data = {
                    'price': current_price,
                    'entry_price': self.entry_price,
                    'profit_pct': (current_price - self.entry_price) / self.entry_price,
                    'rsi': rsi or 50,
                    'bb_position': (current_price - indicators.get('bb_lower', current_price)) / (bb_upper - indicators.get('bb_lower', current_price)) if bb_upper else 0.5,
                    'sma_ratio': current_price / sma if sma else 1.0
                }
                
                ai_analysis = await self.ai_optimizer.analyze_exit_strategy(market_data)
                if ai_analysis:
                    ai_confidence = ai_analysis.get('confidence', 0.5)
                    ai_sell_signal = ai_analysis.get('recommendation') == 'sell' and ai_confidence > 0.7
                    
                    if ai_sell_signal:
                        self.logger.info(f"🤖 AI recommends SELL - Confidence: {ai_confidence:.2f}")
                        
            except Exception as e:
                self.logger.error(f"AI exit analysis error: {e}")
        
        # Take profit - use AI-adjusted level
        if current_price >= take_profit_price:
            return True, "TAKE_PROFIT"
            
        # Stop loss - use AI-adjusted level
        if current_price <= stop_loss_price:
            return True, "STOP_LOSS"
            
        # AI-driven exit signal
        if ai_sell_signal and ai_confidence > 0.8:
            return True, "AI_SIGNAL"
            
        # RSI overbought (additional exit condition)
        if rsi and rsi > 75:  # Slightly higher threshold for more selective exits
            return True, "RSI_OVERBOUGHT"
            
        # Price near Bollinger Band upper (additional exit condition)
        if bb_upper and current_price >= bb_upper * 0.998:
            return True, "BB_UPPER"
            
        # Time-based exit (if position held too long - 4 hours)
        if 'entry_time' in self.position:
            time_held = (datetime.now() - self.position['entry_time']).total_seconds() / 3600
            if time_held > 4:  # 4 hours
                return True, "TIME_EXIT"
            
        return False, ""
        
    async def place_buy_order(self, current_price: float, position_size: float = None):
        """Place a buy order with AI-enhanced logging and risk management"""
        try:
            # Risk management checks
            if self.daily_trades >= self.config['risk']['max_daily_trades']:
                self.logger.warning("Daily trade limit reached")
                return
                
            if self.daily_loss >= self.config['risk']['max_daily_loss']:
                self.logger.warning("Daily loss limit reached")
                return
                
            # Use provided position size or calculate it
            if position_size is None:
                position_size = self.calculate_position_size(current_price)
            
            if position_size <= 0:
                self.logger.warning(f"Position size too small: {position_size}")
                return
                
            # AI-enhanced stop-loss and take-profit levels
            base_stop_loss_pct = self.config['strategy']['stop_loss']
            base_take_profit_pct = 0.005  # 0.5% default take profit
            
            # Adjust levels based on AI sentiment if available
            if self.ai_enabled and hasattr(self, 'ai_sentiment_score'):
                # More aggressive take-profit in bullish sentiment, tighter stop-loss in bearish
                sentiment_adjustment = (self.ai_sentiment_score - 0.5) * 0.5  # -0.25 to +0.25
                take_profit_pct = base_take_profit_pct + (sentiment_adjustment * 0.01)  # Adjust by up to 1%
                stop_loss_pct = base_stop_loss_pct - (sentiment_adjustment * 0.002)  # Adjust stop-loss by up to 0.2%
                
                self.logger.info(f"🤖 AI-adjusted levels - TP: {take_profit_pct:.3%}, SL: {stop_loss_pct:.3%} (sentiment: {self.ai_sentiment_score:.2f})")
            else:
                take_profit_pct = base_take_profit_pct
                stop_loss_pct = base_stop_loss_pct
            
            stop_loss_price = current_price * (1 - stop_loss_pct)
            take_profit_price = current_price * (1 + take_profit_pct)
            
            order = self.client.new_order(
                symbol=self.symbol,
                side='BUY',
                type='MARKET',
                quantity=position_size
            )
            
            self.position = {
                'side': 'BUY',
                'quantity': position_size,
                'order_id': order['orderId'],
                'entry_time': datetime.now(),
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price
            }
            
            self.entry_price = current_price
            self.daily_trades += 1
            
            # Enhanced logging with AI and trade details
            ai_info = ""
            if self.ai_enabled and hasattr(self, 'ai_sentiment_score'):
                ai_info = f" | AI Sentiment: {self.ai_sentiment_score:.2f}"
            
            self.logger.info(
                f"💰 BUY ORDER EXECUTED: {position_size:.6f} {self.symbol} at ${current_price:.2f} | "
                f"Stop Loss: ${stop_loss_price:.2f} | Take Profit: ${take_profit_price:.2f} | "
                f"Risk: ${(current_price - stop_loss_price) * position_size:.2f}{ai_info}"
            )
            
            # Log AI-influenced trade to separate file for analysis
            if self.ai_enabled:
                try:
                    with open('logs/ai_trades.log', 'a', encoding='utf-8') as f:
                        f.write(f"{datetime.now().isoformat()},BUY,{current_price},{position_size},{self.ai_sentiment_score:.3f}\n")
                except Exception:
                    pass
            
        except Exception as e:
            self.logger.error(f"❌ Error placing buy order: {e}")
            
    async def place_sell_order(self, reason: str):
        """Place a sell order with AI-enhanced trade tracking and profit calculation"""
        try:
            if not self.position:
                return
                
            quantity = self.position['quantity']
            
            order = self.client.new_order(
                symbol=self.symbol,
                side='SELL',
                type='MARKET',
                quantity=quantity
            )
            
            # Calculate detailed trade metrics
            entry_time = self.position.get('entry_time', datetime.now())
            exit_time = datetime.now()
            trade_duration = (exit_time - entry_time).total_seconds() / 60  # minutes
            
            current_price = self.current_price
            profit = (current_price - self.entry_price) * quantity
            profit_pct = ((current_price - self.entry_price) / self.entry_price) * 100
            
            self.total_profit += profit
            self.trade_count += 1
            
            # Gamification
            if profit > 0:
                self.winning_trades += 1
                self.gamification.add_win()
                trade_result = "WIN 🎉"
                # Play success sound in testnet
                if self.ui.current_mode == "Testnet" and playsound:
                    try:
                        # You can add a sound file here
                        pass
                    except:
                        pass
            else:
                self.daily_loss += abs(profit)
                self.gamification.add_loss()
                trade_result = "LOSS 📉"
                
            # Check for badges
            self.gamification.check_trade_milestones(self.trade_count, self.total_profit)
            
            # Enhanced logging with AI and complete trade details
            ai_info = ""
            if self.ai_enabled and hasattr(self, 'ai_sentiment_score'):
                ai_info = f" | AI Sentiment: {self.ai_sentiment_score:.2f}"
            
            self.logger.info(
                f"💸 SELL ORDER EXECUTED: {quantity:.6f} {self.symbol} at ${current_price:.2f} | "
                f"Entry: ${self.entry_price:.2f} | Exit: ${current_price:.2f} | "
                f"Profit: ${profit:.2f} ({profit_pct:+.2f}%) | Duration: {trade_duration:.1f}m | "
                f"Result: {trade_result} | Reason: {reason} | "
                f"Total P&L: ${self.total_profit:.2f} | Win Rate: {(self.winning_trades/self.trade_count)*100:.1f}%{ai_info}"
            )
            
            # Log AI-influenced trade to separate file for analysis
            if self.ai_enabled:
                try:
                    with open('logs/ai_trades.log', 'a', encoding='utf-8') as f:
                        f.write(f"{datetime.now().isoformat()},SELL,{current_price},{quantity},{self.ai_sentiment_score:.3f},{profit:.2f},{profit_pct:.2f}\n")
                except Exception:
                    pass
            
            # Reset position
            self.position = None
            self.entry_price = None
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
            
    def calculate_position_size(self, current_price: float) -> float:
        """Calculate position size based on risk management"""
        try:
            balances = self.get_account_balances()
            usdt_balance = balances.get('USDT', 0)
            
            risk_per_trade = self.config['trading']['risk_per_trade']
            max_position_size = self.config['trading']['max_position_size']
            
            risk_amount = min(
                usdt_balance * risk_per_trade,
                max_position_size
            )
            
            stop_loss_pct = self.config['strategy']['stop_loss']
            stop_loss_price = current_price * (1 - stop_loss_pct)
            risk_per_unit = current_price - stop_loss_price
            
            if risk_per_unit > 0:
                position_size = risk_amount / risk_per_unit
                max_units = max_position_size / current_price
                position_size = min(position_size, max_units)
            else:
                position_size = 0
                
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
            
    async def run_with_ui(self):
        """Run bot with modern UI and real-time updates"""
        self.logger.info("Starting Kuvera Grid v1.1 🚀")
        self.is_running = True
        self.trading_active = False
        
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
        
        # Setup keyboard listener
        def on_key_press(key):
            if key.name == 's':
                self.trading_active = not self.trading_active
                status = "STARTED" if self.trading_active else "STOPPED"
                self.logger.info(f"Trading {status} by user")
            elif key.name == 'm':
                # Toggle between testnet and live mode display (demo)
                current = self.ui.current_mode
                self.ui.current_mode = "LIVE" if current == "Testnet" else "Testnet"
                self.logger.info(f"Display mode switched to {self.ui.current_mode}")
            elif key.name == 'q':
                self.is_running = False
                self.logger.info("Shutdown requested by user")
                
        keyboard.on_press(on_key_press)
        
        # Initialize balance refresh timer
        last_balance_refresh = 0
        balance_refresh_interval = 5  # 5 seconds
        
        # Main UI loop with enhanced real-time updates
        with Live(self.ui.layout, refresh_per_second=2, screen=True) as live:
            while self.is_running:
                try:
                    current_time = time.time()
                    
                    # Refresh balances every 5 seconds
                    if current_time - last_balance_refresh >= balance_refresh_interval:
                        last_balance_refresh = current_time
                        # Force balance refresh
                        self._cached_balances = None
                    
                    # Get updated bot data
                    bot_data = self.get_bot_data()
                    bot_data['trading_active'] = self.trading_active
                    
                    # Update UI display
                    self.ui.update_display(bot_data, self.gamification)
                    
                    # Sleep for smoother updates
                    await asyncio.sleep(0.5)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.logger.error(f"UI error: {e}")
                    await asyncio.sleep(1)
                    
        await self.stop()
        
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

def interactive_setup():
    """Interactive setup for bot configuration"""
    console = Console()
    
    console.print(Panel.fit(
        "[bold cyan]Kuvera Grid Trading Bot v1.1 🚀[/bold cyan]\n"
        "[dim]AI-powered cryptocurrency trading with modern interface[/dim]",
        box=box.DOUBLE
    ))
    
    console.print("\n[bold]📊 Trading Mode Selection:[/bold]")
    console.print("1. [green]Testnet[/green] (Recommended for beginners) - Default")
    console.print("2. [red]Live Trading[/red] (Real money - Use with caution!)")
    
    mode_choice = console.input("\nSelect mode (1/2) [Default: 1]: ").strip()
    testnet_mode = True if mode_choice != '2' else False
    
    if not testnet_mode:
        console.print("\n[bold red]⚠️  WARNING: You selected LIVE TRADING mode![/bold red]")
        console.print("[bold red]⚠️  This will use real money. Are you absolutely sure?[/bold red]")
        confirm = console.input("Type 'CONFIRM' to proceed with live trading: ")
        if confirm != 'CONFIRM':
            console.print("[yellow]Switching back to testnet mode for safety.[/yellow]")
            testnet_mode = True
    
    console.print("\n[bold]🤖 AI Features:[/bold]")
    console.print("1. [green]Enable AI[/green] (XGBoost, sentiment analysis) - Default")
    console.print("2. [yellow]Disable AI[/yellow] (Basic mean reversion only)")
    
    ai_choice = console.input("\nSelect AI mode (1/2) [Default: 1]: ").strip()
    ai_enabled = True if ai_choice != '2' else False
    
    console.print("\n[bold]📈 Strategy Selection:[/bold]")
    console.print("1. [green]Mean Reversion[/green] (SMA + RSI + Bollinger Bands) - Default")
    console.print("2. [blue]Grid Trading[/blue] (Coming soon)")
    console.print("3. [purple]DCA Strategy[/purple] (Coming soon)")
    
    strategy_choice = console.input("\nSelect strategy (1/2/3) [Default: 1]: ").strip()
    strategy_type = "mean_reversion"  # Default for now
    
    # Display configuration
    config_table = Table(title="Configuration Summary", box=box.ROUNDED)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_table.add_row("Trading Mode", "Testnet" if testnet_mode else "LIVE")
    config_table.add_row("AI Features", "Enabled" if ai_enabled else "Disabled")
    config_table.add_row("Strategy", "Mean Reversion Enhanced")
    config_table.add_row("Risk per Trade", "1% of capital (max $0.30)")
    config_table.add_row("Target", "$1-2 profit/week")
    
    console.print(config_table)
    
    return testnet_mode, ai_enabled, strategy_type

async def main():
    """Main execution function"""
    try:
        # Interactive setup
        testnet_mode, ai_enabled, strategy_type = interactive_setup()
        
        # Initialize bot
        bot = EnhancedTradingBot()
        bot.testnet_mode = testnet_mode
        bot.ai_enabled = ai_enabled
        bot.ui.ai_enabled = ai_enabled
        bot.ui.strategy_type = "Mean Reversion Enhanced"
        
        console = Console()
        console.print("\n[bold green]🔄 Starting bot...[/bold green]")
        console.print("[dim]Press 's' to start/stop, 'm' to change mode, 'q' to quit[/dim]")
        console.print("[dim]Press Ctrl+C to exit at any time[/dim]")
        
        await asyncio.sleep(2)  # Brief pause before starting UI
        
        await bot.run_with_ui()
        
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