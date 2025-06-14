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
    import talib
except ImportError:
    xgb = None
    StandardScaler = None
    talib = None

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
        
    def create_header(self) -> Panel:
        """Create header panel"""
        title = Text("Kuvera Grid v1.1 🚀", style="bold cyan")
        subtitle = Text(f"Mode: {self.current_mode} | AI: {'Enabled' if self.ai_enabled else 'Disabled'} | Strategy: {self.strategy_type}", style="dim")
        return Panel(Text.assemble(title, "\n", subtitle), box=box.ROUNDED)
        
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
        
    def create_status_panel(self, websocket_status: str, api_status: str, target_progress: float) -> Panel:
        """Create status panel"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Status", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("WebSocket:", websocket_status)
        table.add_row("API:", api_status)
        
        # Progress bar for weekly target
        progress_bar = "█" * int(target_progress * 20) + "░" * (20 - int(target_progress * 20))
        table.add_row("Target:", f"[{progress_bar}] {target_progress:.0%}")
        
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
        """Update the display with current data"""
        self.layout["header"].update(self.create_header())
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
            bot_data.get('target_progress', 0)
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
        
        # AI/ML components
        self.ml_model = None
        self.scaler = None
        self.setup_ml_components()
        
        # Weekly target tracking
        self.weekly_target = 2.0  # $2 per week
        self.week_start_profit = self.total_profit
        
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs('logs', exist_ok=True)
        
        log_level = getattr(logging, self.config['monitoring']['log_level'])
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/bot_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('KuveraBot')
        
    def setup_binance_client(self):
        """Setup Binance API client"""
        testnet_mode = getattr(self, 'testnet_mode', self.config['api']['testnet'])
        
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
        """Setup machine learning components"""
        if xgb and StandardScaler:
            self.ml_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            self.scaler = StandardScaler()
            self.logger.info("ML components initialized")
        else:
            self.logger.warning("ML libraries not available, using basic strategy")
            
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
        
    def get_account_balances(self) -> Dict[str, float]:
        """Get account balances"""
        try:
            account_info = self.client.account(recvWindow=60000)
            balances = {}
            
            for balance in account_info['balances']:
                free_balance = float(balance['free'])
                if free_balance > 0:
                    balances[balance['asset']] = free_balance
                    
            return balances
        except Exception as e:
            self.logger.error(f"Error getting balances: {e}")
            return {}
            
    def get_bot_data(self) -> dict:
        """Get current bot data for UI display"""
        win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0
        
        # Calculate weekly progress
        weekly_profit = self.total_profit - self.week_start_profit
        target_progress = min(weekly_profit / self.weekly_target, 1.0)
        
        return {
            'trades': self.trade_count,
            'win_rate': win_rate,
            'profit': self.total_profit,
            'position': 'BTC/USDT' if self.position else None,
            'risk': self.config['trading']['risk_per_trade'] * 100,
            'balances': self.get_account_balances(),
            'websocket_status': 'Connected (btcusdt@kline_5m)' if self.ws_client else 'Disconnected',
            'api_status': 'OK',
            'target_progress': target_progress
        }
        
    def should_buy_enhanced(self, current_price: float, indicators: Dict[str, float]) -> bool:
        """Enhanced buy logic with multiple indicators"""
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
            
        # Require at least 2 conditions
        return sum(conditions) >= 2
        
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
        """Process enhanced trading signals"""
        try:
            # Check buy signal
            if self.should_buy_enhanced(current_price, indicators):
                self.logger.info(f"BUY SIGNAL: Price ${current_price:.2f}")
                await self.place_buy_order(current_price)
                
            # Check sell signal
            should_sell, reason = self.should_sell_enhanced(current_price, indicators)
            if should_sell:
                self.logger.info(f"SELL SIGNAL ({reason}): Price ${current_price:.2f}")
                await self.place_sell_order(reason)
                
        except Exception as e:
            self.logger.error(f"Error processing signals: {e}")
            
    def should_sell_enhanced(self, current_price: float, indicators: Dict[str, float]) -> Tuple[bool, str]:
        """Enhanced sell logic"""
        if not self.position or not self.entry_price:
            return False, ""
            
        sma = indicators.get('sma')
        rsi = indicators.get('rsi')
        bb_upper = indicators.get('bb_upper')
        
        # Take profit condition
        if sma:
            exit_threshold = self.config['strategy']['exit_threshold']
            sell_threshold = sma * (1 + exit_threshold)
            
            if current_price >= sell_threshold:
                return True, "TAKE_PROFIT"
                
        # RSI overbought
        if rsi and rsi > 70:
            return True, "RSI_OVERBOUGHT"
            
        # Price near Bollinger Band upper
        if bb_upper and current_price >= bb_upper * 0.99:
            return True, "BB_UPPER"
            
        # Stop loss condition
        stop_loss_pct = self.config['strategy']['stop_loss']
        stop_loss_price = self.entry_price * (1 - stop_loss_pct)
        
        if current_price <= stop_loss_price:
            return True, "STOP_LOSS"
            
        return False, ""
        
    async def place_buy_order(self, current_price: float):
        """Place a buy order with gamification"""
        try:
            position_size = self.calculate_position_size(current_price)
            
            if position_size <= 0:
                return
                
            order = self.client.new_order(
                symbol=self.symbol,
                side='BUY',
                type='MARKET',
                quantity=position_size
            )
            
            self.position = {
                'side': 'BUY',
                'quantity': position_size,
                'order_id': order['orderId']
            }
            
            self.entry_price = current_price
            self.daily_trades += 1
            
            self.logger.info(f"BUY ORDER PLACED: {position_size:.6f} {self.symbol} at ${current_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error placing buy order: {e}")
            
    async def place_sell_order(self, reason: str):
        """Place a sell order with gamification"""
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
            
            current_price = self.current_price
            profit = (current_price - self.entry_price) * quantity
            self.total_profit += profit
            self.trade_count += 1
            
            # Gamification
            if profit > 0:
                self.winning_trades += 1
                self.gamification.add_win()
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
                
            # Check for badges
            self.gamification.check_trade_milestones(self.trade_count, self.total_profit)
            
            # Reset position
            self.position = None
            self.entry_price = None
            self.last_trade_time = datetime.now()
            
            self.logger.info(f"SELL ORDER ({reason}): Profit ${profit:.2f} | Total: ${self.total_profit:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error placing sell order: {e}")
            
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
        """Run bot with modern UI"""
        self.logger.info("Starting Kuvera Grid v1.1 🚀")
        self.is_running = True
        
        # Start WebSocket
        await self.start_websocket()
        
        # Setup keyboard listener
        def on_key_press(key):
            if key.name == 's':
                self.is_running = not self.is_running
            elif key.name == 'm':
                # Toggle mode (for demo purposes)
                pass
            elif key.name == 'q':
                self.is_running = False
                
        keyboard.on_press(on_key_press)
        
        # Main UI loop
        with Live(self.ui.layout, refresh_per_second=1, screen=True) as live:
            while self.is_running:
                try:
                    bot_data = self.get_bot_data()
                    self.ui.update_display(bot_data, self.gamification)
                    await asyncio.sleep(1)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.logger.error(f"UI error: {e}")
                    
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