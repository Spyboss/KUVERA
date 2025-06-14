#!/usr/bin/env python3
"""
Live Trading Bot for Mean Reversion Strategy
Connects to Binance WebSocket for real-time data and execution
"""

import asyncio
import json
import logging
import os
import time
import yaml
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from binance.spot import Spot
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TradingBot:
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the trading bot"""
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        # Setup logging
        self.setup_logging()
        
        # Initialize Binance client
        self.setup_binance_client()
        
        # Trading state
        self.symbol = self.config['trading']['symbol']
        self.is_running = False
        self.position = None
        self.entry_price = None
        self.last_trade_time = None
        
        # Price data buffer (store last 100 candles for SMA calculation)
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
        
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs('logs', exist_ok=True)
        
        log_level = getattr(logging, self.config['monitoring']['log_level'])
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('TradingBot')
        
    def setup_binance_client(self):
        """Setup Binance API client with dynamic key selection"""
        # Determine if using testnet mode
        testnet_mode = getattr(self, 'testnet_mode', self.config['api']['testnet'])
        
        # Load appropriate API keys based on mode
        if testnet_mode:
            api_key = os.getenv('BINANCE_TESTNET_API_KEY') or os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_TESTNET_SECRET_KEY') or os.getenv('BINANCE_SECRET_KEY')
            base_url = "https://testnet.binance.vision"
            self.logger.info("Using Binance Testnet")
        else:
            api_key = os.getenv('BINANCE_LIVE_API_KEY')
            api_secret = os.getenv('BINANCE_LIVE_SECRET_KEY')
            base_url = "https://api.binance.com"
            self.logger.warning("Using Binance LIVE trading - real money at risk!")
        
        # Validate API keys
        if not api_key or not api_secret:
            mode_str = "testnet" if testnet_mode else "live"
            self.logger.error(f"API credentials not found for {mode_str} mode")
            self.logger.error(f"Please check your .env file for BINANCE_{mode_str.upper()}_API_KEY and BINANCE_{mode_str.upper()}_SECRET_KEY")
            raise ValueError(f"Missing {mode_str} API credentials")
            
        # First create client to get server time
        temp_client = Spot(
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url
        )
        
        # Get server time for synchronization
        try:
            server_time = temp_client.time()
            local_time = int(time.time() * 1000)
            time_offset = server_time['serverTime'] - local_time
            self.logger.info(f"Time offset with Binance server: {time_offset}ms")
            
            # Store the offset for manual timestamp adjustment
            self.time_offset = time_offset
            
        except Exception as e:
            self.logger.warning(f"Could not sync time with server: {e}")
            self.time_offset = 0
        
        # Create final client with larger recvWindow to handle timestamp differences
        self.client = Spot(
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url
        )
        
        # Test connection with extended recvWindow
        try:
            account_info = self.client.account(recvWindow=60000)
            self.logger.info("Successfully connected to Binance API")
            
            # Log account balance
            for balance in account_info['balances']:
                if float(balance['free']) > 0:
                    self.logger.info(f"Balance: {balance['asset']} = {balance['free']}")
                    
        except Exception as e:
            self.logger.error(f"Failed to connect to Binance API: {e}")
            raise
            
    def calculate_sma(self, period: int = None) -> Optional[float]:
        """Calculate Simple Moving Average"""
        if period is None:
            period = self.config['strategy']['sma_period']
            
        if len(self.price_buffer) < period:
            return None
            
        prices = list(self.price_buffer)[-period:]
        return sum(prices) / len(prices)
        
    def calculate_position_size(self, current_price: float) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account balance
            account_info = self.client.account()
            usdt_balance = 0.0
            
            for balance in account_info['balances']:
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free'])
                    break
                    
            # Calculate risk amount
            risk_per_trade = self.config['trading']['risk_per_trade']
            max_position_size = self.config['trading']['max_position_size']
            
            risk_amount = min(
                usdt_balance * risk_per_trade,
                max_position_size
            )
            
            # Calculate position size based on stop loss
            stop_loss_pct = self.config['strategy']['stop_loss']
            stop_loss_price = current_price * (1 - stop_loss_pct)
            risk_per_unit = current_price - stop_loss_price
            
            if risk_per_unit > 0:
                position_size = risk_amount / risk_per_unit
                # Ensure we don't exceed maximum position value
                max_units = max_position_size / current_price
                position_size = min(position_size, max_units)
            else:
                position_size = 0
                
            # Round to appropriate precision
            return round(position_size, 6)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
            
    def should_buy(self, current_price: float, sma: float) -> bool:
        """Check if buy conditions are met"""
        # Check if we already have a position
        if self.position:
            return False
            
        # Check cooldown period
        if self.last_trade_time:
            cooldown = self.config['risk']['cooldown_period']
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < cooldown:
                return False
                
        # Check daily limits
        max_daily_trades = self.config['risk']['max_daily_trades']
        max_daily_loss = self.config['risk']['max_daily_loss']
        
        if self.daily_trades >= max_daily_trades:
            return False
            
        if self.daily_loss >= max_daily_loss:
            return False
            
        # Check entry condition: price dips below SMA threshold
        entry_threshold = self.config['strategy']['entry_threshold']
        buy_threshold = sma * (1 - entry_threshold)
        
        return current_price <= buy_threshold
        
    def should_sell(self, current_price: float, sma: float) -> tuple[bool, str]:
        """Check if sell conditions are met"""
        if not self.position or not self.entry_price:
            return False, ""
            
        # Take profit condition
        exit_threshold = self.config['strategy']['exit_threshold']
        sell_threshold = sma * (1 + exit_threshold)
        
        if current_price >= sell_threshold:
            return True, "TAKE_PROFIT"
            
        # Stop loss condition
        stop_loss_pct = self.config['strategy']['stop_loss']
        stop_loss_price = self.entry_price * (1 - stop_loss_pct)
        
        if current_price <= stop_loss_price:
            return True, "STOP_LOSS"
            
        return False, ""
        
    async def place_buy_order(self, current_price: float):
        """Place a buy order"""
        try:
            position_size = self.calculate_position_size(current_price)
            
            if position_size <= 0:
                self.logger.warning("Position size too small, skipping buy order")
                return
                
            # Place market buy order
            order = self.client.new_order(
                symbol=self.symbol,
                side='BUY',
                type='MARKET',
                quantity=position_size
            )
            
            self.logger.info(f"BUY ORDER PLACED: {order}")
            
            # Update position state
            self.position = {
                'side': 'BUY',
                'quantity': position_size,
                'order_id': order['orderId']
            }
            
            self.entry_price = current_price
            self.daily_trades += 1
            
            # Send alert
            await self.send_alert(f"BUY ORDER: {position_size:.6f} {self.symbol} at ${current_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error placing buy order: {e}")
            
    async def place_sell_order(self, reason: str):
        """Place a sell order"""
        try:
            if not self.position:
                return
                
            quantity = self.position['quantity']
            
            # Place market sell order
            order = self.client.new_order(
                symbol=self.symbol,
                side='SELL',
                type='MARKET',
                quantity=quantity
            )
            
            self.logger.info(f"SELL ORDER PLACED ({reason}): {order}")
            
            # Calculate profit/loss
            current_price = self.current_price
            profit = (current_price - self.entry_price) * quantity
            self.total_profit += profit
            self.trade_count += 1
            
            if profit > 0:
                self.winning_trades += 1
            else:
                self.daily_loss += abs(profit)
                
            # Reset position state
            self.position = None
            self.entry_price = None
            self.last_trade_time = datetime.now()
            
            # Send alert
            await self.send_alert(
                f"SELL ORDER ({reason}): {quantity:.6f} {self.symbol} at ${current_price:.2f}\n"
                f"Profit: ${profit:.2f} | Total: ${self.total_profit:.2f}"
            )
            
            # Check profit target
            profit_threshold = self.config['monitoring']['profit_alert_threshold']
            if self.total_profit >= profit_threshold:
                await self.send_alert(f"🎉 PROFIT TARGET REACHED: ${self.total_profit:.2f}")
                
        except Exception as e:
            self.logger.error(f"Error placing sell order: {e}")
            
    async def send_alert(self, message: str):
        """Send alert notification"""
        self.logger.info(f"ALERT: {message}")
        
        # Here you could add email/SMS/Discord notifications
        # For now, just log the alert
        
    def handle_kline_data(self, client, message):
        """Handle incoming kline (candlestick) data"""
        try:
            # Extract kline data from message
            kline = message['k']
            close_price = float(kline['c'])
            timestamp = kline['t']
            
            # Only process closed candles
            if not kline['x']:  # x = is_closed
                return
                
            self.current_price = close_price
            
            # Add to price buffer
            self.price_buffer.append(close_price)
            
            # Calculate SMA
            sma = self.calculate_sma()
            if sma is None:
                self.logger.debug(f"Insufficient data for SMA calculation (need {self.config['strategy']['sma_period']} candles)")
                return
                
            self.logger.debug(f"Price: ${close_price:.2f}, SMA: ${sma:.2f}, Timestamp: {timestamp}")
            
            # Reset daily counters if new day
            current_date = datetime.now().date()
            if self.current_date != current_date:
                self.current_date = current_date
                self.daily_trades = 0
                self.daily_loss = 0.0
                self.logger.info(f"New trading day: {current_date}")
                
            # Check trading signals
            asyncio.create_task(self.process_signals(close_price, sma))
            
        except KeyError as e:
            self.logger.error(f"Missing key in kline data: {e}")
        except ValueError as e:
            self.logger.error(f"Invalid price data: {e}")
        except Exception as e:
            self.logger.error(f"Kline error: {e}")
            
    async def process_signals(self, current_price: float, sma: float):
        """Process trading signals"""
        try:
            # Check buy signal
            if self.should_buy(current_price, sma):
                self.logger.info(f"BUY SIGNAL: Price ${current_price:.2f} below SMA ${sma:.2f}")
                await self.place_buy_order(current_price)
                
            # Check sell signal
            should_sell, reason = self.should_sell(current_price, sma)
            if should_sell:
                self.logger.info(f"SELL SIGNAL ({reason}): Price ${current_price:.2f}")
                await self.place_sell_order(reason)
                
        except Exception as e:
            self.logger.error(f"Error processing signals: {e}")
            
    async def start_websocket(self):
        """Start WebSocket connection for real-time data with reconnection handling"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries and self.is_running:
            try:
                self.logger.info(f"Starting WebSocket connection (attempt {retry_count + 1}/{max_retries})...")
                
                # Create WebSocket client
                self.ws_client = SpotWebsocketStreamClient(
                    on_message=self.handle_kline_data,
                    is_combined=True
                )
                
                # Subscribe to kline stream (5-minute candles)
                stream_name = f"{self.symbol.lower()}@kline_5m"
                self.ws_client.kline(symbol=self.symbol, interval='5m')
                
                self.logger.info(f"Successfully subscribed to {stream_name}")
                return  # Success, exit retry loop
                
            except asyncio.CancelledError:
                self.logger.info("WebSocket connection cancelled")
                raise
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Error starting WebSocket (attempt {retry_count}): {e}")
                
                if retry_count < max_retries:
                    wait_time = min(2 ** retry_count, 30)  # Exponential backoff, max 30s
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error("Max WebSocket connection retries exceeded")
                    raise
            
    async def run(self):
        """Main bot execution loop"""
        self.logger.info("Starting Trading Bot...")
        self.is_running = True
        
        try:
            # Start WebSocket connection
            await self.start_websocket()
            
            # Keep the bot running
            while self.is_running:
                await asyncio.sleep(1)
                
                # Periodic status update
                if datetime.now().second == 0:  # Every minute
                    win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0
                    self.logger.info(
                        f"Status - Trades: {self.trade_count}, "
                        f"Win Rate: {win_rate:.1f}%, "
                        f"Profit: ${self.total_profit:.2f}, "
                        f"Position: {'YES' if self.position else 'NO'}"
                    )
                    
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
        except Exception as e:
            self.logger.error(f"Bot error: {e}")
        finally:
            await self.stop()
            
    async def stop(self):
        """Stop the trading bot"""
        self.logger.info("Stopping Trading Bot...")
        self.is_running = False
        
        if self.ws_client:
            self.ws_client.stop()
            
        # Final status report
        win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0
        
        self.logger.info("Final Statistics:")
        self.logger.info(f"Total Trades: {self.trade_count}")
        self.logger.info(f"Winning Trades: {self.winning_trades}")
        self.logger.info(f"Win Rate: {win_rate:.1f}%")
        self.logger.info(f"Total Profit: ${self.total_profit:.2f}")
        
def interactive_setup():
    """Interactive setup for bot configuration"""
    print("🚀 Kuvera Grid Trading Bot v1.0")
    print("================================")
    print("Welcome to the interactive setup!")
    print()
    
    # Trading mode selection
    print("📊 Trading Mode Selection:")
    print("1. Testnet (Recommended for beginners) - Default")
    print("2. Live Trading (Real money - Use with caution!)")
    mode_choice = input("Select mode (1/2) [Default: 1]: ").strip()
    testnet_mode = True if mode_choice != '2' else False
    
    if not testnet_mode:
        print("⚠️  WARNING: You selected LIVE TRADING mode!")
        print("⚠️  This will use real money. Are you absolutely sure?")
        confirm = input("Type 'CONFIRM' to proceed with live trading: ")
        if confirm != 'CONFIRM':
            print("Switching back to testnet mode for safety.")
            testnet_mode = True
    
    # AI features selection
    print("\n🤖 AI Features:")
    print("1. Enable AI (Sentiment analysis, strategy optimization) - Default")
    print("2. Disable AI (Basic mean reversion only)")
    ai_choice = input("Select AI mode (1/2) [Default: 1]: ").strip()
    ai_enabled = True if ai_choice != '2' else False
    
    # Display configuration
    print("\n📋 Configuration Summary:")
    print(f"Trading Mode: {'Testnet' if testnet_mode else 'LIVE'}")
    print(f"AI Features: {'Enabled' if ai_enabled else 'Disabled'}")
    print(f"Strategy: Mean Reversion (BTC/USDT, 5m candles)")
    print(f"Risk per trade: 1% of capital (max $0.30)")
    print(f"Target: $1-2 profit/week")
    print()
    
    return testnet_mode, ai_enabled

async def main():
    """Main execution function with interactive setup"""
    try:
        # Interactive setup
        testnet_mode, ai_enabled = interactive_setup()
        
        # Initialize bot with configuration
        bot = TradingBot()
        bot.testnet_mode = testnet_mode
        bot.ai_enabled = ai_enabled
        
        # Update config based on user choices
        bot.config['api']['testnet'] = testnet_mode
        if testnet_mode:
            bot.config['api']['base_url'] = "https://testnet.binance.vision"
        else:
            bot.config['api']['base_url'] = "https://api.binance.com"
        
        print("\n🔄 Starting bot...")
        print("Press Ctrl+C to stop the bot at any time.")
        print("\n" + "="*50)
        
        await bot.run()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Bot stopped by user.")
        print("Thank you for using Kuvera Grid Trading Bot!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your configuration and try again.")
        
if __name__ == "__main__":
    asyncio.run(main())