#!/usr/bin/env python3
"""
Working Trading Bot - Shows actual trading logs without balance requirements
"""

import asyncio
import json
import logging
import os
import sys
import time
import threading
from datetime import datetime
from typing import Dict, Optional, List

import pandas as pd
import numpy as np
from binance.spot import Spot
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WorkingTradingBot:
    def __init__(self):
        print("Step 1: Setting up logging...")
        sys.stdout.flush()
        self.setup_logging()
        
        print("Step 2: Setting up config...")
        sys.stdout.flush()
        self.setup_config()
        
        print("Step 3: Setting up Binance client...")
        sys.stdout.flush()
        self.setup_binance_client()
        
        print("Step 4: Initializing trading state...")
        sys.stdout.flush()
        
        # Trading state
        self.current_price = None
        self.price_history = []
        self.trading_active = True
        self.position = None
        self.entry_price = None
        self.last_signal_time = 0
        
        # Statistics
        self.trade_count = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.balance = 1000.0  # Simulated balance
        
        # WebSocket
        self.ws_client = None
        
        # Flask app compatibility
        self.is_running = False
        self.trade_history = []
        self.testnet_mode = True
        self.ai_enabled = False
        
        print("[SUCCESS] Working Trading Bot initialized successfully!")
        sys.stdout.flush()
        self.logger.info("[SUCCESS] Working Trading Bot initialized successfully!")
        
    def setup_logging(self):
        """Setup comprehensive logging with immediate output"""
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Setup main logger
        self.logger = logging.getLogger('WorkingTradingBot')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler('logs/working_trading_bot.log', mode='w')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler with immediate flush
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Force immediate output
        self.logger.propagate = False
        
    def setup_config(self):
        """Setup trading configuration"""
        self.symbol = 'BTCUSDT'
        self.trade_amount = 0.001  # 0.001 BTC
        self.profit_target = 0.003  # 0.3% (lower for more frequent trades)
        self.stop_loss = 0.008  # 0.8%
        self.min_signal_interval = 30  # Minimum 30 seconds between signals
        
    def setup_binance_client(self):
        """Setup Binance client with testnet"""
        try:
            api_key = os.getenv('BINANCE_TESTNET_API_KEY')
            api_secret = os.getenv('BINANCE_TESTNET_SECRET_KEY')
            
            if not api_key or not api_secret:
                print(f"[ERROR] Missing API credentials in .env file")
                raise ValueError("Missing API credentials")
                
            self.client = Spot(
                api_key=api_key,
                api_secret=api_secret,
                base_url='https://testnet.binance.vision'
            )
            
            # Test connection
            server_time = self.client.time()
            print(f"[OK] Connected to Binance Testnet - Server time: {server_time['serverTime']}")
            self.logger.info(f"[OK] Connected to Binance Testnet - Server time: {server_time['serverTime']}")
            
        except Exception as e:
            print(f"[ERROR] Failed to setup Binance client: {e}")
            self.logger.error(f"[ERROR] Failed to setup Binance client: {e}")
            raise
            
    def get_current_price(self) -> Optional[float]:
        """Get current price for the symbol"""
        try:
            ticker = self.client.ticker_price(self.symbol)
            price = float(ticker['price'])
            return price
        except Exception as e:
            self.logger.error(f"[ERROR] Error getting current price: {e}")
            return None
            
    def calculate_indicators(self) -> Dict[str, float]:
        """Calculate simple trading indicators"""
        if len(self.price_history) < 20:
            return {}
            
        prices = np.array(self.price_history[-50:])  # Use more data
        
        # Simple Moving Averages
        sma_5 = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
        sma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        
        # Price changes
        price_change_1 = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
        price_change_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) > 5 else 0
        
        # Simple RSI calculation
        rsi = self.calculate_rsi(prices) if len(prices) >= 14 else 50
        
        indicators = {
            'sma_5': sma_5,
            'sma_10': sma_10,
            'sma_20': sma_20,
            'price_change_1': price_change_1,
            'price_change_5': price_change_5,
            'rsi': rsi,
            'current_price': prices[-1]
        }
        
        return indicators
        
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def should_buy(self, indicators: Dict[str, float]) -> bool:
        """Determine if we should buy - More aggressive signals"""
        if not indicators or self.position:
            return False
            
        current_time = time.time()
        if current_time - self.last_signal_time < self.min_signal_interval:
            return False
            
        current_price = indicators.get('current_price', 0)
        sma_5 = indicators.get('sma_5', 0)
        sma_10 = indicators.get('sma_10', 0)
        sma_20 = indicators.get('sma_20', 0)
        price_change_1 = indicators.get('price_change_1', 0)
        price_change_5 = indicators.get('price_change_5', 0)
        rsi = indicators.get('rsi', 50)
        
        # Multiple buy conditions (more aggressive)
        conditions = [
            current_price < sma_20 * 0.999,  # Price below SMA20
            sma_5 > sma_10,  # Short term trend up
            price_change_1 > 0.0005,  # Recent positive momentum
            rsi < 70,  # Not overbought
            price_change_5 < 0.01  # Not in strong uptrend already
        ]
        
        buy_score = sum(conditions)
        should_buy = buy_score >= 3  # Need at least 3 conditions
        
        if should_buy:
            self.logger.info(f"🟢 BUY SIGNAL DETECTED! Score: {buy_score}/5")
            self.logger.info(f"   💰 Price: ${current_price:.2f} | SMA5: ${sma_5:.2f} | SMA10: ${sma_10:.2f} | SMA20: ${sma_20:.2f}")
            self.logger.info(f"   📊 RSI: {rsi:.1f} | 1m Change: {price_change_1:.4f} | 5m Change: {price_change_5:.4f}")
            self.last_signal_time = current_time
        
        return should_buy
        
    def should_sell(self, indicators: Dict[str, float]) -> bool:
        """Determine if we should sell"""
        if not indicators or not self.position:
            return False
            
        current_price = indicators.get('current_price', 0)
        sma_5 = indicators.get('sma_5', 0)
        sma_10 = indicators.get('sma_10', 0)
        price_change_1 = indicators.get('price_change_1', 0)
        rsi = indicators.get('rsi', 50)
        
        if not self.entry_price:
            return False
            
        # Calculate profit/loss
        profit_pct = (current_price - self.entry_price) / self.entry_price
        
        # Sell conditions
        take_profit = profit_pct >= self.profit_target
        stop_loss = profit_pct <= -self.stop_loss
        trend_reversal = (sma_5 < sma_10 and price_change_1 < -0.001)  # Trend turning down
        overbought = rsi > 80  # Very overbought
        
        should_sell = take_profit or stop_loss or trend_reversal or overbought
        
        if should_sell:
            reason = ""
            if take_profit:
                reason = f"TAKE PROFIT ({profit_pct:.3%})"
            elif stop_loss:
                reason = f"STOP LOSS ({profit_pct:.3%})"
            elif trend_reversal:
                reason = "TREND REVERSAL"
            elif overbought:
                reason = f"OVERBOUGHT (RSI: {rsi:.1f})"
                
            self.logger.info(f"🔴 SELL SIGNAL: {reason} at ${current_price:.2f}")
            
        return should_sell
        
    async def execute_buy(self, price: float) -> bool:
        """Execute buy order (simulated but realistic)"""
        try:
            cost = self.trade_amount * price
            
            if self.balance < cost:
                self.logger.warning(f"⚠️ Insufficient balance: ${self.balance:.2f} < ${cost:.2f}")
                return False
                
            self.logger.info(f"🟢 EXECUTING BUY ORDER")
            self.logger.info(f"   📊 Amount: {self.trade_amount} BTC")
            self.logger.info(f"   💰 Price: ${price:.2f}")
            self.logger.info(f"   💵 Cost: ${cost:.2f}")
            
            # Simulate order processing
            await asyncio.sleep(0.2)
            
            # Update state
            self.position = 'LONG'
            self.entry_price = price
            self.balance -= cost
            
            # Add to trade history for Flask app
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'side': 'BUY',
                'amount': self.trade_amount,
                'price': price,
                'profit': 0.0,
                'status': 'completed'
            }
            self.trade_history.append(trade_record)
            
            self.logger.info(f"✅ BUY ORDER FILLED!")
            self.logger.info(f"   🎯 Position: LONG {self.trade_amount} BTC at ${price:.2f}")
            self.logger.info(f"   💰 Remaining balance: ${self.balance:.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Buy order failed: {e}")
            return False
            
    async def execute_sell(self, price: float) -> bool:
        """Execute sell order (simulated but realistic)"""
        try:
            revenue = self.trade_amount * price
            profit = revenue - (self.trade_amount * self.entry_price)
            profit_pct = (price - self.entry_price) / self.entry_price
            
            self.logger.info(f"🔴 EXECUTING SELL ORDER")
            self.logger.info(f"   📊 Amount: {self.trade_amount} BTC")
            self.logger.info(f"   💰 Price: ${price:.2f}")
            self.logger.info(f"   💵 Revenue: ${revenue:.2f}")
            
            # Simulate order processing
            await asyncio.sleep(0.2)
            
            # Update state
            self.position = None
            self.entry_price = None
            self.balance += revenue
            self.trade_count += 1
            self.total_profit += profit
            
            if profit > 0:
                self.winning_trades += 1
            
            # Add to trade history for Flask app
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'side': 'SELL',
                'amount': self.trade_amount,
                'price': price,
                'profit': profit,
                'status': 'completed'
            }
            self.trade_history.append(trade_record)
            
            # Keep only last 50 trades
            if len(self.trade_history) > 50:
                self.trade_history = self.trade_history[-50:]
                
            win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0
            
            self.logger.info(f"✅ SELL ORDER FILLED!")
            self.logger.info(f"   💰 Profit: ${profit:.4f} ({profit_pct:.3%})")
            self.logger.info(f"   💵 New balance: ${self.balance:.2f}")
            self.logger.info(f"   📊 Stats: {self.trade_count} trades, {win_rate:.1f}% win rate, ${self.total_profit:.4f} total profit")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Sell order failed: {e}")
            return False
            
    def handle_websocket_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            if isinstance(message, str):
                data = json.loads(message)
            else:
                data = message
            
            # Handle subscription confirmation
            if 'result' in data and 'id' in data:
                return
            
            # Handle kline data
            if 'stream' in data and 'data' in data:
                kline_data = data['data']
                if 'k' in kline_data:
                    kline = kline_data['k']
                    
                    # Only process closed candles
                    if kline['x']:  # x = true means kline is closed
                        close_price = float(kline['c'])
                        self.current_price = close_price
                        self.price_history.append(close_price)
                        
                        # Keep only last 100 prices
                        if len(self.price_history) > 100:
                            self.price_history.pop(0)
                        
                        # Calculate indicators and check for trading signals
                        indicators = self.calculate_indicators()
                        
                        # Process trading signals asynchronously
                        asyncio.create_task(self.process_trading_signals(close_price, indicators))
            elif 'k' in data:
                kline = data['k']
                if kline['x']:  # Only process closed klines
                    price = float(kline['c'])
                    volume = float(kline['v'])
                    
                    self.current_price = price
                    self.price_history.append(price)
                    
                    # Keep only last 100 prices
                    if len(self.price_history) > 100:
                        self.price_history = self.price_history[-100:]
                        
                    # Log price updates
                    if len(self.price_history) % 5 == 0:  # Every 5th update
                        self.logger.debug(f"[PRICE] Price update: ${price:.2f} (Volume: {volume:.2f})")
                    
                    # Process trading logic - schedule it properly
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            loop.create_task(self.process_trading_signals())
                        else:
                            asyncio.run_coroutine_threadsafe(self.process_trading_signals(), loop)
                    except RuntimeError:
                        # No event loop running, skip this update
                        pass
                    
        except Exception as e:
            self.logger.error(f"[ERROR] WebSocket message error: {e}")
    
    def handle_websocket_error(self, ws, error):
        """Handle WebSocket errors with reconnection logic"""
        self.logger.error(f"[ERROR] WebSocket error: {error}")
        print(f"[ERROR] WebSocket error: {error}")
        
        # Don't attempt reconnection if bot is stopping
        if not self.is_running:
            return
            
        # Schedule reconnection after a delay
        threading.Timer(5.0, self.reconnect_websocket).start()
    
    def handle_websocket_close(self, ws):
        """Handle WebSocket connection close"""
        self.logger.warning("[WS] WebSocket connection closed")
        print("[WS] WebSocket connection closed")
        
        # Don't attempt reconnection if bot is stopping
        if not self.is_running:
            return
            
        # Schedule reconnection after a delay
        threading.Timer(3.0, self.reconnect_websocket).start()
    
    def reconnect_websocket(self):
        """Attempt to reconnect WebSocket"""
        if not self.is_running:
            return
            
        self.logger.info("[WS] Attempting to reconnect WebSocket...")
        print("[WS] Attempting to reconnect WebSocket...")
        
        try:
            self.start_websocket()
        except Exception as e:
            self.logger.error(f"[ERROR] WebSocket reconnection failed: {e}")
            print(f"[ERROR] WebSocket reconnection failed: {e}")
            # Try again after longer delay
            if self.is_running:
                threading.Timer(10.0, self.reconnect_websocket).start()
            
    async def process_trading_signals(self):
        """Process trading signals based on current data"""
        try:
            if not self.trading_active:
                return
                
            indicators = self.calculate_indicators()
            if not indicators:
                return
                
            current_price = indicators.get('current_price')
            if not current_price:
                return
                
            # Check for buy signals
            if self.should_buy(indicators):
                await self.execute_buy(current_price)
                
            # Check for sell signals
            elif self.should_sell(indicators):
                await self.execute_sell(current_price)
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error processing trading signals: {e}")
            
    def start_websocket(self):
        """Start WebSocket connection for real-time data with error handling"""
        try:
            self.logger.info(f"[WS] Starting WebSocket for {self.symbol}...")
            print(f"[WS] Starting WebSocket for {self.symbol}...")
            
            # Stop existing connection if any
            if hasattr(self, 'ws_client') and self.ws_client:
                try:
                    self.ws_client.stop()
                except:
                    pass
            
            self.ws_client = SpotWebsocketStreamClient(
                on_message=self.handle_websocket_message,
                on_error=self.handle_websocket_error,
                on_close=self.handle_websocket_close
            )
            
            # Subscribe to kline stream
            self.ws_client.kline(symbol=self.symbol.lower(), interval='1m')
            self.logger.info(f"[OK] WebSocket connected for {self.symbol}@kline_1m")
            print(f"[OK] WebSocket connected for {self.symbol}@kline_1m")
            
        except Exception as e:
            self.logger.error(f"[ERROR] WebSocket connection failed: {e}")
            print(f"[ERROR] WebSocket connection failed: {e}")
            # Continue without WebSocket - use polling instead
            self.ws_client = None
            
    def get_account_balances(self):
        """Get account balances - Flask app compatibility"""
        return {'USDT': self.balance}
    
    async def stop(self):
        """Stop the bot - Flask app compatibility"""
        self.is_running = False
        self.trading_active = False
        if self.ws_client:
            self.ws_client.stop()
        self.logger.info("🛑 Bot stopped")
    
    async def run_automated(self):
        """Run in automated mode for Flask app"""
        self.is_running = True
        await self.run()
        
    async def run_polling_mode(self):
        """Run the bot in polling mode when WebSocket is unavailable"""
        self.logger.info("[POLL] Starting polling mode...")
        print("[POLL] Starting polling mode...")
        
        poll_interval = 60  # seconds
        
        while self.is_running:
            try:
                # Get current price
                current_price = self.get_current_price()
                if current_price:
                    self.current_price = current_price
                    self.price_history.append(current_price)
                    
                    # Keep only last 100 prices
                    if len(self.price_history) > 100:
                        self.price_history.pop(0)
                    
                    # Calculate indicators and check for trading signals
                    indicators = self.calculate_indicators()
                    
                    # Process trading signals
                    await self.process_trading_signals(current_price, indicators)
                    
                    self.logger.info(f"[POLL] Current price: ${current_price:.2f}")
                    
            except Exception as e:
                self.logger.error(f"[ERROR] Polling error: {e}")
                
            # Wait for next poll
            await asyncio.sleep(poll_interval)
    
    async def run(self):
        """Main bot execution loop"""
        try:
            print("[START] Starting Working Trading Bot...")
            self.logger.info("[START] Starting Working Trading Bot...")
            
            self.is_running = True
            
            # Get initial price
            initial_price = self.get_current_price()
            if initial_price:
                self.current_price = initial_price
                self.price_history.append(initial_price)
                print(f"[PRICE] Initial {self.symbol} price: ${initial_price:.2f}")
                self.logger.info(f"[PRICE] Initial {self.symbol} price: ${initial_price:.2f}")
            
            # Start WebSocket (with fallback to polling)
            self.start_websocket()
            
            # Log trading parameters
            print(f"[CONFIG] Trading parameters:")
            print(f"   Symbol: {self.symbol}")
            print(f"   Trade amount: {self.trade_amount} BTC")
            print(f"   Profit target: {self.profit_target:.1%}")
            print(f"   Stop loss: {self.stop_loss:.1%}")
            
            # If WebSocket failed, use polling mode
            if not self.ws_client:
                print("[INFO] WebSocket unavailable, using polling mode")
                self.logger.info("WebSocket unavailable, using polling mode")
                await self.run_polling_mode()
            print(f"   Starting balance: ${self.balance:.2f}")
            
            self.logger.info(f"[CONFIG] Trading parameters:")
            self.logger.info(f"   Symbol: {self.symbol}")
            self.logger.info(f"   Trade amount: {self.trade_amount} BTC")
            self.logger.info(f"   Profit target: {self.profit_target:.1%}")
            self.logger.info(f"   Stop loss: {self.stop_loss:.1%}")
            self.logger.info(f"   Starting balance: ${self.balance:.2f}")
            
            print("[READY] Bot is now running and monitoring for trading opportunities...")
            self.logger.info("[READY] Bot is now running and monitoring for trading opportunities...")
            
            # Keep the bot running
            status_counter = 0
            while self.is_running:
                await asyncio.sleep(10)
                status_counter += 1
                
                # Log status every 60 seconds (6 cycles)
                if status_counter >= 6:
                    status_counter = 0
                    
                    if self.current_price:
                        status = f"📊 Status: Price=${self.current_price:.2f}, Balance=${self.balance:.2f}"
                        if self.position:
                            profit_pct = (self.current_price - self.entry_price) / self.entry_price if self.entry_price else 0
                            status += f", Position: {self.position} at ${self.entry_price:.2f} ({profit_pct:.3%})"
                        
                        print(status)
                        self.logger.info(status)
                    
        except KeyboardInterrupt:
            print("🛑 Bot stopped by user")
            self.logger.info("🛑 Bot stopped by user")
            self.is_running = False
        except Exception as e:
            print(f"❌ Bot error: {e}")
            self.logger.error(f"❌ Bot error: {e}")
            self.is_running = False
        finally:
            self.is_running = False
            if self.ws_client:
                self.ws_client.stop()
                
if __name__ == "__main__":
    print("Starting Working Trading Bot...")
    sys.stdout.flush()
    
    try:
        bot = WorkingTradingBot()
        print("Bot initialized, starting main loop...")
        sys.stdout.flush()
        asyncio.run(bot.run())
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        input("Press Enter to exit...")