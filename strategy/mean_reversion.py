import backtrader as bt
import talib
import numpy as np
from datetime import datetime, timedelta

class MeanReversionStrategy(bt.Strategy):
    """
    Mean Reversion Strategy:
    - Buy when price dips 0.5% below 20-period SMA
    - Sell when price rises 0.5% above SMA
    - Stop-loss at 0.8% below entry
    - Max 1% of capital per trade
    """
    
    params = (
        ('sma_period', 20),
        ('entry_threshold', 0.005),  # 0.5%
        ('exit_threshold', 0.005),   # 0.5%
        ('stop_loss', 0.008),        # 0.8%
        ('risk_per_trade', 0.01),    # 1%
        ('max_position_size', 0.30), # $0.30 max per trade
        ('cooldown_period', 300),    # 5 minutes between trades
    )
    
    def __init__(self):
        # Initialize indicators
        self.sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.sma_period
        )
        
        # Track trade state
        self.order = None
        self.entry_price = None
        self.last_trade_time = None
        self.daily_trades = 0
        self.daily_loss = 0.0
        self.current_date = None
        
        # Performance tracking
        self.trade_count = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')
        
    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.log(f'BUY EXECUTED: Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size:.6f}, '
                        f'Cost: {order.executed.value:.2f}')
            else:
                profit = (order.executed.price - self.entry_price) * order.executed.size
                self.total_profit += profit
                self.trade_count += 1
                
                if profit > 0:
                    self.winning_trades += 1
                    
                self.log(f'SELL EXECUTED: Price: {order.executed.price:.2f}, '
                        f'Profit: {profit:.2f}, '
                        f'Total Profit: {self.total_profit:.2f}')
                        
                self.entry_price = None
                self.last_trade_time = self.data.datetime.datetime(0)
                self.daily_trades += 1
                
                if profit < 0:
                    self.daily_loss += abs(profit)
                    
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order {order.status}')
            
        self.order = None
        
    def next(self):
        """Main strategy logic"""
        # Reset daily counters
        current_date = self.data.datetime.date(0)
        if self.current_date != current_date:
            self.current_date = current_date
            self.daily_trades = 0
            self.daily_loss = 0.0
            
        # Skip if we don't have enough data for SMA
        if len(self.data) < self.params.sma_period:
            return
            
        # Check if we're in cooldown period
        if self.last_trade_time:
            time_since_last_trade = (self.data.datetime.datetime(0) - self.last_trade_time).total_seconds()
            if time_since_last_trade < self.params.cooldown_period:
                return
                
        # Risk management checks
        if self.daily_trades >= 5:  # Max 5 trades per day
            return
            
        if self.daily_loss >= 1.0:  # Max $1 daily loss
            return
            
        # Skip if we have a pending order
        if self.order:
            return
            
        current_price = self.data.close[0]
        sma_value = self.sma[0]
        
        # Calculate position size based on risk management
        account_value = self.broker.getvalue()
        risk_amount = min(
            account_value * self.params.risk_per_trade,
            self.params.max_position_size
        )
        
        # Calculate position size (accounting for stop loss)
        stop_loss_price = current_price * (1 - self.params.stop_loss)
        risk_per_share = current_price - stop_loss_price
        
        if risk_per_share > 0:
            position_size = risk_amount / risk_per_share
            # Ensure we don't exceed maximum position value
            max_shares = self.params.max_position_size / current_price
            position_size = min(position_size, max_shares)
        else:
            position_size = 0
            
        # Entry condition: Buy when price dips 0.5% below SMA
        if not self.position:
            buy_threshold = sma_value * (1 - self.params.entry_threshold)
            
            if current_price <= buy_threshold and position_size > 0:
                # Calculate the actual cost
                trade_cost = current_price * position_size
                
                if trade_cost <= account_value * 0.9:  # Keep 10% cash buffer
                    self.order = self.buy(size=position_size)
                    self.log(f'BUY SIGNAL: Price: {current_price:.2f}, '
                            f'SMA: {sma_value:.2f}, '
                            f'Size: {position_size:.6f}, '
                            f'Cost: {trade_cost:.2f}')
                            
        # Exit conditions
        elif self.position:
            # Take profit: Sell when price rises 0.5% above SMA
            sell_threshold = sma_value * (1 + self.params.exit_threshold)
            
            # Stop loss: Sell when price drops 0.8% below entry
            stop_loss_price = self.entry_price * (1 - self.params.stop_loss)
            
            if current_price >= sell_threshold:
                self.order = self.sell(size=self.position.size)
                self.log(f'TAKE PROFIT: Price: {current_price:.2f}, '
                        f'SMA: {sma_value:.2f}')
                        
            elif current_price <= stop_loss_price:
                self.order = self.sell(size=self.position.size)
                self.log(f'STOP LOSS: Price: {current_price:.2f}, '
                        f'Entry: {self.entry_price:.2f}')
                        
    def stop(self):
        """Called when backtest ends"""
        win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0
        
        self.log(f'Strategy Results:')
        self.log(f'Total Trades: {self.trade_count}')
        self.log(f'Winning Trades: {self.winning_trades}')
        self.log(f'Win Rate: {win_rate:.1f}%')
        self.log(f'Total Profit: ${self.total_profit:.2f}')
        self.log(f'Final Portfolio Value: ${self.broker.getvalue():.2f}')