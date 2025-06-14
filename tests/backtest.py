#!/usr/bin/env python3
"""
Backtesting Script for Mean Reversion Strategy
Validates strategy performance on 2024 BTC/USDT data
"""

import backtrader as bt
import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from binance.spot import Spot
from strategy.mean_reversion import MeanReversionStrategy

class BacktestRunner:
    def __init__(self, config_path='config/config.yaml'):
        """Initialize backtest runner with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.results = {}
        
    def fetch_historical_data(self, symbol='BTCUSDT', interval='5m', start_date='2024-01-01', end_date='2024-12-31'):
        """
        Fetch historical data from Binance
        Note: This uses public API, no authentication required
        """
        print(f"Fetching historical data for {symbol}...")
        
        try:
            # Initialize Binance client (public API)
            client = Spot()
            
            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            
            # Fetch klines data
            klines = client.klines(symbol=symbol, interval=interval, startTime=start_ts, endTime=end_ts, limit=1000)
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # Convert price columns to float
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in price_columns:
                df[col] = df[col].astype(float)
                
            # Keep only OHLCV data
            df = df[price_columns]
            
            print(f"Fetched {len(df)} data points from {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            print("Using sample data for demonstration...")
            return self.generate_sample_data(start_date, end_date)
            
    def generate_sample_data(self, start_date, end_date, initial_price=45000):
        """
        Generate sample BTC/USDT data for testing when API is unavailable
        """
        print("Generating sample data...")
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate 5-minute intervals
        date_range = pd.date_range(start=start, end=end, freq='5T')
        
        # Generate realistic price movements
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0, 0.002, len(date_range))  # 0.2% volatility
        
        # Add some trend and mean reversion
        trend = np.sin(np.arange(len(date_range)) * 2 * np.pi / (24 * 12 * 7)) * 0.001  # Weekly cycle
        returns += trend
        
        # Calculate prices
        prices = [initial_price]
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
            
        # Create OHLCV data
        df = pd.DataFrame(index=date_range)
        df['close'] = prices
        
        # Generate OHLC from close prices
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        
        # Add some intrabar volatility
        volatility = df['close'] * 0.001  # 0.1% intrabar volatility
        df['high'] = df[['open', 'close']].max(axis=1) + volatility * np.random.uniform(0, 1, len(df))
        df['low'] = df[['open', 'close']].min(axis=1) - volatility * np.random.uniform(0, 1, len(df))
        
        # Generate volume
        df['volume'] = np.random.uniform(100, 1000, len(df))
        
        return df
        
    def run_backtest(self, data):
        """
        Run backtest with the mean reversion strategy
        """
        print("Running backtest...")
        
        # Initialize Cerebro
        cerebro = bt.Cerebro()
        
        # Add strategy with parameters from config
        strategy_params = self.config['strategy']
        cerebro.addstrategy(
            MeanReversionStrategy,
            sma_period=strategy_params['sma_period'],
            entry_threshold=strategy_params['entry_threshold'],
            exit_threshold=strategy_params['exit_threshold'],
            stop_loss=strategy_params['stop_loss'],
            risk_per_trade=self.config['trading']['risk_per_trade'],
            max_position_size=self.config['trading']['max_position_size']
        )
        
        # Convert DataFrame to Backtrader data feed
        data_feed = bt.feeds.PandasData(
            dataname=data,
            datetime=None,  # Use index
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=None
        )
        
        cerebro.adddata(data_feed)
        
        # Set initial capital and commission
        initial_cash = self.config['backtest']['initial_cash']
        commission = self.config['backtest']['commission']
        
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=commission)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        print(f'Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}')
        
        # Run backtest
        results = cerebro.run()
        strategy_instance = results[0]
        
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - initial_cash) / initial_cash * 100
        
        print(f'Final Portfolio Value: ${final_value:.2f}')
        print(f'Total Return: {total_return:.2f}%')
        
        # Extract analyzer results
        trade_analyzer = strategy_instance.analyzers.trades.get_analysis()
        sharpe_ratio = strategy_instance.analyzers.sharpe.get_analysis()
        drawdown = strategy_instance.analyzers.drawdown.get_analysis()
        returns = strategy_instance.analyzers.returns.get_analysis()
        
        # Store results
        self.results = {
            'initial_cash': initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'trade_analysis': trade_analyzer,
            'sharpe_ratio': sharpe_ratio.get('sharperatio', 0) if sharpe_ratio else 0,
            'max_drawdown': drawdown.get('max', {}).get('drawdown', 0) if drawdown else 0,
            'returns': returns
        }
        
        # Plot results
        self.plot_results(cerebro, data)
        
        return self.results
        
    def plot_results(self, cerebro, data):
        """
        Plot backtest results
        """
        try:
            # Create plots directory if it doesn't exist
            os.makedirs('backtests', exist_ok=True)
            
            # Plot with Backtrader
            fig = cerebro.plot(style='candlestick', barup='green', bardown='red')[0][0]
            plt.savefig('backtests/backtest_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create custom performance plot
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Price and SMA
            plt.subplot(3, 1, 1)
            plt.plot(data.index, data['close'], label='BTC/USDT Price', alpha=0.7)
            
            # Calculate and plot SMA
            sma_period = self.config['strategy']['sma_period']
            sma = data['close'].rolling(window=sma_period).mean()
            plt.plot(data.index, sma, label=f'SMA({sma_period})', color='orange')
            
            plt.title('BTC/USDT Price and Moving Average')
            plt.ylabel('Price (USDT)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Volume
            plt.subplot(3, 1, 2)
            plt.plot(data.index, data['volume'], label='Volume', alpha=0.7, color='purple')
            plt.title('Trading Volume')
            plt.ylabel('Volume')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Returns distribution
            plt.subplot(3, 1, 3)
            returns = data['close'].pct_change().dropna()
            plt.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
            plt.title('Returns Distribution')
            plt.xlabel('Returns')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('backtests/performance_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Plots saved to backtests/ directory")
            
        except Exception as e:
            print(f"Error creating plots: {e}")
            
    def print_detailed_results(self):
        """
        Print detailed backtest results
        """
        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        
        print(f"Initial Capital: ${self.results['initial_cash']:.2f}")
        print(f"Final Value: ${self.results['final_value']:.2f}")
        print(f"Total Return: {self.results['total_return']:.2f}%")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {self.results['max_drawdown']:.2f}%")
        
        # Trade analysis
        trades = self.results['trade_analysis']
        if trades:
            total_trades = trades.get('total', {}).get('total', 0)
            won_trades = trades.get('won', {}).get('total', 0)
            lost_trades = trades.get('lost', {}).get('total', 0)
            
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
            print(f"\nTrade Statistics:")
            print(f"Total Trades: {total_trades}")
            print(f"Winning Trades: {won_trades}")
            print(f"Losing Trades: {lost_trades}")
            print(f"Win Rate: {win_rate:.1f}%")
            
            if 'won' in trades and 'pnl' in trades['won']:
                avg_win = trades['won']['pnl']['average']
                print(f"Average Win: ${avg_win:.2f}")
                
            if 'lost' in trades and 'pnl' in trades['lost']:
                avg_loss = trades['lost']['pnl']['average']
                print(f"Average Loss: ${avg_loss:.2f}")
                
        print("\n" + "="*60)
        
        # Risk assessment
        print("\nRISK ASSESSMENT:")
        print("- Trading fees (0.1% per trade) are included")
        print("- Slippage not modeled - expect 0.01-0.05% impact in live trading")
        print("- Market gaps and low liquidity periods not simulated")
        print("- Backtest uses 5-minute data - tick-level execution may vary")
        
        # Recommendations
        if self.results['total_return'] > 0 and win_rate >= 60:
            print("\n✅ Strategy shows promise for live testing")
        elif win_rate < 50:
            print("\n⚠️  Low win rate - consider adjusting parameters")
        else:
            print("\n⚠️  Mixed results - proceed with caution")
            
def main():
    """Main execution function"""
    print("Starting Mean Reversion Strategy Backtest")
    print("="*50)
    
    # Initialize backtest runner
    runner = BacktestRunner()
    
    # Fetch historical data
    config = runner.config['backtest']
    data = runner.fetch_historical_data(
        symbol=runner.config['trading']['symbol'],
        start_date=config['start_date'],
        end_date=config['end_date']
    )
    
    # Run backtest
    results = runner.run_backtest(data)
    
    # Print results
    runner.print_detailed_results()
    
    # Save results to file
    os.makedirs('backtests', exist_ok=True)
    
    with open('backtests/results.txt', 'w') as f:
        f.write(f"Backtest Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Initial Capital: ${results['initial_cash']:.2f}\n")
        f.write(f"Final Value: ${results['final_value']:.2f}\n")
        f.write(f"Total Return: {results['total_return']:.2f}%\n")
        f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}\n")
        f.write(f"Max Drawdown: {results['max_drawdown']:.2f}%\n")
        
    print(f"\nResults saved to backtests/results.txt")
    
if __name__ == "__main__":
    main()