#!/usr/bin/env python3
"""
Data Management Utilities
Handles data fetching, storage, and preprocessing for the trading bot
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
from binance.spot import Spot
import logging

class DataManager:
    def __init__(self, data_dir='data'):
        """Initialize data manager"""
        self.data_dir = data_dir
        self.logger = logging.getLogger('DataManager')
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
    def fetch_historical_klines(self, symbol='BTCUSDT', interval='5m', 
                               start_date=None, end_date=None, limit=1000):
        """
        Fetch historical kline data from Binance
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of klines to fetch
            
        Returns:
            pandas.DataFrame: OHLCV data
        """
        try:
            client = Spot()
            
            # Convert dates to timestamps if provided
            start_ts = None
            end_ts = None
            
            if start_date:
                start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            if end_date:
                end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
                
            # Fetch klines
            klines = client.klines(
                symbol=symbol,
                interval=interval,
                startTime=start_ts,
                endTime=end_ts,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Process data
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # Convert price columns to float
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in price_columns:
                df[col] = df[col].astype(float)
                
            # Keep only OHLCV data
            df = df[price_columns]
            
            self.logger.info(f"Fetched {len(df)} klines for {symbol} ({interval})")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching klines: {e}")
            return None
            
    def save_data(self, data, filename, format='csv'):
        """
        Save data to file
        
        Args:
            data: Data to save (DataFrame, dict, etc.)
            filename: Output filename
            format: File format ('csv', 'json', 'pickle')
        """
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            if format == 'csv' and isinstance(data, pd.DataFrame):
                data.to_csv(filepath)
            elif format == 'json':
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif format == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            self.logger.info(f"Data saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            
    def load_data(self, filename, format='csv'):
        """
        Load data from file
        
        Args:
            filename: Input filename
            format: File format ('csv', 'json', 'pickle')
            
        Returns:
            Loaded data
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            self.logger.warning(f"File not found: {filepath}")
            return None
            
        try:
            if format == 'csv':
                return pd.read_csv(filepath, index_col=0, parse_dates=True)
            elif format == 'json':
                with open(filepath, 'r') as f:
                    return json.load(f)
            elif format == 'pickle':
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return None
            
    def calculate_technical_indicators(self, data):
        """
        Calculate technical indicators for the data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional indicator columns
        """
        df = data.copy()
        
        try:
            # Simple Moving Averages
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price change indicators
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(5)
            df['volatility'] = df['price_change'].rolling(window=20).std()
            
            self.logger.info("Technical indicators calculated")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return data
            
    def resample_data(self, data, timeframe='1H'):
        """
        Resample data to different timeframe
        
        Args:
            data: DataFrame with OHLCV data
            timeframe: Target timeframe ('1H', '4H', '1D', etc.)
            
        Returns:
            Resampled DataFrame
        """
        try:
            resampled = data.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            self.logger.info(f"Data resampled to {timeframe}")
            return resampled
            
        except Exception as e:
            self.logger.error(f"Error resampling data: {e}")
            return data
            
    def clean_data(self, data):
        """
        Clean and validate data
        
        Args:
            data: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        df = data.copy()
        
        try:
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            # Sort by index
            df = df.sort_index()
            
            # Remove rows with missing OHLC data
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            # Validate OHLC relationships
            invalid_rows = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            )
            
            if invalid_rows.any():
                self.logger.warning(f"Removing {invalid_rows.sum()} invalid OHLC rows")
                df = df[~invalid_rows]
                
            # Remove extreme outliers (price changes > 50%)
            price_change = df['close'].pct_change().abs()
            outliers = price_change > 0.5
            
            if outliers.any():
                self.logger.warning(f"Removing {outliers.sum()} outlier rows")
                df = df[~outliers]
                
            self.logger.info(f"Data cleaned: {len(df)} rows remaining")
            return df
            
        except Exception as e:
            self.logger.error(f"Error cleaning data: {e}")
            return data
            
    def get_market_hours_data(self, data, market='crypto'):
        """
        Filter data for specific market hours
        
        Args:
            data: DataFrame with datetime index
            market: Market type ('crypto', 'forex', 'stock')
            
        Returns:
            Filtered DataFrame
        """
        if market == 'crypto':
            # Crypto markets are 24/7
            return data
        elif market == 'forex':
            # Forex: Monday 00:00 UTC to Friday 22:00 UTC
            mask = (
                (data.index.dayofweek < 5) |  # Monday to Friday
                ((data.index.dayofweek == 6) & (data.index.hour < 22))  # Friday until 22:00
            )
            return data[mask]
        elif market == 'stock':
            # Stock market: 9:30 AM to 4:00 PM EST, Monday to Friday
            # This is a simplified version - adjust for your specific market
            mask = (
                (data.index.dayofweek < 5) &  # Monday to Friday
                (data.index.hour >= 14) &     # 9:30 AM EST = 14:30 UTC
                (data.index.hour < 21)        # 4:00 PM EST = 21:00 UTC
            )
            return data[mask]
        else:
            return data
            
    def export_for_analysis(self, data, filename='analysis_data.csv'):
        """
        Export data with indicators for external analysis
        
        Args:
            data: DataFrame to export
            filename: Output filename
        """
        try:
            # Add technical indicators
            analysis_data = self.calculate_technical_indicators(data)
            
            # Add additional analysis columns
            analysis_data['hour'] = analysis_data.index.hour
            analysis_data['day_of_week'] = analysis_data.index.dayofweek
            analysis_data['month'] = analysis_data.index.month
            
            # Save to CSV
            self.save_data(analysis_data, filename, format='csv')
            
            self.logger.info(f"Analysis data exported to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error exporting analysis data: {e}")
            
def main():
    """Example usage of DataManager"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize data manager
    dm = DataManager()
    
    # Fetch recent BTC data
    print("Fetching BTC/USDT data...")
    data = dm.fetch_historical_klines(
        symbol='BTCUSDT',
        interval='5m',
        start_date='2024-01-01',
        end_date='2024-01-31',
        limit=1000
    )
    
    if data is not None:
        print(f"Fetched {len(data)} data points")
        print(data.head())
        
        # Clean data
        clean_data = dm.clean_data(data)
        
        # Add technical indicators
        data_with_indicators = dm.calculate_technical_indicators(clean_data)
        
        # Save data
        dm.save_data(data_with_indicators, 'btc_5m_data.csv')
        
        # Export for analysis
        dm.export_for_analysis(data_with_indicators, 'btc_analysis.csv')
        
        print("Data processing complete!")
    else:
        print("Failed to fetch data")
        
if __name__ == "__main__":
    main()