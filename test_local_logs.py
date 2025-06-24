#!/usr/bin/env python3
"""
Test script to verify log generation locally
"""

import logging
import time
from datetime import datetime

# Setup logging similar to the app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('test_bot')

def test_log_generation():
    """Generate test logs to verify the logging system"""
    print("Starting log generation test...")
    
    # Generate various types of logs
    logger.info("🚀 Starting Enhanced Trading Bot test...")
    logger.info("📊 Loading configuration...")
    logger.info("🤖 AI analysis enabled")
    logger.info("📈 Strategy: Mean Reversion Grid")
    logger.info("🌐 Connecting to Binance API...")
    
    time.sleep(1)
    
    logger.info("📊 Market analysis: BTC/USDT price: $67,234.56")
    logger.info("🤖 AI sentiment analysis: Bullish (confidence: 0.78)")
    logger.info("📈 RSI indicator: 45.2 (oversold signal)")
    logger.info("🟢 BUY signal generated for BTC/USDT")
    logger.info("💰 Position opened: BUY 0.001 BTC at $67,234.56")
    
    time.sleep(1)
    
    logger.warning("⚠️ High volatility detected")
    logger.info("📊 Grid level adjustment: +2.5%")
    logger.info("🔴 SELL signal generated for partial position")
    logger.info("💰 Position closed: SELL 0.0005 BTC at $67,890.12")
    logger.info("💵 Profit: +$0.33 (+0.97%)")
    
    time.sleep(1)
    
    logger.error("❌ API rate limit warning")
    logger.info("⏳ Waiting 60 seconds before next request...")
    logger.info("✅ Trading bot operational - monitoring market...")
    
    print("Log generation test completed!")

if __name__ == '__main__':
    test_log_generation()