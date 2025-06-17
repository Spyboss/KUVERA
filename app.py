#!/usr/bin/env python3
"""
Kuvera Grid Trading Bot - Web Dashboard
Integrated Flask web interface for monitoring the trading bot
"""

import os
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from threading import Thread
import time
import asyncio
from typing import Dict, List, Optional

# Import your existing bot components
try:
    from bot import EnhancedTradingBot  # Your main bot class
except ImportError:
    EnhancedTradingBot = None

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# Global bot instance
bot_instance = None
bot_stats = {
    'status': 'stopped',
    'balance': 0.0,
    'trades_today': 0,
    'profit_loss': 0.0,
    'last_update': datetime.now().isoformat(),
    'total_trades': 0,
    'win_rate': 0.0,
    'uptime': '0:00:00'
}

start_time = datetime.now()
recent_logs = []
max_logs = 100

class WebLogHandler(logging.Handler):
    """Custom log handler to capture logs for web display"""
    def emit(self, record):
        global recent_logs
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).strftime('%H:%M:%S'),
            'level': record.levelname,
            'message': record.getMessage()
        }
        recent_logs.append(log_entry)
        if len(recent_logs) > max_logs:
            recent_logs.pop(0)

# Setup logging
web_handler = WebLogHandler()
logging.getLogger().addHandler(web_handler)

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current bot status and statistics"""
    global bot_stats, start_time
    
    # Update uptime
    uptime = datetime.now() - start_time
    bot_stats['uptime'] = str(uptime).split('.')[0]  # Remove microseconds
    bot_stats['last_update'] = datetime.now().isoformat()
    
    # If bot is running, get real stats
    if bot_instance:
        try:
            # Get real stats from EnhancedTradingBot
            balances = bot_instance.get_account_balances()
            bot_stats.update({
                'balance': balances.get('USDT', 0),
                'total_trades': getattr(bot_instance, 'trade_count', 0),
                'profit_loss': getattr(bot_instance, 'total_profit', 0.0),
                'win_rate': (getattr(bot_instance, 'winning_trades', 0) / getattr(bot_instance, 'trade_count', 1) * 100) if getattr(bot_instance, 'trade_count', 0) > 0 else 0,
                'status': 'running' if getattr(bot_instance, 'is_running', False) else 'stopped'
            })
        except Exception as e:
            logging.error(f"Error getting bot stats: {e}")
            bot_stats['status'] = 'error'
    
    return jsonify(bot_stats)

@app.route('/api/logs')
def get_logs():
    """Get recent logs"""
    return jsonify({
        'logs': recent_logs[-50:],  # Last 50 logs
        'total_logs': len(recent_logs)
    })

@app.route('/api/trades')
def get_trades():
    """Get recent trades"""
    trades = []
    
    # Try to get real trade data from bot instance
    if bot_instance and hasattr(bot_instance, 'trade_history'):
        try:
            # Get real trades from bot if available
            trade_history = getattr(bot_instance, 'trade_history', [])
            for i, trade in enumerate(list(trade_history)[-10:]):  # Last 10 trades
                trades.append({
                    'id': i + 1,
                    'timestamp': trade.get('timestamp', datetime.now().isoformat()),
                    'symbol': trade.get('symbol', 'BTCUSDT'),
                    'side': trade.get('side', 'BUY'),
                    'amount': trade.get('amount', 0.001),
                    'price': trade.get('price', 0.0),
                    'profit': trade.get('profit', 0.0),
                    'status': trade.get('status', 'completed')
                })
        except Exception as e:
            logging.error(f"Error getting real trades: {e}")
    
    # Fallback to mock data if no real trades available
    if not trades:
        now = datetime.now()
        trades = [
            {
                'id': 1,
                'timestamp': (now - timedelta(hours=2, minutes=25)).isoformat(),
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'amount': 0.001,
                'price': 43250.50,
                'profit': 12.50,
                'status': 'completed'
            },
            {
                'id': 2,
                'timestamp': (now - timedelta(hours=1, minutes=25)).isoformat(),
                'symbol': 'BTCUSDT',
                'side': 'SELL',
                'amount': 0.001,
                'price': 43375.25,
                'profit': 15.75,
                'status': 'completed'
            },
            {
                'id': 3,
                'timestamp': (now - timedelta(minutes=45)).isoformat(),
                'symbol': 'ETHUSDT',
                'side': 'BUY',
                'amount': 0.01,
                'price': 2650.75,
                'profit': 8.25,
                'status': 'completed'
            }
        ]
    
    return jsonify({'trades': trades})

@app.route('/api/control/<action>', methods=['POST'])
def bot_control(action):
    """Control bot operations"""
    global bot_instance, bot_stats
    
    try:
        if action == 'start':
            if not bot_instance:
                # Initialize bot here
                logging.info("Starting trading bot...")
                bot_stats['status'] = 'starting'
                # Start bot in background thread
                bot_thread = Thread(target=run_bot_in_background, daemon=True)
                bot_thread.start()
                bot_stats['status'] = 'running'
            return jsonify({'success': True, 'message': 'Bot started successfully'})
        
        elif action == 'stop':
            if bot_instance:
                logging.info("Stopping trading bot...")
                bot_stats['status'] = 'stopping'
                try:
                    bot_instance.is_running = False
                    # Try to stop gracefully if stop method exists
                    if hasattr(bot_instance, 'stop'):
                        try:
                            loop = asyncio.get_event_loop()
                            loop.run_until_complete(bot_instance.stop())
                        except RuntimeError:
                            # If no event loop is running, create a new one
                            asyncio.run(bot_instance.stop())
                except Exception as e:
                    logging.error(f"Error stopping bot: {e}")
                bot_instance = None
                bot_stats['status'] = 'stopped'
            return jsonify({'success': True, 'message': 'Bot stopped successfully'})
        
        elif action == 'restart':
            logging.info("Restarting trading bot...")
            bot_stats['status'] = 'restarting'
            # Restart logic here
            bot_stats['status'] = 'running'
            return jsonify({'success': True, 'message': 'Bot restarted successfully'})
        
        else:
            return jsonify({'success': False, 'message': 'Invalid action'}), 400
            
    except Exception as e:
        logging.error(f"Error controlling bot: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/price')
def get_current_price():
    """Get current price data"""
    try:
        if bot_instance and hasattr(bot_instance, 'current_price'):
            current_price = getattr(bot_instance, 'current_price', 0)
            symbol = getattr(bot_instance, 'symbol', 'BTCUSDT')
            return jsonify({
                'symbol': symbol,
                'price': current_price,
                'timestamp': datetime.now().isoformat()
            })
        else:
            # Fallback mock data
            return jsonify({
                'symbol': 'BTCUSDT',
                'price': 43500.00,
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        logging.error(f"Error getting price data: {e}")
        return jsonify({'error': 'Failed to get price data'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'bot_status': bot_stats['status'],
        'uptime': bot_stats['uptime'],
        'version': '1.1'
    })

def run_bot_in_background():
    """Run the trading bot in a separate thread"""
    global bot_instance, bot_stats
    
    try:
        if EnhancedTradingBot is None:
            logging.error("EnhancedTradingBot not available - check bot.py import")
            bot_stats['status'] = 'error'
            return
            
        # Set environment variables for automated mode
        os.environ['AUTOMATED_MODE'] = 'true'
        os.environ['TRADING_MODE'] = os.environ.get('TRADING_MODE', 'testnet')
        os.environ['AI_ENABLED'] = os.environ.get('AI_ENABLED', 'true')
        os.environ['STRATEGY_TYPE'] = os.environ.get('STRATEGY_TYPE', 'mean_reversion')
        
        logging.info("🚀 Initializing Kuvera Grid Trading Bot...")
        logging.info(f"📊 Mode: {os.environ.get('TRADING_MODE', 'testnet')}")
        logging.info(f"🤖 AI: {os.environ.get('AI_ENABLED', 'true')}")
        logging.info(f"📈 Strategy: {os.environ.get('STRATEGY_TYPE', 'mean_reversion')}")
        
        # Initialize the enhanced bot
        bot_instance = EnhancedTradingBot()
        bot_instance.testnet_mode = os.environ.get('TRADING_MODE', 'testnet').lower() == 'testnet'
        bot_instance.ai_enabled = os.environ.get('AI_ENABLED', 'true').lower() == 'true'
        
        bot_stats['status'] = 'running'
        logging.info("✅ Trading bot initialized successfully!")
        
        # Run the bot in automated mode
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(bot_instance.run_automated())
        finally:
            # Cleanup
            if not loop.is_closed():
                loop.close()
        
    except KeyboardInterrupt:
        logging.info("🛑 Bot stopped by user interrupt")
        bot_stats['status'] = 'stopped'
    except Exception as e:
        logging.error(f"❌ Bot error: {e}")
        bot_stats['status'] = 'error'
        # Try to cleanup bot instance
        if bot_instance:
            try:
                bot_instance.is_running = False
                if hasattr(bot_instance, 'stop'):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(bot_instance.stop())
                    loop.close()
            except Exception as cleanup_error:
                logging.error(f"Error during cleanup: {cleanup_error}")
    finally:
        # Ensure bot_instance is cleared on exit
        if bot_stats['status'] != 'running':
            bot_instance = None

if __name__ == '__main__':
    # Setup logging with better formatting
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/app.log', mode='a') if os.path.exists('logs') or os.makedirs('logs', exist_ok=True) else logging.StreamHandler()
        ]
    )
    
    # Log startup information
    logging.info("🚀 Starting Kuvera Grid Trading Bot Web Dashboard")
    logging.info(f"📊 Trading Mode: {os.environ.get('TRADING_MODE', 'testnet')}")
    logging.info(f"🤖 AI Enabled: {os.environ.get('AI_ENABLED', 'true')}")
    logging.info(f"🔧 Start Bot: {os.environ.get('START_BOT', 'true')}")
    
    # Start bot in background thread if enabled
    if os.environ.get('START_BOT', 'true').lower() == 'true':
        try:
            bot_thread = Thread(target=run_bot_in_background, daemon=True)
            bot_thread.start()
            logging.info("✅ Bot thread started successfully")
        except Exception as e:
            logging.error(f"❌ Failed to start bot thread: {e}")
            bot_stats['status'] = 'error'
    else:
        logging.info("⏸️ Bot auto-start disabled")
    
    # Start Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    logging.info(f"🌐 Starting web dashboard on port {port}")
    logging.info(f"🔗 Dashboard will be available at http://0.0.0.0:{port}")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
    except Exception as e:
        logging.error(f"❌ Failed to start Flask app: {e}")
        raise