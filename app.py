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
    from bot import TradingBot  # Your main bot class
except ImportError:
    TradingBot = None

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
    if bot_instance and hasattr(bot_instance, 'get_stats'):
        try:
            real_stats = bot_instance.get_stats()
            bot_stats.update(real_stats)
        except Exception as e:
            logging.error(f"Error getting bot stats: {e}")
    
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
    # This would connect to your bot's trade history
    # For now, return mock data with proper timestamps
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
                # You would start your bot here
                # bot_instance = TradingBot()
                # bot_instance.start()
                bot_stats['status'] = 'running'
            return jsonify({'success': True, 'message': 'Bot started successfully'})
        
        elif action == 'stop':
            if bot_instance:
                logging.info("Stopping trading bot...")
                bot_stats['status'] = 'stopping'
                # bot_instance.stop()
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

@app.route('/health')
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'bot_status': bot_stats['status'],
        'uptime': bot_stats['uptime']
    })

def run_bot_in_background():
    """Run the trading bot in a separate thread"""
    global bot_instance, bot_stats
    
    try:
        # Initialize and run your bot here
        logging.info("Initializing trading bot...")
        logging.info("Loading configuration files...")
        logging.info("Connecting to exchange API...")
        logging.info("Setting up grid trading strategy...")
        
        # This is where you'd start your actual bot
        # bot_instance = TradingBot()
        # bot_instance.run()
        
        # For now, just simulate bot running
        bot_stats['status'] = 'running'
        logging.info("Trading bot started successfully!")
        
        # Simulate some activity logs
        activity_counter = 0
        
        while True:
            if bot_stats['status'] == 'running':
                # Simulate bot activity
                bot_stats['last_update'] = datetime.now().isoformat()
                
                # Generate some sample log activity every few cycles
                activity_counter += 1
                if activity_counter % 4 == 0:
                    logging.info("Monitoring market conditions...")
                elif activity_counter % 6 == 0:
                    logging.info("Grid levels updated based on price movement")
                elif activity_counter % 8 == 0:
                    logging.warning("High volatility detected - adjusting strategy")
                elif activity_counter % 10 == 0:
                    logging.info("Portfolio rebalancing completed")
                
                # Update other stats as needed
            
            time.sleep(30)  # Update every 30 seconds
            
    except Exception as e:
        logging.error(f"Bot error: {e}")
        bot_stats['status'] = 'error'

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Start bot in background thread
    if os.environ.get('START_BOT', 'true').lower() == 'true':
        bot_thread = Thread(target=run_bot_in_background, daemon=True)
        bot_thread.start()
        logging.info("Bot thread started")
    
    # Start Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    logging.info(f"Starting web dashboard on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)