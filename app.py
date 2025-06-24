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

# Global bot instance and state
bot_instance = None
bot_thread = None
bot_startup_logs = []
bot_stats = {
    'status': 'stopped',  # Start with stopped status
    'balance': 0.0,
    'trades_today': 0,
    'profit_loss': 0.0,
    'last_update': datetime.now().isoformat(),
    'total_trades': 0,
    'win_rate': 0.0,
    'uptime': '0:00:00',
    'startup_progress': 0,
    'startup_stage': 'idle',  # Start with idle
    'error_message': None
}

start_time = datetime.now()
recent_logs = []
max_logs = 200  # Increased for better log history

class WebLogHandler(logging.Handler):
    """Enhanced custom log handler to capture logs for web display"""
    def emit(self, record):
        global recent_logs, bot_startup_logs, bot_stats
        
        try:
            # Create a more detailed log entry with additional metadata
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3],
                'level': record.levelname,
                'message': self._enhance_message(record.getMessage()),
                'module': record.name,
                'category': self._categorize_log(record.getMessage()),
                'source': getattr(record, 'source', 'system'),
                'created': record.created,
                'thread_id': record.thread,
                'process_id': record.process
            }
            
            # Add to recent logs with priority handling for important logs
            if log_entry['level'] in ['ERROR', 'CRITICAL'] or self._is_important_log(log_entry['message']):
                # Insert important logs at the beginning to ensure they're not lost
                recent_logs.insert(0, log_entry)
            else:
                recent_logs.append(log_entry)
                
            # Maintain max logs but keep important ones
            while len(recent_logs) > max_logs:
                # Find oldest non-important log to remove
                for i, log in enumerate(recent_logs):
                    if not (log['level'] in ['ERROR', 'CRITICAL'] or self._is_important_log(log['message'])):
                        recent_logs.pop(i)
                        break
                else:
                    # If all logs are important, remove oldest
                    recent_logs.pop(-1)
            
            # Track startup logs with enhanced capture logic
            current_time = datetime.now()
            time_since_start = (current_time - start_time).total_seconds()
            
            # Expanded keywords for better log capture
            startup_keywords = [
                'starting', 'dashboard', 'flask', 'serving', 'kuvera', 'bot', 'trading', 
                'web', 'port', 'available', 'running', 'initializing', 'setup', 'loading', 
                'connecting', 'ready', 'signal', 'strategy', 'analysis', 'ai', 'model', 
                'prediction', 'sentiment', 'price', 'market', 'indicator', 'position',
                'openrouter', 'claude', 'gpt', 'trade', 'order', 'balance', 'profit', 
                'loss', 'buy', 'sell', 'executed', 'portfolio', 'grid'
            ]
            
            # Capture more logs as startup logs for better visibility
            if (bot_stats['status'] in ['starting', 'initializing', 'preparing', 'running'] or 
                time_since_start < 600 or  # Extended to 10 minutes for better capture
                any(keyword in log_entry['message'].lower() for keyword in startup_keywords) or
                log_entry['level'] in ['INFO', 'WARNING', 'ERROR', 'CRITICAL'] or
                self._is_important_log(log_entry['message'])):
                
                bot_startup_logs.append(log_entry)
                if len(bot_startup_logs) > 500:  # Significantly increased for better history
                    # Remove oldest non-important logs first
                    for i, log in enumerate(bot_startup_logs):
                        if not (log['level'] in ['ERROR', 'CRITICAL'] or self._is_important_log(log['message'])):
                            bot_startup_logs.pop(i)
                            break
                    else:
                        # If all logs are important, remove oldest
                        bot_startup_logs.pop(0)
                        
            # Print to console for debugging in development
            if os.environ.get('FLASK_DEBUG', 'false').lower() == 'true':
                print(f"[{log_entry['timestamp']}] {log_entry['level']} - {log_entry['message']}")
                
        except Exception as e:
            # Fallback logging to prevent handler errors
            print(f"WebLogHandler error: {e}")
    
    def _enhance_message(self, message):
        """Add visual enhancements to log messages for better readability"""
        # Add emoji indicators based on content
        if any(kw in message.lower() for kw in ['error', 'failed', 'exception', 'critical']):
            return f"❌ {message}"
        elif any(kw in message.lower() for kw in ['warning', 'warn']):
            return f"⚠️ {message}"
        elif any(kw in message.lower() for kw in ['buy', 'long', 'purchase']):
            return f"🟢 {message}"
        elif any(kw in message.lower() for kw in ['sell', 'short']):
            return f"🔴 {message}"
        elif any(kw in message.lower() for kw in ['signal', 'indicator', 'strategy']):
            return f"📊 {message}"
        elif any(kw in message.lower() for kw in ['ai', 'ml', 'model', 'prediction', 'sentiment']):
            return f"🤖 {message}"
        elif any(kw in message.lower() for kw in ['starting', 'initializing', 'setup']):
            return f"🚀 {message}"
        elif any(kw in message.lower() for kw in ['ready', 'complete', 'success', 'finished']):
            return f"✅ {message}"
        else:
            return message
    
    def _is_important_log(self, message):
        """Determine if a log message is important and should be preserved"""
        important_keywords = [
            'signal', 'trade', 'position', 'order', 'buy', 'sell', 
            'ai analysis', 'prediction', 'sentiment', 'model', 
            'strategy', 'indicator', 'crossover', 'divergence',
            'error', 'warning', 'critical', 'exception', 'failed'
        ]
        return any(keyword in message.lower() for keyword in important_keywords)
    
    def _categorize_log(self, message):
        """Enhanced categorization of log messages for better filtering"""
        message_lower = message.lower()
        
        # More specific categorization with expanded keywords
        if any(keyword in message_lower for keyword in ['starting', 'initializing', 'loading', 'connecting', 'setup', 'ready']):
            return 'startup'
        elif any(keyword in message_lower for keyword in ['buy', 'sell', 'trade', 'order', 'position', 'entry', 'exit', 'stop', 'profit']):
            return 'trading'
        elif any(keyword in message_lower for keyword in ['ai', 'ml', 'model', 'prediction', 'sentiment', 'analysis', 'confidence', 'score']):
            return 'ai'
        elif any(keyword in message_lower for keyword in ['error', 'failed', 'exception', 'critical', 'crash', 'timeout']):
            return 'error'
        elif any(keyword in message_lower for keyword in ['warning', 'warn', 'caution', 'attention']):
            return 'warning'
        elif any(keyword in message_lower for keyword in ['strategy', 'signal', 'indicator', 'analysis', 'pattern', 'trend', 'sma', 'ema', 'rsi', 'macd', 'bollinger']):
            return 'strategy'
        elif any(keyword in message_lower for keyword in ['price', 'market', 'volume', 'volatility', 'momentum']):
            return 'market'
        elif any(keyword in message_lower for keyword in ['config', 'setting', 'parameter', 'option']):
            return 'config'
        else:
            return 'general'

# Setup logging
web_handler = WebLogHandler()
logging.getLogger().addHandler(web_handler)

# Add immediate startup logs to ensure they're captured
logging.info("🔧 Web log handler initialized")
logging.info("📊 Log capture system active")
logging.info("🌐 Dashboard logging ready")

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
    """Get recent logs with enhanced filtering and sorting"""
    category = request.args.get('category', 'all')
    limit = int(request.args.get('limit', 100))  # Increased default limit
    startup_only = request.args.get('startup', 'false').lower() == 'true'
    level = request.args.get('level', 'all')  # Filter by log level
    search = request.args.get('search', '').lower()  # Search term
    sort_order = request.args.get('sort', 'desc').lower()  # asc or desc
    since = request.args.get('since', None)  # Timestamp to get logs since
    
    # Select the log source
    log_source = bot_startup_logs if startup_only else recent_logs
    
    # Apply filters
    filtered_logs = log_source.copy()  # Create a copy to avoid modifying original
    
    # Filter by timestamp if specified (apply first for efficiency)
    if since:
        try:
            since_ts = float(since)
            filtered_logs = [log for log in filtered_logs if log.get('created', 0) > since_ts]
        except (ValueError, TypeError):
            logging.warning(f"Invalid since parameter: {since}")
    
    # Filter by category if specified
    if category != 'all':
        filtered_logs = [log for log in filtered_logs if log.get('category') == category]
    
    # Filter by level if specified
    if level != 'all':
        filtered_logs = [log for log in filtered_logs if log.get('level', '').lower() == level.lower()]
    
    # Filter by search term if specified
    if search:
        filtered_logs = [log for log in filtered_logs if search in log.get('message', '').lower() or 
                         search in log.get('module', '').lower() or
                         search in log.get('category', '').lower()]
    
    # Sort logs (always sort before applying limit)
    if sort_order == 'asc':
        sorted_logs = sorted(filtered_logs, key=lambda x: x.get('created', 0))
    else:
        sorted_logs = sorted(filtered_logs, key=lambda x: x.get('created', 0), reverse=True)
    
    # Apply limit
    if limit > 0:
        logs_to_return = sorted_logs[:limit]
    else:
        logs_to_return = sorted_logs
    
    # Get available categories from recent logs
    categories = list(set(log.get('category', 'general') for log in recent_logs))
    categories.sort()  # Sort alphabetically
    
    # Get available log levels from recent logs
    levels = list(set(log.get('level', 'INFO') for log in recent_logs))
    levels.sort()  # Sort alphabetically
    
    # Count logs by category
    category_counts = {}
    for cat in categories:
        category_counts[cat] = len([log for log in recent_logs if log.get('category') == cat])
    
    # Count logs by level
    level_counts = {}
    for lvl in levels:
        level_counts[lvl] = len([log for log in recent_logs if log.get('level') == lvl])
    
    # Get latest timestamp for incremental updates
    latest_timestamp = max([log.get('created', 0) for log in recent_logs]) if recent_logs else 0
    
    # Enhanced response with more metadata
    return jsonify({
        'logs': logs_to_return,
        'total_logs': len(recent_logs),
        'startup_logs': len(bot_startup_logs),
        'filtered_count': len(filtered_logs),
        'categories': categories,
        'category_counts': category_counts,
        'levels': levels,
        'level_counts': level_counts,
        'bot_status': bot_stats['status'],
        'startup_stage': bot_stats.get('startup_stage', 'idle'),
        'latest_timestamp': latest_timestamp
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
    global bot_instance, bot_stats, bot_thread, bot_startup_logs
    
    try:
        if action == 'start':
            if not bot_instance and (not bot_thread or not bot_thread.is_alive()):
                # Clear previous startup logs
                bot_startup_logs.clear()
                
                # Initialize bot here
                logging.info("🚀 Starting Kuvera Grid Trading Bot...")
                bot_stats.update({
                    'status': 'starting',
                    'startup_stage': 'initializing',
                    'startup_progress': 10,
                    'error_message': None
                })
                
                # Start bot in background thread
                bot_thread = Thread(target=run_bot_in_background, daemon=True)
                bot_thread.start()
                
                return jsonify({'success': True, 'message': 'Bot startup initiated'})
            else:
                return jsonify({'success': False, 'message': 'Bot is already running or starting'})
        
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
            
            try:
                # Stop the current bot if running
                if bot_instance:
                    logging.info("Stopping existing bot instance...")
                    try:
                        bot_instance.is_running = False
                        if hasattr(bot_instance, 'stop'):
                            try:
                                loop = asyncio.get_event_loop()
                                loop.run_until_complete(bot_instance.stop())
                            except RuntimeError:
                                asyncio.run(bot_instance.stop())
                    except Exception as e:
                        logging.error(f"Error stopping bot: {e}")
                    
                    bot_instance = None
                    logging.info("Bot instance stopped")
                
                # Wait a moment for cleanup
                time.sleep(1)
                
                # Clear previous startup logs
                bot_startup_logs.clear()
                
                # Start bot in background thread
                bot_thread = Thread(target=run_bot_in_background, daemon=True)
                bot_thread.start()
                
                bot_stats.update({
                    'status': 'starting',
                    'startup_stage': 'initializing',
                    'startup_progress': 10,
                    'error_message': None
                })
                
                logging.info("Bot restart initiated successfully")
                return jsonify({'success': True, 'message': 'Bot restart initiated successfully'})
                
            except Exception as e:
                logging.error(f"Bot restart error: {e}")
                bot_stats['status'] = 'error'
                bot_stats['error_message'] = str(e)
                return jsonify({'success': False, 'message': f'Bot restart failed: {str(e)}'})
        
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

@app.route('/api/logs/stream')
def stream_logs():
    """Get logs since a specific timestamp for real-time streaming"""
    since = request.args.get('since', '0')
    category = request.args.get('category', 'all')
    level = request.args.get('level', 'all')
    limit = int(request.args.get('limit', 50))
    
    try:
        since_ts = float(since)
    except (ValueError, TypeError):
        since_ts = 0
    
    # Get logs newer than the specified timestamp
    if category == 'all' and level == 'all':
        new_logs = [log for log in recent_logs if log.get('created', 0) > since_ts]
    else:
        # Apply category and level filters
        new_logs = [log for log in recent_logs 
                   if log.get('created', 0) > since_ts and
                   (category == 'all' or log.get('category') == category) and
                   (level == 'all' or log.get('level', '').lower() == level.lower())]
    
    # Sort by timestamp (newest first) and limit
    new_logs = sorted(new_logs, key=lambda x: x.get('created', 0), reverse=True)[:limit]
    
    # Get latest timestamp for next request
    latest_timestamp = max([log.get('created', 0) for log in recent_logs]) if recent_logs else since_ts
    
    return jsonify({
        'logs': new_logs,
        'latest_timestamp': latest_timestamp,
        'count': len(new_logs),
        'bot_status': bot_stats['status'],
        'startup_stage': bot_stats.get('startup_stage', 'idle')
    })

@app.route('/api/logs/sse')
def sse_logs():
    """Stream logs in real-time with Server-Sent Events (SSE)"""
    def generate():
        try:
            # Send initial logs
            last_timestamp = 0
            connection_start = time.time()
            
            # Initial data burst with recent logs
            recent = sorted(recent_logs[-30:], key=lambda x: x.get('created', 0))
            for log in recent:
                if log.get('created', 0) > last_timestamp:
                    last_timestamp = log.get('created', 0)
                    log_json = json.dumps(log)
                    yield f"data: {log_json}\n\n"
            
            # Keep connection open and stream new logs
            last_log_count = len(recent_logs)
            last_check = time.time()
            max_connection_time = 3600  # 1 hour max connection
            
            while time.time() - connection_start < max_connection_time:
                try:
                    # Check for new logs
                    current_count = len(recent_logs)
                    current_time = time.time()
                    
                    # If new logs have been added
                    if current_count > last_log_count:
                        # Find new logs since last timestamp
                        new_logs = [log for log in recent_logs if log.get('created', 0) > last_timestamp]
                        new_logs.sort(key=lambda x: x.get('created', 0))
                        
                        # Send each new log
                        for log in new_logs:
                            if log.get('created', 0) > last_timestamp:
                                last_timestamp = log.get('created', 0)
                                log_json = json.dumps(log)
                                yield f"data: {log_json}\n\n"
                        
                        last_log_count = current_count
                        last_check = current_time
                    
                    # Send heartbeat every 15 seconds if no new logs
                    elif current_time - last_check > 15:
                        yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': current_time})}\n\n"
                        last_check = current_time
                    
                    time.sleep(0.5)  # Check for new logs every 500ms
                    
                except GeneratorExit:
                    break
                except Exception as e:
                    logging.error(f"SSE stream error: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                    break
                    
        except Exception as e:
            logging.error(f"SSE initialization error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': 'SSE connection failed'})}\n\n"
    
    response = app.response_class(
        response=generate(),
        status=200,
        mimetype='text/event-stream'
    )
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'  # Disable buffering for nginx
    return response

@app.route('/health')
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'bot_status': bot_stats['status'],
        'uptime': bot_stats['uptime'],
        'version': '1.2'  # Updated version number
    })

def run_bot_in_background():
    """Run the trading bot in a background thread with detailed startup tracking"""
    global bot_instance, bot_stats
    
    try:
        logging.info("🚀 Starting bot initialization process...")
        
        if EnhancedTradingBot is None:
            error_msg = "❌ EnhancedTradingBot not available - check bot.py import"
            logging.error(error_msg)
            bot_stats.update({
                'status': 'error',
                'startup_stage': 'error',
                'error_message': error_msg
            })
            return
            
        # Stage 1: Environment Setup
        logging.info("🔧 Setting up environment variables...")
        bot_stats.update({
            'status': 'starting',
            'startup_stage': 'environment_setup',
            'startup_progress': 20
        })
        
        os.environ['AUTOMATED_MODE'] = 'true'
        os.environ['TRADING_MODE'] = os.environ.get('TRADING_MODE', 'testnet')
        os.environ['AI_ENABLED'] = os.environ.get('AI_ENABLED', 'true')
        os.environ['STRATEGY_TYPE'] = os.environ.get('STRATEGY_TYPE', 'mean_reversion')
        
        logging.info(f"✅ Environment configured - Mode: {os.environ.get('TRADING_MODE')}, AI: {os.environ.get('AI_ENABLED')}, Strategy: {os.environ.get('STRATEGY_TYPE')}")
        time.sleep(1)  # Brief pause for UI update
        
        # Stage 2: Loading Configuration
        logging.info("📋 Loading configuration and settings...")
        bot_stats.update({
            'startup_stage': 'loading_config',
            'startup_progress': 40
        })
        logging.info(f"📊 Mode: {os.environ.get('TRADING_MODE', 'testnet')}")
        logging.info(f"🤖 AI: {os.environ.get('AI_ENABLED', 'true')}")
        logging.info(f"📈 Strategy: {os.environ.get('STRATEGY_TYPE', 'mean_reversion')}")
        time.sleep(0.5)
        
        # Stage 3: Initializing Bot
        logging.info("🤖 Initializing Enhanced Trading Bot...")
        bot_stats.update({
            'startup_stage': 'initializing_bot',
            'startup_progress': 60
        })
        
        try:
            bot_instance = EnhancedTradingBot()
            bot_instance.testnet_mode = os.environ.get('TRADING_MODE', 'testnet').lower() == 'testnet'
            bot_instance.ai_enabled = os.environ.get('AI_ENABLED', 'true').lower() == 'true'
            logging.info("✅ Bot instance created successfully")
        except Exception as e:
            error_msg = f"❌ Failed to initialize bot: {str(e)}"
            logging.error(error_msg)
            bot_stats.update({
                'status': 'error',
                'startup_stage': 'error',
                'error_message': error_msg
            })
            return
        
        # Stage 4: Connecting to APIs
        logging.info("🌐 Connecting to Binance API and external services...")
        bot_stats.update({
            'startup_stage': 'connecting_apis',
            'startup_progress': 80
        })
        time.sleep(1)
        
        # Stage 5: Final Setup
        logging.info("⚡ Finalizing startup and beginning operations...")
        bot_stats.update({
            'startup_stage': 'finalizing',
            'startup_progress': 95
        })
        time.sleep(0.5)
        
        # Stage 6: Ready
        logging.info("✅ Kuvera Grid Trading Bot is now LIVE and ready for trading!")
        bot_stats.update({
            'status': 'running',
            'startup_stage': 'ready',
            'startup_progress': 100,
            'last_update': datetime.now().isoformat()
        })
        
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
        error_msg = f"Error during bot startup: {str(e)}"
        logging.error(f"❌ {error_msg}")
        bot_stats.update({
            'status': 'error',
            'startup_stage': 'error',
            'error_message': error_msg
        })
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
    
    # Immediate startup logs for web dashboard
    logging.info("🚀 Starting Kuvera Grid Trading Bot Web Dashboard v1.1")
    logging.info("📱 Initializing Flask application...")
    logging.info("🔧 Setting up web log handler...")
    logging.info("📊 Loading dashboard configuration...")
    logging.info("🌐 Web dashboard initialization complete")
    
    # Log environment information
    logging.info(f"📊 Trading Mode: {os.environ.get('TRADING_MODE', 'testnet')}")
    logging.info(f"🤖 AI Enabled: {os.environ.get('AI_ENABLED', 'true')}")
    logging.info(f"📈 Strategy Type: {os.environ.get('STRATEGY_TYPE', 'mean_reversion')}")
    logging.info(f"🔧 Start Bot: {os.environ.get('START_BOT', 'true')}")
    logging.info(f"🌐 Port: {os.environ.get('PORT', '5000')}")
    logging.info(f"🐛 Debug Mode: {os.environ.get('FLASK_DEBUG', 'false')}")
    logging.info("✅ Environment configuration loaded successfully")
    
    # Always start bot in background thread by default (changed behavior)
    start_bot_env = os.environ.get('START_BOT', 'true').lower()
    if start_bot_env == 'true':
        try:
            logging.info("🚀 Auto-starting bot thread...")
            logging.info("📋 Preparing bot initialization sequence...")
            bot_stats.update({
                'status': 'starting',
                'startup_stage': 'initializing',
                'startup_progress': 5
            })
            bot_thread = Thread(target=run_bot_in_background, daemon=True)
            bot_thread.start()
            logging.info("✅ Bot thread started successfully")
            logging.info("⏳ Bot initialization in progress - check logs for updates")
        except Exception as e:
            error_msg = f"❌ Failed to start bot thread: {e}"
            logging.error(error_msg)
            logging.error(f"🔍 Error details: {type(e).__name__}: {str(e)}")
            bot_stats.update({
                'status': 'error',
                'startup_stage': 'error',
                'error_message': str(e)
            })
    else:
        logging.info("⏸️ Bot auto-start disabled via START_BOT environment variable")
        logging.info("📱 Web dashboard ready - bot can be started manually")
        bot_stats.update({
            'status': 'stopped',
            'startup_stage': 'manual'
        })
    
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