#!/bin/bash

# Railway startup script for Kuvera Grid Trading Bot
echo "🚀 Starting Kuvera Grid Trading Bot on Railway..."

# Set default values for environment variables if not provided
export PORT=${PORT:-8000}
export FLASK_DEBUG=${FLASK_DEBUG:-false}
export AUTOMATED_MODE=${AUTOMATED_MODE:-true}
export TRADING_MODE=${TRADING_MODE:-testnet}
export AI_ENABLED=${AI_ENABLED:-true}
export STRATEGY_TYPE=${STRATEGY_TYPE:-mean_reversion}
export START_BOT=${START_BOT:-true}

# Validate required environment variables
if [ -z "$BINANCE_API_KEY" ] || [ -z "$BINANCE_SECRET_KEY" ]; then
    echo "⚠️  Warning: Binance API keys not set. Bot will run in demo mode."
fi

if [ -z "$BINANCE_TESTNET_API_KEY" ] || [ -z "$BINANCE_TESTNET_SECRET_KEY" ]; then
    echo "⚠️  Warning: Binance testnet API keys not set."
fi

# Log environment info
echo "📊 Configuration:"
echo "   Port: $PORT"
echo "   Trading Mode: $TRADING_MODE"
echo "   Automated Mode: $AUTOMATED_MODE"
echo "   AI Enabled: $AI_ENABLED"
echo "   Strategy: $STRATEGY_TYPE"

# Start the application
echo "🌐 Starting Flask web dashboard..."
exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --max-requests 1000 --preload app:app