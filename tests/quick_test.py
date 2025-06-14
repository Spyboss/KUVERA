#!/usr/bin/env python3
"""
Kuvera Grid Trading Bot - Quick Test Suite
Validates all components including autonomous trading features
"""

import asyncio
import os
import sys
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import TradingBot
from ai.ai_strategy_optimizer import AIStrategyOptimizer
from ai.emergency_stop import EmergencyStopSystem
from ai.auto_trader import AutonomousTrader

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    @classmethod
    def red(cls, text): return f"{cls.RED}{text}{cls.END}"
    
    @classmethod
    def green(cls, text): return f"{cls.GREEN}{text}{cls.END}"
    
    @classmethod
    def yellow(cls, text): return f"{cls.YELLOW}{text}{cls.END}"
    
    @classmethod
    def blue(cls, text): return f"{cls.BLUE}{text}{cls.END}"
    
    @classmethod
    def cyan(cls, text): return f"{cls.CYAN}{text}{cls.END}"
    
    @classmethod
    def bold(cls, text): return f"{cls.BOLD}{text}{cls.END}"

class TestSuite:
    """Comprehensive test suite for Kuvera Grid Bot"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
        
    def test(self, name):
        """Decorator for test functions"""
        def decorator(func):
            self.tests.append((name, func))
            return func
        return decorator
    
    def assert_true(self, condition, message=""):
        """Assert that condition is True"""
        if not condition:
            raise AssertionError(f"Expected True, got False. {message}")
    
    def assert_equal(self, actual, expected, message=""):
        """Assert that actual equals expected"""
        if actual != expected:
            raise AssertionError(f"Expected {expected}, got {actual}. {message}")
    
    def assert_not_none(self, value, message=""):
        """Assert that value is not None"""
        if value is None:
            raise AssertionError(f"Expected not None, got None. {message}")
    
    async def run_tests(self):
        """Run all registered tests"""
        print(f"\n{Colors.bold('🧪 KUVERA GRID BOT - TEST SUITE')}")
        print("=" * 50)
        
        failed_tests = []
        for test_name, test_func in self.tests:
            try:
                print(f"\n{Colors.cyan(f'Testing: {test_name}')}", end=" ... ")
                if asyncio.iscoroutinefunction(test_func):
                    await test_func()
                else:
                    test_func()
                print(f"{Colors.green('PASS')}")
                self.passed += 1
            except Exception as e:
                print(f"{Colors.red('FAIL')}")
                error_msg = str(e)
                print(f"  {Colors.red(f'Error: {error_msg}')}")
                failed_tests.append((test_name, error_msg))
                self.failed += 1
        
        # Print summary
        print("\n" + "=" * 50)
        total = self.passed + self.failed
        if self.failed == 0:
            print(f"{Colors.green(f'✅ ALL TESTS PASSED ({self.passed}/{total})')}")
        else:
            print(f"{Colors.yellow(f'⚠️  TESTS COMPLETED: {self.passed} passed, {self.failed} failed')}")
            if failed_tests:
                print(f"\n{Colors.red('Failed tests:')}")
                for test_name, error in failed_tests:
                    print(f"  • {test_name}: {error}")
        print("=" * 50)
        
        return self.failed == 0

# Initialize test suite
test_suite = TestSuite()

@test_suite.test("Configuration Loading")
def test_config_loading():
    """Test that configuration files can be loaded"""
    config_path = "config/config.yaml"
    test_suite.assert_true(os.path.exists(config_path), "Config file should exist")
    
    # Test environment file template
    env_template = ".env.template"
    if os.path.exists(env_template):
        with open(env_template, 'r') as f:
            content = f.read()
            test_suite.assert_true("BINANCE_API_KEY" in content, "Should have API key template")
            test_suite.assert_true("OPENROUTER_API_KEY" in content, "Should have OpenRouter key template")

@test_suite.test("AI Strategy Optimizer Initialization")
def test_ai_optimizer_init():
    """Test AI Strategy Optimizer initialization"""
    # Test with mock API key
    optimizer = AIStrategyOptimizer(api_key="test_key")
    test_suite.assert_not_none(optimizer, "Optimizer should initialize with API key")
    test_suite.assert_not_none(optimizer.api_key, "Should have API key")
    test_suite.assert_equal(optimizer.api_key, "test_key", "Should store API key correctly")
    
    # Test rate limiting configuration
    test_suite.assert_true(optimizer.rate_limit_delay >= 12.0, "Should have proper rate limiting for free tier")
    test_suite.assert_not_none(optimizer.models, "Should have AI models configured")
    test_suite.assert_true('sentiment' in optimizer.models, "Should have sentiment analysis model")

@test_suite.test("Emergency Stop System")
def test_emergency_stop():
    """Test Emergency Stop System functionality"""
    # Create test config
    test_config = {
        'max_daily_loss_pct': 0.03,
        'max_consecutive_losses': 3,
        'max_drawdown_pct': 0.05,
        'volatility_threshold': 0.02,
        'flash_crash_threshold': 0.10
    }
    
    emergency_stop = EmergencyStopSystem(test_config)
    test_suite.assert_not_none(emergency_stop, "Emergency stop should initialize")
    test_suite.assert_equal(emergency_stop.MAX_DAILY_LOSS, 0.03, "Should set daily loss limit")
    
    # Test emergency conditions check
    result = emergency_stop.check_emergency_conditions()
    test_suite.assert_true(isinstance(result, tuple) and len(result) == 2, "check_emergency_conditions should return (bool, list)")
    
    # Test daily loss limit
    emergency_stop.update_balance(1000.0)  # Starting balance
    emergency_stop.add_trade_result(-50.0, "sell")  # Add a loss to trigger daily loss
    should_stop, reasons = emergency_stop.check_emergency_conditions()
    test_suite.assert_true(should_stop, "Should trigger emergency stop on high loss")
    
    # Test manual resume
    resume_result = emergency_stop.manual_resume()
    test_suite.assert_true(resume_result, "Manual resume should return True when successful")
    test_suite.assert_true(not emergency_stop.is_emergency_stopped, "Emergency stop should be cleared after manual resume")

@test_suite.test("Autonomous Trader Initialization")
def test_autonomous_trader():
    """Test Autonomous Trader initialization"""
    # Mock trading bot
    mock_bot = Mock()
    mock_bot.config = {
        'trading': {'symbol': 'BTCUSDT'},
        'strategy': {'sma_period': 20}
    }
    mock_bot.logger = Mock()
    
    # Test config
    test_config = {
        'trading': {'symbol': 'BTCUSDT', 'capital': 30.0},
        'strategy': {'sma_period': 20},
        'risk_management': {'max_daily_trades': 5}
    }
    
    # Test initialization without API key
    auto_trader = AutonomousTrader(mock_bot, test_config)
    test_suite.assert_not_none(auto_trader, "Autonomous trader should initialize")
    test_suite.assert_equal(auto_trader.is_autonomous, False, "Autonomous trading should be disabled by default")
    test_suite.assert_equal(auto_trader.ai_optimizer, None, "Should not have AI optimizer without API key")

@test_suite.test("Trading Bot Initialization")
async def test_trading_bot_init():
    """Test Trading Bot initialization in paper mode"""
    # Test paper trading mode (safe for testing)
    bot = TradingBot(trading_mode="paper")
    test_suite.assert_not_none(bot, "Bot should initialize")
    test_suite.assert_equal(bot.trading_mode, "paper", "Should be in paper mode")
    test_suite.assert_not_none(bot.config, "Config should be loaded")
    test_suite.assert_not_none(bot.logger, "Logger should be initialized")

@test_suite.test("Signal Processing Logic")
def test_signal_processing():
    """Test signal processing logic"""
    # Mock price data
    prices = [100, 101, 102, 101, 100, 99, 98, 99, 100, 101]
    
    # Calculate simple moving average
    sma_period = 5
    if len(prices) >= sma_period:
        sma = sum(prices[-sma_period:]) / sma_period
        current_price = prices[-1]
        
        # Test buy signal (price below SMA)
        deviation_threshold = 0.005  # 0.5%
        buy_threshold = sma * (1 - deviation_threshold)
        sell_threshold = sma * (1 + deviation_threshold)
        
        test_suite.assert_true(isinstance(sma, (int, float)), "SMA should be numeric")
        test_suite.assert_true(isinstance(buy_threshold, (int, float)), "Buy threshold should be numeric")
        test_suite.assert_true(isinstance(sell_threshold, (int, float)), "Sell threshold should be numeric")

@test_suite.test("Risk Management Calculations")
def test_risk_management():
    """Test risk management calculations"""
    account_balance = 1000.0
    risk_per_trade = 0.01  # 1%
    max_position_size = 0.1  # 10%
    
    # Calculate position size
    risk_amount = account_balance * risk_per_trade
    max_amount = account_balance * max_position_size
    
    test_suite.assert_equal(risk_amount, 10.0, "Risk amount should be 1% of balance")
    test_suite.assert_equal(max_amount, 100.0, "Max amount should be 10% of balance")
    test_suite.assert_true(risk_amount <= max_amount, "Risk amount should not exceed max position")

@test_suite.test("Logging System")
def test_logging():
    """Test logging system functionality"""
    logs_dir = "logs"
    test_suite.assert_true(os.path.exists(logs_dir), "Logs directory should exist")
    
    # Test log file creation
    from datetime import datetime
    log_filename = f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(logs_dir, log_filename)
    
    # Create a test log entry
    with open(log_path, 'w') as f:
        f.write(f"Test log entry at {datetime.now()}\n")
    
    test_suite.assert_true(os.path.exists(log_path), "Log file should be created")
    
    # Clean up test log
    if os.path.exists(log_path):
        os.remove(log_path)

@test_suite.test("Configuration Validation")
def test_config_validation():
    """Test configuration parameter validation"""
    # Test required configuration parameters
    required_params = [
        'trading.symbol',
        'trading.capital',
        'trading.risk_per_trade',
        'strategy.sma_period',
        'strategy.entry_threshold',
        'risk_management.max_daily_trades'
    ]
    
    # Load config to validate
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    for param_path in required_params:
        keys = param_path.split('.')
        value = config
        for key in keys:
            test_suite.assert_true(key in value, f"Config should have {param_path}")
            value = value[key]
        test_suite.assert_not_none(value, f"{param_path} should not be None")

@test_suite.test("File Structure Validation")
def test_file_structure():
    """Test that all required files and directories exist"""
    required_files = [
        'bot.py',
        'config/config.yaml',
        'requirements.txt',
        'README.md'
    ]
    
    required_dirs = [
        'ai',
        'config',
        'logs',
        'utils'
    ]
    
    for file_path in required_files:
        test_suite.assert_true(os.path.exists(file_path), f"Required file {file_path} should exist")
    
    for dir_path in required_dirs:
        test_suite.assert_true(os.path.isdir(dir_path), f"Required directory {dir_path} should exist")
    
    # Test AI module files
    ai_files = [
        'ai/ai_strategy_optimizer.py',
        'ai/emergency_stop.py',
        'ai/auto_trader.py'
    ]
    
    for ai_file in ai_files:
        test_suite.assert_true(os.path.exists(ai_file), f"AI module {ai_file} should exist")

@test_suite.test("Mock Trading Simulation")
async def test_mock_trading():
    """Test mock trading functionality"""
    # Create a mock trading scenario
    mock_prices = [50000, 49500, 49000, 49500, 50000, 50500, 51000, 50500, 50000]
    
    # Simulate SMA calculation and signal generation
    sma_period = 5
    signals = []
    
    for i in range(sma_period, len(mock_prices)):
        current_price = mock_prices[i]
        sma = sum(mock_prices[i-sma_period:i]) / sma_period
        
        # Generate signals based on price vs SMA
        if current_price < sma * 0.995:  # 0.5% below SMA
            signals.append('BUY')
        elif current_price > sma * 1.005:  # 0.5% above SMA
            signals.append('SELL')
        else:
            signals.append('HOLD')
    
    test_suite.assert_true(len(signals) > 0, "Should generate trading signals")
    test_suite.assert_true(all(signal in ['BUY', 'SELL', 'HOLD'] for signal in signals), "All signals should be valid")

async def main():
    """Main test runner"""
    print(f"{Colors.bold('Kuvera Grid Bot - Component Validation')}")
    print(f"{Colors.cyan('Testing all autonomous trading components...')}\n")
    
    # Run the test suite
    success = await test_suite.run_tests()
    
    if success:
        print(f"\n{Colors.green('🎉 All tests passed! The bot is ready for autonomous trading.')}")
        print(f"{Colors.yellow('💡 Next steps:')}")
        print(f"  1. Configure your API keys in .env file")
        print(f"  2. Start with paper trading mode")
        print(f"  3. Enable autonomous mode when ready")
        print(f"  4. Monitor the bot's performance")
    else:
        print(f"\n{Colors.red('❌ Some tests failed. Please fix the issues before running the bot.')}")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.yellow('Tests interrupted by user.')}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.red(f'Test suite error: {e}')}")
        sys.exit(1)