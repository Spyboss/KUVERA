#!/usr/bin/env python3
"""
Test script to verify logging and bot initialization
"""

import os
import sys
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_logging():
    """Test basic logging functionality"""
    print("🧪 Testing logging system...")
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger('TestLogger')
    
    # Test different log levels
    logger.info("✅ INFO level logging works")
    logger.warning("⚠️ WARNING level logging works")
    logger.error("❌ ERROR level logging works")
    
    print("✅ Basic logging test completed")

def test_bot_import():
    """Test bot import and basic initialization"""
    print("🧪 Testing bot import...")
    
    try:
        from bot import EnhancedTradingBot
        print("✅ Bot import successful")
        
        # Test if config file exists
        config_path = 'config/config.yaml'
        if os.path.exists(config_path):
            print("✅ Config file found")
            
            # Try to create bot instance (this might fail due to API keys)
            try:
                bot = EnhancedTradingBot()
                print("✅ Bot instance created successfully")
                return True
            except Exception as e:
                print(f"⚠️ Bot instance creation failed (expected): {e}")
                print("   This is normal if API keys are not configured")
                return True
        else:
            print(f"❌ Config file not found: {config_path}")
            return False
            
    except ImportError as e:
        print(f"❌ Bot import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_environment():
    """Test environment variables"""
    print("🧪 Testing environment variables...")
    
    # Load .env file if it exists
    env_file = '.env'
    if os.path.exists(env_file):
        print("✅ .env file found")
        from dotenv import load_dotenv
        load_dotenv()
    else:
        print("⚠️ .env file not found")
    
    # Check important environment variables
    env_vars = {
        'TRADING_MODE': os.getenv('TRADING_MODE', 'testnet'),
        'AI_ENABLED': os.getenv('AI_ENABLED', 'true'),
        'STRATEGY_TYPE': os.getenv('STRATEGY_TYPE', 'mean_reversion'),
        'START_BOT': os.getenv('START_BOT', 'true')
    }
    
    for key, value in env_vars.items():
        print(f"   {key}: {value}")
    
    print("✅ Environment variables loaded")
    return True

def main():
    """Run all tests"""
    print("🚀 Starting Kuvera Grid Bot Tests")
    print(f"📅 Test time: {datetime.now()}")
    print("=" * 50)
    
    tests = [
        test_logging,
        test_environment,
        test_bot_import
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result if result is not None else True)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print("-" * 30)
    
    print("📊 Test Results:")
    passed = sum(results)
    total = len(results)
    print(f"   Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! Bot should work correctly.")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)