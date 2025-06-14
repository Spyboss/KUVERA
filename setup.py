#!/usr/bin/env python3
"""
Setup Script for Binance Trading Bot
Automated installation and configuration
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_header():
    """Print setup header"""
    print("🤖 Binance Trading Bot Setup")
    print("=" * 40)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
        
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_requirements():
    """Install Python requirements"""
    print("\nInstalling Python packages...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ Python packages installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def install_talib():
    """Install TA-Lib with platform-specific instructions"""
    print("\nInstalling TA-Lib...")
    
    try:
        # Try to import TA-Lib first
        import talib
        print("✅ TA-Lib already installed")
        return True
        
    except ImportError:
        print("TA-Lib not found. Installing...")
        
        if sys.platform.startswith('win'):
            print("\n📋 Windows TA-Lib Installation:")
            print("1. Download TA-Lib wheel from:")
            print("   https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
            print("2. Choose the correct wheel for your Python version")
            print("3. Install with: pip install downloaded_wheel.whl")
            print("\nAlternatively, try:")
            
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "TA-Lib"])
                print("✅ TA-Lib installed successfully")
                return True
            except subprocess.CalledProcessError:
                print("❌ Automatic TA-Lib installation failed")
                print("   Please install manually using the instructions above")
                return False
                
        else:
            # Linux/Mac installation
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "TA-Lib"])
                print("✅ TA-Lib installed successfully")
                return True
            except subprocess.CalledProcessError:
                print("❌ TA-Lib installation failed")
                print("   On Linux/Mac, you may need to install system dependencies first:")
                print("   sudo apt-get install libta-lib-dev  # Ubuntu/Debian")
                print("   brew install ta-lib  # macOS")
                return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    directories = [
        'logs',
        'backtests',
        'data',
        'config'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
        
    return True

def setup_environment_file():
    """Setup .env file if it doesn't exist"""
    print("\nSetting up environment file...")
    
    env_file = Path('.env')
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
        
    # Create .env file with template
    env_content = """
# Binance API Credentials
# IMPORTANT: Never commit this file to version control!
# Get your API keys from: https://testnet.binance.vision/ (testnet) or https://www.binance.com/en/my/settings/api-management (live)

BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here

# Trading Environment
TRADING_MODE=testnet  # Options: testnet, live

# Alerts (optional - for email/SMS notifications)
ALERT_EMAIL=your_email@example.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@example.com
SMTP_PASSWORD=your_app_password
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content.strip())
        print("✅ .env file created")
        print("   ⚠️  Remember to update it with your actual API credentials")
        return True
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    print("\nTesting package imports...")
    
    packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('yaml', 'PyYAML'),
        ('binance.spot', 'binance-connector'),
        ('backtrader', 'backtrader'),
        ('matplotlib', 'matplotlib'),
        ('dotenv', 'python-dotenv')
    ]
    
    failed_imports = []
    
    for package, pip_name in packages:
        try:
            __import__(package)
            print(f"✅ {pip_name}")
        except ImportError:
            print(f"❌ {pip_name}")
            failed_imports.append(pip_name)
            
    # Test TA-Lib separately
    try:
        import talib
        print("✅ TA-Lib")
    except ImportError:
        print("❌ TA-Lib")
        failed_imports.append('TA-Lib')
        
    if failed_imports:
        print(f"\n⚠️  Failed imports: {', '.join(failed_imports)}")
        print("   Please install missing packages manually")
        return False
        
    print("\n✅ All packages imported successfully")
    return True

def check_api_setup():
    """Check if API credentials are configured"""
    print("\nChecking API configuration...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_SECRET_KEY')
        
        if not api_key or api_key == 'your_api_key_here':
            print("⚠️  API key not configured")
            return False
            
        if not api_secret or api_secret == 'your_secret_key_here':
            print("⚠️  API secret not configured")
            return False
            
        print("✅ API credentials configured")
        return True
        
    except Exception as e:
        print(f"❌ Error checking API configuration: {e}")
        return False

def run_basic_test():
    """Run a basic functionality test"""
    print("\nRunning basic functionality test...")
    
    try:
        # Test configuration loading
        import yaml
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration file loaded")
        
        # Test strategy import
        sys.path.append('.')
        from strategy.mean_reversion import MeanReversionStrategy
        print("✅ Strategy class imported")
        
        # Test data manager
        from utils.data_manager import DataManager
        dm = DataManager()
        print("✅ Data manager initialized")
        
        print("\n✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n🎉 Setup Complete!")
    print("=" * 40)
    print("\n📋 Next Steps:")
    print("\n1. Configure API Credentials:")
    print("   - Edit .env file with your Binance API keys")
    print("   - Get testnet keys from: https://testnet.binance.vision/")
    print("\n2. Run Backtest:")
    print("   python backtest.py")
    print("\n3. Start Live Trading (testnet):")
    print("   python bot.py")
    print("\n4. Monitor Performance:")
    print("   .\\monitor.ps1")
    print("\n⚠️  Important Reminders:")
    print("   - Always test on testnet first")
    print("   - Never commit .env file to version control")
    print("   - Start with small amounts")
    print("   - Monitor the bot closely")
    print("\n📚 Documentation:")
    print("   - Read README.md for detailed instructions")
    print("   - Check logs/ directory for trading logs")
    print("   - Review config/config.yaml for parameters")

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        return False
        
    # Create directories
    if not create_directories():
        return False
        
    # Install requirements
    if not install_requirements():
        print("\n⚠️  Some packages failed to install. You may need to install them manually.")
        
    # Install TA-Lib
    if not install_talib():
        print("\n⚠️  TA-Lib installation failed. Please install it manually.")
        
    # Setup environment file
    if not setup_environment_file():
        return False
        
    # Test imports
    if not test_imports():
        print("\n⚠️  Some packages are missing. Please install them before proceeding.")
        
    # Run basic test
    if not run_basic_test():
        print("\n⚠️  Basic functionality test failed. Please check the installation.")
        
    # Print next steps
    print_next_steps()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Setup completed successfully!")
        else:
            print("\n❌ Setup completed with warnings. Please review the messages above.")
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
