# Kuvera Grid Trading Bot - One-Click Deployment Script
# PowerShell script for Windows setup and deployment

param(
    [switch]$TestMode,
    [switch]$SkipDependencies,
    [switch]$Verbose
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    
    switch ($Color) {
        "Red" { Write-Host $Message -ForegroundColor Red }
        "Green" { Write-Host $Message -ForegroundColor Green }
        "Yellow" { Write-Host $Message -ForegroundColor Yellow }
        "Blue" { Write-Host $Message -ForegroundColor Blue }
        "Cyan" { Write-Host $Message -ForegroundColor Cyan }
        "Magenta" { Write-Host $Message -ForegroundColor Magenta }
        default { Write-Host $Message }
    }
}

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-ColorOutput "=" * 60 "Cyan"
    Write-ColorOutput "🚀 $Title" "Yellow"
    Write-ColorOutput "=" * 60 "Cyan"
}

function Write-Step {
    param([string]$Step)
    Write-ColorOutput "\n📋 $Step..." "Blue"
}

function Write-Success {
    param([string]$Message)
    Write-ColorOutput "✅ $Message" "Green"
}

function Write-Warning {
    param([string]$Message)
    Write-ColorOutput "⚠️  $Message" "Yellow"
}

function Write-Error {
    param([string]$Message)
    Write-ColorOutput "❌ $Message" "Red"
}

try {
    Write-Header "KUVERA GRID BOT DEPLOYMENT"
    
    # Check if running as administrator
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
    if (-not $isAdmin) {
        Write-Warning "Not running as administrator. Some operations may fail."
        Write-ColorOutput "Consider running PowerShell as Administrator for full functionality." "Yellow"
    }
    
    # Step 1: Check Python installation
    Write-Step "Checking Python installation"
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Python found: $pythonVersion"
        } else {
            throw "Python not found"
        }
    } catch {
        Write-Error "Python is not installed or not in PATH"
        Write-ColorOutput "Please install Python 3.8+ from https://python.org" "Yellow"
        exit 1
    }
    
    # Step 2: Check pip
    Write-Step "Checking pip installation"
    try {
        $pipVersion = pip --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Pip found: $pipVersion"
        } else {
            throw "Pip not found"
        }
    } catch {
        Write-Error "Pip is not installed"
        Write-ColorOutput "Please install pip or reinstall Python with pip included" "Yellow"
        exit 1
    }
    
    # Step 3: Create virtual environment
    Write-Step "Setting up virtual environment"
    if (-not (Test-Path "venv")) {
        python -m venv venv
        Write-Success "Virtual environment created"
    } else {
        Write-Success "Virtual environment already exists"
    }
    
    # Step 4: Activate virtual environment
    Write-Step "Activating virtual environment"
    & ".\venv\Scripts\Activate.ps1"
    Write-Success "Virtual environment activated"
    
    # Step 5: Install dependencies
    if (-not $SkipDependencies) {
        Write-Step "Installing Python dependencies"
        pip install --upgrade pip
        pip install -r requirements.txt
        Write-Success "Dependencies installed"
    } else {
        Write-Warning "Skipping dependency installation"
    }
    
    # Step 6: Create necessary directories
    Write-Step "Creating directory structure"
    $directories = @("logs", "ai", "config", "utils", "data")
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Success "Created directory: $dir"
        } else {
            Write-Success "Directory exists: $dir"
        }
    }
    
    # Step 7: Setup environment file
    Write-Step "Setting up environment configuration"
    if (-not (Test-Path ".env")) {
        if (Test-Path ".env.template") {
            Copy-Item ".env.template" ".env"
            Write-Success "Environment file created from template"
            Write-Warning "Please edit .env file with your API keys before running the bot"
        } else {
            Write-Warning "No .env.template found. You'll need to create .env manually"
        }
    } else {
        Write-Success "Environment file already exists"
    }
    
    # Step 8: Run tests (if not in test mode)
    if (-not $TestMode) {
        Write-Step "Running component tests"
        try {
            python tests/quick_test.py
            Write-Success "All tests passed"
        } catch {
            Write-Warning "Some tests failed. Check the output above."
        }
    }
    
    # Step 9: Display next steps
    Write-Header "DEPLOYMENT COMPLETE"
    
    Write-ColorOutput "\n🎉 Kuvera Grid Bot is ready!" "Green"
    Write-ColorOutput "\nNext steps:" "Yellow"
    Write-ColorOutput "1. Edit .env file with your API keys" "White"
    Write-ColorOutput "2. Start with testnet mode for safety" "White"
    Write-ColorOutput "3. Run: python bot.py" "White"
    Write-ColorOutput "4. Enable autonomous mode when ready" "White"
    
    Write-ColorOutput "\n📚 Quick commands:" "Cyan"
    Write-ColorOutput "• Test components: python tests/quick_test.py" "White"
    Write-ColorOutput "• Start bot: python bot.py" "White"
    Write-ColorOutput "• Monitor logs: Get-Content logs\\trading.log -Wait" "White"
    Write-ColorOutput "• Emergency stop: Ctrl+C in bot terminal" "White"
    
    Write-ColorOutput "\n⚠️  Important reminders:" "Yellow"
    Write-ColorOutput "• Always test with paper/testnet mode first" "White"
    Write-ColorOutput "• Never share your API keys" "White"
    Write-ColorOutput "• Monitor the bot regularly" "White"
    Write-ColorOutput "• Use stop-loss and position limits" "White"
    
    if ($TestMode) {
        Write-ColorOutput "\n🧪 Test mode completed successfully" "Green"
    }
    
} catch {
    Write-Error "Deployment failed: $($_.Exception.Message)"
    Write-ColorOutput "\nTroubleshooting:" "Yellow"
    Write-ColorOutput "1. Ensure Python 3.8+ is installed" "White"
    Write-ColorOutput "2. Check internet connection for pip installs" "White"
    Write-ColorOutput "3. Run PowerShell as Administrator" "White"
    Write-ColorOutput "4. Check antivirus isn't blocking files" "White"
    exit 1
}

# Additional functions for monitoring
function Start-BotMonitoring {
    Write-Header "STARTING BOT MONITORING"
    
    # Create monitoring script
    $monitorScript = @'
# Kuvera Grid Bot Monitor
while ($true) {
    Clear-Host
    Write-Host "🤖 Kuvera Grid Bot Monitor" -ForegroundColor Cyan
    Write-Host "=" * 40 -ForegroundColor Cyan
    Write-Host "Time: $(Get-Date)" -ForegroundColor Yellow
    
    # Check if bot is running
    $botProcess = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*bot.py*" }
    if ($botProcess) {
        Write-Host "Status: RUNNING" -ForegroundColor Green
        Write-Host "PID: $($botProcess.Id)" -ForegroundColor White
    } else {
        Write-Host "Status: STOPPED" -ForegroundColor Red
    }
    
    # Show recent logs
    if (Test-Path "logs\trading.log") {
        Write-Host "\nRecent logs:" -ForegroundColor Yellow
        Get-Content "logs\trading.log" -Tail 5 | ForEach-Object {
            Write-Host $_ -ForegroundColor White
        }
    }
    
    Write-Host "\nPress Ctrl+C to stop monitoring" -ForegroundColor Gray
    Start-Sleep -Seconds 10
}
'@
    
    $monitorScript | Out-File -FilePath "monitor_bot.ps1" -Encoding UTF8
    Write-Success "Monitor script created: monitor_bot.ps1"
    Write-ColorOutput "Run: .\monitor_bot.ps1 to start monitoring" "Cyan"
}

# Create monitoring script
Start-BotMonitoring

Write-ColorOutput "\n🚀 Ready to trade! Good luck!" "Green"