# Trading Bot Monitor Script
# Monitors profit/drawdown and sends alerts

param(
    [string]$ConfigPath = "config\config.yaml",
    [string]$LogPath = "logs",
    [int]$CheckInterval = 60  # Check every 60 seconds
)

# Function to read YAML config (simplified)
function Read-Config {
    param([string]$Path)
    
    if (-not (Test-Path $Path)) {
        Write-Error "Config file not found: $Path"
        exit 1
    }
    
    $config = @{}
    $content = Get-Content $Path
    
    foreach ($line in $content) {
        if ($line -match '^\s*([^#:]+):\s*(.+)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim().Trim('"')
            
            # Try to convert to number if possible
            if ($value -match '^[0-9.]+$') {
                $config[$key] = [double]$value
            } else {
                $config[$key] = $value
            }
        }
    }
    
    return $config
}

# Function to get latest log file
function Get-LatestLogFile {
    param([string]$LogDirectory)
    
    $logFiles = Get-ChildItem -Path $LogDirectory -Filter "trading_bot_*.log" | Sort-Object LastWriteTime -Descending
    
    if ($logFiles.Count -eq 0) {
        return $null
    }
    
    return $logFiles[0].FullName
}

# Function to parse trading statistics from log
function Get-TradingStats {
    param([string]$LogFile)
    
    if (-not (Test-Path $LogFile)) {
        return $null
    }
    
    $stats = @{
        TotalProfit = 0.0
        TotalTrades = 0
        WinningTrades = 0
        CurrentDrawdown = 0.0
        LastUpdate = (Get-Date)
        HasPosition = $false
    }
    
    # Read last 100 lines for recent activity
    $lines = Get-Content $LogFile -Tail 100
    
    foreach ($line in $lines) {
        # Look for profit updates
        if ($line -match 'Total: \$([0-9.-]+)') {
            $stats.TotalProfit = [double]$matches[1]
        }
        
        # Count trades
        if ($line -match 'SELL ORDER') {
            $stats.TotalTrades++
            
            # Check if it was a winning trade
            if ($line -match 'Profit: \$([0-9.-]+)' -and [double]$matches[1] -gt 0) {
                $stats.WinningTrades++
            }
        }
        
        # Check for current position
        if ($line -match 'BUY ORDER PLACED') {
            $stats.HasPosition = $true
        }
        
        if ($line -match 'SELL ORDER PLACED') {
            $stats.HasPosition = $false
        }
        
        # Update last activity time
        if ($line -match '^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})') {
            try {
                $stats.LastUpdate = [DateTime]::ParseExact($matches[1], "yyyy-MM-dd HH:mm:ss", $null)
            } catch {
                # Ignore parsing errors
            }
        }
    }
    
    # Calculate drawdown (simplified - just negative profit)
    if ($stats.TotalProfit -lt 0) {
        $stats.CurrentDrawdown = [Math]::Abs($stats.TotalProfit)
    }
    
    return $stats
}

# Function to send alert
function Send-Alert {
    param(
        [string]$Message,
        [string]$Type = "INFO"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $alertMessage = "[$timestamp] [$Type] $Message"
    
    # Display alert
    switch ($Type) {
        "SUCCESS" { Write-Host $alertMessage -ForegroundColor Green }
        "WARNING" { Write-Host $alertMessage -ForegroundColor Yellow }
        "ERROR" { Write-Host $alertMessage -ForegroundColor Red }
        default { Write-Host $alertMessage -ForegroundColor Cyan }
    }
    
    # Log alert to file
    $alertLogPath = "logs\alerts.log"
    Add-Content -Path $alertLogPath -Value $alertMessage
    
    # Optional: Send email alert (uncomment and configure)
    # Send-EmailAlert -Message $Message -Type $Type
    
    # Optional: Play sound alert
    if ($Type -eq "WARNING" -or $Type -eq "ERROR") {
        [System.Console]::Beep(800, 500)
    } elseif ($Type -eq "SUCCESS") {
        [System.Console]::Beep(1000, 200)
        Start-Sleep -Milliseconds 100
        [System.Console]::Beep(1200, 200)
    }
}

# Function to send email alert (optional)
function Send-EmailAlert {
    param(
        [string]$Message,
        [string]$Type
    )
    
    # Configure these variables with your email settings
    $smtpServer = $env:SMTP_SERVER
    $smtpPort = $env:SMTP_PORT
    $emailFrom = $env:SMTP_USERNAME
    $emailTo = $env:ALERT_EMAIL
    $emailPassword = $env:SMTP_PASSWORD
    
    if (-not $smtpServer -or -not $emailFrom -or -not $emailTo) {
        return  # Email not configured
    }
    
    try {
        $subject = "Trading Bot Alert - $Type"
        $body = @"
Trading Bot Alert

Type: $Type
Time: $(Get-Date)
Message: $Message

This is an automated alert from your trading bot.
"@
        
        $credential = New-Object System.Management.Automation.PSCredential($emailFrom, (ConvertTo-SecureString $emailPassword -AsPlainText -Force))
        
        Send-MailMessage -SmtpServer $smtpServer -Port $smtpPort -UseSsl -Credential $credential -From $emailFrom -To $emailTo -Subject $subject -Body $body
        
        Write-Host "Email alert sent successfully" -ForegroundColor Green
    } catch {
        Write-Host "Failed to send email alert: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Function to check bot health
function Test-BotHealth {
    param([hashtable]$Stats)
    
    $issues = @()
    
    # Check if bot is active (last update within 10 minutes)
    $timeSinceUpdate = (Get-Date) - $Stats.LastUpdate
    if ($timeSinceUpdate.TotalMinutes -gt 10) {
        $issues += "Bot appears inactive (last update: $($Stats.LastUpdate))"
    }
    
    # Check for excessive losses
    if ($Stats.CurrentDrawdown -gt 2.0) {
        $issues += "High drawdown detected: $($Stats.CurrentDrawdown.ToString('F2'))"
    }
    
    # Check win rate if we have enough trades
    if ($Stats.TotalTrades -gt 10) {
        $winRate = ($Stats.WinningTrades / $Stats.TotalTrades) * 100
        if ($winRate -lt 40) {
            $issues += "Low win rate: $($winRate.ToString('F1'))%"
        }
    }
    
    return $issues
}

# Main monitoring function
function Start-Monitoring {
    Write-Host "🤖 Trading Bot Monitor Started" -ForegroundColor Green
    Write-Host "==============================" -ForegroundColor Green
    Write-Host "Config: $ConfigPath"
    Write-Host "Log Path: $LogPath"
    Write-Host "Check Interval: $CheckInterval seconds"
    Write-Host "Press Ctrl+C to stop monitoring"
    Write-Host ""
    
    # Load configuration
    $config = Read-Config -Path $ConfigPath
    $profitThreshold = if ($config.ContainsKey('profit_alert_threshold')) { $config['profit_alert_threshold'] } else { 1.0 }
    $drawdownThreshold = if ($config.ContainsKey('drawdown_alert_threshold')) { $config['drawdown_alert_threshold'] } else { 0.5 }
    
    Write-Host "Profit Alert Threshold: $profitThreshold"
    Write-Host "Drawdown Alert Threshold: $drawdownThreshold"
    Write-Host ""
    
    $lastProfitAlert = 0.0
    $lastDrawdownAlert = 0.0
    $startTime = Get-Date
    
    try {
        while ($true) {
            # Get latest log file
            $logFile = Get-LatestLogFile -LogDirectory $LogPath
            
            if ($logFile) {
                # Parse trading statistics
                $stats = Get-TradingStats -LogFile $logFile
                
                if ($stats) {
                    # Display current status
                    $runtime = (Get-Date) - $startTime
                    $winRate = if ($stats.TotalTrades -gt 0) { ($stats.WinningTrades / $stats.TotalTrades) * 100 } else { 0 }
                    
                    Clear-Host
                    Write-Host "🤖 Trading Bot Monitor - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
                    Write-Host "=" * 60 -ForegroundColor Green
                    Write-Host ""
                    Write-Host "📊 PERFORMANCE SUMMARY" -ForegroundColor Cyan
                    Write-Host "Total Profit: `$" -NoNewline
                    if ($stats.TotalProfit -ge 0) {
                        Write-Host $stats.TotalProfit.ToString('F2') -ForegroundColor Green
                    } else {
                        Write-Host $stats.TotalProfit.ToString('F2') -ForegroundColor Red
                    }
                    Write-Host "Total Trades: $($stats.TotalTrades)"
                    Write-Host "Winning Trades: $($stats.WinningTrades)"
                    Write-Host "Win Rate: " -NoNewline
                    if ($winRate -ge 60) {
                        Write-Host "$($winRate.ToString('F1'))%" -ForegroundColor Green
                    } elseif ($winRate -ge 40) {
                        Write-Host "$($winRate.ToString('F1'))%" -ForegroundColor Yellow
                    } else {
                        Write-Host "$($winRate.ToString('F1'))%" -ForegroundColor Red
                    }
                    Write-Host "Current Drawdown: `$" -NoNewline
                    if ($stats.CurrentDrawdown -le $drawdownThreshold) {
                        Write-Host $stats.CurrentDrawdown.ToString('F2') -ForegroundColor Green
                    } else {
                        Write-Host $stats.CurrentDrawdown.ToString('F2') -ForegroundColor Red
                    }
                    Write-Host "Has Position: " -NoNewline
                    if ($stats.HasPosition) {
                        Write-Host "YES" -ForegroundColor Yellow
                    } else {
                        Write-Host "NO" -ForegroundColor Gray
                    }
                    Write-Host "Last Update: $($stats.LastUpdate.ToString('yyyy-MM-dd HH:mm:ss'))"
                    Write-Host "Runtime: $($runtime.ToString('hh\:mm\:ss'))"
                    Write-Host ""
                    
                    # Check for alerts
                    
                    # Profit target alert
                    if ($stats.TotalProfit -ge $profitThreshold -and $stats.TotalProfit -gt $lastProfitAlert) {
                        Send-Alert -Message "🎉 PROFIT TARGET REACHED: `$$($stats.TotalProfit.ToString('F2'))" -Type "SUCCESS"
                        $lastProfitAlert = $stats.TotalProfit
                    }
                    
                    # Drawdown alert
                    if ($stats.CurrentDrawdown -ge $drawdownThreshold -and $stats.CurrentDrawdown -gt $lastDrawdownAlert) {
                        Send-Alert -Message "⚠️ HIGH DRAWDOWN: `$$($stats.CurrentDrawdown.ToString('F2'))" -Type "WARNING"
                        $lastDrawdownAlert = $stats.CurrentDrawdown
                    }
                    
                    # Health check
                    $healthIssues = Test-BotHealth -Stats $stats
                    if ($healthIssues.Count -gt 0) {
                        Write-Host "⚠️ HEALTH ISSUES:" -ForegroundColor Red
                        foreach ($issue in $healthIssues) {
                            Write-Host "  - $issue" -ForegroundColor Red
                            Send-Alert -Message $issue -Type "ERROR"
                        }
                        Write-Host ""
                    }
                    
                    # Progress towards daily target
                    $dailyTarget = 1.0  # $1 per day target
                    $progressPct = ($stats.TotalProfit / $dailyTarget) * 100
                    Write-Host "📈 DAILY PROGRESS" -ForegroundColor Cyan
                    Write-Host "Target: `$$($dailyTarget.ToString('F2'))"
                    Write-Host "Progress: " -NoNewline
                    if ($progressPct -ge 100) {
                        Write-Host "$($progressPct.ToString('F1'))% (TARGET REACHED!)" -ForegroundColor Green
                    } elseif ($progressPct -ge 50) {
                        Write-Host "$($progressPct.ToString('F1'))%" -ForegroundColor Yellow
                    } else {
                        Write-Host "$($progressPct.ToString('F1'))%" -ForegroundColor Gray
                    }
                    
                    # Risk assessment
                    Write-Host ""
                    Write-Host "⚖️ RISK STATUS" -ForegroundColor Cyan
                    $riskLevel = "LOW"
                    $riskColor = "Green"
                    
                    if ($stats.CurrentDrawdown -gt 1.0) {
                        $riskLevel = "HIGH"
                        $riskColor = "Red"
                    } elseif ($stats.CurrentDrawdown -gt 0.5 -or $winRate -lt 50) {
                        $riskLevel = "MEDIUM"
                        $riskColor = "Yellow"
                    }
                    
                    Write-Host "Risk Level: " -NoNewline
                    Write-Host $riskLevel -ForegroundColor $riskColor
                    
                } else {
                    Write-Host "No trading statistics available" -ForegroundColor Yellow
                }
            } else {
                Write-Host "No log files found in $LogPath" -ForegroundColor Yellow
            }
            
            Write-Host ""
            Write-Host "Next check in $CheckInterval seconds... (Ctrl+C to stop)" -ForegroundColor Gray
            
            # Wait for next check
            Start-Sleep -Seconds $CheckInterval
        }
    } catch {
        if ($_.Exception.Message -notmatch "terminated by the user") {
            Write-Host "Monitor error: $($_.Exception.Message)" -ForegroundColor Red
        }
    } finally {
        Write-Host ""
        Write-Host "Monitor stopped." -ForegroundColor Yellow
    }
}

# Create logs directory if it doesn't exist
if (-not (Test-Path $LogPath)) {
    New-Item -ItemType Directory -Path $LogPath -Force | Out-Null
}

# Start monitoring
Start-Monitoring