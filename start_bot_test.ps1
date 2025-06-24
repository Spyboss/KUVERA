# Start bot and test endpoints
Write-Host "Starting the bot..." -ForegroundColor Yellow

try {
    # Use the correct endpoint format: /api/control/start
    $response = Invoke-WebRequest -Uri 'https://kuvera-production.up.railway.app/api/control/start' -Method POST
    Write-Host "Bot start response: $($response.StatusCode)" -ForegroundColor Green
    
    $data = $response.Content | ConvertFrom-Json
    Write-Host "Success: $($data.success)" -ForegroundColor Green
    Write-Host "Message: $($data.message)" -ForegroundColor Green
    
    # Wait for bot to start
    Write-Host "Waiting for bot to initialize..." -ForegroundColor Yellow
    Start-Sleep -Seconds 20
    
    # Test endpoints again
    Write-Host "\nTesting endpoints after bot start:" -ForegroundColor Green
    
    # Test bot status first
    Write-Host "\n1. Testing bot status:" -ForegroundColor Yellow
    try {
        $statusResponse = Invoke-WebRequest -Uri 'https://kuvera-production.up.railway.app/api/status' -Method GET
        $statusData = $statusResponse.Content | ConvertFrom-Json
        Write-Host "Status: $($statusResponse.StatusCode)" -ForegroundColor Green
        Write-Host "Bot Status: $($statusData.status)" -ForegroundColor Green
        Write-Host "Startup Stage: $($statusData.startup_stage)" -ForegroundColor Green
        Write-Host "Uptime: $($statusData.uptime)" -ForegroundColor Green
    } catch {
        Write-Host "Status Error: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    # Test logs endpoint
    Write-Host "\n2. Testing logs endpoint:" -ForegroundColor Yellow
    try {
        $logsResponse = Invoke-WebRequest -Uri 'https://kuvera-production.up.railway.app/api/logs?limit=10' -Method GET
        $logsData = $logsResponse.Content | ConvertFrom-Json
        Write-Host "Logs Status: $($logsResponse.StatusCode)" -ForegroundColor Green
        Write-Host "Total logs: $($logsData.total_logs)" -ForegroundColor Green
        Write-Host "Recent logs count: $($logsData.logs.Count)" -ForegroundColor Green
        Write-Host "Bot status: $($logsData.bot_status)" -ForegroundColor Green
        Write-Host "Startup stage: $($logsData.startup_stage)" -ForegroundColor Green
        
        if ($logsData.logs.Count -gt 0) {
            Write-Host "\nSample logs:" -ForegroundColor Cyan
            for ($i = 0; $i -lt [Math]::Min(3, $logsData.logs.Count); $i++) {
                $log = $logsData.logs[$i]
                Write-Host "  [$($log.level)] $($log.message)" -ForegroundColor Gray
            }
        }
    } catch {
        Write-Host "Logs Error: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    # Test SSE endpoint
    Write-Host "\n3. Testing SSE endpoint:" -ForegroundColor Yellow
    try {
        $sseResponse = Invoke-WebRequest -Uri 'https://kuvera-production.up.railway.app/api/logs/sse' -Method GET -TimeoutSec 5
        Write-Host "SSE Status: $($sseResponse.StatusCode)" -ForegroundColor Green
        Write-Host "Content-Type: $($sseResponse.Headers['Content-Type'])" -ForegroundColor Green
        Write-Host "First 200 chars: $($sseResponse.Content.Substring(0, [Math]::Min(200, $sseResponse.Content.Length)))" -ForegroundColor Cyan
    } catch {
        Write-Host "SSE Error: $($_.Exception.Message)" -ForegroundColor Red
    }
    
} catch {
    Write-Host "Start Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "\nTest completed!" -ForegroundColor Green