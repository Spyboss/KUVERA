# Test API endpoints after deployment
Write-Host "Testing Railway deployment..." -ForegroundColor Green

# Test SSE endpoint
Write-Host "\n1. Testing SSE endpoint:" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri 'https://kuvera-production.up.railway.app/api/logs/sse' -Method GET -TimeoutSec 10
    Write-Host "SSE Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "Content-Type: $($response.Headers['Content-Type'])" -ForegroundColor Green
} catch {
    Write-Host "SSE Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Test regular logs endpoint
Write-Host "\n2. Testing logs endpoint:" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri 'https://kuvera-production.up.railway.app/api/logs?limit=20' -Method GET
    $data = $response.Content | ConvertFrom-Json
    Write-Host "Logs Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "Total logs: $($data.total_logs)" -ForegroundColor Green
    Write-Host "Startup logs: $($data.startup_logs)" -ForegroundColor Green
    Write-Host "Startup stage: $($data.startup_stage)" -ForegroundColor Green
    Write-Host "Recent logs count: $($data.logs.Count)" -ForegroundColor Green
} catch {
    Write-Host "Logs Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Test bot status
Write-Host "\n3. Testing bot status:" -ForegroundColor Yellow
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

# Test health endpoint
Write-Host "\n4. Testing health endpoint:" -ForegroundColor Yellow
try {
    $healthResponse = Invoke-WebRequest -Uri 'https://kuvera-production.up.railway.app/health' -Method GET
    $healthData = $healthResponse.Content | ConvertFrom-Json
    Write-Host "Health Status: $($healthResponse.StatusCode)" -ForegroundColor Green
    Write-Host "App Status: $($healthData.status)" -ForegroundColor Green
    Write-Host "Version: $($healthData.version)" -ForegroundColor Green
} catch {
    Write-Host "Health Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "\nTest completed!" -ForegroundColor Green