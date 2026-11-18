# Starts the whole NeuroQuest project on Windows: MongoDB (Docker), backend, ML service and the game.
# Usage (from this folder):  powershell -ExecutionPolicy Bypass -File .\start-all.ps1
# Each service opens in its own window; close a window to stop that service.

$root = $PSScriptRoot
$jdk = Get-ChildItem "C:\Program Files\Eclipse Adoptium" -Directory -ErrorAction SilentlyContinue |
       Where-Object Name -like "jdk-21*" | Select-Object -First 1
$maven = "C:\tools\apache-maven-3.9.9\bin"

# 1. Docker + MongoDB
if (-not (docker version --format '{{.Server.Version}}' 2>$null)) {
    Write-Host "Starting Docker Desktop..."
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    while (-not (docker version --format '{{.Server.Version}}' 2>$null)) { Start-Sleep -Seconds 2 }
}
if (docker ps -a --filter name=neuroquest-mongo --format "{{.Names}}") {
    docker start neuroquest-mongo | Out-Null
} else {
    docker run -d --name neuroquest-mongo -p 27017:27017 --restart unless-stopped mongo:8.0 | Out-Null
}
Write-Host "MongoDB running on port 27017"

# 2. Backend (port 8080)
if (Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue) {
    Write-Warning "Port 8080 is already in use (another program, e.g. the load balancer?). Stop it first."
} else {
    $env:JAVA_HOME = $jdk.FullName
    Start-Process powershell -WorkingDirectory "$root\backend" -ArgumentList "-NoExit", "-Command",
        "`$env:JAVA_HOME='$($jdk.FullName)'; `$env:Path='$($jdk.FullName)\bin;$maven;' + `$env:Path; mvn spring-boot:run"
}

# 3. ML service (port 8000)
Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command",
    "python -m uvicorn app:app --app-dir ml/service --port 8000"

# 4. Game (port 5180)
Start-Process powershell -WorkingDirectory "$root\frontend" -ArgumentList "-NoExit", "-Command",
    "npx vite --port 5180 --strictPort"

Write-Host ""
Write-Host "Starting... give it about a minute, then open http://localhost:5180"
Write-Host "Model Dashboard: http://localhost:5180/#dashboard"
