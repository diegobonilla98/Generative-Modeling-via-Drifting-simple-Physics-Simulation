$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "Node.js is not installed or not in PATH" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please install Node.js:" -ForegroundColor Cyan
    Write-Host "  1. Download from: https://nodejs.org/" -ForegroundColor Cyan
    Write-Host "  2. Or run as Administrator: winget install OpenJS.NodeJS.LTS" -ForegroundColor Cyan
    exit 1
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "Error: npm is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

$packageJsonPath = Join-Path $scriptPath "package.json"
if (-not (Test-Path $packageJsonPath)) {
    Write-Host "Error: package.json not found" -ForegroundColor Red
    exit 1
}

$nodeModulesPath = Join-Path $scriptPath "node_modules"
if (-not (Test-Path $nodeModulesPath)) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    Set-Location $scriptPath
    npm install
}

Write-Host "Starting development server..." -ForegroundColor Green
Set-Location $scriptPath
npm run dev
