param(
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = $ProjectRoot
}
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}
$OutputDir = (Resolve-Path $OutputDir).Path
$Archive = Join-Path $OutputDir "llm_econ_linux_package_$Timestamp.tar.gz"

$TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) "llm_econ_project_pack_$Timestamp"
$PackageRoot = Join-Path $TempRoot "llm_econ_project"

New-Item -ItemType Directory -Path $PackageRoot -Force | Out-Null

$IncludeItems = @(
    "run_simulation.py",
    "run_experiment.py",
    "random_tax_simulation.py",
    "run_random_calibration.py",
    "run_saez_experiment.py",
    "saez_simulation.py",
    "preview_saez_schedule.py",
    "validate_calibration_csv.py",
    "requirements.txt",
    "README.md",
    "AGENTS.md",
    "llm_agent",
    "ai_economist",
    "deployment"
)

foreach ($item in $IncludeItems) {
    $src = Join-Path $ProjectRoot $item
    if (-not (Test-Path $src)) {
        Write-Warning "Missing item, skipped: $item"
        continue
    }
    $dst = Join-Path $PackageRoot $item
    if ((Get-Item $src).PSIsContainer) {
        Copy-Item -Path $src -Destination $dst -Recurse -Force
    } else {
        Copy-Item -Path $src -Destination $dst -Force
    }
}

$ExcludedDirs = @(
    ".git", "venv", ".venv", "__pycache__", "simulation_results", "linux_simulation_results", "thesis"
)
$ExcludedFiles = @(
    "*.pdf", "*.docx", "*.doc", ".env", "*.env", "credentials.json"
)

foreach ($dir in $ExcludedDirs) {
    Get-ChildItem -Path $PackageRoot -Recurse -Force -Directory -Filter $dir -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force
}
foreach ($pattern in $ExcludedFiles) {
    Get-ChildItem -Path $PackageRoot -Recurse -Force -File -Filter $pattern -ErrorAction SilentlyContinue |
        Remove-Item -Force
}

Push-Location $TempRoot
try {
    tar -czf $Archive "llm_econ_project"
} finally {
    Pop-Location
    Remove-Item -LiteralPath $TempRoot -Recurse -Force
}

Write-Host "Created package: $Archive"
