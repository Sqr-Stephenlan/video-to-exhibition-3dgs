# Windows PowerShell equivalent of dev.sh
# Usage: .\dev.ps1 {doctor|bootstrap|python|pip} ...
param(
    [Parameter(Position=0)]
    [string]$Command = "doctor",
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$RemainingArgs
)

$Root = $PSScriptRoot
Set-Location $Root

$VenDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { "venv" }
$Py = Join-Path $VenDir "Scripts" "python.exe"
$Pip = Join-Path $VenDir "Scripts" "pip.exe"

function Find-SystemPython {
    $candidates = @("python3", "python")
    foreach ($cmd in $candidates) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($found) { return $found.Source }
    }
    throw "No system Python found. Install Python 3 before bootstrapping."
}

function Doctor {
    Write-Host "Project root: $Root"
    Write-Host "Venv dir: $VenDir"
    if (Test-Path $Py) {
        $ver = & $Py --version 2>&1
        Write-Host "Python: $($Py)"
        Write-Host "Version: $($ver)"
    } elseif (Test-Path $VenDir) {
        Write-Host "Venv status: present but $Py is not executable"
    } else {
        Write-Host "Venv status: missing"
    }
}

function Bootstrap {
    if (Test-Path "$Py") {
        Write-Host "Existing virtual environment found at $VenDir"
        Doctor
        return
    }

    if (Test-Path "$VenDir") {
        Write-Error "$VenDir exists but not executable; refusing to overwrite."
        exit 1
    }

    $systemPy = Find-SystemPython
    & $systemPy -m venv $VenDir
    & $Py -m pip install --upgrade pip

    if (Test-Path "$Root\requirements.txt") {
        & $Py -m pip install -r "$Root\requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    }

    Doctor
}

function Ensure-Venv {
    if (Test-Path $Py) { return }
    Write-Error "No usable virtual environment at $VenDir. Run '.\dev.ps1 bootstrap'."
    exit 1
}

switch ($Command) {
    "doctor"   { Doctor }
    "bootstrap"{ Bootstrap }
    "python"   { Ensure-Venv; & $Py @RemainingArgs }
    "pip"      { Ensure-Venv; & $Py -m pip @RemainingArgs }
    "pytest"   { Ensure-Venv; & $Py -m pytest @RemainingArgs }
    default    { Write-Error "Unknown: $Command. Usage: .\dev.ps1 {doctor|bootstrap|python|pip|pytest} ..."; exit 2 }
}
