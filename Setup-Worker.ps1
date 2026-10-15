[CmdletBinding()]
param([string]$PythonLauncher = 'py', [switch]$DownloadModels, [switch]$AcceptModelTerms)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $PythonLauncher -3.11 -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 x64 is required; no global Python changes were made.' }
}
$env:PIP_REQUIRE_VIRTUALENV = 'true'
& $venvPython -m pip install --upgrade 'pip==25.3'
if ($LASTEXITCODE -ne 0) { throw 'Could not prepare isolated pip.' }
& $venvPython -m pip install 'torch==2.11.0' 'torchaudio==2.11.0' --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Could not install the pinned CUDA 12.8 PyTorch packages.' }
& $venvPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed; no service was started.' }
New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot 'state') -Force | Out-Null
& $venvPython -m pip freeze | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'state\installed-requirements.txt') -Encoding utf8
if ($DownloadModels) {
    if (-not $AcceptModelTerms) { throw 'Read THIRD-PARTY.md and accept the model licenses/gated repository terms before using -AcceptModelTerms.' }
    & $venvPython (Join-Path $PSScriptRoot 'scripts\provision_models.py') --accept-model-terms
    if ($LASTEXITCODE -ne 0) { throw 'Model provisioning failed; resume it after correcting the reported issue.' }
}
Write-Output 'Dependencies prepared. No tests, inference, Sem changes or services were started.'
