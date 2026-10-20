[CmdletBinding()]
param([string]$PythonLauncher = 'py', [string]$PythonExecutable = '', [switch]$DownloadModels, [switch]$AcceptModelTerms,
      # Vox actor engine (explicit): an existing folder holding voxcpm2-actor-q4_k.gguf, and a folder holding
      # audiocpp.dll + the approved ggml/CUDA/MSVC DLLs. Every sha256 is verified; files are copied into the
      # KevFLIX profile unless -VoxReferenceInPlace. Nothing is downloaded and nothing is taken from Sem's folders.
      [string]$VoxModelSource = '', [string]$VoxRuntimeSource = '', [switch]$VoxReferenceInPlace)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    if (-not $PythonExecutable -and (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'runtime\python\python.exe'))) {
        $PythonExecutable = Join-Path $PSScriptRoot 'runtime\python\python.exe'
    }
    if ($PythonExecutable) { & $PythonExecutable -m venv (Join-Path $PSScriptRoot '.venv') }
    else { & $PythonLauncher -3.11 -m venv (Join-Path $PSScriptRoot '.venv') }
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
if ($VoxModelSource -or $VoxRuntimeSource) {
    $voxArgs = @((Join-Path $PSScriptRoot 'scripts\provision_vox.py'))
    if ($VoxModelSource) { $voxArgs += @('--model-source', $VoxModelSource) }
    if ($VoxRuntimeSource) { $voxArgs += @('--runtime-source', $VoxRuntimeSource) }
    if ($VoxReferenceInPlace) { $voxArgs += '--reference-in-place' }
    & $venvPython @voxArgs
    if ($LASTEXITCODE -ne 0) { throw 'Vox actor engine provisioning failed (see the message above); dubbing cannot use Vox until it succeeds.' }
}
Write-Output 'Dependencies prepared. No tests, inference, Sem changes or services were started.'
