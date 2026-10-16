[CmdletBinding()]
param([string]$SevenZip = 'C:\Program Files\7-Zip\7z.exe')
$ErrorActionPreference = 'Stop'
$version = '9.0.2'
$expected = '4d2060a8b34a940aa47d785142055bb92a63053781e55f2ace4546edd519a8f5'
$url = 'https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-9.0.2-full_build-shared.7z'
$downloads = Join-Path $PSScriptRoot 'setup-records\downloads'
$destination = Join-Path $PSScriptRoot 'tools'
$archive = Join-Path $downloads 'ffmpeg-9.0.2-full_build-shared.7z'
New-Item -ItemType Directory -Path $downloads,$destination -Force | Out-Null
if (-not (Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest -Uri $url -OutFile $archive
}
$digest = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($digest -ne $expected) { throw 'Downloaded archive differs from the pinned publisher checksum; nothing extracted.' }
$installed = Join-Path $destination 'ffmpeg-9.0.2-full_build-shared'
if (-not (Test-Path -LiteralPath $installed)) {
    if (Test-Path -LiteralPath $SevenZip) { & $SevenZip x $archive ('-o'+$destination) -y }
    else {
        $extractPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $extractPython)) { $extractPython = Join-Path $PSScriptRoot 'runtime\python\python.exe' }
        if (-not (Test-Path -LiteralPath $extractPython)) { throw 'Install KevFLIX PC or prepare its virtual environment first.' }
        $extractLibraries = Join-Path $downloads 'extractor-libraries'
        & $extractPython -m pip install --target $extractLibraries 'py7zr==1.0.0'
        if ($LASTEXITCODE -ne 0) { throw 'Could not install the isolated archive extractor.' }
        & $extractPython -c 'import sys; sys.path.insert(0,sys.argv[1]); import py7zr; a=py7zr.SevenZipFile(sys.argv[2]); a.extractall(sys.argv[3]); a.close()' $extractLibraries $archive $destination
    }
    if ($LASTEXITCODE -ne 0) { throw 'FFmpeg extraction failed.' }
}
[ordered]@{
    version=$version; source=$url; sha256=$digest; directory=$installed
    license='GPL-3.0, Gyan full shared build; notices included'
    executed=$false
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'setup-records\ffmpeg-install.json') -Encoding utf8
Write-Output 'Private FFmpeg files installed. No FFmpeg command or media probe was run.'
