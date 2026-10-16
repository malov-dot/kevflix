[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$desktop = [Environment]::GetFolderPath('Desktop')
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut((Join-Path $desktop 'KevFLIX PC.lnk'))
$shortcut.TargetPath = Join-Path $env:SystemRoot 'System32\cmd.exe'
$shortcut.Arguments = '/c ""' + (Join-Path $PSScriptRoot 'Open-KevFLIX-PC.cmd') + '""'
$shortcut.WorkingDirectory = $PSScriptRoot
$shortcut.WindowStyle = 7
$shortcut.Description = 'KevFLIX PC control panel and TV pairing'
$shortcut.IconLocation = (Join-Path $env:SystemRoot 'System32\shell32.dll') + ',14'
$shortcut.Save()
Write-Output 'KevFLIX PC shortcut created on the desktop.'
