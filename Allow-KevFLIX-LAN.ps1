#Requires -RunAsAdministrator
[CmdletBinding()]
param([string]$ConfigPath = '')
$ErrorActionPreference = 'Stop'
if (-not $ConfigPath) {
    # The user's settings live in the KevFLIX profile (same rule as relay/app_runtime/paths.py).
    if ($env:KEVFLIX_PROFILE) { $profileRoot = $env:KEVFLIX_PROFILE }
    else {
        $pointer = Join-Path $env:LOCALAPPDATA 'KevFLIX\profile-location.json'
        if (Test-Path -LiteralPath $pointer) { $profileRoot = (Get-Content -LiteralPath $pointer -Raw | ConvertFrom-Json).profile_root }
        else { $profileRoot = Join-Path $env:LOCALAPPDATA 'KevFLIX\profile' }
    }
    $ConfigPath = Join-Path $profileRoot 'relay-config.json'
    if (-not (Test-Path -LiteralPath $ConfigPath)) { $ConfigPath = Join-Path $PSScriptRoot 'relay-config.json' }
}
$cfg = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$address = [Net.IPAddress]::Parse([string]$cfg.bind)
$octets = $address.GetAddressBytes()
$private = $address.AddressFamily -eq [Net.Sockets.AddressFamily]::InterNetwork -and (
    $octets[0] -eq 10 -or
    ($octets[0] -eq 192 -and $octets[1] -eq 168) -or
    ($octets[0] -eq 172 -and $octets[1] -ge 16 -and $octets[1] -le 31))
if (-not $private) { throw 'Configure bind with the PC private LAN IPv4 address before adding this rule.' }
$port = [int]$cfg.port
if ($port -lt 1024 -or $port -gt 65535) { throw 'Worker port is outside the allowed range.' }
$interface = Get-NetIPAddress -IPAddress $address.IPAddressToString -ErrorAction Stop
$profile = Get-NetConnectionProfile -InterfaceIndex $interface.InterfaceIndex -ErrorAction Stop
if ($profile.NetworkCategory.ToString() -ne 'Private') {
    throw 'The worker network is not marked Private. Review that Windows network setting first; this helper does not change it.'
}
$listener = @(Get-NetTCPConnection -LocalAddress $address.IPAddressToString -LocalPort $port -State Listen -ErrorAction Stop)
if ($listener.Count -ne 1) { throw 'Start exactly one KevFLIX worker before configuring its firewall rule.' }
$process = Get-CimInstance Win32_Process -Filter ('ProcessId='+$listener[0].OwningProcess)
if ($process.CommandLine -notmatch '(?i)(-m\s+relay\.server(?:\s|$)|scripts[\\/]serve\.py)') {
    throw 'The listening process is not the KevFLIX relay; nothing changed.'
}
$program = [IO.Path]::GetFullPath([string]$process.ExecutablePath)
if (-not (Test-Path -LiteralPath $program -PathType Leaf)) { throw 'The worker executable could not be identified.' }
$ruleName = 'KevFLIX-Worker-Private-LAN'
$existing = Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output 'The named KevFLIX rule already exists. It was left unchanged.'
    exit 0
}
$rule = @{
    Name=$ruleName; DisplayName='KevFLIX worker - private LAN only'
    Direction='Inbound'; Action='Allow'; Enabled='True'; Profile='Private'
    Protocol='TCP'; LocalPort=$port; LocalAddress=$address.IPAddressToString
    RemoteAddress='LocalSubnet'; Program=$program; EdgeTraversalPolicy='Block'
}
New-NetFirewallRule @rule | Out-Null
Write-Output 'KevFLIX is allowed on its configured port from the private local subnet.'
Write-Output 'No other firewall rules, network profiles, Sem services or Plex files were changed.'
