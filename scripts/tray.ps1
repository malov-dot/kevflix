# KevFLIX tray icon (Windows PowerShell 5.1, no extra software).
# Shows in the notification area whether the PC worker answers: orange K = running, grey K = stopped. Right-click: open
# KevFLIX PC, start or stop the worker, exit the icon. Double-click opens KevFLIX PC. The worker keeps running when the
# icon is closed. Started at login by the "KevFLIX Tray" scheduled task (relay/startup_tasks.py).
#   -SelfTest : create the icon once, check the worker, log the result and exit (used by the tests).
param([switch]$SelfTest)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$root = Split-Path -Parent $PSScriptRoot

function Get-ProfileRoot {
    if ($env:KEVFLIX_PROFILE) { return $env:KEVFLIX_PROFILE }
    $base = Join-Path $env:LOCALAPPDATA 'KevFLIX'
    $pointer = Join-Path $base 'profile-location.json'
    if (Test-Path -LiteralPath $pointer) {
        try {
            $value = (Get-Content -Raw -Encoding UTF8 -LiteralPath $pointer | ConvertFrom-Json).profile_root
            if ($value) { return [Environment]::ExpandEnvironmentVariables([string]$value) }
        } catch {}
    }
    return (Join-Path $base 'profile')
}
$profileRoot = Get-ProfileRoot
$logFile = Join-Path (Join-Path $profileRoot 'logs') 'tray.log'
function Write-Log([string]$text) {
    try {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $logFile) | Out-Null
        Add-Content -LiteralPath $logFile -Value ('{0} {1}' -f (Get-Date -Format s), $text)
    } catch {}
}

# one icon per login session
$mutexName = if ($SelfTest) { 'Local\KevFLIXTraySelfTest' } else { 'Local\KevFLIXTray' }
$mutex = New-Object System.Threading.Mutex($false, $mutexName)
if (-not $mutex.WaitOne(0)) { Write-Log 'another tray icon is already running; exiting'; exit 0 }

function Get-Interpreter([string]$name) {
    foreach ($candidate in @((Join-Path $root ".venv\Scripts\$name"), (Join-Path $root "runtime\python\$name"))) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    return $null
}
function Get-WorkerUrl {
    try {
        $config = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $profileRoot 'relay-config.json') | ConvertFrom-Json
        if ($config.public_url) { return ([string]$config.public_url).TrimEnd('/') }
    } catch {}
    return 'http://127.0.0.1:8787'
}
function Get-WorkerInfo {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri ((Get-WorkerUrl) + '/tv-api/version') -TimeoutSec 2
        $info = $response.Content | ConvertFrom-Json
        if ($info.product -eq 'KevFLIX') { return $info }
    } catch {}
    return $null
}

function New-KIcon([System.Drawing.Color]$color) {
    $bitmap = New-Object System.Drawing.Bitmap 32, 32
    $g = [System.Drawing.Graphics]::FromImage($bitmap)
    $g.SmoothingMode = 'AntiAlias'
    $g.TextRenderingHint = 'AntiAliasGridFit'
    $g.Clear([System.Drawing.Color]::Transparent)
    $back = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(255, 23, 24, 29))
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $path.AddArc(0, 0, 10, 10, 180, 90); $path.AddArc(22, 0, 10, 10, 270, 90)
    $path.AddArc(22, 22, 10, 10, 0, 90); $path.AddArc(0, 22, 10, 10, 90, 90); $path.CloseFigure()
    $g.FillPath($back, $path)
    $font = New-Object System.Drawing.Font('Segoe UI', 19, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
    $format = New-Object System.Drawing.StringFormat
    $format.Alignment = 'Center'; $format.LineAlignment = 'Center'
    $g.DrawString('K', $font, (New-Object System.Drawing.SolidBrush $color), (New-Object System.Drawing.RectangleF 0, 0, 32, 33), $format)
    $g.Dispose()
    return [System.Drawing.Icon]::FromHandle($bitmap.GetHicon())
}
$iconRunning = New-KIcon ([System.Drawing.Color]::FromArgb(255, 239, 173, 53))
$iconStopped = New-KIcon ([System.Drawing.Color]::FromArgb(255, 138, 141, 150))

$notify = New-Object System.Windows.Forms.NotifyIcon
$notify.Icon = $iconStopped
$notify.Text = 'KevFLIX - checking...'
$menu = New-Object System.Windows.Forms.ContextMenuStrip
$status = $menu.Items.Add('KevFLIX'); $status.Enabled = $false
[void]$menu.Items.Add('-')
$open = $menu.Items.Add('Open KevFLIX PC')
$start = $menu.Items.Add('Start server')
$stop = $menu.Items.Add('Stop server')
[void]$menu.Items.Add('-')
$exit = $menu.Items.Add('Exit tray icon')
$notify.ContextMenuStrip = $menu

$script:running = $null
function Update-State([bool]$announce) {
  try {
    $info = Get-WorkerInfo
    $now = [bool]$info
    if ($now) {
        $notify.Icon = $iconRunning
        $notify.Text = ('KevFLIX - running ({0})' -f $info.version)
        $status.Text = ('Server running - version {0}' -f $info.version)
    } else {
        $notify.Icon = $iconStopped
        $notify.Text = 'KevFLIX - server stopped'
        $status.Text = 'Server stopped'
    }
    $start.Enabled = -not $now
    $stop.Enabled = $now
    if ($null -ne $script:running -and $script:running -ne $now) {
        Write-Log ('server state changed: ' + $(if ($now) { 'running' } else { 'stopped' }))
        if ($announce) {
            $notify.BalloonTipTitle = 'KevFLIX'
            $notify.BalloonTipText = $(if ($now) { 'The server is running.' } else { 'The server stopped.' })
            $notify.ShowBalloonTip(3000)
        }
    }
    $script:running = $now
    return $now
  } catch {
    Write-Log ('status update failed: ' + $_.Exception.Message)
    return $false
  }
}

function Start-Hidden([string]$file, [string]$arguments) {
    Start-Process -FilePath $file -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden
}
$open.add_Click({
  try {
    $py = Get-Interpreter 'pythonw.exe'
    if ($py) { Start-Hidden $py ('"{0}"' -f (Join-Path $root 'scripts\open_pc.py')); Write-Log 'open requested' }
  } catch { Write-Log ('open failed: ' + $_.Exception.Message) }
})
$notify.add_DoubleClick({ $open.PerformClick() })
$start.add_Click({
  try {
    $py = Get-Interpreter 'pythonw.exe'
    if ($py) { Start-Hidden $py ('"{0}"' -f (Join-Path $root 'scripts\run_worker.py')); Write-Log 'start requested' }
  } catch { Write-Log ('start failed: ' + $_.Exception.Message) }
})
$stop.add_Click({
  try {
    $py = Get-Interpreter 'python.exe'
    if ($py) { Start-Hidden $py ('-I -B "{0}"' -f (Join-Path $root 'scripts\stop_worker.py')); Write-Log 'stop requested' }
  } catch { Write-Log ('stop failed: ' + $_.Exception.Message) }
})

Write-Log ('tray starting (install {0}, profile {1})' -f $root, $profileRoot)
$notify.Visible = $true
[void](Update-State $false)

if ($SelfTest) {
    Write-Log ('selftest ok: worker answering = {0}; icon visible = {1}' -f $script:running, $notify.Visible)
    $notify.Visible = $false; $notify.Dispose(); $menu.Dispose()
    exit 0
}

# nothing in a handler may end the icon: log it and carry on
$ErrorActionPreference = 'Continue'
[System.Windows.Forms.Application]::SetUnhandledExceptionMode([System.Windows.Forms.UnhandledExceptionMode]::CatchException)
[System.Windows.Forms.Application]::add_ThreadException({ param($sender, $e) Write-Log ('unhandled error: ' + $e.Exception.Message) })
$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 5000
$timer.add_Tick({ [void](Update-State $true) })
$timer.Start()
$context = New-Object System.Windows.Forms.ApplicationContext
$exit.add_Click({ $timer.Stop(); $notify.Visible = $false; $notify.Dispose(); $context.ExitThread() })
try { [System.Windows.Forms.Application]::Run($context) } finally { Write-Log 'tray closed' }
