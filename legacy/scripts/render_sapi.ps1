<#
Windows desktop speech rendering for Movie Dub's dub stage (scripts/generate-dub.py).

Uses System.Speech (SAPI 5) voices already installed on this machine, so dubbed
speech is generated on local hardware with no network and no mock audio.

Modes:
  -Mode voices                  print one enabled voice name per line; exit 3 if none
  -Mode render -JobsFile f.json render each job {"out","voice","text","pitch","rate"}

Each line is rendered by its own synthesizer instance whose output target is a
WAV file, so the file is flushed and closed before the caller measures it.
Exit codes: 0 ok, 3 no SAPI voices, 4 render failed.
#>
param(
    [string]$Mode = 'voices',
    [string]$JobsFile
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech

function Get-SapiVoices {
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    try { , @($s.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo.Name }) }
    finally { $s.Dispose() }
}

$voices = Get-SapiVoices

if ($Mode -eq 'voices') {
    if ($voices.Count -eq 0) { exit 3 }
    $voices | ForEach-Object { Write-Output $_ }
    exit 0
}

if ($voices.Count -eq 0) { Write-Error 'no enabled System.Speech voices installed'; exit 3 }

# PS 5.1 emits the deserialized root as ONE object, so @(...) would wrap the
# whole array in another array; unwrap explicitly.
$parsed = Get-Content -Raw -Encoding UTF8 $JobsFile | ConvertFrom-Json
$jobs = @()
foreach ($item in $parsed) { $jobs += $item }
if ($jobs.Count -eq 0) { Write-Error 'no render jobs given'; exit 4 }

foreach ($job in $jobs) {
    if (-not ($voices -contains $job.voice)) { Write-Error ('voice not installed: ' + $job.voice); exit 4 }
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    try {
        $synth.SelectVoice($job.voice)
        $synth.SetOutputToWaveFile($job.out)
        $safe = [System.Security.SecurityElement]::Escape($job.text)
        $rate = [int]$job.rate
        $pitch = [int]$job.pitch
        # SSML treats a bare percentage as an ABSOLUTE speed, so a "faster"
        # line must be written with an explicit sign: +60% is 1.6x normal.
        $rateText = ('+{0}%' -f $rate)
        $pitchText = ('{0}{1}st' -f ($(if ($pitch -lt 0) { '' } else { '+' })), $pitch)
        $ssml = "<speak version='1.0' xml:lang='en-US'><prosody pitch='" + $pitchText + "' rate='" + $rateText + "'>" + $safe + "</prosody></speak>"
        $synth.SpeakSsml($ssml)
    }
    finally {
        $synth.Dispose()   # disposes the output stream, flushing the WAV to disk
    }
    if (-not (Test-Path $job.out)) { Write-Error ('renderer produced no file: ' + $job.out); exit 4 }
}
exit 0
