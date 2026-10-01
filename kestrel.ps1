<#
.SYNOPSIS
  Kestrel launcher for Windows (behaves like kestrel.sh on Linux).
.DESCRIPTION
  Builds the kyty_emulator command line from presets/<preset>.ini, kestrel.ini and
  kestrel.local.ini, then runs it (or prints it with -DryRun, or records a paste-able
  report with -Report). Works on Windows PowerShell 5.1 and pwsh.
.EXAMPLE
  kestrel.bat -Game D:\Games\PS5\PPSA01325
  kestrel.bat -DryRun -Preset ds-high
  kestrel.bat -Report -Seconds 120
#>
param(
    [switch]$DryRun,
    [switch]$Report,
    [int]$Seconds = 300,
    [string]$Preset = '',
    [string]$Config = '',
    [string]$Game = '',
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

# ---------------------------------------------------------------- output helpers

$script:ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$script:Warnings = New-Object System.Collections.Generic.List[string]
$script:Inv = [System.Globalization.CultureInfo]::InvariantCulture

# Plain stdout so output survives redirection (Write-Host does not on 5.1).
function Say([string]$Text) { [Console]::Out.WriteLine($Text) }

function Fail([string]$Text) {
    [Console]::Error.WriteLine('kestrel: error: ' + $Text)
    exit 2
}

function Warn([string]$Text) {
    $script:Warnings.Add('warning: ' + $Text)
}

function Show-Usage {
    Say 'Usage: kestrel.bat [-DryRun] [-Report] [-Seconds N] [-Preset NAME] [-Config FILE] [-Game DIR] [-Help]'
    Say ''
    Say '  -DryRun        print the emulator command and exit without launching'
    Say '  -Report        run for -Seconds (default 300), then print a summary block to paste back'
    Say '  -Seconds N     report length in seconds (default 300)'
    Say '  -Preset NAME   preset from presets\ (overrides [launcher] preset)'
    Say '  -Config FILE   settings file to use instead of kestrel.ini'
    Say '  -Game DIR      game folder (overrides [game] path)'
    Say '  -Help          this text'
    Say ''
    Say 'Settings live in kestrel.ini; put your own in kestrel.local.ini (ignored by git).'
    Say 'See docs\QUICKSTART.md for setup.'
}

# ---------------------------------------------------------------- INI handling

# Parse an INI file into a hashtable keyed by "section.key" (lower case).
# Comments are whole lines starting with ; or #; values are never unquoted or cut.
function Read-Ini([string]$Path) {
    $map = @{}
    $section = ''
    foreach ($raw in [System.IO.File]::ReadAllLines($Path)) {
        $line = $raw.TrimEnd("`r").Trim()
        if ($line -eq '' -or $line.StartsWith(';') -or $line.StartsWith('#')) { continue }
        if ($line -match '^\[(.*)\]$') { $section = $Matches[1].Trim().ToLowerInvariant(); continue }
        $eq = $line.IndexOf('=')
        if ($eq -lt 1) { continue }
        $key = $line.Substring(0, $eq).Trim().ToLowerInvariant()
        $map[$section + '.' + $key] = $line.Substring($eq + 1).Trim()
    }
    return $map
}

# Overlay: a key from $Top wins only when its value is non-empty.
function Merge-Ini([hashtable]$Base, [hashtable]$Top) {
    foreach ($k in $Top.Keys) {
        if ($Top[$k] -ne '') { $Base[$k] = $Top[$k] }
    }
}

function Get-Val([string]$Key) {
    if ($script:Settings.ContainsKey($Key)) { return [string]$script:Settings[$Key] }
    return ''
}

# on/true/yes/1 -> $true; off/false/no/0/empty -> $false; anything else is an error.
function Get-Switch([string]$Key) {
    $v = (Get-Val $Key).ToLowerInvariant()
    if ($v -in @('on', 'true', 'yes', '1')) { return $true }
    if ($v -in @('', 'off', 'false', 'no', '0')) { return $false }
    Fail ("invalid value '" + $v + "' for " + $Key + ' (use on/off)')
}

# A number-like setting where empty and 0 mean "off". Returns '' when off.
function Get-NonZero([string]$Key) {
    $v = Get-Val $Key
    $n = 0.0
    if ($v -eq '') { return '' }
    if ([double]::TryParse($v, [System.Globalization.NumberStyles]::Float, $script:Inv, [ref]$n) -and $n -eq 0) { return '' }
    return $v
}

function Get-PresetNames {
    $dir = Join-Path $script:ScriptDir 'presets'
    if (-not (Test-Path -LiteralPath $dir)) { return @() }
    return @(Get-ChildItem -LiteralPath $dir -Filter '*.ini' | Sort-Object Name | ForEach-Object { $_.BaseName })
}

# Fills $script:Settings and $script:ConfigFiles; returns the preset name used.
function Load-Settings {
    $cfg = $Config
    if ($cfg -eq '') { $cfg = Join-Path $script:ScriptDir 'kestrel.ini' }
    elseif (-not [System.IO.Path]::IsPathRooted($cfg)) { $cfg = [System.IO.Path]::GetFullPath($cfg) }
    if (-not (Test-Path -LiteralPath $cfg -PathType Leaf)) { Fail ('config file not found: ' + $cfg) }
    $main = Read-Ini $cfg

    $local = Join-Path $script:ScriptDir 'kestrel.local.ini'
    $localIni = $null
    if (Test-Path -LiteralPath $local -PathType Leaf) { $localIni = Read-Ini $local }

    # Preset: -Preset, else kestrel.local.ini, else kestrel.ini (same order as the values).
    $name = $Preset
    if ($name -eq '' -and $null -ne $localIni -and $localIni.ContainsKey('launcher.preset')) { $name = $localIni['launcher.preset'] }
    if ($name -eq '' -and $main.ContainsKey('launcher.preset')) { $name = $main['launcher.preset'] }
    if ($name -eq '') { $name = 'default' }
    $presetFile = Join-Path (Join-Path $script:ScriptDir 'presets') ($name + '.ini')
    if ($name -notmatch '^[A-Za-z0-9._-]+$' -or -not (Test-Path -LiteralPath $presetFile -PathType Leaf)) {
        Fail ("unknown preset '" + $name + "'. Available presets: " + ((Get-PresetNames) -join ', '))
    }

    $script:Settings = Read-Ini $presetFile
    $script:ConfigFiles = @($presetFile, $cfg)
    Merge-Ini $script:Settings $main
    if ($null -ne $localIni) {
        Merge-Ini $script:Settings $localIni
        $script:ConfigFiles += $local
    }
    if ($Game -ne '') { $script:Settings['game.path'] = $Game }
    return $name
}

# ---------------------------------------------------------------- emulator

function Find-Emulator {
    $cfg = Get-Val 'launcher.emulator'
    if ($cfg -ne '') {
        if (-not [System.IO.Path]::IsPathRooted($cfg)) { $cfg = Join-Path $script:ScriptDir $cfg }
        if (-not (Test-Path -LiteralPath $cfg -PathType Leaf)) { Fail ('emulator not found: ' + $cfg + ' (from [launcher] emulator)') }
        return [System.IO.Path]::GetFullPath($cfg)
    }
    foreach ($rel in @('_Build\windows\install\kyty_emulator.exe', '_Build\windows\kyty_emulator.exe')) {
        $p = Join-Path $script:ScriptDir $rel
        if (Test-Path -LiteralPath $p -PathType Leaf) { return $p }
    }
    if ($DryRun) {
        Warn 'emulator not built yet; using placeholder path (see docs/QUICKSTART.md)'
        return ''
    }
    Fail 'emulator not found: build first, see docs/QUICKSTART.md'
}

# Quote one argument per CommandLineToArgvW rules (only when needed).
function Quote-Arg([string]$Arg) {
    if ($Arg -ne '' -and $Arg -notmatch '[\s"]') { return $Arg }
    $s = $Arg -replace '(\\*)"', '$1$1\"'
    $s = $s -replace '(\\+)$', '$1$1'
    return '"' + $s + '"'
}

function Join-Args([object[]]$ArgList) {
    return (@($ArgList | ForEach-Object { Quote-Arg ([string]$_) }) -join ' ')
}

# Run "<emu> --help" (15 s limit). Returns the combined text, or $null if it failed.
function Get-HelpText([string]$Emu) {
    $out = [System.IO.Path]::GetTempFileName()
    $err = [System.IO.Path]::GetTempFileName()
    try {
        $p = Start-Process -FilePath $Emu -ArgumentList '--help' -WorkingDirectory (Split-Path -Parent $Emu) `
            -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -PassThru
        if (-not $p.WaitForExit(15000)) {
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
            return $null
        }
        return ([System.IO.File]::ReadAllText($out) + "`n" + [System.IO.File]::ReadAllText($err))
    } catch {
        return $null
    } finally {
        Remove-Item -LiteralPath $out, $err -Force -ErrorAction SilentlyContinue
    }
}

# True if the help text lists $Flag as a whole word followed by whitespace.
function Test-Flag([string]$Flag) {
    if ($script:AssumeAll) { return $true }
    if ($null -eq $script:HelpText) { return $false }
    return [regex]::IsMatch($script:HelpText, '(?<![\w-])' + [regex]::Escape($Flag) + '(?=\s)')
}

function Get-Banner {
    if ($script:AssumeAll -or $null -eq $script:HelpText) { return 'unknown' }
    foreach ($l in ($script:HelpText -split "`r?`n")) {
        if ($l.Trim() -ne '') { return $l.Trim() }
    }
    return 'unknown'
}

# ---------------------------------------------------------------- argument mapping

function Need-Flag([string]$IniKey, [string]$Flag, [string]$Ending) {
    Warn ($IniKey + ' needs ' + $Flag + ' (coming soon, not in this build)' + $Ending)
}

# Builds the emulator argument list from the settings. $ReportDir switches on report mode.
function Build-Args([string]$GamePath, [string]$ReportDir) {
    $a = New-Object System.Collections.Generic.List[string]
    $inReport = ($ReportDir -ne '')

    $a.Add('--game'); $a.Add($GamePath)
    if (Get-Switch 'game.redzone') {
        if (Test-Flag '--redzone') { $a.Add('--redzone') } else { Warn "game.redzone needs --redzone (not in this build); skipped" }
    }

    $v = Get-Val 'video.render_scale'; if ($v -ne '') { $a.Add('--render-scale'); $a.Add($v) }
    if (Get-Switch 'video.fullscreen') { $a.Add('--fullscreen') }
    $v = Get-Val 'video.width';        if ($v -ne '') { $a.Add('--screen-width'); $a.Add($v) }
    $v = Get-Val 'video.height';       if ($v -ne '') { $a.Add('--screen-height'); $a.Add($v) }
    $v = Get-Val 'video.rt_mode';      if ($v -ne '') { $a.Add('--rt-mode'); $a.Add($v) }
    $v = Get-Val 'video.vertex_fetch'; if ($v -ne '') { $a.Add('--vertex-fetch'); $a.Add($v) }
    $v = Get-Val 'video.present_mode'; if ($v -ne '') { $a.Add('--present-mode'); $a.Add($v) }

    # Upscaler: linear is the emulator's built-in behaviour, so only pass it when supported.
    $up = (Get-Val 'video.upscaler').ToLowerInvariant()
    $hasUp = Test-Flag '--upscaler'
    if ($up -eq 'fsr1') {
        if ($hasUp) {
            $a.Add('--upscaler'); $a.Add('fsr1')
            $v = Get-Val 'video.fsr_sharpness'
            if ($v -ne '' -and (Test-Flag '--fsr-sharpness')) { $a.Add('--fsr-sharpness'); $a.Add($v) }
        } else {
            Need-Flag 'video.upscaler=fsr1' '--upscaler' '; skipped'
        }
    } elseif (($up -eq 'linear' -or $up -eq '') -and $hasUp) {
        $a.Add('--upscaler'); $a.Add('linear')
    } elseif ($up -ne 'linear' -and $up -ne '') {
        Fail ("invalid value '" + $up + "' for video.upscaler (use linear or fsr1)")
    }

    # Game-speed patch: its frame cap is only applied together with the patch.
    $cap = Get-NonZero 'speed.frame_cap'
    if (Get-Switch 'speed.game_speed_patch') {
        $pn = Get-Val 'speed.patch_name'
        if ($pn -eq '') {
            Warn 'speed.game_speed_patch is on but there is no game-speed patch for this preset; skipped'
        } elseif (Test-Flag '--patch') {
            $a.Add('--patch'); $a.Add($pn)
            $pc = Get-NonZero 'speed.patch_frame_cap'
            if ($cap -eq '' -and $pc -ne '') { $cap = $pc }
        } else {
            Warn 'speed.game_speed_patch needs --patch (not in this build); skipped (patch_frame_cap not applied)'
        }
    }
    if ($cap -ne '') { $a.Add('--frame-cap'); $a.Add($cap) }

    if (Get-Switch 'debug.overlay') {
        if (Test-Flag '--perf-overlay') { $a.Add('--perf-overlay') }
        else { Need-Flag 'debug.overlay' '--perf-overlay' '; the window title shows fps instead' }
    }

    # Frame-time log: report mode forces its own file and ignores the ini value.
    if ($inReport) {
        if (Test-Flag '--frame-time-log') { $a.Add('--frame-time-log'); $a.Add((Join-Path $ReportDir 'frametimes.txt')) }
    } else {
        $v = Get-Val 'debug.frame_time_log'
        if ($v -ne '') {
            if (Test-Flag '--frame-time-log') { $a.Add('--frame-time-log'); $a.Add($v) }
            else { Need-Flag 'debug.frame_time_log' '--frame-time-log' '; skipped' }
        }
    }

    if ($inReport) {
        $a.Add('--fps-log'); $a.Add('1')
    } else {
        $v = Get-NonZero 'debug.fps_log'; if ($v -ne '') { $a.Add('--fps-log'); $a.Add($v) }
    }

    # Guest log: report mode always writes it inside the report folder.
    $guestLog = ''
    if ($inReport) {
        $guestLog = Join-Path $ReportDir 'guest.log'
    } elseif (Get-Switch 'debug.guest_log') {
        $logs = Join-Path $script:ScriptDir 'kestrel-logs'
        if (-not $DryRun) { [void](New-Item -ItemType Directory -Force -Path $logs) }
        $guestLog = Join-Path $logs ((Get-Date -Format 'yyyyMMdd-HHmmss') + '.guest.log')
    }
    if ($guestLog -ne '') { $a.Add('--printf-direction'); $a.Add('File'); $a.Add('--printf-output-file'); $a.Add($guestLog) }

    $extra = Get-Val 'debug.extra_args'
    if ($extra -ne '') { foreach ($t in ($extra -split '\s+')) { if ($t -ne '') { $a.Add($t) } } }
    return ,$a.ToArray()
}

# Report-safe copy of the arguments: --game, log and report paths shrink to their last name.
function Redact-Args([string[]]$ArgList) {
    $out = New-Object System.Collections.Generic.List[string]
    for ($i = 0; $i -lt $ArgList.Length; $i++) {
        $x = $ArgList[$i]
        if ($i -gt 0 -and $ArgList[$i - 1] -in @('--game', '--printf-output-file', '--frame-time-log')) {
            $x = Split-Path -Leaf ($x.TrimEnd('\', '/'))
        }
        $out.Add($x)
    }
    return ,$out.ToArray()
}

# ---------------------------------------------------------------- report statistics

function F1([double]$x) { return $x.ToString('0.0', $script:Inv) }

# $Dts = frame times in ms, $TotalSec = wall time they span.
function Format-FrameStats([string]$Label, [System.Collections.Generic.List[double]]$Dts, [double]$TotalSec) {
    $n = $Dts.Count
    if ($n -lt 1 -or $TotalSec -le 0) { return ('  ' + $Label + ': no frames') }
    $sorted = $Dts.ToArray()
    [Array]::Sort($sorted)
    [Array]::Reverse($sorted)   # slowest first
    $slow = 0
    foreach ($d in $sorted) { if ($d -gt 33.3) { $slow++ } }
    $r = New-Object System.Collections.Generic.List[string]
    foreach ($pct in @(0.01, 0.10)) {
        $k = [Math]::Max(1, [int][Math]::Ceiling($n * $pct))
        $sum = 0.0
        for ($i = 0; $i -lt $k; $i++) { $sum += $sorted[$i] }
        $r.Add((F1 (1000.0 / ($sum / $k))))
    }
    return ('  ' + $Label + ': frames ' + $n + ', avg fps ' + (F1 ($n / $TotalSec)) + ', 1% low ' + $r[0] +
        ', 10% low ' + $r[1] + ', max frame ' + (F1 $sorted[0]) + ' ms, >33.3 ms ' + (F1 (100.0 * $slow / $n)) + '%')
}

# frametimes.txt: one "frame time_us" per line. Returns report lines, or $null if unusable.
function Get-FrameTimeStats([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    $ts = New-Object System.Collections.Generic.List[long]
    foreach ($l in [System.IO.File]::ReadAllLines($Path)) {
        $f = $l.Trim() -split '\s+'
        $t = 0L
        if ($f.Length -ge 2 -and [long]::TryParse($f[1], [ref]$t)) { $ts.Add($t) }
    }
    if ($ts.Count -lt 3) { return $null }
    $last = $ts[$ts.Count - 1]
    $winStart = 0
    while ($winStart -lt $ts.Count - 1 -and $ts[$winStart] -lt ($last - 60000000L)) { $winStart++ }
    $all = New-Object System.Collections.Generic.List[double]
    $win = New-Object System.Collections.Generic.List[double]
    for ($i = 1; $i -lt $ts.Count; $i++) {
        $d = ($ts[$i] - $ts[$i - 1]) / 1000.0
        $all.Add($d)
        if ($i -gt $winStart) { $win.Add($d) }
    }
    return @(
        (Format-FrameStats 'whole run' $all (($last - $ts[0]) / 1e6)),
        (Format-FrameStats 'last 60 s ' $win (($last - $ts[$winStart]) / 1e6))
    )
}

# Fallback: per-second "fps: N frame: N t=Ns" samples from the guest log.
function Get-FpsLogStats([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    $fps = New-Object System.Collections.Generic.List[double]
    $lastFrame = ''
    foreach ($l in [System.IO.File]::ReadAllLines($Path)) {
        if ($l -match '^fps: (\d+) frame: (\d+) t=(\d+)s') {
            $fps.Add([double]$Matches[1]); $lastFrame = $Matches[2]
        }
    }
    if ($fps.Count -eq 0) { return $null }
    $lines = @()
    $lines += '  (no frame-time log; using the guest log fps samples)'
    $sets = @(@('whole run', 0), @('last 60 samples', [Math]::Max(0, $fps.Count - 60)))
    foreach ($s in $sets) {
        $sub = $fps.GetRange($s[1], $fps.Count - $s[1])
        $m = $sub | Measure-Object -Average -Minimum -Maximum
        $lines += ('  ' + $s[0] + ': samples ' + $sub.Count + ', avg ' + (F1 $m.Average) + ' fps, min ' + (F1 $m.Minimum) + ', max ' + (F1 $m.Maximum))
    }
    $lines += ('  last frame number: ' + $lastFrame)
    return $lines
}

# ---------------------------------------------------------------- report system info

function Get-GitInfo {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if (-not $git) { return 'unknown' }
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $rev = (& git -C $script:ScriptDir rev-parse --short HEAD 2>$null | Select-Object -First 1)
        if (-not $rev) { return 'unknown' }
        $dirty = (& git -C $script:ScriptDir status --porcelain -uno 2>$null)
        if ($dirty) { return ($rev + ' (modified)') }
        return [string]$rev
    } catch { return 'unknown' } finally { $ErrorActionPreference = $old }
}

function Get-SystemLines {
    $o = @()
    try {
        $os = Get-CimInstance Win32_OperatingSystem
        $o += ('os: ' + $os.Caption.Trim() + ' ' + $os.Version)
    } catch { $o += 'os: unknown' }
    try {
        $cpu = @(Get-CimInstance Win32_Processor | ForEach-Object { $_.Name.Trim() }) -join '; '
        $o += ('cpu: ' + $cpu)
    } catch { $o += 'cpu: unknown' }
    try {
        $ram = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
        $o += ('ram: ' + (F1 ($ram / 1GB)) + ' GB')
    } catch { $o += 'ram: unknown' }
    try {
        foreach ($g in @(Get-CimInstance Win32_VideoController)) { $o += ('gpu: ' + $g.Name + ' (driver ' + $g.DriverVersion + ')') }
    } catch { $o += 'gpu: unknown' }
    return $o
}

# ---------------------------------------------------------------- report mode

function Invoke-Report([string]$Emu, [string[]]$EmuArgs, [string]$Dir, [string]$PresetName, [string]$GamePath) {
    $emuDir = Split-Path -Parent $Emu
    $log = Join-Path $Dir 'emulator.log'
    $errLog = Join-Path $Dir 'emulator.err.tmp'
    Say ('Recording ' + $Seconds + ' seconds; play normally. The game will be closed automatically.')

    $p = Start-Process -FilePath $Emu -ArgumentList (Join-Args $EmuArgs) -WorkingDirectory $emuDir `
        -RedirectStandardOutput $log -RedirectStandardError $errLog -NoNewWindow -PassThru
    $null = $p.Handle   # keep the handle so ExitCode is readable after exit (PS 5.1)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $nextNote = 30
    try {
        while ($sw.Elapsed.TotalSeconds -lt $Seconds -and -not $p.HasExited) {
            [void]$p.WaitForExit(1000)
            if ($sw.Elapsed.TotalSeconds -ge $nextNote -and -not $p.HasExited) {
                Say ('  ... ' + [int]$sw.Elapsed.TotalSeconds + ' of ' + $Seconds + ' s')
                $nextNote += 30
            }
        }
    } finally {
        # Also runs on Ctrl+C: never leave the emulator behind.
        if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; [void]$p.WaitForExit(10000) }
    }
    $actual = [int][Math]::Round($sw.Elapsed.TotalSeconds)
    $ended = 'timer'
    $early = $false
    if ($sw.Elapsed.TotalSeconds -lt $Seconds) { $early = $true; $ended = 'exited early (code ' + $p.ExitCode + ')' }

    # Merge stderr into emulator.log (Start-Process cannot share one file for both).
    if (Test-Path -LiteralPath $errLog) {
        $e = [System.IO.File]::ReadAllText($errLog)
        if ($e -ne '') { [System.IO.File]::AppendAllText($log, $e) }
        Remove-Item -LiteralPath $errLog -Force -ErrorAction SilentlyContinue
    }

    Say ''
    Say '----- Kestrel report (paste everything below) -----'
    Say ('kestrel commit: ' + (Get-GitInfo))
    Say ('emulator: ' + (Get-Banner))
    foreach ($l in (Get-SystemLines)) { Say $l }
    Say ('preset: ' + $PresetName)
    Say ('game: ' + (Split-Path -Leaf ($GamePath.TrimEnd('\', '/'))))
    Say ('args: ' + (Join-Args (Redact-Args $EmuArgs)))
    Say ('duration: requested ' + $Seconds + ' s, actual ' + $actual + ' s')
    Say ('ended: ' + $ended)
    Say 'frame stats:'
    $stats = Get-FrameTimeStats (Join-Path $Dir 'frametimes.txt')
    if ($null -eq $stats) { $stats = Get-FpsLogStats (Join-Path $Dir 'guest.log') }
    if ($null -eq $stats) { $stats = @('  no frame data (the game did not reach rendering?)') }
    foreach ($l in $stats) { Say $l }
    if ($early -and (Test-Path -LiteralPath $log)) {
        Say 'last lines of emulator.log:'
        foreach ($l in (Get-Content -LiteralPath $log -Tail 15)) { Say ('  ' + $l) }
    }
    Say ('report folder: ' + $Dir)
    Say '----- end of report -----'
}

# ---------------------------------------------------------------- main

if ($Help) { Show-Usage; exit 0 }
if ($Rest -and $Rest.Count -gt 0) {
    [Console]::Error.WriteLine('kestrel: error: unknown argument(s): ' + ($Rest -join ' '))
    Show-Usage
    exit 2
}
if ($Seconds -lt 1) { Fail '-Seconds must be at least 1' }

$presetName = Load-Settings
$emu = Find-Emulator
$script:AssumeAll = $false
$script:HelpText = $null
if ($emu -eq '') {
    $script:AssumeAll = $true
    $emu = 'kyty_emulator'
    $emuDir = $script:ScriptDir
} else {
    $emuDir = Split-Path -Parent $emu
    $script:HelpText = Get-HelpText $emu
    if ($null -eq $script:HelpText) { Warn 'could not run the emulator with --help; optional flags will be skipped' }
}

$gamePath = Get-Val 'game.path'
if ($gamePath -eq '') {
    if (-not $DryRun) { Fail 'no game folder: set [game] path in kestrel.local.ini or pass -Game DIR' }
    Warn 'no game folder set; using placeholder <game dir>'
    $gamePath = '<game dir>'
} elseif (-not $DryRun -and -not (Test-Path -LiteralPath $gamePath -PathType Container)) {
    Fail ('game folder not found: ' + $gamePath)
}

$reportDir = ''
if ($Report) {
    $reportDir = Join-Path (Join-Path $script:ScriptDir 'kestrel-logs') ('report-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
    if (-not $DryRun) { [void](New-Item -ItemType Directory -Force -Path $reportDir) }
}

$emuArgs = Build-Args $gamePath $reportDir

Say ('preset: ' + $presetName)
Say ('config: ' + ($script:ConfigFiles -join ', '))
Say ('emulator: ' + (Get-Banner))
foreach ($w in $script:Warnings) { Say $w }
Say ('saves: ' + (Join-Path $emuDir '_SaveData'))
Say ('command: ' + (Quote-Arg $emu) + ' ' + (Join-Args $emuArgs))

if ($DryRun) { exit 0 }

if ($Report) {
    Invoke-Report $emu $emuArgs $reportDir $presetName $gamePath
    exit 0
}

$proc = Start-Process -FilePath $emu -ArgumentList (Join-Args $emuArgs) -WorkingDirectory $emuDir -NoNewWindow -PassThru
$null = $proc.Handle
try {
    $proc.WaitForExit()
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
}
exit $proc.ExitCode
