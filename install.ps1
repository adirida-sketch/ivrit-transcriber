<#
  Ivrit Transcriber - one-time installer (Windows / PowerShell)

  - Checks for Python 3.10+ (offers to install it with winget if missing)
  - Creates the dedicated folder (default: %USERPROFILE%\IvritTranscriber)
  - Copies the scripts there and creates a private Python environment (.venv)
  - pip install faster-whisper ; pip install -U av ; pip install -U anthropic sherpa-onnx
  - Creates Desktop + Start Menu shortcuts for the GUI

  Usage:  powershell -ExecutionPolicy Bypass -File install.ps1 [-InstallDir <path>]
#>
param(
    [string]$InstallDir = (Join-Path $env:USERPROFILE 'IvritTranscriber')
)

$ErrorActionPreference = 'Stop'
$Src = $PSScriptRoot
$AppFiles = @('app.py', 'insights.py', 'speakers.py', 'winfront.py', 'transcribe_ivrit.py', 'start.cmd', 'transcribe.cmd', 'install.ps1', 'install.cmd', 'README.md', 'CHANGELOG.md', 'LICENSE', 'VERSION')

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

function Find-Python {
    $candidates = @(
        @{ Exe = 'python'; Pre = @() },
        @{ Exe = 'py';     Pre = @('-3') }
    )
    foreach ($c in $candidates) {
        if (-not (Get-Command $c.Exe -ErrorAction SilentlyContinue)) { continue }
        try {
            $pre = $c.Pre
            $out = & $c.Exe @pre -c "import sys; print('%d.%d|%s' % (sys.version_info[0], sys.version_info[1], sys.executable))" 2>$null
        } catch { continue }
        if ("$out" -match '^(\d+)\.(\d+)\|(.+)$') {
            if ([int]$Matches[1] -eq 3 -and [int]$Matches[2] -ge 10) { return $Matches[3].Trim() }
        }
    }
    return $null
}

function Invoke-Checked($exe, [string[]]$argList) {
    & $exe @argList
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $exe $($argList -join ' ')" }
}

function Test-Internet {
    # Any HTTP answer (even an error code) means we are connected
    try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch {}
    foreach ($u in @('https://pypi.org/simple/pip/', 'https://www.python.org/', 'https://huggingface.co/')) {
        try {
            $req = [Net.WebRequest]::Create($u)
            $req.Method = 'HEAD'
            $req.Timeout = 8000
            $req.GetResponse().Close()
            return $true
        } catch {
            $e = $_.Exception
            while ($e -and -not ($e -is [Net.WebException])) { $e = $e.InnerException }
            if ($e -and $e.Response) { return $true }
        }
    }
    return $false
}

function Show-NoInternet($what) {
    Write-Host ''
    Write-Host '****************************************************************' -ForegroundColor Red
    Write-Host ' NO INTERNET CONNECTION' -ForegroundColor Red
    Write-Host " The installer needs internet to $what." -ForegroundColor Red
    Write-Host ' Please connect to the internet and run the installer again.' -ForegroundColor Red
    Write-Host '****************************************************************' -ForegroundColor Red
}

function Test-Libraries($pyExe) {
    if (-not (Test-Path $pyExe)) { return $false }
    & $pyExe -c 'import faster_whisper, av, numpy' 2>$null
    return ($LASTEXITCODE -eq 0)
}

# ------------------------------------------------------------------ running app (update while it is open)
$AppPort = 8765

function Invoke-AppApi($path) {
    # Note: every HTTP request resets the app's idle timer, so poll sparingly (Test-AppPort does not)
    try {
        $req = [Net.WebRequest]::Create("http://127.0.0.1:$AppPort$path")
        $req.Proxy = $null
        $req.Timeout = 3000
        $resp = $req.GetResponse()
        try {
            $sr = New-Object IO.StreamReader($resp.GetResponseStream())
            return ($sr.ReadToEnd() | ConvertFrom-Json)
        } finally { $resp.Close() }
    } catch { return $null }
}

function Test-AppPort {
    $c = New-Object Net.Sockets.TcpClient
    try {
        $ar = $c.BeginConnect('127.0.0.1', $AppPort, $null, $null)
        if (-not $ar.AsyncWaitHandle.WaitOne(500)) { return $false }
        $c.EndConnect($ar)
        return $true
    } catch { return $false } finally { $c.Close() }
}

function Stop-AppServer {
    try {
        $ids = @(Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $AppPort -State Listen -ErrorAction SilentlyContinue |
                 Select-Object -ExpandProperty OwningProcess -Unique)
        foreach ($id in $ids) {
            $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
            if ($proc -and $proc.ProcessName -match '^pythonw?$') { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
        }
    } catch {}
}

function Wait-TranscriptionDone {
    $st = Invoke-AppApi '/status?since=0'
    if (-not ($st -and $st.running)) { return }
    Write-Host "A transcription is in progress ($($st.file)). Waiting for it to finish - it will not be interrupted..." -ForegroundColor Yellow
    while ($st -and $st.running) {
        Start-Sleep -Seconds 5
        $st = Invoke-AppApi '/status?since=0'
    }
    Write-Host 'The transcription finished.'
}

function Close-RunningApp {
    if (-not (Test-AppPort)) { return }
    $info = Invoke-AppApi '/ping'
    if (-not ($info -and $info.app -eq 'ivrit-transcriber')) { return }
    Write-Step 'Ivrit Transcriber is open'
    Write-Host "Version $($info.version) is running (folder: $($info.folder))."
    Wait-TranscriptionDone
    Write-Host 'Please close the Ivrit Transcriber window so it can be updated.' -ForegroundColor Yellow
    Write-Host 'Waiting for it to close... (or press Enter once the window is closed)'
    try { while ([Console]::KeyAvailable) { [void][Console]::ReadKey($true) } } catch {}
    while (Test-AppPort) {
        $enter = $false
        try { while ([Console]::KeyAvailable) { if ([Console]::ReadKey($true).Key -eq 'Enter') { $enter = $true } } } catch {}
        if ($enter) {
            # The window is closed but the server is still up (it exits on its own within a few minutes) - stop it now
            $st = Invoke-AppApi '/status?since=0'
            if ($st -and $st.running) { Wait-TranscriptionDone; Write-Host 'Press Enter once the window is closed.' }
            else { Stop-AppServer; Start-Sleep -Milliseconds 500 }
            continue
        }
        Start-Sleep -Milliseconds 500
    }
    Write-Host 'The transcriber is closed - continuing.' -ForegroundColor Green
}

Write-Host '========================================' -ForegroundColor Green
Write-Host '   Ivrit Transcriber - installation'      -ForegroundColor Green
Write-Host '========================================' -ForegroundColor Green
$AppVersion = 'dev'
if (Test-Path (Join-Path $Src 'VERSION')) { $AppVersion = (Get-Content (Join-Path $Src 'VERSION') -TotalCount 1).Trim() }
Write-Host "Version:        $AppVersion"
Write-Host "Install folder: $InstallDir"

# An open window keeps running the old version (and locks .venv files) - close it first
Close-RunningApp

# ------------------------------------------------------------------ 1. Python
Write-Step 'Looking for Python 3.10 or newer'
$py = Find-Python
if (-not $py) {
    Write-Host 'Python 3.10+ was not found.' -ForegroundColor Yellow
    if (-not (Test-Internet)) {
        Show-NoInternet 'download Python'
        Write-Host 'Alternatively install Python 3.10+ manually and tick "Add python.exe to PATH".' -ForegroundColor Yellow
        exit 1
    }
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        $ans = Read-Host 'Install Python 3.12 now with winget (adds python.exe to PATH)? [Y/n]'
        if ($ans -notmatch '^[nN]') {
            winget install --id Python.Python.3.12 -e --scope user --accept-package-agreements --accept-source-agreements `
                --override '/quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0'
            $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
            $py = Find-Python
        }
    }
}
if (-not $py) {
    Write-Host ''
    Write-Host 'Please install Python 3.10+ from https://www.python.org/downloads/' -ForegroundColor Yellow
    Write-Host 'IMPORTANT: on the first installer screen tick  "Add python.exe to PATH".' -ForegroundColor Yellow
    Write-Host 'Then open a NEW PowerShell window and run this installer again.' -ForegroundColor Yellow
    exit 1
}
Write-Host "Using Python: $py"

# ------------------------------------------------------------------ 2. Folder + files
Write-Step "Creating the transcriber folder"
New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir 'ui') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir 'Transcripts') | Out-Null
$InstallDir = (Resolve-Path $InstallDir).Path

if ((Resolve-Path $Src).Path.TrimEnd('\') -ne $InstallDir.TrimEnd('\')) {
    foreach ($f in $AppFiles) {
        $p = Join-Path $Src $f
        if (Test-Path $p) { Copy-Item $p -Destination $InstallDir -Force }
    }
    Copy-Item (Join-Path $Src 'ui\index.html') -Destination (Join-Path $InstallDir 'ui') -Force
    Write-Host "Files copied to $InstallDir"
} else {
    Write-Host 'Running from the install folder - nothing to copy.'
}

# ------------------------------------------------------------------ 3. Python environment + libraries
Write-Step 'Creating a private Python environment (.venv)'
$venv = Join-Path $InstallDir '.venv'
$vpy = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $vpy)) { Invoke-Checked $py @('-m', 'venv', $venv) }

Write-Step 'Checking the internet connection'
if (Test-Internet) {
    Write-Host 'Connected.'
    Write-Step 'Installing libraries: faster-whisper, av, anthropic, sherpa-onnx (this can take a few minutes)'
    try {
        Invoke-Checked $vpy @('-m', 'pip', 'install', '--upgrade', 'pip')
        Invoke-Checked $vpy @('-m', 'pip', 'install', 'faster-whisper')
        Invoke-Checked $vpy @('-m', 'pip', 'install', '-U', 'av')
        Invoke-Checked $vpy @('-m', 'pip', 'install', '-U', 'anthropic')   # optional summary / title / tasks with Claude
        Invoke-Checked $vpy @('-m', 'pip', 'install', '-U', 'sherpa-onnx') # optional speaker separation (local)
    } catch {
        if (-not (Test-Internet)) { Show-NoInternet 'download the Python libraries (the connection was lost)'; exit 1 }
        throw
    }
} elseif (Test-Libraries $vpy) {
    Write-Host 'No internet connection - skipping library updates (the installed libraries are OK).' -ForegroundColor Yellow
} else {
    Show-NoInternet 'download the Python libraries (about 300 MB)'
    exit 1
}

Write-Step 'Checking the installation'
# No double quotes inside -c: Windows PowerShell 5.1 strips them when calling native programs
Invoke-Checked $vpy @('-c', "import faster_whisper, av, numpy; print('faster-whisper', faster_whisper.__version__, '| av', av.__version__)")

# ------------------------------------------------------------------ 4. Shortcuts
Write-Step 'Creating shortcuts'
$ws = New-Object -ComObject WScript.Shell
$pythonw = Join-Path $venv 'Scripts\pythonw.exe'
foreach ($dir in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
    $lnk = $ws.CreateShortcut((Join-Path $dir 'Ivrit Transcriber.lnk'))
    $lnk.TargetPath = $pythonw
    $lnk.Arguments = '"' + (Join-Path $InstallDir 'app.py') + '"'
    $lnk.WorkingDirectory = $InstallDir
    $lnk.IconLocation = "$env:SystemRoot\System32\SndVol.exe,0"
    $lnk.Description = "Hebrew audio transcription (ivrit.ai) - version $AppVersion"
    $lnk.Save()
}
Write-Host 'Desktop + Start Menu shortcut: "Ivrit Transcriber"'

Write-Host ''
Write-Host '========================================' -ForegroundColor Green
Write-Host " Done! Ivrit Transcriber $AppVersion is installed."                                   -ForegroundColor Green
Write-Host '========================================' -ForegroundColor Green
Write-Host "Folder:  $InstallDir"
Write-Host "Results: $InstallDir\Transcripts\<date_time>\  (one folder per transcription)"
Write-Host 'GUI:     double-click "Ivrit Transcriber" on the Desktop (or start.cmd in the folder)'
Write-Host 'Console: cd "' -NoNewline; Write-Host $InstallDir -NoNewline; Write-Host '" ; .\transcribe.cmd .\my_audio.m4a'
Write-Host 'Note: the first transcription downloads the model (~1.6 GB) once - internet is required for it.'

Write-Host "`nOpening the transcriber..." -ForegroundColor Green
Start-Process -FilePath $pythonw -ArgumentList ('"' + (Join-Path $InstallDir 'app.py') + '"') -WorkingDirectory $InstallDir
exit 0
