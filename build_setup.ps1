<#
  Builds dist\Setup-IvritTranscriber-v<VERSION>.cmd - a single self-extracting installer.

  The .cmd holds all app files as a base64 zip after a ::PAYLOAD:: marker.
  When run, it extracts them to %TEMP%, runs install.ps1 and cleans up.

  Usage:  powershell -ExecutionPolicy Bypass -File build_setup.ps1
#>
$ErrorActionPreference = 'Stop'
$Src = $PSScriptRoot
$Files = @('install.ps1', 'install.cmd', 'app.py', 'insights.py', 'speakers.py', 'winfront.py', 'transcribe_ivrit.py', 'start.cmd', 'transcribe.cmd', 'README.md', 'CHANGELOG.md', 'LICENSE', 'VERSION', 'ui\index.html')

$dist = Join-Path $Src 'dist'
$stage = Join-Path $env:TEMP ('ivt-build-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $dist, (Join-Path $stage 'ui') | Out-Null
try {
    foreach ($f in $Files) { Copy-Item (Join-Path $Src $f) -Destination (Join-Path $stage $f) }
    $zip = Join-Path $env:TEMP ('ivt-build-' + [guid]::NewGuid().ToString('N') + '.zip')
    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip
    $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($zip))
    Remove-Item $zip
} finally {
    Remove-Item $stage -Recurse -Force
}

$header = @'
@echo off
rem ================================================================
rem  Ivrit Transcriber - single-file installer
rem  Double-click to install into %USERPROFILE%\IvritTranscriber
rem  Other folder:  Setup-IvritTranscriber.cmd D:\Transcriber
rem ================================================================
setlocal
set "IVT_SELF=%~f0"
set "IVT_DIR=%TEMP%\IvritTranscriberSetup-%RANDOM%%RANDOM%"
echo Extracting installer files...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $t=[IO.File]::ReadAllText($env:IVT_SELF); $m=':'+':PAYLOAD'+'::'; $b=$t.Substring($t.LastIndexOf($m)+$m.Length) -replace '\s',''; New-Item -ItemType Directory -Force -Path $env:IVT_DIR | Out-Null; $z=Join-Path $env:IVT_DIR 'payload.zip'; [IO.File]::WriteAllBytes($z,[Convert]::FromBase64String($b)); Expand-Archive -LiteralPath $z -DestinationPath $env:IVT_DIR -Force; Remove-Item $z"
if errorlevel 1 (
  echo Extraction failed.
  pause
  exit /b 1
)
if "%~1"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%IVT_DIR%\install.ps1"
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%IVT_DIR%\install.ps1" -InstallDir "%~1"
)
set "IVT_RC=%errorlevel%"
rmdir /s /q "%IVT_DIR%" 2>nul
if not "%IVT_RC%"=="0" (
  echo.
  echo The installation did not complete - see the messages above.
  pause
  exit /b %IVT_RC%
)
echo.
echo This window will close in 5 seconds...
"%SystemRoot%\System32\timeout.exe" /t 5 >nul
exit /b 0
'@

$lines = New-Object System.Collections.Generic.List[string]
$lines.AddRange([string[]]($header -split "`r?`n"))
$lines.Add('::PAYLOAD::')
for ($i = 0; $i -lt $b64.Length; $i += 100) { $lines.Add($b64.Substring($i, [Math]::Min(100, $b64.Length - $i))) }

$version = (Get-Content (Join-Path $Src 'VERSION') -TotalCount 1).Trim()
Get-ChildItem $dist -Filter 'Setup-IvritTranscriber*.cmd' | Remove-Item
$out = Join-Path $dist "Setup-IvritTranscriber-v$version.cmd"
[IO.File]::WriteAllText($out, ($lines -join "`r`n") + "`r`n", [Text.Encoding]::ASCII)
Write-Host "Built: $out ($([Math]::Round((Get-Item $out).Length / 1KB)) KB)" -ForegroundColor Green
