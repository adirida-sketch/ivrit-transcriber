<#
  Publishes the version written in VERSION:
    1. checks: on main, everything committed, CHANGELOG.md has a section for it, tag is new
    2. creates the git tag vX.Y.Z and pushes main + the tag
    3. builds dist\Setup-IvritTranscriber-vX.Y.Z.cmd
    4. creates a GitHub Release with the changelog notes and the installer attached

  Before running: bump VERSION, add "## [X.Y.Z] - YYYY-MM-DD" to CHANGELOG.md, commit.

  Usage:  powershell -ExecutionPolicy Bypass -File release.ps1
#>
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Invoke-Checked($exe, [string[]]$argList) {
    & $exe @argList
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $exe $($argList -join ' ')" }
}

$version = (Get-Content VERSION -TotalCount 1).Trim()
if ($version -notmatch '^\d+\.\d+\.\d+$') { throw "VERSION must look like 1.2.3 (found '$version')" }
$tag = "v$version"
Write-Host "Releasing $tag" -ForegroundColor Cyan

# ---- checks
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne 'main') { throw "Release from 'main' only (current branch: $branch)" }
if (git status --porcelain) { throw 'There are uncommitted changes - commit them first.' }
git rev-parse -q --verify "refs/tags/$tag" | Out-Null
if ($LASTEXITCODE -eq 0) { throw "Tag $tag already exists - bump VERSION first." }
if (git ls-remote --tags origin "refs/tags/$tag") { throw "Tag $tag already exists on GitHub - bump VERSION first." }

$changelog = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'CHANGELOG.md'), [Text.Encoding]::UTF8)
$m = [regex]::Match($changelog, '(?ms)^## \[' + [regex]::Escape($version) + '\][^\n]*\n(.*?)(?=^## \[|\z)')
if (-not $m.Success -or -not $m.Groups[1].Value.Trim()) { throw "CHANGELOG.md has no section '## [$version]'." }
$notes = $m.Groups[1].Value.Trim()

# ---- tag + push
Invoke-Checked git @('tag', '-a', $tag, '-m', "Version $version")
Invoke-Checked git @('push', 'origin', 'main')
Invoke-Checked git @('push', 'origin', $tag)

# ---- build installer
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'build_setup.ps1')
if ($LASTEXITCODE -ne 0) { throw 'build_setup.ps1 failed' }
$setup = Join-Path $PSScriptRoot "dist\Setup-IvritTranscriber-$tag.cmd"

# ---- GitHub release
$notesFile = Join-Path $env:TEMP "ivt-release-$version.md"
[IO.File]::WriteAllText($notesFile, $notes, (New-Object Text.UTF8Encoding($false)))
try {
    Invoke-Checked gh @('release', 'create', $tag, $setup, '--title', "Ivrit Transcriber $tag", '--notes-file', $notesFile)
} finally {
    Remove-Item $notesFile -ErrorAction SilentlyContinue
}
Write-Host "`nReleased $tag" -ForegroundColor Green
