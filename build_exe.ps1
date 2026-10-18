$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

$buildDir = "dist_build"

Write-Host "==> Checking PyInstaller"
python -m pip install --upgrade pyinstaller

Write-Host "==> Cleaning old build"
if (Test-Path "build") { Remove-Item "build" -Recurse -Force }
if (Test-Path $buildDir) { Remove-Item $buildDir -Recurse -Force }

Write-Host "==> Building MADNE.exe"
python -m PyInstaller --noconfirm --clean --onefile --windowed --name MADNE --distpath $buildDir --workpath build --specpath build --collect-submodules madne launcher.py

Write-Host "==> Updating dist (preserve runtime and local config)"
New-Item -ItemType Directory -Path "dist" -Force | Out-Null
Copy-Item (Join-Path $buildDir "MADNE.exe") "dist\MADNE.exe" -Force
if (Test-Path ".env.example") { Copy-Item ".env.example" "dist\.env.example" -Force }
if (Test-Path "README.md") { Copy-Item "README.md" "dist\README.md" -Force }
if (Test-Path "dist\.env") { Remove-Item "dist\.env" -Force }
Remove-Item $buildDir -Recurse -Force

Write-Host "==> Running EXE selftest"
$exe = Join-Path $PWD "dist\MADNE.exe"
$process = Start-Process -FilePath $exe -ArgumentList "--selftest" -WorkingDirectory (Join-Path $PWD "dist") -PassThru -Wait
if ($process.ExitCode -ne 0) {
    throw "Selftest failed with exit code $($process.ExitCode)"
}
$result = Join-Path $PWD "dist\selftest_result.txt"
if (Test-Path $result) { Get-Content $result }

Write-Host "==> Cleaning selftest artifacts"
$realResult = Join-Path $PWD "dist\selftest_real_result.txt"
if (Test-Path $realResult) { Remove-Item $realResult -Force }
if (Test-Path $result) { Remove-Item $result -Force }
$selftestData = Join-Path $PWD "dist\runtime_selftest"
if (Test-Path $selftestData) { Remove-Item $selftestData -Recurse -Force }

Write-Host ""
Write-Host "==> Build complete: $exe"
Write-Host "    Double-click the exe to launch the desktop UI."
Write-Host "    Testers enter their own API key in the API configuration window."
Write-Host "    dist\runtime and dist\config are local files; do NOT include them when distributing."
