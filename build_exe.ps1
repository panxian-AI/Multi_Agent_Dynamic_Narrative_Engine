$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "==> Checking PyInstaller"
python -m pip install --upgrade pyinstaller

Write-Host "==> Cleaning old build"
if (Test-Path "build") { Remove-Item "build" -Recurse -Force }
if (Test-Path "dist") { Remove-Item "dist" -Recurse -Force }
if (Test-Path "MADNE.spec") { Remove-Item "MADNE.spec" -Force }

Write-Host "==> Building MADNE.exe"
python -m PyInstaller --noconfirm --clean --onefile --windowed --name MADNE --collect-submodules madne launcher.py

Write-Host "==> Copying docs (API key is NOT copied)"
if (Test-Path ".env.example") { Copy-Item ".env.example" "dist\.env.example" -Force }
if (Test-Path "README.md") { Copy-Item "README.md" "dist\README.md" -Force }
if (Test-Path "dist\.env") { Remove-Item "dist\.env" -Force }

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
