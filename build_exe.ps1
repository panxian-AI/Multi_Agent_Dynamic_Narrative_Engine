param(
    # Python interpreter; defaults to .venv\Scripts\python.exe, then PATH python
    [string]$Python = "",
    # Skip dependency installation (use when the environment is ready)
    [switch]$SkipDeps,
    # Build a directory bundle (faster startup; recommended with torch/CUDA)
    [switch]$OneDir,
    # Pass an empty string for CPU-only torch: -TorchIndex ""
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

$buildDir = "dist_build"

if (-not $Python) {
    $venvPython = Join-Path $PWD ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $Python = $venvPython
    } else {
        $Python = "python"
    }
}

function Invoke-Python {
    param([string[]]$Arguments)
    Write-Host "    > $Python $($Arguments -join ' ')"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $Python $($Arguments -join ' ')"
    }
}

Write-Host "==> Python: $Python"
Invoke-Python @("-V")

if (-not $SkipDeps) {
    Write-Host "==> Installing pip / build tools"
    Invoke-Python @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

    Write-Host "==> Installing torch (CUDA 12.8 wheels by default)"
    $torchArgs = @("-m", "pip", "install", "--upgrade", "torch")
    if ($TorchIndex) {
        $torchArgs += @("--index-url", $TorchIndex)
    }
    try {
        Invoke-Python $torchArgs
    } catch {
        Write-Warning "CUDA torch install failed, falling back to PyPI: $($_.Exception.Message)"
        Invoke-Python @("-m", "pip", "install", "--upgrade", "torch")
    }

    Write-Host "==> Installing diffusers / transformers / accelerate / project deps"
    # The [image] extra declares torch/diffusers/transformers/accelerate/Pillow.
    # torch is already installed above, so pip reuses it.
    Invoke-Python @("-m", "pip", "install", "-e", ".[image]")
    Invoke-Python @("-m", "pip", "install", "--upgrade", "pyinstaller")

    Write-Host "==> Verifying local SD backend"
    $verify = "import torch, diffusers, transformers, accelerate; " +
        "print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); " +
        "print('diffusers', diffusers.__version__, '| transformers', transformers.__version__, '| accelerate', accelerate.__version__)"
    Invoke-Python @("-c", $verify)
} else {
    Write-Host "==> SkipDeps: dependency installation skipped"
}

Write-Host "==> Cleaning old build"
if (Test-Path "build") { Remove-Item "build" -Recurse -Force }
if (Test-Path $buildDir) { Remove-Item $buildDir -Recurse -Force }

Write-Host "==> Building MADNE.exe"
$collectArgs = @(
    "--collect-submodules", "madne",
    # diffusers pipelines/schedulers are imported dynamically
    "--collect-submodules", "diffusers",
    "--collect-data", "diffusers",
    # transformers: only CLIP (text encoder for SD1.5 / SD2.1 / SDXL)
    "--collect-submodules", "transformers.models.clip",
    "--collect-data", "transformers",
    "--collect-submodules", "accelerate",
    "--collect-submodules", "huggingface_hub",
    "--collect-submodules", "safetensors",
    "--hidden-import", "tqdm.auto"
)
$modeArg = if ($OneDir) { "--onedir" } else { "--onefile" }
if (-not $OneDir) {
    Write-Warning "onefile packs torch/CUDA into a multi-GB exe with slow first start; consider -OneDir"
}

Invoke-Python (@(
    "-m", "PyInstaller", "--noconfirm", "--clean", $modeArg,
    "--windowed", "--name", "MADNE",
    "--distpath", $buildDir, "--workpath", "build", "--specpath", "build"
) + $collectArgs + @("launcher.py"))

Write-Host "==> Updating dist (preserve runtime and local config)"
New-Item -ItemType Directory -Path "dist" -Force | Out-Null
if ($OneDir) {
    $target = Join-Path $PWD "dist\MADNE"
    if (Test-Path $target) { Remove-Item $target -Recurse -Force }
    Copy-Item (Join-Path $buildDir "MADNE") $target -Recurse -Force
    $exe = Join-Path $target "MADNE.exe"
    if (Test-Path ".env.example") { Copy-Item ".env.example" (Join-Path $target ".env.example") -Force }
    if (Test-Path "README.md") { Copy-Item "README.md" (Join-Path $target "README.md") -Force }
    if (Test-Path (Join-Path $target ".env")) { Remove-Item (Join-Path $target ".env") -Force }
} else {
    Copy-Item (Join-Path $buildDir "MADNE.exe") "dist\MADNE.exe" -Force
    if (Test-Path ".env.example") { Copy-Item ".env.example" "dist\.env.example" -Force }
    if (Test-Path "README.md") { Copy-Item "README.md" "dist\README.md" -Force }
    if (Test-Path "dist\.env") { Remove-Item "dist\.env" -Force }
    $exe = Join-Path $PWD "dist\MADNE.exe"
}
Remove-Item $buildDir -Recurse -Force

Write-Host "==> Running EXE selftest"
$workDir = Split-Path $exe -Parent
$process = Start-Process -FilePath $exe -ArgumentList "--selftest" -WorkingDirectory $workDir -PassThru -Wait
if ($process.ExitCode -ne 0) {
    throw "Selftest failed with exit code $($process.ExitCode)"
}
$result = Join-Path $workDir "selftest_result.txt"
if (Test-Path $result) { Get-Content $result }

Write-Host "==> Cleaning selftest artifacts"
$realResult = Join-Path $workDir "selftest_real_result.txt"
if (Test-Path $realResult) { Remove-Item $realResult -Force }
if (Test-Path $result) { Remove-Item $result -Force }
$selftestData = Join-Path $workDir "runtime_selftest"
if (Test-Path $selftestData) { Remove-Item $selftestData -Recurse -Force }

Write-Host ""
Write-Host "==> Build complete: $exe"
Write-Host "    Double-click the exe to launch the desktop UI."
Write-Host "    Testers enter their own API key in the API configuration window."
Write-Host "    runtime and config next to the exe are local files; do NOT include them when distributing."
