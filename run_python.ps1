$ErrorActionPreference = "Stop"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$python = if ($env:FEDSPECTRA_PYTHON) { $env:FEDSPECTRA_PYTHON } else { "python" }
& $python @args
exit $LASTEXITCODE
