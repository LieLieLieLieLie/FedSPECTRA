$ErrorActionPreference = "Stop"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$python = if ($env:FEDSPECTRA_PYTHON) { $env:FEDSPECTRA_PYTHON } else { "python" }

& $python run_experiments.py --all
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python run_esc50.py all
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python run_ablations.py all
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python run_experiments.py --dataset urbansound8k --seed 2027 --sensitivity
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python run_experiments.py --dataset urbansound8k --seed 2027 --heterogeneity
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python analyze_and_plot.py
exit $LASTEXITCODE
