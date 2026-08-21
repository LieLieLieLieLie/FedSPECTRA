$ErrorActionPreference = "Stop"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$python = if ($env:FEDSPECTRA_PYTHON) { $env:FEDSPECTRA_PYTHON } else { "python" }

& $python run_experiments.py --all
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python run_esc50_cv.py --methods FedAvg FedExP FedDisco FPL FedLESAM --run-tag v2
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python run_esc50_locked.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python run_experiments.py --dataset urbansound8k --seed 2027 --ablation
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python run_ablation_multiseed.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python run_staged_ablation.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python run_high_shift_ablation.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python run_experiments.py --dataset urbansound8k --seed 2027 --sensitivity
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python run_experiments.py --dataset urbansound8k --seed 2027 --heterogeneity
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python analyze_and_plot.py
exit $LASTEXITCODE
