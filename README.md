# FedSPECTRA

Official experiment code for **FedSPECTRA: Reliability-Shrunk Paired Prototypes for Heterogeneous Federated Signal Learning**.

The repository contains the federated training pipeline, five comparison methods, ablations, ESC-50 five-fold evaluation, statistical analysis, and the scripts used to generate the paper figures. Datasets, cached features, checkpoints, logs, tables, and figures are intentionally excluded.

## Environment

- Python 3.9
- PyTorch 2.0.0
- CUDA-capable GPU recommended

```bash
python -m pip install -r requirements.txt
```

The experiments were run with `D:/anaconda3/envs/pytorch2.0.0_py3.9/python.exe`. On Windows PowerShell, optionally point the launchers to another interpreter:

```powershell
$env:FEDSPECTRA_PYTHON = "C:/path/to/python.exe"
```

## Datasets

Download the datasets from their official sources:

- [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) (use the official `audio/` and `metadata/` directories).
- [ESC-50](https://github.com/karolpiczak/ESC-50) (clone or extract the official repository).

The default layout is:

```text
data/
├── UrbanSound8K/
│   ├── audio/fold1 ... fold10/
│   └── metadata/UrbanSound8K.csv
└── ESC-50-master/
    ├── audio/
    └── meta/esc50.csv
```

Alternatively set one or more environment variables:

```powershell
$env:FEDSPECTRA_DATA_ROOT = "D:/datasets"
$env:FEDSPECTRA_URBANSOUND_ROOT = "D:/datasets/UrbanSound8K"
$env:FEDSPECTRA_ESC50_ROOT = "D:/datasets/ESC-50-master"
```

The first run converts the official audio files into cached 64-bin log-mel tensors under `results/models/`. UrbanSound8K may also use preprocessed `processed/fold_1.npy` through `processed/fold_10.npy` files containing `waveform` and `target` fields.

## Reproduction

Run from the repository root. The complete Windows workflow is:

```powershell
./run_all.ps1
```

Individual stages can be executed as follows:

```powershell
# UrbanSound8K: six methods, five seeds
python run_experiments.py --all

# ESC-50: five baselines followed by the locked FedSPECTRA protocol
python run_esc50_cv.py --methods FedAvg FedExP FedDisco FPL FedLESAM --run-tag v2
python run_esc50_locked.py

# Ablations and controlled stress tests
python run_experiments.py --dataset urbansound8k --seed 2027 --ablation
python run_ablation_multiseed.py
python run_staged_ablation.py
python run_high_shift_ablation.py
python run_experiments.py --dataset urbansound8k --seed 2027 --sensitivity
python run_experiments.py --dataset urbansound8k --seed 2027 --heterogeneity

# Tables, statistics, and the three experiment figures used in the paper
python analyze_and_plot.py
```

For a short pipeline check:

```powershell
python run_experiments.py --dataset urbansound8k --method FedSPECTRA --seed 2027 --quick
python test_invariants.py
```

## Output layout

```text
results/
├── figures/   # PDF visualizations
├── models/    # checkpoints, predictions, histories, and caches
└── tables/    # CSV and LaTeX tables
```

All dataset splits, baseline configurations, random seeds, and ESC-50 validation-contained selections are encoded in the scripts. Test-fold statistics are not used for hyperparameter selection.

