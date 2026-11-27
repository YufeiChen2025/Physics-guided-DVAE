# Fault diagnosis in offshore production wells using a dynamical variational autoencoder with physics-guided features

This research bundle implements window-level fault diagnosis on Petrobras 3W. A dynamical variational autoencoder (DVAE) produces a temporal latent representation. Seven engineering features are concatenated with it before classification. An auxiliary rule classifier and a pressure–flow–temperature conditional predictor contribute training losses.

See [Method and experiment map](docs/METHOD.md), [Data sources](docs/DATA.md), [Verification](docs/VERIFICATION.md), and the [Chinese quick start](README_zh.md).

## 1. Check the packaged results

From the extracted directory, with Python 3.11:

~~~sh
python reproduce.py verify
python reproduce.py report --include-extras
python export_tables.py --output-dir tables --include-extras
~~~

These commands need only the Python standard library. The export includes per-class metrics and confusion matrices. Historical checkpoints, results, and the 1.1.1 test predictions are in reference/. New outputs go under work/ by default.

## 2. Install and download

The verified environment was Windows, Python 3.11, PyTorch 2.3.0+cu118, an RTX 4060 Laptop GPU with 8 GB VRAM, NumPy 1.26.4, pandas 3.0.2, and PyArrow 25.0.0. Use Python 3.11 and preferably a CUDA-capable NVIDIA GPU for retraining.

~~~sh
python -m venv .venv
~~~

Activate with .venv\Scripts\Activate.ps1 in PowerShell or source .venv/bin/activate in Bash, then run:

~~~sh
python -m pip install -r requirements.txt
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
~~~

For CPU use on Windows or Linux, use https://download.pytorch.org/whl/cpu as the PyTorch index. For other supported platforms, see the [PyTorch 2.3.0 instructions](https://pytorch.org/get-started/previous-versions/#v230). A CPU build is slower. For 2.0.0, use --num-workers 0 or 1 if worker memory is limited.

Download the public archives in [Data sources](docs/DATA.md). The commands below assume each data directory is directly under data/ and contains the numbered class folders.

## 3. Re-evaluate historical checkpoints

~~~sh
python reproduce.py evaluate --version 2.0.0 --data-root data/3w_dataset_2.0.0
python reproduce.py evaluate --version 1.1.1 --data-root data/3w_dataset_1.1.1
python reproduce.py compare
python reproduce.py report --source evaluate
python export_tables.py --source evaluate --output-dir tables-evaluated
~~~

Use --groups E or --groups D --seeds 42 to evaluate only the corresponding complete model. Add --include-extras to the 2.0.0 command for the four recurrent comparisons and the chain zero-decay diagnostic. 1.1.1 rebuilds the mask-aware raw data cache on first use and applies the archived train-fitted feature transform for checkpoint evaluation. Use --work-dir PATH to put the cache and generated outputs on another drive. Use the same path for compare, report, and table exports.

## 4. Retrain

~~~sh
python reproduce.py train --version 2.0.0 --data-root data/3w_dataset_2.0.0 --work-dir retrain
python reproduce.py train --version 1.1.1 --data-root data/3w_dataset_1.1.1 --work-dir retrain --baseline scratch
python reproduce.py report --source train --work-dir retrain
~~~

2.0.0 starts from a newly trained A0. For 1.1.1, --baseline scratch trains the earlier Stage 1, adapts it into P3, then trains the joint groups. Its default mode, --baseline p3, starts P3 from the packaged earlier Stage 1. Both versions support joint-stage training from a packaged historical baseline:

~~~sh
python reproduce.py train --version 2.0.0 --data-root data/3w_dataset_2.0.0 --groups E --baseline reference --work-dir warmstart
python reproduce.py train --version 1.1.1 --data-root data/3w_dataset_1.1.1 --groups D --seeds 42 --baseline reference --work-dir warmstart
~~~

Only run one training invocation per work directory at a time. Completed runs can be reused. The DVAE variants support epoch-boundary continuation where their trainer saves resume state. To evaluate a newly trained checkpoint again, use --checkpoint-source trained.

## Contents and interpretation

| Path | Contents |
|---|---|
| code/ | Dataset preparation, models, training and evaluation |
| protocols/ | Fixed split lists, per-run settings and frozen preprocessing parameters |
| reference/ | Historical checkpoints and outputs |
| docs/ | Method details, data and verification |
| SHA256SUMS.txt | Integrity manifest |

The 3W releases use different sensors, classes, missing-value handling, and evidence definitions. The results describe the corresponding fixed splits. Historical tests had prior exposure during development, and these experiments provide no independent blind or well-disjoint validation.
