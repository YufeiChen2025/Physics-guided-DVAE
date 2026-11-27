# Verification

The packaged model state tensors were checked against their local archived originals after export. Model weights and normalization data were preserved. Machine-specific launch paths were removed from the distributed weights and result metadata. The entry points supply the package paths at run time.

## Full checkpoint replay

The 3W 1.1.1 cache was rebuilt from 1,978 original CSV instances using the packaged split. The original counts of 276,108 training, 61,744 validation and 57,401 test windows were recovered. The fixed split and training normalization were checked against the archived configuration.

On Windows with Python 3.11, PyTorch 2.3.0+cu118 and an RTX 4060 Laptop GPU, all 20 checkpoint evaluations recovered their historical confusion matrices and Macro-F1 values. This covers ten 2.0.0 runs across the main and supplemental groups, and five 1.1.1 configurations at each of two seeds. For 1.1.1 all 57,401 window labels and predicted class IDs of each run also matched the stored predictions exactly. [Machine-readable comparison](verification.json).

## Training checks

Small subsets of real windows were used for one-epoch checks. The 2.0.0 checks covered A0, E, E-ZERO-DECAY and GRU-OBS7D. The 1.1.1 checks covered the earlier Stage 1, P3 and all four joint branches (B, C, D and EVIDENCE). Full checkpoint replay and small training checks are separate. Full-scale retraining of every run, independent blind testing, and execution on a separate operating system were outside this verification.
