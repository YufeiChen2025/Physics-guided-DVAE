# Verification

## Chain-only ablations

The released 3W 2.0.0 group D uses the archived DVAE + chain checkpoint (historical experiment F). All 85,717 test windows were re-evaluated and the confusion matrix and classification metrics matched the original. The classifier receives eight DVAE features. The conditional predictor retains the 2.0.0 protocol's joint weight decay of 1e-5.

The released 3W 1.1.1 group C was trained from the corresponding P3 baseline for seeds 42 and 2025. The training used the fixed split and continuation schedule, with a separate zero-weight-decay optimizer group for the conditional predictor. Each seed used the classification checkpoint selected by validation Macro-F1. The classifier receives the 24-dimensional latent summary. Both checkpoints were separately evaluated on the 57,401-window test split, and the prediction IDs and confusion matrices matched the released results. A small one-epoch paired check also verified identical model tensors across the packaging cleanup for 1.1.1 groups B, C and D.

## Other published configurations

3W 2.0.0 groups B, C, E and E-ZERO-DECAY use the fixed seven-feature implementation. The retained archived model tensors and normalization statistics were checked when the package was exported, and complete checkpoint replays matched the original confusion matrices. Other 3W 2.0.0 and 1.1.1 results are the archived checkpoint replays described by the underlying experiment protocols. The 3W 1.1.1 cache was rebuilt from 1,978 original CSV instances, recovering 276,108 training, 61,744 validation and 57,401 test windows.

[Machine-readable checks](verification.json). Historical test sets had development exposure. These checks describe the fixed splits used in this repository.
