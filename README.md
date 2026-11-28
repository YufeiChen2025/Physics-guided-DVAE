# Physics-guided DVAE

Window-level fault diagnosis for offshore production wells using the Petrobras 3W data.

The repository includes the 7D model, reproducibility scripts, frozen splits, checkpoints, and prediction results.

| Dataset | Chain-only ablation | Test Macro-F1 |
|---|---|---:|
| 3W 2.0.0 | D, DVAE + Chain | 0.849425 |
| 3W 1.1.1 | C, Adapted DVAE + Chain | 0.864739 ± 0.032804 |

Start with the [reproduction guide](offshore_well_fault_diagnosis/README.md) or [中文说明](offshore_well_fault_diagnosis/README_zh.md).
Read the [experiment design](offshore_well_fault_diagnosis/docs/METHOD.md) and [complete tables](offshore_well_fault_diagnosis/reference/TABLES.md).
