# Method and experiment map

The model classifies 128-sample windows with a stride of 128. The DVAE uses recurrent inference, a learned transition prior and a measurement model. A summary of the posterior means is fused with the seven evidence features in the evidence-enabled runs. The rule head contributes auxiliary cross-entropy. The conditional prediction branch contributes Gaussian negative log likelihood. For diagnosis, the main classifier uses the temporal representation and the evidence, when enabled.

## 3W 2.0.0 main experiment

- Input is 27 channels. The window label is the majority of valid point labels. Transition labels 101–109 map to classes 1–9. Inputs are forward- and backward-filled within instances, standardized using training-file statistics, and clipped to [-20, 20].
- The temporal mean of the posterior latent means gives eight classifier inputs. Evidence adds seven more: water-cut proxy rise and roughness, mean effective DHSV opening, PCK opening drop, mean effective PCK opening, and low-opening evidence for DHSV and PCK. Effective openings mix a decoder estimate with a latent estimate. The feature values pass through tanh. The water-cut proxy and effective openings are model-derived quantities.
- The base loss is L_rec + 0.001 L_KL + 2.0 L_cls. The rule and chain weights are 0.0005 and 0.001. The chain uses history 64, gap 4 and horizon 16. The flow and temperature terms are weighted 0.005 and 1. At training the temperature branch receives a 0.15/0.85 blend of observed and predicted flow. At evaluation it uses predicted flow. Its historical latent is a slice of the whole-window posterior.
- A0 trains with Adam, learning rate 0.001, up to 30 epochs and patience 6. B–E start from the same best A0, use learning rate 0.0003 and train up to ten further epochs with patience 4. All use batch size 32, no class weighting, weight decay 1e-5 and seed 42. Model selection uses validation Macro-F1.

| Group | 7D | Rule | Chain | Archived test Macro-F1 |
|---|---|---|---|---:|
| A0 | | | | 0.830514 |
| B | Yes | | | 0.866317 |
| C | Yes | Yes | | 0.860861 |
| D | | | Yes | 0.849425 |
| E | Yes | Yes | Yes | 0.901326 |

B is the evidence-only control. D uses only the DVAE representation at classification time and adds the conditional prediction loss during training. B–E have an additional training stage compared with A0. The original E's feature selection included historical test feedback. The file-level split can include instances from the same well in different partitions. Whole-window encoding and preprocessing make this an offline recognition setup.

Supplemental groups GRU, GRU-OBS7D, LSTM, and LSTM-OBS7D are full-split recurrent comparisons. The OBS7D features are observation-derived proxies and differ from the DVAE evidence. E-ZERO-DECAY uses a separate optimizer group with zero weight decay for the chain and obtains test Macro-F1 0.853234. The original E's auxiliary parameter collapse and this diagnostic limit physical interpretations of the chain.

## 3W 1.1.1 adaptation

- Seven observed inputs are P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, P-JUS-CKGL and QGL. The raw observation mask is retained before forward filling and concatenated for a 14-dimensional encoder input. Normalization uses finite original observations from training instances. The split groups exact duplicates and time-overlapping instances from the same well.
- The posterior latent means are pooled with their temporal mean, population standard deviation and last-minus-first difference, yielding 24 dimensions. The new seven features are four pressure/temperature end–start differences, two first-difference RMS values, and pressure–pressure correlation. The mask controls feature validity. Train-fitted empirical CDF transformations, shared learned positive gains and tanh are applied before feature fusion.
- The P3 base loss is 0.1 L_rec + 0.001 L_KL + 2.0 L_cls. Reconstruction is fixed-variance Gaussian NLL on observed elements. The main classifier uses square-root inverse-frequency weights. The rule classifier is initialized from training class prototypes and can update shared evidence gains through its loss.
- C and D independently encode the first 64 points for the conditional branch and use a gap of 4 and horizon of 16. The flow target is QGL and the temperature targets are T-TPT and T-JUS-CKP. The flow/temperature term weights are 0.005/1. The rule and chain outer weights are 0.0005 and 0.001. The chain has zero weight decay in C and D, and other parameters have 1e-5.
- A is the adapted P3 baseline. Other groups start from A and use Adam, learning rate 0.0003, batch size 32, at most ten additional epochs and patience 4. The best completed epoch is selected by validation Macro-F1. Seeds 42 and 2025 share the fixed data split.

| Group | New 7D | Rule | Chain | Test Macro-F1 mean ± sample SD |
|---|---|---|---|---:|
| A | | | | 0.857004 ± 0.063406 |
| B | Yes | Yes | | 0.919606 ± 0.010340 |
| C | | | Yes | 0.864739 ± 0.032804 |
| D | Yes | Yes | Yes | 0.928259 ± 0.004193 |
| EVIDENCE | Yes | | | 0.901884 ± 0.019916 |

C classifies from the 24-dimensional DVAE representation. The joint-stage EVIDENCE control isolates the feature input under the same continuation schedule. Comparisons with A include extra joint training. The test split had historical exposure. The two releases have distinct sensor and label configurations.

## Source map

| Component | File |
|---|---|
| 2.0.0 dataset | code/TRANS/DVAE_3W/threew/dataset.py |
| 2.0.0 DVAE training and chain | code/TRANS/DVAE_3W/train_3w_dvae_classifier.py |
| 2.0.0 7D features and rule loss | code/DVAE_PINN/physics_guided_7d/unified_physical_rule_loss.py |
| 2.0.0 recurrent baselines | code/TRANS/DVAE_3W/train_recurrent_obs7d.py |
| 1.1.1 CSV and masks | code/DVAE_PINN/experiments_3w_v1_1_1/src/prepare.py |
| 1.1.1 Stage 1 and P3 | code/DVAE_PINN/experiments_3w_v1_1_1/src/train.py and tuning_20261028/ |
| 1.1.1 features and rule | code/DVAE_PINN/adapted_p3_new7_20261110/worker.py |
| 1.1.1 chain training | code/DVAE_PINN/chain_adaptation_20261111/experiment.py and code/adapted.py |

The 3W 2.0.0 evidence implementation computes the seven features in a fixed order. [中文实验配置与结果](实验配置与结果.md). The packaged entry point is reproduce.py.
