# DVAE-PINN 当前 7D 主路线 Loss 完整表达式说明

## 1. 总体目标

当前方案面向 3W 数据集的 0-9 十类故障诊断。由于数据集中缺少完整井筒几何参数、PVT 参数、阀门流量系数、真实含水率以及完整边界条件，因此把物理知识写成可学习的弱约束与可解释证据特征。

当前主路线使用 **7D rule evidence**，并在训练阶段加入轻量辅助项：

1. `e_rule`：训练和测试阶段都计算，作为分类器输入的一部分。
2. `L_chain`：训练阶段使用的压力-流量/温度条件概率链辅助损失。
3. `L_rule_energy`：训练阶段使用的轻量规则能量辅助损失，当前 7D full best 中权重为 `0.0005`。

核心思想是：保留 DVAE 对多变量时序过程的概率建模能力，同时让分类器显式接收可解释的物理证据，避免完全依赖黑箱 latent 表征。

## 2. 当前主路线结论

```text
Stage 1:
  run = dvae_baseline_full_clip20_v1
  作用 = 训练原始 DVAE 全量窗口 baseline
  rule_evidence = false
  use_unified_physical_rule = false
  class_weighting = none
  clip_norm = 20
  best val macro-F1 = 0.8459
  test macro-F1 = 0.8305

Stage 2:
  run = full_candidate_drop_macro_positive_dims
  作用 = 从 Stage 1 checkpoint 初始化，接入 7D rule evidence，并启用轻量 L_rule_energy 与 L_chain，全模型微调
  rule_evidence = true
  use_unified_physical_rule = true
  use_rule_energy_logits = false
  pinn_weight = 0.0005
  chain_prior_weight = 0.001
  class_weighting = none
  lr = 3e-4
  clip_norm = 20
  split = 3w_seed42_full_val15_test15.json
  train / val / test = 392484 / 87405 / 85717
  best val macro-F1 = 0.8861
  test accuracy = 0.8898
  test macro-F1 = 0.9013
  test balanced accuracy = 0.9165
```

## 3. 模型结构

输入窗口为：

```text
x_{1:T} in R^{T x D}
```

其中 `T` 是时间窗口长度，`D` 是传感器变量数。当前 3W 接入中 `D=27`。

整体结构为：

```text
x_{1:T} -> DVAE encoder -> q_phi(z_{1:T} | x_{1:T})
z_{1:T} -> DVAE decoder -> p_theta(x_{1:T} | z_{1:T})
e_rule = f_rule(x_{1:T}, x_hat_{1:T}, z_mu_{1:T})
h_cls = concat(mean_t(z_mu_{1:T}), e_rule)
logits_cls = classifier(h_cls)
y_hat = softmax(logits_cls)
```

各符号含义：

```text
x_{1:T}      : 标准化后的传感器窗口。
z_{1:T}      : DVAE 时序潜变量。
z_mu_{1:T}   : encoder posterior 的均值，用作确定性 latent 表示。
x_hat_{1:T}  : decoder 根据 z_mu 重构出的传感器序列均值。
e_rule       : 由 x、x_hat 和 z_mu 计算出的 7D 物理证据。
h_cls        : 分类头输入，由窗口 latent 均值和 7D rule evidence 拼接得到。
logits_cls   : 十类故障分类 logits。
y            : 真实故障类别标签。
```

最终分类输入为：

```text
h_cls = concat(mean_t(z_mu), e_rule_7d)
```

## 4. 当前 7D Rule Evidence

当前最终 `e_rule` 包含 7 维：

```text
e_rule_7d = [
  water_rise,
  water_smooth,
  dhsv_open_mean,
  pck_drop,
  pck_open_mean,
  dhsv_low_evidence,
  pck_low_evidence
]
```

各维含义：

```text
water_rise        : 隐含含水状态在窗口前后段的变化。
water_smooth      : 隐含含水状态在窗口内的平滑性。
dhsv_open_mean    : DHSV 有效开度的窗口均值。
pck_drop          : PCK 有效开度在窗口内的下降幅度。
pck_open_mean     : PCK 有效开度的窗口均值。
dhsv_low_evidence : DHSV 长时间低开度证据。
pck_low_evidence  : PCK 长时间低开度证据。
```

原 10D 版本包含但当前删除的维度：

```text
dhsv_drop   : DHSV 有效开度下降幅度。
pck_smooth  : PCK 有效开度时间平滑性。
dp_trend    : 管柱压差或相关压力差趋势。
```

删除原因：维度级消融表明，这三个维度组合在 full 数据上会削弱泛化能力。删除后得到的 7D 版本在 full split 上达到当前最优：`Macro-F1=0.9013`，`Balanced Acc=0.9165`，并将之前关键的 `8->5` 混淆压到 `0`。

## 5. 有效开度计算

DHSV/PCK 有效开度融合 DVAE 重构状态和 latent 开度头，辅助计算规则证据模块中的dhsv_open_mean，pck_drop，pck_open_mean，dhsv_low_evidence，pck_low_evidence维度：

```text
a_latent_t = sigmoid(g_phi(z_mu_t))
a_state_t  = clamp(x_hat_open_t, 0, 1)
a_eff_t    = alpha * a_state_t + (1 - alpha) * a_latent_t
```

其中：

```text
a_latent_t : 由 latent 读出的隐含物理开度。
a_state_t  : 由 decoder 重构出的观测开度。
a_eff_t    : 用于计算 rule evidence 的有效开度。
alpha      : 重构状态与 latent 状态的混合权重。
```

这样做是为了缓解传感器观测不完整、噪声较大、部分物理状态不可直接观测的问题。重构值提供与原始变量一致的状态估计，latent 开度头提供更平滑的隐含物理趋势。

## 6. Loss 总表达式

当前 7D 主路线的训练总损失为：

```text
L_total =
    L_rec
  + beta * L_KL
  + lambda_cls * L_CE
  + lambda_chain * L_chain
  + lambda_rule * L_rule_energy
```

展开为：

```text
L_total =
    - E_{q_phi(z|x)} [ log p_theta(x | z) ]
  + beta * KL(q_phi(z | x) || p_theta(z))
  + lambda_cls * CE(classifier(concat(mean_t(z_mu), e_rule_7d)), y)
  + lambda_chain * L_chain
  + lambda_rule * CE(rule_head(tanh(e_rule_7d)) / tau, y)
```

当前 7D full best 中：

```text
beta = 0.001
lambda_cls = 2.0
lambda_chain = 0.001
lambda_rule = pinn_weight = 0.0005
use_rule_energy_logits = false
class_weighting = none
```

注意：`L_rule_energy` 只作为训练辅助损失参与优化，不在推理阶段额外把 rule logits 融合到分类 logits 中。推理阶段使用的是：

```text
logits_cls = classifier(concat(mean_t(z_mu), e_rule_7d))
```

## 7. `L_rec`：重构负对数似然

DVAE decoder 输出观测变量的概率分布：

```text
p_theta(x_t | z_t)
```

重构损失为：

```text
L_rec = - E_{q_phi(z|x)} [ sum_t log p_theta(x_t | z_t) ]
```

该项作用在整个时间窗口 `x_{1:T}` 上，关注每个时间步、每个传感器变量的重构误差。它要求 latent 保留压力、温度、流量、阀门状态等过程信息，为后续 evidence 计算和分类提供稳定基础。

## 8. `L_KL`：潜变量动态先验约束

encoder 给出 posterior：

```text
q_phi(z_{1:T} | x_{1:T})
```

DVAE transition prior 给出动态先验：

```text
p_theta(z_{1:T})
```

KL 项为：

```text
L_KL = KL(q_phi(z_{1:T} | x_{1:T}) || p_theta(z_{1:T}))
```

该项约束 posterior 不要偏离 DVAE 动态先验，避免 latent 只为分类边界服务而丢失时序生成结构。

## 9. `L_CE`：故障分类损失

分类输入为：

```text
h_cls = concat(mean_t(z_mu), e_rule_7d)
logits_cls = classifier(h_cls)
```

分类损失为：

```text
L_CE = CE(logits_cls, y)
```

该项完成最终 0-9 十类故障诊断任务。`mean_t(z_mu)` 提供窗口级时序表征，`e_rule_7d` 提供物理证据，两者共同输入分类器。

## 10. `L_chain`：条件概率链辅助损失

条件概率链描述压力、流量、温度之间的传播关系：

```text
P_history + z_history -> Q_future
T_history + P_history + Q_hat + z_history -> T_future
```

损失为：

```text
L_chain =
    lambda_flow * NLL(Q_future | P_history, z_history)
  + lambda_temp * NLL(T_future | T_history, P_history, Q_hat, z_history)
```

其中：

```text
P_history : 压力变量历史片段。
Q_future  : 未来流量目标，主要包括 QGL、QBS。
T_history : 温度变量历史片段。
T_future  : 未来温度目标。
z_history : DVAE posterior latent 的历史片段。
Q_hat     : 由压力历史和 latent 预测出的未来流量。
```

每个 NLL 使用 Gaussian NLL：

```text
NLL(u | mu, sigma)
= 0.5 * mean( ((u - mu)^2 / sigma^2) + 2log(sigma) + log(2pi) )
```

该项用于补充故障传播过程约束，避免模型只学习同一时刻变量相关性。

## 11. `L_rule_energy`：统一规则能量辅助损失

规则能量项将 7D evidence 映射为十类规则 logits：

```text
logits_rule = rule_head(tanh(e_rule_7d)) / tau
```

对应损失为：

```text
L_rule_energy = CE(logits_rule, y)
```

该项只在训练阶段使用，当前权重为：

```text
lambda_rule = pinn_weight = 0.0005
```

它的作用是轻量约束 `e_rule_7d` 的类别可分性，帮助模型更稳定地使用物理证据。但它不是主分类头，也不在推理阶段通过 `use_rule_energy_logits` 叠加到分类 logits。已经做过实验验证，在推理阶段如果加入规则能量logits，会重复利用这部分信息，导致结果不如不在推理阶段启用。



## 12. 不纳入当前主路线的探索项

后续围绕边界混淆做过若干探索，包括：

```text
hard-boundary weighted CE
local boundary logit-margin loss
conditional_pair gate
evidence-conditioned gate
7D + hydrate/normal-stability 新增 evidence
```

这些方案在 medium 或诊断实验中没有稳定超过当前 7D 主路线，因此不写入最终 loss。它们可以作为后续分析记录，但不属于当前主路线表达式。

## 13. 推荐训练路径

当前推荐训练路径为两阶段：

```text
Stage 1: 训练原始 DVAE 全量窗口 baseline。
Stage 2: 从 Stage 1 checkpoint warm-start，接入 7D rule evidence，启用轻量 L_rule_energy 与 L_chain，全模型微调。
```

Stage 2 推荐设置：

```text
init_checkpoint = dvae_baseline_full_clip20_v1.pth
use_rule_evidence_features = true
use_unified_physical_rule = true
use_rule_energy_logits = false
RULE_EVIDENCE_KEEP = water_rise,water_smooth,dhsv_open_mean,pck_drop,pck_open_mean,dhsv_low_evidence,pck_low_evidence
RULE_ENERGY_KEEP = water_rise,water_smooth,dhsv_open_mean,pck_drop,pck_open_mean,dhsv_low_evidence,pck_low_evidence
pinn_weight = 0.0005
use_chain_prior_aux = true
chain_prior_weight = 0.001
class_weighting = none
lr = 3e-4
clip_norm = 20
freeze_dvae = false
```

采用 warm-start 是为了避免从零开始同时学习重构、分类、物理证据和辅助规则时目标过多、latent 不稳定。先让 DVAE 学到基础时序结构，再让 encoder、decoder、classifier 一起适配 7D rule evidence，可以保持原有时序表征，同时增强物理可解释诊断能力。

