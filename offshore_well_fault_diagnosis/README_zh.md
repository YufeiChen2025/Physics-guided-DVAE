# 中文快速说明

这个包包含海上生产井故障诊断论文的模型、实验配置、固定划分、原始结果和模型权重。模型用 DVAE 学习时序潜在表征，再与七维工况特征融合分类。规则辅助分类与压力–流量–温度预测参与训练。

- 3W 2.0.0 是十分类主实验，含 A0、C、D、E 和仅七维证据的 B。补充组包含 GRU／LSTM 全量对比和 chain 零衰减诊断。
- 3W 1.1.1 是九分类适配实验，含两种子的 A、B、C、D 和仅新 7D 对照。该版本用带原始观测掩码的压力和温度统计量构造工况特征。
- 解压后执行 python reproduce.py verify，可直接核对历史结果。按 [英文说明](README.md) 安装依赖，按 [数据说明](docs/DATA.md) 下载 Petrobras 的两个原始数据版本。重跑已有权重用 evaluate，重新训练用 train。
- 新输出默认写入 work/。运行 python reproduce.py --help 可以查看具体参数。

更多细节见 [方法与实验口径](docs/METHOD.md) 和 [验证记录](docs/VERIFICATION.md)。
