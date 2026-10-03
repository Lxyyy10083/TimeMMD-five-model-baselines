# TaTS Full：预测相关性选择与图条件未来特征分布

## 交付边界

本实现只新增一个 `full` 方法，不提供消融模式。保留原入口默认 `original`，新增 `scripts/main_forecast_essay_full.sh`，沿用原 essay 的九领域、24 步历史、12 步 decoder 上下文和各领域四个预测长度。脚本默认只运行 full+iTransformer；通过 `MODELS` 可比较 full 下不同 backbone。

按要求，开发过程中没有运行训练、测试、smoke test、模型前向、编译检查或指标评估。执行过的 Python 代码仅用标准库读取 CSV，统计数据规模和日期顺序。可运行性尚未经过实际执行验证，不能据此声称精度提升。没有修改任何 `models/` 下的 backbone 文件，也没有更改冻结 LLM 的权重、层数设置、tokenizer 或 token embedding 池化算法。

对旧实验文件的两处修改仅为：GPT-2 本地路径允许配置；原本无条件 `.cuda()` 的未使用可训练投影改为 `.to(self.device)`，使 CPU 初始化不在这里失败。

## 阅读所得与融合选择

本地 GANF 的 `models/GANF.py` 先对各变量的时间序列编码，再经图传播生成条件，最后计算条件流的 log probability；`train_water.py` 通过矩阵指数的迹约束邻接矩阵无环。`models/NF.py` 给出了 MAF/RealNVP、正反变换和 Jacobian 的累积。

直接把 GANF 接到 TaTS 后面有三个问题：GANF 的原任务是观测密度异常检测；当前 Time-MMD 使用单变量目标，没有原论文中的多传感器图；高维密度模块容易在几百行数据上过拟合。另一个不应照搬的细节是 GANF 的条件编码使用当前观测，预测任务则必须限制条件为预测原点之前可获得的信息。

本地 TaTS 将池化文本经 MLP 压缩后拼为额外通道，再读取 backbone 的第零输出通道。这个方式不能让通道独立的 DLinear/PatchTST 利用文本。旧流程还读取未来行的 `prior_history_avg`，其预测时可得性未经证明；full 不读取该列。旧训练仅保存 backbone，未同步保存文本 MLP；full 保存整个可训练系统。

参考仓库：[Lxyyy10083/TaTS-NF](https://github.com/Lxyyy10083/TaTS-NF)。以实际读取的本地代码为实施基础；不假设参考仓库名称意味着已包含可直接复用的流改进。

## 框架流程

```text
历史数值 X ──历史归一化与小型 GRU──历史状态 h──────────────┐
                                                        │
历史 fact ──原冻结 embedding/池化──可训练文本投影 u         │
                                  │                     │
                      历史条件预测相关性门控 g             │
                                  │                     │
                           选择后的文本 s                │
                         ╱                 ╲             │
          有界标量输入精炼                  条件融合 c ◀───┘
                    │                          ▲
          可替换 backbone ──基础预测 b──────────┘
                    │                          │
                    │              图自回归条件流 p(r | c)
                    │                          │
                    │                未来残差特征均值/方差
                    └─────────不确定性约束修正───┘
                                     │
                                   预测 ŷ
```

### 1. 先判断是否有预测相关的有用信息

历史数值只使用当前输入窗口的均值、标准差归一化；GRU 输出时间状态。门控输入为历史状态、文本投影及其逐元素乘积，输出每个历史时点的软选择权重；另有通道门控。空文本强制为零，文本摘要只从选择后的文本得到。

训练时，用共享的未来固定特征目标构造两个轻量探针：仅历史状态预测、历史状态加选择后文本预测。以两者预测误差的相对改善作为停止梯度的软标签，监督门控；两个探针同时受真实特征回归监督。它衡量的是任务上的预测增益，不是单纯模态相似度。此辅助比较发生在一个 full 模型的损失内部，不是另外运行消融实验。

软标签为 `sigmoid((relative_gain-0.05)/0.025)`，无增益时约为0.12，避免无信息文本仍被监督成0.5；其中 `relative_gain=(error_ts-error_joint)/(0.1+error_ts)`，0.1 稳定项避免小误差除法放大。这是可训练的相关性代理，不是已识别的因果贡献；训练误差改善仍可能过拟合，所以采用低容量、dropout、稀疏惩罚与验证集早停。所有标签仅用于训练损失，推理门控只看历史。

选择后的文本通过 `0.1 × 历史标准差 × tanh(标量投影)` 精炼输入。该投影零初始化，数值序列占主导；通道数保持 1。因此文本可以进入通道独立 backbone，而不需要修改它们内部。

### 2. 再建模当前条件下的未来特征分布

采用固定正交 DCT 基 B，取前 K 个低频系数。对未来残差定义：

`r = ((Y - stop_gradient(b)) / historical_scale)^T B / sqrt(H)`。

固定目标避免学习目标编码器与流相互配合、将目标压成常数。除以 `sqrt(H)` 让不同预测长度的特征量级更可比。K 根据训练段规模和预测长度自动限制；不把大量重叠窗口当作独立样本。

图节点是这些有明确固定含义的未来轨迹系数。每个三角流层学习全局稀疏边，并用当前条件调节父节点强度。第 k 个系数的平移和尺度只能依赖其有序父节点与历史条件；Jacobian 为三角形，可精确累加对数行列式。尺度为 `0.7*tanh(raw)`，避免指数无界。逆变换按同一顺序逐系数生成，多个流层反向求逆。第二层采用相反次序，提高表达能力。

这里将 GANF 的“依赖结构组织条件密度”迁移为“历史多模态条件组织未来轨迹特征之间的依赖”，而非另接一个异常检测模型。每层用严格三角掩码保证 DAG，不引入小样本上不稳定的矩阵指数增广拉格朗日优化；各层组成的总映射不解释为一张具有真实因果意义的 DAG。

训练密度时允许看到目标的前序系数，这是自回归似然的合法因子分解；推理时这些系数由流顺序生成，不需要未来真实目标。条件包括历史状态、已选文本和基础预测的固定特征，不含未来观测或未来文本。

### 3. 分布实际参与预测

固定 Sobol 正态对偶样本经可微逆流生成残差特征，再还原为低频残差轨迹，估计均值与方差。对偶样本保存为 buffer，训练/评估使用同一组样本；模型处于 eval 时该采样步骤不引入随机波动。

最终预测为 `ŷ = b + historical_scale × sigmoid(a)/(1+variance) × residual_mean`。初始化 `a=-2`，避免分布尚未学好时大幅修正 backbone。均值由完整逆流样本估计，不把 `inverse(0)` 误称为分布均值。

流不是仅用于额外损失的装饰：预测 MSE 通过采样均值及方差反馈进入流，流的条件同时受到相关性门控控制。高频部分继续交给 backbone。导出的残差方差只是低维残差分布的模型估计，未校准，不能直接当作完整未来轨迹的置信区间。

### 4. 联合目标

`L = MSE(ŷ,Y) + warmup*0.05*NLL(r|c) + 0.05*(流均值特征对齐 + 两个探针回归) + 0.02*相关性BCE + 0.001*(门控均值 + 图边稀疏项)`。

NLL 按 K 归一化，默认五轮 warmup。条件流及密度目标使用 float32；支持用户主动启用 AMP，统一优化器覆盖所有可训练模块。目标对 backbone 基础预测停止梯度，让密度损失不直接推动 backbone 操纵残差标签；预测 MSE 仍训练 backbone。相关性探针标签也停止梯度。

## 本地数据统计与容量

以下为实际 CSV 记录数，按 70%/10%/20% 的整数边界划分；训练窗口使用 L=24。窗口范围对应各领域最短至最长预测长度。

| 领域 | 总行数 | 训练/验证/测试行数 | 训练窗口数范围 | 空 fact |
|---|---:|---|---|---:|
| Agriculture | 496 | 347 / 50 / 99 | 318 → 312 | 95 |
| Climate | 496 | 347 / 50 / 99 | 318 → 312 | 0 |
| Economy | 423 | 296 / 43 / 84 | 267 → 261 | 0 |
| Energy | 1479 | 1035 / 149 / 295 | 1000 → 964 | 1 |
| Environment | 15248 | 10673 / 1526 / 3049 | 10602 → 10314 | 234 |
| Health | 1389 | 972 / 140 / 277 | 937 → 901 | 2 |
| Security | 297 | 207 / 31 / 59 | 178 → 172 | 0 |
| SocialGood | 900 | 630 / 90 / 180 | 601 → 595 | 367 |
| Traffic | 531 | 371 / 54 / 106 | 342 → 336 | 0 |

空 fact 表按 CSV 空字符串统计；运行时另把 nan/none/No information available 视为无文本。

| 训练行数 | 框架宽度 | backbone d_model / d_ff / e_layers | 流层数 | dropout | AdamW 衰减 |
|---|---:|---|---:|---:|---:|
| <700 | 16 | 32 / 64 / 1 | 1 | 0.30 | 0.01 |
| 700–2999 | 24 | 64 / 128 / 1 | 2 | 0.20 | 0.005 |
| ≥3000 | 32 | 128 / 256 / 2 | 2 | 0.15 | 0.001 |

`effective_blocks = floor(train_rows / max(L,H))`；`K=min(8,H,max(2,floor(sqrt(effective_blocks))))`。这是保守容量启发式，不是统计独立样本数或最优超参数的证明。统一四个注意力头；DLinear 不使用 d_model 等 Transformer 参数。小数据 batch 上限16，其余32，实际不得超过训练窗口数。梯度裁剪1.0，余弦学习率，验证 MSE 早停最多等待10轮；不监控测试集来选择轮数。

## 数据与运行一致性

- Economy 原文件从2024倒排到1989；full 在内存中稳定升序排列后划分。原 CSV 不改写。
- scaler 只拟合训练行；验证/测试窗口仅向前借用24步历史，未来标签全部处于各自划分。
- 预测原点解释为最后一个已观测周期的 `end_date`，历史文本只有在其 `end_date` 不晚于该原点时才可用。这依赖数据集 end_date 确实代表事实可用时间；代码不能证明文章实际发布日期或原数据抓取过程没有前视信息。
- 不使用 `prior_history_avg`、`preds` 或未来 fact。测试标签只用于离线评估。
- 冻结文本 lookup 分64行计算，池化结果缓存CPU；只优化新增模块和 backbone。这里仍是原 TaTS 的 token embedding 池化，不冒称 LLM 全上下文语义编码。
- train/val/test 使用同一个 full 前向，不丢弃尾批。保存 backbone、文本投影、门控、图流、探针、固定基和采样 buffer；同时保存配置、训练归一化量、优化器/调度器状态。当前没有自动续训入口。
- 默认在训练标准化量纲评估；`--inverse` 切换到原量纲。`feature_nll` 始终是归一化低维残差特征的密度，不能与完整原始轨迹密度等同。
- full 修改了时间顺序、未来先验使用、尾批与容量策略；因此与旧脚本已有结果的差异不能全部归因于模型创新。若论文要求严格基线对比，需另行统一数据协议再重跑；本次只交付 full，不添加或执行基线/消融。

## 使用

依赖沿用原 README 环境（Python 3.11、PyTorch 2.5、transformers、pandas、scikit-learn、sktime、reformer_pytorch 等）。本次没有安装依赖或下载模型。GPT-2 默认使用原服务器本地目录，也可以指定其他包含同一模型与 tokenizer 的目录。

```bash
# 在 TaTS-main-change-Readgpt 目录，用 Bash 启动全套 full 实验。
GPT2_PATH=/your/local/gpt2 bash scripts/main_forecast_essay_full.sh

# 可选：只比较 full 使用不同 backbone，仍无消融。
MODELS="iTransformer DLinear PatchTST Transformer" \
GPT2_PATH=/your/local/gpt2 bash scripts/main_forecast_essay_full.sh
```

支持的 full backbone 为 iTransformer、DLinear、PatchTST、Transformer；接口是单变量历史与日历特征输入，输出未来单变量轨迹。不声称其他原仓库模型的特殊实现已经完成适配。full 当前使用单设备，原 GPT-2 本地加载、六层配置、池化方式保持不变。脚本的 `GPU` 为物理设备选择，进程内使用逻辑 `cuda:0`。

单次示例：

```bash
python run.py --variant full --task_name long_term_forecast --is_training 1 \
  --model_id Agriculture_2025_24_6_iTransformer_TaTS_GPT2 \
  --model iTransformer --data custom --root_path ./data --data_path Agriculture.csv \
  --seq_len 24 --label_len 12 --pred_len 6 --freq m --target OT \
  --llm_model GPT2 --gpt2_path /your/local/gpt2 --seed 2025 --des full \
  --train_epochs 50 --patience 10 --num_workers 0 \
  --checkpoints ./checkpoints/full --save_name ./results/full/summary.jsonl
```

用户运行脚本后才会执行常规训练、验证和最终测试集评估。输出为 `log/full`、`checkpoints/full`、`results/full`。每个结果目录包括预测、真实值、残差方差、修正增益、历史文本门控、指标和配置；汇总为 JSONL。只评估已有模型时，以完全相同的模型标识和配置改为 `--is_training 0`，将恢复 full checkpoint。

## 文件映射

- `layers/PredictiveGraphFlow.py`：相关性、精炼、图条件流、采样反馈与联合目标。
- `data_provider/full_data.py`：规模策略、时间排序、数据划分、原冻结嵌入缓存与窗口。
- `exp/exp_full_forecasting.py`：完整模块生命周期、优化、早停、保存恢复与结果导出。
- `run.py`：full 参数与实验分派；默认原实现入口保留。
- `scripts/main_forecast_essay_full.sh`：九领域 × 四预测长度的 full-only 实验。

清理时递归检查 TaTS 目录，未发现日志、训练检查点或结果文件，实际没有可删除产物；GANF 原有检查点保留。清理命令曾被自动审批策略拦截，随后只读复查确认无需删除。
