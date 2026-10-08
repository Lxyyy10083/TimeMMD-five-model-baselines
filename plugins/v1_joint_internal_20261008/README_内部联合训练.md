# V1内部联合训练：代码阅读与实验入口

本版本按“插件放入每个模型框架、模块loss参与主干训练”的要求实现。
代码新增于`plugins/v1_joint_internal_20261008`，旧V1/V1-P1代码与结果保留。
这不是已完成的新实验结果；代码验证不代表180任务已经训练收敛。

## 1. 真实接入位置

|模型|框架内部接入位置|可训练范围|
|---|---|---|
|TaTS/iTransformer|每个encoder层输出，原projection之前|时序主干、原生文本投影、内部模块与flow|
|CFA/PatchTST|每个encoder层输出，原FlattenHead之前|主干与原生CFA adapter保留且可训练，新增内部模块与flow|
|MM-TSFlib/Informer|encoder/decoder每层输出，原decoder.projection之前|主干、原生文本预测分支、新增模块与flow|
|SpecTF|历史/未来复数频谱路径，标量解码和irfft之前|原生频域模型与文本投影、新增复数适配器与flow|
|Aurora|decoder各层输出，原linear_head之前|全部时序encoder/decoder、原生融合及预测头；大型视觉/文本编码器冻结|

`JointForecast`是一个完整可训练的`nn.Module`。原生网络、内部适配器与
条件分布头都是其注册子模块。层hook直接替换原层的输出，后续原预测头
读取修改后的特征；不是将文件中保存的预测传给外置校正器。

## 2. 阅读重点

1. `native.py::JointForecast._install_layers`：四个实数模型的层接入，保留tuple输出结构。
2. `native.py::JointForecast._spectral_fuse`：SpecTF的复数特征接入。
3. `core.py::HistoryCondition`：历史GRU、文本0/1/2滞后对齐、当前文本与缺失mask。
4. `core.py::InternalAdapter`：历史条件改变内部隐藏表示，增量限幅与恒等初始化。
5. `core.py::JointDistribution.finish`：实时原生预测与hidden作为条件，不detach。
6. `core.py::JointDistribution.density`：真实未来数值的条件NLL，包含尺度Jacobian。
7. `core.py::JointDistribution.objective`：点预测与模块损失，共同反传到主干。
8. `train.py::run`：主干和插件分别设学习率，同一个optimizer与backward更新。
9. `train.py`的eligible/converged：不选epoch0、验证平台期、预算上限拒绝完成。
10. `data.py::JointWindows.original_test`：主表对照原始未加插件的180任务。

## 3. 分布及损失

条件为截至当前t的历史数值、历史文本及实时原模型表示。
未来target只用于训练损失，forward条件不读取未来target或未来文本。

从两条完整H维条件残差分布（主干条件/额外语义条件）的概率混合，得到
未来标准化数值分布`Y = native_prediction + local_history_scale * residual`。
H=1表示下一时点分布；H>1同时建模完整未来轨迹，不再截断4个DCT系数。
H=1的条件仿射专家是高斯形状，两专家混合；不宣称可表示任意单点密度。

四层条件仿射耦合流复用已有内部实验的`TrajectoryFlow`。每层只读取不变
坐标及历史条件，因此可逆且Jacobian可计算。采用GANF的条件密度思想，
未直接调用GANF原版异常检测模型、MAF/RealNVP类或自由DAG优化。

预测直接使用混合分布的采样均值。点预测loss能够更新flow；NLL能够
更新时序主干、内部适配器和flow。没有独立于flow的decision头，
没有外部alpha回退，也没有用epoch0初始化作为最终插件结果。

默认设置：hidden=32，每专家16个固定Sobol正负配对样本；损失为
`MSE + 0.2 MAE + 0.03 NLL + 0.01 utility_BCE + 0.005 correction_regularizer`。
NLL前10轮升温。主干lr=1e-4，插件lr=0.003，AdamW weight_decay=1e-4，
batch=16，梯度裁剪1。这些是待验证的联合训练起始设置，不能把旧P1的
验证赢家当作新训练协议下已证明的最优参数。

## 4. 收敛和原始180组对照

至少训练40轮；五模型每个任务分别训练一套主干/插件权重，种子默认
2026/2027/2028，共540次独立联合拟合。不是五模型共享一套插件。
验证评分为`0.5*MSE/initial_val_MSE + 0.5*MAE/initial_val_MAE`，
连续25轮无超过1e-5的改善，且最近验证曲线没有显著继续恢复，才停止。
学习率遇平台减半，下限1e-8。至少完成8轮与NLL升温后才可选检查点。
这是操作性的验证平台期，不是数学全局收敛证明。

2000轮是可记录、可扩展的安全上限；未达平台期则保存NOT_CONVERGED和
resume.pt，不产生COMPLETED、不做最终测试。增加`--epochs`可断点续训，
优化器、调度器和随机状态一起恢复。修改源代码或训练配置则拒绝复用输出。

`run_all.py`先完成全部联合训练，再冻结checkpoint与源码SHA256，最后
读取test。主表使用原始180组的`base_pred`及target，核验测试窗口、历史
窗口和归一化尺度，两个版本使用同一个原始target重算MSE/MAE。
三种子统计为指标均值与标准差，不是预测集成。

## 5. 输入协议的明确边界

当前原生加载器复用`internal_semflow_full_lab/native.py`。它保留五模型
原生架构与CFA文本adapter，但原生侧使用已有内部实验的BERT文本接口；
TaTS/SpecTF历史实验使用过GPT2，部分原始提示构造也不同。因此主表能
回答是否超过原始保存结果，不能把差异全部解释为只有新增插件造成。
恢复逐行原始LLM/提示处理尚未在本版本实现，原LLM资产也不在本机。

SpecTF复用独立`DistributionSpecTF`覆写，其原频段切片路径保留
`use_all_bands=False`。Aurora通过可微模型路径读取内部状态，不调用
`generate/no_grad`；旧flow_match/retriever不用于新条件密度预测路径。

## 6. 代码验证与运行

在baseline目录（本机）：

```powershell
.venv/Scripts/python.exe experiment_vcs/server_baseline/plugins/v1_joint_internal_20261008/verify.py
.venv/Scripts/python.exe experiment_vcs/server_baseline/plugins/v1_joint_internal_20261008/run_all.py --dry-run
```

验证覆盖原生网络内部接入、NLL到主干/适配器梯度、点loss到flow梯度、
实际参数更新、缺失文本、H=1/6/336流可逆性。Aurora的测试使用原生
Transformer层和点头，冻结文本/视觉资产使用小型替身，不声称验证了
正式预训练模型权重。正式训练必须有本地Aurora与BERT资产。

正式环境准备好后，在仓库根目录运行：

```bash
python plugins/v1_joint_internal_20261008/run_all.py \
  --inputs-root /path/to/v1_frozen_inputs_20261006 \
  --bert /path/to/bert-base-uncased \
  --output /path/to/new_joint_outputs
```

原生预训练权重仍由仓库`models/aurora`加载。每个模型独立子进程，
避免原仓库`models/layers/utils`模块名称冲突。只使用调用者指定的设备，
没有自动部署、远程训练或终止其他进程的动作。
