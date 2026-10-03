# GANF 思想迁移：共享后置分布插件的原理、实验与设置

**整理日期：2026-10-03。** 本文核对了 GANF 原论文、归档参考代码，以及当前共享后置插件和参数实验的实际实现。结果来自已经完成的实验记录，本文整理没有新增训练。

## 1. 当前实现的准确定位

当前插件接在五个底模的预测输出之后：读取历史数值、历史文本和底模未来预测，学习未来预测残差的条件分布，再产生受控修正。

五个底模为 **TaTS、MM-TSFlib、SpecTF、CFA、Aurora**。底模冻结，插件训练不更新底模参数，也没有读取底模内部隐藏状态。

“共享”具体指：**同一领域 × 同一步长 × 同一种子，五个模型共用一个插件及其权重**。五个模型各自有验证集选出的修正系数 α。不同领域、不同步长、不同种子分别训练，因此正式实验是 36 × 3 = 108 个插件训练实例，而不是整个实验共用一个权重文件。

当前方法可以称为“GANF 思想启发的多模态条件残差校正”。实验尚未完成底模内部的深度融合与联合训练；也尚未证明文本语义带来了额外收益。

## 2. GANF 原论文：需要理解的机制

### 2.1 论文与任务

GANF 全称 Graph-Augmented Normalizing Flow，论文为 *Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series*，作者 Enyan Dai、Jie Chen，发表于 **ICLR 2022**。其任务是多条时序的无监督异常检测：学习正常数据的联合密度，用低密度识别异常。[论文信息](https://arxiv.org/abs/2202.07857)

### 2.2 条件密度、依赖图与训练

归一化流用可逆变换把数据映射到简单分布；变量替换公式使密度可计算，逆变换使采样可行：

$$
\log p(x\mid d)=\log q(f(x;d))+
\log\left|\det\frac{\partial f(x;d)}{\partial x}\right|.
$$

GANF 将各条时序作为图节点，用 DAG 表达父节点条件依赖，并沿时间展开：

$$
p(\mathcal X)=\prod_i\prod_t
p(x_t^i\mid \mathrm{pa}(x^i)_{1:t},x^i_{1:t-1}).
$$

RNN 编码历史，图依赖编码器汇总父节点状态和自身过去状态；条件流据此估计密度。原论文使用连续邻接矩阵，并以无环约束联合学习图与网络：

$$
h(A)=\operatorname{tr}\!\left(e^{A\circ A}\right)-n=0,
\qquad
\mathcal L_c=\mathcal L_{\mathrm{NLL}}+\lambda h(A)+\frac{c}{2}h(A)^2.
$$

异常检测、密度估计和图随时间的变化构成原论文实验内容。[原论文第 3、5、6 节](https://arxiv.org/html/2202.07857)

官方实现可查看 LSTM、图聚合以及 MAF/RealNVP 条件密度接口。[官方代码](https://github.com/enyandai/ganf)

### 2.3 本项目迁移的思路

本项目希望回答“历史文本和数值条件下，未来可能怎样变化”。迁移时采用三条设计原则：

1. 输出一个可学习的条件分布，而非仅增加一个确定性线性修正。
2. 在分布变量之间表达依赖，避免把未来各部分误差完全独立处理。
3. 将数值与文本作为条件输入，比较加入文本前后对残差分布的解释能力。

这些是本项目的设计解释。实际模块与 GANF 原论文的图节点、图学习方式和任务均有差异，下面分别说明。

## 3. 从 GANF 到当前插件：哪些保留、哪些改造

| 项目 | GANF 原方法 | 当前插件实际实现 |
|---|---|---|
| 学习对象 | 多条时序观测的联合密度 | 底模未来残差的低维系数条件密度 |
| 输出用途 | 异常评分、密度分析 | 点预测校正与分布摘要 |
| 图节点 | 各条时序 | 未来残差的 DCT 系数 |
| 图结构 | 通过约束学习 DAG | 固定系数顺序，严格下三角掩码保证无环 |
| 可学习部分 | 图与网络参数 | 下三角边权、编码器、流、语义门控与修正幅度 |
| 条件 | 数值历史与图父节点 | 数值历史、底模预测、历史文本 |
| 流结构 | 官方代码提供 MAF/RealNVP | 自定义单个三角自回归仿射流 |
| 训练目标 | 密度似然与 DAG 约束 | 点误差、系数密度、语义效用、修正正则 |
| 参数共享 | 原方法内部的节点共享 | 本项目跨五个冻结底模共享插件 |

当前没有复现原论文的自由拓扑 DAG 搜索、增广拉格朗日训练或原始异常检测实验。当前边权是同一插件内的全局可学习参数，并非随每个样本变化的动态邻接矩阵。边权也没有稀疏约束或因果识别保证。

## 4. 模块的计算过程

### 4.1 统一输入和两层尺度

每个 batch 的统一接口为：

| 输入 | 形状 | 含义 |
|---|---|---|
| base | B × H × 1 | 冻结底模预测 |
| history | B × L × 1 | 历史 OT 数值 |
| text | B × L × 768 | 历史文本 BERT 向量 |
| text_mask | B × L | 有效文本标记 |
| target | B × H × 1 | 训练/评估目标，推理不需要 |

外层统一标准化使用原数值训练段的均值和标准差。插件内部再用当前历史窗口均值 ℓ 和标准差 s 构造局部尺度：

$$
\ell=\operatorname{mean}(x_{\mathrm{history}}),\quad
s=\max(\operatorname{std}(x_{\mathrm{history}}),0.1).
$$

数值历史输入 GRU 前变为 (x−ℓ)/s；残差分布学习 (y−base)/s；生成修正后乘回 s。ℓ、s 均 detach。

**这一步乘回 s 是恢复局部尺度，并不是恢复原始物理单位。最终 MSE/MAE 仍在外层训练段标准化空间计算。**

### 4.2 历史文本与数值条件

数值编码器为 hidden=32 的 GRU。文本使用冻结 bert-base-uncased：

- 只读 fact 列；当前插件不使用 preds 列。
- 最长 256 token，按 attention_mask 对最后一层 token 表示取均值，再 L2 归一化。
- 每个领域缓存一次 768 维向量。
- 空字符串与 No information available 标记无效。
- 文本先投影到 32 维；结合 GRU 状态、文本与数值交互项、数值状态变化幅度，为 lag=0、1、2 的候选文本计算权重。
- 获胜配置在有效历史时间点均匀聚合；时间注意力作为参数变体参加过筛选。

“均匀聚合”只描述时间维聚合；候选 lag 的权重仍是可学习的。

数据读取限制文本位于预测原点之前的历史行。**代码没有独立验证每条 fact 的真实发布时间或检索来源是否含未来内容**，因此因果可用性仍依赖数据制作过程；不能仅凭切片就保证没有文本泄漏。

### 4.3 用 DCT 表达未来残差轨迹

令 r=(y−base)/s，H 为预测长度，Φ 为 H × K 的正交 DCT 基。代码中的系数和重建为：

$$
a=\frac{r\Phi}{\sqrt H},\qquad
\tilde r=\sqrt H\,a\Phi^\top,\qquad K=\min(K_{\mathrm{配置}},H).
$$

保留前 K 个系数，允许插件以较低维度学习整段未来误差的水平和缓慢变化。获胜配置 K=4。

重要限制：当 K<H 时，高频残差被舍弃；重建的轨迹处于低维子空间。**代码计算的是 K 维系数空间的密度，不能称为完整 H 维未来轨迹的非退化精确密度。**这是当前分布建模能力的边界。

### 4.4 三角图流

代码定义：

$$
A_{kj}=\mathbf 1(j<k)\,\sigma(w_{kj}).
$$

每个系数的 shift 和 log_scale 由条件向量、节点 embedding、带权的前序系数计算：

$$
a_k=\mu_k(a_{<k},c)+\exp(u_k(a_{<k},c))z_k,\quad z_k\sim\mathcal N(0,1),
$$

其中 u=0.6 tanh(network_output)，限制尺度变化。条件密度负对数为：

$$
\operatorname{NLL}(a\mid c)=
\sum_k\left[
\frac12\left((a_k-\mu_k)e^{-u_k}\right)^2+
\frac12\log(2\pi)+u_k
\right].
$$

严格三角结构使 Jacobian 行列式可计算，采样按系数顺序生成。边权初始 logit=-2，最后一层输出参数初始化为零，使初始流为恒等映射。

这个设计避免了显式无环约束优化，但固定了系数顺序。所得图表示残差频率系数之间的统计依赖，不能直接解释成领域实体或文本事件之间的因果图。

### 4.5 数值条件与多模态条件共用一个流

同一流分别接受两个条件：

- c_ts：最后一个 GRU 状态与底模预测的 DCT 特征。
- c_joint：数值状态、聚合文本及底模预测特征。

于是得到两个系数分布，并用语义门控 g 混合：

$$
p(a\mid c)=(1-g)p_{\mathrm{ts}}(a)+g p_{\mathrm{joint}}(a).
$$

g 由数值状态、文本摘要、文本覆盖率和余弦相似度生成；没有有效文本时归零。两个分支共享流参数，但条件编码网络分别学习。

训练阶段以加入文本前后的 NLL 差构造效用监督：

$$
g^*=\sigma\left(
\frac{\operatorname{NLL}_{\mathrm{ts}}-\operatorname{NLL}_{\mathrm{joint}}-0.05}{0.25}
\right).
$$

教师 g* detach；真实未来只用于训练损失，推理 g 来自历史条件。这是一个学习目标，并不等于语义贡献已经被实验验证。

### 4.6 从分布到点预测

每个条件用 16 个固定 Sobol 正负配对噪声采样，重建残差轨迹，得到混合均值 m 与混合方差 v；方差包含分支内部方差及分支均值差异。

插件先给出原始候选修正：

$$
\Delta=s\cdot C\,\tanh(m)\,
\frac{\sigma(\gamma)}{1+\beta v},
\qquad
\hat y_{\mathrm{raw}}=base+\Delta.
$$

最后加入每个模型自己的验证门控：

$$
\hat y_m=base_m+\alpha_m\Delta_m.
$$

C 是修正上限参数，γ 是可训练幅度 logit，β 控制方差收缩。获胜配置 C=1、γ 初值=0、β=0；γ 后续可训练，最终增益不固定为 0.5。

模型共享插件权重，但各模型输入的 base 不同，所以修正量可以不同。获胜配置没有底模 ID embedding；只有相关参数变体启用了该条件。

分布均值经过 tanh、幅度缩放与 α 后用于点预测，**最终预测并不是未经变换的分布期望**。当前优化 MSE+MAE，尚未实现专门的条件中位数输出或经过校准的预测区间。

### 4.7 总损失和语义显示

获胜配置的训练目标为：

$$
\mathcal L=
\operatorname{MSE}(\hat y_{\mathrm{raw}},y)
+0.2\operatorname{MAE}(\hat y_{\mathrm{raw}},y)
+0.03\mathcal L_{\mathrm{NLL}}
+0.01\mathcal L_{\mathrm{utility}}
+0.005\operatorname{mean}(\hat y_{\mathrm{raw}}-base)^2.
$$

NLL 使用混合密度并除以 K；效用损失是有文本样本的门控二元交叉熵。α 不参与 batch 反向传播，它用于验证集 checkpoint 选择及最终评估。

代码的 semantic_profile 提供文本效用、预计方向、预计变化、分布不确定度和文本条件引起的均值偏移。这些是数值摘要，尚未建立“具体事件→趋势解释”的自然语言展示。其不确定度也未校准到最终 α 修正后的预测空间，不能当作可信区间。

## 5. 数据设置与公平性

### 5.1 九领域和预测长度

使用 TaTS-main-change-Readgpt 所提供数据，经 benchmark/prepare_readgpt.py 整理。

| 领域 | 预测长度 H | 初筛长度 |
|---|---|---:|
| Agriculture | 6、8、10、12 | 6 |
| Climate | 6、8、10、12 | 8 |
| Economy | 6、8、10、12 | 8 |
| Energy | 12、24、36、48 | 36 |
| Environment | 48、96、192、336 | 336 |
| Health | 12、24、36、48 | 24 |
| Security | 6、8、10、12 | 10 |
| SocialGood | 6、8、10、12 | 8 |
| Traffic | 6、8、10、12 | 8 |

共同输入长度 L=24，单变量 OT。删除缺失 OT，按日期排序，重复日期保留最后一条；源文件和整理后文件的 hash 记录在 manifest 中。整体按时间使用约 70% 训练、10% 验证、20% 测试，实际边界包含整数取整。

### 5.2 底模训练段和插件训练段不同

底模原数值训练段用于训练底模并拟合全局标准化。插件使用保存的底模验证预测构造样本：

1. 按时间取验证预测窗口前约 70% 为插件 fit。
2. 其余约 30% 为插件 holdout。
3. 从 fit 尾部再删 H−1 个预测原点，避免 fit 与 holdout 的目标区间重叠。
4. 原测试段用于锁定配置后的最终评估。

例如 H=12 时，删除最后 11 个 fit 原点。历史输入可以与相邻窗口重叠，隔离的是预测目标重叠。

**插件消耗了额外的验证段监督，这是方法成本，应在论文比较中披露。**底模过去也可能使用该验证段选择 checkpoint，因此插件 holdout 不是整个流程从未使用过的数据。

### 5.3 哪些一致，哪些不同

| 项目 | 五模型之间的关系 |
|---|---|
| 插件读取的 OT、文本、时间边界和目标 | 一致 |
| 插件历史长度、预测长度、全局标准化 | 一致 |
| 插件架构、训练协议和获胜超参数 | 一致 |
| 同一领域/步长/种子的插件权重 | 共用 |
| 底模原始预测 | 不同 |
| 底模内部结构、原始训练设置 | 沿用此前冻结模型；本轮未重新统一训练 |
| 模型 ID 条件 | 获胜配置关闭 |
| 模型专属 α | 分别在 holdout 选择 |
| 文本可用性 | 共用历史缓存和 mask |

因此，这轮比较控制的是“同一后置插件对不同冻结底模的作用”，并没有重新验证五个底模内部训练预算完全相同。

## 6. 参数实验：实际执行内容

### 6.1 14 个配置

下表完整对应 experiment.py::variants()。U 表示均匀时间聚合，A 表示时间注意力；K 实际不超过 H。

| 配置 | C | γ初值 | β | K上限 | NLL | 效用 | 聚合 | ID |
|---|---:|---:|---:|---:|---:|---:|---|---|
| control | 0.5 | −2 | 1 | 4 | 0.03 | 0.01 | U | 否 |
| amplitude_1 | 1 | 0 | 1 | 4 | 0.03 | 0.01 | U | 否 |
| amplitude_2 | 2 | 0 | 1 | 4 | 0.03 | 0.01 | U | 否 |
| no_variance_shrink | 1 | 0 | 0 | 4 | 0.03 | 0.01 | U | 否 |
| coeff_8 | 1 | 0 | 0.5 | 8 | 0.03 | 0.01 | U | 否 |
| coeff_16 | 1 | 0 | 0.5 | 16 | 0.03 | 0.01 | U | 否 |
| nll_010 | 1 | 0 | 0.5 | 4 | 0.1 | 0.01 | U | 否 |
| nll_030 | 1 | 0 | 0.5 | 4 | 0.3 | 0.01 | U | 否 |
| utility_005 | 1 | 0 | 0.5 | 4 | 0.1 | 0.05 | U | 否 |
| time_attention | 1 | 0 | 0.5 | 4 | 0.1 | 0.01 | A | 否 |
| model_condition | 1 | 0 | 0.5 | 4 | 0.1 | 0.01 | U | 是 |
| combined | 2 | 0 | 0.25 | 8 | 0.1 | 0.05 | A | 是 |
| numeric_ablation | 2 | 0 | 0.25 | 8 | 0.1 | 0.05 | A | 是 |
| time_shuffle_ablation | 2 | 0 | 0.25 | 8 | 0.1 | 0.05 | A | 是 |

前三类组合/消融配置的 MAE 权重=0.5、修正正则=0.001，并按每个底模 fit 基础误差平衡点损失。其余配置分别为 0.2、0.005、不平衡。

numeric_ablation 去掉文本，time_shuffle_ablation 只在当前历史窗口内共同打乱 text/mask。它保留文本集合，主要检验时间对应关系。多个配置同时改变了多个参数，不能当作严格单因素实验。

### 6.2 训练设置

| 参数 | 实际值 |
|---|---|
| 优化器 | AdamW |
| 学习率 | 0.001，固定 |
| weight_decay | 0.0001 |
| batch_size | 256 |
| 梯度范数裁剪 | 1.0 |
| hidden / text_dim | 32 / 768 |
| 条件采样数 | 16 |
| 文本 lag | 0、1、2 |
| 初筛/复核最多 epoch | 24 |
| 正式最多 epoch | 80 |
| minimum_epochs | 8 |
| patience | 8 |
| checkpoint 改善阈值 | 综合相对分数降低超过 0.00001 |
| 正式随机种子 | 2026、2027、2028 |
| DataLoader workers | 0 |
| torch CPU threads | 4 |
| 实际服务器 | RTX 4090 24GB，16核 CPU、120GB 内存 |
| 已使用环境 | PyTorch 2.5.1 / CUDA 12.4、transformers 4.40.2 |

底模与 BERT 均不更新。每个插件 fit 合并五模型样本并 shuffle；底模样本数量一致，合并并不会创造新的独立时间观测。

### 6.3 epoch 与 α 怎么选

每个 epoch，对每个底模枚举 α∈{0、0.25、0.5、0.75、1}。只有 holdout MSE、MAE 都严格优于原底模的非零 α 才可入选，再最小化：

$$
S_m=\frac12\left(
\frac{\mathrm{MSE}_{m,\mathrm{corrected}}}{\mathrm{MSE}_{m,\mathrm{base}}}
+
\frac{\mathrm{MAE}_{m,\mathrm{corrected}}}{\mathrm{MAE}_{m,\mathrm{base}}}
\right).
$$

共享 checkpoint 按五模型平均 S_m 选择。保留初始化 epoch=0 作为回退选项。注意：**验证集双指标改善并不保证测试集也改善。**

### 6.4 实验阶段与次数

1. **初筛**：14 配置 × 九领域各一个步长 × seed2026，126 次训练。
2. **复核**：晋级三个真实文本配置 no_variance_shrink、amplitude_2、combined；九领域 × seed2027，27 次训练。
3. **全局锁定**：仅根据隔离 holdout，选 no_variance_shrink。
4. **正式训练**：九领域 × 四步长 × 三种子，108 次训练。
5. **测试**：108 个插件分别评估五个底模，540 条种子记录。
6. **汇总**：按模型/领域/长度取三种子指标均值和标准差，得到 180 项结果。
7. **赢家消融**：固定赢家参数，九领域 × 两种子 × 去文本/窗口内打乱文本，另 36 次训练。

无文本消融在初筛综合分数更优，但实验事先将其定位为诊断对照，未允许其晋级多模态插件候选。这个选择限制应明确披露。

正式预算最初 40 epoch。观察训练/holdout 记录后，在最终测试前延长到 80；部分触及预算且最优点靠后的任务按同种子重训，旧记录保留在 budget40。上述次数不包含额外预算重训，不能把 297 当作全部优化运行次数。

## 7. 两个流程图

### 7.1 单个样本

~~~mermaid
flowchart TD
    X[历史OT：训练段全局标准化] --> N[窗口局部标准化与GRU]
    T[历史fact文本] --> B[冻结BERT与有效mask]
    B --> A[数值条件下的lag对齐与文本聚合]
    N --> A
    P[冻结底模未来预测] --> D[DCT预测特征]
    N --> C1[数值条件编码]
    D --> C1
    N --> C2[多模态条件编码]
    A --> C2
    D --> C2
    C1 --> F[共用三角图流]
    C2 --> F
    A --> G[语义效用门控]
    N --> G
    F --> R[16次配对采样与残差轨迹重建]
    G --> R
    R --> Q[均值方差与受限修正]
    Q --> AL[模型专属验证α]
    P --> OUT[最终预测]
    AL --> OUT
~~~

### 7.2 完整实验

~~~mermaid
flowchart TD
    A[保存原代码与冻结底模预测] --> B[统一数据接口与文本缓存]
    B --> C[fit尾部删除H减1个原点]
    C --> D[14配置初筛：126次训练]
    D --> E[3个真实文本配置复核：27次]
    E --> F[仅按holdout锁定配置]
    F --> G[36任务乘3种子：108次正式训练]
    G --> H[检查训练预算与保存最佳checkpoint]
    H --> I[固定epoch和各底模α]
    I --> J[测试540条记录]
    J --> K[180项均值标准差与MSE和MAE图]
    F --> L[固定赢家参数的文本消融：36次]
    L --> M[检查语义贡献]
    K --> R[归档报告和权重]
    M --> R
~~~

## 8. 完成的结果及其解释

下表是每模型 36 项任务相对原底模的误差变化宏平均；负数表示下降。三种子平均的是评估指标，没有做三模型预测集成。

| 模型 | 平均MSE变化 | 平均MAE变化 | 实际启用/36 | 双指标改善 | 至少一项退化 |
|---|---:|---:|---:|---:|---:|
| TaTS | −2.68% | −2.02% | 19 | 16 | 3 |
| MM-TSFlib | −5.32% | −3.76% | 20 | 19 | 1 |
| SpecTF | −0.59% | −0.40% | 11 | 10 | 1 |
| CFA | −0.42% | −0.25% | 10 | 6 | 4 |
| Aurora | −4.53% | −2.11% | 16 | 11 | 5 |

合计 62/180 双指标改善，14/180 至少一项退化，104/180 在统计容差内持平。启用数排除 epoch=0 恒等 checkpoint；这些 checkpoint 的配对浮点采样会产生极小非零修正。变化小于 0.00001% 按持平统计，原记录保留。

本轮增大 NLL 至 0.1/0.3、效用至 0.05 已实际参加实验，但全局赢家仍使用 0.03/0.01。收益更可能与扩大修正幅度、取消方差收缩有关；赢家同时改变多个参数，所以仍需严格匹配的单因素对照才能拆分贡献。

### 8.1 语义贡献是否成立？

赢家参数匹配的 holdout 消融如下：

| 输入 | MSE变化 | MAE变化 | 综合变化 |
|---|---:|---:|---:|
| 真实文本 | −2.28% | −1.32% | −1.80% |
| 无文本 | −2.72% | −1.67% | −2.19% |
| 窗口内打乱文本 | −2.31% | −1.32% | −1.81% |

真实文本没有胜过无文本，打乱后的表现也很接近。因此当前可以报告预测校正收益，**不能将其归因为已验证的深度语义融合收益**。

### 8.2 为什么改善有限？

- 低维 DCT 修正难以覆盖高频误差，尤其长预测长度；Environment 的 K4 残差能量占比领域平均约 30.75%。
- 五个底模的误差规律不同，共享权重与共享 checkpoint 可能产生目标冲突。
- 获胜配置没有模型身份条件，主要靠底模预测特征区分误差模式。
- 文本聚合和效用教师较弱；教师来自同一学习系统的 NLL 优势，不是独立事件标签。
- α 门控、tanh 与低维修正提高了稳定性，同时使大量任务回退原预测。
- MSE 与 MAE 对最优点估计的偏好不同，简单混合损失没有专门学习条件中位数。
- 当前没有把分布学习反馈到底模隐藏特征，难以纠正底模内部错失的语义信息。

这些是从实现及消融结果提出的机制解释，不能当作各因素已经被实验单独证明。

### 8.3 收敛与统计限制

19/108 正式训练达到 80 epoch 上限，最优 epoch 接近上限；它们不能标为完全收敛。早停也只是验证指标的停止规则。

Security/12 隔离后每底模仅剩 3 个 fit 原点。五模型合并产生 15 条残差样本，但对应同一组 3 个时间原点，不能视为 15 个独立观测。三种子标准差描述优化随机性，不是独立时间块的置信区间。

测试集在先前实验已被查看，本轮是探索性复现。当前也没有完整报告 CRPS、区间覆盖率、PIT 等分布校准指标，所以尚不足以声称已经学到了可靠的完整未来分布。

## 9. 代码位置与设置流程

项目代码根目录：

C:/Users/32113/OneDrive/Desktop/baseline/experiment_vcs/server_baseline

服务器实验根目录：

/root/autodl-tmp/carma_workspace

| 文件/目录（相对代码根） | 作用 |
|---|---|
| benchmark/prepare_readgpt.py | 固定数据整理、时间边界与manifest |
| plugins/make_adapter_dataset.py | 对齐冻结预测、目标与历史，生成fit/holdout/test |
| plugins/semflow/data.py | 统一读取、历史文本切片和BERT缓存 |
| plugins/semflow_tuning_v2/model.py | DCT、图流、条件融合、损失与修正 |
| plugins/semflow_tuning_v2/experiment.py | 参数定义、审计、训练、选择和评估 |
| plugins/semflow_tuning_v2/winner_ablation.py | 固定赢家参数的文本消融 |
| plugins/semflow_tuning_v2/report.py | 指标汇总、图和分析 |
| plugins/semflow_tuning_v2/build_workbook.mjs | Excel报告生成 |
| plugins/references/GANF-main/models/GANF.py | 原GANF参考结构 |
| plugins/references/GANF-main/models/NF.py | 原条件流参考实现 |

### 9.1 运行前的数据布局

~~~text
carma_workspace/
  benchmark/readgpt_data/<domain>/
    <domain>.csv
    manifest.json
  models/bert-base-uncased/
  plugins/adapters/<model>/<domain>/<H>/
    fit.npz
    holdout.npz
    test.npz
  plugins/semflow/cache/<domain>_bert.npz
  plugins/semflow_tuning_v2/
~~~

每个 npz 包含 base_pred、target、history 等数组。当前训练从 semflow/data.py 动态读取BERT缓存，不使用早期 npz 中的旧文本向量。

make_adapter_dataset.py 会核对数组形状和训练段标准化后的目标；若预测与目标的时间或尺度不一致，会抛错。它依赖此前各底模预测文件，不能只下载插件代码就直接运行。

### 9.2 复现命令

以下是**复现说明，本次整理文档没有执行这些训练指令**。在具备上述数据、冻结预测与BERT的服务器工作目录运行：

~~~bash
cd /root/autodl-tmp/carma_workspace
/root/miniconda3/bin/python -u plugins/semflow_tuning_v2/experiment.py --stage all --screen-epochs 24 --final-epochs 80 --minimum-epochs 8 --patience 8 --batch-size 256 --lr 0.001
~~~

也可以顺序运行 stage：audit → screen → confirm → fit → evaluate。确认完成后进行赢家消融：

~~~bash
/root/miniconda3/bin/python -u plugins/semflow_tuning_v2/winner_ablation.py
~~~

生成图和指标报告：

~~~bash
/root/miniconda3/bin/python plugins/semflow_tuning_v2/report.py --directory plugins/semflow_tuning_v2/outputs
~~~

已完成的 fit_result.json 会被跳过，正式测试在配置锁定和正式训练完成后进行。若需一轮全新实验，应为新实验建立独立工作副本与输出目录，避免复用旧缓存记录；脚本不会对所有训练参数自动比较并强制失效旧结果。

### 9.3 输出与版本记录

每个训练实例保存 module.pt、fit_result.json、逐轮trace、best_epoch和五个底模 α。最终种子结果为 test_seed_results.csv；180 项汇总为 results_180_mean_std.csv。

本地交付目录：

C:/Users/32113/OneDrive/Desktop/baseline/GANF_parameter_experiment_20261003

其中 experiment_artifacts.tar.gz 已归档权重和训练记录；服务器另有逐模型测试预测 npz，未包含在本地权重归档中。

截至本文核对，实验结果记录提交为 **9b2151b**。历史关键提交：2c6c315 参数实验实现，b5e4d91 预算扩展和赢家消融。代码仓库为 [TimeMMD-five-model-baselines](https://github.com/Lxyyy10083/TimeMMD-five-model-baselines)。

## 10. 与用户原始目标的对应关系

| 原始目标 | 当前完成程度 |
|---|---|
| 历史数值/文本条件下学习未来分布 | 已实现低维残差系数条件分布；完整轨迹分布尚未完成 |
| 五个模型可使用同一接口 | 已完成共享后置接口 |
| 每个模型内部深度融合 | 尚未完成 |
| 学习图结构依赖 | 已实现固定顺序的残差系数边权学习 |
| 语义显示 | 有数值摘要，缺乏具体事件的可解释展示 |
| 全部任务MSE与MAE下降 | 未达到，有退化与持平 |
| 五个模型全部充分收敛 | 未能确认，19组仍触及预算上限 |
| 证明文本提供额外收益 | 当前消融不支持 |
| 原baseline保持可追溯 | 冻结预测与新插件实验分开，并有Git记录 |

下一步若进行内部融合，应为每个底模建立独立插件实例，通过底模隐藏状态条件化未来分布，让分布损失进入联合训练；同时保留共享后置版作为对照。应优先验证真实文本相对无文本的额外贡献，并补充分布校准评估，再判断是否实现了语义条件下的未来分布学习。

---

## 附录：已交付结果入口

- [180项结果CSV](C:/Users/32113/OneDrive/Desktop/baseline/GANF_parameter_experiment_20261003/results_180_mean_std.csv)
- [540项种子记录](C:/Users/32113/OneDrive/Desktop/baseline/GANF_parameter_experiment_20261003/test_seed_results.csv)
- [参数设置JSON](C:/Users/32113/OneDrive/Desktop/baseline/GANF_parameter_experiment_20261003/experiment_config.json)
- [MSE对比图](C:/Users/32113/OneDrive/Desktop/baseline/GANF_parameter_experiment_20261003/MSE_180_cases.png)
- [MAE对比图](C:/Users/32113/OneDrive/Desktop/baseline/GANF_parameter_experiment_20261003/MAE_180_cases.png)
- [训练曲线](C:/Users/32113/OneDrive/Desktop/baseline/GANF_parameter_experiment_20261003/training_curves.png)

