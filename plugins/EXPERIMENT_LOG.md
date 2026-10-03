# CARMA 实验记录

## 版本和隔离

- 2026-10-03，从服务器 `/root/autodl-tmp/baseline` 下载未改动的五模型代码与 benchmark 脚本；源码归档 SHA-256：`7888a1101e6766ab3bd927843c1d550e4252fc07ed6a7c1b27db9d46fe288537`。
- Git 原始快照：`87e149dd88ed4887ae8a46d035adf90b100bdf64`。数据、权重、旧结果不纳入 Git；它们仍在原服务器目录。
- 此后的修改均在本地 `experiment_vcs/server_baseline` Git 仓库提交，再复制到服务器 `/root/autodl-tmp/carma_workspace`。原服务器 `/root/autodl-tmp/baseline` 中的模型源文件没有覆盖。
- 服务器实验目录复用原目录的数据、预训练权重和三模型已有的 ReadGPT 检查点/测试结果。CFA 检查点在实验目录内新训练；Aurora 在实验目录内推断。

## 设计与数据口径

- 数据：`benchmark/readgpt_data`，来自 `TaTS-main-change-Readgpt`。每个领域和预测步长取对应 manifest 的 `seq_len`、训练/验证/测试边界。
- 五模型原有设置由 `benchmark/run_all.py` 生成。CARMA 对所有模型共用 64 维确定性文本哈希，文本只取预测时点以前的 `fact`。数值只用训练段拟合的标准化参数。
- 原模型验证集预测按时间排序导出。CARMA 用验证段前 70% 拟合，后 30% 做早停和启用判断；只有验证 MSE、MAE 同时低于原模型时启用。测试标签只参与最终一次评估。CFA 的原始训练程序未保留检查点，因此在隔离副本中按 ReadGPT 设置重新训练。
- Aurora 加了监督训练适配器后，其结果称 **Aurora+CARMA（监督适配）**，不称零样本。

## 修改历史

| Git 提交 | 内容 |
|---|---|
| `87e149d` | 原服务器源码快照 |
| `05a75b2` | 新增独立 CARMA 模块，无原模型改动 |
| `7a9cb40`, `f19d280` | 在隔离副本导出有序验证预测；清理生成文件 |
| `2ec7052`, `63a6743` | 新增可续跑预测导出与目录修复 |
| `489b89c`, `9ab3b67` | 加入按 ReadGPT 切分构造适配数据与测试评估；精确选择预测文件 |
| `ff02fcb`, `a9162bd` | 全量实验调度；适配 Aurora 命名 |
| `03ee583` | 修正 SpecTF 入口，改用实际运行的 `exp_long_term_forecasting_SpecTF.py` |

## 已完成的先导任务（Agriculture, horizon 6）

先导任务用于确认五模型的数据与计算链路；完整 180 组结果尚未完成。以下均为同一任务测试集归一化尺度指标，正式分析应以全部任务和完成后的聚合表为准。

| 模型 | 原 MSE → CARMA MSE | 原 MAE → CARMA MAE | 验证集是否启用 |
|---|---:|---:|---|
| TaTS | 0.495795 → 0.483323 | 0.597170 → 0.588846 | 是 |
| CFA | 0.167851 → 0.167851 | 0.289601 → 0.289601 | 否 |
| SpecTF | 0.209841 → 0.203519 | 0.299484 → 0.297046 | 是 |
| Aurora | 0.922083 → 0.889748 | 0.690809 → 0.680458 | 是 |
| MM-TSFlib | 3.020123 → 2.954844 | 1.279375 → 1.264180 | 是 |

## 批量运行

在隔离服务器目录执行：

```bash
cd /root/autodl-tmp/carma_workspace
/root/miniconda3/bin/python -u plugins/run_full_carma.py
```

按 `plugins/adapters/<模型>/<领域>/<步长>/result.json` 续跑；每组各有日志，`plugins/runs/carma_status.csv` 记录状态。若任务失败，应先查看对应日志和 Git 提交，再修复/提交/同步并续跑。
