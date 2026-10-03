"""Create the complete 180-case MSE/MAE report and comparison charts."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ORDER = ['Agriculture', 'Climate', 'Economy', 'Energy', 'Environment',
         'Health', 'Security', 'SocialGood', 'Traffic']
MODELS = ['TaTS', 'MM-TSFlib', 'SpecTF', 'CFA', 'Aurora']


def chart(frame, metric, out):
    rows = frame[['domain', 'horizon']].drop_duplicates()
    rows['domain'] = pd.Categorical(rows.domain, ORDER, ordered=True)
    rows = rows.sort_values(['domain', 'horizon'])
    labels = [f'{x.domain} / {x.horizon}' for x in rows.itertuples()]
    pivot = frame.pivot(index=['domain', 'horizon'], columns='model',
                        values=f'{metric}_delta_pct').reindex(
                            pd.MultiIndex.from_frame(rows[['domain', 'horizon']]))[MODELS]
    values = pivot.to_numpy()
    bound = max(2.0, np.nanpercentile(np.abs(values), 98))
    fig, ax = plt.subplots(figsize=(10, 13))
    img = ax.imshow(values, cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
    ax.set_xticks(range(len(MODELS)), MODELS)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title(f'Semantic graph flow: test {metric.upper()} change (%)\nNegative = improvement; 0 = holdout gate disabled')
    ax.set_xticks(np.arange(-.5, len(MODELS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which='minor', color='#dddddd', linewidth=.5)
    fig.colorbar(img, ax=ax, shrink=.65, label='Change (%)')
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=HERE / 'runs/all_results.csv')
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    path = args.input
    frame = pd.read_csv(path)
    if len(frame) != 180 or frame[['model', 'domain', 'horizon']].duplicated().any():
        raise ValueError(f'expected 180 unique cases, got {len(frame)}')
    report = args.output
    report.mkdir(exist_ok=True)
    frame.to_csv(report / 'complete_metrics.csv', index=False, encoding='utf-8-sig')
    chart(frame, 'mse', report / 'MSE_180_cases.png')
    chart(frame, 'mae', report / 'MAE_180_cases.png')
    active = frame[frame.alpha > 0]
    both = (frame.mse_delta_pct < 0) & (frame.mae_delta_pct < 0)
    worse = (frame.mse_delta_pct > 0) | (frame.mae_delta_pct > 0)
    table = frame.groupby('model').agg(
        active=('alpha', lambda x: int((x > 0).sum())),
        mean_mse_change=('mse_delta_pct', 'mean'),
        mean_mae_change=('mae_delta_pct', 'mean'),
        median_mse_change=('mse_delta_pct', 'median'),
        median_mae_change=('mae_delta_pct', 'median'))
    table['both_better'] = frame.groupby('model').apply(
        lambda x: int(((x.mse_delta_pct < 0) & (x.mae_delta_pct < 0)).sum()),
        include_groups=False)
    table['any_worse'] = frame.groupby('model').apply(
        lambda x: int(((x.mse_delta_pct > 0) | (x.mae_delta_pct > 0)).sum()),
        include_groups=False)
    table = table.reindex(MODELS)
    table.to_csv(report / 'model_summary.csv', encoding='utf-8-sig')
    lines = ['# 语义条件图流：完整测试结果', '',
             f'- 共 {len(frame)} 组：9 领域 × 4 步长 × 5 模型。',
             f'- 验证集启用修正：{len(active)}/180；测试集 MSE 与 MAE 同时改善：{int(both.sum())}/180。',
             f'- 测试集至少一个指标变差：{int(worse.sum())}/180；其余为保持原预测或两指标改善。',
             '- 负的变化百分比表示误差下降。所有指标都在训练集拟合的标准化空间计算。',
             '- 门控和修正系数只在验证集选取，测试集未参与选择。因此不能保证全部 180 项在测试集下降。',
             '', '## 各模型汇总', '',
             '| 模型 | 启用 / 36 | 两指标改善 / 36 | 至少一项变差 / 36 | 平均 MSE 变化 | 平均 MAE 变化 |',
             '|---|---:|---:|---:|---:|---:|']
    for name, row in table.iterrows():
        lines.append(f'| {name} | {int(row.active)} | {int(row.both_better)} | '
                     f'{int(row.any_worse)} | {row.mean_mse_change:+.2f}% | '
                     f'{row.mean_mae_change:+.2f}% |')
    lines += ['', '## 图与逐项数据', '',
              '![MSE 对比](MSE_180_cases.png)', '',
              '![MAE 对比](MAE_180_cases.png)', '',
              '逐项完整数值见 `complete_metrics.csv`。每个实验的训练轮数和验证轨迹见服务器 `runs/{domain}/{horizon}/result.json`。']
    (report / '结果分析.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:7]))


if __name__ == '__main__':
    main()
