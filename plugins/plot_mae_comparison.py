"""Plot same-task MAE before/after CARMA from results_180.csv.

Run locally after the completed experiment has been downloaded. No test-set
selection or retraining is performed here.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path(r'C:\Users\32113\OneDrive\Desktop\baseline\CARMA_results')
CSV = ROOT / 'results_180.csv'
MODELS = ('MM-TSFlib', 'CFA', 'SpecTF', 'TaTS', 'Aurora')
COLORS = {'MM-TSFlib': '#1b9e77', 'CFA': '#d95f02', 'SpecTF': '#7570b3',
          'TaTS': '#e7298a', 'Aurora': '#66a61e'}


def read_rows():
    with CSV.open(newline='', encoding='utf-8-sig') as stream:
        source = list(csv.DictReader(stream))
    if len(source) != 180:
        raise ValueError(f'Expected 180 task rows, got {len(source)}')
    rows = []
    for row in source:
        if row['status'] != 'ok':
            continue
        before = float(row['base_mae'])
        after = float(row['carma_mae'])
        if not math.isfinite(before) or not math.isfinite(after) or before < 0 or after < 0:
            raise ValueError(f'Invalid MAE for {row["model"]}/{row["domain"]}/{row["horizon"]}')
        rows.append(dict(model=row['model'], domain=row['domain'],
                         horizon=int(row['horizon']), before=before, after=after,
                         enabled=row['enabled_by_validation'] == '1'))
    return rows, len(source) - len(rows)


def heatmap(rows, missing):
    domains = sorted({r['domain'] for r in rows})
    horizons = {d: sorted({r['horizon'] for r in rows if r['domain'] == d}) for d in domains}
    lookup = {(r['domain'], r['horizon'], r['model']): r for r in rows}
    changes = [(r['after'] / r['before'] - 1) * 100 for r in rows if r['before'] > 0]
    limit = max(5, min(50, float(np.percentile(np.abs(changes), 95)))) if changes else 5
    fig, axes = plt.subplots(3, 3, figsize=(18, 16), constrained_layout=True)
    cmap = plt.get_cmap('RdYlGn_r').copy()
    cmap.set_bad('#e5e7eb')
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    last = None
    for ax, domain in zip(axes.flat, domains):
        hs = horizons[domain]
        matrix = np.full((len(MODELS), len(hs)), np.nan)
        for i, model in enumerate(MODELS):
            for j, horizon in enumerate(hs):
                row = lookup.get((domain, horizon, model))
                if row and row['before'] > 0:
                    matrix[i, j] = 100 * (row['after'] / row['before'] - 1)
        last = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
        ax.set_title(domain.replace('Health_US', 'Health'), fontsize=14, weight='bold')
        ax.set_xticks(range(len(hs)), [str(h) for h in hs])
        ax.set_yticks(range(len(MODELS)), MODELS, fontsize=9)
        ax.set_xlabel('Forecast horizon')
        for i in range(len(MODELS)):
            for j in range(len(hs)):
                value = matrix[i, j]
                label = 'n/a' if np.isnan(value) else f'{value:+.1f}%'
                ax.text(j, i, label, ha='center', va='center', fontsize=9,
                        color='black', weight='bold' if np.isfinite(value) and abs(value) > 8 else 'normal')
        ax.set_xticks(np.arange(-.5, len(hs), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(MODELS), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.2)
        ax.tick_params(which='minor', bottom=False, left=False)
    for ax in list(axes.flat)[len(domains):]:
        ax.axis('off')
    if last is not None:
        cbar = fig.colorbar(last, ax=axes, shrink=.65, pad=.015)
        cbar.set_label('MAE change vs unchanged baseline (%)\nNegative = improvement; colors clipped at 95th percentile')
    fig.suptitle(f'CARMA vs unchanged baselines: test MAE by task ({len(rows)}/180 complete, {missing} missing)',
                 fontsize=18, weight='bold')
    path = ROOT / 'MAE_九领域五模型四步长_相对变化.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return path


def scatter(rows, missing):
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    positive = [r for r in rows if r['before'] > 0 and r['after'] > 0]
    values = [v for r in positive for v in (r['before'], r['after'])]
    if not values:
        raise ValueError('No positive MAE pairs to plot')
    lower, upper = min(values) * .8, max(values) * 1.25
    ax.plot([lower, upper], [lower, upper], '--', color='#555555', lw=1.2,
            label='No change')
    for model in MODELS:
        subset = [r for r in positive if r['model'] == model]
        ax.scatter([r['before'] for r in subset], [r['after'] for r in subset],
                   label=model, color=COLORS[model], s=43, alpha=.78,
                   edgecolors='white', linewidths=.4)
    ax.set(xscale='log', yscale='log', xlim=(lower, upper), ylim=(lower, upper),
           xlabel='Unchanged baseline test MAE', ylabel='CARMA test MAE')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, which='both', alpha=.18)
    ax.legend(loc='best', frameon=True)
    improved = sum(r['after'] < r['before'] - 1e-12 for r in rows)
    worsened = sum(r['after'] > r['before'] + 1e-12 for r in rows)
    unchanged = len(rows) - improved - worsened
    ax.set_title(f'Test MAE: baseline vs CARMA | lower is better\n'
                 f'{improved} improved · {worsened} worsened · {unchanged} unchanged · {missing} missing')
    ax.text(.03, .97, 'Below dashed line = CARMA improves MAE', transform=ax.transAxes,
            va='top', fontsize=10, bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    path = ROOT / 'MAE_五模型_修改前后散点对比.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return path


def main():
    rows, missing = read_rows()
    heat = heatmap(rows, missing)
    dots = scatter(rows, missing)
    by_model = defaultdict(list)
    for row in rows:
        by_model[row['model']].append(row)
    lines = ['# MAE 修改前后对比图', '',
             '两张图均使用本次 ReadGPT 对齐数据、同一模型同一领域同一步长的测试预测。原值来自未接 CARMA 的模型输出；新值来自验证集决定是否启用的 CARMA 输出。',
             '', f'完整任务：{len(rows)}/180；未完成：{missing}。MAE 越低越好，热力图负值表示改善。',
             '', f'- 热力图：`{heat.name}`', f'- 散点图：`{dots.name}`',
             '- 逐项原值、新值：`results_180.csv` 与 `CARMA_五模型九领域MSE_MAE.xlsx`。',
             '', '| 模型 | MAE 下降 | 上升 | 不变 | 已完成 |', '|---|---:|---:|---:|---:|']
    for model in MODELS:
        group = by_model[model]
        down = sum(r['after'] < r['before'] - 1e-12 for r in group)
        up = sum(r['after'] > r['before'] + 1e-12 for r in group)
        lines.append(f'| {model} | {down} | {up} | {len(group)-down-up} | {len(group)} |')
    lines += ['', '图中的百分比为 `(CARMA MAE / 原 MAE − 1) × 100%`。色阶在变化绝对值的第 95 百分位截断，单元格标注仍为实际百分比。',
              '第一版 `TimeMMD_五模型九领域MAE.xlsx` 与本次 ReadGPT 对齐数据不同，不用于计算图中的差值。', '']
    (ROOT / 'MAE_对比说明.md').write_text('\n'.join(lines), encoding='utf-8')
    print(heat)
    print(dots)


if __name__ == '__main__':
    main()
