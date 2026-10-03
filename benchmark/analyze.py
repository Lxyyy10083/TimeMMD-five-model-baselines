"""Validate and visualize completed Time-MMD baseline MSE results."""
from pathlib import Path
import shutil
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get('BENCHMARK_RUNS_DIR', ROOT / 'benchmark' / 'runs'))
DATA = Path(os.environ.get('BENCHMARK_DATA_DIR', ROOT / 'benchmark' / 'data'))
OUT = Path(os.environ.get('BENCHMARK_RESULT_DIR', ROOT / 'benchmark_results'))
MODELS = ['MM-TSFlib', 'CFA', 'SpecTF', 'TaTS', 'Aurora']


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(RUNS / 'results.csv')
    df = df.drop_duplicates(['model', 'domain', 'horizon'], keep='last')
    expected = 9 * 4 * len(MODELS)
    good = df[df.status.eq('ok')].copy()
    if len(good) != expected or good.mse.isna().any():
        missing = expected - len(good)
        raise RuntimeError(f'Incomplete benchmark: {len(good)}/{expected} valid rows; {missing} missing')
    good['horizon'] = good.horizon.astype(int)
    good['mse'] = good.mse.astype(float)
    train_variance = {}
    drift = {}
    for domain in good.domain.unique():
        domain_dir = DATA / domain
        meta = json.loads((domain_dir / 'manifest.json').read_text(encoding='utf-8'))
        series = pd.read_csv(domain_dir / f'{domain}.csv', usecols=['OT']).iloc[:, 0]
        train = series.iloc[:meta['train_end']]
        train_variance[domain] = float(train.var(ddof=0))
        drift[domain] = (float(train.mean()), float(series.iloc[meta['test_start']:].mean()))
    good['raw_mse'] = good.apply(lambda row: row.mse * train_variance[row.domain], axis=1)
    good.to_csv(OUT / 'mse_long.csv', index=False)
    pivot = good.pivot(index=['domain', 'horizon'], columns='model', values='mse')[MODELS]
    pivot.to_csv(OUT / 'mse_table.csv')
    winners = pd.DataFrame({'best_model': pivot.idxmin(axis=1),
                            'best_standardized_mse': pivot.min(axis=1)})
    winners.to_csv(OUT / 'best_by_task.csv')
    good.pivot(index=['domain', 'horizon'], columns='model', values='raw_mse')[MODELS].to_csv(
        OUT / 'raw_mse_table.csv')
    ranks = pivot.rank(axis=1, method='average')
    summary = pd.DataFrame({
        'mean_rank': ranks.mean(),
        'wins': (ranks.eq(1)).sum(),
        'median_relative_mse': pivot.div(pivot.min(axis=1), axis=0).median(),
    }).sort_values('mean_rank')
    summary.to_csv(OUT / 'model_summary.csv')
    shutil.copy2(RUNS / 'convergence_audit.csv', OUT / 'convergence_audit.csv')
    domain_ranks = ranks.groupby(level='domain').mean().round(2)
    domain_ranks.to_csv(OUT / 'domain_mean_ranks.csv')
    domain_leaders = domain_ranks.idxmin(axis=1)
    display_mse = pivot.round(6).copy()
    display_mse.index = [f'{domain} / {horizon}' for domain, horizon in display_mse.index]

    domains = list(pivot.index.get_level_values(0).unique())
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
    colors = dict(zip(MODELS, plt.cm.tab10.colors[:len(MODELS)]))
    for ax, domain in zip(axes.flat, domains):
        part = pivot.loc[domain]
        for model in MODELS:
            ax.plot(part.index, part[model], marker='o', linewidth=1.8,
                    label=model, color=colors[model])
        ax.set_title(domain)
        ax.set_xlabel('Forecast horizon')
        ax.set_ylabel('Test MSE (train-scaled OT)')
        ax.set_yscale('log')
        ax.grid(alpha=.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               bbox_to_anchor=(.5, -.025))
    fig.suptitle('Time-MMD: test MSE across nine domains and four horizons', fontsize=16)
    fig.savefig(OUT / 'mse_by_domain.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 13), constrained_layout=True)
    im = ax.imshow(ranks.values, vmin=1, vmax=5, cmap='YlGnBu_r', aspect='auto')
    ax.set_xticks(range(len(MODELS)), MODELS, rotation=30, ha='right')
    ax.set_yticks(range(len(pivot)), [f'{d} · {h}' for d, h in pivot.index])
    for y in range(len(pivot)):
        for x in range(len(MODELS)):
            ax.text(x, y, f'{ranks.iloc[y, x]:.0f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, label='Rank within domain and horizon (1 = lowest MSE)')
    ax.set_title('Time-MMD baseline ranks across 36 forecast tasks')
    fig.savefig(OUT / 'mse_ranks.png', dpi=180)
    plt.close(fig)

    winner_counts = summary.wins.to_dict()
    report = [
        '# Time-MMD baseline results', '',
        'Five repositories evaluated on the same prepared OT target, chronological '
        '70/10/20 split, training-only standardization, and domain-specific '
        'lookback/horizon schedule. Four trained models used seed 2025, up to '
        f'{int(good.epochs.max())} epochs and patience 10. Aurora used its published pretrained weights '
        'for zero-shot inference.', '',
        'Variants: MM-TSFlib/Informer+BERT, CFA/PatchTST+BERT with CFA injection, '
        'SpecTF/TimeMixer+GPT-2, TaTS/iTransformer+GPT-2, and Aurora zero-shot.', '',
        'MSE is in units of the training-standardized OT series. Compare models '
        'within a domain/horizon; cross-domain raw MSE magnitudes are not directly '
        'comparable.', '',
        'These are one-seed reference runs using fixed upstream model variants, '
        'not a hyperparameter search. Aurora uses published weights while the '
        'other four models train on each domain; interpret its comparison accordingly.', '',
        '## Overall rank', '', summary.to_markdown(), '',
        '## Mean rank by domain', '', domain_ranks.to_markdown(), '',
        '## Lowest-MSE counts', '',
        ', '.join(f'{k}: {v}' for k, v in winner_counts.items()), '',
        '## Domain leaders by mean rank', '',
        ', '.join(f'{domain}: {model}' for domain, model in domain_leaders.items()), '',
        'Security has a large chronological distribution shift: its test OT mean '
        f'is {drift["Security"][1] / drift["Security"][0]:.1f} times its training mean. '
        'This helps explain why every model has much larger standardized MSE there; '
        'within-task ranks remain the useful comparison.', '',
        '## Test MSE for all 36 tasks', '', display_mse.to_markdown(), '',
        '## Files', '',
        '- `mse_long.csv`: all 180 completed observations, including raw-unit MSE',
        '- `mse_table.csv`: 36 × 5 standardized MSE matrix',
        '- `best_by_task.csv`: lowest-MSE model for each domain and horizon',
        '- `raw_mse_table.csv`: 36 × 5 MSE matrix in original OT units',
        '- `model_summary.csv`: mean rank and win count',
        '- `domain_mean_ranks.csv`: mean rank of each model within each domain',
        '- `mse_by_domain.png`: MSE curves by horizon',
        '- `mse_ranks.png`: rank heatmap', '',
        '- `convergence_audit.csv`: all 180 final stop reasons and validation minima', '',
    ]
    (OUT / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    print(summary.to_string())
    print(f'Saved {expected} results in {OUT}')


if __name__ == '__main__':
    main()
