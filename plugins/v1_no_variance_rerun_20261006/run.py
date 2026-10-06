"""Execute only the previously locked screenshot configuration, from fresh fits."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'outputs'
VARIANT = 'no_variance_shrink'
MODELS = ['TaTS', 'MM-TSFlib', 'SpecTF', 'CFA', 'Aurora']


def save(name, value):
    OUT.mkdir(exist_ok=True)
    p = OUT / name
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(p)


def original():
    snapshot = json.loads((HERE / 'SOURCE_SNAPSHOT.json').read_text())
    for row in snapshot['source_files']:
        if hashlib.sha256((HERE / row['snapshot']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Original V1 source has changed: ' + row['snapshot'])
    sys.path.insert(0, str(HERE))
    import experiment
    experiment.OUTPUT = OUT
    import torch
    torch.set_num_threads(4)
    return experiment


def arguments():
    return SimpleNamespace(bert=str(Path('/xiliang/LXY/baseline_v3_lab_20261004/models/bert-base-uncased')),
                           screen_epochs=24, final_epochs=80, minimum_epochs=8, patience=8,
                           batch_size=256, lr=.001)


def all_tasks(exp):
    return [(m['domain'], h, s) for m in exp.manifests() for h in m['horizons']
            for s in [2026, 2027, 2028]]


def worker(index):
    exp = original()
    args = arguments()
    cfg = exp.variants()[VARIANT]
    for domain, horizon, seed in all_tasks(exp)[index::2]:
        save(f'PROGRESS_WORKER{index}.json', dict(stage='train', domain=domain, horizon=horizon, seed=seed))
        embedding, mask = exp.text_cache(domain, Path(args.bert))
        exp.train_case('final', VARIANT, cfg, domain, horizon, seed, args, embedding, mask)
        save(f'PROGRESS_WORKER{index}.json', dict(stage='completed_fit', domain=domain, horizon=horizon, seed=seed))


def summarize():
    import csv
    import numpy as np
    from collections import defaultdict
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (OUT / 'test_seed_results.csv').open(encoding='utf-8-sig') as f:
        seeds = list(csv.DictReader(f))
    if len(seeds) != 540:
        raise ValueError('Final result requires 540 test records')
    fits = {}
    capped = []
    for p in (OUT / 'final' / VARIANT).glob('*/*/*/fit_result.json'):
        r = json.loads(p.read_text())
        fits[(r['domain'], r['horizon'], r['seed'])] = r
        stale = r['epochs_run'] - r['best_epoch']
        if r['epochs_run'] >= 80 and stale < 8:
            capped.append(dict(domain=r['domain'], horizon=r['horizon'], seed=r['seed'],
                               best_epoch=r['best_epoch'], epochs_run=r['epochs_run']))
    groups = defaultdict(list)
    for r in seeds:
        groups[(r['model'], r['domain'], int(r['horizon']))].append(r)
    results = []
    for (model, domain, horizon), group in groups.items():
        if len(group) != 3:
            raise ValueError('Expected three independent plugin seeds per model task')
        v = dict(model=model, domain=domain, horizon=horizon, seed_count=3,
                 enabled_seeds=sum(float(r['alpha']) > 0 and fits[(domain, horizon, int(r['seed']))]['best_epoch'] > 0 for r in group))
        for k in ['mse', 'mae']:
            v['baseline_' + k] = float(group[0]['baseline_' + k])
            values = [float(r['plugin_' + k]) for r in group]
            v['plugin_' + k] = float(np.mean(values))
            v[k + '_std'] = float(np.std(values, ddof=1))
            v[k + '_delta_pct'] = 100 * (v['plugin_' + k] / v['baseline_' + k] - 1)
        results.append(v)
    save('RESULTS_180.json', results)
    summary = []
    for model in MODELS:
        selected = [r for r in results if r['model'] == model]
        summary.append(dict(model=model, tasks=len(selected), enabled_cases=sum(r['enabled_seeds'] > 0 for r in selected),
                            both_better=sum(r['mse_delta_pct'] < -1e-5 and r['mae_delta_pct'] < -1e-5 for r in selected),
                            any_worse=sum(r['mse_delta_pct'] > 1e-5 or r['mae_delta_pct'] > 1e-5 for r in selected),
                            unchanged=sum(abs(r['mse_delta_pct']) <= 1e-5 and abs(r['mae_delta_pct']) <= 1e-5 for r in selected),
                            mean_mse_change=float(np.mean([r['mse_delta_pct'] for r in selected])),
                            mean_mae_change=float(np.mean([r['mae_delta_pct'] for r in selected]))))
    save('MODEL_SUMMARY.json', summary)
    save('CONVERGENCE_BUDGET.json', dict(fits=108, capped_without_validation_plateau=capped,
                                       stopping='original V1: at least 8 epochs, patience 8, budget 80',
                                       all_validation_plateau=not capped))
    for name, rows in [('results_180_mean_std.csv', results), ('model_summary.csv', summary)]:
        with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    domains = ['Agriculture', 'Climate', 'Economy', 'Energy', 'Environment', 'Health', 'Security', 'SocialGood', 'Traffic']
    for metric in ['mse', 'mae']:
        fig, axes = plt.subplots(1, 5, figsize=(22, 6), layout='constrained', sharey=True)
        for model, ax in zip(MODELS, axes):
            grid = np.array([[r[metric + '_delta_pct'] for r in sorted(
                [r for r in results if r['model'] == model and r['domain'] == domain], key=lambda r:r['horizon'])]
                for domain in domains])
            im = ax.imshow(grid, cmap='RdYlGn_r', vmin=-30, vmax=30, aspect='auto')
            for i in range(9):
                for j in range(4):
                    ax.text(j, i, f'{grid[i,j]:+.1f}', ha='center', va='center', fontsize=8)
            ax.set_title(model);ax.set_yticks(range(9), domains);ax.set_xticks(range(4), ['H1', 'H2', 'H3', 'H4'])
        fig.colorbar(im, ax=list(axes), label='Error change (%) | negative = improvement', shrink=.8)
        fig.suptitle(f'V1 no_variance_shrink rerun: {metric.upper()} vs original frozen baseline | 3-seed mean')
        fig.savefig(OUT / (metric.upper() + '_180_cases.png'), dpi=180);plt.close(fig)
    lines = ['# V1 no_variance_shrink复现实验结果', '',
             '沿用截图对应的原始冻结预测输入、原始代码、80轮预算及验证集alpha选择。'
             '三种子为独立插件训练性能平均，不是三个底模重训，也不是预测集成。', '',
             '负值表示误差下降；低于0.00001%的变化视为持平。', '',
             '|模型|任务数|启用|双改善|至少一项退化|保持原预测|平均MSE变化|平均MAE变化|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['model']}|36|{r['enabled_cases']}|{r['both_better']}|{r['any_worse']}|{r['unchanged']}|"
                     f"{r['mean_mse_change']:+.2f}%|{r['mean_mae_change']:+.2f}%|")
    lines += ['', f'达到80轮且未满足验证平台期的训练数：{len(capped)}。保持原版预算不会被描述为全部收敛。',
              '', '代码与输入相同，但实验室设备、Python/CUDA运行环境不同，不承诺逐位相同。'
              '数据中holdout只用于选权重和alpha，测试只在全部108次拟合结束后评估。']
    (OUT / 'V1复现实验结果.md').write_text('\n'.join(lines), encoding='utf-8')


def launch():
    exp = original()
    if Path(sys.prefix).resolve() != Path('/xiliang/LXY/envs/lxy'):
        raise RuntimeError('Use only the authorized lxy environment')
    # Recover old data before any training. Missing inputs are a prerequisite.
    for m in exp.manifests():
        for h in m['horizons']:
            for model in MODELS:
                for split in ['fit', 'holdout', 'test']:
                    if not (ROOT / 'plugins/adapters' / model / m['domain'] / str(h) / (split + '.npz')).exists():
                        raise FileNotFoundError(f'Missing original V1 frozen input {model}/{m["domain"]}/{h}/{split}')
    if len(all_tasks(exp)) != 108:
        raise ValueError('Nine domains, four horizons, three plugin seeds are required')
    args = arguments()
    save('winner.json', dict(variant=VARIANT, parameters=exp.variants()[VARIANT],
                             selected_on='Previously locked screenshot configuration; no new parameter selection'))
    save('RUN_PLAN.json', dict(snapshot=json.loads((HERE / 'SOURCE_SNAPSHOT.json').read_text()),
                              arguments=vars(args), models=MODELS, tasks=all_tasks(exp),
                              selection='holdout-only alpha in [0,.25,.5,.75,1]; both errors must improve',
                              base_forecasters='original AutoDL forecasts, frozen; not refit',
                              gpus=[1, 2], environment=sys.prefix))
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True)
    memory = {int(line.split(',')[0]):int(line.split(',')[1]) for line in gpu.stdout.splitlines()}
    if any(memory[i] >= 100 for i in [1, 2]):
        raise RuntimeError('Requested GPUs are occupied; refusing to interfere with other jobs')
    logs = OUT / 'logs';logs.mkdir(exist_ok=True)
    jobs = []
    for index, gpu_id in enumerate([1, 2]):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
        stream = (logs / f'worker{index}.log').open('a')
        process = subprocess.Popen([sys.executable, '-u', str(HERE / 'run.py'), '--worker', str(index)],
                                   env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        jobs.append((process, stream))
    while any(p.poll() is None for p, _ in jobs):
        complete = len(list((OUT / 'final' / VARIANT).glob('*/*/*/fit_result.json')))
        save('STATUS.json', dict(stage='training', completed_fits=complete, total_fits=108,
                                 completed_model_seed_records=complete * 5, total_model_seed_records=540))
        time.sleep(30)
    for _, stream in jobs:stream.close()
    failed = [p.returncode for p, _ in jobs if p.returncode]
    if failed:
        save('INCOMPLETE.json', dict(failed_worker_codes=failed));raise RuntimeError('Training worker failed; see logs')
    save('STATUS.json', dict(stage='evaluating', completed_fits=108, total_fits=108))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1')
    with (logs / 'evaluate.log').open('a') as f:
        subprocess.run([sys.executable, '-u', str(HERE / 'run.py'), '--evaluate'],
                       env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    summarize()
    save('FULL_COMPLETED.json', dict(final_fits=108, test_rows=540, model_tasks=180,
                                    variant=VARIANT, all_final_evaluated=True,
                                    all_validation_plateau=json.loads((OUT / 'CONVERGENCE_BUDGET.json').read_text())['all_validation_plateau']))
    save('STATUS.json', dict(stage='complete', completed_fits=108, total_fits=108, test_rows=540))
    print('V1_RERUN_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--worker', type=int, choices=[0, 1]);p.add_argument('--evaluate', action='store_true')
    args = p.parse_args()
    try:
        if args.worker is not None:worker(args.worker)
        elif args.evaluate:original().evaluate(arguments(), VARIANT)
        else:launch()
    except Exception as error:
        save('INCOMPLETE.json', dict(worker=args.worker, evaluate=args.evaluate, error=str(error)))
        raise
