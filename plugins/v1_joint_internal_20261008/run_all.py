"""每模型独立训练；全部达到验证平台期后锁定权重，再对照原始180组。"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = ('TaTS', 'MM-TSFlib', 'SpecTF', 'CFA', 'Aurora')


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def collect(out, jobs):
    grouped = {}
    for model, domain, horizon, seed in jobs:
        path = out / 'internal' / model / domain / str(horizon) / str(seed) / 'EVALUATED.json'
        if not path.exists():
            continue
        grouped.setdefault((model, domain, horizon), []).append(json.loads(path.read_text(encoding='utf-8')))
    rows = []
    expected_seeds = set(j[-1] for j in jobs)
    for (model, domain, horizon), values in grouped.items():
        if {v['seed'] for v in values} != expected_seeds:
            continue
        baseline = values[0]['original']
        if any(v['original'] != baseline for v in values):
            raise ValueError('original reference differs across seeds')
        row = dict(model=model, domain=domain, horizon=horizon, seeds=len(values))
        for metric in ('mse', 'mae'):
            actual = [v['test'][metric] for v in values]
            row.update({f'original_{metric}': baseline[metric], metric: float(np.mean(actual)),
                f'{metric}_std': float(np.std(actual, ddof=1)) if len(actual) > 1 else 0.0,
                f'{metric}_decrease_pct': 100*(1-np.mean(actual)/max(baseline[metric], 1e-12))})
        rows.append(row)
    if rows:
        with (out / 'results_180.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    save(out / 'RESULTS.json', rows)
    save(out / 'SUMMARY.json', dict(tasks=len(rows), total_tasks=len(set(j[:3] for j in jobs)),
        mse_decrease_pct=float(np.mean([r['mse_decrease_pct'] for r in rows])) if rows else None,
        mae_decrease_pct=float(np.mean([r['mae_decrease_pct'] for r in rows])) if rows else None,
        both_better=sum(r['mse_decrease_pct'] > 1e-5 and r['mae_decrease_pct'] > 1e-5 for r in rows),
        any_worse=sum(r['mse_decrease_pct'] < -1e-5 or r['mae_decrease_pct'] < -1e-5 for r in rows)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs-root', default=str(ROOT.parent / 'v1_frozen_inputs_20261006'))
    p.add_argument('--bert', default=str(ROOT / 'models/bert-base-uncased'))
    p.add_argument('--output', default=str(HERE / 'outputs'))
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--domains', nargs='+')
    p.add_argument('--horizon', type=int)
    p.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--epoch-extension', type=int, default=2000)
    p.add_argument('--max-safety-epochs', type=int, default=0,
                   help='0 permits resumable cap extensions until plateau; errors still stop')
    p.add_argument('--cuda-memory-fraction', type=float, default=0.8)
    p.add_argument('--device', default='')
    p.add_argument('--dry-run', action='store_true', help='write and display the task plan without training')
    args = p.parse_args()
    if args.epochs < 1 or args.epoch_extension < 1 or args.max_safety_epochs < 0:
        p.error('invalid epoch budget')
    if args.max_safety_epochs and args.max_safety_epochs < args.epochs:
        p.error('max-safety-epochs cannot be smaller than initial epochs')
    if not 0 < args.cuda_memory_fraction <= 1:
        p.error('cuda-memory-fraction must be in (0,1]')
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        p.error('duplicate models or seeds')
    inputs, out = Path(args.inputs_root).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for path in sorted((inputs / 'benchmark/readgpt_data').glob('*/manifest.json')):
        meta = json.loads(path.read_text(encoding='utf-8'))
        if args.domains and meta['domain'] not in args.domains:
            continue
        for horizon in meta['horizons']:
            if args.horizon and horizon != args.horizon:
                continue
            for model in args.models:
                for seed in args.seeds:
                    jobs.append((model, meta['domain'], horizon, seed))
    if not jobs:
        p.error('no original experiment tasks matched')
    plan = dict(jobs=jobs, comparison='original frozen 180 experiments', variant='internal',
        inputs_root=str(inputs), bert=str(Path(args.bert).resolve()),
        cuda_memory_fraction=args.cuda_memory_fraction,
        convergence_budget=dict(initial=args.epochs, extension=args.epoch_extension,
                                maximum=args.max_safety_epochs, cap_exit_code=42),
        training='independent joint backbone/plugin fit; trained checkpoint only; no alpha bypass')
    if (out / 'PLAN.json').exists() and json.loads((out / 'PLAN.json').read_text(encoding='utf-8')) != json.loads(json.dumps(plan)):
        raise ValueError('existing experiment plan differs')
    save(out / 'PLAN.json', plan)
    if args.dry_run:
        print(json.dumps(dict(tasks=len(set(j[:3] for j in jobs)), fits=len(jobs),
            plan=str(out / 'PLAN.json'), experiments_started=False), ensure_ascii=False))
        return
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    trained = []
    # 不读取/选择空闲GPU；使用调用者指定的CUDA_VISIBLE_DEVICES，默认单进程顺序执行。
    for stage in ('train', 'evaluate'):
        if stage == 'evaluate':
            save(out / 'TEST_LOCK.json', dict(plan=plan, trained=trained))
        for model, domain, horizon, seed in jobs:
            dest = out / 'internal' / model / domain / str(horizon) / str(seed)
            log = out / 'logs' / f'{stage}_{model}_{domain}_{horizon}_{seed}.log'
            log.parent.mkdir(exist_ok=True)
            command = [sys.executable, '-u', str(HERE / 'train.py'), '--stage', stage,
                '--model', model, '--domain', domain, '--horizon', str(horizon), '--seed', str(seed),
                '--inputs-root', str(inputs), '--bert', args.bert, '--output', str(out),
                '--cuda-memory-fraction', str(args.cuda_memory_fraction)]
            if args.device:
                command.extend(['--device', args.device])
            cap = args.epochs
            budget_path = dest / 'BUDGET_EXTENSIONS.json'
            if budget_path.exists():
                cap = max(cap, json.loads(budget_path.read_text(encoding='utf-8'))['next_cap'])
            while True:
                save(out / 'STATUS.json', dict(stage=stage, model=model, domain=domain,
                    horizon=horizon, seed=seed, current_safety_cap=cap))
                with log.open('a', encoding='utf-8') as stream:
                    stream.write(f'\nSTART stage={stage} seed={seed} safety_cap={cap}\n'); stream.flush()
                    result = subprocess.run(command + ['--epochs', str(cap)], cwd=ROOT,
                        env=env, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode != 42 or stage != 'train':
                    break
                marker = json.loads((dest / 'NOT_CONVERGED.json').read_text(encoding='utf-8'))
                if marker['safety_limit'] != cap or not (dest / 'resume.pt').exists():
                    raise ValueError('budget exit has no matching recoverable checkpoint')
                if args.max_safety_epochs and cap >= args.max_safety_epochs:
                    break
                next_cap = cap + args.epoch_extension
                if args.max_safety_epochs:
                    next_cap = min(next_cap, args.max_safety_epochs)
                history = json.loads(budget_path.read_text(encoding='utf-8'))['history'] if budget_path.exists() else []
                history.append(dict(previous_cap=cap, next_cap=next_cap, time_unix=time.time()))
                save(budget_path, dict(next_cap=next_cap, history=history))
                print(f'CONTINUE {model}/{domain}/{horizon}/{seed}: {cap} -> {next_cap}', flush=True)
                cap = next_cap
            if result.returncode:
                save(out / 'FAILED.json', dict(stage=stage, task=[model, domain, horizon, seed], log=str(log)))
                raise SystemExit(f'{stage} failed; preserved log: {log}')
            if stage == 'train':
                completion = json.loads((dest / 'COMPLETED.json').read_text(encoding='utf-8'))
                if not completion['converged'] or completion['best_epoch'] < 1:
                    raise ValueError('refuse untrained/unconverged checkpoint')
                trained.append(dict(task=[model, domain, horizon, seed], source_sha256=completion['source_sha256'],
                    checkpoint_sha256=hashlib.sha256((dest / 'checkpoint.pt').read_bytes()).hexdigest()))
            else:
                collect(out, jobs)
    rows = collect(out, jobs)
    if len(rows) != len(set(j[:3] for j in jobs)):
        raise ValueError('incomplete task/seed results')
    audited = []
    for model, domain, horizon, seed in jobs:
        record = json.loads((out / 'internal' / model / domain / str(horizon) / str(seed) / 'COMPLETED.json').read_text(encoding='utf-8'))
        if not record['converged'] or record['best_epoch'] < 1:
            raise ValueError('final convergence audit failed')
        audited.append(dict(model=model, domain=domain, horizon=horizon, seed=seed,
                            epochs_run=record['epochs_run'], best_epoch=record['best_epoch'], converged=True))
    with (out / 'convergence_audit.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(audited[0]))
        writer.writeheader(); writer.writerows(audited)
    save(out / 'FULL_COMPLETED.json', dict(tasks=len(rows), fits=len(jobs), all_validation_plateau=True))


if __name__ == '__main__':
    main()
