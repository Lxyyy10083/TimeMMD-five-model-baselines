"""Export chronologically ordered validation forecasts from frozen baseline checkpoints.

CFA has no retained checkpoint in the supplied baseline, so it is retrained
with the same ReadGPT data/arguments before its validation predictions.
"""
import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('benchmark_run_all', ROOT / 'benchmark/run_all.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def result_dir(model):
    if model == 'Aurora':
        return ROOT / 'Aurora-main/TimeMMD/val_results'
    repo = {'MM-TSFlib': 'MM-TSFlib-main', 'CFA': 'cfa-main',
            'SpecTF': 'SpecTF-main', 'TaTS': 'TaTS-main'}[model]
    return ROOT / repo / 'val_results'


def existing_predictions(model, domain, horizon):
    pattern = f'*_{model}_{domain}_{horizon}_*'
    return list(result_dir(model).glob(pattern + '/pred.npy'))


def run_one(model, domain, horizon, seq_len, epochs, seed):
    if existing_predictions(model, domain, horizon):
        print('SKIP', model, domain, horizon, flush=True)
        return 'ok'
    benchmark.OUT.mkdir(parents=True, exist_ok=True)
    cwd, args = benchmark.command(model, domain, horizon, seq_len, epochs, seed)
    if model != 'Aurora':
        idx = args.index('--is_training') + 1
        args[idx] = '1' if model == 'CFA' else '0'
    env = os.environ.copy()
    env.update(CARMA_EXPORT_SPLIT='val', CARMA_KEEP_CHECKPOINTS='1',
               BASELINE_BERT_PATH=str(ROOT / 'models/bert-base-uncased'),
               BASELINE_GPT2_PATH=str(ROOT / 'models/gpt2'),
               HF_HOME='/root/autodl-tmp/hf', TOKENIZERS_PARALLELISM='false',
               MPLBACKEND='Agg', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    logs = ROOT / 'plugins/runs'
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f'export_{model}_{domain}_{horizon}.log'
    print('START', model, domain, horizon, flush=True)
    start = time.monotonic()
    with log.open('w', encoding='utf-8') as stream:
        outcome = subprocess.run(args, cwd=cwd, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT)
    pred = existing_predictions(model, domain, horizon)
    status = 'ok' if outcome.returncode == 0 and len(pred) == 1 else f'error:{outcome.returncode};matches={len(pred)}'
    with (logs / 'export_status.csv').open('a', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        if stream.tell() == 0:
            writer.writerow(['model', 'domain', 'horizon', 'status', 'seconds', 'log'])
        writer.writerow([model, domain, horizon, status, round(time.monotonic() - start, 1), str(log)])
    print('DONE', model, domain, horizon, status, flush=True)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=benchmark.MODELS)
    parser.add_argument('--domain')
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args()
    for path in sorted(benchmark.DATA.glob('*/manifest.json')):
        meta = json.loads(path.read_text(encoding='utf-8'))
        domain = meta['domain']
        if args.domain and domain != args.domain:
            continue
        for horizon in meta['horizons']:
            if args.horizon and horizon != args.horizon:
                continue
            for model in benchmark.MODELS:
                if args.model and model != args.model:
                    continue
                status = run_one(model, domain, horizon, meta['seq_len'], args.epochs, args.seed)
                if status != 'ok':
                    raise RuntimeError(f'{model} {domain} {horizon}: {status}')


if __name__ == '__main__':
    main()
