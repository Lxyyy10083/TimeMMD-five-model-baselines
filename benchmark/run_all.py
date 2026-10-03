"""Unified Time-MMD benchmark launcher for the five upstream repositories.

All jobs use the prepared OT series, 70/10/20 chronological split, training-only
standardization, one seed, and identical domain-specific lookbacks/horizons.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('BENCHMARK_DATA_DIR', ROOT / 'benchmark' / 'data'))
OUT = Path(os.environ.get('BENCHMARK_RUNS_DIR', ROOT / 'benchmark' / 'runs'))
MODELS = ('MM-TSFlib', 'CFA', 'SpecTF', 'TaTS', 'Aurora')
FREQ = {'Environment': 'd', 'Energy': 'w', 'Health_US': 'w', 'Health': 'w'}
FIELDS = ('model', 'domain', 'horizon', 'seq_len', 'seed', 'epochs', 'status',
          'mse', 'seconds', 'log')
RESULTS_LOCK = threading.Lock()


def command(model, domain, horizon, seq_len, epochs, seed):
    data_dir = DATA / domain
    py = sys.executable
    identifier = f'{DATA.name}_{model}_{domain}_{horizon}_{seed}'
    freq = FREQ.get(domain, 'm')
    train_batch = 64 if domain == 'Environment' else 16
    if model == 'Aurora':
        workdir = ROOT / 'Aurora-main' / 'TimeMMD'
        args = [py, '-u', 'run_longExp.py', '--is_training', '0', '--model_id', identifier,
                '--data', domain, '--root_path', str(data_dir), '--data_path', f'{domain}.csv',
                '--features', 'S', '--freq', freq, '--seq_len', str(seq_len), '--label_len', '0',
                '--pred_len', str(horizon), '--batch_size', '8', '--num_workers', '0',
                '--model_path', str(ROOT / 'models' / 'aurora'),
                '--inference_token_len', str(min(seq_len, 48)), '--random_seed', str(seed)]
    else:
        repo = {'MM-TSFlib': 'MM-TSFlib-main', 'CFA': 'cfa-main',
                'SpecTF': 'SpecTF-main', 'TaTS': 'TaTS-main'}[model]
        workdir = ROOT / repo
        model_name = {'MM-TSFlib': 'Informer', 'CFA': 'PatchTST',
                      'SpecTF': 'TimeMixer', 'TaTS': 'iTransformer'}[model]
        args = [py, '-u', 'run.py', '--task_name', 'long_term_forecast',
                '--is_training', '1', '--model_id', identifier, '--model', model_name,
                '--data', 'custom', '--root_path', str(data_dir), '--data_path', f'{domain}.csv',
                '--features', 'S', '--target', 'OT', '--freq', freq, '--seq_len', str(seq_len),
                '--label_len', str(seq_len // 2 if model != 'SpecTF' else 0),
                '--pred_len', str(horizon), '--enc_in', '1', '--dec_in', '1', '--c_out', '1',
                '--train_epochs', str(epochs), '--patience', '10', '--batch_size', str(train_batch),
                '--num_workers', '0', '--seed', str(seed), '--gpu', '0',
                '--text_len', '4', '--des', 'baseline', '--save_name', str(OUT / 'legacy_metrics.txt')]
        if model == 'MM-TSFlib':
            args += ['--llm_model', 'BERT', '--use_fullmodel', '0']
        elif model == 'CFA':
            args += ['--llm_model', 'BERT', '--use_fullmodel', '0',
                     '--use_text_integrated', '1', '--text_injection_mode', 'cfa']
        elif model == 'SpecTF':
            args += ['--llm_model', 'GPT2', '--llm_path', str(ROOT / 'models' / 'gpt2'),
                     '--method', 'SpecTF', '--down_sampling_layers', '2',
                     '--down_sampling_method', 'avg', '--down_sampling_window', '2',
                     '--mm_emb_size', '4', '--mm_hidden_size', '16', '--n_ts_features', '1',
                     '--text_emb', '8', '--llm_emb_size', '768', '--proj_per_freq',
                     '--fuse_history', '--use_product']
        elif model == 'TaTS':
            args += ['--llm_model', 'GPT2', '--text_emb', '12', '--prior_weight', '0.5']
    return workdir, args


def run_one(model, domain, horizon, seq_len, epochs, seed):
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / f'{model}_{domain}_{horizon}.log'
    workdir, args = command(model, domain, horizon, seq_len, epochs, seed)
    env = os.environ.copy()
    env['BASELINE_BERT_PATH'] = str(ROOT / 'models' / 'bert-base-uncased')
    env['BASELINE_GPT2_PATH'] = str(ROOT / 'models' / 'gpt2')
    env['HF_HOME'] = '/root/autodl-tmp/hf'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['MPLBACKEND'] = 'Agg'
    env['OMP_NUM_THREADS'] = '4'
    env['MKL_NUM_THREADS'] = '4'
    env['OPENBLAS_NUM_THREADS'] = '4'
    env['NUMEXPR_NUM_THREADS'] = '4'
    start = time.monotonic()
    print(f'START {model} {domain} {horizon}', flush=True)
    with log_path.open('w', encoding='utf-8') as log:
        result = subprocess.run(args, cwd=workdir, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=86400)
    seconds = round(time.monotonic() - start, 1)
    output = log_path.read_text(encoding='utf-8', errors='replace')
    matches = re.findall(r'\bmse\s*:\s*([0-9.+eE-]+)', output)
    mse = float(matches[-1]) if matches else ''
    status = 'ok' if result.returncode == 0 and mse != '' else f'error:{result.returncode}'
    row = dict(model=model, domain=domain, horizon=horizon, seq_len=seq_len,
               seed=seed, epochs=epochs, status=status, mse=mse,
               seconds=seconds, log=str(log_path.relative_to(ROOT)))
    print(f'DONE {model} {domain} {horizon} {status} mse={mse} seconds={seconds}', flush=True)
    with RESULTS_LOCK:
        with (OUT / 'results.csv').open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--domain')
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    done = set()
    results_path = OUT / 'results.csv'
    if results_path.exists():
        with results_path.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                if row['status'] == 'ok' or not args.retry_errors:
                    done.add((row['model'], row['domain'], int(row['horizon'])))
    jobs = []
    for manifest_path in sorted(DATA.glob('*/manifest.json')):
        meta = json.loads(manifest_path.read_text(encoding='utf-8'))
        domain = meta['domain']
        if args.domain and args.domain != domain:
            continue
        for horizon in meta['horizons']:
            if args.horizon and horizon != args.horizon:
                continue
            for model in MODELS:
                if args.model and args.model != model:
                    continue
                if (model, domain, horizon) in done:
                    continue
                jobs.append((model, domain, horizon, meta['seq_len'], args.epochs, args.seed))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, *job) for job in jobs]
        for future in as_completed(futures):
            future.result()


if __name__ == '__main__':
    main()
