"""Resumable five-model, nine-domain, four-horizon CARMA experiment."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DATA = ROOT / 'benchmark/readgpt_data'
RUNS = HERE / 'runs'


def call(script, args, log, env):
    command = [sys.executable, '-u', str(HERE / script), *map(str, args)]
    with log.open('a', encoding='utf-8') as stream:
        stream.write('\n$ ' + ' '.join(command) + '\n')
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{script} exited {result.returncode}; see {log}')


def run_case(model, domain, horizon, env):
    out = HERE / 'adapters' / model / domain / str(horizon)
    if (out / 'result.json').exists():
        return 'skip'
    RUNS.mkdir(parents=True, exist_ok=True)
    log = RUNS / f'carma_{model}_{domain}_{horizon}.log'
    start = time.monotonic()
    try:
        for split in ('val', 'test'):
            call('export_predictions.py', ['--model', model, '--domain', domain,
                                           '--horizon', horizon, '--split', split], log, env)
        call('make_adapter_dataset.py', ['--model', model, '--domain', domain,
                                         '--horizon', horizon], log, env)
        call('carma_v1/train_adapter.py', ['--train', out / 'fit.npz',
                                           '--val', out / 'holdout.npz', '--out', out / 'adapter.pt',
                                           '--epochs', 40, '--patience', 8], log, env)
        call('evaluate_adapter.py', ['--directory', out], log, env)
        status = 'ok'
    except Exception as exc:
        status = 'error:' + str(exc)
    with (RUNS / 'carma_status.csv').open('a', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        if stream.tell() == 0:
            writer.writerow(['model', 'domain', 'horizon', 'status', 'seconds', 'log'])
        writer.writerow([model, domain, horizon, status, round(time.monotonic() - start, 1), log])
    print('DONE', model, domain, horizon, status, flush=True)
    return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model')
    p.add_argument('--domain')
    p.add_argument('--horizon', type=int)
    args = p.parse_args()
    env = os.environ.copy()
    env['BENCHMARK_DATA_DIR'] = str(DATA)
    domains = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(DATA.glob('*/manifest.json'))]
    models = ('TaTS', 'MM-TSFlib', 'SpecTF', 'CFA', 'Aurora')
    cases = 0
    for domain in domains:
        if args.domain and domain['domain'] != args.domain:
            continue
        for horizon in domain['horizons']:
            if args.horizon and horizon != args.horizon:
                continue
            for model in models:
                if args.model and model != args.model:
                    continue
                run_case(model, domain['domain'], horizon, env)
                cases += 1
    print('ATTEMPTED', cases, flush=True)


if __name__ == '__main__':
    main()
