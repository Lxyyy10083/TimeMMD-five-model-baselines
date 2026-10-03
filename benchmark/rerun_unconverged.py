"""Finish benchmark tasks that failed or stopped before early stopping."""
import csv
import json
import subprocess
import sys
from pathlib import Path

from run_all import DATA, OUT, run_one


def main():
    subprocess.run([sys.executable, str(Path(__file__).with_name('audit.py'))], check=True)
    with (OUT / 'convergence_audit.csv').open(newline='', encoding='utf-8') as stream:
        pending = [row for row in csv.DictReader(stream)
                   if row['status'] != 'ok' or row['stop_reason'] == 'hit_epoch_cap']
    print(f'Rerunning {len(pending)} unfinished tasks', flush=True)
    for row in pending:
        model, domain, horizon = row['model'], row['domain'], int(row['horizon'])
        meta = json.loads((DATA / domain / 'manifest.json').read_text(encoding='utf-8'))
        epochs = 200 if row['stop_reason'] == 'hit_epoch_cap' else 100
        run_one(model, domain, horizon, meta['seq_len'], epochs, 2025)
    subprocess.run([sys.executable, str(Path(__file__).with_name('audit.py'))], check=True)


if __name__ == '__main__':
    main()
