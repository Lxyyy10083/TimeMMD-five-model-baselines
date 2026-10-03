"""Audit validation early stopping for each completed benchmark run."""
import csv
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get('BENCHMARK_RUNS_DIR', ROOT / 'benchmark' / 'runs'))
with (RUNS / 'results.csv').open(newline='', encoding='utf-8') as stream:
    rows = list(csv.DictReader(stream))
latest = {(r['model'], r['domain'], r['horizon']): r for r in rows}
audit = []
for row in latest.values():
    log = (ROOT / row['log']).read_text(encoding='utf-8', errors='replace')
    epochs = [int(x) for x in re.findall(r'Epoch:\s*(\d+)\s*cost time', log)]
    losses = [float(x) for x in re.findall(r'Vali Loss:\s*([0-9.eE+-]+)', log)]
    state = ('pretrained_zero_shot' if row['model'] == 'Aurora' else
             'early_stopped' if 'Early stopping' in log else
             'hit_epoch_cap' if epochs and epochs[-1] >= int(row['epochs']) else
             'error_or_incomplete')
    audit.append({'model': row['model'], 'domain': row['domain'],
                  'horizon': row['horizon'], 'status': row['status'],
                  'stop_reason': state, 'epochs_run': epochs[-1] if epochs else 0,
                  'best_val_loss': min(losses) if losses else ''})
with (RUNS / 'convergence_audit.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
    writer.writeheader()
    writer.writerows(audit)
print('rows', len(audit), 'cap', sum(r['stop_reason'] == 'hit_epoch_cap' for r in audit),
      'errors', sum(r['status'] != 'ok' for r in audit))
