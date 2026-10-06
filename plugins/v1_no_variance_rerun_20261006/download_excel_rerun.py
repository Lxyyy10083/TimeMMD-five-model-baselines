"""Retrieve a separately executed V1 rerun and prepare measured workbook inputs."""
from pathlib import Path
import csv
import getpass
import hashlib
import json
import os
import sys
import zipfile
import paramiko

BASE = Path(sys.argv[1]).resolve()
REMOTE = '/xiliang/LXY/baseline_v1_no_variance_20261006_rerun2'
OUT = BASE / 'GANF_V1_MSE_Excel_rerun2_20261006'
OUT.mkdir(exist_ok=True)
cfg = paramiko.SSHConfig()
with Path.home().joinpath('.ssh/config').open() as f:
    cfg.parse(f)
h = cfg.lookup('Remote')
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.WarningPolicy())
password = getpass.getpass('Lab password: ')
c.connect(h['hostname'], port=int(h['port']), username=h['user'], password=password,
          timeout=20, auth_timeout=20, allow_agent=False, look_for_keys=False)
del password
archive = OUT.with_suffix('.zip')
with c.open_sftp() as s:
    s.get(REMOTE + '/completed_v1_rerun2_results.zip', str(archive))
c.close()
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        if os.path.commonpath([str(OUT), str((OUT / name).resolve())]) != str(OUT):
            raise ValueError('Unsafe archive member')
    z.extractall(OUT)
complete = json.loads((OUT / 'FULL_COMPLETED.json').read_text())
launch = json.loads((OUT / 'V1_LAUNCH.json').read_text())
assert complete['final_fits'] == 108 and complete['test_rows'] == 540
assert launch['workspace'] == REMOTE and launch['fresh_fits']
domains = ['Agriculture', 'Climate', 'Economy', 'Energy', 'Environment', 'Health', 'Security', 'SocialGood', 'Traffic']
models = ['TaTS', 'MM-TSFlib', 'SpecTF', 'CFA', 'Aurora']
rows = list(csv.DictReader((OUT / 'test_seed_results.csv').open(encoding='utf-8-sig')))
assert len(rows) == 540
for row in rows:
    for key in ['horizon', 'seed', 'test_windows']:
        row[key] = int(row[key])
    for key in ['alpha', 'baseline_mse', 'baseline_mae', 'plugin_mse', 'plugin_mae', 'raw_mse', 'raw_mae', 'mse_delta_pct', 'mae_delta_pct', 'gate_mean', 'correction_std_units']:
        row[key] = float(row[key])
    fit = json.loads((OUT / 'final/no_variance_shrink' / str(row['seed']) / row['domain'] / str(row['horizon']) / 'fit_result.json').read_text())
    row['best_epoch'] = fit['best_epoch']
    row['epochs_run'] = fit['epochs_run']
    row['validation_plateau'] = fit['epochs_run'] - fit['best_epoch'] >= 8
    row['holdout_mse_ratio'] = fit['holdout'][row['model']]['mse_ratio']
    row['holdout_mae_ratio'] = fit['holdout'][row['model']]['mae_ratio']
rows.sort(key=lambda r:(domains.index(r['domain']), r['horizon'], models.index(r['model']), r['seed']))
groups = [rows[i:i+3] for i in range(0, 540, 3)]
assert len(groups) == 180
for group in groups:
    assert {r['seed'] for r in group} == {2026, 2027, 2028}
    assert len({(r['domain'], r['horizon'], r['model']) for r in group}) == 1
    assert max(r['baseline_mse'] for r in group) - min(r['baseline_mse'] for r in group) < 1e-10
budget = json.loads((OUT / 'CONVERGENCE_BUDGET.json').read_text())
payload = dict(domains=domains, models=models, rows=rows,
               completed=complete, launch=launch, convergence=budget,
               source_commit='b5e4d91', seeds=[2026, 2027, 2028],
               archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
(OUT / 'workbook_inputs.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(dict(downloaded=str(OUT), rows=len(rows), tasks=len(groups),
                      capped=len(budget['capped_without_validation_plateau'])), ensure_ascii=False))
