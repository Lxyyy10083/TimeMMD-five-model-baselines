"""Collect CARMA MSE/MAE for every model, domain, and prediction horizon."""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
MODELS = ('MM-TSFlib', 'CFA', 'SpecTF', 'TaTS', 'Aurora')
FIELDS = ('model', 'domain', 'horizon', 'status', 'enabled_by_validation',
          'base_mse', 'carma_mse', 'delta_mse', 'base_mae', 'carma_mae', 'delta_mae')


def main():
    rows = []
    for path in sorted((ROOT / 'benchmark/readgpt_data').glob('*/manifest.json')):
        meta = json.loads(path.read_text(encoding='utf-8'))
        for horizon in meta['horizons']:
            for model in MODELS:
                result = HERE / 'adapters' / model / meta['domain'] / str(horizon) / 'result.json'
                row = dict(model=model, domain=meta['domain'], horizon=horizon,
                           status='missing' if not result.exists() else 'ok')
                if result.exists():
                    item = json.loads(result.read_text(encoding='utf-8'))
                    row.update(enabled_by_validation=int(item['enabled_by_validation']),
                               base_mse=item['base_test']['mse'],
                               carma_mse=item['adapter_test']['mse'],
                               delta_mse=item['adapter_test']['mse'] - item['base_test']['mse'],
                               base_mae=item['base_test']['mae'],
                               carma_mae=item['adapter_test']['mae'],
                               delta_mae=item['adapter_test']['mae'] - item['base_test']['mae'])
                rows.append(row)
    with (HERE / 'results_180.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['model']].append(row)
    report = ['# CARMA 实验汇总', '',
              '数据：ReadGPT 对齐的 TimeMMD。MSE/MAE 数值越低越好；差值 = CARMA − 原模型。', '',
              '| 模型 | 已完成/总数 | 验证启用 | 测试 MSE 下降 | 测试 MAE 下降 | 两项同时下降 |',
              '|---|---:|---:|---:|---:|---:|']
    for model in MODELS:
        subset = [r for r in grouped[model] if r['status'] == 'ok']
        report.append(f'| {model} | {len(subset)}/{len(grouped[model])} | '
                      f'{sum(int(r["enabled_by_validation"]) for r in subset)} | '
                      f'{sum(r["delta_mse"] < 0 for r in subset)} | '
                      f'{sum(r["delta_mae"] < 0 for r in subset)} | '
                      f'{sum(r["delta_mse"] < 0 and r["delta_mae"] < 0 for r in subset)} |')
    unfinished = [f'{r["model"]}/{r["domain"]}/{r["horizon"]}' for r in rows if r['status'] != 'ok']
    report += ['', f'未完成：{len(unfinished)} 组。', '']
    if unfinished:
        report += ['前 20 个未完成任务：' + ', '.join(unfinished[:20]), '']
    (HERE / 'SUMMARY.md').write_text('\n'.join(report), encoding='utf-8')
    print(f'completed={len(rows)-len(unfinished)} total={len(rows)} missing={len(unfinished)}')


if __name__ == '__main__':
    main()
