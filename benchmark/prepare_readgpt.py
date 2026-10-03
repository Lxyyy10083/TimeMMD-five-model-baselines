"""Canonical Time-MMD adapter for the nine CSVs supplied with TaTS Readgpt.

All five model loaders receive the same chronological OT and text columns.
Only observed targets are retained; historical-average features use past OT.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'TaTS-main-change-Readgpt' / 'data'
OUT = ROOT / 'benchmark' / 'readgpt_data'
HORIZONS = {
    'Agriculture': (6, 8, 10, 12),
    'Climate': (6, 8, 10, 12),
    'Economy': (6, 8, 10, 12),
    'Energy': (12, 24, 36, 48),
    'Environment': (48, 96, 192, 336),
    'Health': (12, 24, 36, 48),
    'Security': (6, 8, 10, 12),
    'SocialGood': (6, 8, 10, 12),
    'Traffic': (6, 8, 10, 12),
}
SEQ_LEN = 24


def prepare_domain(domain: str) -> dict:
    source = SOURCE / f'{domain}.csv'
    raw = pd.read_csv(source, low_memory=False)
    dates = pd.to_datetime(raw['date'], errors='coerce')
    starts = pd.to_datetime(raw['start_date'], errors='coerce')
    ends = pd.to_datetime(raw['end_date'], errors='coerce')
    frame = pd.DataFrame({
        'date': dates, 'start_date': starts, 'end_date': ends,
        'OT': pd.to_numeric(raw['OT'], errors='coerce'),
        'fact': raw['fact'].fillna('No information available').astype(str),
        'preds': raw['preds'].fillna('No information available').astype(str),
    })
    if frame[['date', 'start_date', 'end_date']].isna().any().any():
        raise ValueError(f'{domain}: invalid date in supplied dataset')
    missing = int(frame.OT.isna().sum())
    frame = frame[frame.OT.notna()].sort_values('date', kind='stable')
    duplicates = int(frame.date.duplicated().sum())
    frame = frame.drop_duplicates('date', keep='last').reset_index(drop=True)
    frame['prior_history_avg'] = (frame.OT.shift(1).expanding(min_periods=1)
                                   .mean().fillna(frame.OT.iloc[0]))
    frame['Final_Search_4'] = frame['fact']
    for col in ('date', 'start_date', 'end_date'):
        frame[col] = frame[col].dt.strftime('%Y-%m-%d')
    columns = ['date', 'start_date', 'end_date', 'OT', 'prior_history_avg',
               'fact', 'preds', 'Final_Search_4']
    output_dir = OUT / domain
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f'{domain}.csv'
    frame.to_csv(output, columns=columns, index=False)
    n = len(frame)
    train_end = int(n * .7)
    test_start = n - int(n * .2)
    if test_start - train_end <= max(HORIZONS[domain]):
        raise ValueError(f'{domain}: validation segment is too short')
    meta = {
        'domain': domain, 'source': str(source.relative_to(ROOT)),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'rows_source': len(raw), 'rows': n, 'missing_ot_dropped': missing,
        'duplicate_dates_dropped': duplicates, 'train_end': train_end,
        'validation_end': test_start, 'test_start': test_start,
        'seq_len': SEQ_LEN, 'horizons': HORIZONS[domain],
        'target': 'OT', 'features': 'S', 'csv': str(output.relative_to(ROOT)),
        'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (output_dir / 'manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


if __name__ == '__main__':
    for name in HORIZONS:
        print(json.dumps(prepare_domain(name)))
