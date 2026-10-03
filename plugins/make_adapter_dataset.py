"""Create CARMA arrays from ordered ReadGPT validation/test predictions.

All text vectors are deterministic signed token hashes. They are shared by
all five models, and only rows before each forecast origin are used.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIM = 64


def text_vector(raw):
    if not isinstance(raw, str) or not raw.strip() or raw.strip().lower() == 'no information available':
        return np.zeros(DIM, dtype=np.float32), 0.0
    tokens = re.findall(r'(?u)\b\w+\b', raw.lower()[:4000])[:512]
    if not tokens:
        return np.zeros(DIM, dtype=np.float32), 0.0
    vector = np.zeros(DIM, dtype=np.float32)
    for token in tokens:
        digest = hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest()
        index = int.from_bytes(digest[:4], 'little') % DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    vector /= max(np.linalg.norm(vector), 1e-6)
    return vector, 1.0


def repo_dir(model):
    return ROOT / ({'MM-TSFlib': 'MM-TSFlib-main', 'CFA': 'cfa-main',
                    'SpecTF': 'SpecTF-main', 'TaTS': 'TaTS-main',
                    'Aurora': 'Aurora-main/TimeMMD'}[model])


def prediction_files(model, domain, horizon, split):
    folder = repo_dir(model) / ('val_results' if split == 'val' else 'results')
    prefix = '' if model == 'Aurora' else 'long_term_forecast_'
    pattern = f'{prefix}readgpt_data_{model}_{domain}_{horizon}_*'
    found = list(folder.glob(pattern + '/pred.npy'))
    if len(found) != 1:
        raise FileNotFoundError(f'{folder}/{pattern}/pred.npy: found {len(found)}')
    return found[0], found[0].with_name('true.npy')


def build(model, domain, horizon, split):
    source = ROOT / 'benchmark/readgpt_data' / domain
    meta = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(source / f'{domain}.csv')
    values = frame['OT'].to_numpy(dtype=np.float32)
    train_end, val_end, length = meta['train_end'], meta['validation_end'], meta['seq_len']
    mean = values[:train_end].mean()
    sd = values[:train_end].std()
    if sd < 1e-8:
        raise ValueError('constant training series')
    scaled = ((values - mean) / sd).astype(np.float32)
    if 'fact' not in frame:
        raise ValueError('fact column missing')
    text_features = [text_vector(s) for s in frame['fact'].tolist()]
    text = np.stack([x[0] for x in text_features])
    mask = np.asarray([x[1] for x in text_features], dtype=np.float32)
    pred_path, true_path = prediction_files(model, domain, horizon, split)
    pred = np.load(pred_path).astype(np.float32)
    true = np.load(true_path).astype(np.float32)
    origin_start = train_end if split == 'val' else val_end
    expected_n = (val_end if split == 'val' else len(values)) - origin_start - horizon + 1
    if pred.shape != (expected_n, horizon, 1) or true.shape != pred.shape:
        raise ValueError(f'{model} {domain} {horizon} {split}: shapes {pred.shape}, {true.shape}, expected {(expected_n,horizon,1)}')
    origins = np.arange(origin_start, origin_start + expected_n)
    expected = np.stack([scaled[o:o + horizon] for o in origins])[..., None]
    if not np.allclose(true, expected, rtol=2e-4, atol=2e-4):
        delta = np.max(np.abs(true - expected))
        raise ValueError(f'{model} {domain} {horizon} {split}: target alignment/scale mismatch, max abs {delta}')
    return dict(base_pred=pred, target=true,
                history=np.stack([scaled[o - length:o] for o in origins])[..., None],
                text=np.stack([text[o - length:o] for o in origins]),
                text_mask=np.stack([mask[o - length:o] for o in origins]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--domain', required=True)
    p.add_argument('--horizon', type=int, required=True)
    args = p.parse_args()
    val = build(args.model, args.domain, args.horizon, 'val')
    test = build(args.model, args.domain, args.horizon, 'test')
    n = len(val['target'])
    split = max(1, min(n - 1, int(n * 0.7)))
    if split < 8 or n - split < 4:
        raise ValueError(f'validation windows too few for adapter: {n}')
    out = ROOT / 'plugins/adapters' / args.model / args.domain / str(args.horizon)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'fit.npz', **{k: v[:split] for k, v in val.items()})
    np.savez_compressed(out / 'holdout.npz', **{k: v[split:] for k, v in val.items()})
    np.savez_compressed(out / 'test.npz', **test)
    print(f'{args.model} {args.domain} {args.horizon}: fit={split} holdout={n - split} test={len(test["target"])}')


if __name__ == '__main__':
    main()
