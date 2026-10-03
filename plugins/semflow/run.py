"""Train one shared semantic distribution per domain/horizon, evaluate five frozen models."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from data import DATA, MODELS, AdapterDataset, text_cache
from model import SemanticGraphFlow, SemanticGraphFlowConfig

HERE = Path(__file__).resolve().parent
OUT = HERE / 'runs'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_all(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batches(dataset, batch_size=128, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=DEVICE.type == 'cuda')


def numpy_predict(module, dataset):
    module.eval()
    pieces, profiles = [], []
    with torch.inference_mode():
        for batch in batches(dataset, 256):
            base, _, history, text, mask = [x.to(DEVICE) for x in batch]
            pred, state = module(base, history, text, mask)
            profile = module.semantic_profile(state, history, pred)
            pieces.append(pred.cpu().numpy())
            profiles.append(np.stack([profile[k].cpu().numpy() for k in
                ('text_utility', 'expected_direction', 'expected_change',
                 'model_uncertainty', 'text_shift')], axis=-1))
    return np.concatenate(pieces), np.concatenate(profiles)


def metrics(pred, target):
    delta = pred - target
    return {'mse': float(np.mean(delta ** 2, dtype=np.float64)),
            'mae': float(np.mean(np.abs(delta), dtype=np.float64))}


def validation_loss(module, holdouts):
    scores = []
    for ds in holdouts.values():
        pred, _ = numpy_predict(module, ds)
        base = metrics(ds.base, ds.target)
        new = metrics(pred, ds.target)
        scores.append(0.5 * (new['mse'] / max(base['mse'], 1e-9)
                            + new['mae'] / max(base['mae'], 1e-9)))
    return float(np.mean(scores))


def choose_alpha(base, plugin, target):
    baseline = metrics(base, target)
    chosen, best = 0.0, 2.0
    for alpha in (0.25, 0.5, 0.75, 1.0):
        current = metrics(base + alpha * (plugin - base), target)
        if current['mse'] >= baseline['mse'] * 0.998:
            continue
        if current['mae'] >= baseline['mae'] * 0.998:
            continue
        score = 0.5 * (current['mse'] / baseline['mse'] +
                       current['mae'] / baseline['mae'])
        if score < best:
            chosen, best = alpha, score
    return chosen


def run_case(domain, horizon, args):
    dest = OUT / domain / str(horizon)
    if (dest / 'result.json').exists() and not args.force:
        print('SKIP', domain, horizon, flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True)
    seed_all()
    start = time.monotonic()
    embedding, mask = text_cache(domain, Path(args.bert), args.bert_batch)
    fit = {name: AdapterDataset(name, domain, horizon, 'fit', embedding, mask)
           for name in MODELS}
    holdout = {name: AdapterDataset(name, domain, horizon, 'holdout', embedding, mask)
               for name in MODELS}
    test = {name: AdapterDataset(name, domain, horizon, 'test', embedding, mask)
            for name in MODELS}
    config = SemanticGraphFlowConfig(text_dim=embedding.shape[1], horizon=horizon,
                                     coefficients=min(args.coefficients, horizon),
                                     samples=args.samples)
    module = SemanticGraphFlow(config).to(DEVICE)
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr, weight_decay=1e-4)
    fit_loader = batches(ConcatDataset(list(fit.values())), args.batch_size, True)
    best, stale, best_epoch = float('inf'), 0, 0
    train_log = []
    for epoch in range(1, args.epochs + 1):
        module.train()
        total, n_batches = 0.0, 0
        for batch in fit_loader:
            base, target, history, text, text_mask = [x.to(DEVICE) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            pred, state = module(base, history, text, text_mask)
            loss, _ = module.objective(pred, target, base, state)
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite loss: {domain}/{horizon}/epoch{epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n_batches += 1
        val = validation_loss(module, holdout)
        row = {'epoch': epoch, 'train_loss': total / n_batches, 'val_ratio': val}
        train_log.append(row)
        print('EPOCH', domain, horizon, row, flush=True)
        if val < best - args.min_delta:
            best, stale, best_epoch = val, 0, epoch
            torch.save({'config': config.__dict__, 'state_dict': module.state_dict(),
                        'epoch': epoch, 'val_ratio': val}, dest / 'module.pt')
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(dest / 'module.pt', map_location=DEVICE, weights_only=False)
    module.load_state_dict(checkpoint['state_dict'])
    rows = []
    for name in MODELS:
        hold_pred, _ = numpy_predict(module, holdout[name])
        alpha = choose_alpha(holdout[name].base, hold_pred, holdout[name].target)
        test_pred, profile = numpy_predict(module, test[name])
        chosen = test[name].base + alpha * (test_pred - test[name].base)
        baseline = metrics(test[name].base, test[name].target)
        modified = metrics(chosen, test[name].target)
        np.save(dest / f'{name}_pred.npy', chosen)
        np.save(dest / f'{name}_profile.npy', profile)
        row = {'model': name, 'domain': domain, 'horizon': horizon, 'alpha': alpha,
               'baseline_mse': baseline['mse'], 'baseline_mae': baseline['mae'],
               'plugin_mse': modified['mse'], 'plugin_mae': modified['mae'],
               'mse_delta_pct': 100 * (modified['mse'] / baseline['mse'] - 1),
               'mae_delta_pct': 100 * (modified['mae'] / baseline['mae'] - 1),
               'test_windows': len(test[name])}
        rows.append(row)
        print('RESULT', row, flush=True)
    result = {'domain': domain, 'horizon': horizon, 'checkpoint_epoch': best_epoch,
              'holdout_ratio': best, 'seconds': time.monotonic() - start,
              'config': config.__dict__, 'train_log': train_log, 'rows': rows}
    (dest / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert', default='/root/autodl-tmp/carma_workspace/models/bert-base-uncased')
    parser.add_argument('--domain')
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--bert-batch', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--min-delta', type=float, default=0.0005)
    parser.add_argument('--coefficients', type=int, default=4)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    manifests = [json.loads(p.read_text(encoding='utf-8'))
                 for p in sorted(DATA.glob('*/manifest.json'))]
    for item in manifests:
        if args.domain and item['domain'] != args.domain:
            continue
        for horizon in item['horizons']:
            if args.horizon and args.horizon != horizon:
                continue
            run_case(item['domain'], horizon, args)
    files = list(OUT.glob('*/*/result.json'))
    rows = [row for path in files for row in json.loads(path.read_text())['rows']]
    if rows:
        with (OUT / 'all_results.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print('COMPLETED CASES', len(files), 'ROWS', len(rows), flush=True)


if __name__ == '__main__':
    main()
