"""Model-specific calibration of the shared flow, selected only on holdout."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch

from data import DATA, MODELS, AdapterDataset, text_cache
from model import SemanticGraphFlow, SemanticGraphFlowConfig
from run import DEVICE, OUT, batches, metrics, numpy_predict, seed_all

HERE = Path(__file__).resolve().parent
REFINED = HERE / 'refined'


def candidate(module, ds):
    pred, _ = numpy_predict(module, ds)
    base = metrics(ds.base, ds.target)
    best = {'alpha': 0.0, 'score': 1.0}
    for alpha in (0.25, 0.5, 0.75, 1.0):
        score = metrics(ds.base + alpha * (pred - ds.base), ds.target)
        mse_ratio = score['mse'] / max(base['mse'], 1e-12)
        mae_ratio = score['mae'] / max(base['mae'], 1e-12)
        if mse_ratio < 0.998 and mae_ratio < 0.998:
            combined = 0.5 * (mse_ratio + mae_ratio)
            if combined < best['score']:
                best = {'alpha': alpha, 'score': combined}
    return best


def run_case(domain, horizon, embedding, mask, args):
    shared = OUT / domain / str(horizon) / 'module.pt'
    ckpt = torch.load(shared, map_location='cpu', weights_only=False)
    config = SemanticGraphFlowConfig(**ckpt['config'])
    for name in MODELS:
        dest = REFINED / domain / str(horizon) / name
        if (dest / 'result.json').exists() and not args.force:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        seed_all()
        fit = AdapterDataset(name, domain, horizon, 'fit', embedding, mask)
        hold = AdapterDataset(name, domain, horizon, 'holdout', embedding, mask)
        test = AdapterDataset(name, domain, horizon, 'test', embedding, mask)
        module = SemanticGraphFlow(config).to(DEVICE)
        module.load_state_dict(ckpt['state_dict'])
        best = candidate(module, hold)
        best['epoch'] = 0
        best_state = copy.deepcopy(module.state_dict())
        optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr,
                                      weight_decay=1e-4)
        loader = batches(fit, args.batch_size, True)
        stale, log = 0, []
        for epoch in range(1, args.epochs + 1):
            module.train()
            losses = []
            for batch in loader:
                base, target, history, text, text_mask = [x.to(DEVICE) for x in batch]
                optimizer.zero_grad(set_to_none=True)
                pred, state = module(base, history, text, text_mask)
                loss, _ = module.objective(pred, target, base, state)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'nonfinite: {domain}/{horizon}/{name}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.item()))
            contender = candidate(module, hold)
            log.append({'epoch': epoch, 'loss': float(np.mean(losses)), **contender})
            if contender['score'] < best['score'] - args.min_delta:
                best = {**contender, 'epoch': epoch}
                best_state = copy.deepcopy(module.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        module.load_state_dict(best_state)
        test_pred, profile = numpy_predict(module, test)
        final = test.base + best['alpha'] * (test_pred - test.base)
        baseline_metrics = metrics(test.base, test.target)
        modified_metrics = metrics(final, test.target)
        np.save(dest / 'pred.npy', final)
        np.save(dest / 'profile.npy', profile)
        if best['alpha']:
            torch.save({'config': config.__dict__, 'state_dict': best_state,
                        'alpha': best['alpha'], 'epoch': best['epoch']}, dest / 'module.pt')
        row = {'model': name, 'domain': domain, 'horizon': horizon,
               'alpha': best['alpha'], 'best_epoch': best['epoch'],
               'baseline_mse': baseline_metrics['mse'],
               'baseline_mae': baseline_metrics['mae'],
               'plugin_mse': modified_metrics['mse'],
               'plugin_mae': modified_metrics['mae'],
               'mse_delta_pct': 100 * (modified_metrics['mse'] / baseline_metrics['mse'] - 1),
               'mae_delta_pct': 100 * (modified_metrics['mae'] / baseline_metrics['mae'] - 1),
               'test_windows': len(test)}
        (dest / 'result.json').write_text(json.dumps({'row': row, 'holdout': best,
                                                     'train_log': log}, indent=2),
                                          encoding='utf-8')
        print('RESULT', row, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert', default='/root/autodl-tmp/carma_workspace/models/bert-base-uncased')
    parser.add_argument('--domain')
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--min-delta', type=float, default=0.0003)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    manifests = [json.loads(path.read_text(encoding='utf-8'))
                 for path in sorted(DATA.glob('*/manifest.json'))]
    for item in manifests:
        domain = item['domain']
        if args.domain and args.domain != domain:
            continue
        embedding, mask = text_cache(domain, Path(args.bert))
        for horizon in item['horizons']:
            if args.horizon and args.horizon != horizon:
                continue
            run_case(domain, horizon, embedding, mask, args)
    files = sorted(REFINED.glob('*/*/*/result.json'))
    rows = [json.loads(p.read_text(encoding='utf-8'))['row'] for p in files]
    if rows:
        with (REFINED / 'all_results.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print('COMPLETED REFINED', len(rows), flush=True)


if __name__ == '__main__':
    main()
