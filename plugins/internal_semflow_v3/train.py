"""Train one model/domain/horizon with isolated outputs and explicit controls."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from internal_semflow_v3.native import NativeForecast, config_from_upstream
from internal_semflow_v3.data import AlignedWindows


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def _evaluate(model, loader, device):
    model.eval()
    total_squared = total_absolute = 0.0
    total_nll = 0.0
    count = 0
    predictions, targets, origins = [], [], []
    gates, coverages, energies = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch)
            error = (pred - batch['target']).double()
            total_squared += float(error.square().sum())
            total_absolute += float(error.abs().sum())
            count += error.numel()
            state = model.distribution.state
            if model.has_distribution:
                residual = ((batch['target'] - state['prediction']) / model.distribution.memory.scale).squeeze(-1) + state['flow_center']
                # Include the local residual-scale Jacobian for standardized-target NLL.
                nll = model.distribution.flow.nll(residual, state['context'])
                nll = nll + model.horizon * model.distribution.memory.scale[:, 0, 0].log()
                total_nll += float(nll.sum())
                draws = (state['base'][:, None] +
                         model.distribution.memory.scale[:, None] * state['draws'][..., None])
                low = torch.quantile(draws, 0.05, dim=1)
                high = torch.quantile(draws, 0.95, dim=1)
                coverages.append(float(((batch['target'] >= low) &
                                        (batch['target'] <= high)).float().sum()))
                # Empirical CRPS, deterministic sampling approximation.
                raw = draws.squeeze(-1)
                term = (raw - batch['target'].squeeze(-1)[:, None]).abs().sum()
                pair = (raw[:, :, None] - raw[:, None, :]).abs().sum()
                s = raw.shape[1]
                energies.append(float(term / s - 0.5 * pair / (s*s)))
            predictions.append(pred.cpu().numpy())
            targets.append(batch['target'].cpu().numpy())
            origins.append(batch['origin'].cpu().numpy())
            gates.append(float(model.distribution.memory.gate.mean()))
            model.distribution.release()
    metrics = dict(mse=total_squared/count, mae=total_absolute/count,
                   gate_mean=float(np.mean(gates)))
    if model.has_distribution:
        metrics.update(nll_per_step=total_nll/count, crps_approx=sum(energies)/count,
                       coverage90_approx=sum(coverages)/count)
    return metrics, dict(pred=np.concatenate(predictions), target=np.concatenate(targets),
                         origins=np.concatenate(origins))


def evaluate(model, loader, device):
    # Informer's native ProbAttention samples indices even in eval mode.
    # Fixed evaluation RNG removes noise from checkpoint selection; restore the
    # training RNG afterwards so validation does not reset training shuffles.
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(2026)
        if devices:
            torch.cuda.manual_seed_all(2026)
        return _evaluate(model, loader, device)


def run(args):
    if args.stage == 'evaluate' and not (Path(args.output)/'DESIGN_LOCKED.json').exists():
        raise ValueError('test evaluation requires DESIGN_LOCKED.json documenting validation selection')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = config_from_upstream(args.model, args.domain, args.horizon, 24, args.seed)
    # Create the model before loading its warm-start weights.
    model = NativeForecast(args.model, config, args.horizon, args.variant, args.width).to(device)
    if args.control_checkpoint:
        saved_control = torch.load(args.control_checkpoint, map_location=device, weights_only=False)
        native_weights = {k:v for k,v in saved_control['state_dict'].items()
                          if k.startswith(('backbone.', 'text_encoder.'))}
        missing, unexpected = model.load_state_dict(native_weights, strict=False)
        if unexpected or any(k.startswith(('backbone.', 'text_encoder.')) for k in missing):
            raise ValueError(f'control checkpoint mismatch: {missing}, {unexpected}')
    # The native model must exist before data imports, so its module namespace is active.
    datasets = {split: AlignedWindows(args.domain, args.horizon, split, config, args.bert,
                    args.model == 'Aurora', args.variant) for split in ('train', 'val', 'test')}
    train = datasets['train']
    hold = datasets['val']
    loaders = {k: DataLoader(ds, batch_size=args.batch_size, shuffle=(k=='train'),
                            num_workers=0, drop_last=False) for k, ds in datasets.items()}
    dest = Path(args.output) / args.variant / args.model / args.domain / str(args.horizon) / str(args.seed)
    identity_arguments = {k:v for k,v in vars(args).items() if k != 'stage'}
    identity = dict(arguments=identity_arguments, native_config=vars(config) if config else None,
                    manifest=train.meta, train_mean=train.mean, train_std=train.std)
    # Source fingerprint covers plugin and selected native repo, excluding artifacts.
    root = HERE.parents[1]
    source = sorted(HERE.glob('*.py'))
    from internal_semflow_v3.native import REPOS
    source += sorted((root / REPOS[args.model]).rglob('*.py'))
    digest = hashlib.sha256()
    for file in source:
        digest.update(str(file.relative_to(root)).encode())
        digest.update(file.read_bytes())
    identity['source_sha256'] = digest.hexdigest()
    if (dest/'run_config.json').exists():
        old = json.loads((dest/'run_config.json').read_text(encoding='utf-8'))
        if old != identity:
            raise ValueError('existing output has a different config/source; choose a new output root')
        if args.stage == 'evaluate':
            if not (dest/'COMPLETED.json').exists():
                raise ValueError('training must complete before test evaluation')
            saved = torch.load(dest/'checkpoint.pt', map_location=device, weights_only=False)
            model.load_state_dict(saved['state_dict'])
            metrics, arrays = evaluate(model, loaders['test'], device)
            np.savez_compressed(dest/'test_predictions.npz', **arrays)
            write_json(dest/'EVALUATED.json', dict(model=args.model, domain=args.domain,
                horizon=args.horizon, seed=args.seed, variant=args.variant,
                test=metrics, best_epoch=saved['epoch']))
            print('EVALUATED', json.dumps(metrics), flush=True)
            return
        if (dest/'COMPLETED.json').exists():
            print('ALREADY_COMPLETED', dest, flush=True)
            return
        raise ValueError('incomplete previous run; preserve it and choose a new output root')
    if args.stage == 'evaluate':
        raise ValueError('no training record exists for this configuration')
    write_json(dest/'run_config.json', identity)
    write_json(dest/'STARTED.json', dict(device=str(device), windows={k:len(v) for k,v in datasets.items()},
        parameters=sum(p.numel() for p in model.parameters()),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    distribution_parameters = []
    backbone_parameters = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (distribution_parameters if name.startswith(('distribution.', 'fusions.'))
             else backbone_parameters).append(p)
    optimizer = torch.optim.AdamW([
        dict(params=backbone_parameters, lr=args.backbone_lr),
        dict(params=distribution_parameters, lr=args.plugin_lr)], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    weights = dict(nll_weight=args.nll_weight, mae_weight=args.mae_weight,
                   energy_weight=args.energy_weight, regularity_weight=0.001)
    # Score is normalized by a matched native initialization, not by test metrics.
    initial, _ = evaluate(model, loaders['val'], device)
    normal_mse = max(initial['mse'], 1e-8)
    normal_mae = max(initial['mae'], 1e-8)
    # The warm-start checkpoint is a valid candidate; never discard it merely
    # because an additional training epoch was required.
    best_score, best_epoch, stale = 1.0, 0, 0
    torch.save(dict(state_dict=model.state_dict(), epoch=0, validation=initial,
                    identity=identity), dest/'checkpoint.pt')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    trace = []
    started = time.monotonic()
    epochs_run = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = []
        # Ramp auxiliary density loss over early epochs; point path always trains.
        current = {**weights, 'nll_weight': args.nll_weight * min(1, epoch/args.warmup)}
        for batch in loaders['train']:
            batch = {k:v.to(device) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            model(batch)
            loss, parts = model.objective(batch['target'], **current)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite objective at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            totals.append(dict(loss=float(loss.detach()), **parts))
            model.distribution.release()
        val, _ = evaluate(model, loaders['val'], device)
        score = 0.5 * (val['mse']/normal_mse + val['mae']/normal_mae)
        scheduler.step(score)
        epochs_run = epoch
        trace.append(dict(epoch=epoch, score=score, validation=val,
                          train={k:float(np.mean([r[k] for r in totals])) for k in totals[0]}))
        if score < best_score - 1e-5:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch, validation=val,
                            identity=identity), dest/'checkpoint.pt')
        else:
            stale += 1
        write_json(dest/'training.json', dict(best_epoch=best_epoch, epochs_run=epoch,
                                             capped=False, trace=trace))
        print('EPOCH', epoch, json.dumps(val), 'best', best_epoch, flush=True)
        if epoch >= args.minimum_epochs and stale >= args.patience:
            break
    saved = torch.load(dest/'checkpoint.pt', map_location=device, weights_only=False)
    model.load_state_dict(saved['state_dict'])
    # Test is a separate explicit stage after the entire experiment design is locked.
    metrics, _ = evaluate(model, loaders['val'], device)
    record = dict(model=args.model, domain=args.domain, horizon=args.horizon, seed=args.seed,
                  variant=args.variant, validation=metrics, best_epoch=best_epoch, epochs_run=epochs_run,
                  capped=epochs_run==args.epochs, seconds=time.monotonic()-started, trace=trace)
    write_json(dest/'COMPLETED.json', record)
    print('COMPLETED', json.dumps({k:v for k,v in record.items() if k!='trace'}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--control-checkpoint', default='')
    p.add_argument('--stage', choices=['train', 'evaluate'], default='train')
    p.add_argument('--model', choices=['SpecTF','CFA','TaTS','MM-TSFlib','Aurora'], required=True)
    p.add_argument('--domain', required=True)
    p.add_argument('--horizon', type=int, required=True)
    p.add_argument('--variant', choices=['control','internal','fusion_only','distribution_only',
                                        'numeric','shuffle'], default='internal')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--minimum-epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--backbone-lr', type=float, default=0.0001)
    p.add_argument('--plugin-lr', type=float, default=0.001)
    p.add_argument('--nll-weight', type=float, default=0.1)
    p.add_argument('--mae-weight', type=float, default=0.5)
    p.add_argument('--energy-weight', type=float, default=0.05)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--bert', default=str(HERE.parents[1]/'models/bert-base-uncased'))
    p.add_argument('--output', default=str(HERE/'outputs'))
    args = p.parse_args()
    if args.warmup < 1 or args.epochs < 1:
        p.error('epochs and warmup must be positive')
    run(args)


if __name__ == '__main__':
    main()

