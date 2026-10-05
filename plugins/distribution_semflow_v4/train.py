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
from distribution_semflow_v4.native import NativeForecast, config_from_upstream
from distribution_semflow_v4.data import AlignedWindows


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def checkpoint_state(model):
    # Frozen Aurora tensors are restored from the copied pretrained model.
    # Store learned parameters and buffers, avoiding 800MB writes every epoch.
    names = {name for name,p in model.named_parameters() if p.requires_grad}
    names.update(name for name,_ in model.named_buffers())
    return {k:v for k,v in model.state_dict().items() if k in names}


def restore_checkpoint(model, saved):
    missing, unexpected = model.load_state_dict(saved['state_dict'], strict=False)
    required = {name for name,p in model.named_parameters() if p.requires_grad}
    required.update(name for name,_ in model.named_buffers())
    if unexpected or any(k in required for k in missing):
        raise ValueError(f'incomplete learned state: {missing}, {unexpected}')


def export_distribution(dest, arrays, mean, std):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # One fixed first test origin, selected before observing any test errors.
    keys=['pred','target','native_pred','distribution_mean','p05','p50','p95']
    raw={k:arrays[k][0].reshape(-1)*std+mean for k in keys}
    raw['origin']=arrays['origins'][:1]
    np.savez_compressed(dest/'first_origin_raw_distribution.npz',**raw)
    with (dest/'distribution_profile.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f)
        w.writerow(['origin','mean_prediction_raw','mean_distribution_raw','mean_p05_raw','mean_p95_raw',
            'semantic_mixture_probability','semantic_effect_raw','probability_above_last'])
        for i,origin in enumerate(arrays['origins']):
            w.writerow([int(origin),float(arrays['pred'][i].mean()*std+mean),
                float(arrays['distribution_mean'][i].mean()*std+mean),float(arrays['p05'][i].mean()*std+mean),
                float(arrays['p95'][i].mean()*std+mean),float(arrays['text_gate'][i].mean()),
                float(arrays['text_effect'][i].mean()*std),float(arrays['probability_above_last'][i].mean())])
    steps=np.arange(1,len(raw['pred'])+1)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    ax=axes[0]
    ax.fill_between(steps,raw['p05'],raw['p95'],alpha=.2,label='Approx. 90% conditional interval')
    for k,label in [('target','Observed'),('native_pred','Native path'),('distribution_mean','Distribution mean'),('pred','Distribution decision')]:
        ax.plot(steps,raw[k],label=label)
    ax.set_xlabel('Forecast step');ax.set_ylabel('Original target units');ax.legend(fontsize=7)
    draws=arrays['first_distribution_draws'][0,:,0]*std+mean
    weights=arrays['first_distribution_weights'][0,:,0]
    axes[1].hist(draws,bins=12,weights=weights,density=True,alpha=.6)
    axes[1].axvline(raw['pred'][0],label='Distribution decision',color='red')
    axes[1].axvline(raw['target'][0],label='Observed',color='black',linestyle='--')
    axes[1].set_title('Weighted conditional samples at forecast step 1')
    axes[1].legend(fontsize=7)
    fig.suptitle('Fixed first test origin; empirical distribution approximation')
    fig.tight_layout();fig.savefig(dest/'conditional_distribution.png',dpi=160);plt.close(fig)


def _evaluate(model, loader, device):
    model.eval()
    squared=absolute=count=total_nll=total_crps=covered=0.0
    arrays={k:[] for k in ['pred','target','origins']}
    gate_total=0.0
    with torch.inference_mode():
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            pred=model(batch);error=(pred-batch['target']).double()
            squared+=float(error.square().sum());absolute+=float(error.abs().sum());count+=error.numel()
            arrays['pred'].append(pred.cpu().numpy());arrays['target'].append(batch['target'].cpu().numpy())
            arrays['origins'].append(batch['origin'].cpu().numpy())
            if model.has_distribution:
                st=model.distribution.state
                arrays.setdefault('native_pred',[]).append(st['base'].cpu().numpy())
                if 'first_distribution_draws' not in arrays:
                    arrays['first_distribution_draws']=[st['draws'][:1].cpu().numpy()]
                    arrays['first_distribution_weights']=[st['weights'][:1].cpu().numpy()]
                nll,_,_=model.distribution.density(batch['target'])
                total_nll+=float(nll.sum())
                summaries=model.distribution.summaries()
                for k,v in summaries.items():arrays.setdefault(k,[]).append(v.cpu().numpy())
                y=batch['target'].squeeze(-1)
                covered+=float(((y>=summaries['p05'])&(y<=summaries['p95'])).sum())
                draws=st['draws'];w=st['weights']
                crps=((draws-y[:,None]).abs()*w).sum(1)
                crps=crps-.5*((draws[:,:,None]-draws[:,None,:]).abs()*w[:,:,None]*w[:,None,:]).sum((1,2))
                total_crps+=float(crps.sum());gate_total+=float(st['gate'].sum())
            model.distribution.release()
    metrics=dict(mse=squared/count,mae=absolute/count)
    if model.has_distribution:
        metrics.update(nll_per_step=total_nll/count,crps_approx=total_crps/count,
            coverage90_approx=covered/count,gate_mean=gate_total/count)
    return metrics,{k:np.concatenate(v) for k,v in arrays.items()}


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
        if not json.loads(Path(args.control_checkpoint).with_name('COMPLETED.json').read_text())['converged']:
            raise ValueError('warm start requires a converged native reference')
        native_weights = {k:v for k,v in saved_control['state_dict'].items()
                          if k.startswith(('backbone.', 'text_encoder.'))}
        missing, unexpected = model.load_state_dict(native_weights, strict=False)
        required={k for k,p in model.named_parameters() if p.requires_grad and k.startswith(('backbone.', 'text_encoder.'))}
        required.update(k for k,_ in model.named_buffers() if k.startswith(('backbone.', 'text_encoder.')))
        if unexpected or any(k in required for k in missing):
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
    from distribution_semflow_v4.native import REPOS
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
            if not json.loads((dest/'COMPLETED.json').read_text())['converged']:
                raise ValueError('test evaluation requires validation convergence')
            saved = torch.load(dest/'checkpoint.pt', map_location=device, weights_only=False)
            restore_checkpoint(model, saved)
            metrics, arrays = evaluate(model, loaders['test'], device)
            np.savez_compressed(dest/'test_predictions.npz', **arrays)
            write_json(dest/'EVALUATED.json', dict(model=args.model, domain=args.domain,
                horizon=args.horizon, seed=args.seed, variant=args.variant,
                test=metrics, best_epoch=saved['epoch']))
            if model.has_distribution:
                export_distribution(dest, arrays, train.mean, train.std)
            print('EVALUATED', json.dumps(metrics), flush=True)
            return
        if (dest/'COMPLETED.json').exists():
            print('ALREADY_COMPLETED', dest, flush=True)
            return
        if not (dest/'resume.pt').exists():
            print('RESTART_BEFORE_FIRST_RESUME_CHECKPOINT', dest, flush=True)
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
                   energy_weight=args.energy_weight, regularity_weight=0.02)
    # Score is normalized by a matched native initialization, not by test metrics.
    resuming = (dest/'resume.pt').exists()
    initial, _ = evaluate(model, loaders['val'], device)
    normal_mse = max(initial['mse'], 1e-8)
    normal_mae = max(initial['mae'], 1e-8)
    # The warm-start checkpoint is a valid candidate; never discard it merely
    # because an additional training epoch was required.
    best_score, best_epoch, stale = 1.0, 0, 0
    if not resuming:
        torch.save(dict(state_dict=checkpoint_state(model), epoch=0, validation=initial,
                        identity=identity), dest/'checkpoint.pt')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    trace = []
    started = time.monotonic()
    epochs_run = 0
    start_epoch = 1
    prior_seconds = 0.0
    if resuming:
        resume = torch.load(dest/'resume.pt', map_location=device, weights_only=False)
        restore_checkpoint(model, resume)
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        best_score, best_epoch, stale = resume['best_score'], resume['best_epoch'], resume['stale']
        normal_mse, normal_mae = resume['normal_mse'], resume['normal_mae']
        trace = resume['trace']; start_epoch = resume['epoch'] + 1
        prior_seconds = resume['seconds']
        random.setstate(resume['python_rng']); np.random.set_state(resume['numpy_rng'])
        torch.set_rng_state(resume['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in resume['cuda_rng']])
        print('RESUMED', start_epoch, flush=True)
    converged = False
    for epoch in range(start_epoch, args.epochs + 1):
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
            torch.save(dict(state_dict=checkpoint_state(model), epoch=epoch, validation=val,
                            identity=identity), dest/'checkpoint.pt')
        else:
            stale += 1
        should_stop = epoch >= args.minimum_epochs and stale >= args.patience
        if epoch % 10 == 0 or should_stop or epoch == args.epochs:
            payload = dict(state_dict=checkpoint_state(model), epoch=epoch,
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                best_score=best_score, best_epoch=best_epoch, stale=stale,
                normal_mse=normal_mse, normal_mae=normal_mae, trace=trace,
                seconds=prior_seconds+time.monotonic()-started,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
            torch.save(payload, dest/'resume.tmp')
            (dest/'resume.tmp').replace(dest/'resume.pt')
        write_json(dest/'training.json', dict(best_epoch=best_epoch, epochs_run=epoch,
                                             capped=False, trace=trace))
        print('EPOCH', epoch, json.dumps(val), 'best', best_epoch, flush=True)
        if should_stop:
            converged = True
            break
    if not converged:
        write_json(dest/'NEEDS_MORE_TRAINING.json', dict(epoch=epochs_run, stale=stale))
        raise RuntimeError('safety epoch limit reached without convergence; resume with extended limit in a new recorded protocol')
    saved = torch.load(dest/'checkpoint.pt', map_location=device, weights_only=False)
    restore_checkpoint(model, saved)
    # Test is a separate explicit stage after the entire experiment design is locked.
    metrics, _ = evaluate(model, loaders['val'], device)
    record = dict(model=args.model, domain=args.domain, horizon=args.horizon, seed=args.seed,
                  variant=args.variant, validation=metrics, best_epoch=best_epoch, epochs_run=epochs_run,
                  capped=False, converged=True, stopping='validation plateau',
                  seconds=prior_seconds+time.monotonic()-started, trace=trace)
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

