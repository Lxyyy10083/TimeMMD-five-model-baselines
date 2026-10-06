"""Resumable holdout-only parameter search, followed by locked test evaluation."""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from semflow.data import DATA, MODELS, AdapterDataset, text_cache
from model import SemanticGraphFlow, SemanticGraphFlowConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT = HERE / 'outputs'
PILOT = {'Agriculture': 6, 'Climate': 8, 'Economy': 8, 'Energy': 36,
         'Environment': 336, 'Health': 24, 'Security': 10, 'SocialGood': 8, 'Traffic': 8}


def variants():
    common = dict(coefficients=4, hidden=32, samples=16, correction_limit=0.5,
                  correction_init=-2.0, variance_beta=1.0, pooling='uniform',
                  model_conditioned=False, nll_weight=0.03, utility_weight=0.01,
                  mae_weight=0.2, regularity_weight=0.005, balanced=False, text_mode='real')
    stronger = {**common, 'correction_limit': 1.0, 'correction_init': 0.0,
                'variance_beta': 0.5}
    combined = {**stronger, 'correction_limit': 2.0, 'variance_beta': 0.25,
                'coefficients': 8, 'pooling': 'attention', 'model_conditioned': True,
                'nll_weight': 0.1, 'utility_weight': 0.05, 'mae_weight': 0.5,
                'regularity_weight': 0.001, 'balanced': True}
    return {
        'control': common,
        'amplitude_1': {**common, 'correction_limit': 1.0, 'correction_init': 0.0},
        'amplitude_2': {**common, 'correction_limit': 2.0, 'correction_init': 0.0},
        'no_variance_shrink': {**stronger, 'variance_beta': 0.0},
        'coeff_8': {**stronger, 'coefficients': 8},
        'coeff_16': {**stronger, 'coefficients': 16},
        'nll_010': {**stronger, 'nll_weight': 0.1},
        'nll_030': {**stronger, 'nll_weight': 0.3},
        'utility_005': {**stronger, 'nll_weight': 0.1, 'utility_weight': 0.05},
        'time_attention': {**stronger, 'nll_weight': 0.1, 'pooling': 'attention'},
        'model_condition': {**stronger, 'nll_weight': 0.1, 'model_conditioned': True},
        'combined': combined,
        'numeric_ablation': {**combined, 'text_mode': 'none'},
        'time_shuffle_ablation': {**combined, 'text_mode': 'shuffle'},
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def metric(pred, true):
    e = pred.astype(np.float64) - true.astype(np.float64)
    return dict(mse=float(np.mean(e * e)), mae=float(np.mean(np.abs(e))))


def manifests():
    return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(DATA.glob('*/manifest.json'))]


class Tagged(Dataset):
    def __init__(self, data, tag):
        self.data, self.tag = data, tag

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return (*self.data[i], self.tag)


def load_split(domain, horizon, split, embedding, mask):
    return {name: AdapterDataset(name, domain, horizon, split, embedding, mask) for name in MODELS}


def loader(dataset, shuffle=False, batch=256):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=0,
                      pin_memory=DEVICE.type == 'cuda')


def alter_text(text, mask, mode):
    if mode == 'none':
        return torch.zeros_like(text), torch.zeros_like(mask)
    if mode == 'shuffle':
        # Reorder only text already inside each causal history; no future text.
        generator = torch.Generator().manual_seed(9173 + text.shape[1])
        order = torch.randperm(text.shape[1], generator=generator).to(text.device)
        return text[:, order], mask[:, order]
    return text, mask


def prediction(module, data, tag, mode):
    module.eval()
    preds, profiles = [], []
    with torch.inference_mode():
        for batch in loader(Tagged(data, tag)):
            base, target, history, text, mask, model_id = [x.to(DEVICE) for x in batch]
            text, mask = alter_text(text, mask, mode)
            pred, state = module(base, history, text, mask, model_id)
            preds.append(pred.cpu().numpy())
            correction = (pred - base) / state['scale']
            profiles.append(torch.stack((state['gate'], state['mixture_variance'].mean(-1),
                                         correction.abs().mean((1, 2))), -1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(profiles)


def select_alpha(base, raw, target):
    baseline = metric(base, target)
    best = dict(alpha=0.0, mse_ratio=1.0, mae_ratio=1.0, score=1.0)
    raw_metric = metric(raw, target)
    for alpha in (0.25, 0.5, 0.75, 1.0):
        current = metric(base + alpha * (raw - base), target)
        mse = current['mse'] / max(baseline['mse'], 1e-12)
        mae = current['mae'] / max(baseline['mae'], 1e-12)
        # Same rule for every variant: both holdout metrics must improve.
        if mse < 1.0 and mae < 1.0 and 0.5 * (mse + mae) < best['score']:
            best = dict(alpha=alpha, mse_ratio=mse, mae_ratio=mae, score=0.5*(mse + mae))
    return {**best, 'raw_mse_ratio': raw_metric['mse'] / max(baseline['mse'], 1e-12),
            'raw_mae_ratio': raw_metric['mae'] / max(baseline['mae'], 1e-12)}


def holdout_score(module, hold, cfg):
    rows = {}
    for i, name in enumerate(MODELS):
        raw, profile = prediction(module, hold[name], i, cfg['text_mode'])
        rows[name] = {**select_alpha(hold[name].base, raw, hold[name].target),
                      'gate_mean': float(profile[:, 0].mean()),
                      'variance_mean': float(profile[:, 1].mean()),
                      'correction_std_units': float(profile[:, 2].mean())}
    return float(np.mean([r['score'] for r in rows.values()])), rows


def train_case(stage, variant, cfg, domain, horizon, seed, args, embedding, mask):
    dest = OUTPUT / stage / variant / str(seed) / domain / str(horizon)
    if (dest / 'fit_result.json').exists():
        previous = json.loads((dest / 'fit_result.json').read_text(encoding='utf-8'))
        if stage == 'final' and getattr(args, 'require_convergence', False):
            if not previous.get('converged') or previous.get('convergence_protocol') != 'v2_raw_five_model_plateau':
                raise RuntimeError('Existing final fit has a different convergence protocol')
            return previous
        extend = (stage == 'final' and previous['epochs_run'] == 40
                  and args.final_epochs > 40 and previous['best_epoch'] >= 40 - args.patience)
        if not extend:
            return previous
        archive = dest / 'budget40'
        archive.mkdir(exist_ok=True)
        shutil.copy2(dest / 'fit_result.json', archive / 'fit_result.json')
        shutil.copy2(dest / 'module.pt', archive / 'module.pt')
    dest.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    fit = load_split(domain, horizon, 'fit', embedding, mask)
    hold = load_split(domain, horizon, 'holdout', embedding, mask)
    fit_count = len(fit[MODELS[0]])
    usable_count = fit_count - (horizon - 1)
    if usable_count < 1:
        raise ValueError(f'no purged fit windows for {domain}/{horizon}')
    # Last retained fit target ends strictly before the first holdout origin.
    combined = ConcatDataset([Tagged(Subset(ds, range(usable_count)), i)
                              for i, ds in enumerate(fit.values())])
    training = loader(combined, True, args.batch_size)
    config_keys = SemanticGraphFlowConfig.__dataclass_fields__
    config = SemanticGraphFlowConfig(**{k: v for k, v in cfg.items() if k in config_keys},
                                      horizon=horizon, text_dim=embedding.shape[-1])
    if config.coefficients > horizon:
        config = SemanticGraphFlowConfig(**{**asdict(config), 'coefficients': horizon})
    module = SemanticGraphFlow(config).to(DEVICE)
    normal_mse, normal_mae = [], []
    for ds in fit.values():
        base_metric = metric(ds.base[:usable_count], ds.target[:usable_count])
        normal_mse.append(base_metric['mse']); normal_mae.append(base_metric['mae'])
    mse_scale = torch.tensor(normal_mse, device=DEVICE, dtype=torch.float32)
    mae_scale = torch.tensor(normal_mae, device=DEVICE, dtype=torch.float32)
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr, weight_decay=1e-4)
    best_score, best_rows = holdout_score(module, hold, cfg)
    require_convergence = stage == 'final' and getattr(args, 'require_convergence', False)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, threshold=1e-5,
        threshold_mode='abs', min_lr=1e-6) if require_convergence else None)
    raw_best = {m: 0.5*(r['raw_mse_ratio']+r['raw_mae_ratio']) for m,r in best_rows.items()}
    raw_stale = {m: 0 for m in MODELS}
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    stale, trace = 0, []
    epochs = args.final_epochs if stage == 'final' else args.screen_epochs
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        module.train()
        totals = dict(total=0.0, point=0.0, nll=0.0, utility=0.0, regularity=0.0)
        n = 0
        for batch in training:
            base, target, history, text, text_mask, model_id = [x.to(DEVICE) for x in batch]
            text, text_mask = alter_text(text, text_mask, cfg['text_mode'])
            optimizer.zero_grad(set_to_none=True)
            pred, state = module(base, history, text, text_mask, model_id)
            normalizers = (mse_scale[model_id], mae_scale[model_id]) if cfg['balanced'] else None
            loss, components = module.objective(pred, target, base, state,
                nll_weight=cfg['nll_weight'], utility_weight=cfg['utility_weight'],
                mae_weight=cfg['mae_weight'], regularity_weight=cfg['regularity_weight'],
                normalizers=normalizers)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite loss {stage}/{variant}/{seed}/{domain}/{horizon}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            size = len(base); n += size
            totals['total'] += float(loss.detach()) * size
            for k in components:
                totals[k] += float(components[k]) * size
        score, rows = holdout_score(module, hold, cfg)
        if require_convergence:
            for m,r in rows.items():
                raw_score = 0.5*(r['raw_mse_ratio']+r['raw_mae_ratio'])
                if raw_score < raw_best[m]-1e-5:
                    raw_best[m], raw_stale[m] = raw_score, 0
                else:
                    raw_stale[m] += 1
            scheduler.step(float(np.mean([0.5*(r['raw_mse_ratio']+r['raw_mae_ratio']) for r in rows.values()])))
        trace.append(dict(epoch=epoch, **{k: v/n for k, v in totals.items()},
                          holdout_score=score, models=rows,
                          lr=optimizer.param_groups[0]['lr'], raw_stale=dict(raw_stale),
                          correction_logit=float(module.correction_logit.detach())))
        if score < best_score - 1e-5:
            best_score, best_rows, best_epoch = score, rows, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if require_convergence:
            write_json(dest/'training_progress.json', dict(epoch=epoch, best_epoch=best_epoch,
                lr=optimizer.param_groups[0]['lr'], selected_stale=stale, raw_stale=raw_stale,
                domain=domain, horizon=horizon, seed=seed, variant=variant))
        if (epoch >= args.minimum_epochs and stale >= args.patience
                and (not require_convergence or min(raw_stale.values()) >= args.patience)):
            break
    converged = (len(trace) >= args.minimum_epochs and stale >= args.patience
                 and (not require_convergence or min(raw_stale.values()) >= args.patience))
    if require_convergence and not converged:
        write_json(dest/'NOT_CONVERGED.json', dict(epochs_run=len(trace), best_epoch=best_epoch,
            selected_stale=stale, raw_stale=raw_stale, trace=trace))
        raise RuntimeError(f'Final fit did not converge at safety cap: {variant}/{seed}/{domain}/{horizon}')
    torch.save(dict(config=asdict(config), state_dict=best_state, variant=cfg,
                    alphas={k: v['alpha'] for k, v in best_rows.items()}, seed=seed), dest / 'module.pt')
    result = dict(stage=stage, variant=variant, seed=seed, domain=domain, horizon=horizon,
                  parameters=cfg, model_config=asdict(config), epochs_run=len(trace),
                  converged=converged, convergence_protocol=('v2_raw_five_model_plateau' if require_convergence else 'screen_budget'),
                  stopping_reason=('validation_plateau' if converged else 'screen_budget_cap'),
                  maximum_epochs=epochs, minimum_epochs=args.minimum_epochs, patience=args.patience,
                  raw_stale=raw_stale, selected_stale=stale, final_lr=optimizer.param_groups[0]['lr'],
                  best_epoch=best_epoch, holdout_score=best_score, holdout=best_rows,
                  raw_fit_windows=fit_count, fit_windows=usable_count,
                  purge=horizon-1, holdout_windows=len(hold[MODELS[0]]),
                  seconds=round(time.monotonic()-started, 2), trace=trace)
    write_json(dest / 'fit_result.json', result)
    with (OUTPUT / 'progress.jsonl').open('a', encoding='utf-8') as log:
        log.write(json.dumps({k: result[k] for k in ('stage','variant','seed','domain','horizon',
                                                    'best_epoch','holdout_score','seconds')})+'\n')
    print('FIT_DONE', stage, variant, seed, domain, horizon, 'score', best_score,
          'epoch', best_epoch, 'seconds', result['seconds'], flush=True)
    del module, optimizer, training, combined, fit, hold, best_state
    gc.collect(); torch.cuda.empty_cache()
    return result


def ranking(stage, names, seeds):
    ranking_rows = []
    for name in names:
        paths = list((OUTPUT / stage / name).glob('*/*/*/fit_result.json'))
        results = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
        results = [r for r in results if r['seed'] in seeds]
        expected = len(PILOT) * len(seeds)
        if len(results) != expected:
            raise ValueError(f'incomplete ranking {stage}/{name}: {len(results)}/{expected}')
        entries = [x for r in results for x in r['holdout'].values()]
        ranking_rows.append(dict(variant=name, cases=len(results),
            score=float(np.mean([r['holdout_score'] for r in results])),
            mse_ratio=float(np.mean([e['mse_ratio'] for e in entries])),
            mae_ratio=float(np.mean([e['mae_ratio'] for e in entries])),
            raw_mse_ratio=float(np.mean([e['raw_mse_ratio'] for e in entries])),
            raw_mae_ratio=float(np.mean([e['raw_mae_ratio'] for e in entries])),
            enabled=int(sum(e['alpha'] > 0 for e in entries))))
    return sorted(ranking_rows, key=lambda x: x['score'])


def audit():
    rows = []
    for meta in manifests():
        domain = meta['domain']
        for horizon in meta['horizons']:
            for name in MODELS:
                folder = HERE.parent / 'adapters' / name / domain / str(horizon)
                with np.load(folder / 'fit.npz') as fit:
                    n = len(fit['target']); usable = n - (horizon-1)
                    if usable < 1:
                        raise ValueError(f'no purged calibration data: {domain}/{horizon}')
                    residual = (fit['target'][:usable] - fit['base_pred'][:usable]).squeeze(-1)
                    if not np.isfinite(residual).all():
                        raise ValueError('nonfinite calibration arrays')
                    h = np.arange(horizon)[:,None] + 0.5
                    k = np.arange(horizon)[None,:]
                    basis = np.cos(np.pi*h*k/horizon)*np.sqrt(2/horizon)
                    basis[:,0] /= np.sqrt(2)
                    coeff = residual @ basis
                    total = np.square(coeff).sum()
                    energy = {f'energy_k{size}': float(np.square(coeff[:,:min(size,horizon)]).sum()/max(total,1e-12))
                              for size in (4,8,16,32)}
                    baseline = metric(fit['base_pred'][:usable], fit['target'][:usable])
                    rows.append(dict(model=name,domain=domain,horizon=horizon,fit_raw=n,
                                     fit_purged=usable,fit_mse=baseline['mse'],fit_mae=baseline['mae'],**energy))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/'data_audit.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    write_json(OUTPUT/'experiment_config.json',dict(variants=variants(),pilot=PILOT,
                 protocol='fit tail purged H-1 origins; holdout selects epoch+alpha; test only after winner lock'))
    print('AUDIT_DONE',len(rows),flush=True)


def screen(args):
    names = list(variants())
    for meta in manifests():
        domain = meta['domain']; horizon = PILOT[domain]
        embedding, mask = text_cache(domain, Path(args.bert))
        for name in names:
            train_case('screen', name, variants()[name], domain, horizon, 2026, args, embedding, mask)
    scores=ranking('screen',names,[2026])
    write_json(OUTPUT/'screen_ranking.json',scores)
    promoted=[r['variant'] for r in scores if 'ablation' not in r['variant']][:3]
    write_json(OUTPUT/'promoted.json',promoted)
    return promoted


def confirm(args, promoted):
    for meta in manifests():
        domain=meta['domain'];horizon=PILOT[domain]
        embedding,mask=text_cache(domain,Path(args.bert))
        for name in promoted:
            train_case('screen',name,variants()[name],domain,horizon,2027,args,embedding,mask)
    scores=ranking('screen',promoted,[2026,2027])
    winner=scores[0]['variant']
    write_json(OUTPUT/'confirmation_ranking.json',scores)
    write_json(OUTPUT/'winner.json',dict(variant=winner,parameters=variants()[winner],
               ranking=scores,selected_on='purged holdout only, 9 domains x 5 models x 2 seeds'))
    print('WINNER_LOCKED',winner,flush=True)
    return winner


def final_fit(args,winner):
    for meta in manifests():
        domain=meta['domain'];embedding,mask=text_cache(domain,Path(args.bert))
        for horizon in meta['horizons']:
            for seed in (2026,2027,2028):
                train_case('final',winner,variants()[winner],domain,horizon,seed,args,embedding,mask)


def evaluate(args,winner):
    paths=sorted((OUTPUT/'final'/winner).glob('*/*/*/fit_result.json'))
    if len(paths)!=108:
        raise ValueError(f'only {len(paths)}/108 final fits; refusing partial test evaluation')
    rows=[]
    for meta in manifests():
        domain=meta['domain'];embedding,mask=text_cache(domain,Path(args.bert))
        for horizon in meta['horizons']:
            test=load_split(domain,horizon,'test',embedding,mask)
            for seed in (2026,2027,2028):
                folder=OUTPUT/'final'/winner/str(seed)/domain/str(horizon)
                result_path=folder/'test_result.json'
                if result_path.exists():
                    rows.extend(json.loads(result_path.read_text(encoding='utf-8')))
                    continue
                checkpoint=torch.load(folder/'module.pt',map_location=DEVICE,weights_only=False)
                module=SemanticGraphFlow(SemanticGraphFlowConfig(**checkpoint['config'])).to(DEVICE)
                module.load_state_dict(checkpoint['state_dict'])
                case_rows=[]
                for tag,name in enumerate(MODELS):
                    raw,profile=prediction(module,test[name],tag,checkpoint['variant']['text_mode'])
                    alpha=checkpoint['alphas'][name]
                    chosen=test[name].base+alpha*(raw-test[name].base)
                    base=metric(test[name].base,test[name].target)
                    tuned=metric(chosen,test[name].target);raw_metrics=metric(raw,test[name].target)
                    row=dict(model=name,domain=domain,horizon=horizon,seed=seed,variant=winner,
                             alpha=alpha,baseline_mse=base['mse'],baseline_mae=base['mae'],
                             plugin_mse=tuned['mse'],plugin_mae=tuned['mae'],
                             raw_mse=raw_metrics['mse'],raw_mae=raw_metrics['mae'],
                             mse_delta_pct=100*(tuned['mse']/base['mse']-1),
                             mae_delta_pct=100*(tuned['mae']/base['mae']-1),
                             gate_mean=float(profile[:,0].mean()),
                             correction_std_units=float(profile[:,2].mean()),test_windows=len(raw))
                    np.savez_compressed(folder/f'{name}_test.npz',raw=raw,chosen=chosen,profile=profile)
                    case_rows.append(row)
                write_json(result_path,case_rows);rows.extend(case_rows)
                del module;gc.collect();torch.cuda.empty_cache()
    with (OUTPUT/'test_seed_results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    write_json(OUTPUT/'COMPLETED.json',dict(winner=winner,final_fits=len(paths),test_rows=len(rows),
               seed_count=3,model_tasks=180))
    print('EXPERIMENT_COMPLETED',len(rows),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['all','audit','screen','confirm','fit','evaluate'],default='all')
    p.add_argument('--bert',default='/root/autodl-tmp/carma_workspace/models/bert-base-uncased')
    p.add_argument('--screen-epochs',type=int,default=24)
    p.add_argument('--final-epochs',type=int,default=80)
    p.add_argument('--minimum-epochs',type=int,default=8)
    p.add_argument('--patience',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--lr',type=float,default=0.001)
    args=p.parse_args()
    torch.set_num_threads(4)
    write_json(OUTPUT / f'runtime_{args.stage}.json', vars(args))
    if args.stage in ('all','audit'):audit()
    if args.stage in ('all','screen'):promoted=screen(args)
    else:promoted=json.loads((OUTPUT/'promoted.json').read_text()) if (OUTPUT/'promoted.json').exists() else []
    if args.stage in ('all','confirm'):winner=confirm(args,promoted)
    else:winner=json.loads((OUTPUT/'winner.json').read_text())['variant'] if (OUTPUT/'winner.json').exists() else None
    if args.stage in ('all','fit'):final_fit(args,winner)
    if args.stage in ('all','evaluate'):evaluate(args,winner)


if __name__=='__main__':
    main()
