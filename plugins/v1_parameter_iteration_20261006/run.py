"""V1-P2: trained outputs, validation-only hyperparameters, no identity fallback.

Run with /xiliang/LXY/envs/lxy/bin/python. Base forecasters remain frozen.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import copy
import csv
import gc
import hashlib
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'outputs'
sys.path.insert(0, str(HERE.parent))
from semflow.data import DATA, MODELS, AdapterDataset, text_cache
from model import SemanticGraphFlow, SemanticGraphFlowConfig

DEVICE = torch.device('cuda')
SEEDS = [2026, 2027, 2028]
PILOT = {'Agriculture': 6, 'Climate': 8, 'Economy': 8, 'Energy': 36,
         'Environment': 336, 'Health': 24, 'Security': 10, 'SocialGood': 8, 'Traffic': 8}
PROTOCOL = dict(name='trained_checkpoint_recent_trend_v1', minimum_epochs=40,
                first_checkpoint_epoch=8, patience=20, trend_window=20,
                score_tolerance=1e-4, maximum_epochs=1000,
                scheduler_patience=5, scheduler_factor=0.5, min_lr=1e-8,
                required_lr_reductions=3, batch_size=256)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def atomic_checkpoint(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric(pred, target):
    error = pred.astype(np.float64) - target.astype(np.float64)
    return {'mse': float(np.mean(error ** 2)), 'mae': float(np.mean(np.abs(error)))}


def configs():
    control = dict(hidden=32, coefficients=4, samples=16, correction_limit=1.,
                   correction_init=0., variance_beta=0., pooling='uniform',
                   model_conditioned=False, gate_conditioned=False, utility_margin=.05,
                   gate_threshold=0., gate_temperature=1., utility_temperature=.25,
                   nll_weight=.03, utility_weight=.01, mae_weight=.2,
                   regularity_weight=.005, balanced=False, lr=.003)
    base = {**control, 'model_conditioned': True, 'gate_conditioned': True, 'balanced': True}
    changes = {
        'conditioned_balanced': {},
        'amplitude_025': {'correction_limit': .25},
        'amplitude_200': {'correction_limit': 2.},
        'nll_000': {'nll_weight': 0.},
        'nll_010': {'nll_weight': .1},
        'nll_030': {'nll_weight': .3},
        'utility_010': {'utility_weight': .1},
        'mae_100': {'mae_weight': 1.},
        'gate_low': {'gate_threshold': -1.},
        'gate_high': {'gate_threshold': 1.},
        'teacher_strict': {'utility_margin': .2},
        'lr_001': {'lr': .001},
    }
    return {'v1_corrected_control': control,
            **{name: {**base, **change} for name, change in changes.items()}}


class Tagged(Dataset):
    def __init__(self, data, tag):
        self.data, self.tag = data, tag

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return (*self.data[i], self.tag)


def loader(data, shuffle=False):
    return DataLoader(data, batch_size=PROTOCOL['batch_size'], shuffle=shuffle,
                      num_workers=0, pin_memory=True)


def splits(domain, horizon, split, embeddings, mask):
    return {m: AdapterDataset(m, domain, horizon, split, embeddings, mask) for m in MODELS}


def predict(module, data, model_id):
    module.eval()
    values, profiles = [], []
    with torch.inference_mode():
        for batch in loader(Tagged(data, model_id)):
            base, target, history, text, mask, tag = [v.to(DEVICE) for v in batch]
            pred, state = module(base, history, text, mask, tag)
            if not torch.isfinite(pred).all():
                raise FloatingPointError('Nonfinite prediction')
            values.append(pred.cpu().numpy())
            profiles.append(torch.stack((state['gate'], state['mixture_variance'].mean(-1),
                ((pred-base)/state['scale']).abs().mean((1, 2))), -1).cpu().numpy())
    return np.concatenate(values), np.concatenate(profiles)


def validate(module, hold):
    rows = {}
    for tag, name in enumerate(MODELS):
        pred, profile = predict(module, hold[name], tag)
        actual = metric(pred, hold[name].target)
        baseline = metric(hold[name].base, hold[name].target)
        mr = actual['mse'] / max(baseline['mse'], 1e-12)
        ar = actual['mae'] / max(baseline['mae'], 1e-12)
        rows[name] = {**actual, 'baseline_mse': baseline['mse'],
                      'baseline_mae': baseline['mae'], 'mse_ratio': mr,
                      'mae_ratio': ar, 'score': max(mr, ar),
                      'gate_mean': float(profile[:, 0].mean()),
                      'correction_std_units': float(profile[:, 2].mean())}
    return rows


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def train(stage, variant, domain, horizon, seed, embeddings, mask):
    cfg = configs()[variant]
    dest = OUT / stage / variant / str(seed) / domain / str(horizon)
    result_path = dest / 'fit_result.json'
    screen_source = OUT/'screen'/variant/str(seed)/domain/str(horizon)
    if stage == 'final' and not result_path.exists() and (screen_source/'fit_result.json').exists():
        previous = json.loads((screen_source/'fit_result.json').read_text())
        if previous['parameters'] == cfg and previous['protocol'] == PROTOCOL and previous['validation_stopped']:
            dest.mkdir(parents=True, exist_ok=True)
            for m in MODELS:
                shutil.copyfile(screen_source/f'{m}.pt', dest/f'{m}.pt')
            save(result_path, {**previous, 'stage': 'final', 'reused_from': 'screen',
                               'reuse_reason': 'Identical data, seed, parameters and complete training protocol'})
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['protocol'] != PROTOCOL or result['parameters'] != cfg:
            raise ValueError('Existing fit has different parameters/protocol')
        if not result['validation_stopped'] or min(result['best_epochs'].values()) < 8:
            raise ValueError('Existing fit is incomplete or has an initialization checkpoint')
        return result
    dest.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    fit = splits(domain, horizon, 'fit', embeddings, mask)
    hold = splits(domain, horizon, 'holdout', embeddings, mask)
    count = len(fit[MODELS[0]]) - (horizon - 1)
    if count < 1:
        raise ValueError('No fit windows remain after target-overlap purge')
    for split in (fit, hold):
        for name in MODELS:
            if not np.array_equal(split[name].target, split[MODELS[0]].target):
                raise ValueError('Models use different targets')
            if not np.array_equal(split[name].history, split[MODELS[0]].history):
                raise ValueError('Models use different histories')
    combined = ConcatDataset([Tagged(Subset(fit[m], range(count)), tag)
                              for tag, m in enumerate(MODELS)])
    training = loader(combined, shuffle=True)
    fields = SemanticGraphFlowConfig.__dataclass_fields__
    model_cfg = {k: v for k, v in cfg.items() if k in fields}
    model_cfg['coefficients'] = min(model_cfg['coefficients'], horizon)
    config = SemanticGraphFlowConfig(**model_cfg, horizon=horizon, text_dim=embeddings.shape[-1])
    module = SemanticGraphFlow(config).to(DEVICE)
    normal = [metric(fit[m].base[:count], fit[m].target[:count]) for m in MODELS]
    mse_scale = torch.tensor([r['mse'] for r in normal], device=DEVICE)
    mae_scale = torch.tensor([r['mae'] for r in normal], device=DEVICE)
    optimizer = torch.optim.AdamW(module.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        factor=.5, patience=PROTOCOL['scheduler_patience'], threshold=1e-4,
        threshold_mode='abs', min_lr=PROTOCOL['min_lr'], eps=1e-12)
    initial = validate(module, hold)  # Diagnostic only; excluded from checkpoint selection.
    initial_state = cpu_state(module)
    best_scores = {m: float('inf') for m in MODELS}
    best_rows, best_states = {}, {}
    best_epochs, stale = {m: 0 for m in MODELS}, {m: 0 for m in MODELS}
    trace, resume_epoch, previous_seconds = [], 0, 0.
    resume = dest / 'resume.pt'
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['parameters'] != cfg or state['protocol'] != PROTOCOL:
            raise ValueError('Resume state differs from the locked experiment')
        module.load_state_dict(state['last_state'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        best_scores, best_rows = state['best_scores'], state['best_rows']
        best_states, best_epochs, stale = state['best_states'], state['best_epochs'], state['stale']
        trace, resume_epoch, previous_seconds = state['trace'], state['epoch'], state['seconds']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    started = time.monotonic()
    validation_stopped = False
    trend = {m: None for m in MODELS}
    for epoch in range(resume_epoch + 1, PROTOCOL['maximum_epochs'] + 1):
        module.train()
        totals = dict(total=0., point=0., nll=0., utility=0., regularity=0.)
        seen = 0
        for batch in training:
            base, target, history, text, text_mask, tags = [v.to(DEVICE) for v in batch]
            optimizer.zero_grad(set_to_none=True)
            pred, state = module(base, history, text, text_mask, tags)
            normalizers = (mse_scale[tags], mae_scale[tags]) if cfg['balanced'] else None
            loss, pieces = module.objective(pred, target, base, state,
                nll_weight=cfg['nll_weight'], utility_weight=cfg['utility_weight'],
                mae_weight=cfg['mae_weight'], regularity_weight=cfg['regularity_weight'],
                normalizers=normalizers)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss: {variant}/{domain}/{horizon}/{seed}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            size = len(base)
            seen += size
            totals['total'] += float(loss.detach()) * size
            for name, value in pieces.items():
                totals[name] += float(value) * size
        rows = validate(module, hold)
        score = float(np.mean([r['score'] for r in rows.values()]))
        scheduler.step(score)
        if epoch >= PROTOCOL['first_checkpoint_epoch']:
            current_state = None
            for m, row in rows.items():
                if row['score'] < best_scores[m] - PROTOCOL['score_tolerance']:
                    if current_state is None:
                        current_state = cpu_state(module)
                    best_scores[m], best_rows[m] = row['score'], row
                    best_epochs[m], best_states[m], stale[m] = epoch, current_state, 0
                else:
                    stale[m] += 1
        trace.append(dict(epoch=epoch, **{k: v/seen for k, v in totals.items()},
                          validation_score=score, models=rows, stale=dict(stale),
                          lr=optimizer.param_groups[0]['lr']))
        window = PROTOCOL['trend_window']
        if len(trace) >= window:
            half = window // 2
            for m in MODELS:
                scores = [r['models'][m]['score'] for r in trace[-window:]]
                trend[m] = float(np.mean(scores[half:]) - np.mean(scores[:half]))
        lr_reduced = optimizer.param_groups[0]['lr'] <= cfg['lr'] / (2 ** PROTOCOL['required_lr_reductions'])
        no_recent_improvement = all(v is not None and v >= -PROTOCOL['score_tolerance'] for v in trend.values())
        validation_stopped = (epoch >= PROTOCOL['minimum_epochs'] and
            min(stale.values()) >= PROTOCOL['patience'] and lr_reduced and no_recent_improvement)
        progress = dict(stage=stage, variant=variant, domain=domain, horizon=horizon,
            seed=seed, epoch=epoch, best_epochs=best_epochs, stale=stale,
            recent_score_change=trend, lr=optimizer.param_groups[0]['lr'],
            validation_stopped=validation_stopped)
        save(dest / 'training_progress.json', progress)
        save(OUT / 'CURRENT_FIT.json', progress)
        if epoch % 20 == 0 or validation_stopped or epoch == PROTOCOL['maximum_epochs']:
            atomic_checkpoint(resume, dict(parameters=cfg, protocol=PROTOCOL, epoch=epoch,
                last_state=cpu_state(module), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                best_scores=best_scores, best_rows=best_rows, best_states=best_states,
                best_epochs=best_epochs, stale=stale, trace=trace,
                seconds=previous_seconds + time.monotonic()-started,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
        if validation_stopped:
            break
    if not validation_stopped:
        save(dest / 'NOT_CONVERGED.json', progress)
        raise RuntimeError(f'Safety cap reached; refusing convergence claim: {variant}/{seed}/{domain}/{horizon}')
    if min(best_epochs.values()) < PROTOCOL['first_checkpoint_epoch']:
        raise ValueError('A model has no eligible trained checkpoint')
    for m in MODELS:
        if not any(not torch.equal(best_states[m][k], initial_state[k]) for k in initial_state):
            raise RuntimeError('Selected checkpoint is identical to initialization')
        atomic_checkpoint(dest / f'{m}.pt', dict(config=asdict(config), state_dict=best_states[m],
            parameters=cfg, best_epoch=best_epochs[m], model=m, seed=seed))
    result = dict(stage=stage, variant=variant, domain=domain, horizon=horizon, seed=seed,
        parameters=cfg, protocol=PROTOCOL, initial_holdout=initial, holdout=best_rows,
        best_epochs=best_epochs, epochs_run=epoch, validation_stopped=True,
        stopping_reason='validation_no_further_improvement_after_lr_reductions',
        recent_score_change=trend, stale=stale, final_lr=optimizer.param_groups[0]['lr'],
        raw_fit_windows=len(fit[MODELS[0]]), fit_windows=count, purge=horizon-1,
        holdout_windows=len(hold[MODELS[0]]), seconds=previous_seconds + time.monotonic()-started,
        trace=trace)
    save(result_path, result)
    with (OUT / 'progress.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({k: result[k] for k in ['stage','variant','domain','horizon','seed',
            'epochs_run','best_epochs','seconds']}) + '\n')
    print('FIT_DONE', stage, variant, seed, domain, horizon, 'epochs', epoch,
          'best', best_epochs, 'seconds', round(result['seconds'], 1), flush=True)
    del module, optimizer, training, best_states
    gc.collect()
    torch.cuda.empty_cache()
    return result


def training_paths(stage, variant):
    return list((OUT / stage / variant).glob('*/*/*/fit_result.json'))


def screen_ranking():
    table = []
    for name in configs():
        paths = training_paths('screen', name)
        if len(paths) != 9:
            raise ValueError('Incomplete pilot candidate')
        values = [x for p in paths for x in json.loads(p.read_text())['holdout'].values()]
        table.append(dict(variant=name, score=float(np.mean([r['score'] for r in values])),
            validation_both_better=sum(r['mse_ratio'] < 1.-1e-7 and r['mae_ratio'] < 1.-1e-7 for r in values),
            validation_model_tasks=len(values), fits=len(paths)))
    return sorted(table, key=lambda r: (r['score'], -r['validation_both_better'], r['variant']))


def lock_finalists():
    ranking = screen_ranking()
    save(OUT / 'screen_ranking.json', ranking)
    candidates = [r['variant'] for r in ranking if r['variant'] != 'v1_corrected_control'][:2]
    finalists = ['v1_corrected_control', *candidates]
    value = dict(finalists=finalists, parameters={n: configs()[n] for n in finalists},
        selected_on='all 9 pilot domains, five raw trained model outputs, validation only',
        ranking=ranking, full_model_task_count=180, full_seeds=SEEDS)
    save(OUT / 'FINALISTS_LOCK.json', value)
    return finalists


def lock_model_selections(finalists, manifests):
    selections = []
    for meta in manifests:
        d = meta['domain']
        for h in meta['horizons']:
            for m in MODELS:
                candidates = []
                for name in finalists:
                    rs = [json.loads((OUT/'final'/name/str(s)/d/str(h)/'fit_result.json').read_text()) for s in SEEDS]
                    if not all(r['validation_stopped'] for r in rs):
                        raise ValueError('Unfinished final training')
                    mr = float(np.mean([r['holdout'][m]['mse_ratio'] for r in rs]))
                    ar = float(np.mean([r['holdout'][m]['mae_ratio'] for r in rs]))
                    candidates.append(dict(variant=name, mse_ratio=mr, mae_ratio=ar,
                                           score=max(mr, ar)))
                ranked = sorted(candidates, key=lambda x: (x['score'], x['variant']))
                selections.append(dict(model=m, domain=d, horizon=h, variant=ranked[0]['variant'],
                    validation=ranked, same_configuration_for_all_three_seeds=True))
    save(OUT / 'TEST_LOCK.json', dict(selections=selections, locked_before_new_test_access=True,
        policy='raw trained output, no external alpha, no fallback', target_dual_improvements=150,
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    return selections


def evaluate(selections, manifests):
    # Only the locked selected variant and matched control are tested/reported.
    index = {(r['domain'], r['horizon'], r['model']): r for r in selections}
    seed_rows = []
    for meta in manifests:
        d = meta['domain']
        embeddings, mask = text_cache(d, Path('/xiliang/LXY/unused_offline_bert'))
        for h in meta['horizons']:
            tests = splits(d, h, 'test', embeddings, mask)
            for tag, m in enumerate(MODELS):
                selected = index[d, h, m]['variant']
                for seed in SEEDS:
                    for label, variant in [('V1-P2', selected), ('matched_control', 'v1_corrected_control')]:
                        folder = OUT/'final'/variant/str(seed)/d/str(h)
                        checkpoint = torch.load(folder/f'{m}.pt', map_location=DEVICE, weights_only=False)
                        if checkpoint['best_epoch'] < PROTOCOL['first_checkpoint_epoch']:
                            raise ValueError('Initialization checkpoint cannot be evaluated')
                        module = SemanticGraphFlow(SemanticGraphFlowConfig(**checkpoint['config'])).to(DEVICE)
                        module.load_state_dict(checkpoint['state_dict'])
                        pred, profile = predict(module, tests[m], tag)
                        baseline = metric(tests[m].base, tests[m].target)
                        actual = metric(pred, tests[m].target)
                        destination = OUT/'predictions'/label/str(seed)/d/str(h)
                        destination.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(destination/f'{m}.npz', pred=pred, profile=profile)
                        row = dict(version=label, variant=variant, model=m, domain=d, horizon=h,
                            seed=seed, best_epoch=checkpoint['best_epoch'], baseline_mse=baseline['mse'],
                            baseline_mae=baseline['mae'], mse=actual['mse'], mae=actual['mae'],
                            mse_delta_pct=100*(actual['mse']/max(baseline['mse'],1e-12)-1),
                            mae_delta_pct=100*(actual['mae']/max(baseline['mae'],1e-12)-1),
                            equal_prediction_points=int(np.count_nonzero(pred == tests[m].base)),
                            prediction_points=int(pred.size), test_windows=len(pred),
                            max_abs_correction=float(np.max(np.abs(pred-tests[m].base))),
                            gate_mean=float(profile[:, 0].mean()))
                        seed_rows.append(row)
                        del module
                        gc.collect()
                        torch.cuda.empty_cache()
            save(OUT/'STATUS.json', dict(stage='evaluation', domain=d, horizon=h,
                 completed_test_records=len(seed_rows), expected_test_records=1080))
    if len(seed_rows) != 1080:
        raise ValueError('Incomplete test evaluation')
    write_csv('test_seed_results.csv', seed_rows)
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[row['version'], row['domain'], row['horizon'], row['model']].append(row)
    aggregates = []
    for key, rows in grouped.items():
        if len(rows) != 3 or len({r['seed'] for r in rows}) != 3:
            raise ValueError('Three independent plugin seeds required')
        row = {k: rows[0][k] for k in ['version','variant','domain','horizon','model','baseline_mse','baseline_mae']}
        for k in ['mse','mae']:
            row[k] = float(np.mean([r[k] for r in rows]))
            row[k+'_std'] = float(np.std([r[k] for r in rows], ddof=1))
            row[k+'_delta_pct'] = 100*(row[k]/max(row['baseline_'+k],1e-12)-1)
        # Exclude floating point noise from improvement counts.
        row['both_better'] = row['mse_delta_pct'] < -1e-5 and row['mae_delta_pct'] < -1e-5
        row['any_worse'] = row['mse_delta_pct'] > 1e-5 or row['mae_delta_pct'] > 1e-5
        aggregates.append(row)
    primary = [r for r in aggregates if r['version']=='V1-P2']
    if len(primary) != 180:
        raise ValueError('Incomplete 5x9x4 primary results')
    write_csv('results_180.csv', primary)
    save(OUT/'RESULTS_180.json', primary)
    write_csv('matched_control_180.csv', [r for r in aggregates if r['version']=='matched_control'])
    fits = [json.loads(p.read_text()) for p in (OUT/'final').glob('*/*/*/*/fit_result.json')]
    summary = dict(version='V1-P2', primary_model_tasks=180, plugin_seed_count=3,
        test_rows=1080, both_better=sum(r['both_better'] for r in primary),
        any_worse=sum(r['any_worse'] for r in primary), target=150,
        target_achieved=sum(r['both_better'] for r in primary)>=150,
        final_fits=len(fits), all_validation_stopped=all(r['validation_stopped'] for r in fits),
        all_selected_checkpoints_trained=all(min(r['best_epochs'].values())>=8 for r in fits),
        all_final_evaluated=True, alpha_zero_fallback=False,
        note='Validation stopping is not a proof of global convergence. Old test sets were already inspected in earlier iterations.')
    save(OUT/'FULL_COMPLETED.json', summary)
    save(OUT/'STATUS.json', {'stage':'complete', **summary})
    lines = ['# V1-P2 参数实验结果', '', f"180项中MSE与MAE同时下降：**{summary['both_better']}/180**；目标150项，{'达到' if summary['target_achieved'] else '未达到'}。", '',
        '所有主结果为训练后插件直接输出，未使用alpha=0回退。每项为三个插件随机种子的指标均值，原始底模预测保持固定。', '',
        '变化百分比=(修改后−原版)/原版×100%，负数表示改善。完整180项见results_180.csv；匹配训练规则的对照见matched_control_180.csv。', '',
        '完成指满足预先规定的验证停止条件，不证明数学上的全局收敛。旧测试集已在此前迭代中被查看，本轮是开发性实验，论文结论还需要新的独立测试集。', '',
        '|模型|双指标下降项数|至少一项上升项数|', '|---|---:|---:|']
    for m in MODELS:
        rows = [r for r in primary if r['model']==m]
        lines.append(f"|{m}|{sum(r['both_better'] for r in rows)}/36|{sum(r['any_worse'] for r in rows)}/36|")
    (OUT/'V1-P2参数实验结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('V1_P2_COMPLETE', json.dumps(summary), flush=True)


def main():
    if Path(sys.prefix).resolve() != Path('/xiliang/LXY/envs/lxy'):
        raise RuntimeError('Experiments must run in lxy only')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != os.environ.get('V1_P2_GPU'):
        raise RuntimeError('GPU selection must match launch record')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.05)
    OUT.mkdir(exist_ok=True)
    marker = json.loads((ROOT/'INPUT_DOWNLOAD_COMPLETE.json').read_text())
    for r in marker['records']:
        if hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest() != r['sha256']:
            raise ValueError('Frozen input changed: '+r['path'])
    source = json.loads((HERE/'SOURCE_MANIFEST.json').read_text())
    for name, sha in source['files'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest() != sha:
            raise ValueError('Deployed source changed: '+name)
    save(OUT/'INPUT_AUDIT.json', dict(files=len(marker['records']), all_sha256_equal=True))
    plan = dict(version='V1-P2', backup_tag='pre_v1_p2_iteration_20261006', configs=configs(),
        pilot=PILOT, protocol=PROTOCOL, screening_fits=117, finalist_count=3,
        final_fits=324, identical_pilot_final_fits_reused=27, new_final_fits=297,
        seeds=SEEDS, test_rows=1080, target_dual_improvements=150,
        alpha_zero_fallback=False, selection='validation only; no test ranking; raw trained output',
        forecasters='original five frozen forecasters', architecture='V1 graph flow with conditioned gate ablation')
    save(OUT/'PLAN.json', plan)
    metas = [json.loads(p.read_text()) for p in sorted(DATA.glob('*/manifest.json'))]
    if len(metas)!=9:
        raise ValueError('Nine domains required')
    if not (OUT/'FINALISTS_LOCK.json').exists():
        done = 0
        for meta in metas:
            d, h = meta['domain'], PILOT[meta['domain']]
            embeddings, mask = text_cache(d, Path('/xiliang/LXY/unused_offline_bert'))
            for name in configs():
                save(OUT/'STATUS.json', dict(stage='screen', variant=name, domain=d,
                     horizon=h, seed=2026, completed_fits=done, total_fits=117))
                train('screen', name, d, h, 2026, embeddings, mask)
                done += 1
        finalists = lock_finalists()
    else:
        finalists = json.loads((OUT/'FINALISTS_LOCK.json').read_text())['finalists']
    done = 0
    for meta in metas:
        d = meta['domain']
        embeddings, mask = text_cache(d, Path('/xiliang/LXY/unused_offline_bert'))
        for h in meta['horizons']:
            for seed in SEEDS:
                for name in finalists:
                    save(OUT/'STATUS.json', dict(stage='final', variant=name, domain=d, horizon=h,
                         seed=seed, completed_fits=done, total_fits=108*len(finalists)))
                    train('final', name, d, h, seed, embeddings, mask)
                    done += 1
    for name in finalists:
        if len(training_paths('final', name)) != 108:
            raise ValueError('All final fits are required before test access')
    if not (OUT/'TEST_LOCK.json').exists():
        selections = lock_model_selections(finalists, metas)
    else:
        selections = json.loads((OUT/'TEST_LOCK.json').read_text())['selections']
    evaluate(selections, metas)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save(OUT/'FAILED.json', dict(error=str(error), type=type(error).__name__))
        raise
