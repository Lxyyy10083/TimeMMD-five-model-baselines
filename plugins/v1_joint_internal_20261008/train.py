"""单模型内部联合训练、断点续训和验证选优；训练阶段不读取测试集。"""
from __future__ import annotations
import argparse
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'plugins'))
from plugins.v1_joint_internal_20261008.native import JointForecast, config_from_upstream, REPOS
from plugins.v1_joint_internal_20261008.data import JointWindows

MODELS = ('TaTS', 'MM-TSFlib', 'SpecTF', 'CFA', 'Aurora')


class TrainingBudgetReached(RuntimeError):
    """Exit 42 means recoverable safety cap, never convergence or arbitrary failure."""


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def state_dict(model):
    # Aurora的大型冻结预训练参数由本地资产加载，保存所有实际可训练参数及buffer。
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    buffers = {name for name, _ in model.named_buffers()}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in trainable or k in buffers}


def restore(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    required = {name for name, p in model.named_parameters() if p.requires_grad}
    if unexpected or required.intersection(missing):
        raise ValueError(f'checkpoint mismatch: unexpected={unexpected}, trainable_missing={required.intersection(missing)}')


def evaluate(model, loader, device):
    # Informer评估也会采样注意力索引；固定评估RNG后恢复，避免污染训练RNG。
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.manual_seed(2026)
        model.eval()
        predictions, targets, origins, draws, weights, nlls = [], [], [], [], [], []
        for raw in loader:
            batch = {k: v.to(device) for k, v in raw.items()}
            prediction = model(batch)
            predictions.append(prediction.cpu().numpy())
            targets.append(batch['target'].cpu().numpy())
            origins.append(batch['origin'].cpu().numpy())
            if model.has_distribution:
                nlls.extend(model.distribution.density(batch['target'])[0].cpu().numpy().tolist())
                draws.append(model.distribution.state['draws'].cpu().numpy())
                weights.append(model.distribution.state['sample_weights'].cpu().numpy())
            model.distribution.release()
    arrays = dict(pred=np.concatenate(predictions), target=np.concatenate(targets), origins=np.concatenate(origins))
    error = arrays['pred'].astype(np.float64) - arrays['target'].astype(np.float64)
    metrics = dict(mse=float(np.mean(error ** 2)), mae=float(np.mean(np.abs(error))))
    if draws:
        arrays.update(draws=np.concatenate(draws), sample_weights=np.concatenate(weights))
        metrics['nll_per_step'] = float(np.mean(nlls) / model.horizon)
    return metrics, arrays


def source_digest(model):
    files = list(HERE.glob('*.py'))
    files += list((ROOT / 'plugins/internal_semflow_full_lab').glob('*.py'))
    files += list((ROOT / REPOS[model]).rglob('*.py'))
    digest = hashlib.sha256()
    for file in sorted(files):
        digest.update(str(file.relative_to(ROOT)).replace('\\', '/').encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def run(args):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device)
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    inputs = Path(args.inputs_root).resolve()
    manifest = json.loads((inputs / 'benchmark/readgpt_data' / args.domain / 'manifest.json').read_text())
    config = config_from_upstream(args.model, args.domain, args.horizon, manifest['seq_len'], args.seed)
    model = JointForecast(args.model, config, args.horizon, args.variant, args.width, args.samples).to(device)
    dest = Path(args.output).resolve() / args.variant / args.model / args.domain / str(args.horizon) / str(args.seed)
    dest.mkdir(parents=True, exist_ok=True)
    settings = {k: v for k, v in vars(args).items() if k not in ('stage', 'epochs', 'device')}
    identity = dict(settings=settings, source_sha256=source_digest(args.model), manifest=manifest,
                    native_config=vars(config) if config else None)
    identity_path = dest / 'run_config.json'
    if identity_path.exists() and json.loads(identity_path.read_text(encoding='utf-8')) != identity:
        raise ValueError('existing run has different source/config; use a new output directory')
    write_json(identity_path, identity)
    dataset = lambda split: JointWindows(inputs, args.domain, args.horizon, split, config, args.bert, args.model == 'Aurora')
    make_loader = lambda ds, shuffle=False: DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)
    if args.stage == 'evaluate':
        completion = json.loads((dest / 'COMPLETED.json').read_text(encoding='utf-8'))
        if not completion['converged'] or not (Path(args.output) / 'TEST_LOCK.json').exists():
            raise ValueError('test requires completed convergence and TEST_LOCK.json')
        saved = torch.load(dest / 'checkpoint.pt', map_location=device, weights_only=False)
        lock = json.loads((Path(args.output) / 'TEST_LOCK.json').read_text(encoding='utf-8'))
        entries = [entry for entry in lock['trained'] if entry['task'] == [args.model, args.domain, args.horizon, args.seed]]
        if len(entries) != 1 or entries[0]['source_sha256'] != identity['source_sha256'] or entries[0]['checkpoint_sha256'] != hashlib.sha256((dest / 'checkpoint.pt').read_bytes()).hexdigest():
            raise ValueError('test checkpoint/source differs from locked training')
        restore(model, saved['state_dict'])
        test = dataset('test')
        metrics, arrays = evaluate(model, make_loader(test), device)
        original = test.original_test(args.model)
        if not np.array_equal(arrays['origins'], test.origins) or arrays['target'].shape != original['target'].shape:
            raise ValueError('original 180-task test windows do not match')
        if not np.allclose(arrays['target'], original['target'], rtol=1e-6, atol=1e-6):
            raise ValueError('original and new test targets use different scales or ordering')
        if not np.allclose(test.numeric[np.stack([np.arange(o-test.length, o) for o in test.origins])],
                           original['history'], rtol=1e-6, atol=1e-6):
            raise ValueError('original and new historical windows differ')
        # 使用原始冻结target重算两组指标，保证主表完全相同的评价目标。
        target = original['target'].astype(np.float64)
        def error_metrics(pred):
            diff = pred.astype(np.float64) - target
            return dict(mse=float(np.mean(diff ** 2)), mae=float(np.mean(abs(diff))))
        baseline = error_metrics(original['base_pred'])
        actual = error_metrics(arrays['pred'])
        arrays.update(original_pred=original['base_pred'], original_target=original['target'])
        np.savez_compressed(dest / 'test_predictions.npz', **arrays)
        write_json(dest / 'EVALUATED.json', dict(model=args.model, domain=args.domain, horizon=args.horizon,
            seed=args.seed, variant=args.variant, original=baseline, test={**metrics, **actual},
            decrease_pct={k: 100 * (1-actual[k]/max(baseline[k], 1e-12)) for k in ('mse', 'mae')},
            target_max_abs_diff=float(np.max(np.abs(arrays['target'] - original['target'])))))
        return
    if (dest / 'COMPLETED.json').exists():
        return
    # 【修改10】训练段训练，验证段选权重；这里不构造test Dataset。
    training, holdout = dataset('train'), dataset('val')
    train_loader, val_loader = make_loader(training, True), make_loader(holdout)
    plugin, backbone = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (plugin if name.startswith(('distribution.', 'fusions.')) else backbone).append(parameter)
    # 【修改11】底模和内部插件参数同时进入优化器，使用各自的学习率。
    groups = [dict(params=backbone, lr=args.backbone_lr)]
    if plugin:
        groups.append(dict(params=plugin, lr=args.plugin_lr))
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5,
        threshold=1e-5, threshold_mode='abs', min_lr=1e-8, eps=1e-12)
    initial, _ = evaluate(model, val_loader, device)
    normal_mse, normal_mae = max(initial['mse'], 1e-8), max(initial['mae'], 1e-8)
    best_score, best_epoch, stale, start_epoch, trace = float('inf'), None, 0, 1, []
    prior_seconds = 0.0
    resume_path = dest / 'resume.pt'
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        restore(model, saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        best_score, best_epoch, stale = saved['best_score'], saved['best_epoch'], saved['stale']
        normal_mse, normal_mae = saved['normal_mse'], saved['normal_mae']
        trace, start_epoch, prior_seconds = saved['trace'], saved['epoch'] + 1, saved['seconds']
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    write_json(dest / 'STARTED.json', dict(device=str(device), train_windows=len(training), val_windows=len(holdout),
        backbone_trainable=sum(p.numel() for p in backbone), plugin_trainable=sum(p.numel() for p in plugin)))
    started = time.monotonic()
    converged = False
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); totals = []; counts = []
        for raw in train_loader:
            batch = {k: v.to(device) for k, v in raw.items()}
            optimizer.zero_grad(set_to_none=True)
            model(batch)
            loss, parts = model.objective(batch['target'], mae_weight=args.mae_weight,
                nll_weight=args.nll_weight * min(1, epoch/args.warmup),
                utility_weight=args.utility_weight, regularity_weight=args.regularity_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite joint training loss')
            # 【修改12】一次backward同时更新原模型隐藏层、内部适配器和条件flow。
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            totals.append(dict(loss=float(loss.detach()), **parts)); counts.append(len(batch['target']))
            model.distribution.release()
        val, _ = evaluate(model, val_loader, device)
        score = 0.5 * (val['mse']/normal_mse + val['mae']/normal_mae)
        scheduler.step(score)
        # 初始化epoch=0仅作诊断；至少训练8轮且完成辅助损失升温才可选权重。
        eligible = epoch >= max(args.checkpoint_min_epoch, args.warmup)
        if eligible and score < best_score - 1e-5:
            best_score, best_epoch, stale = score, epoch, 0
            save_checkpoint(dest / 'checkpoint.pt', dict(state_dict=state_dict(model), epoch=epoch, validation=val))
        elif eligible:
            stale += 1
        trace.append(dict(epoch=epoch, score=score, validation=val,
            train={k: float(np.average([r[k] for r in totals], weights=counts)) for k in totals[0]},
            lr=[g['lr'] for g in optimizer.param_groups], stale=stale))
        # 最佳验证分数无改善还需最近曲线不再明显恢复，避免过早宣称平台期。
        stable_recent = len(trace) >= args.patience and args.patience >= 2
        if stable_recent:
            recent = [r['score'] for r in trace[-args.patience:]]
            half = len(recent)//2
            stable_recent = np.mean(recent[:half]) - np.mean(recent[half:]) <= 1e-5
        converged = bool(epoch >= args.minimum_epochs and best_epoch is not None and
                         stale >= args.patience and stable_recent)
        if epoch % 10 == 0 or converged or epoch == args.epochs:
            save_checkpoint(resume_path, dict(state_dict=state_dict(model), epoch=epoch,
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), best_score=best_score,
                best_epoch=best_epoch, stale=stale, normal_mse=normal_mse, normal_mae=normal_mae,
                trace=trace, seconds=prior_seconds+time.monotonic()-started,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else []))
        write_json(dest / 'training.json', dict(converged=converged, best_epoch=best_epoch, trace=trace))
        print('EPOCH', epoch, json.dumps(val), 'best', best_epoch, flush=True)
        if converged:
            break
    if not converged:
        write_json(dest / 'NOT_CONVERGED.json', dict(safety_limit=args.epochs, best_epoch=best_epoch,
            message='resume by increasing --epochs; cap is not convergence'))
        raise TrainingBudgetReached('safety limit reached without validation plateau; no final test evaluated')
    saved = torch.load(dest / 'checkpoint.pt', map_location=device, weights_only=False)
    restore(model, saved['state_dict'])
    write_json(dest / 'COMPLETED.json', dict(converged=True, best_epoch=best_epoch,
        epochs_run=trace[-1]['epoch'], validation=saved['validation'], source_sha256=identity['source_sha256']))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--domain', required=True)
    p.add_argument('--horizon', type=int, required=True)
    p.add_argument('--variant', choices=('internal', 'control'), default='internal')
    p.add_argument('--stage', choices=('train', 'evaluate'), default='train')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--minimum-epochs', type=int, default=40)
    p.add_argument('--checkpoint-min-epoch', type=int, default=8)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--backbone-lr', type=float, default=1e-4)
    p.add_argument('--plugin-lr', type=float, default=0.003)
    p.add_argument('--nll-weight', type=float, default=0.03)
    p.add_argument('--mae-weight', type=float, default=0.2)
    p.add_argument('--utility-weight', type=float, default=0.01)
    p.add_argument('--regularity-weight', type=float, default=0.005)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--device', default='')
    p.add_argument('--cuda-memory-fraction', type=float, default=0.8)
    p.add_argument('--inputs-root', default=str(ROOT.parent / 'v1_frozen_inputs_20261006'))
    p.add_argument('--bert', default=str(ROOT / 'models/bert-base-uncased'))
    p.add_argument('--output', default=str(HERE / 'outputs'))
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if min(args.horizon, args.epochs, args.minimum_epochs, args.checkpoint_min_epoch, args.batch_size, args.warmup) < 1 or args.patience < 2:
        raise SystemExit('invalid positive training parameter or patience < 2')
    if not 0 < args.cuda_memory_fraction <= 1:
        raise SystemExit('cuda-memory-fraction must be in (0,1]')
    try:
        run(args)
    except TrainingBudgetReached as error:
        print(str(error), file=sys.stderr, flush=True)
        raise SystemExit(42)
