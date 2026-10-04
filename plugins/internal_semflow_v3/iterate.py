"""Validation-only selection; retrained controls account for extra training budget."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs_20261004'
OLD = ROOT / 'plugins/internal_semflow/outputs_full_20261003'
AURORA = ROOT / 'plugins/internal_semflow/outputs_full_aurora_v2_20261004'
MODELS = ['SpecTF', 'CFA', 'TaTS', 'MM-TSFlib', 'Aurora']
CONFIGS = {
    'continued_control': dict(variant='control', plugin_lr=.0003, nll_weight=0., energy_weight=0.),
    'balanced': dict(variant='internal', plugin_lr=.0003, nll_weight=.03, energy_weight=.02),
    'strong': dict(variant='internal', plugin_lr=.001, nll_weight=.1, energy_weight=.05),
}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def taskpath(trial, model, domain, horizon):
    return OUT/trial/CONFIGS[trial]['variant']/model/domain/str(horizon)/'2026'


def run(trial, model, domain, horizon, stage):
    cfg = CONFIGS[trial]
    base = AURORA if model == 'Aurora' else OLD
    checkpoint = base/'control'/model/domain/str(horizon)/'2026/checkpoint.pt'
    out = OUT/trial
    out.mkdir(exist_ok=True)
    marker = taskpath(trial, model, domain, horizon)/('COMPLETED.json' if stage=='train' else 'EVALUATED.json')
    if marker.exists():
        return
    cmd = [sys.executable, '-u', str(HERE/'train.py'), '--stage', stage,
           '--model', model, '--domain', domain, '--horizon', str(horizon),
           '--output', str(out), '--control-checkpoint', str(checkpoint),
           '--epochs', '300', '--minimum-epochs', '30', '--patience', '25',
           '--backbone-lr', '.00002', '--warmup', '20']
    for key, value in cfg.items():
        cmd.extend(['--'+key.replace('_','-'), str(value)])
    log = OUT/'logs'/f'{stage}_{trial}_{model}_{domain}_{horizon}.log'
    log.parent.mkdir(exist_ok=True)
    print(stage, trial, model, domain, horizon, flush=True)
    save(OUT/'PROGRESS.json', dict(stage=stage,trial=trial,model=model,domain=domain,horizon=horizon))
    with log.open('a', encoding='utf-8') as stream:
        code = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
    if code:
        save(OUT/'FAILED.json',dict(stage=stage,trial=trial,model=model,domain=domain,horizon=horizon,log=str(log)))
        raise RuntimeError(f'failed: {log}')


def main():
    OUT.mkdir(exist_ok=True)
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg')
    save(OUT/'PLAN.json',dict(models=MODELS,configs=CONFIGS,seed=2026,train_jobs=540,
         selection='mean validation MSE/continued_control MSE and MAE/continued_control MAE',
         test_policy='all selection locked before evaluating any new tests',
         controls='same initialized backbone, same additional epochs/optimizer; no source changes to V2'))
    # Preserve the active Aurora job and avoid GPU competition.
    while not (AURORA/'SUITE_evaluate_COMPLETED.json').exists():
        if (AURORA/'RESUME_FAILED.json').exists():
            raise RuntimeError('Aurora V2 failed; inspect before proceeding')
        save(OUT/'PROGRESS.json', dict(stage='waiting_for_aurora_v2'))
        time.sleep(30)
    tasks=[]
    for path in sorted((ROOT/'benchmark/readgpt_data').glob('*/manifest.json')):
        meta=json.loads(path.read_text())
        tasks.extend((meta['domain'], h) for h in meta['horizons'])
    if len(tasks)!=36:
        raise ValueError('expected 36 domain/horizon tasks')
    selected=[]
    for model in MODELS:
        for domain,horizon in tasks:
            for trial in CONFIGS:
                run(trial,model,domain,horizon,'train')
            control=json.loads((taskpath('continued_control',model,domain,horizon)/'COMPLETED.json').read_text())
            def score(trial):
                v=json.loads((taskpath(trial,model,domain,horizon)/'COMPLETED.json').read_text())['validation']
                c=control['validation']
                return .5*(v['mse']/max(c['mse'],1e-8)+v['mae']/max(c['mae'],1e-8))
            trial=min(['balanced','strong'], key=score)
            selected.append(dict(model=model,domain=domain,horizon=horizon,trial=trial,
                                 validation_score=score(trial)))
            save(OUT/'VALIDATION_SELECTION.json',selected)
    for trial in CONFIGS:
        save(OUT/trial/'DESIGN_LOCKED.json',dict(selection=selected,configs=CONFIGS,test_used=False))
    # Evaluate exactly the chosen plugin and the continued control, never choose by test.
    rows=[]
    for item in selected:
        model,domain,horizon,trial=(item[k] for k in ['model','domain','horizon','trial'])
        for t in ['continued_control',trial]:
            run(t,model,domain,horizon,'evaluate')
        c=json.loads((taskpath('continued_control',model,domain,horizon)/'EVALUATED.json').read_text())['test']
        v=json.loads((taskpath(trial,model,domain,horizon)/'EVALUATED.json').read_text())['test']
        oldroot=AURORA if model=='Aurora' else OLD
        old=json.loads((oldroot/'control'/model/domain/str(horizon)/'2026/EVALUATED.json').read_text())['test']
        rows.append(dict(**item,continued_control=c,internal=v,previous_control=old,
                    changes_vs_continued={k:100*(v[k]/max(c[k],1e-12)-1) for k in ['mse','mae']},
                    changes_vs_previous={k:100*(v[k]/max(old[k],1e-12)-1) for k in ['mse','mae']}))
        save(OUT/'RESULTS.json',rows)
    subprocess.run([sys.executable,str(HERE/'report.py')],cwd=ROOT,check=True)
    save(OUT/'FULL_COMPLETED.json',dict(tasks=180,train_jobs=540,evaluation_jobs=360))
    print('V3_ALL_FIVE_COMPLETED',flush=True)


if __name__=='__main__':
    main()
