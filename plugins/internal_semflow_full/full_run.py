"""Finish pilot, train all five models, then evaluate a predeclared design."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT/'plugins/internal_semflow'
MODELS = ['TaTS','MM-TSFlib','Aurora','SpecTF','CFA']


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pilot',default=str(PLUGIN/'outputs_v3_cfa'))
    p.add_argument('--output',default=str(PLUGIN/'outputs_full_20261003'))
    args=p.parse_args()
    pilot,out=Path(args.pilot),Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    write(out/'FULL_PLAN.json',dict(models=MODELS,variants=['control','internal'],
        seeds=[2026],domains=9,horizons_per_domain=4,training_jobs=360,
        note='Independent weights per model. Single-seed comparison. Pilot completes first.'))
    print('WAITING_FOR_CFA_PILOT',str(pilot),flush=True)
    while not (pilot/'SUITE_train_COMPLETED.json').exists():
        if (pilot/'FAILED.json').exists():
            write(out/'WAIT_FAILED.json',dict(pilot_failure=json.loads((pilot/'FAILED.json').read_text())))
            print('PILOT_FAILED',flush=True)
            return 1
        time.sleep(30)
    # Fixed architecture and weights, no selection from final test outcomes.
    write(out/'DESIGN_LOCKED.json',dict(models=MODELS,variants=['control','internal'],
        nll_weight=0.1,mae_weight=0.5,energy_weight=0.05,epochs=200,patience=15,
        backbone_lr=0.0001,plugin_lr=0.001,seed=2026,
        rationale='Evaluate the predeclared internal architecture against matched retraining and saved original baselines; no test-driven parameter changes.'))
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg')
    successes,failures=[],[]
    for model in MODELS:
        command=[sys.executable,'-u',str(PLUGIN/'launch.py'),'--suite','full','--models',model,
                 '--variants','control','internal','--stage','train','--output',str(out)]
        print('FULL_MODEL_START',model,flush=True)
        with (out/f'driver_train_{model}.log').open('a',encoding='utf-8') as log:
            code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        (successes if code==0 else failures).append(model)
        write(out/'FULL_PROGRESS.json',dict(train_completed_models=successes,failed_models=failures))
        print('FULL_MODEL_DONE',model,'code',code,flush=True)
    # No tests until the entire training queue has been attempted.
    evaluated=[]
    for model in successes:
        command=[sys.executable,'-u',str(PLUGIN/'launch.py'),'--suite','full','--models',model,
                 '--variants','control','internal','--stage','evaluate','--output',str(out)]
        with (out/f'driver_evaluate_{model}.log').open('a',encoding='utf-8') as log:
            code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if code==0:
            evaluated.append(model)
        else:
            failures.append(model+':evaluation')
        write(out/'FULL_PROGRESS.json',dict(train_completed_models=successes,evaluated_models=evaluated,
                                             failed_models=failures))
    if failures:
        write(out/'FULL_INCOMPLETE.json',dict(evaluated_models=evaluated,failures=failures))
        return 1
    result=subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('report.py')),
                           '--directory',str(out)],cwd=ROOT,env=env)
    if result.returncode:
        return result.returncode
    write(out/'FULL_COMPLETED.json',dict(models=evaluated,tasks=180,
                                        train_jobs=360,evaluation_records=360))
    print('ALL_FIVE_COMPLETED',flush=True)
    return 0


if __name__=='__main__':
    sys.exit(main())

