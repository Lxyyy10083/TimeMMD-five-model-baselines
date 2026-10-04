"""Small fixed five-model pilot; all writes and subprocesses stay under LXY."""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'outputs_pilot'
MODELS=['SpecTF','CFA','TaTS','MM-TSFlib','Aurora']
TASKS=[('Agriculture',12),('Economy',10)]


def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def collect_results():
    rows=[]
    for model in MODELS:
        for domain,horizon in TASKS:
            values={}
            for variant in ['control','internal']:
                p=OUT/variant/model/domain/str(horizon)/'2026/EVALUATED.json'
                if p.exists():values[variant]=json.loads(p.read_text())['test']
            if len(values)!=2:continue
            rows.append(dict(model=model,domain=domain,horizon=horizon,**values,
                change={k:100*(values['internal'][k]/max(values['control'][k],1e-12)-1)
                        for k in ['mse','mae']}))
    save('RESULTS.json',rows)
    return rows


def main():
    assert str(ROOT).startswith('/xiliang/LXY/')
    assert sys.prefix=='/xiliang/LXY/envs/lxy', 'Use only lxy environment'
    OUT.mkdir(exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1',
        PYTHONNOUSERSITE='1',HF_HOME='/xiliang/LXY/.cache/huggingface',
        XDG_CACHE_HOME='/xiliang/LXY/.cache',MPLCONFIGDIR='/xiliang/LXY/.cache/matplotlib',
        TORCH_HOME='/xiliang/LXY/.cache/torch',TMPDIR='/xiliang/LXY/tmp',
        CUDA_CACHE_PATH='/xiliang/LXY/.cache/cuda',
        TORCH_EXTENSIONS_DIR='/xiliang/LXY/.cache/torch_extensions',
        TORCHINDUCTOR_CACHE_DIR='/xiliang/LXY/.cache/torchinductor',
        TRITON_CACHE_DIR='/xiliang/LXY/.cache/triton')
    plan=dict(models=MODELS,tasks=TASKS,train_jobs=20,test_jobs=20,seed=2026,
        variants=['control','internal'],epochs=60,minimum_epochs=10,patience=10,batch_size=32,
        backbone_lr=.0001,plugin_lr=.0003,nll_weight=.03,energy_weight=.02,mae_weight=.5,
        warmstart=False,selection='fixed parameters, validation checkpoint only',
        reason='reduced scope requested; paired fresh training, no AutoDL checkpoint dependency')
    save('PLAN.json',plan)
    # Design is fixed before test evaluation; no hyperparameter sweep in this pilot.
    save('DESIGN_LOCKED.json',plan)
    for model in MODELS:
        for domain,horizon in TASKS:
            for stage in ['train','evaluate']:
                for variant in ['control','internal']:
                    dest=OUT/variant/model/domain/str(horizon)/'2026'
                    marker=dest/('COMPLETED.json' if stage=='train' else 'EVALUATED.json')
                    if marker.exists():continue
                    item=dict(stage=stage,model=model,domain=domain,horizon=horizon,variant=variant)
                    save('PROGRESS.json',item)
                    log=OUT/'logs'/f'{stage}_{model}_{domain}_{horizon}_{variant}.log'
                    log.parent.mkdir(exist_ok=True)
                    cmd=[sys.executable,'-u',str(HERE/'train.py'),'--stage',stage,'--model',model,
                        '--domain',domain,'--horizon',str(horizon),'--variant',variant,
                        '--output',str(OUT),'--epochs','60','--minimum-epochs','10','--patience','10',
                        '--batch-size','32','--backbone-lr','.0001','--plugin-lr','.0003',
                        '--nll-weight','.03','--energy-weight','.02','--warmup','10']
                    print('START',item,flush=True)
                    with log.open('a') as stream:
                        code=subprocess.run(cmd,cwd=ROOT,env=os.environ,stdout=stream,stderr=subprocess.STDOUT).returncode
                    if code:
                        save('FAILED.json',dict(**item,log=str(log),code=code))
                        raise RuntimeError(f'actual training failed: {log}')
                    print('DONE',item,flush=True)
            collect_results()
        subprocess.run([sys.executable,str(HERE/'report_pilot.py')],cwd=ROOT,env=os.environ,check=True)
    subprocess.run([sys.executable,str(HERE/'report_pilot.py')],cwd=ROOT,env=os.environ,check=True)
    save('FULL_COMPLETED.json',dict(train_jobs=20,test_jobs=20,tasks=10))
    print('PILOT_COMPLETED',flush=True)


if __name__=='__main__':main()
