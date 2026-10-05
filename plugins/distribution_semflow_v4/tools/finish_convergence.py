"""Continue capped jobs without changing sources/configurations of running jobs."""
from pathlib import Path
import concurrent.futures
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
ORIGINAL=HERE/'outputs_full_fitted'
FINAL=HERE/'outputs_final_converged'
DRIVER_PID=427963

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def original_driver_alive():
    p=Path('/proc')/str(DRIVER_PID)/'cmdline'
    try:return str(HERE/'run_full.py').encode() in p.read_bytes()
    except FileNotFoundError:return False

def available_gpu():
    while True:
        text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        for line in text.splitlines():
            index,memory=map(int,line.split(','))
            if index in [1,2] and memory<100:return index
        time.sleep(60)

def extend(source):
    original=source
    while True:
        config=json.loads((source/'run_config.json').read_text())
        budget=int(config['arguments']['epochs'])*2
        if budget>16000:raise RuntimeError(f'additional training still required: {source}')
        assert (source/'resume.pt').exists(),f'cannot resume without optimizer/RNG state: {source}'
        relative=source.relative_to(source.parents[4])
        output=HERE/f'outputs_extended_{budget}'
        destination=output/relative
        args=dict(config['arguments'],output=str(output),epochs=budget)
        destination.mkdir(parents=True,exist_ok=True)
        if not (destination/'run_config.json').exists():
            for name in ['checkpoint.pt','resume.pt','training.json']:
                # Copy: subsequent training must never overwrite original checkpoint data.
                shutil.copyfile(source/name,destination/name)
            adapted={**config,'arguments':args}
            write(destination/'run_config.json',adapted)
            write(destination/'CONTINUATION.json',dict(source=str(source),original_source=str(original),
                from_epoch=json.loads((source/'NEEDS_MORE_TRAINING.json').read_text())['epoch'],
                old_limit=config['arguments']['epochs'],new_limit=budget,
                change='epoch safety budget only; source, data, optimizer, scheduler, RNG and stopping tolerance unchanged'))
        write(output/'DESIGN_LOCKED.json',dict(original_plan=json.loads((ORIGINAL/'PLAN.json').read_text()),
            extension_budget=budget,selection='validation only; continuation of capped jobs'))
        gpu=available_gpu();env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
        for stage in ['train','evaluate']:
            marker=destination/('COMPLETED.json' if stage=='train' else 'EVALUATED.json')
            if marker.exists():continue
            command=[sys.executable,'-u',str(HERE/'train.py'),'--stage',stage]
            for key,value in args.items():command.extend(['--'+key.replace('_','-'),str(value)])
            log=destination/f'continuation_{stage}.log'
            print('CONTINUATION_START',stage,str(destination),'gpu',gpu,flush=True)
            with log.open('a') as f:
                code=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
            if code:
                if stage=='train' and (destination/'NEEDS_MORE_TRAINING.json').exists():
                    source=destination;break
                raise RuntimeError(f'continuation failed: {log}')
        else:
            return destination

def main():
    assert str(ROOT).startswith('/xiliang/LXY/')
    assert sys.prefix=='/xiliang/LXY/envs/lxy'
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TMPDIR='/xiliang/LXY/tmp',
        XDG_CACHE_HOME='/xiliang/LXY/.cache',HF_HOME='/xiliang/LXY/.cache/huggingface',
        MPLCONFIGDIR='/xiliang/LXY/.cache/matplotlib',TORCH_HOME='/xiliang/LXY/.cache/torch',
        CUDA_CACHE_PATH='/xiliang/LXY/.cache/cuda',TORCH_EXTENSIONS_DIR='/xiliang/LXY/.cache/torch_extensions',
        TORCHINDUCTOR_CACHE_DIR='/xiliang/LXY/.cache/torchinductor',TRITON_CACHE_DIR='/xiliang/LXY/.cache/triton',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg')
    FINAL.mkdir(exist_ok=True)
    write(FINAL/'CONTINUATION_PLAN.json',dict(trigger='recorded safety-cap failures',
        original_output=str(ORIGINAL),budgets=[4000,8000,16000],
        stopping='unchanged: at least 45 epochs, 25 stale validation epochs, improvement threshold 1e-5',
        device_policy='wait for original queue to finish, then use an idle GPU 1 or 2; never GPU0'))
    while original_driver_alive():
        p=ORIGINAL/'STATUS.json'
        if p.exists():write(FINAL/'STATUS.json',{**json.loads(p.read_text()),'phase':'original full queue; continuation scheduled'})
        time.sleep(60)
    assert (ORIGINAL/'INCOMPLETE.json').exists() or (ORIGINAL/'FULL_COMPLETED.json').exists(), 'original queue stopped unexpectedly'
    resolved={};failures=[]
    # Serial continuation prevents claiming a GPU between subprocesses used by another worker.
    for source in sorted(ORIGINAL.glob('*/*/*/*/2026')):
        if (source/'EVALUATED.json').exists():resolved[str(source.relative_to(ORIGINAL))]=source;continue
        try:
            if not (source/'NEEDS_MORE_TRAINING.json').exists():raise RuntimeError(f'non-cap failure needs inspection: {source}')
            result=extend(source);resolved[str(source.relative_to(ORIGINAL))]=result
            write(FINAL/'RESOLVED_SOURCES.json',{k:str(v) for k,v in resolved.items()})
        except Exception as e:failures.append(dict(source=str(source),error=str(e)))
    if failures:
        write(FINAL/'INCOMPLETE.json',dict(resolved_runs=len(resolved),failures=failures));return
    assert len(resolved)==360, f'expected 360 converged evaluated runs, got {len(resolved)}'
    for relative,source in resolved.items():
        completed=json.loads((source/'COMPLETED.json').read_text());assert completed['converged']
        destination=FINAL/relative;destination.mkdir(parents=True,exist_ok=True)
        for p in source.iterdir():
            if not p.is_file() or p.suffix not in ['.json','.csv','.md','.png','.npz','.log']:continue
            q=destination/p.name
            if not q.exists():os.link(p,q)
        write(destination/'RESULT_SOURCE.json',dict(source=str(source),note='read-only result assembly; original training identity preserved'))
    write(FINAL/'PLAN.json',{**json.loads((ORIGINAL/'PLAN.json').read_text()),
        'continuation_plan':json.loads((FINAL/'CONTINUATION_PLAN.json').read_text())})
    write(FINAL/'RESOLVED_SOURCES.json',{k:str(v) for k,v in resolved.items()})
    spec=importlib.util.spec_from_file_location('distribution_result_driver',HERE/'run_full.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    driver.OUT=FINAL
    rows=driver.collect();driver.report(rows);assert len(rows)==180
    write(FINAL/'FAILURES.json',[])
    write(FINAL/'RECOVERED_INITIAL_FAILURES.json',[json.loads(p.read_text()) for p in ORIGINAL.glob('FAILURES_GPU*.json')])
    write(FINAL/'FULL_COMPLETED.json',dict(pairs=180,train_jobs=360,test_jobs=360,
        all_validation_converged=True,source='original evaluated runs plus recorded resumable extensions'))
    write(FINAL/'STATUS.json',dict(completed_pairs=180,total_pairs=180,phase='complete'))
    print('ALL_CONVERGED_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        write(FINAL/'INCOMPLETE.json',dict(error=str(e)));raise
