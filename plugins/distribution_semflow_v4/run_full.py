"""Fixed full paired experiment. One worker per explicitly selected idle GPU."""
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'outputs_full_fitted'
MODELS=['SpecTF','CFA','TaTS','MM-TSFlib','Aurora']
DOMAINS=['Agriculture','Climate','Economy','Energy','Environment','Health','Security','SocialGood','Traffic']
LOCK=threading.Lock()
PREVIOUS=HERE.parent/'internal_semflow_full_lab/outputs_full'

def save(name,value):
    path=OUT/name; temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8'); temp.replace(path)

def tasks():
    return [(m,d,h) for d in DOMAINS
            for h in json.loads((ROOT/'benchmark/readgpt_data'/d/'manifest.json').read_text())['horizons'] for m in MODELS]

def collect():
    rows=[]
    previous={(r['model'],r['domain'],r['horizon']):r for r in json.loads((PREVIOUS/'RESULTS.json').read_text())}
    historical={(r['model'],r['domain'],r['horizon']):r['original'] for r in json.loads((HERE/'historical_comparison_180.json').read_text())}
    for m,d,h in tasks():
        values={}; convergence={}; arrays={}
        for v in ['control','internal']:
            dest=OUT/v/m/d/str(h)/'2026'
            if (dest/'EVALUATED.json').exists():
                values[v]=json.loads((dest/'EVALUATED.json').read_text())['test']
                convergence[v]=json.loads((dest/'COMPLETED.json').read_text())
                import numpy as np
                with np.load(dest/'test_predictions.npz') as a:
                    arrays[v]=(a['origins'].copy(),a['target'].copy())
        if len(values)!=2:continue
        import numpy as np
        if not np.array_equal(arrays['control'][0],arrays['internal'][0]) or not np.array_equal(arrays['control'][1],arrays['internal'][1]):
            raise ValueError(f'paired target/origin mismatch {m}/{d}/{h}')
        rows.append(dict(model=m,domain=d,horizon=h,**values,
            previous_internal=previous[(m,d,h)]['internal'],
            previous_native=previous[(m,d,h)]['control'],
            decrease_vs_previous_internal_pct={k:100*(1-values['internal'][k]/max(previous[(m,d,h)]['internal'][k],1e-12)) for k in ['mse','mae']},
            historical_original=historical[(m,d,h)],
            decrease_vs_historical_pct={k:100*(1-values['internal'][k]/max(historical[(m,d,h)][k],1e-12)) for k in ['mse','mae']},
            decrease_pct={k:100*(1-values['internal'][k]/max(values['control'][k],1e-12)) for k in ['mse','mae']},
            convergence={v:{k:c[k] for k in ['best_epoch','epochs_run','converged','seconds']} for v,c in convergence.items()}))
    save('RESULTS.json',rows)
    fields=['model','domain','horizon','control_mse','internal_mse','mse_decrease_pct','control_mae','internal_mae','mae_decrease_pct','control_epochs','internal_epochs']
    with (OUT/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow(dict(model=r['model'],domain=r['domain'],horizon=r['horizon'],
            control_mse=r['control']['mse'],internal_mse=r['internal']['mse'],mse_decrease_pct=r['decrease_pct']['mse'],
            control_mae=r['control']['mae'],internal_mae=r['internal']['mae'],mae_decrease_pct=r['decrease_pct']['mae'],
            control_epochs=r['convergence']['control']['epochs_run'],internal_epochs=r['convergence']['internal']['epochs_run']))
    return rows

def report(rows):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines=['# 九领域五模型四步长：收敛训练对比','',f'已完成配对：{len(rows)}/180。正百分比代表误差下降；未完成项保留为空。',
        '主要比较：从相同收敛权重开始、相同追加训练规则的无插件原版 control。historical_original 是旧实验保存的原版数值，训练协议不同，只作参考。',
        '指标为训练段统计量标准化尺度；使用验证集最优权重。单种子2026，验证平台期不等同于全局最优证明。','',
        '|模型|领域|步长|原版MSE|插件MSE|下降%|原版MAE|插件MAE|下降%|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        c,v,d=r['control'],r['internal'],r['decrease_pct']
        lines.append(f"|{r['model']}|{r['domain']}|{r['horizon']}|{c['mse']:.6f}|{v['mse']:.6f}|{d['mse']:.2f}|{c['mae']:.6f}|{v['mae']:.6f}|{d['mae']:.2f}|")
    (OUT/'收敛实验结果.md').write_text('\n'.join(lines),encoding='utf-8')
    for metric in ['mse','mae']:
        fig,axes=plt.subplots(1,5,figsize=(22,6),sharey=True)
        for ax,m in zip(axes,MODELS):
            grid=np.full((9,4),np.nan)
            for r in rows:
                if r['model']==m:
                    di=DOMAINS.index(r['domain'])
                    horizons=json.loads((ROOT/'benchmark/readgpt_data'/r['domain']/'manifest.json').read_text())['horizons']
                    grid[di,horizons.index(r['horizon'])]=r['decrease_pct'][metric]
            im=ax.imshow(grid,cmap='RdYlGn',vmin=-30,vmax=30,aspect='auto')
            for i in range(9):
                for j in range(4):
                    if np.isfinite(grid[i,j]):ax.text(j,i,f'{grid[i,j]:.1f}',ha='center',va='center',fontsize=8)
            ax.set_title(m);ax.set_xticks(range(4),['H1','H2','H3','H4']);ax.set_yticks(range(9),DOMAINS)
        fig.colorbar(im,ax=list(axes),label='Error decrease (%)',shrink=.8)
        fig.suptitle(f'{metric.upper()} reduction vs matched native control ({len(rows)}/180 complete)')
        fig.savefig(OUT/f'{metric.upper()}_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)

def worker(gpu,assigned):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
    failures=[]
    for model,domain,horizon in assigned:
        for variant in ['control','internal']:
            for stage in ['train','evaluate']:
                dest=OUT/variant/model/domain/str(horizon)/'2026'
                marker=dest/('COMPLETED.json' if stage=='train' else 'EVALUATED.json')
                if marker.exists():continue
                item=dict(gpu=gpu,stage=stage,model=model,domain=domain,horizon=horizon,variant=variant)
                with LOCK:save(f'PROGRESS_GPU{gpu}.json',item)
                log=OUT/'logs'/f'{stage}_{model}_{domain}_{horizon}_{variant}.log'
                log.parent.mkdir(exist_ok=True)
                cmd=[sys.executable,'-u',str(HERE/'train.py'),'--stage',stage,'--model',model,
                    '--domain',domain,'--horizon',str(horizon),'--variant',variant,'--output',str(OUT),
                    '--epochs','2000','--minimum-epochs','45','--patience','25','--batch-size','32',
                    '--backbone-lr','.00002','--plugin-lr','.0003','--nll-weight','.01','--energy-weight','.005','--warmup','20',
                    '--control-checkpoint',str(PREVIOUS/'control'/model/domain/str(horizon)/'2026/checkpoint.pt')]
                print('START',item,flush=True)
                with log.open('a') as stream:
                    code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
                if code:
                    failures.append(dict(**item,code=code,log=str(log)))
                    with LOCK:save(f'FAILURES_GPU{gpu}.json',failures)
                    print('FAILED',item,flush=True)
                    break
                print('DONE',item,flush=True)
            else:continue
            # Preserve failure and proceed to other pairs; no manufactured metric.
        with LOCK:
            rows=collect();report(rows);save('STATUS.json',dict(completed_pairs=len(rows),total_pairs=180))
    return failures

def main():
    assert str(ROOT).startswith('/xiliang/LXY/')
    assert sys.prefix=='/xiliang/LXY/envs/lxy'
    OUT.mkdir(exist_ok=True)
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg',
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HF_HOME='/xiliang/LXY/.cache/huggingface',
        XDG_CACHE_HOME='/xiliang/LXY/.cache',MPLCONFIGDIR='/xiliang/LXY/.cache/matplotlib',TORCH_HOME='/xiliang/LXY/.cache/torch',
        TMPDIR='/xiliang/LXY/tmp',CUDA_CACHE_PATH='/xiliang/LXY/.cache/cuda',TORCH_EXTENSIONS_DIR='/xiliang/LXY/.cache/torch_extensions',
        TORCHINDUCTOR_CACHE_DIR='/xiliang/LXY/.cache/torchinductor',TRITON_CACHE_DIR='/xiliang/LXY/.cache/triton',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    plan=dict(models=MODELS,tasks=tasks(),train_jobs=360,test_jobs=360,seed=2026,
        variants=['control','internal'],minimum_epochs=45,patience=25,safety_epoch_limit=2000,
        convergence='validation composite improvement >1e-5 resets patience; cap is failure, not convergence',
        score='.5*MSE/initial_validation_MSE + .5*MAE/initial_validation_MAE',
        batch_size=32,backbone_lr=.00002,plugin_lr=.0003,nll_weight=.01,energy_weight=.005,mae_weight=.5,
        regularity_weight=.02,utility_weight=.002,feature_weight=.005,density_ramp=20,
        distribution='full trajectory two-expert affine-flow mixture, one probability per trajectory',
        prediction='Bayes action from weighted flow samples, squared + .5 smoothed absolute risk',
        samples_per_expert=16,decision_smoothing=.02,decision_bisections=24,
        plugin_first_eligible_epoch=20,epoch_zero='native reference only; not an eligible fitted plugin',
        selection='fixed before any test; validation checkpoint only; no test-driven tuning',
        warmstart=True,warmstart_reference=str(PREVIOUS),gpus=[1,2],resume_interval=10,
        original='native architectures without new plugin, trained with same aligned protocol; historical original is separate reference')
    if (OUT/'PLAN.json').exists() and json.loads((OUT/'PLAN.json').read_text())!=plan:raise ValueError('fixed plan changed')
    if not (PREVIOUS/'FULL_COMPLETED.json').exists():raise ValueError('converged references required')
    save('PLAN.json',plan);save('DESIGN_LOCKED.json',plan)
    # Sequential preprocessing prevents the two workers racing on shared cache files.
    sys.path.insert(0,str(HERE.parent));os.environ['CUDA_VISIBLE_DEVICES']='1'
    from semflow.data import text_cache
    for d in DOMAINS:
        directory=ROOT/'benchmark/readgpt_data'/d
        manifest=json.loads((directory/'manifest.json').read_text())
        actual=hashlib.sha256((directory/f'{d}.csv').read_bytes()).hexdigest()
        if actual!=manifest['sha256']:raise ValueError(f'canonical CSV hash mismatch: {d}')
        text_cache(d,ROOT/'models/bert-base-uncased')
        print('DATA_READY',d,flush=True)
    jobs=tasks();assert len(jobs)==180
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,gpu,jobs[i::2]) for i,gpu in enumerate([1,2])]
        failures=sum([f.result() for f in futures],[])
    rows=collect();report(rows);save('FAILURES.json',failures)
    if len(rows)==180 and not failures:
        save('FULL_COMPLETED.json',dict(pairs=180,train_jobs=360,test_jobs=360,all_validation_converged=True))
    else:save('INCOMPLETE.json',dict(pairs=len(rows),failures=failures))
    print('FULL_FINISHED',len(rows),'failures',len(failures),flush=True)

if __name__=='__main__':main()
