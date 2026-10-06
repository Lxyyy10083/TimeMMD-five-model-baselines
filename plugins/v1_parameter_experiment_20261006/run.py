"""V1-P1: validation-selected training hyperparameters and locked gate sensitivity."""
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import csv
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'outputs'
REFERENCE = ROOT / 'reference_v1'
sys.path.insert(0, str(HERE))
import experiment as exp

exp.OUTPUT = OUT
torch.set_num_threads(4)
MODELS = list(exp.MODELS)
SEEDS = [2026, 2027, 2028]

def save(name, value):
    exp.write_json(OUT/name, value)

def csv_save(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def configs():
    base = {**exp.variants()['no_variance_shrink'], 'utility_margin':0.05, 'lr':0.001}
    changes = {
        'v1_control':{},
        'nll_010':{'nll_weight':0.1}, 'nll_030':{'nll_weight':0.3},
        'utility_005':{'utility_weight':0.05}, 'utility_010':{'utility_weight':0.1},
        'mae_050':{'mae_weight':0.5}, 'mae_100':{'mae_weight':1.0},
        'regularity_000':{'regularity_weight':0.0}, 'regularity_002':{'regularity_weight':0.02},
        'amplitude_05':{'correction_limit':0.5}, 'amplitude_2':{'correction_limit':2.0},
        'teacher_margin_000':{'utility_margin':0.0}, 'teacher_margin_015':{'utility_margin':0.15},
        'lr_0003':{'lr':0.0003}, 'lr_003':{'lr':0.003},
        'model_balanced':{'balanced':True},
    }
    return {name:{**base,**change} for name,change in changes.items()}

def args(cfg, formal=False):
    return SimpleNamespace(bert='/xiliang/LXY/baseline_v3_lab_20261004/models/bert-base-uncased',
                           screen_epochs=24, final_epochs=2000, minimum_epochs=32 if formal else 8,
                           patience=20 if formal else 8, require_convergence=formal,
                           batch_size=256, lr=cfg['lr'])

def policies():
    original = [0.25,0.5,0.75,1.0]
    dense = sorted(set(original+[i/10 for i in range(1,11)]))
    return [dict(name=f'tau{tau:g}_{grid}', threshold=tau, alphas=values)
            for tau in [0.0,0.0025,0.005,0.01,0.02]
            for grid,values in [('original',original),('dense',dense)]]

def check_inputs():
    marker = json.loads((ROOT/'INPUT_DOWNLOAD_COMPLETE.json').read_text())
    for record in marker['records']:
        p=ROOT/record['path']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Frozen input changed: '+record['path'])
    snapshot=json.loads((HERE/'SOURCE_MANIFEST.json').read_text())
    for name,sha in snapshot['files'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Deployed source changed: '+name)
    save('INPUT_AUDIT.json',dict(files=len(marker['records']),all_sha256_equal=True,
                                  five_forecasters='unchanged original frozen forecasts'))

def ranking(names, seeds):
    table=[]
    for name in names:
        results=[json.loads((OUT/'screen'/name/str(seed)/domain/str(h)/'fit_result.json').read_text())
                 for domain,h in exp.PILOT.items() for seed in seeds]
        entries=[r for result in results for r in result['holdout'].values()]
        per_model={m:{k:float(np.mean([r['holdout'][m][k] for r in results]))
                       for k in ['mse_ratio','mae_ratio']} for m in MODELS}
        table.append(dict(variant=name,score=float(np.mean([r['score'] for r in entries])),
                          mse_ratio=float(np.mean([r['mse_ratio'] for r in entries])),
                          mae_ratio=float(np.mean([r['mae_ratio'] for r in entries])),
                          enabled=sum(r['alpha']>0 for r in entries),fits=len(results),models=per_model))
    # At numerical ties prefer unchanged V1. No test metrics enter this ranking.
    return sorted(table,key=lambda r:(round(r['score'],10),r['variant']!='v1_control',r['variant']))

def fit(stage,name,domain,horizon,seed,done,total):
    cfg=configs()[name]
    save('STATUS.json',dict(stage=stage,variant=name,domain=domain,horizon=horizon,seed=seed,
                            completed_fits=done,total_fits=total))
    embedding,mask=exp.text_cache(domain,Path(args(cfg).bert))
    exp.OUTPUT = OUT/'convergence_v3' if stage=='final' else OUT
    try:
        return exp.train_case(stage,name,cfg,domain,horizon,seed,args(cfg,stage=='final'),embedding,mask)
    finally:
        exp.OUTPUT = OUT

def select_policy(base,raw,target,policy):
    baseline=exp.metric(base,target)
    best=dict(alpha=0.0,mse_ratio=1.0,mae_ratio=1.0,score=1.0)
    for alpha in policy['alphas']:
        val=exp.metric(base+alpha*(raw-base),target)
        mr=val['mse']/max(baseline['mse'],1e-12)
        ar=val['mae']/max(baseline['mae'],1e-12)
        score=0.5*(mr+ar)
        if mr<1-policy['threshold'] and ar<1-policy['threshold'] and score<best['score']:
            best=dict(alpha=alpha,mse_ratio=mr,mae_ratio=ar,score=score)
    return best

def evaluate(winner):
    records=[]
    reference_metrics={}
    matched_metrics={}
    selection_log=[]
    for meta in exp.manifests():
        d=meta['domain']
        embedding,mask=exp.text_cache(d,Path(args(configs()[winner]).bert))
        for h in meta['horizons']:
            hold=exp.load_split(d,h,'holdout',embedding,mask)
            # Only after all training and global winner/policies have been locked.
            test=exp.load_split(d,h,'test',embedding,mask)
            for seed in SEEDS:
                for label,folder in [
                    ('reference_v1',REFERENCE/str(seed)/d/str(h)),
                    ('matched_v1',OUT/'convergence_v3/final/v1_control'/str(seed)/d/str(h)),
                    ('V1-P1',OUT/'convergence_v3/final'/winner/str(seed)/d/str(h))]:
                    checkpoint=torch.load(folder/'module.pt',map_location=exp.DEVICE,weights_only=False)
                    result=json.loads((folder/'fit_result.json').read_text())
                    module=exp.SemanticGraphFlow(exp.SemanticGraphFlowConfig(**checkpoint['config'])).to(exp.DEVICE)
                    module.load_state_dict(checkpoint['state_dict'])
                    dest=OUT/'predictions'/label/str(seed)/d/str(h)
                    dest.mkdir(parents=True,exist_ok=True)
                    for tag,m in enumerate(MODELS):
                        raw_hold,_=exp.prediction(module,hold[m],tag,checkpoint['variant']['text_mode'])
                        raw_test,profile=exp.prediction(module,test[m],tag,checkpoint['variant']['text_mode'])
                        baseline=exp.metric(test[m].base,test[m].target)
                        raw_metric=exp.metric(raw_test,test[m].target)
                        key=(d,h,seed,m)
                        original_alpha=checkpoint['alphas'][m]
                        original_chosen=test[m].base+original_alpha*(raw_test-test[m].base)
                        if label=='reference_v1':
                            reference_metrics[key]=exp.metric(original_chosen,test[m].target)
                            matched_metrics[key]=reference_metrics[key]
                        elif label=='matched_v1':
                            matched_metrics[key]=exp.metric(original_chosen,test[m].target)
                        np.savez_compressed(dest/f'{m}.npz',raw_holdout=raw_hold,raw_test=raw_test,
                                            original_chosen=original_chosen,profile=profile)
                        for policy in policies():
                            selection=select_policy(hold[m].base,raw_hold,hold[m].target,policy)
                            if policy['name']=='tau0_original':
                                # Preserve the exact V1 alpha at its selected checkpoint.
                                selection={**selection,**result['holdout'][m]}
                            elif result['best_epoch']==0:
                                selection=dict(alpha=0.0,mse_ratio=1.0,mae_ratio=1.0,score=1.0)
                            alpha=selection['alpha']
                            chosen=test[m].base+alpha*(raw_test-test[m].base)
                            metric=exp.metric(chosen,test[m].target)
                            row=dict(version=label,training_variant=winner if label=='V1-P1' else 'v1_control',
                                     policy=policy['name'],threshold=policy['threshold'],model=m,domain=d,
                                     horizon=h,seed=seed,alpha=alpha,best_epoch=result['best_epoch'],
                                     epochs_run=result['epochs_run'],enabled=int(alpha>0 and result['best_epoch']>0),
                                     baseline_mse=baseline['mse'],baseline_mae=baseline['mae'],
                                     mse=metric['mse'],mae=metric['mae'],raw_mse=raw_metric['mse'],raw_mae=raw_metric['mae'],
                                     mse_decrease_pct=100*(1-metric['mse']/baseline['mse']),
                                     mae_decrease_pct=100*(1-metric['mae']/baseline['mae']),
                                     v1_mse=reference_metrics[key]['mse'],v1_mae=reference_metrics[key]['mae'],
                                     mse_decrease_vs_v1_pct=100*(1-metric['mse']/reference_metrics[key]['mse']),
                                     mae_decrease_vs_v1_pct=100*(1-metric['mae']/reference_metrics[key]['mae']),
                                     matched_v1_mse=matched_metrics[key]['mse'],matched_v1_mae=matched_metrics[key]['mae'],
                                     mse_decrease_vs_matched_v1_pct=100*(1-metric['mse']/matched_metrics[key]['mse']),
                                     mae_decrease_vs_matched_v1_pct=100*(1-metric['mae']/matched_metrics[key]['mae']),
                                     holdout_mse_ratio=selection['mse_ratio'],holdout_mae_ratio=selection['mae_ratio'],
                                     test_windows=len(raw_test))
                            records.append(row)
                            selection_log.append({k:row[k] for k in ['version','policy','model','domain','horizon','seed','alpha','holdout_mse_ratio','holdout_mae_ratio']})
                    del module
                    gc.collect();torch.cuda.empty_cache()
                save('STATUS.json',dict(stage='evaluation',domain=d,horizon=h,seed=seed,
                                        completed_test_records=len(records),expected_test_records=16200))
    # Fill the converged comparator for historical rows after both controls were evaluated.
    for r in records:
        m=matched_metrics[(r['domain'],r['horizon'],r['seed'],r['model'])]
        for k in ['mse','mae']:
            r['matched_v1_'+k]=m[k]
            r[k+'_decrease_vs_matched_v1_pct']=100*(1-r[k]/m[k])
    if len(records)!=16200:raise ValueError('Incomplete gate sensitivity results')
    csv_save('test_seed_results.csv',records)
    save('HOLDOUT_SELECTIONS.json',selection_log)
    return records

def report(records,winner):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=defaultdict(list)
    for r in records:groups[(r['version'],r['policy'],r['model'],r['domain'],r['horizon'])].append(r)
    means=[]
    for key,rs in groups.items():
        if len(rs)!=3:raise ValueError('Every task requires three seeds')
        r=rs[0]
        result=dict(version=key[0],policy=key[1],model=key[2],domain=key[3],horizon=key[4],
                    enabled_seeds=sum(x['enabled'] for x in rs))
        for k in ['mse','mae']:
            result['baseline_'+k]=r['baseline_'+k]
            result[k]=float(np.mean([x[k] for x in rs]))
            result[k+'_std']=float(np.std([x[k] for x in rs],ddof=1))
            result['v1_'+k]=float(np.mean([x['v1_'+k] for x in rs]))
            result[k+'_decrease_pct']=100*(1-result[k]/result['baseline_'+k])
            result[k+'_decrease_vs_v1_pct']=100*(1-result[k]/result['v1_'+k])
            result['matched_v1_'+k]=float(np.mean([x['matched_v1_'+k] for x in rs]))
            result[k+'_decrease_vs_matched_v1_pct']=100*(1-result[k]/result['matched_v1_'+k])
        means.append(result)
    csv_save('results_all_policies.csv',means)
    primary=[r for r in means if r['version']=='V1-P1' and r['policy']=='tau0_original']
    if len(primary)!=180:raise ValueError('Incomplete primary experiment')
    csv_save('results_180.csv',primary)
    save('RESULTS_180.json',primary)
    summary=[]
    for version in ['reference_v1','matched_v1','V1-P1']:
        for policy in policies():
            for m in MODELS:
                rs=[r for r in means if r['version']==version and r['policy']==policy['name'] and r['model']==m]
                summary.append(dict(version=version,policy=policy['name'],threshold=policy['threshold'],model=m,
                                    tasks=36,enabled=sum(r['enabled_seeds']>0 for r in rs),
                                    both_better=sum(r['mse_decrease_pct']>1e-5 and r['mae_decrease_pct']>1e-5 for r in rs),
                                    any_worse=sum(r['mse_decrease_pct']<-1e-5 or r['mae_decrease_pct']<-1e-5 for r in rs),
                                    mse_decrease_pct=float(np.mean([r['mse_decrease_pct'] for r in rs])),
                                    mae_decrease_pct=float(np.mean([r['mae_decrease_pct'] for r in rs])),
                                    mse_decrease_vs_v1_pct=float(np.mean([r['mse_decrease_vs_v1_pct'] for r in rs])),
                                    mae_decrease_vs_v1_pct=float(np.mean([r['mae_decrease_vs_v1_pct'] for r in rs])),
                                    mse_decrease_vs_matched_v1_pct=float(np.mean([r['mse_decrease_vs_matched_v1_pct'] for r in rs])),
                                    mae_decrease_vs_matched_v1_pct=float(np.mean([r['mae_decrease_vs_matched_v1_pct'] for r in rs]))))
    csv_save('model_policy_summary.csv',summary)
    save('MODEL_SUMMARY.json',[r for r in summary if r['policy']=='tau0_original'])
    convergence=[]
    for p in (OUT/'convergence_v3/final').glob('*/*/*/*/fit_result.json'):
        r=json.loads(p.read_text())
        convergence.append({k:r[k] for k in ['variant','domain','horizon','seed','best_epoch','epochs_run','converged','stopping_reason','final_lr','raw_stale','selected_stale']})
    save('CONVERGENCE_BUDGET.json',dict(fits=len(convergence),all_validation_plateau=all(r['converged'] for r in convergence),records=convergence))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for k,ax in zip(['mse','mae'],axes):
        for version,offset,color in [('reference_v1',-.26,'#8F9DAB'),('matched_v1',0,'#E8A745'),('V1-P1',.26,'#367FB5')]:
            vals=[next(r for r in summary if r['version']==version and r['policy']=='tau0_original' and r['model']==m)[k+'_decrease_pct'] for m in MODELS]
            ax.bar(np.arange(5)+offset,vals,width=.25,label=version,color=color)
        ax.axhline(0,color='black',linewidth=.6);ax.set_xticks(range(5),MODELS,rotation=20)
        ax.set_title(k.upper()+' decrease vs original (%)');ax.legend()
    fig.savefig(OUT/'V1_P1_model_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for k,ax in zip(['mse','mae'],axes):
        for m in MODELS:
            rs=[next(r for r in summary if r['version']=='V1-P1' and r['policy']==f'tau{t:g}_original' and r['model']==m) for t in [0,.0025,.005,.01,.02]]
            ax.plot([r['threshold']*100 for r in rs],[r[k+'_decrease_pct'] for r in rs],marker='o',label=m)
        ax.axhline(0,color='black',linewidth=.6);ax.set_xlabel('Holdout minimum dual gain (%)')
        ax.set_title(k.upper()+' decrease vs original (%)');ax.legend(fontsize=8)
    fig.savefig(OUT/'threshold_sensitivity.png',dpi=180);plt.close(fig)
    lines=[(HERE/'V1参数实验.md').read_text(),'', '## 本次完成结果','',
           f"仅按验证集锁定的训练配置：`{winner}`。主结果采用原V1阈值0与原alpha网格，不根据测试结果改变配置。",'',
           '|模型|原V1 MSE减小%|V1-P1 MSE减小%|原V1 MAE减小%|V1-P1 MAE减小%|',
           '|---|---:|---:|---:|---:|']
    for m in MODELS:
        ref=next(r for r in summary if r['version']=='reference_v1' and r['policy']=='tau0_original' and r['model']==m)
        new=next(r for r in summary if r['version']=='V1-P1' and r['policy']=='tau0_original' and r['model']==m)
        lines.append(f"|{m}|{ref['mse_decrease_pct']:+.4f}|{new['mse_decrease_pct']:+.4f}|{ref['mae_decrease_pct']:+.4f}|{new['mae_decrease_pct']:+.4f}|")
    lines += ['',f'正式阶段共{len(convergence)}次独立拟合，全部满足五个底模原始插件验证评分和选用评分的平台期条件。历史80轮V1仅作历史参考。', '',
              '|模型|收敛V1 MSE减小%|收敛V1 MAE减小%|调参相对收敛V1 MSE再减小%|调参相对收敛V1 MAE再减小%|',
              '|---|---:|---:|---:|---:|']
    for m in MODELS:
        ref=next(r for r in summary if r['version']=='matched_v1' and r['policy']=='tau0_original' and r['model']==m)
        new=next(r for r in summary if r['version']=='V1-P1' and r['policy']=='tau0_original' and r['model']==m)
        lines.append(f"|{m}|{ref['mse_decrease_pct']:+.4f}|{ref['mae_decrease_pct']:+.4f}|{new['mse_decrease_vs_matched_v1_pct']:+.4f}|{new['mae_decrease_vs_matched_v1_pct']:+.4f}|")
    lines += ['',
              '阈值与网格的全部预设结果见model_policy_summary.csv和results_all_policies.csv。它们是完整敏感性结果，不根据测试排名宣布新的全局赢家。', '',
              '![原V1与参数实验主结果](V1_P1_model_comparison.png)', '',
              '![验证门槛敏感性](threshold_sensitivity.png)', '']
    (OUT/'V1参数实验结果.md').write_text('\n'.join(lines),encoding='utf-8')
    return convergence

def main():
    if Path(sys.prefix).resolve()!=Path('/xiliang/LXY/envs/lxy'):raise RuntimeError('Use lxy only')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=os.environ.get('V1_P1_GPU'):
        raise RuntimeError('GPU must match the idle device recorded by the launcher')
    torch.cuda.set_per_process_memory_fraction(.05)
    OUT.mkdir(exist_ok=True)
    check_inputs()
    save('PLAN.json',dict(name='V1-P1',backup_tag='pre_v1_hparam_20261006',
                          configs=configs(),pilot=exp.PILOT,policies=policies(),
                          selection='holdout-only training winner; gates are locked sensitivity, never test selection',
                          seeds=SEEDS,protocol='v3_raw_five_model_plateau',final_safety_epochs=2000,
                          minimum_epochs=32,patience=20,scheduler=dict(factor=.5,patience=5,min_lr=1e-8,eps=1e-12),
                          screening=dict(maximum_epochs=24,minimum_epochs=8,patience=8),
                          comparator='fresh converged V1 control plus frozen historical 80-epoch V1'))
    if not (OUT/'winner.json').exists():
        done=0
        for d,h in exp.PILOT.items():
            for name in configs():
                fit('screen',name,d,h,2026,done,144);done+=1
        rank=ranking(list(configs()),[2026])
        save('screen_ranking.json',rank)
        promoted=['v1_control']+[r['variant'] for r in rank if r['variant']!='v1_control'][:2]
        save('promoted.json',promoted)
        done=0
        for d,h in exp.PILOT.items():
            for name in promoted:
                fit('screen',name,d,h,2027,done,27);done+=1
        confirm=ranking(promoted,[2026,2027])
        save('confirmation_ranking.json',confirm)
        save('winner.json',dict(variant=confirm[0]['variant'],parameters=configs()[confirm[0]['variant']],
                                selected_on='9 pilot domains x 5 models x 2 plugin seeds, holdout only',ranking=confirm))
    winner=json.loads((OUT/'winner.json').read_text())['variant']
    final_variants=list(dict.fromkeys(['v1_control',winner]))
    done=0
    for name in final_variants:
        for meta in exp.manifests():
            for h in meta['horizons']:
                for seed in SEEDS:
                    fit('final',name,meta['domain'],h,seed,done,108*len(final_variants));done+=1
    for name in final_variants:
        fits=list((OUT/'convergence_v3/final'/name).glob('*/*/*/fit_result.json'))
        if len(fits)!=108 or not all(json.loads(p.read_text()).get('converged') for p in fits):
            raise RuntimeError('Require all final fits converged before testing')
    save('TEST_LOCK.json',dict(winner=winner,policies=policies(),locked_before_test=True,
                               primary_policy='tau0_original',timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    records=evaluate(winner)
    convergence=report(records,winner)
    save('FULL_COMPLETED.json',dict(screen_fits=144,confirmation_fits=27,final_fits=len(convergence),
                                    matched_control_fits=108,winner_fits=108,unique_training_variants=final_variants,
                                    reference_v1_fits_reused=108,test_rows=16200,primary_model_tasks=180,
                                    winner=winner,all_final_evaluated=True,all_validation_plateau=all(r['converged'] for r in convergence)))
    save('STATUS.json',dict(stage='complete',winner=winner,final_fits=len(convergence),test_rows=16200))
    print('V1_P1_COMPLETE',winner,flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        save('FAILED.json',dict(error=str(error),type=type(error).__name__))
        raise
