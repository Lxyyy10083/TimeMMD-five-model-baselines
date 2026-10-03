"""Compare complete five-model tests against both controls; never select winners."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
MODELS=['TaTS','MM-TSFlib','Aurora','SpecTF','CFA']
DOMAINS=['Agriculture','Climate','Economy','Energy','Environment','Health',
         'Security','SocialGood','Traffic']


def metric(pred,target,training_precision=False):
    # Evaluation subtracts float32 tensors before accumulating in float64.
    e=(pred-target).astype(np.float64) if training_precision else pred.astype(np.float64)-target.astype(np.float64)
    return dict(mse=float(np.mean(e**2)),mae=float(np.mean(np.abs(e))))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    p.add_argument('--aurora-directory',type=Path)
    args=p.parse_args()
    out=args.directory
    rows=[]
    models=args.models
    for model in models:
        for domain in DOMAINS:
            meta=json.loads((ROOT/'benchmark/readgpt_data'/domain/'manifest.json').read_text())
            for horizon in meta['horizons']:
                original_path=ROOT/'plugins/adapters'/model/domain/str(horizon)/'test.npz'
                with np.load(original_path) as original:
                    old=metric(original['base_pred'],original['target'])
                    old_target=original['target']
                tasks={}
                for variant in ['control','internal']:
                    directory=args.aurora_directory if model=='Aurora' and args.aurora_directory else out
                    path=directory/variant/model/domain/str(horizon)/'2026'
                    item=json.loads((path/'EVALUATED.json').read_text())
                    with np.load(path/'test_predictions.npz') as data:
                        if data['target'].shape!=old_target.shape or not np.allclose(
                            data['target'],old_target,rtol=2e-4,atol=2e-4):
                            raise ValueError(f'target mismatch: {model}/{domain}/{horizon}/{variant}')
                        value=metric(data['pred'],data['target'],training_precision=True)
                    if not all(np.isclose(value[k],item['test'][k],rtol=1e-8) for k in value):
                        raise ValueError('saved prediction and recorded metric disagree')
                    tasks[variant]=value
                row=dict(model=model,domain=domain,horizon=horizon,original=old,**tasks)
                for reference in ['original','control']:
                    for name in ['mse','mae']:
                        row[f'{name}_change_vs_{reference}']=100*(tasks['internal'][name]/
                            max(row[reference][name],1e-12)-1)
                rows.append(row)
    (out/'comparison_180.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    summary=[]
    for model in models:
        tasks=[r for r in rows if r['model']==model]
        r=dict(model=model,tasks=len(tasks))
        for ref in ['original','control']:
            for name in ['mse','mae']:
                r[f'{name}_macro_change_vs_{ref}']=float(np.mean(
                    [t[f'{name}_change_vs_{ref}'] for t in tasks]))
            r[f'both_better_vs_{ref}']=sum(t[f'mse_change_vs_{ref}'] < -1e-5 and
                t[f'mae_change_vs_{ref}'] < -1e-5 for t in tasks)
            r[f'any_worse_vs_{ref}']=sum(t[f'mse_change_vs_{ref}'] > 1e-5 or
                t[f'mae_change_vs_{ref}'] > 1e-5 for t in tasks)
        summary.append(r)
    (out/'model_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for name in ['mse','mae']:
        for ref in ['original','control']:
            keys=[(r['domain'],r['horizon']) for r in rows if r['model']==models[0]]
            lookup={(r['model'],r['domain'],r['horizon']):r for r in rows}
            values=np.array([[lookup[m,d,h][f'{name}_change_vs_{ref}'] for m in models] for d,h in keys])
            limit=max(1,float(np.max(np.abs(values))))
            fig,ax=plt.subplots(figsize=(10,13))
            plot=ax.imshow(values,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
            ax.set_xticks(range(len(models)),models)
            ax.set_yticks(range(len(keys)),[f'{d}/{h}' for d,h in keys],fontsize=8)
            for i in range(len(keys)):
                for j in range(len(models)):
                    ax.text(j,i,f'{values[i,j]:+.1f}',ha='center',va='center',fontsize=7)
            ax.set_title(f'{name.upper()} change vs {ref} (%) | seed 2026 | negative = improvement')
            fig.colorbar(plot,ax=ax,label='Change (%)',shrink=.6)
            fig.tight_layout()
            fig.savefig(out/f'{name.upper()}_vs_{ref}.png',dpi=180)
            plt.close(fig)
    lines=[f'# {len(models)}模型内部融合实验结果','',
           '九领域、四预测长度、seed2026。负值表示误差下降。原baseline和匹配control分别比较；'
           '新输入与训练协议有变化，收益归因优先参考匹配control。','',
           '|模型|MSE vs 原baseline|MAE vs 原baseline|MSE vs control|MAE vs control|双指标改善 vs control /36|',
           '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append('|'+r['model']+'|'+'|'.join(f'{r[f"{n}_macro_change_vs_{ref}"]:+.2f}%'
            for ref in ['original','control'] for n in ['mse','mae'])+
            f'|{r["both_better_vs_control"]}|')
    all_models=all(r['mse_macro_change_vs_control']<0 and r['mae_macro_change_vs_control']<0 for r in summary)
    all_tasks=all(r['mse_change_vs_control']<-1e-5 and r['mae_change_vs_control']<-1e-5 for r in rows)
    lines.extend(['',f'{len(models)}模型的36任务平均双指标是否都改善：{all_models}。',
                  f'全部{len(rows)}任务的双指标是否逐项都改善：{all_tasks}。',
                  '平均提升与逐任务全部提升是不同判据；没有对退化项隐藏或回退。'])
    (out/'五模型结果分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':
    main()
