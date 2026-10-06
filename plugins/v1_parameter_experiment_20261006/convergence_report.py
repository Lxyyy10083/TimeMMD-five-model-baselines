"""Audit every formal fit and draw actual validation curves, without retraining."""
from pathlib import Path
import csv
import json

def verify_and_plot(dest, expected):
    results=[json.loads(p.read_text(encoding='utf-8')) for p in (dest/'convergence_v3/final').glob('*/*/*/*/fit_result.json')]
    assert len(results)==expected
    rows=[]
    for r in results:
        assert r['converged'] and r['convergence_protocol']=='v3_raw_five_model_plateau'
        assert r['epochs_run']>=32 and r['selected_stale']>=20 and min(r['raw_stale'].values())>=20
        assert len(r['trace'])==r['epochs_run'] and r['stopping_reason']=='validation_plateau'
        for m,count in r['raw_stale'].items():
            rows.append(dict(variant=r['variant'],model=m,domain=r['domain'],horizon=r['horizon'],
                seed=r['seed'],epochs_run=r['epochs_run'],best_epoch=r['best_epoch'],
                selected_stale=r['selected_stale'],raw_stale=count,final_lr=r['final_lr']))
    with (dest/'逐模型收敛核对.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=sorted({r['variant'] for r in results})
    fig,axes=plt.subplots(len(names),2,figsize=(13,4*len(names)),squeeze=False,layout='constrained')
    for i,name in enumerate(names):
        r=max([r for r in results if r['variant']==name],key=lambda r:r['epochs_run'])
        t=r['trace'];x=[v['epoch'] for v in t]
        axes[i,0].plot(x,[v['total'] for v in t],label='Training objective')
        for m in r['raw_stale']:
            axes[i,1].plot(x,[.5*(v['models'][m]['raw_mse_ratio']+v['models'][m]['raw_mae_ratio']) for v in t],label=m)
        axes[i,1].plot(x,[v['holdout_score'] for v in t],label='Alpha-selected score',color='black',linestyle='--')
        for j in range(2):
            axes[i,j].set_title(f"{name}: {r['domain']} H={r['horizon']} seed={r['seed']}")
            axes[i,j].set_xlabel('Epoch');axes[i,j].axvline(r['best_epoch'],color='gray',linestyle=':')
            axes[i,j].legend(fontsize=8)
    fig.savefig(dest/'正式实验收敛曲线.png',dpi=180);plt.close(fig)
    p=dest/'V1参数实验结果.md'
    text=p.read_text(encoding='utf-8')
    text+=f'\n## 正式收敛逐项复核\n\n{len(results)}次独立正式拟合及{len(rows)}条逐模型平台期记录全部通过条件核对。下图展示各配置训练最久的任务；全部逐轮损失、学习率和五模验证指标保存在每组fit_result.json的trace中。\n\n![实际收敛曲线](正式实验收敛曲线.png)\n'
    p.write_text(text,encoding='utf-8')
