"""Export all tasks, including regressions and capped training runs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=Path(__file__).parent/'outputs_20261004'
    rows=json.loads((out/'RESULTS.json').read_text())
    models=['TaTS','MM-TSFlib','SpecTF','CFA','Aurora']
    summary=[]
    for m in models:
        tasks=[r for r in rows if r['model']==m]
        summary.append(dict(model=m,tasks=len(tasks),
            **{f'{k}_macro_change':float(np.mean([r['changes_vs_continued'][k] for r in tasks])) for k in ['mse','mae']},
            both_better=sum(all(r['changes_vs_continued'][k]<0 for k in ['mse','mae']) for r in tasks)))
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# V3 五模型结果','',
           '九领域、四长度、单种子2026。配置只按验证集选择。负百分比为误差降低。',
           '主要比较对象为同等追加训练的 continued control；旧 control 同时保留在 RESULTS.json。',
           '采用旧实验 checkpoint 初始化，属于继续训练实验，不能直接与从头训练归为同等预算。','',
           '|模型|平均MSE变化|平均MAE变化|双指标改善任务数|',
           '|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['model']}|{r['mse_macro_change']:+.2f}%|{r['mae_macro_change']:+.2f}%|{r['both_better']}/36|")
    capped=[]
    for p in out.glob('*/*/*/*/*/2026/COMPLETED.json'):
        x=json.loads(p.read_text())
        if x['capped']: capped.append(str(p.relative_to(out)))
    lines.extend(['',f'达到300 epoch上限的训练数：{len(capped)}。达到上限不代表已经收敛。',
                  '本轮设计参考了上一轮测试退化情况，复用了相同测试集，结果属于探索性迭代；最终结论需要独立数据或新时间段验证。'])
    (out/'V3结果分析.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'CAPPED_RUNS.json').write_text(json.dumps(capped,indent=2),encoding='utf-8')
    keys=[(r['domain'],r['horizon']) for r in rows if r['model']==models[0]]
    lookup={(r['model'],r['domain'],r['horizon']):r for r in rows}
    for metric in ['mse','mae']:
        values=np.array([[lookup[m,d,h]['changes_vs_continued'][metric] for m in models] for d,h in keys])
        limit=max(1,float(np.max(abs(values))))
        fig,ax=plt.subplots(figsize=(10,13))
        plot=ax.imshow(values,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        ax.set_xticks(range(5),models)
        ax.set_yticks(range(36),[f'{d}/{h}' for d,h in keys],fontsize=8)
        for i in range(36):
            for j in range(5): ax.text(j,i,f'{values[i,j]:+.1f}',ha='center',va='center',fontsize=7)
        ax.set_title(f'V3 {metric.upper()} change vs continued control (%)')
        fig.colorbar(plot,ax=ax,shrink=.6)
        fig.tight_layout()
        fig.savefig(out/f'{metric.upper()}_vs_continued_control.png',dpi=180)
        plt.close(fig)


if __name__=='__main__': main()
