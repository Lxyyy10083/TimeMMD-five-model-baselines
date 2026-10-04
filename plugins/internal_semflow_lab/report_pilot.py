import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(__file__).parent/'outputs_pilot'
rows=json.loads((out/'RESULTS.json').read_text())
lines=['# 实验室服务器精简迭代结果','',
       '五模型、Agriculture/12与Economy/10、单种子2026。同设置重新训练control与插件。负数表示误差下降。',
       '这是用于检查迭代方向的小规模实验，不能代表九领域四长度均有效。','',
       '|模型|领域|长度|Control MSE|插件MSE|变化|Control MAE|插件MAE|变化|',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    c,v,d=r['control'],r['internal'],r['change']
    lines.append(f"|{r['model']}|{r['domain']}|{r['horizon']}|{c['mse']:.6f}|{v['mse']:.6f}|{d['mse']:+.2f}%|{c['mae']:.6f}|{v['mae']:.6f}|{d['mae']:+.2f}%|")
capped=[]
for p in out.glob('*/*/*/*/2026/COMPLETED.json'):
    x=json.loads(p.read_text())
    if x['capped']:capped.append(str(p.relative_to(out)))
lines.extend(['',f'达到60 epoch训练上限的组数：{len(capped)}。不将达到上限称作收敛。',
              '本轮缩短了训练预算；不能与此前完整实验直接归因为插件效果，优先比较本轮paired control。'])
(out/'精简实验结果.md').write_text('\n'.join(lines),encoding='utf-8')
for metric in ['mse','mae']:
    fig,ax=plt.subplots(figsize=(12,4))
    labels=[r['model']+'\n'+r['domain']+'/'+str(r['horizon']) for r in rows]
    x=np.arange(len(rows))
    ax.bar(x-.2,[r['control'][metric] for r in rows],.4,label='Control')
    ax.bar(x+.2,[r['internal'][metric] for r in rows],.4,label='Internal V3')
    ax.set_xticks(x,labels,fontsize=7)
    ax.set_ylabel(metric.upper());ax.legend()
    fig.tight_layout();fig.savefig(out/f'{metric.upper()}_comparison.png',dpi=180);plt.close(fig)
