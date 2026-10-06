from pathlib import Path
import csv
import hashlib
import json
import statistics

BASE=Path(__file__).resolve().parents[2]
SRC=BASE/'V1参数实验_20261006/RESULTS_180.json'
OUT=BASE/'outputs/v1_p1_original_comparison_20261006'
OUT.mkdir(parents=True,exist_ok=True)
rows=json.loads(SRC.read_text(encoding='utf-8'))
models=['TaTS','MM-TSFlib','SpecTF','CFA','Aurora']
domains=['Agriculture','Climate','Economy','Energy','Environment','Health','Security','SocialGood','Traffic']
rows=sorted(rows,key=lambda r:(domains.index(r['domain']),r['horizon'],models.index(r['model'])))
assert len(rows)==180 and len({(r['domain'],r['horizon'],r['model']) for r in rows})==180
old=json.loads((BASE/'GANF_V1_MSE_Excel_rerun2_20261006/RESULTS_180.json').read_text(encoding='utf-8'))
old={(r['domain'],r['horizon'],r['model']):r for r in old}
for r in rows:
    assert r['version']=='V1-P1' and r['policy']=='tau0_original'
    for k in ['mse','mae']:
        assert r['baseline_'+k]>0
        assert abs(r['baseline_'+k]-old[(r['domain'],r['horizon'],r['model'])]['baseline_'+k])<1e-8
        assert abs(100*(1-r[k]/r['baseline_'+k])-r[k+'_decrease_pct'])<1e-8
summary=[]
for m in models:
    rs=[r for r in rows if r['model']==m]
    assert len(rs)==36
    s=dict(model=m,tasks=36)
    for k in ['mse','mae']:
        s['original_'+k+'_mean']=statistics.mean(r['baseline_'+k] for r in rs)
        s['module_'+k+'_mean']=statistics.mean(r[k] for r in rs)
        s[k+'_mean_decrease_pct']=100*(1-s['module_'+k+'_mean']/s['original_'+k+'_mean'])
        s[k+'_task_macro_decrease_pct']=statistics.mean(r[k+'_decrease_pct'] for r in rs)
    s['mse_lower_tasks']=sum(r['mse_decrease_pct']>1e-5 for r in rs)
    s['mse_higher_tasks']=sum(r['mse_decrease_pct']<-1e-5 for r in rs)
    s['mse_unchanged_tasks']=36-s['mse_lower_tasks']-s['mse_higher_tasks']
    summary.append(s)
data=dict(rows=rows,models=models,domains=domains,summary=summary,
    source=str(SRC),source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
    baseline_matches_frozen_unmodified_reference=True)
(OUT/'comparison_inputs.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
fields=['domain','horizon','model','baseline_mse','mse','mse_decrease_pct','baseline_mae','mae','mae_decrease_pct','enabled_seeds']
with (OUT/'原始baseline与模块_180项.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
md=['# 原始 baseline 与当前 V1-P1 插件的直接对比','','## 对比对象与更正','',
    '之前回复的“未修改 baseline：MSE下降2.7367%”表示插件相对无插件原版的逐任务下降率宏平均，并非原版自身下降。该行标签不够明确。本报告直接列出两者实际误差。','',
    '- 原始 baseline：五个原模型的冻结测试预测，不含GANF/V1插件；对应数据字段baseline_mse、baseline_mae。已逐项核对与原始冻结参考一致，180项无不一致。',
    '- 当前添加模块：本轮验证集锁定的V1-P1配置lr_003；对应mse、mae。保持原版底模预测冻结，训练后置条件分布插件；不重训原模型。',
    '- 使用相同TimeMMD ReadGPT数据、时间切分、目标、预测步长、测试窗口及误差计算空间。本轮没有新增反归一化操作。',
    '- 插件结果为三种子指标均值，不是预测集成。主结果沿用验证alpha选择：0/0.25/0.5/0.75/1；alpha=0时保留原版预测。测试集不参与alpha或参数选择。',
    '- 下降率=(原始误差−模块误差)/原始误差×100%；正值改善，负值退化。原版与自己比较时下降率当然是0%。','',
    '## 五模型MSE实际均值','',
    '每模型36项（九领域×四步长）的简单数值均值。不同领域误差量级差异大，该汇总可能被Security等领域支配；判断单项表现应看下方逐领域与实际步长的明细。','',
    '|模型|原始baseline MSE均值|添加模块MSE均值|由两均值计算的下降率|逐任务下降率宏平均|',
    '|---|---:|---:|---:|---:|']
for s in summary:
    md.append(f"|{s['model']}|{s['original_mse_mean']:.6f}|{s['module_mse_mean']:.6f}|{s['mse_mean_decrease_pct']:+.4f}%|{s['mse_task_macro_decrease_pct']:+.4f}%|")
md+=['','两种下降率的公式不同：','',
    '- 均值下降率=1−平均(模块MSE)/平均(原始MSE)，相当于按各任务原始MSE大小加权其下降率。',
    '- 逐任务下降率宏平均=平均(1−模块MSE/原始MSE)，每任务权重相同。',
    '- 因此Aurora的逐任务宏平均虽然改善4.8939%，但MSE简单均值略增，不能宣称其平均实际MSE也下降。此处没有根据测试结果换用其他阈值或赢家。','',
    '## 180项实际结果','',
    'MSE和MAE均越小越好。Excel中的绿色表示下降、红色表示增加、白色附近表示持平；改善判定容差为下降率绝对值0.00001%。','']
for d in domains:
    md += [f'### {d}','','|步长|模型|原始MSE|模块MSE|MSE下降率|原始MAE|模块MAE|MAE下降率|','|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['domain']==d:
            md.append(f"|{r['horizon']}|{r['model']}|{r['baseline_mse']:.6f}|{r['mse']:.6f}|{r['mse_decrease_pct']:+.4f}%|{r['baseline_mae']:.6f}|{r['mae']:.6f}|{r['mae_decrease_pct']:+.4f}%|")
    md.append('')
md+=['## 来源','',f'结果文件：`{SRC}`。SHA256：`{data["source_sha256"]}`。','',
    '本报告重新组织已完成实验的结果，未更改训练代码、模型权重、测试预测或选择策略。历史V1及同收敛规则V1对照不作为本表的原始baseline。','']
(OUT/'原始baseline与模块直接对比.md').write_text('\n'.join(md),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
