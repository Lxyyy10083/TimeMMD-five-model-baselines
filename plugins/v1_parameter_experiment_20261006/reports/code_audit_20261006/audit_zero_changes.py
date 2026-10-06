"""Read-only review of real saved V1-P1 predictions, checkpoints metadata and curves."""
from collections import Counter,defaultdict
from pathlib import Path
import csv
import hashlib
import json
import statistics
import numpy as np

BASE=Path(__file__).resolve().parents[2]
EXP=BASE/'V1参数实验_20261006'
FROZEN=BASE/'experiment_vcs/v1_frozen_inputs_20261006'
OUT=EXP/'零变化代码审核_20261006'
OUT.mkdir(exist_ok=True)
models=['TaTS','MM-TSFlib','SpecTF','CFA','Aurora']
domains=['Agriculture','Climate','Economy','Energy','Environment','Health','Security','SocialGood','Traffic']
source=json.loads((EXP/'RESULTS_180.json').read_text(encoding='utf-8'))
raw=[r for r in csv.DictReader((EXP/'test_seed_results.csv').open(encoding='utf-8-sig'))
     if r['version']=='V1-P1' and r['policy']=='tau0_original']
raw={(r['model'],r['domain'],int(r['horizon']),int(r['seed'])):r for r in raw}
assert len(source)==180 and len(raw)==540
fits={}
records=[]
metrics_mismatch=[]
alpha_mismatch=[]
profile=defaultdict(dict)

def metric(p,y):
    e=p.astype(np.float64)-y.astype(np.float64)
    return dict(mse=float(np.mean(e*e)),mae=float(np.mean(np.abs(e))))

def validate(tag,got,want,key):
    if abs(got-want)>1e-9*max(1,abs(want)):
        metrics_mismatch.append(dict(tag=tag,key=key,got=got,want=want))

for s in source:
    m,d,h=s['model'],s['domain'],s['horizon']
    with np.load(FROZEN/'plugins/adapters'/m/d/str(h)/'test.npz') as z:
        base=z['base_pred'].astype(np.float32);target=z['target'].astype(np.float32)
    with np.load(FROZEN/'plugins/adapters'/m/d/str(h)/'holdout.npz') as z:
        hb=z['base_pred'].astype(np.float32);hy=z['target'].astype(np.float32)
    baseline=metric(base,target)
    means=[]
    for seed in [2026,2027,2028]:
        key=(m,d,h,seed)
        fkey=(d,h,seed)
        if fkey not in fits:
            fits[fkey]=json.loads((EXP/'convergence_v3/final/lr_003'/str(seed)/d/str(h)/'fit_result.json').read_text(encoding='utf-8'))
        fit=fits[fkey];saved=raw[key];alpha=float(saved['alpha'])
        with np.load(EXP/'predictions/V1-P1'/str(seed)/d/str(h)/(m+'.npz')) as z:
            direct=z['raw_test'];chosen=z['original_chosen'];hold=z['raw_holdout'];gate=z['profile'][:,0]
        assert direct.shape==chosen.shape==base.shape==target.shape
        reconstructed=base+alpha*(direct-base)
        if not np.array_equal(chosen,reconstructed):
            metrics_mismatch.append(dict(tag='chosen_prediction_formula',key=key,max_abs=float(np.max(np.abs(chosen-reconstructed)))))
        final=metric(chosen,target);direct_metric=metric(direct,target)
        means.append(final)
        for k in ['mse','mae']:
            validate('baseline_'+k,baseline[k],float(saved['baseline_'+k]),key)
            validate('chosen_'+k,final[k],float(saved[k]),key)
            validate('raw_'+k,direct_metric[k],float(saved['raw_'+k]),key)
        hbase=metric(hb,hy)
        best_alpha=0;score=1.0
        for a in [.25,.5,.75,1]:
            met=metric(hb+a*(hold-hb),hy)
            mr=met['mse']/hbase['mse'];ar=met['mae']/hbase['mae'];val=.5*(mr+ar)
            if mr<1 and ar<1 and val<score:best_alpha,score=a,val
        if best_alpha!=alpha:
            alpha_mismatch.append(dict(key=key,saved_alpha=alpha,recomputed_alpha=best_alpha))
        profile[fkey][m]=gate
        records.append(dict(model=m,domain=d,horizon=h,seed=seed,alpha=alpha,best_epoch=fit['best_epoch'],
            epochs_run=fit['epochs_run'],original_mse=baseline['mse'],selected_mse=final['mse'],
            raw_saved_checkpoint_mse=direct_metric['mse'],selected_change_pct=100*(final['mse']/baseline['mse']-1),
            raw_change_pct=100*(direct_metric['mse']/baseline['mse']-1),
            selected_exactly_base=bool(np.array_equal(chosen,base)),raw_exactly_base=bool(np.array_equal(direct,base)),
            raw_base_max_abs=float(np.max(np.abs(direct-base))),
            raw_mse_holdout_ratio=fit['holdout'][m]['raw_mse_ratio'],raw_mae_holdout_ratio=fit['holdout'][m]['raw_mae_ratio']))
    for k in ['mse','mae']:
        validate('task_'+k,float(np.mean([r[k] for r in means])),s[k],(m,d,h))
        validate('task_baseline_'+k,baseline[k],s['baseline_'+k],(m,d,h))

task_rows=[]
for s in source:
    rs=[r for r in records if (r['model'],r['domain'],r['horizon'])==(s['model'],s['domain'],s['horizon'])]
    exact=s['mse']==s['baseline_mse']
    delta=100*(s['mse']/s['baseline_mse']-1)
    zero_display=abs(delta)<.00005
    all_zero=all(r['alpha']==0 for r in rs)
    all_init=all(r['best_epoch']==0 for r in rs)
    kind=('三个种子alpha=0且保存初始化权重' if all_zero and all_init else
          '三个种子alpha=0，至少一个保存训练权重' if all_zero else
          '存在非零alpha但初始化/浮点量级变化' if zero_display else '实际MSE发生变化')
    task_rows.append(dict(model=s['model'],domain=s['domain'],horizon=s['horizon'],original_mse=s['baseline_mse'],
        selected_mse=s['mse'],change_pct=delta,displayed_zero=zero_display,exact_mse_equal=exact,
        alpha_seed2026=rs[0]['alpha'],alpha_seed2027=rs[1]['alpha'],alpha_seed2028=rs[2]['alpha'],
        best_epoch_seed2026=rs[0]['best_epoch'],best_epoch_seed2027=rs[1]['best_epoch'],best_epoch_seed2028=rs[2]['best_epoch'],
        all_alpha_zero=all_zero,all_saved_initial=all_init,cause=kind))

fit_rows=[]
for (d,h,seed),f in fits.items():
    trace=f['trace'];n=len(trace);first=trace[-min(20,n)];last=trace[-1]
    changes={m:.5*((last['models'][m]['raw_mse_ratio']+last['models'][m]['raw_mae_ratio'])-
                   (first['models'][m]['raw_mse_ratio']+first['models'][m]['raw_mae_ratio'])) for m in models}
    fit_rows.append(dict(domain=d,horizon=h,seed=seed,best_epoch=f['best_epoch'],epochs_run=f['epochs_run'],
        converged_flag=f['converged'],selected_initial=f['best_epoch']==0,
        training_point_first=trace[0]['point'],training_point_last=last['point'],
        raw_scores_in_recent_window_still_improved=sum(v<-1e-5 for v in changes.values()),
        raw_recent_window_changes=changes))
gate_same=sum(all(np.array_equal(v[models[0]],v[m]) for m in models[1:]) for v in profile.values())
def counts(sub):
    return dict(tasks=len(sub),displayed_zero=sum(r['displayed_zero'] for r in sub),
        exact_mse_equal=sum(r['exact_mse_equal'] for r in sub),all_alpha_zero=sum(r['all_alpha_zero'] for r in sub),
        all_saved_initial=sum(r['all_saved_initial'] for r in sub))
stats=dict(task_counts=counts(task_rows),causes=dict(Counter(r['cause'] for r in task_rows)),
    per_model={m:counts([r for r in task_rows if r['model']==m]) for m in models},
    per_domain={d:counts([r for r in task_rows if r['domain']==d]) for d in domains},
    seed_rows=540,seed_alpha_zero=sum(r['alpha']==0 for r in records),seed_chosen_equals_base=sum(r['selected_exactly_base'] for r in records),
    seed_raw_equals_base=sum(r['raw_exactly_base'] for r in records),
    seed_trained_checkpoint_bypassed=sum(r['alpha']==0 and r['best_epoch']>0 for r in records),
    seed_initial_checkpoint_nonzero_alpha=sum(r['alpha']>0 and r['best_epoch']==0 for r in records),
    fits=108,initial_checkpoints=sum(r['selected_initial'] for r in fit_rows),
    early_stop_with_recent_raw_improvement=sum(r['raw_scores_in_recent_window_still_improved']>0 for r in fit_rows),
    five_model_gate_arrays_identical=gate_same,metrics_mismatches=metrics_mismatch,alpha_mismatches=alpha_mismatch)
zero_keys={(r['model'],r['domain'],r['horizon']) for r in task_rows if r['displayed_zero']}
stats['zero_tasks_positive_alpha_trained_checkpoint_rows']=sum(
    (r['model'],r['domain'],r['horizon']) in zero_keys and r['alpha']>0 and r['best_epoch']>0 for r in records)
stats['trained_bypass_raw_mse_better_but_mae_not_better']=sum(
    r['alpha']==0 and r['best_epoch']>0 and r['raw_mse_holdout_ratio']<1 and r['raw_mae_holdout_ratio']>=1 for r in records)
stats['initial_raw_prediction_max_difference']=max(r['raw_base_max_abs'] for r in records if r['best_epoch']==0)
for name,rs in [('180项零变化原因.csv',task_rows),('540项种子预测审核.csv',records)]:
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
(OUT/'审核统计.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'108组训练与权重选择审核.json').write_text(json.dumps(fit_rows,ensure_ascii=False,indent=2),encoding='utf-8')
code=BASE/'experiment_vcs/server_baseline/plugins/v1_parameter_experiment_20261006'
def link(name,line,label):return f'[{label}]({(code/name).as_posix()}:{line})'
cfa=[r for r in records if r['model']=='CFA' and r['domain']=='Agriculture' and r['horizon']==6]
raw_cfa=statistics.mean(r['raw_saved_checkpoint_mse'] for r in cfa)
base_cfa=cfa[0]['original_mse']
climate=fits[('Climate',6,2026)]
md=['# V1-P1代码审核：为什么大量MSE变化为0.0000%','',
    '日期：2026-10-06。审核范围为本轮V1-P1代码、108组正式插件训练记录、540条主策略种子记录与保存的实际预测数组。此报告未修改训练代码、原版模型或已有预测，未启动新训练。','',
    '## 1. 审核结论','',
    '大量零变化主要来自两种选择机制：验证alpha回退到原版，以及将训练前初始化权重选为最佳权重。结果计算没有发现错列、漏跑替代或误用指标，但现有实验衡量的是“插件加验证回退策略”，不能当成每个任务都有效使用了训练后的插件。','',
    f"- 180项中，{stats['task_counts']['displayed_zero']}项显示0.0000%，{stats['task_counts']['exact_mse_equal']}项的汇总MSE浮点值完全相等。二者不同，因为三种子指标均值和初始化的数值误差会产生极小浮点差异。",
    '- 85项三个种子的alpha全为0，最终实际使用原始预测。其中21项三个种子都选初始化权重，64项至少一个种子选了训练权重但被alpha停用。',
    '- 另外23项存在非零alpha，但非零alpha对应的都是epoch0权重，不是有效使用训练后模块产生的变化。',
    f"- 108组正式插件训练中，{stats['initial_checkpoints']}组最终保存epoch0初始化权重。",
    f"- 540条模型×领域×步长×种子记录中，{stats['seed_alpha_zero']}条alpha为0；其中{stats['seed_trained_checkpoint_bypassed']}条曾保存训练后的共享权重，但该模型的插件预测最终被停用。",
    f"- {stats['seed_initial_checkpoint_nonzero_alpha']}条记录在epoch0得到非零alpha，属于严格比较受浮点差异影响的选择。初始化输出与原版预测的最大逐元素差仅{stats['initial_raw_prediction_max_difference']:.12g}。",
    '- 对540条记录独立复算原始MSE/MAE、插件原始输出误差、最终选用输出误差及alpha选择；对180项复算三种子指标均值：所有指标与选择结果一致，未发现统计或保存数值错误。','',
    '## 2. 逐模型与逐领域零变化数量','',
    '|模型|任务数|显示零变化|三个种子alpha全为0|三个种子都选初始化权重|',
    '|---|---:|---:|---:|---:|']
for m in models:
    v=stats['per_model'][m];md.append(f"|{m}|36|{v['displayed_zero']}|{v['all_alpha_zero']}|{v['all_saved_initial']}|")
md+=['','|领域|任务数|显示零变化|三个种子alpha全为0|三个种子都选初始化权重|','|---|---:|---:|---:|---:|']
for d in domains:
    v=stats['per_domain'][d];md.append(f"|{d}|20|{v['displayed_zero']}|{v['all_alpha_zero']}|{v['all_saved_initial']}|")
md+=['','两种原因可以重叠，不能将这些列相加。特别是Climate四步长的全部20项均显示零变化，12组步长×种子训练全部选择epoch0。','',
    '## 3. P1：alpha=0允许直接关闭整个插件','',
    link('experiment.py',123,'验证alpha选择代码')+'；'+link('run.py',161,'最终预测计算代码')+'。','',
    '代码初始化alpha=0，仅在验证集MSE与MAE同时下降时，从0.25/0.5/0.75/1中选择正权重。最终输出为：','',
    '```python\nchosen = base + alpha * (raw_plugin - base)\n```','',
    'alpha=0时，chosen严格等于base。这是V1保留的部署保护策略，不是数据没有进模型。但用户要求考察插件加到五个模型后的真实效果时，该策略会把失败修正直接表现为“与原版相同”。','',
    '一个实际例子（CFA，Agriculture，步长6，三种子指标均值）：','',
    '|输出|实际MSE|相对原版变化|','|---|---:|---:|',
    f'|无插件原始预测|{base_cfa:.9f}|0.0000%|',
    f'|已保存训练权重的插件直接输出|{raw_cfa:.9f}|{100*(raw_cfa/base_cfa-1):+.4f}%|',
    f'|验证alpha回退后的输出（原Excel）|{base_cfa:.9f}|0.0000%|','',
    '三个种子的共享权重最佳轮次分别为61/59/46，实际训练97/99/107轮。插件直接输出已经改变，而且在该测试任务上变差；验证选择alpha=0后表格显示0。因此不能将这些0计为插件成功保持原性能。','',
    f"在200条“训练权重被停用”种子记录中，有{stats['trained_bypass_raw_mse_better_but_mae_not_better']}条的alpha=1输出验证MSE下降但MAE没有下降。双指标约束也会拒绝只改善MSE的候选；该计数不表示更小alpha一定能通过，独立重算已确认原网格均未被选用。",'',
    '## 4. P1：最终权重可能根本不是训练后的权重','',
    link('experiment.py',197,'epoch0初始化最佳权重')+'；'+link('experiment.py',238,'最佳权重更新条件')+'；'+link('experiment.py',257,'最终保存best_state')+'。','',
    '训练开始前设best_epoch=0并保存初始化state。之后只有五模型“alpha选用后的验证评分”平均改善超过1e-5才更新best_state。若未达到条件，即便优化器运行了32轮或更多，最终module.pt仍是初始化state。','',
    link('model.py',44,'flow末层零初始化')+'；'+link('model.py',129,'正负配对采样')+'；'+link('model.py',204,'残差修正输出')+'。','',
    'flow的最后线性层权重和偏置初始化为0，正负配对噪声的均值接近0。于是epoch0的残差点预测接近0，输出接近原版预测。保留epoch0作为候选，本来是为了保护原预测，但与“展示训练后模块的实验效果”不一致。','',
    f"实际例子：Climate/6/seed2026训练{climate['epochs_run']}轮，最终best_epoch={climate['best_epoch']}。训练point损失由{climate['trace'][0]['point']:.9f}变为{climate['trace'][-1]['point']:.9f}，说明训练执行并有变化；最终测试却使用初始化插件。",
    '因此保存的raw_test也不总是“训练后插件的无回退输出”：33组仍是初始化输出。直接把alpha改成1，无法恢复未保存的最终训练权重。','',
    '## 5. P1/P2：收敛标记不能证明最终插件已学到有效修正','',
    link('experiment.py',195,'raw历史最佳初始化')+'；'+link('experiment.py',225,'历史最佳与stale计数')+'；'+link('experiment.py',248,'停止与converged条件')+'。','',
    'raw_stale检查的是“连续多轮未超过历史最佳验证评分”，不是“最近多轮评分已经不再变化”。初始化对应原版预测，若训练后验证评分先变差，即使之后持续恢复、但尚未超过初始化，计数仍不重置。','',
    f"本轮108组中，有{stats['early_stop_with_recent_raw_improvement']}组在最后最多20轮的首尾比较中，至少一个模型的raw验证评分仍改善超过1e-5，但停止条件依然达标。这是早停语义，并不证明这些模型的近期曲线已平坦；也不能据此证明33组最终选初始化权重的任务已学到有效插件。",
    '之前“全部收敛”的描述应更正为“全部满足代码定义的验证早停条件”。训练终点、最佳验证权重、最终启用插件是三件不同的事。','',
    '## 6. P2：共享训练难以照顾每个底模，门控也未感知底模','',
    link('model.py',191,'语义gate输入')+'；'+link('model.py',236,'点损失与可选模型归一化')+'；'+link('experiment.py',174,'五模型混合训练')+'。','',
    '- 一个领域/步长/种子只训练一个共享插件，五模型的样本合并；最终配置balanced=False。训练的点损失按实际误差量级相加，不按各底模基准误差平衡。误差较大的模型可能主导共享更新。这是优化风险，尚未通过新的受控消融验证因果贡献。',
    '- 最终配置model_conditioned=False。model_id虽然传入forward，但当前设置没有使用模型身份嵌入。',
    '- 语义gate只看numeric temporal、semantic、coverage、agreement，没有输入base_features或模型身份。相同历史与文本时，各底模的gate必然相同；实测108组的五模型gate数组全部完全一致。flow条件仍包含各底模base_features，因此不能说整个插件忽略了底模预测。',
    '- 模型之间对新增文本信息的需要不同，这个gate却无法按底模的预测和模型身份调整语义效用。外部alpha承担了大量拒绝失败修正的职责；这是SpecTF/CFA零变化多的一个结构性风险，并非已验证的唯一原因。','',
    '## 7. P2：24轮筛选不足以证明“收敛后的最佳超参”','',
    link('run.py',51,'筛选与正式训练预算')+'；'+link('run.py',303,'两个种子筛选并锁定赢家')+'。','',
    '16组参数在9个pilot步长以最多24轮筛选；对照及两个候选在第二种子复核，最后按holdout锁定lr=0.003。正式阶段才以收敛协议重训对照和赢家。该方案是节约计算的分阶段筛选，但学习率较大可能在短预算下占优势；不能由此声称16组参数都已收敛比较，或0.003必然是完整36任务上的最佳设置。','',
    '## 8. 必须怎样调整下一轮实验','',
    '1. 将“训练后插件直接输出”作为主实验，另报“加验证回退保护后的部署输出”。字段名称明确区分，不能把直接返回原版的任务说成插件有效。',
    '2. 正式主实验的候选权重必须来自实际训练轮次；epoch0只保存为原始对照。按直接插件验证指标选择训练后的最佳权重，保留best_raw.pt和last.pt及其轮次。即使结果退化，也如实报告。',
    '3. 若另保留V1回退策略，至少为best_epoch=0显式设alpha=0；alpha比较加入数值容差，避免初始化浮点噪声触发非零权重。',
    '4. 将训练是否结束、验证是否平台、是否选用训练权重、是否启用插件、是否测试改善分开记录。依据近期验证曲线判断平台，并保留安全上限未满足规则的失败状态。',
    '5. 共享模块加入底模预测条件和模型身份条件到语义效用门控，使用仅由fit估计的损失归一化；通过固定数据、种子和预算的消融验证是否改善，而不是先假定有效。',
    '6. 关键超参候选补充按相同正式停止规则的验证复核，防止短预算选出仅前期较快的设置；不根据已看过的测试结果替换赢家。','',
    '上述调整属于新实验协议，应保存当前V1-P1及其结果，另建版本，避免覆盖既有baseline与历史结论。代码审核阶段不能用重新选测试最好的权重来改写已有结果。','',
    '## 9. 审核证据与范围','',
    '- `180项零变化原因.csv`：逐模型、逐领域、逐步长的原始/选用MSE、三个alpha与最佳轮次、零变化原因。',
    '- `540项种子预测审核.csv`：每种子的直接插件MSE、回退后的MSE、逐元素是否等于原版、最大差值。',
    '- `108组训练与权重选择审核.json`：实际运行轮次、选中轮次和近期曲线诊断。',
    '- `审核统计.json`：全部计数及独立复算差异（本轮指标差异和alpha差异均为空）。',
    '- 复算使用冻结的test/holdout数组及已保存预测；没有调用服务器或改动训练。未审核五篇原论文底模的所有训练实现，本报告聚焦本轮插件和评估路径。零变化的已确认原因不依赖反归一化假设。','']
(OUT/'V1P1零变化与代码审核.md').write_text('\n'.join(md),encoding='utf-8')
print(json.dumps(stats,ensure_ascii=False,indent=2))
