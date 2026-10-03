"""Summarize the locked three-seed experiment, parameter ranking, and diagnostics."""
from __future__ import annotations
import argparse
import json
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER=['Agriculture','Climate','Economy','Energy','Environment','Health','Security','SocialGood','Traffic']
MODELS=['TaTS','MM-TSFlib','SpecTF','CFA','Aurora']
HERE=Path(__file__).resolve().parent


def read_json(path):return json.loads(path.read_text(encoding='utf-8'))


def chart_heatmap(frame,metric,dest):
    keys=frame[['domain','horizon']].drop_duplicates().copy()
    keys['domain']=pd.Categorical(keys.domain,ORDER,ordered=True)
    keys=keys.sort_values(['domain','horizon'])
    pivot=frame.pivot(index=['domain','horizon'],columns='model',values=f'{metric}_delta_pct')
    values=pivot.reindex(pd.MultiIndex.from_frame(keys))[MODELS].to_numpy()
    limit=max(1.0,np.ceil(np.nanmax(np.abs(values))))
    fig,ax=plt.subplots(figsize=(10,13))
    img=ax.imshow(values,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
    ax.set_xticks(range(5),MODELS)
    ax.set_yticks(range(len(keys)),[f'{r.domain} / {r.horizon}' for r in keys.itertuples()],fontsize=8)
    for i in range(len(keys)):
        for j in range(5):
            value=values[i,j]
            ax.text(j,i,f'{value:+.1f}',ha='center',va='center',fontsize=7,
                    color='white' if abs(value)>0.6*limit else '#222222')
    ax.set_title(f'Tuned graph flow: {metric.upper()} change vs frozen baseline (%)\nMean of three independent training seeds; negative = improvement')
    fig.colorbar(img,ax=ax,shrink=.6,label='Change (%)')
    fig.tight_layout();fig.savefig(dest/f'{metric.upper()}_180_cases.png',dpi=180);plt.close(fig)


def charts(mean,summary,previous,ranking,audit,records,dest,winner):
    for metric in ('mse','mae'):chart_heatmap(mean,metric,dest)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for metric,ax in zip(('mse','mae'),axes):
        old=previous.groupby('model')[f'{metric}_delta_pct'].mean().reindex(MODELS)
        new=summary.set_index('model')[f'mean_{metric}_change'].reindex(MODELS)
        x=np.arange(5)
        ax.bar(x-.18,old,.36,label='Previous plugin',color='#a8b1be')
        ax.bar(x+.18,new,.36,label='Tuned (3-seed mean)',color='#276a9e')
        ax.axhline(0,color='#777',linewidth=.8)
        ax.set_xticks(x,MODELS,rotation=12);ax.set_ylabel('Macro mean change (%)')
        ax.set_title(metric.upper());ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'model_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    scores=pd.DataFrame(ranking).sort_values('score')
    ax.barh(scores.variant,100*(scores.score-1),color=['#c77927' if 'ablation' in v else '#276a9e' for v in scores.variant])
    ax.invert_yaxis();ax.set_xlabel('Holdout composite relative error change (%)')
    ax.set_title('Parameter screening: 9 domains, 5 models, seed 2026\nIncludes per-model alpha selection; test metrics are excluded')
    fig.tight_layout();fig.savefig(dest/'parameter_ranking.png',dpi=180);plt.close(fig)
    grouped=audit.groupby('domain')[['energy_k4','energy_k8','energy_k16']].mean().reindex(ORDER)
    fig,ax=plt.subplots(figsize=(12,4.5));x=np.arange(9)
    for offset,(key,label) in zip((-.25,0,.25),(('energy_k4','K=4'),('energy_k8','K=8'),('energy_k16','K=16'))):
        ax.bar(x+offset,100*grouped[key],.25,label=label)
    ax.set_xticks(x,ORDER,rotation=20);ax.set_ylabel('Residual energy retained (%)');ax.set_ylim(0,105)
    ax.set_title('Purged calibration residuals: mean DCT energy coverage across models and horizons')
    ax.legend();fig.tight_layout();fig.savefig(dest/'DCT_energy.png',dpi=180);plt.close(fig)
    if records:
        pilot={'Agriculture':6,'Climate':8,'Economy':8,'Energy':36,'Environment':336,'Health':24,
               'Security':10,'SocialGood':8,'Traffic':8}
        fig,axes=plt.subplots(3,3,figsize=(13,10))
        for domain,ax in zip(ORDER,axes.flat):
            for r in records:
                if r['domain']!=domain or r['horizon']!=pilot[domain]:continue
                ax.plot([e['epoch'] for e in r['trace']],
                        [100*(e['holdout_score']-1) for e in r['trace']],label=str(r['seed']))
            ax.axhline(0,color='#aaa',linewidth=.5);ax.set_title(f'{domain} / {pilot[domain]}')
            ax.set_xlabel('Epoch');ax.set_ylabel('Holdout error change (%)');ax.legend(fontsize=7)
        fig.suptitle(f'Final validation trajectories: {winner}',fontsize=14)
        fig.tight_layout();fig.savefig(dest/'training_curves.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
    dest=args.directory;seeds=pd.read_csv(dest/'test_seed_results.csv')
    if len(seeds)!=540 or seeds[['model','domain','horizon','seed']].duplicated().any():
        raise ValueError('Expected 540 unique test records')
    winner=read_json(dest/'winner.json');screen=read_json(dest/'screen_ranking.json')
    params=read_json(dest/'experiment_config.json')['variants'];audit=pd.read_csv(dest/'data_audit.csv')
    previous=pd.read_csv(HERE.parent/'semflow/refined_results/complete_metrics.csv')
    records=[]
    archive=dest/'experiment_artifacts.tar.gz'
    if archive.exists():
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                parts=Path(member.name).parts
                if len(parts)==6 and parts[0]=='final' and parts[-1]=='fit_result.json':
                    records.append(json.load(tar.extractfile(member)))
    epoch_map={(r['domain'],r['horizon'],r['seed']):r['best_epoch'] for r in records}
    if len(epoch_map)!=108:raise ValueError('Missing final checkpoint metadata')
    seeds['best_epoch']=[epoch_map[(r.domain,r.horizon,r.seed)] for r in seeds.itertuples()]
    seeds['effective_enabled']=(seeds.alpha>0)&(seeds.best_epoch>0)
    rows=[]
    for (domain,horizon,model),group in seeds.groupby(['domain','horizon','model']):
        if len(group)!=3:raise ValueError('Missing seed')
        row=dict(domain=domain,horizon=int(horizon),model=model,
                 baseline_mse=float(group.baseline_mse.iloc[0]),baseline_mae=float(group.baseline_mae.iloc[0]),
                 plugin_mse=float(group.plugin_mse.mean()),plugin_mae=float(group.plugin_mae.mean()),
                 mse_std=float(group.plugin_mse.std(ddof=1)),mae_std=float(group.plugin_mae.std(ddof=1)),
                 enabled_seeds=int(group.effective_enabled.sum()),
                 both_better_seeds=int(((group.mse_delta_pct < -1e-5)&(group.mae_delta_pct < -1e-5)).sum()),
                 alpha_mean=float(group.alpha.mean()),gate_mean=float(group.gate_mean.mean()),
                 correction_std_units=float(group.correction_std_units.mean()),
                 raw_mse=float(group.raw_mse.mean()),raw_mae=float(group.raw_mae.mean()),
                 test_windows=int(group.test_windows.iloc[0]))
        row['mse_delta_pct']=100*(row['plugin_mse']/row['baseline_mse']-1)
        row['mae_delta_pct']=100*(row['plugin_mae']/row['baseline_mae']-1)
        rows.append(row)
    mean=pd.DataFrame(rows).merge(previous[['domain','horizon','model','plugin_mse','plugin_mae']].rename(
          columns={'plugin_mse':'previous_mse','plugin_mae':'previous_mae'}),on=['domain','horizon','model'],validate='one_to_one')
    mean['domain']=pd.Categorical(mean.domain,ORDER,ordered=True)
    mean['model']=pd.Categorical(mean.model,MODELS,ordered=True)
    mean=mean.sort_values(['domain','horizon','model']).reset_index(drop=True)
    mean.to_csv(dest/'results_180_mean_std.csv',index=False,encoding='utf-8-sig')
    summary_rows=[]
    for name in MODELS:
        data=mean[mean.model==name]
        summary_rows.append(dict(model=name,tasks=len(data),enabled_cases=int((data.enabled_seeds>0).sum()),
             both_better=int(((data.mse_delta_pct < -1e-5)&(data.mae_delta_pct < -1e-5)).sum()),
             any_worse=int(((data.mse_delta_pct > 1e-5)|(data.mae_delta_pct > 1e-5)).sum()),
             unchanged=int(((data.mse_delta_pct.abs() <= 1e-5)&(data.mae_delta_pct.abs() <= 1e-5)).sum()),
             mean_mse_change=float(data.mse_delta_pct.mean()),mean_mae_change=float(data.mae_delta_pct.mean()),
             raw_mse_change=float((100*(data.raw_mse/data.baseline_mse-1)).mean()),
             raw_mae_change=float((100*(data.raw_mae/data.baseline_mae-1)).mean())))
    summary=pd.DataFrame(summary_rows);summary.to_csv(dest/'model_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(screen).to_csv(dest/'parameter_ranking.csv',index=False,encoding='utf-8-sig')
    charts(mean,summary,previous,screen,audit,records,dest,winner['variant'])
    ablation=read_json(dest/'winner_ablation.json') if (dest/'winner_ablation.json').exists() else None
    both=int(summary.both_better.sum());worse=int(summary.any_worse.sum());unchanged=int(summary.unchanged.sum())
    lines=['# 参数实验结果与分析','',f'锁定配置：`{winner["variant"]}`。筛选依据为九领域五模型、两种随机种子的隔离 holdout；最终覆盖 36 个领域/步长、三个种子，共 540 条测试记录，汇总为 180 个模型任务。',
           '',f'三种子平均结果中，{both}/180 项 MSE 与 MAE 同时下降，{worse}/180 项至少一个指标上升，{unchanged}/180 项维持原预测。这里的“平均”是独立训练性能平均，不是对三个预测做集成。','',
           '## 与原底模比较','',
           '|模型|启用任务 / 36|双指标改善|至少一项退化|平均 MSE 变化|平均 MAE 变化|',
           '|---|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'|{r.model}|{r.enabled_cases}|{r.both_better}|{r.any_worse}|{r.mean_mse_change:+.2f}%|{r.mean_mae_change:+.2f}%|')
    lines+=['','负值表示误差下降。宏平均按每项相对变化计算，避免 Security 的绝对误差主导汇总；全部 MSE/MAE 与原底模在相同训练段标准化空间计算。初始化 epoch=0 的图流是恒等映射，配对采样存在约 1e-9 历史标准差单位的浮点误差，会让严格小于 1 的选择规则产生无意义的非零 alpha。启用数排除了这些恒等 checkpoint；结果本身完整保留，变化小于 0.00001% 的项目按持平统计。',
            '', '## 参数筛选与收益归因','',
            '本轮同时检查了修正幅度、方差收缩、DCT 维度、NLL 权重、语义效用权重、语义注意力和模型条件化。较大损失权重是否有效以验证排行为准；组合配置有多个参数同时改变，不能归因于单一因素。',
            '', '|配置|修正上限|初始 logit|方差系数|K 上限|NLL 权重|语义权重|首轮验证综合变化|',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in screen:
        c=params[row['variant']]
        lines.append(f'|{row["variant"]}|{c["correction_limit"]}|{c["correction_init"]}|{c["variance_beta"]}|{c["coefficients"]}|{c["nll_weight"]}|{c["utility_weight"]}|{100*(row["score"]-1):+.2f}%|')
    lines+=['','短步长下 K 被截断为 H，因此部分 K=8/16 试验实际维度相同。首轮排行包含无文本和时间打乱消融；晋级规则在运行前限定为三个真实文本配置，消融用于诊断，不作为多模态插件候选。',
            '', '同协议 control 修复了 checkpoint 选择、目标窗口重叠，并统一采用 holdout 双指标严格改善即可启用。因此相对历史版的变化同时包含协议影响；参数效果应优先比较本轮同协议 control。',
            '', '## 文本贡献检查','']
    if ablation:
        lines+=['以下均为同一赢家参数、九领域五模型、两个种子的 holdout 对照；不参与测试集上的配置选择。','',
                '|输入|验证 MSE 变化|验证 MAE 变化|验证综合变化|','|---|---:|---:|---:|']
        for row in [ablation['real_text'],*ablation['ablations']]:
            label={'winner_numeric':'无文本','winner_shuffle':'窗口内打乱文本'}.get(row['variant'],'真实文本')
            lines.append(f'|{label}|{100*(row["mse_ratio"]-1):+.2f}%|{100*(row["mae_ratio"]-1):+.2f}%|{100*(row["score"]-1):+.2f}%|')
        real=ablation['real_text']['score'];numeric=next(r['score'] for r in ablation['ablations'] if r['variant']=='winner_numeric')
        lines.append('')
        lines.append('真实文本在本对照中优于无文本，但尚需独立复现才能确认语义贡献。' if real<numeric else
                     '真实文本未优于无文本。当前收益可能主要来自历史数值残差校准，不能声称已证实深度语义融合带来额外收益。')
    if records:
        capped=sum(r['epochs_run']>=80 for r in records)
        cap_selected=sum(r['epochs_run']>=80 and r['best_epoch']>=72 for r in records)
        lines+=['','## 训练与数据局限','',f'正式训练记录共 {len(records)} 组，{capped} 组达到 80 轮上限，其中 {cap_selected} 组最佳 epoch 接近上限。其余使用验证集早停。达到上限的组不应被表述为已经完全收敛；训练和验证曲线见 `training_curves.png`。']
    lines+=['','Security/12 经 H-1 隔离后每底模只有 3 个 fit 预测原点；重叠窗口也不等于独立样本。三种子离散度反映优化随机性，不能代替独立时间块的泛化验证。测试集曾在历史实验中被查看，本轮属于探索性复现。',
            '', '## 图与完整数据','', '![各模型对比](model_comparison.png)','',
            '![参数排行](parameter_ranking.png)','',
            '完整 180 项均值、种子标准差、上一版指标和原始候选指标见 `results_180_mean_std.csv`；540 项独立种子指标见 `test_seed_results.csv`。权重和全部逐轮记录已归档在 `experiment_artifacts.tar.gz`。']
    (dest/'参数实验结果分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    detail_columns=['domain','horizon','model','baseline_mse','previous_mse','plugin_mse','mse_std',
                    'mse_delta_pct','baseline_mae','previous_mae','plugin_mae','mae_std','mae_delta_pct',
                    'enabled_seeds','both_better_seeds']
    param_rows=[]
    for r in screen:
        c=params[r['variant']]
        param_rows.append([r['variant'],c['correction_limit'],c['correction_init'],c['variance_beta'],
                          c['coefficients'],c['nll_weight'],c['utility_weight'],c['pooling'],
                          c['model_conditioned'],c['text_mode'],r['mse_ratio']-1,r['mae_ratio']-1,r['score']-1])
    payload=dict(winner=winner,summary=json.loads(summary.to_json(orient='split')),
                 detail=json.loads(mean[detail_columns].to_json(orient='split',double_precision=15)),
                 parameters=param_rows,seeds=json.loads(seeds.to_json(orient='split',double_precision=15)))
    (dest/'workbook_data.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    print(summary.to_string(index=False));print('both/worse/unchanged',both,worse,unchanged)


if __name__=='__main__':main()
