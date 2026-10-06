from pathlib import Path
import csv
import json
import statistics
import sys

BASE = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent.parent
MODELS = ['TaTS','MM-TSFlib','SpecTF','CFA','Aurora']
TOL = 0.00001

def read_csv(name):
    return list(csv.DictReader((BASE/name).open(encoding='utf-8-sig')))

def read_json(name):
    return json.loads((BASE/name).read_text(encoding='utf-8'))

sources = [
 ('CARMA', 'CARMA_results/results_180.csv', 'carma', 1),
 ('GANF共享初版','GANF_semflow_results/complete_metrics.csv','frozen',1),
 ('GANF单模型校准','GANF_semflow_results/refined/complete_metrics.csv','frozen',1),
 ('V1','GANF_V1_MSE_Excel_rerun2_20261006/RESULTS_180.json','v1',3),
 ('V2','GANF_internal_fusion_20261003/五模型V2完整结果_20261004/comparison_180.json','internal',1),
 ('V3全量收敛','GANF_internal_fusion_20261003/九领域五模型全量收敛结果_20261005/RESULTS.json','internal',1),
 ('V4','GANF_internal_fusion_20261003/V4五模型九领域完整结果_20261005/RESULTS.json','internal',1),
]
versions = []
for name,filename,kind,seeds in sources:
    raw = read_json(filename) if filename.endswith('.json') else read_csv(filename)
    assert len(raw) == 180, (name,len(raw))
    observations = []
    for r in raw:
        if kind == 'carma':
            old = {k:float(r['base_'+k]) for k in ['mse','mae']}
            new = {k:float(r['carma_'+k]) for k in ['mse','mae']}
            control = old
        elif kind in ['v1','frozen']:
            old = {k:float(r['baseline_'+k]) for k in ['mse','mae']}
            new = {k:float(r['plugin_'+k]) for k in ['mse','mae']}
            control = old
        else:
            old = r.get('historical_original',r.get('original'))
            control,new = r['control'],r['internal']
        assert old and control and new
        observations.append(dict(model=r['model'],domain=r['domain'],horizon=int(r['horizon']),
                                 historical=old,control=control,modified=new,
                                 decrease_control={k:100*(1-new[k]/control[k]) for k in ['mse','mae']},
                                 decrease_historical={k:100*(1-new[k]/old[k]) for k in ['mse','mae']}))
    assert len({(r['model'],r['domain'],r['horizon']) for r in observations})==180
    models = []
    for m in MODELS:
        cases = [r for r in observations if r['model']==m]
        assert len(cases)==36
        entry = dict(model=m)
        for comparison in ['control','historical']:
            key = 'decrease_'+comparison
            entry[comparison] = dict(
                mse_decrease_pct=statistics.mean(r[key]['mse'] for r in cases),
                mae_decrease_pct=statistics.mean(r[key]['mae'] for r in cases),
                both_better=sum(all(r[key][k]>TOL for k in ['mse','mae']) for r in cases),
                any_worse=sum(any(r[key][k]<-TOL for k in ['mse','mae']) for r in cases),
                unchanged=sum(all(abs(r[key][k])<=TOL for k in ['mse','mae']) for r in cases),
                mse_better=sum(r[key]['mse']>TOL for r in cases),
                mae_better=sum(r[key]['mae']>TOL for r in cases))
        models.append(entry)
    total={}
    for comparison in ['control','historical']:
        total[comparison] = {k:statistics.mean(m[comparison][k] for m in models) for k in ['mse_decrease_pct','mae_decrease_pct']}
        for k in ['both_better','any_worse','unchanged','mse_better','mae_better']:
            total[comparison][k] = sum(m[comparison][k] for m in models)
        total[comparison]['both_means_better_models'] = sum(m[comparison]['mse_decrease_pct']>TOL and m[comparison]['mae_decrease_pct']>TOL for m in models)
    versions.append(dict(name=name,source=filename,seeds=seeds,models=models,total=total,observations=observations))

reference={(r['model'],r['domain'],r['horizon']):r['historical'] for r in next(v for v in versions if v['name']=='V1')['observations']}
for v in versions:
    v['historical_baseline_max_relative_difference'] = max(abs(r['historical'][k]/reference[(r['model'],r['domain'],r['horizon'])][k]-1) for r in v['observations'] for k in ['mse','mae'])
    assert v['historical_baseline_max_relative_difference']<1e-5

out=BASE/'实验版本汇总_20261006'
out.mkdir(exist_ok=True)
(out/'版本汇总数据.json').write_text(json.dumps(versions,ensure_ascii=False,indent=2),encoding='utf-8')

core = [v for v in versions if v['name'] in ['V1','V2','V3全量收敛','V4']]
lines = ['# 已完成实验版本汇总：V1是不是效果最好？','',
         '日期：2026-10-06。根据本地已完成的逐项结果重新计算，没有启动新训练，没有修改任何模型代码。', '',
         '## 结论','',
         '**V1目前的整体平均MSE降幅最大，也是在核心V1至V4中唯一让五个模型的平均MSE、MAE都下降的版本。但不能说V1在每个模型、每个指标或每个任务上都是最好。**', '',
         '- 相对共同历史baseline，V1整体MSE平均减小2.71%，是已核对的七个完整版本中最大；整体MAE减小1.71%。',
         '- 同口径下，V2整体MAE减小2.94%，比V1更大；V2双指标改善98项，V1为62项，所以V1也不是改善覆盖数量最多。',
         '- 相对共同历史baseline，TaTS的V2、CFA和Aurora的V3结果优于V1。这些版本重新训练了主干，训练协议发生变化，不能把全部差异归功于插件。',
         '- 相对各轮匹配control，V4在SpecTF上平均MSE减小6.38%、MAE减小6.42%，但其余模型未实现双指标平均改善，不能推广成五模型通用成功。',
         '- V1有104/180项持平、14/180项至少一项退化，仍未达到九领域、五模型、四步长全面双指标下降的目标。', '',
         '## 1. 统一符号和统计口径','',
         '`误差减小百分比 = (对照误差 − 修改后误差) / 对照误差 × 100%`。', '',
         '**正数代表改善，负数代表误差增加。**MSE与MAE本身越小越好。所有均值先逐任务计算相对变化，再对180项作算术平均；每模型固定36项，因此与五模型宏平均一致。不直接将Security等不同领域的绝对MSE混合后计算百分比。', '',
         '数量判定沿用V1的0.00001%容差，避免恒等初始化的浮点误差被计作改善。双改善要求MSE和MAE同时下降；至少一项退化要求其中任意一项上升；持平要求两项均在容差内。', '',
         '两种对照分别回答不同问题：', '',
         '1. **共同历史baseline**：回答相对最初未添加模块的保存结果，实际误差是否更低。七个版本的历史对照数值逐项核对一致，最大相对差异小于2.3e-7。训练方式与随机种子数不同，所以这不是严格同预算的模块因果排名。',
         '2. **各轮匹配control**：回答同一轮相同数据接口、训练预算和微调方式下，模块本身是否带来额外收益。V1对照为冻结历史预测；V2至V4对照重新训练或追加训练。不同轮control不同，不能将这些百分比当作完全相同底模下的直接排名。', '',
         '## 2. 与共同历史baseline比较：实际指标结果','',
         '|完整版本|平均MSE减小%|平均MAE减小%|五模型平均双下降|任务双改善/180|至少一项退化/180|双持平/180|',
         '|---|---:|---:|---:|---:|---:|---:|']
for v in versions:
    t=v['total']['historical']
    lines.append(f"|{v['name']}|{t['mse_decrease_pct']:+.2f}|{t['mae_decrease_pct']:+.2f}|{t['both_means_better_models']}/5|{t['both_better']}|{t['any_worse']}|{t['unchanged']}|")
lines += ['', '不能只按双改善数量决定最好。V2和V3改动更广，改善任务更多，同时退化任务也明显更多；V1通过验证集alpha回退保留了104项原版预测。', '',
          '### 各模型对共同历史baseline的变化','',
          '每格按“**MSE减小% / MAE减小%**”排列。', '',
          '|模型|V1|V2|V3全量收敛|V4|',
          '|---|---:|---:|---:|---:|']
for model in MODELS:
    cells=[]
    for v in core:
        m=next(m for m in v['models'] if m['model']==model)['historical']
        cells.append(f"{m['mse_decrease_pct']:+.2f} / {m['mae_decrease_pct']:+.2f}")
    lines.append('|'+model+'|'+'|'.join(cells)+'|')
lines += ['', '按这组保存的实际结果，核心版本中TaTS的两项平均相对降幅均以V2最大，MM-TSFlib与SpecTF为V1，CFA与Aurora为V3。这里排名按36项相对变化宏平均，不是不同领域绝对误差的混合均值。它是结果层面的比较，不能解释成各版本只有插件结构不同。', '',
          '## 3. 与各轮匹配control比较：模块的额外收益','',
          '|版本|平均MSE减小%|平均MAE减小%|五模型平均双下降|任务双改善/180|至少一项退化/180|双持平/180|',
          '|---|---:|---:|---:|---:|---:|---:|']
for v in core:
    t=v['total']['control']
    lines.append(f"|{v['name']}|{t['mse_decrease_pct']:+.2f}|{t['mae_decrease_pct']:+.2f}|{t['both_means_better_models']}/5|{t['both_better']}|{t['any_worse']}|{t['unchanged']}|")
lines += ['', '这组结果表明：后续内部融合在单模型上有收益，但没有取得跨五模型一致的平均双指标改善。V2相对历史baseline的MAE优势，不能直接当成插件相对匹配control的优势。', '',
          '### 各模型相对匹配control的变化','',
          '|模型|V1|V2|V3全量收敛|V4|',
          '|---|---:|---:|---:|---:|']
for model in MODELS:
    cells=[]
    for v in core:
        m=next(m for m in v['models'] if m['model']==model)['control']
        cells.append(f"{m['mse_decrease_pct']:+.2f} / {m['mae_decrease_pct']:+.2f}")
    lines.append('|'+model+'|'+'|'.join(cells)+'|')
lines += ['', 'Aurora是对照差异尤其明显的例子：V3相对历史原版MSE减小21.99%、MAE减小18.24%，但相对本轮匹配微调control只减小0.96%的MSE，MAE反而增加0.26%。历史Aurora主要采用零样本推理，后续版本允许末层和预测头微调，因此不能将21.99%的下降全部归功于插件。', '',
          'V4的SpecTF相对匹配control改善6.38%/6.42%，相对历史baseline却是−4.55%/−3.28%。说明新模块确实改善了该轮control，但该轮整个配置仍未超越最初的保存结果。', '',
          '## 4. 版本对应与实验范围','',
          '|名称|实现方式|完整任务|种子口径|收敛/预算说明|',
          '|---|---|---:|---|---|',
          '|CARMA|历史残差与文本条件校正，验证决定是否启用|180|单次结果|不据仅完成标记声称全部充分收敛|',
          '|GANF共享初版|语义条件图流，共享后置修正|180|单次结果|保守修正，覆盖率较低|',
          '|GANF单模型校准|共享版基础上分别校准|180|单次结果|167/180项额外校准的最佳轮次为0|',
          '|V1|锁定no_variance_shrink；冻结底模预测，共享DCT残差图流，验证选择alpha|180|三个插件种子，底模预测固定|108次插件拟合，19次触及80轮且未达到验证平台期|',
          '|V2|内部事件融合与条件残差分布，五模型分别训练|180|单种子2026|原版和插件360次训练；有control触及预算上限|',
          '|V3全量收敛|实验室版内部融合，匹配control，完整验证平台期训练|180|单种子2026|360次训练/评估，完成标记显示均满足验证平台期|',
          '|V4|条件混合分布、完整轨迹flow及分布风险决策，原生内部调制|180|单种子2026|360次配对训练/评估，最终结果包含可恢复续训，均满足验证平台期|', '',
          '本文V1指用户截图对应的`no_variance_shrink`赢家，原代码实际存放在`plugins/semflow_tuning_v2`，不要仅凭文件夹中的v2将其误认为本文的V2内部融合。', '',
          'V1采用2026、2027、2028三个插件种子。本次表使用2026-10-06最新一次实验室独立复跑；历史AutoDL及前一次实验室复跑与该结果在两位小数汇总上相同。重复使用同三个种子的复跑不能当作增加了独立随机种子数。', '',
          '以下不加入正式全量排名：', '',
          '- Agriculture/12和Economy/10的10项精简实验：只有两个领域，且两组达到60轮上限，不能外推九领域。',
          '- AutoDL迁移前部分任务、四模型中间下载结果：是后续完整版本的子集，不当作独立完整实验。',
          '- V1参数筛选的numeric_ablation、combined等：主要是验证集筛选/消融，未各自完成同等180项三种子测试；不能把验证排名与正式测试结果混排。',
          '- 最初尚未对齐ReadGPT数据的baseline：数据接口不同，不用于本次模块增益比较。', '',
          '## 5. 为什么更复杂的后续版本没有全面超过V1','',
          '以下将实现事实与需要验证的解释分开。', '',
          '1. **V1保护了底模预测。**冻结底模、有限DCT修正及验证alpha回退都是代码机制。V1的104项持平主要源于保留原预测，而不是每项插件都有效。V2至V4没有同样的逐任务原版回退，完整结果暴露了更多退化。',
          '2. **后续训练改变了对照本身。**数据/文本接口、微调预算、主干训练和收敛要求变化，使新control与历史原版不同。Aurora的例子直接显示额外训练带来的收益不能全记作插件贡献。',
          '3. **内部改动影响了已经有效的原生通路。**V2设计记录确认CFA曾关闭原生pooled-text adapter，SpecTF同时改变频带读取，后续版本再恢复或调整。这些不是只有增加一个模块的单因素实验。',
          '4. **分布拟合与点误差不是同一目标。**V4显式从分布输出，但NLL、CRPS与MSE/MAE联合优化并不能自动同时降低两种点误差。现有结果没有单因素消融，因此不能断言某个损失权重是唯一原因。',
          '5. **V1也未证明额外语义贡献。**原参数实验的匹配验证消融中，真实文本MSE变化−2.28%、MAE变化−1.32%，无文本为−2.72%/−1.67%；真实文本没有优于无文本。V1可能主要获益于数值残差校准，不能用较好的误差结果替代深度多模态融合的证据。', '',
          '## 6. 应当怎样选择下一步基准','',
          '- 若目标是当前五模型都取得平均双指标改善，以V1作为效果参照最合适。',
          '- 若目标是最低实际MAE，不能忽略V2；若只研究TaTS、CFA或Aurora，不能仅因V1整体更均衡而删除后续版本。',
          '- V4在SpecTF相对匹配control上有积极结果，值得保留作独立分支；当前证据不支持给五模型统一采用V4。',
          '- 下一轮必须统一原版起始权重、输入、训练预算、种子、验证选择方式，并同时保留“直接插件”与“验证选择后”结果。否则再次出现对照变动，难以判断模块本身是否更好。',
          '- 已反复使用的测试集结果用于探索性汇总；最终定版应在未参与设计的新时间段或独立数据上确认。', '',
          '## 7. 原始结果来源','',
          '下列文件各提供180项结果；同目录的`版本汇总数据.json`保存重算后的逐任务、逐模型及整体数值。', '']
for v in versions:
    lines.append(f"- **{v['name']}**：`{v['source']}`。")
lines += ['', '原说明中的符号有两种，本文件均转换为正数改善。V1结果与其`MODEL_SUMMARY.json`核对；V3与`SUMMARY.json`核对。V4原文件没有总汇总表，本次由180项`control/internal/historical_original`数值重新计算。', '']
(out/'全部实验版本对比汇总.md').write_text('\n'.join(lines),encoding='utf-8')
with (out/'模型与版本对比.csv').open('w',newline='',encoding='utf-8-sig') as f:
    fields=['version','model','reference','mse_decrease_pct','mae_decrease_pct','both_better','any_worse','unchanged','mse_better','mae_better']
    w=csv.DictWriter(f,fieldnames=fields)
    w.writeheader()
    for v in versions:
        for m in v['models']:
            for reference in ['historical','control']:
                w.writerow(dict(version=v['name'],model=m['model'],reference=reference,**m[reference]))
print(json.dumps(dict(output=str(out),version_count=len(versions),records=sum(len(v['observations']) for v in versions),
                      historical_summary=[dict(version=v['name'],**v['total']['historical']) for v in versions]),ensure_ascii=False,indent=2))
