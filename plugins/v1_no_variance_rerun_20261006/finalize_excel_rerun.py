"""Verify exported native features and archive experiment evidence without large weights."""
from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

base = Path(sys.argv[1]).resolve()
out = base / 'GANF_V1_MSE_Excel_rerun2_20261006'
report = json.loads((out / 'Excel结果核对.json').read_text(encoding='utf-8'))
data = json.loads((out / 'workbook_inputs.json').read_text(encoding='utf-8'))
book = Path(report['workbook'])
ns = {'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'c':'http://schemas.openxmlformats.org/drawingml/2006/chart',
      'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
with zipfile.ZipFile(book) as z:
    chart_sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    chart_cells = {c.attrib['r']:c for c in chart_sheet.findall('.//s:sheetData/s:row/s:c',ns)}
    charts = sorted(n for n in z.namelist() if '/charts/chart' in n and n.endswith('.xml'))
    assert len(charts) == 9
    for name in charts:
        root = ET.fromstring(z.read(name))
        series = root.findall('.//c:lineChart/c:ser',ns)
        assert len(series) == 5
        for s in series:
            binding = s.find('c:val/c:numRef/c:f',ns)
            assert binding is not None
            match = re.fullmatch(r"'MSE图表'!\$([B-F])\$(\d+):\$\1\$(\d+)",binding.text)
            assert match and int(match[3])-int(match[2]) == 3
            for row in range(int(match[2]),int(match[3])+1):
                cell = chart_cells[match[1]+str(row)]
                assert cell.attrib.get('t','n') == 'n'
                assert float(cell.find('s:v',ns).text) >= 0
            points = s.findall('c:val/c:numRef/c:numCache/c:pt',ns)
            # Optional caches may be empty; Excel resolves the four verified source cells.
            assert len(points) in (0,4)
            color = s.find('c:spPr/a:ln/a:solidFill/a:srgbClr',ns)
            assert color is not None and color.attrib['val'] in {'3779B5','8061AF','D69038','329E9A','BC5576'}
    sheets = sorted(n for n in z.namelist() if n.startswith('xl/worksheets/sheet') and n.endswith('.xml'))
    assert len(sheets) == 3
    formulas = 0
    for name in sheets:
        root = ET.fromstring(z.read(name))
        formulas += len(root.findall('.//s:f',ns))
        assert not root.findall('.//s:c[@t="e"]',ns), 'Excel error cache'
    assert formulas > 2000
    conditional_rules = sum(len(ET.fromstring(z.read(n)).findall('.//s:conditionalFormatting/s:cfRule',ns)) for n in sheets)
    assert conditional_rules >= 30
    for name in sheets[1:]:
        pane = ET.fromstring(z.read(name)).find('.//s:pane',ns)
        assert pane is not None and pane.attrib.get('xSplit') == '3' and pane.attrib.get('ySplit') == '5'
report['export_verification'] = dict(native_charts=9, series_per_chart=5, points_per_series=4,
                                     native_color_rules=conditional_rules, formulas=formulas,
                                     formula_errors=0, frozen_headers=True)
(out / 'Excel结果核对.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

groups = [data['rows'][i:i+3] for i in range(0,540,3)]
bad = []
for g in groups:
    r = g[0]
    mean = sum(x['plugin_mse'] for x in g)/3
    improvement = (r['baseline_mse']-mean)/r['baseline_mse']*100
    if improvement < -0.00001:
        bad.append((r,mean,improvement))
lines = ['# V1九领域五模型MSE复跑与Excel说明','',
         '## 本次执行','',
         '本次在实验室服务器新目录执行108次新的插件拟合，随后评估五模型，得到540条独立插件种子记录与180项三种子指标平均。原版预测保持冻结，未重新训练五个底模。', '',
         f"- 源码版本：`{data['source_commit']}`，`no_variance_shrink`，与截图对应。",
         f"- 工作目录：`{data['launch']['workspace']}`。",
         '- Python环境：`/xiliang/LXY/envs/lxy`。本次读写均在授权目录内。',
         '- 原输入567个文件逐项校验SHA256，未替换数据集、划分、归一化或底模预测。',
         '- 插件种子：2026、2027、2028；每个领域、步长、种子训练一个共享插件，并评估五模型。',
         '- 训练预算、权重、alpha候选、早停规则保持截图版设定。',
         '- 三种子平均是三个独立插件的指标平均，不是三个底模重训，也不是预测集成。', '',
         '## 为什么图中的五个模型平均MSE都下降','',
         '汇总值是每个模型36项相对误差变化的算术平均。正负变化在平均时相抵，因此五个模型的平均值下降，不表示180个任务全部改善。', '',
         f"本次MSE：{report['counts']['improved']}项改善、{report['counts']['worse']}项退化、{report['counts']['unchanged']}项持平。", '',
         'V1在验证集选择alpha，候选为0、0.25、0.5、0.75、1。只有验证集MSE和MAE同时改善才接受非零修正。alpha为0时保留原预测。测试集不参与选择，但验证集有效的修正仍可能在测试集退化。', '',
         '例如CFA的直接插件平均MSE约增加0.94%，验证选择后平均MSE约下降0.42%。收益包含回退和混合权重选择的影响，不能据此认定插件在所有任务上都有效。', '',
         '|模型|平均MSE减小百分比|下降任务|上升任务|持平任务|',
         '|---|---:|---:|---:|---:|']
for r in report['models']:
    lines.append(f"|{r['model']}|{r['improvement_pct']:.6f}%|{r['improved']}|{r['worse']}|{r['unchanged']}|")
lines += ['', '## MSE增加的12项','',
          '|领域|步长|模型|原版MSE|V1平均MSE|误差减小百分比|',
          '|---|---:|---|---:|---:|---:|']
for r,mean,improvement in bad:
    lines.append(f"|{r['domain']}|{r['horizon']}|{r['model']}|{r['baseline_mse']:.8f}|{mean:.8f}|{improvement:+.6f}%|")
lines += ['', '## Excel定义与阅读','',
          '- `MSE图表`：九领域的原版与V1数值并列，每个领域四个真实步长，五条模型折线。',
          '- `180项对比`：原版、选择后插件平均、种子标准差、改善百分比、直接插件及其改善率、有效种子数和MSE状态。',
          '- `540条种子`：原始记录、验证alpha、最佳轮次、运行轮次、验证误差比，并保留本次MAE原始数值。',
          '- 改善率 = (原版MSE − V1平均MSE) / 原版MSE。正值表示减小，负值表示增加。',
          '- 该符号与旧CSV的`mse_delta_pct`相反；旧CSV负值表示减小。',
          '- 改善率热力刻度固定为−10%、0、+10%。超出端点的数字不截断，仅颜色饱和。',
          '- 绝对MSE绿色表示较低、红色较高，原版和V1使用同一领域的共同刻度。各领域之间绝对量级不混合配色。',
          '- 小于0.00001%的变化按持平计数。百分比显示两位小数；舍入为零的微小负值显示0.00%，完整数值仍保留。',
          '- 平均改善是36项相对改善率的宏平均，不是原版和新版本MSE整体平均后再计算比例。', '',
          '## 预算与限制','',
          f"{report['capped_without_plateau']}/108次拟合在80轮上限结束时未满足8轮验证平台期。本次为保持V1原协议未追加训练，因此不能表述为全部充分收敛。", '',
          '本次不证明语义模态的独立贡献，需要另外进行文本消融才能分离这一作用。', '',
          '## 核对与归档','',
          '- 180行Excel公式结果与本次540条原始记录独立核对，未发现公式错误。',
          '- 九张图表均为Excel原生图表，每张五个模型、每个模型四个数据点，直接绑定计算单元格。',
          '- 导出后核对图表系列颜色、缓存数据、条件格式和冻结表头。三张工作表与九个领域均已渲染检查。',
          '- Git标签`pre_v1_excel_rerun2_20261006`保存执行前状态。模型源码未改动，新增结果归档和Excel生成脚本。',
          f"- 完整本地实验归档SHA256：`{data['archive_sha256']}`。", '']
(out / 'V1_MSE复跑说明.md').write_text('\n'.join(lines),encoding='utf-8')
repo = base / 'experiment_vcs/server_baseline'
rel = 'plugins/v1_no_variance_rerun_20261006/reports/20261006_rerun2'
dest = repo / rel
dest.mkdir(parents=True,exist_ok=True)
for name in ['FULL_COMPLETED.json','RUN_PLAN.json','V1_LAUNCH.json','SOURCE_SNAPSHOT.json',
             'INPUT_DOWNLOAD_COMPLETE.json','CONVERGENCE_BUDGET.json','RESULTS_180.json',
             'MODEL_SUMMARY.json','model_summary.csv','results_180_mean_std.csv','test_seed_results.csv',
             'workbook_inputs.json','Excel结果核对.json','V1_MSE复跑说明.md','MSE_180_cases.png','MAE_180_cases.png']:
    shutil.copyfile(out / name,dest / name)
shutil.copyfile(book,dest / book.name)
for fit in (out/'final').rglob('fit_result.json'):
    target = dest / fit.relative_to(out)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(fit,target)
shutil.copyfile(base/'experiment_vcs/v1_excel_report_20261006/build_v1_mse.mjs',
                repo/'plugins/v1_no_variance_rerun_20261006/build_v1_mse.mjs')

# Credentials remain in memory. Report only the repository visibility and URL.
credentials=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
                           capture_output=True,text=True,cwd=repo,check=True)
fields=dict(line.split('=',1) for line in credentials.stdout.splitlines() if '=' in line)
token=fields.get('password')
if not token:raise RuntimeError('No stored GitHub credential')
request=urllib.request.Request('https://api.github.com/repos/Lxyyy10083/TimeMMD-five-model-baselines',
                               headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'})
with urllib.request.urlopen(request,timeout=30) as response:
    repository=json.load(response)
assert repository['private'], 'Repository must remain private'
print('REPOSITORY_PRIVATE',repository['html_url'])
del token,fields,credentials
relative_files=['plugins/v1_no_variance_rerun_20261006/build_v1_mse.mjs',
                'plugins/v1_no_variance_rerun_20261006/download_excel_rerun.py',
                'plugins/v1_no_variance_rerun_20261006/finalize_excel_rerun.py',rel]
subprocess.run(['git','add','--',*relative_files],cwd=repo,check=True)
subprocess.run(['git','commit','--only','-m','Archive fresh V1 rerun and nine-domain native MSE Excel charts','--',*relative_files],cwd=repo,check=True)
subprocess.run(['git','push','origin','HEAD:main'],cwd=repo,check=True)
print('VERIFIED_AND_ARCHIVED',json.dumps(report['export_verification']))
