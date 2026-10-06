import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const base = path.resolve(process.argv[2]);
const resultDir = path.join(base, 'GANF_V1_MSE_Excel_rerun2_20261006');
const input = JSON.parse(await fs.readFile(path.join(resultDir, 'workbook_inputs.json'), 'utf8'));
const out = path.join(base, 'outputs', 'v1_mse_rerun2_20261006');
await fs.mkdir(out, {recursive: true});
const rows = input.rows, models = input.models, domains = input.domains;
if(rows.length !== 540 || !input.launch.fresh_fits) throw Error('Require a completed fresh rerun');
const wb = Workbook.create();
const view = wb.worksheets.add('MSE图表');
const detail = wb.worksheets.add('180项对比');
const source = wb.worksheets.add('540条种子');
const navy = '#213C57', text = '#243447', gray = '#F3F5F7';
const colors = ['#3779B5', '#8061AF', '#D69038', '#329E9A', '#BC5576'];
// Suppress a misleading minus sign on changes that round to zero at two decimals.
const ratioFormat = '[>=0.00005]0.00%;[<=-0.00005]-0.00%;"0.00%"';
function setup(sheet, range) {
  sheet.showGridLines = false;
  sheet.getRange(range).format.font = {name: 'Arial', size: 10, color: text};
  sheet.getRange(range).format.rowHeight = 23;
  sheet.getRange(range).format.columnWidth = 15;
  sheet.getRange(range).format.verticalAlignment = 'center';
}
function title(sheet, cell, label) {
  sheet.getRange(cell).values = [[label]];
  sheet.getRange(cell).format.font = {name:'Arial', size: 14, bold:true, color:navy};
}
function header(sheet, range, values) {
  sheet.getRange(range).values = [values];
  sheet.getRange(range).format.fill = navy;
  sheet.getRange(range).format.font = {name:'Arial', size:10, bold:true, color:'#FFFFFF'};
  sheet.getRange(range).format.horizontalAlignment = 'center';
  sheet.getRange(range).format.borders = {insideVertical:{style:'thin', color:'#FFFFFF'}};
  sheet.getRange(range).format.rowHeight = 27;
}
function heat(range) {
  range.setNumberFormat(ratioFormat);
  range.conditionalFormats.add('colorScale', {
    colors:['#E9A2A2','#F4F5F7','#6DC391'], thresholds:[-0.1,0,0.1]
  });
}
function mseHeat(range, min, max) {
  range.setNumberFormat('0.000000');
  range.conditionalFormats.add('colorScale', {
    colors:['#D2EAD9','#F3EAC9','#F0C8C5'], thresholds:[min,(min+max)/2,max]
  });
}

setup(view, 'A1:N302');
setup(detail, 'A1:L185');
setup(source, 'A1:Q545');
view.tabColor = navy;
detail.tabColor = '#859BAE';
view.getRange('G1:G302').format.columnWidth = 3;
view.getRange('A1:A302').format.columnWidth = 18;
view.getRange('H1:H302').format.columnWidth = 18;
detail.getRange('A1:A185').format.columnWidth = 18;
detail.getRange('B1:B185').format.columnWidth = 8;
detail.getRange('C1:C185').format.columnWidth = 15;
detail.getRange('F1:F185').format.columnWidth = 18;
detail.getRange('J1:J185').format.columnWidth = 17;
detail.getRange('K1:K185').format.columnWidth = 11;
detail.getRange('L1:L185').format.columnWidth = 18;
source.getRange('A1:A545').format.columnWidth = 18;
source.getRange('B1:D545').format.columnWidth = 10;
source.getRange('E1:G545').format.columnWidth = 17;
source.getRange('H1:K545').format.columnWidth = 12;
source.getRange('L1:Q545').format.columnWidth = 17;

title(source, 'A2', 'V1 新复跑原始种子记录');
source.getRange('A3').values = [['来源：本次独立复跑 test_seed_results.csv 与 fit_result.json；插件种子 2026、2027、2028。']];
source.getRange('A4').values = [[`V1 源码 b5e4d91；108 次全新拟合；原版预测固定；输入 SHA256 校验；结果归档 SHA256 ${input.archive_sha256}`]];
header(source, 'A5:Q5', ['领域','预测步长','模型','插件种子','原版MSE','直接插件MSE','选择后MSE','验证alpha','最佳轮次','运行轮次','验证平台期','原版MAE','直接插件MAE','选择后MAE','测试窗口数','验证MSE比','验证MAE比']);
source.getRange('A6:Q545').values = rows.map(r => [r.domain,r.horizon,r.model,r.seed,r.baseline_mse,r.raw_mse,r.plugin_mse,r.alpha,r.best_epoch,r.epochs_run,r.validation_plateau?'已达到':'未达到',r.baseline_mae,r.raw_mae,r.plugin_mae,r.test_windows,r.holdout_mse_ratio,r.holdout_mae_ratio]);
source.getRange('E6:G545').setNumberFormat('0.000000');
source.getRange('L6:N545').setNumberFormat('0.000000');
source.getRange('P6:Q545').setNumberFormat('0.000000');
source.getRange('H6:H545').setNumberFormat('0.00');
source.freezePanes.freezeRows(5);
source.freezePanes.freezeColumns(3);
source.tables.add('A5:Q545',true,'SeedObservations');

title(detail, 'A2', '180 项 MSE 对比');
detail.getRange('A3').values = [['改善率 =（原版MSE − V1三种子平均MSE）/ 原版MSE；正值改善，负值退化。']];
detail.getRange('A4').values = [['直接插件为未经过验证alpha回退的输出；三种子平均为指标平均，非预测集成。状态阈值为 ±0.00001%。']];
header(detail, 'A5:L5', ['领域','步长','模型','原版MSE','V1平均MSE','V1种子标准差','MSE改善率','直接插件MSE','直接插件改善率','有效插件种子数','MSE状态','验证选用alpha均值']);
const index = new Map();
const expected = [];
for(let i=0; i<180; i++) {
  const group=rows.slice(i*3,i*3+3),r=group[0],n=i+6,s=i*3+6,e=s+2;
  const key=`${r.domain}|${r.horizon}|${r.model}`;
  index.set(key,n);
  detail.getRange(`A${n}:C${n}`).values = [[r.domain,r.horizon,r.model]];
  detail.getRange(`D${n}:L${n}`).formulas = [[
    `='540条种子'!E${s}`,
    `=AVERAGE('540条种子'!G${s}:G${e})`,
    `=STDEV.S('540条种子'!G${s}:G${e})`,
    `=(D${n}-E${n})/D${n}`,
    `=AVERAGE('540条种子'!F${s}:F${e})`,
    `=(D${n}-H${n})/D${n}`,
    `=COUNTIFS('540条种子'!H${s}:H${e},">0",'540条种子'!I${s}:I${e},">0")`,
    `=IF(G${n}>0.0000001,"改善",IF(G${n}<-0.0000001,"退化","持平"))`,
    `=AVERAGE('540条种子'!H${s}:H${e})`
  ]];
  const mean=group.reduce((a,x)=>a+x.plugin_mse,0)/3, raw=group.reduce((a,x)=>a+x.raw_mse,0)/3;
  const ratio=(r.baseline_mse-mean)/r.baseline_mse;
  expected.push({domain:r.domain,horizon:r.horizon,model:r.model,baseline:r.baseline_mse,mean,raw,ratio,
    status:ratio>1e-7?'改善':ratio< -1e-7?'退化':'持平',row:n});
}
detail.getRange('D6:F185').setNumberFormat('0.000000');
detail.getRange('H6:H185').setNumberFormat('0.000000');
detail.getRange('L6:L185').setNumberFormat('0.00');
heat(detail.getRange('G6:G185'));
heat(detail.getRange('I6:I185'));
detail.getRange('K6:K185').conditionalFormats.add('containsText',{text:'退化',format:{fill:'#F4D8D8',font:{color:'#A12F35'}}});
detail.getRange('K6:K185').conditionalFormats.add('containsText',{text:'改善',format:{fill:'#DFF0E6',font:{color:'#246446'}}});
detail.freezePanes.freezeRows(5);
detail.freezePanes.freezeColumns(3);
detail.tables.add('A5:L185',true,'CaseComparison');

title(view,'A2','V1 九领域五模型 MSE 对比');
view.getRange('A3').values = [['2026-10-06 实验室全新复跑；9 领域 × 4 步长 × 5 模型；每项为 3 个插件种子的指标平均。']];
view.getRange('A4').values = [['正百分比表示误差减小；绿色改善，红色退化，灰色持平。绝对MSE越小越好。']];
// G is a spacer throughout the output sheet; no heading, fill, or border crosses it.
header(view, 'A6:F6', ['模型','任务数','MSE下降','MSE上升','MSE持平','平均改善率']);
header(view, 'H6:J6', ['模型','直接插件改善率','选用后改善率']);
for(let i=0;i<5;i++) {
  const n=i+7,m=models[i];
  view.getRange(`A${n}`).values=[[m]];
  view.getRange(`B${n}:F${n}`).formulas=[[
    `=COUNTIF('180项对比'!$C$6:$C$185,A${n})`,
    `=COUNTIFS('180项对比'!$C$6:$C$185,A${n},'180项对比'!$K$6:$K$185,"改善")`,
    `=COUNTIFS('180项对比'!$C$6:$C$185,A${n},'180项对比'!$K$6:$K$185,"退化")`,
    `=COUNTIFS('180项对比'!$C$6:$C$185,A${n},'180项对比'!$K$6:$K$185,"持平")`,
    `=AVERAGEIF('180项对比'!$C$6:$C$185,A${n},'180项对比'!$G$6:$G$185)`
  ]];
  view.getRange(`H${n}:J${n}`).formulas=[[
    `=A${n}`,
    `=AVERAGEIF('180项对比'!$C$6:$C$185,A${n},'180项对比'!$I$6:$I$185)`,
    `=F${n}`
  ]];
}
view.getRange('A12').values=[['总计']];
view.getRange('B12:E12').formulas=[['=SUM(B7:B11)','=SUM(C7:C11)','=SUM(D7:D11)','=SUM(E7:E11)']];
view.getRange('F12').formulas=[['=AVERAGE(\'180项对比\'!G6:G185)']];
view.getRange('A12:F12').format.fill=gray;
view.getRange('A12:F12').format.font.bold=true;
heat(view.getRange('F7:F12'));
heat(view.getRange('I7:J11'));
view.getRange('A14').values=[['平均改善率是 36 项相对改善率的算术平均。不同领域绝对MSE的量级不同，不直接混合比较。']];
view.getRange('A15').values=[['并非所有任务下降：非零alpha仅要求验证集MSE与MAE同时改善，不能保证测试集也改善。']];
view.getRange('A16').values=[['alpha=0 保留原版预测。种子页提供alpha、最佳轮次、验证误差比，可核对选择规则。']];
view.getRange('A17').values=[['底模使用原版冻结预测；重新训练的是共享后置V1插件。归一化和指标尺度沿用原版缓存。']];
const capped = input.convergence.capped_without_validation_plateau.length;
view.getRange('A18').values=[[`沿用截图版80轮预算；${capped}/108次拟合达到上限且未满足验证平台期，不能称为全部充分收敛。`]];
view.getRange('A20').values=[['领域MSE表：原版与V1使用相同的领域内热力刻度，绿低红高。改善率热力刻度固定为 −10% / 0 / +10%。']];
view.getRange('A21').values=[['每张折线图展示该领域五个模型的V1平均MSE；原版数值并列列出。所有预测步长均显示真实长度。']];
view.getRange('A14:A21').format.font.size=10;

const chartMeta=[];
for(let di=0;di<domains.length;di++) {
  const d=domains[di],start=26+di*30;
  const hs=[...new Set(expected.filter(r=>r.domain===d).map(r=>r.horizon))].sort((a,b)=>a-b);
  title(view,`A${start}`,`${d}  MSE`);
  view.getRange(`A${start+1}`).values=[['V1 平均MSE']];
  view.getRange(`H${start+1}`).values=[['原版 MSE']];
  header(view,`A${start+2}:F${start+2}`,['步长',...models]);
  header(view,`H${start+2}:M${start+2}`,['步长',...models]);
  const vals=expected.filter(r=>r.domain===d).flatMap(r=>[r.baseline,r.mean]);
  const min=Math.min(...vals),max=Math.max(...vals);
  for(let hi=0;hi<4;hi++) {
    const n=start+3+hi;
    view.getRange(`A${n}`).values=[[`${hs[hi]}步`]];
    view.getRange(`H${n}`).values=[[`${hs[hi]}步`]];
    view.getRange(`B${n}:F${n}`).formulas=[models.map(m=>`='180项对比'!E${index.get(`${d}|${hs[hi]}|${m}`)}`)];
    view.getRange(`I${n}:M${n}`).formulas=[models.map(m=>`='180项对比'!D${index.get(`${d}|${hs[hi]}|${m}`)}`)];
  }
  mseHeat(view.getRange(`B${start+3}:F${start+6}`),min,max);
  mseHeat(view.getRange(`I${start+3}:M${start+6}`),min,max);
  view.getRange(`A${start+9}`).values=[['MSE 改善率（正值改善）']];
  header(view,`A${start+10}:F${start+10}`,['步长',...models]);
  for(let hi=0;hi<4;hi++) {
    const n=start+11+hi;
    view.getRange(`A${n}`).values=[[`${hs[hi]}步`]];
    view.getRange(`B${n}:F${n}`).formulas=[models.map(m=>`='180项对比'!G${index.get(`${d}|${hs[hi]}|${m}`)}`)];
  }
  heat(view.getRange(`B${start+11}:F${start+14}`));
  view.getRange(`A${start+17}`).values=[['红色负值 = 测试集MSE增加。']];
  view.getRange(`A${start+18}`).values=[['持平主要来自验证alpha=0的回退。']];
  const chart = view.charts.add('line',view.getRange(`A${start+2}:F${start+6}`));
  chart.title=`${d}  V1 MSE`;
  chart.titleTextStyle.fontSize=14;
  chart.titleTextStyle.typeface='Arial';
  chart.legend={position:'top',textStyle:{typeface:'Arial',fontSize:10}};
  chart.xAxis={axisType:'textAxis',textStyle:{typeface:'Arial',fontSize:10}};
  chart.yAxis={numberFormatCode:max>10?'0.0':'0.000',numberFormatSourceLinked:false,textStyle:{typeface:'Arial',fontSize:10}};
  chart.setPosition(`H${start+9}`,`N${start+27}`);
  if(chart.series.items.length!==5) throw Error('Expected five line series for '+d);
  chart.series.items.forEach((s,i)=>{s.line={fill:colors[i],style:'solid',width:2};});
  chartMeta.push({domain:d,source:`A${start+2}:F${start+6}`,series:chart.series.items.map(s=>({values:s.formula,categories:s.categoryFormula}))});
}
wb.recalculate();
// Verify the consequential calculations against independently computed measured data.
const actual=detail.getRange('D6:L185').values;
expected.forEach((r,i)=> {
  for(const [j,v] of [[0,r.baseline],[1,r.mean],[3,r.ratio],[4,r.raw]]) {
    if(typeof actual[i][j]!=='number' || Math.abs(actual[i][j]-v)>1e-9*Math.max(1,Math.abs(v)))
      throw Error(`Formula mismatch row ${i+6} column ${j}: ${actual[i][j]} vs ${v}`);
  }
  if(actual[i][7]!==r.status) throw Error('Status mismatch');
});
const counts={improved:expected.filter(r=>r.status==='改善').length,worse:expected.filter(r=>r.status==='退化').length,unchanged:expected.filter(r=>r.status==='持平').length};
const totals=view.getRange('C12:E12').values[0];
if(totals.join(',')!==[counts.improved,counts.worse,counts.unchanged].join(',')) throw Error('Summary counts mismatch');
console.log((await wb.inspect({kind:'region',sheetId:'MSE图表',range:'A6:J12',maxChars:2600,tableMaxRows:7,tableMaxCols:10})).ndjson);
for(const [sheet,range] of [[view,'A1:N293'],[detail,'A1:L185'],[source,'A1:Q545']]) {
  const data=sheet.getRange(range).values;
  for(const row of data) for(const cell of row) if(typeof cell==='string' && /^#(REF!|DIV\/0!|VALUE!|N\/A|NAME\?|NUM!|SPILL!|CALC!)/.test(cell)) throw Error('Formula error '+cell);
}
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#SPILL!',options:{useRegex:true,maxResults:20},maxChars:1000})).ndjson);
const renderTargets=[['MSE图表','A1:N22','summary'],['180项对比','A1:L19','detail'],['540条种子','A1:Q15','seeds'],...domains.map((d,i)=>['MSE图表',`A${26+i*30}:N${54+i*30}`,d])];
for(const [sheetName,range,name] of renderTargets) {
  const png=await wb.render({sheetName,range,scale:1,format:'png'});
  await fs.writeFile(path.join(out,`preview_${name}.png`),new Uint8Array(await png.arrayBuffer()));
  console.log('RENDERED',name);
}
const file=path.join(out,'V1_九领域五模型_MSE对比.xlsx');
await (await SpreadsheetFile.exportXlsx(wb)).save(file);
const summary=models.map(m=> {
  const group=expected.filter(r=>r.model===m);
  return {model:m,improvement_pct:group.reduce((a,r)=>a+r.ratio,0)/36*100,improved:group.filter(r=>r.status==='改善').length,worse:group.filter(r=>r.status==='退化').length,unchanged:group.filter(r=>r.status==='持平').length};
});
const review={counts,models:summary,charts:chartMeta,capped_without_plateau:capped,workbook:file,source_commit:input.source_commit,launch:input.launch};
await fs.writeFile(path.join(resultDir,'Excel结果核对.json'),JSON.stringify(review,null,2));
await fs.copyFile(file,path.join(resultDir,path.basename(file)));
console.log('EXPORTED',JSON.stringify(review));
