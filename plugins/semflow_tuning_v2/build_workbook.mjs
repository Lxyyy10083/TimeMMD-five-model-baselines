import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const dir=process.argv[2];
if(!dir) throw new Error('Provide the result directory');
const input=JSON.parse(await fs.readFile(path.join(dir,'workbook_data.json'),'utf8'));
if(input.detail.data.length!==180 || input.seeds.data.length!==540) throw new Error('Incomplete experiment');
const outputDir=path.join(dir,'outputs','semflow_tuning_20261003');
await fs.mkdir(outputDir,{recursive:true});
const wb=Workbook.create();
const summary=wb.worksheets.add('模型汇总');
const detail=wb.worksheets.add('180项结果');
const parameters=wb.worksheets.add('参数筛选');
const source=wb.worksheets.add('540项种子数据');
function col(i){let s='';for(let n=i+1;n>0;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;}
function table(sheet,headers,rows,start=5,widths=[]){
  const last=start+rows.length;const end=col(headers.length-1);
  sheet.showGridLines=false;
  sheet.getRange(`A${start}:${end}${last}`).values=[headers,...rows];
  sheet.getRange(`A${start}:${end}${last}`).format.font={name:'Arial',size:10,color:'#202d3d'};
  sheet.getRange(`A${start}:${end}${last}`).format.verticalAlignment='center';
  sheet.getRange(`A${start}:${end}${last}`).format.rowHeight=22;
  sheet.getRange(`A${start}:${end}${start}`).format={fill:'#253e58',font:{name:'Arial',size:10,bold:true,color:'#ffffff'},
    horizontalAlignment:'center',verticalAlignment:'center',rowHeight:34,wrapText:true};
  for(let i=0;i<headers.length;i++) sheet.getRange(`${col(i)}${start}:${col(i)}${last}`).format.columnWidth=widths[i]??16;
  sheet.tables.add(`A${start}:${end}${last}`,true,`Table${sheet.name.replace(/[^A-Za-z0-9]/g,'')||start}${headers.length}`);
  sheet.freezePanes.freezeRows(start);
  return last;
}
function title(sheet,text,note){
  sheet.getRange('A2').values=[[text]];
  sheet.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#253e58'};
  sheet.getRange('A2').format.rowHeight=25;
  if(note){sheet.getRange('A3').values=[[note]];sheet.getRange('A3').format.font={name:'Arial',size:10,italic:true,color:'#5c6674'};}
}

title(detail,'九领域四步长的 MSE 与 MAE','三种子独立训练的性能均值和样本标准差；负的变化表示误差下降。');
const detailHeaders=['领域','预测步长','模型','原MSE','上一版MSE','本轮MSE均值','MSE标准差','MSE变化',
                     '原MAE','上一版MAE','本轮MAE均值','MAE标准差','MAE变化','启用种子数','双改善种子数'];
const detailRows=input.detail.data.map(row=>row.map((value,i)=>[7,12].includes(i)?null:value));
table(detail,detailHeaders,detailRows,5,[18,11,15,16,16,16,16,14,16,16,16,16,14,12,14]);
detail.getRange('D6:G185').setNumberFormat('0.000000');
detail.getRange('I6:L185').setNumberFormat('0.000000');
detail.getRange('H6').formulas=[['=(F6-D6)/D6']];detail.getRange('H6:H185').fillDown();
detail.getRange('M6').formulas=[['=(K6-I6)/I6']];detail.getRange('M6:M185').fillDown();
for(const range of ['H6:H185','M6:M185']){
  detail.getRange(range).setNumberFormat('0.00%');
  detail.getRange(range).conditionalFormats.add('cellIs',{operator:'lessThan',formula:0,format:{fill:'#e4f0ec',font:{color:'#236543'}}});
  detail.getRange(range).conditionalFormats.add('cellIs',{operator:'greaterThan',formula:0,format:{fill:'#fae8e5',font:{color:'#a13b32'}}});
}
detail.getRange('N6:O185').setNumberFormat('0');
detail.freezePanes.freezeColumns(3);

title(summary,'五模型参数实验汇总',`锁定配置：${input.winner.variant}；9领域 × 4步长 × 3种子。`);
const models=['TaTS','MM-TSFlib','SpecTF','CFA','Aurora'];
table(summary,['模型','任务数','有启用任务','MSE MAE双改善','至少一项退化','保持原预测','平均MSE变化','平均MAE变化'],
      models.map(m=>[m,null,null,null,null,null,null,null]),5,[16,11,14,17,16,15,18,18]);
const modelRange="'180项结果'!$C$6:$C$185";
const mseRange="'180项结果'!$H$6:$H$185";const maeRange="'180项结果'!$M$6:$M$185";
const alphaRange="'180项结果'!$N$6:$N$185";
summary.getRange('B6:H6').formulas=[[
  `=COUNTIF(${modelRange},A6)`,
  `=COUNTIFS(${modelRange},A6,${alphaRange},">0")`,
  `=COUNTIFS(${modelRange},A6,${mseRange},"<-0.0000001",${maeRange},"<-0.0000001")`,
  `=COUNTIFS(${modelRange},A6,${mseRange},">0.0000001")+COUNTIFS(${modelRange},A6,${maeRange},">0.0000001")-COUNTIFS(${modelRange},A6,${mseRange},">0.0000001",${maeRange},">0.0000001")`,
  `=COUNTIFS(${modelRange},A6,${mseRange},">=-0.0000001",${mseRange},"<=0.0000001",${maeRange},">=-0.0000001",${maeRange},"<=0.0000001")`,
  `=AVERAGEIF(${modelRange},A6,${mseRange})`,
  `=AVERAGEIF(${modelRange},A6,${maeRange})`,
]];
summary.getRange('B6:H10').fillDown();summary.getRange('G6:H10').setNumberFormat('0.00%');
summary.getRange('A13').values=[['平均变化为各任务相对变化的宏平均，未经原始误差大小加权。']];
summary.getRange('A14').values=[['原始候选预测及各种子完整指标见“540项种子数据”；协议与消融见结果分析。']];
summary.getRange('A13:A14').format.font={name:'Arial',size:10,color:'#5c6674'};

title(parameters,'参数筛选结果','九领域各一个步长、五模型、seed 2026 的 holdout 结果；测试集未参与排行。');
table(parameters,['配置','修正上限','初始logit','方差系数','K上限','NLL权重','语义权重','文本汇聚','模型条件','文本输入',
                   '验证MSE变化','验证MAE变化','验证综合变化'],input.parameters,5,[30,12,12,12,10,12,12,14,12,14,17,17,17]);
parameters.getRange('B6:G19').setNumberFormat('0.00');
parameters.getRange('K6:M19').setNumberFormat('0.00%');

source.showGridLines=false;
source.getRange('A1').values=[['来源：固定的 TimeMMD ReadGPT 数据、五底模预测，以及插件三个随机种子的独立评估。']];
table(source,input.seeds.columns,input.seeds.data,3,input.seeds.columns.map((v,i)=>i<5?20:18));
source.getRange('F4:U543').setNumberFormat('0.000000');
source.freezePanes.freezeColumns(3);
wb.recalculate();
const expected=input.summary;
for(let i=0;i<models.length;i++){
  const values=summary.getRange(`A${i+6}:H${i+6}`).values[0];
  const row=expected.data.find(r=>r[0]===models[i]);
  if(Number(values[1])!==36 || Number(values[3])!==row[3] || Number(values[4])!==row[4])
    throw new Error(`Summary reconciliation failed: ${models[i]} ${JSON.stringify(values)}`);
  if(Math.abs(Number(values[6])-row[6]/100)>1e-8 || Math.abs(Number(values[7])-row[7]/100)>1e-8)
    throw new Error('Metric average reconciliation failed');
}
const errorScan=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
                               options:{useRegex:true,maxResults:20},maxChars:1000});
console.log(errorScan.ndjson);
console.log((await wb.inspect({kind:'table',range:'模型汇总!A5:H10',include:'values,formulas',tableMaxRows:6,tableMaxCols:8,maxChars:2000})).ndjson);
for(const [name,range] of [['模型汇总','A1:H15'],['180项结果','A1:H12'],['参数筛选','A1:M19'],['540项种子数据','A1:H10']]){
  const preview=await wb.render({sheetName:name,range,scale:1.2});
  await fs.writeFile(path.join(dir,`preview_${name}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const output=await SpreadsheetFile.exportXlsx(wb);
const filename=path.join(outputDir,'参数实验MSE_MAE结果.xlsx');
await output.save(filename);
console.log(filename);
