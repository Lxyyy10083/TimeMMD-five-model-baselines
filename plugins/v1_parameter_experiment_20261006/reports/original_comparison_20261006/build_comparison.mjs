import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

const base=path.resolve(process.argv[2]);
const out=path.join(base,'outputs/v1_p1_original_comparison_20261006');
const input=JSON.parse(await fs.readFile(path.join(out,'comparison_inputs.json'),'utf8'));
const wb=Workbook.create();
const summary=wb.worksheets.add('模型实际误差汇总');
const detail=wb.worksheets.add('180项原版与插件');
const navy='#213C57',body='#243447',font='Arial';
const pct='[>=0.00000005]0.0000%;[<=-0.00000005]-0.0000%;"0.0000%"';
function setup(s,range){
 s.showGridLines=false;s.getRange(range).format.font={name:font,size:10,color:body};
 s.getRange(range).format.rowHeight=24;s.getRange(range).format.columnWidth=18;
 s.getRange(range).format.verticalAlignment='center';
}
function title(s,cell,label){s.getRange(cell).values=[[label]];s.getRange(cell).format.font={name:font,size:14,bold:true,color:navy};}
function header(s,range,labels){
 s.getRange(range).values=[labels];s.getRange(range).format.fill=navy;
 s.getRange(range).format.font={name:font,size:10,bold:true,color:'#FFFFFF'};
 s.getRange(range).format.horizontalAlignment='center';s.getRange(range).format.wrapText=true;
 s.getRange(range).format.rowHeight=40;
 s.getRange(range).format.borders={insideVertical:{style:'thin',color:'#FFFFFF'}};
}
function heat(r){r.setNumberFormat(pct);r.conditionalFormats.add('colorScale',{colors:['#E9A2A2','#FFFFFF','#6DC391'],thresholds:[-.1,0,.1]});}
setup(summary,'A1:T38');setup(detail,'A1:K185');summary.tabColor=navy;
summary.getRange('A1:A38').format.columnWidth=17;
summary.getRange('B1:C38').format.columnWidth=20;
summary.getRange('D1:E38').format.columnWidth=22;
summary.getRange('F1:I38').format.columnWidth=13;
summary.getRange('J1:J38').format.columnWidth=3;
detail.getRange('A1:A185').format.columnWidth=19;
detail.getRange('B1:B185').format.columnWidth=9;
detail.getRange('C1:C185').format.columnWidth=17;
detail.getRange('D1:I185').format.columnWidth=21;
detail.getRange('J1:J185').format.columnWidth=14;
detail.getRange('K1:K185').format.columnWidth=22;

title(detail,'A2','原始 baseline 与当前 V1-P1 模块：180项直接对比');
detail.getRange('A3').values=[['原始列=无插件冻结预测；模块列=当前V1-P1三种子指标均值。下降率=(原始−模块)/原始，正数更好。']];
detail.getRange('A4').values=[['来源：baseline/V1参数实验_20261006/RESULTS_180.json；原始值与冻结无插件参考逐项一致。启用0个种子时保留原版输出。']];
detail.getRange('A3:A4').format.font={name:font,size:10,color:'#586A7B'};
const headers=['领域','预测步长','模型','原始baseline MSE','添加模块MSE','MSE下降率','原始baseline MAE','添加模块MAE','MAE下降率','启用种子数/3','双指标结果'];
header(detail,'A5:K5',headers);
detail.getRange('A6:K185').values=input.rows.map(r=>[r.domain,r.horizon,r.model,r.baseline_mse,r.mse,null,r.baseline_mae,r.mae,null,r.enabled_seeds,null]);
detail.getRange('F6').formulas=[['=(D6-E6)/D6']];detail.getRange('F6:F185').fillDown();
detail.getRange('I6').formulas=[['=(G6-H6)/G6']];detail.getRange('I6:I185').fillDown();
detail.getRange('K6').formulas=[['=IF(AND(F6>0.0000001,I6>0.0000001),"双指标下降",IF(OR(F6<-0.0000001,I6<-0.0000001),"至少一项增加","持平或单项下降"))']];
detail.getRange('K6:K185').fillDown();
detail.getRange('D6:E185').setNumberFormat('0.000000');detail.getRange('G6:H185').setNumberFormat('0.000000');
detail.getRange('D6:J185').format.horizontalAlignment='right';
detail.getRange('D6:D185').format.font.color='#586A7B';detail.getRange('G6:G185').format.font.color='#586A7B';
detail.getRange('E6:E185').format.font.color='#186598';detail.getRange('H6:H185').format.font.color='#186598';
heat(detail.getRange('F6:F185'));heat(detail.getRange('I6:I185'));
detail.freezePanes.freezeRows(5);detail.freezePanes.freezeColumns(3);
detail.tables.add('A5:K185',true,'OriginalModuleComparison');header(detail,'A5:K5',headers);

title(summary,'A2','无插件原版与当前模块：实际 MSE 对比');
summary.getRange('A3').values=[['每模型36项=九领域×四步长。B/C列为MSE简单均值；D列由这两个均值计算下降率。']];
summary.getRange('A4').values=[['E列为逐任务下降率宏平均，口径不同。不同领域误差量级差异大，请同时查看180项明细。']];
header(summary,'A5:I5',['模型','原始MSE均值','模块MSE均值','均值MSE下降率','逐任务下降率均值','MSE下降任务','MSE增加任务','MSE持平任务','任务数']);
title(summary,'A14','MAE 同口径补充');
header(summary,'A15:I15',['模型','原始MAE均值','模块MAE均值','均值MAE下降率','逐任务下降率均值','MAE下降任务','MAE增加任务','MAE持平任务','任务数']);
for(let i=0;i<5;i++){
 const n=i+6,z=i+16,m=input.models[i];
 for(const [row,baseCol,newCol,pctCol] of [[n,'D','E','F'],[z,'G','H','I']]){
  summary.getRange(`A${row}`).values=[[m]];
  summary.getRange(`B${row}:I${row}`).formulas=[[
   `=AVERAGEIFS('180项原版与插件'!$${baseCol}$6:$${baseCol}$185,'180项原版与插件'!$C$6:$C$185,$A${row})`,
   `=AVERAGEIFS('180项原版与插件'!$${newCol}$6:$${newCol}$185,'180项原版与插件'!$C$6:$C$185,$A${row})`,
   `=(B${row}-C${row})/B${row}`,
   `=AVERAGEIFS('180项原版与插件'!$${pctCol}$6:$${pctCol}$185,'180项原版与插件'!$C$6:$C$185,$A${row})`,
   `=COUNTIFS('180项原版与插件'!$C$6:$C$185,$A${row},'180项原版与插件'!$${pctCol}$6:$${pctCol}$185,">0.0000001")`,
   `=COUNTIFS('180项原版与插件'!$C$6:$C$185,$A${row},'180项原版与插件'!$${pctCol}$6:$${pctCol}$185,"<-0.0000001")`,
   `=I${row}-F${row}-G${row}`,
   `=COUNTIFS('180项原版与插件'!$C$6:$C$185,$A${row})`
  ]];
 }
}
for(const r of ['B6:C10','B16:C20'])summary.getRange(r).setNumberFormat('0.000000');
for(const r of ['D6:E10','D16:E20'])heat(summary.getRange(r));
summary.getRange('A23').values=[['Aurora：均值MSE略增，逐任务下降率均值为正；两者不是同一种统计量，不能互相替代。']];
summary.getRange('A24').values=[['绿色=误差下降；红色=误差增加。插件采用验证alpha回退策略，不依据测试误差选择启用。']];
for(const [range,start,end,t] of [['A5:C10','K5','T19','实际MSE均值：原始与模块'],['A15:C20','K22','T36','实际MAE均值：原始与模块']]){
 const c=summary.charts.add('bar',summary.getRange(range));c.setPosition(start,end);c.title=t;
 c.titleTextStyle.typeface=font;c.titleTextStyle.fontSize=12;
 c.legend={position:'top',textStyle:{typeface:font,fontSize:10}};
 c.xAxis={axisType:'textAxis',textStyle:{typeface:font,fontSize:10}};
 c.yAxis={numberFormatCode:'0.00',numberFormatSourceLinked:false,textStyle:{typeface:font,fontSize:10}};
 c.series.items[0].fill='#8F9DAB';c.series.items[1].fill='#367FB5';
}
wb.recalculate();
const computed=summary.getRange('B6:E10').values;
for(let i=0;i<5;i++){
 const s=input.summary[i],want=[s.original_mse_mean,s.module_mse_mean,s.mse_mean_decrease_pct/100,s.mse_task_macro_decrease_pct/100];
 for(let j=0;j<4;j++)if(Math.abs(computed[i][j]-want[j])>1e-10)throw Error(`Summary mismatch ${i} ${j}`);
}
for(let i of [0,90,179]){
 const actual=detail.getRange(`F${i+6}`).values[0][0];
 if(Math.abs(actual-input.rows[i].mse_decrease_pct/100)>1e-10)throw Error('Detail mismatch');
}
console.log((await wb.inspect({kind:'table',range:'模型实际误差汇总!A5:I10',include:'values,formulas',tableMaxRows:6,tableMaxCols:9,maxChars:2200})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},summary:'Formula errors',maxChars:1200})).ndjson);
for(const [sheetName,range,name] of [['模型实际误差汇总','A1:T37','preview_summary.png'],['180项原版与插件','A1:K15','preview_detail.png']]){
 const p=await wb.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(out,name),new Uint8Array(await p.arrayBuffer()));
}
const x=await SpreadsheetFile.exportXlsx(wb);await x.save(path.join(out,'原始baseline_vs_V1P1模块_MSE_MAE.xlsx'));
console.log('EXPORTED',out);
