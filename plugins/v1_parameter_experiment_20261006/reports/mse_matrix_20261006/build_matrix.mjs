import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

const base=path.resolve(process.argv[2]);
const inputPath=path.join(base,'V1参数实验_20261006/RESULTS_180.json');
const rows=JSON.parse(await fs.readFile(inputPath,'utf8'));
const models=['TaTS','MM-TSFlib','SpecTF','CFA','Aurora'];
const domains=['Agriculture','Climate','Economy','Energy','Environment','Health','Security','SocialGood','Traffic'];
const out=path.join(base,'outputs/v1_p1_mse_matrix_20261006');
await fs.mkdir(out,{recursive:true});
const index=new Map(rows.map(r=>[`${r.model}|${r.domain}|${r.horizon}`,r]));
if(rows.length!==180 || index.size!==180)throw Error('Require all 180 tasks');
const horizons=Object.fromEntries(domains.map(d=>[d,[...new Set(rows.filter(r=>r.domain===d).map(r=>r.horizon))].sort((a,b)=>a-b)]));
if(Object.values(horizons).some(h=>h.length!==4))throw Error('Every domain requires four horizons');
function col(n){let s='';while(n>0){n--;s=String.fromCharCode(65+n%26)+s;n=Math.floor(n/26);}return s;}
const lastCol=col(109);
const wb=Workbook.create();
const sheet=wb.worksheets.add('MSE原始与修改对比');
const navy='#213C57',dark='#304E6B',blue='#E7EEF5',body='#243447';
sheet.showGridLines=false;sheet.tabColor=navy;
const all=sheet.getRange(`A1:${lastCol}13`);
all.format.font={name:'Arial',size:10,color:body};all.format.rowHeight=26;
all.format.columnWidth=17;all.format.verticalAlignment='center';
sheet.getRange('A1:A13').format.columnWidth=17;
sheet.getRange('A2').values=[['五模型 × 九领域 × 四步长：原始 MSE / 修改后 MSE / 变化百分比']];
sheet.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:navy};
sheet.getRange('A3').values=[['变化百分比=(修改后MSE−原始MSE)/原始MSE。正号=误差上升，负号=误差下降；绿色下降、红色上升。']];
sheet.getRange('A4').values=[['原始=无插件冻结预测；修改后=本轮V1-P1（lr_003、原alpha策略）的已完成任务结果，各任务为三个种子的MSE指标均值。']];
sheet.getRange('A5').values=[['来源：baseline/V1参数实验_20261006/RESULTS_180.json。原始值已与冻结参考核对一致。']];
sheet.getRange('A3:A5').format.font={name:'Arial',size:10,color:'#586A7B'};
sheet.mergeCells('A6:A8');sheet.getRange('A6').values=[['模型']];
sheet.getRange('A6:A8').format.fill=navy;
sheet.getRange('A6:A8').format.font={name:'Arial',size:10,bold:true,color:'#FFFFFF'};
sheet.getRange('A6:A8').format.horizontalAlignment='center';
sheet.getRange('A9:A13').values=models.map(m=>[m]);
sheet.getRange('A9:A13').format.font.bold=true;
sheet.getRange(`B9:${lastCol}13`).format.horizontalAlignment='right';
sheet.getRange('A6:A13').format.borders={right:{style:'medium',color:navy}};
const checks=[];
for(let di=0;di<domains.length;di++){
 const d=domains[di],start=2+di*12,end=start+11;
 const group=`${col(start)}6:${col(end)}6`;
 sheet.mergeCells(group);sheet.getRange(`${col(start)}6`).values=[[d]];
 sheet.getRange(group).format.fill=di%2?dark:navy;
 sheet.getRange(group).format.font={name:'Arial',size:11,bold:true,color:'#FFFFFF'};
 sheet.getRange(group).format.horizontalAlignment='center';
 sheet.getRange(group).format.rowHeight=30;
 for(let hi=0;hi<4;hi++){
  const h=horizons[d][hi],c=start+hi*3;
  sheet.mergeCells(`${col(c)}7:${col(c+2)}7`);
  sheet.getRange(`${col(c)}7`).values=[[`预测步长 ${h}`]];
  sheet.getRange(`${col(c)}7:${col(c+2)}7`).format.fill=blue;
  sheet.getRange(`${col(c)}7:${col(c+2)}7`).format.font.bold=true;
  sheet.getRange(`${col(c)}7:${col(c+2)}7`).format.horizontalAlignment='center';
  sheet.getRange(`${col(c)}8:${col(c+2)}8`).values=[['原始MSE','修改后MSE','变化百分比']];
  sheet.getRange(`${col(c)}8:${col(c+2)}8`).format.fill='#F3F5F7';
  sheet.getRange(`${col(c)}8:${col(c+2)}8`).format.horizontalAlignment='center';
  const vals=models.map(m=>{
   const r=index.get(`${m}|${d}|${h}`);
   if(!r || r.version!=='V1-P1' || r.policy!=='tau0_original' || r.baseline_mse<=0)throw Error('Invalid task');
   return [r.baseline_mse,r.mse,null];
  });
  sheet.getRange(`${col(c)}9:${col(c+2)}13`).values=vals;
  sheet.getRange(`${col(c)}9:${col(c+1)}13`).setNumberFormat('0.000000');
  sheet.getRange(`${col(c)}9:${col(c)}13`).format.font.color='#586A7B';
  sheet.getRange(`${col(c+1)}9:${col(c+1)}13`).format.font.color='#186598';
  const change=sheet.getRange(`${col(c+2)}9:${col(c+2)}13`);
  change.formulas=models.map((m,i)=>{
   const n=i+9;
   checks.push({address:`${col(c+2)}${n}`,want:(vals[i][1]-vals[i][0])/vals[i][0]});
   return [`=(${col(c+1)}${n}-${col(c)}${n})/${col(c)}${n}`];
  });
  change.setNumberFormat('[>=0.0000005]+0.0000%;[<=-0.0000005]-0.0000%;"0.0000%"');
  change.conditionalFormats.add('colorScale',{colors:['#6DC391','#FFFFFF','#E9A2A2'],thresholds:[-.1,0,.1]});
  change.conditionalFormats.add('cellIs',{operator:'lessThan',formula:-.0000005,format:{font:{color:'#1A7040'}}});
  change.conditionalFormats.add('cellIs',{operator:'greaterThan',formula:.0000005,format:{font:{color:'#AB3030'}}});
  sheet.getRange(`${col(c+2)}7:${col(c+2)}13`).format.borders={right:{style:'thin',color:'#CAD5DE'}};
 }
 sheet.getRange(`${col(end)}6:${col(end)}13`).format.borders={right:{style:'medium',color:navy}};
}
sheet.freezePanes.freezeRows(8);sheet.freezePanes.freezeColumns(1);
wb.recalculate();
for(const t of checks){const got=sheet.getRange(t.address).values[0][0];if(Math.abs(got-t.want)>1e-12)throw Error('Change sign/value mismatch '+t.address);}
if(checks.length!==180)throw Error('Incomplete percent changes');
console.log((await wb.inspect({kind:'table',range:'MSE原始与修改对比!A8:M13',include:'values,formulas',tableMaxRows:6,tableMaxCols:13,maxChars:1800})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},summary:'Formula errors',maxChars:800})).ndjson);
for(let di=0;di<9;di++){
 const start=2+di*12,end=start+11;
 const range=di===0?`A1:${col(end)}13`:`${col(start)}6:${col(end)}13`;
 const preview=await wb.render({sheetName:sheet.name,range,scale:1.5,format:'png'});
 await fs.writeFile(path.join(out,`preview_${domains[di]}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const x=await SpreadsheetFile.exportXlsx(wb);
const file=path.join(out,'九领域_五模型_四步长_MSE原版与修改对比.xlsx');
await x.save(file);
await fs.writeFile(path.join(out,'布局与数据核对.json'),JSON.stringify({models,domains,horizons,model_rows:5,domain_groups:9,horizon_groups:36,metric_values:540,raw_mse_values:360,change_formulas:180,worksheet_count:1,excel_columns:109,last_column:lastCol,change_definition:'(modified-original)/original',source:inputPath},null,2));
console.log('EXPORTED',file);
