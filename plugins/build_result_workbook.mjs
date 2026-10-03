import fs from 'node:fs/promises';
import { SpreadsheetFile, Workbook } from '@oai/artifact-tool';

const input = 'C:/Users/32113/OneDrive/Desktop/baseline/CARMA_results/results_180.csv';
const output = 'C:/Users/32113/OneDrive/Desktop/baseline/CARMA_results/CARMA_五模型九领域MSE_MAE.xlsx';
const lines = (await fs.readFile(input, 'utf8')).replace(/^\uFEFF/, '').trim().split(/\r?\n/);
const data = lines.map((line, row) => line.split(',').map((value, column) => {
  if (row === 0 || value === '') return value;
  return [2, 4, 5, 6, 7, 8, 9, 10].includes(column) ? Number(value) : value;
}));
if (data.length !== 181 || data.some(row => row.length !== 11)) {
  throw new Error(`Expected 180 complete cases and 11 columns, got ${data.length - 1}`);
}
data[0] = ['模型', '领域', '预测步长', '状态', '验证启用',
           '原MSE', 'CARMA MSE', 'MSE差值', '原MAE', 'CARMA MAE', 'MAE差值'];
const wb = Workbook.create();
const sheet = wb.worksheets.add('180组结果');
sheet.getRange('A1:K181').values = data;
sheet.getRange('A1:K1').format.font = { bold: true };
sheet.getRange('A1:K1').format.rowHeight = 26;
sheet.getRange('A1:A181').format.columnWidth = 16;
sheet.getRange('B1:B181').format.columnWidth = 18;
sheet.getRange('C1:E181').format.columnWidth = 13;
sheet.getRange('F1:K181').format.columnWidth = 16;
sheet.getRange('F2:K181').setNumberFormat('0.000000');
sheet.freezePanes.freezeRows(1);
wb.recalculate();
const preview = await wb.render({ sheetName: '180组结果', range: 'A1:K12', scale: 1.4 });
await fs.writeFile('C:/Users/32113/OneDrive/Desktop/baseline/CARMA_results/workbook_preview.png',
                   new Uint8Array(await preview.arrayBuffer()));
const file = await SpreadsheetFile.exportXlsx(wb);
await file.save(output);
console.log(output);
