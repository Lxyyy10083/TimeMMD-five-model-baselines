"""Download completed V1-P1 evidence; credential is read only into memory."""
from pathlib import Path
import csv
import getpass
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
import paramiko

BASE=Path(sys.argv[1]).resolve()
DEST=BASE/'V1参数实验_20261006'
DEST.mkdir(exist_ok=True)
REMOTE='/xiliang/LXY/baseline_v1_parameter_20261006'
cfg=paramiko.SSHConfig()
with Path.home().joinpath('.ssh/config').open() as f:cfg.parse(f)
h=cfg.lookup('Remote')
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.WarningPolicy())
password=getpass.getpass('Lab password for V1-P1 results: ')
c.connect(h['hostname'],port=int(h['port']),username=h['user'],password=password,
          timeout=20,auth_timeout=20,allow_agent=False,look_for_keys=False)
del password
if '--watch' in sys.argv[2:]:
    last=None
    while True:
        with c.open_sftp() as s:
            with s.open(REMOTE+'/plugins/v1_parameter_experiment_20261006/outputs/STATUS.json') as f:
                status=json.loads(f.read().decode())
            launch=json.loads(s.open(REMOTE+'/V1_P1_LAUNCH.json').read().decode())
            try:
                completed=json.loads(s.open(REMOTE+'/plugins/v1_parameter_experiment_20261006/outputs/FULL_COMPLETED.json').read().decode())
            except IOError:
                completed=None
        (DEST/'等待与下载状态.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
        marker=(status.get('stage'),status.get('variant'),status.get('completed_fits'),status.get('completed_test_records'))
        if marker!=last:
            print('PROGRESS',json.dumps(status,ensure_ascii=False),flush=True)
            last=marker
        if completed and completed.get('test_rows')==16200 and completed.get('all_validation_plateau'):
            break
        _,out,err=c.exec_command('/usr/bin/python3 -c '+repr('from pathlib import Path; print(Path("/proc/%s").exists())'%launch['pid']))
        alive=out.read().decode().strip()
        if alive!='True':
            raise RuntimeError('Own training process stopped before complete; inspect its log and unresolved convergence status')
        time.sleep(50)
    pack='''from pathlib import Path
import zipfile
r=Path('/xiliang/LXY/baseline_v1_parameter_20261006')
o=r/'plugins/v1_parameter_experiment_20261006/outputs'
with zipfile.ZipFile(str(r/'completed_v1_p1_results.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in o.rglob('*'):
        if p.is_file(): z.write(str(p),str(p.relative_to(o)))
    for name in ['V1_P1_LAUNCH.json','V1_P1_INITIAL_LAUNCH.json','V1_P1_PRE_CONVERGENCE_LAUNCH.json','V1_P1_CONVERGENCE_V2_LAUNCH.json','V1_P1_CONVERGENCE_V3_LAUNCH.json','V1_P1_LAUNCH.log','INITIAL_FAILURES.json','REFERENCE_V1_SOURCES.json','INPUT_DOWNLOAD_COMPLETE.json']:
        p=r/name
        if p.is_file(): z.write(str(p),'provenance/'+name)
print('ARCHIVE_READY')
'''
    # The command contains no user-supplied text; quote Python source for POSIX shell.
    command='/usr/bin/python3 -c '+"'"+pack.replace("'","'\\''")+"'"
    _,out,err=c.exec_command(command)
    answer=out.read().decode();error=err.read().decode()
    if out.channel.recv_exit_status():raise RuntimeError(error)
    print(answer,flush=True)
archive=DEST.with_suffix('.zip')
with c.open_sftp() as s:s.get(REMOTE+'/completed_v1_p1_results.zip',str(archive))
c.close()
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        if os.path.commonpath([str(DEST),str((DEST/name).resolve())])!=str(DEST):
            raise ValueError('Unsafe archive member')
    z.extractall(DEST)
completed=json.loads((DEST/'FULL_COMPLETED.json').read_text())
assert completed['final_fits'] in [108,216] and completed['test_rows']==16200
assert completed['all_validation_plateau'] and completed['matched_control_fits']==108
primary=json.loads((DEST/'RESULTS_180.json').read_text())
assert len(primary)==180 and len({(r['model'],r['domain'],r['horizon']) for r in primary})==180
raw=list(csv.DictReader((DEST/'test_seed_results.csv').open(encoding='utf-8-sig')))
assert len(raw)==16200
from convergence_report import verify_and_plot
verify_and_plot(DEST,completed['final_fits'])
reference=json.loads((BASE/'GANF_V1_MSE_Excel_rerun2_20261006/RESULTS_180.json').read_text())
reference={(r['model'],r['domain'],r['horizon']):r for r in reference}
for r in primary:
    old=reference[(r['model'],r['domain'],r['horizon'])]
    for key in ['mse','mae']:
        assert abs(r['v1_'+key]-old['plugin_'+key])<1e-6*max(1,old['plugin_'+key])
        assert abs(r['baseline_'+key]-old['baseline_'+key])<1e-8*max(1,old['baseline_'+key])
stats=dict(completed=completed,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
           original_v1_recomputed_matches=True,
           mse_decrease_pct=sum(r['mse_decrease_pct'] for r in primary)/180,
           mae_decrease_pct=sum(r['mae_decrease_pct'] for r in primary)/180,
           mse_decrease_vs_v1_pct=sum(r['mse_decrease_vs_v1_pct'] for r in primary)/180,
           mae_decrease_vs_v1_pct=sum(r['mae_decrease_vs_v1_pct'] for r in primary)/180,
           mse_decrease_vs_matched_v1_pct=sum(r['mse_decrease_vs_matched_v1_pct'] for r in primary)/180,
           mae_decrease_vs_matched_v1_pct=sum(r['mae_decrease_vs_matched_v1_pct'] for r in primary)/180,
           both_better=sum(r['mse_decrease_pct']>1e-5 and r['mae_decrease_pct']>1e-5 for r in primary),
           any_worse=sum(r['mse_decrease_pct']<-1e-5 or r['mae_decrease_pct']<-1e-5 for r in primary),
           unchanged=sum(abs(r['mse_decrease_pct'])<=1e-5 and abs(r['mae_decrease_pct'])<=1e-5 for r in primary))
(DEST/'下载与结果核对.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
report=(DEST/'V1参数实验结果.md').read_text(encoding='utf-8')
report+='\n## 整体主结果核对\n\n'
report+=f"相对未修改原版，180项平均MSE减小{stats['mse_decrease_pct']:+.4f}%，MAE减小{stats['mae_decrease_pct']:+.4f}%。\n\n"
report+=f"相对原V1，平均MSE进一步减小{stats['mse_decrease_vs_v1_pct']:+.4f}%，MAE进一步减小{stats['mae_decrease_vs_v1_pct']:+.4f}%。负值表示反而增加。\n\n"
report+=f"相对相同收敛规则的V1对照，调参MSE进一步减小{stats['mse_decrease_vs_matched_v1_pct']:+.4f}%，MAE进一步减小{stats['mae_decrease_vs_matched_v1_pct']:+.4f}%。\n\n"
report+=f"相对未修改原版双指标改善{stats['both_better']}项，至少一项退化{stats['any_worse']}项，双持平{stats['unchanged']}项。复算原V1与上一轮结果逐项吻合。\n"
(DEST/'V1参数实验结果.md').write_text(report,encoding='utf-8')
repo=BASE/'experiment_vcs/server_baseline'
relative='plugins/v1_parameter_experiment_20261006/reports/20261006'
target=repo/relative;target.mkdir(parents=True,exist_ok=True)
for p in DEST.iterdir():
    if p.is_file() and p.suffix in ['.csv','.json','.md','.png']:
        shutil.copyfile(p,target/p.name)
for p in DEST.rglob('fit_result.json'):
    t=target/p.relative_to(DEST);t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,t)
subprocess.run(['git','-c','core.autocrlf=false','add','--',relative],cwd=repo,check=True)
commit=subprocess.run(['git','-c','core.autocrlf=false','commit','--only','-m','Archive V1-P1 validation-selected weights and complete gate sensitivity results','--',relative],cwd=repo,capture_output=True,text=True)
if commit.returncode:raise RuntimeError(commit.stderr)
subprocess.run(['git','push','origin','HEAD:main'],cwd=repo,check=True)
print(json.dumps(stats,ensure_ascii=False,indent=2))
