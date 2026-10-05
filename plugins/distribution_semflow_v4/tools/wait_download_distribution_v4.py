"""One-off background result retrieval; SSH secret exists only in process memory."""
from pathlib import Path
import csv
import datetime
import getpass
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
import paramiko

BASE=next(p for p in Path(__file__).resolve().parents if (p/'experiment_vcs/server_baseline/.git').exists())
REMOTE='/xiliang/LXY/baseline_v3_lab_20261004'
OUTPUT=REMOTE+'/plugins/distribution_semflow_v4/outputs_final_converged'
STATE=BASE/'GANF_internal_fusion_20261003/V4自动下载状态.json'

def status(**kwargs):
    STATE.write_text(json.dumps(kwargs,ensure_ascii=False,indent=2),encoding='utf-8')

def summarize(out):
    rows=json.loads((out/'RESULTS.json').read_text(encoding='utf-8'))
    assert len(rows)==180
    assert all(r['convergence'][v]['converged'] for r in rows for v in ['control','internal'])
    assert all(r['convergence']['internal']['best_epoch']>=20 for r in rows)
    lines=['# V4完整条件分布实验结果','',
        '180/180配对、360次训练和360次评估完成。全部满足预先约定的验证平台期；插件选中权重均完成20轮分布升温。',
        'MSE、MAE越小越好；下降百分比为(control-plugin)/control*100，正数改善、负数变差。',
        'control与插件均继承上轮相同任务收敛的原生权重，再以相同规则追加训练。单种子2026，平台期不证明全局最优或统计显著性。','',
        '|模型|MSE平均下降%|MAE平均下降%|两项同时改善|','|---|---:|---:|---:|']
    summary=[]
    for m in ['SpecTF','CFA','TaTS','MM-TSFlib','Aurora']:
        a=[r for r in rows if r['model']==m]
        item=dict(model=m,mse_mean_decrease_pct=sum(r['decrease_pct']['mse'] for r in a)/36,
            mae_mean_decrease_pct=sum(r['decrease_pct']['mae'] for r in a)/36,
            both_improved=sum(all(r['decrease_pct'][k]>0 for k in ['mse','mae']) for r in a))
        summary.append(item)
        lines.append(f"|{m}|{item['mse_mean_decrease_pct']:+.2f}|{item['mae_mean_decrease_pct']:+.2f}|{item['both_improved']}/36|")
    lines.extend(['','## 文件','',
        '- comparison.csv：本轮配对MSE/MAE结果。comparison_all_versions.csv：同时列本轮对照、上一轮原生、上一轮插件、历史原版，避免混用比较口径。',
        '- MSE_comparison.png、MAE_comparison.png：所有领域/长度的下降百分比。',
        '- 每任务conditional_distribution.png：固定第一个测试起点的条件分布与预测；distribution_profile.csv：分布摘要及语义增量。',
        '- 每任务test_predictions.npz：标准化尺度下的预测、目标、条件均值、分位数、混合概率等；first_origin_raw_distribution.npz：固定示例的原始单位。',
        '- 学习到的图与门控表示模型内条件依赖，不表示真实因果解释；样本分位区间不是保证覆盖率的置信区间。',
        '- 单种子探索性实验，已保留所有变差项。历史结果训练协议不同，不能全部归因于新增模块。'])
    (out/'完整结果汇总.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=['model','domain','horizon']
    groups=['control','internal','previous_native','previous_internal','historical_original']
    fields += [g+'_'+k for g in groups for k in ['mse','mae']]
    fields += ['decrease_vs_control_'+k+'_pct' for k in ['mse','mae']]
    fields += ['decrease_vs_previous_plugin_'+k+'_pct' for k in ['mse','mae']]
    with (out/'comparison_all_versions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            row={k:r[k] for k in ['model','domain','horizon']}
            for g in groups:
                for k in ['mse','mae']:row[g+'_'+k]=r[g][k]
            for k in ['mse','mae']:
                row['decrease_vs_control_'+k+'_pct']=r['decrease_pct'][k]
                row['decrease_vs_previous_plugin_'+k+'_pct']=r['decrease_vs_previous_internal_pct'][k]
            w.writerow(row)

def archive_git(out):
    repo=BASE/'experiment_vcs/server_baseline'
    relative='plugins/distribution_semflow_v4/reports/20261005'
    dest=repo/relative;dest.mkdir(parents=True,exist_ok=True)
    for p in out.iterdir():
        if p.is_file() and p.suffix in ['.json','.csv','.md','.png']:shutil.copyfile(p,dest/p.name)
    for model in ['SpecTF','CFA','TaTS','MM-TSFlib','Aurora']:
        p=out/'internal'/model/'Agriculture/6/2026/conditional_distribution.png'
        if p.exists():shutil.copyfile(p,dest/(model+'_conditional_distribution.png'))
    subprocess.run(['git','add','--',relative],cwd=repo,check=True)
    subprocess.run(['git','commit','--only','-m','Archive complete fitted conditional-distribution experiment for all five models','--',relative],cwd=repo,check=True)
    subprocess.run(['git','push','origin','HEAD:main'],cwd=repo,check=True)

def work(password):
    cfg=paramiko.SSHConfig();cfg.parse(Path.home().joinpath('.ssh/config').open())
    h=cfg.lookup('Remote')
    status(stage='waiting',remote_output=OUTPUT,local_pid=os.getpid())
    while True:
        c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.WarningPolicy())
        try:
            c.connect(h['hostname'],port=int(h['port']),username=h['user'],password=password,
                timeout=15,auth_timeout=15,allow_agent=False,look_for_keys=False)
            with c.open_sftp() as s:
                try:
                    with s.open(OUTPUT+'/FULL_COMPLETED.json') as f:done=json.loads(f.read())
                except OSError:done=None
                if not done:
                    try:
                        with s.open(OUTPUT+'/INCOMPLETE.json') as f:failed=json.loads(f.read())
                    except OSError:failed=None
                    if failed:
                        status(stage='incomplete_requires_attention',details=failed);return
                    try:
                        with s.open(OUTPUT+'/STATUS.json') as f:progress=json.loads(f.read())
                    except OSError:progress={}
                    status(stage='waiting',remote_output=OUTPUT,progress=progress,local_pid=os.getpid())
                else:
                    assert done['pairs']==180 and done['all_validation_converged']
                    status(stage='packing',completed=done,local_pid=os.getpid())
                    script='''from pathlib import Path
import zipfile
r=Path("/xiliang/LXY/baseline_v3_lab_20261004")
o=r/"plugins/distribution_semflow_v4/outputs_final_converged"
with zipfile.ZipFile(r/"completed_distribution_v4_fitted_results.zip","w",zipfile.ZIP_DEFLATED) as z:
 for p in o.rglob("*"):
  if p.is_file() and p.suffix in [".json",".csv",".md",".png",".npz",".log"]:z.write(p,p.relative_to(o))
 z.write(r/"distribution_v4_fitted_launch.log","distribution_v4_fitted_launch.log")
 z.write(r/"lxy_packages.txt","lxy_packages.txt")
'''
                    _,stdout,stderr=c.exec_command('/usr/bin/python3 -c '+shlex.quote(script))
                    code=stdout.channel.recv_exit_status()
                    if code:raise RuntimeError(stderr.read().decode(errors='replace'))
                    out=BASE/'GANF_internal_fusion_20261003/V4五模型九领域完整结果_20261005'
                    out.mkdir(parents=True,exist_ok=True)
                    archive=out.with_suffix('.zip')
                    status(stage='downloading',local_output=str(out),local_pid=os.getpid())
                    s.get(REMOTE+'/completed_distribution_v4_fitted_results.zip',str(archive))
                    with zipfile.ZipFile(archive) as z:
                        for n in z.namelist():
                            target=(out/n).resolve()
                            if os.path.commonpath([str(out.resolve()),str(target)])!=str(out.resolve()):raise ValueError('archive member outside result directory')
                        z.extractall(out)
                    summarize(out)
                    try:
                        archive_git(out);git_state='pushed_private_repository'
                    except Exception as e:git_state='results_downloaded_git_needs_attention: '+str(e)
                    status(stage='complete',pairs=180,local_output=str(out),git=git_state)
                    return
        except Exception as e:
            status(stage='retrying_connection_or_download',error=str(e),local_pid=os.getpid())
        finally:c.close()
        time.sleep(300)

if __name__=='__main__':
    if '--worker' in sys.argv:
        work(json.loads(sys.stdin.readline())['password'])
    else:
        password=getpass.getpass('Lab SSH password for one-off result retrieval: ')
        log=BASE/'GANF_internal_fusion_20261003/V4自动下载日志.log'
        with log.open('a',encoding='utf-8') as f:
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=str(BASE),
                stdin=subprocess.PIPE,stdout=f,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,text=True)
            p.stdin.write(json.dumps(dict(password=password))+'\n');p.stdin.close()
        del password
        print('RESULT_RETRIEVAL_STARTED',p.pid)
