"""保留一套最佳旧权重，下载配置指标并推送GitHub，然后清理其余旧训练状态。"""
from pathlib import Path, PurePosixPath
import argparse
import datetime
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TARGET = '/xiliang/LXY/baseline_v3_lab_20261004'
CONTROL = '/xiliang/LXY/maintenance_archive_20261008'
PYTHON = '/xiliang/LXY/envs/lxy/bin/python'
BRANCH = 'feat/v1-joint-internal-20261008'
DEST = Path('D:/LXY_server_backup/20261008_best_V3_internal')
REPORT = REPO/'reports/server_cleanup_20261008'
METADATA_NAMES = {'run_config.json','COMPLETED.json','EVALUATED.json','STARTED.json','PLAN.json',
    'SUMMARY.json','RESULTS.json','FULL_COMPLETED.json','CONTINUATION_PLAN.json','RESOLVED_SOURCES.json'}


def status(stage, **details):
    DEST.mkdir(parents=True, exist_ok=True)
    record = dict(stage=stage, local_pid=os.getpid(),
        checked_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), **details)
    temp = DEST/'STATUS.tmp'
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(DEST/'STATUS.json')


def connect(password):
    config = paramiko.SSHConfig()
    with Path.home().joinpath('.ssh/config').open() as f:
        config.parse(f)
    host = config.lookup('Remote')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(host['hostname'], port=int(host['port']), username=host['user'], password=password,
        timeout=20, auth_timeout=20, allow_agent=False, look_for_keys=False)
    client.get_transport().set_keepalive(30)
    return client


def remote(client, command, sink=None):
    _, output, errors = client.exec_command('cd /xiliang/LXY && '+command)
    lines = []
    for line in output:
        if sink:
            sink(line.rstrip())
        else:
            lines.append(line)
    error = errors.read().decode(errors='replace')
    code = output.channel.recv_exit_status()
    if code:
        raise RuntimeError('Remote command failed: '+str(code)+' '+error[-1500:])
    return ''.join(lines)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def local_path(relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or ':' in relative or '\\' in relative:
        raise ValueError('Unsafe local archive path: '+relative)
    root = DEST/'files'
    path = root.joinpath(*p.parts)
    if root.resolve() not in path.resolve().parents:
        raise ValueError('Archive path outside destination')
    for parent in [path, *path.parents]:
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError('Local archive path contains a symlink')
    return path


def git_archive(inventory):
    # GitHub保存全部旧实验参数和指标；唯一保留的180个权重仍位于服务器。
    REPORT.mkdir(parents=True, exist_ok=True)
    names = {'run_config.json','COMPLETED.json','EVALUATED.json','STARTED.json','PLAN.json',
             'SUMMARY.json','RESULTS.json','FULL_COMPLETED.json','CONTINUATION_PLAN.json','RESOLVED_SOURCES.json'}
    for item in inventory['entries']:
        if item['kind']=='file' and PurePosixPath(item['path']).name in names and item['size']<15_000_000:
            source = local_path(item['path'])
            target = REPORT/'legacy_metadata'/PurePosixPath(item['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    (REPORT/'archive_manifest.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    (REPORT/'README.md').write_text(
        '# 2026-10-08旧实验备份与清理\n\n'
        '按180任务的平均相对MSE选择V3插件版，保留其180个checkpoint。'
        '其余旧checkpoint、全部resume及离线wheel副本按用户授权直接删除，不再完整备份。'
        '模型预训练资产、数据、源码、指标、日志和预测结果保留在服务器。当前r3运行目录及lxy环境不在删除范围。\n\n'
        f'本地参数与指标归档：`{DEST}`。保留权重的路径和SHA256见archive_manifest.json。'
        '保留权重未下载到本地；当前r3完成后再处理其下载。旧版本训练协议不同，历史排名不作为严格因果对照结论。\n', encoding='utf-8')
    relative = REPORT.relative_to(REPO).as_posix()
    def git(*args):
        return subprocess.check_output(['git',*args], cwd=REPO, text=True, stderr=subprocess.STDOUT).strip()
    if git('branch','--show-current') != BRANCH:
        raise RuntimeError('Unexpected Git branch; cleanup refused')
    git('add','--',relative)
    staged = git('diff','--cached','--name-only','--',relative)
    if staged:
        git('commit','--only','-m','Archive legacy lab training settings and verified backup manifest','--',relative)
    revision = git('rev-parse','HEAD')
    git('push','origin','HEAD:'+BRANCH)
    published = git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]
    if published != revision:
        raise RuntimeError('GitHub revision mismatch; cleanup refused')
    return revision


def retention_plan():
    v3 = json.loads((REPO/'plugins/internal_semflow_full_lab/reports/20261005/RESULTS.json').read_text(encoding='utf-8'))
    v4 = json.loads((REPO/'plugins/distribution_semflow_v4/reports/20261005/RESULTS.json').read_text(encoding='utf-8'))
    key=lambda r:(r['model'],r['domain'],int(r['horizon']))
    original={key(r):r['historical_original'] for r in v3}
    ranking=[]
    for version,rows in [('V3',v3),('V4',v4)]:
        if len(rows)!=180 or {key(r) for r in rows}!=set(original):
            raise ValueError('Incomplete or unmatched historical results')
        for variant in ['control','internal']:
            improvement={metric:100*(1-sum(r[variant][metric]/original[key(r)][metric] for r in rows)/180) for metric in ['mse','mae']}
            ranking.append(dict(version=version,variant=variant,mean_relative_improvement_pct=improvement))
    winner=max(ranking,key=lambda r:r['mean_relative_improvement_pct']['mse'])
    if (winner['version'],winner['variant'])!=('V3','internal'):
        raise ValueError('Selection changed; inspect ranking before cleanup')
    if not all(r['convergence']['internal']['converged'] for r in v3):
        raise ValueError('Selected version is not fully converged')
    paths=sorted('plugins/internal_semflow_full_lab/outputs_full/internal/'+r['model']+'/'+r['domain']+'/'+str(r['horizon'])+'/2026/checkpoint.pt' for r in v3)
    return dict(criterion='Mean per-task relative MSE against the same historical original; 180 tasks equally weighted',
                selected=winner,ranking=ranking,keep_paths=paths,scope=TARGET,
                note='One complete version, not a per-task mixture. Old protocols differ; retention ranking is not a causal benchmark.')


def work(password):
    client = connect(password)
    try:
        status('preparing_best_version')
        plan=retention_plan()
        (DEST/'retention_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
        with client.open_sftp() as sftp:
            sftp.put(str(DEST/'retention_plan.json'),CONTROL+'/retention_plan.json')
        remote(client, PYTHON+' -u -B '+CONTROL+'/remote.py prepare')
        with client.open_sftp() as sftp:
            sftp.get(CONTROL+'/inventory.json',str(DEST/'source_inventory.json'))
            sftp.get(CONTROL+'/important_metadata.zip',str(DEST/'important_metadata.zip'))
        inventory=json.loads((DEST/'source_inventory.json').read_text(encoding='utf-8'))
        if inventory['root']!=TARGET or inventory['selection']!=plan:
            raise ValueError('Unexpected remote target or retention plan')
        if sha(DEST/'important_metadata.zip')!=inventory['metadata_zip_sha256']:
            raise ValueError('Metadata archive checksum mismatch')
        with zipfile.ZipFile(DEST/'important_metadata.zip') as archive:
            for item in archive.infolist():
                target=local_path(item.filename)
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(item) as source,target.open('wb') as out:
                    shutil.copyfileobj(source,out)
        for item in inventory['entries']:
            if item['kind']=='file' and PurePosixPath(item['path']).name in METADATA_NAMES and item['size']<15_000_000:
                if sha(local_path(item['path']))!=item['sha256']:
                    raise ValueError('Extracted metadata checksum mismatch')
        inventory.update(metadata_verified=True,local_archive=str(DEST),verified_unix=time.time())
        status('publishing_github',retained_checkpoints=len(inventory['retained']))
        inventory['git_commit']=git_archive(inventory)
        (DEST/'verified_local_receipt.json').write_text(json.dumps(inventory,ensure_ascii=False),encoding='utf-8')
        with client.open_sftp() as sftp:
            sftp.put(str(DEST/'verified_local_receipt.json'),CONTROL+'/verified_local_receipt.json')
        status('verifying_best_weights_before_cleanup',git_commit=inventory['git_commit'])
        def update(line):
            try:value=json.loads(line)
            except json.JSONDecodeError:value={'message':line}
            status('server_cleanup',progress=value,git_commit=inventory['git_commit'])
        remote(client,PYTHON+' -u -B '+CONTROL+'/remote.py cleanup',sink=update)
        with client.open_sftp() as sftp:
            sftp.get(CONTROL+'/CLEANUP_COMPLETED.json',str(DEST/'CLEANUP_COMPLETED.json'))
            sftp.get(CONTROL+'/deleted_files.json',str(DEST/'deleted_files.json'))
        usage=remote(client,'du -x -B1 --max-depth=1 '+TARGET+' && du -x -s -B1 /xiliang/LXY')
        (DEST/'disk_usage_after.txt').write_text(usage,encoding='utf-8')
        status('complete',result=json.loads((DEST/'CLEANUP_COMPLETED.json').read_text()),disk_usage_after=usage)
    finally:
        client.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--worker',action='store_true')
    args=p.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    if args.worker:
        secret=json.loads(sys.stdin.readline())
        password=secret.pop('password');secret.clear()
        try:
            work(password)
        except Exception as exc:
            status('needs_attention_no_unverified_deletion',error_type=type(exc).__name__,error=str(exc))
            raise
    else:
        password=getpass.getpass('Lab SSH password (not saved): ')
        with (DEST/'worker.log').open('a',encoding='utf-8') as output:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],
                cwd=REPO,stdin=subprocess.PIPE,stdout=output,stderr=subprocess.STDOUT,text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            process.stdin.write(json.dumps({'password':password})+'\n');process.stdin.close()
        del password
        print(json.dumps(dict(started=True,pid=process.pid,local_archive=str(DEST))),flush=True)


if __name__=='__main__':
    main()
