"""下载完整旧工作目录、校验、归档重要配置到GitHub，最后按清单清理旧训练状态。"""
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
import paramiko

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TARGET = '/xiliang/LXY/baseline_v3_lab_20261004'
CONTROL = '/xiliang/LXY/maintenance_archive_20261008'
PYTHON = '/xiliang/LXY/envs/lxy/bin/python'
BRANCH = 'feat/v1-joint-internal-20261008'
DEST = Path('D:/LXY_server_backup/20261008_legacy_v3')
REPORT = REPO/'reports/server_cleanup_20261008'


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
    # 训练配置、指标、版本计划和全部权重哈希可直接在GitHub审阅；二进制权重完整保存在D盘。
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
        '完整服务器工作目录先下载至本机D盘并校验，再移除旧checkpoint/resume文件和离线安装包副本。'
        '模型预训练资产、数据、源码、指标、日志和预测结果保留在服务器。当前r3运行目录及lxy环境不在删除范围。\n\n'
        f'本地备份：`{DEST}`。完整模型权重在`files/`中；Git保存超参数、全部指标、各文件SHA256与恢复位置。\n', encoding='utf-8')
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


def work(password):
    client = connect(password)
    try:
        status('inventory')
        remote(client, PYTHON+' -B '+CONTROL+'/remote.py inventory')
        with client.open_sftp() as sftp:
            sftp.get(CONTROL+'/inventory.json', str(DEST/'source_inventory.json'))
        inventory = json.loads((DEST/'source_inventory.json').read_text(encoding='utf-8'))
        if inventory['root'] != TARGET:
            raise ValueError('Unexpected remote target')
        files = [item for item in inventory['entries'] if item['kind']=='file']
        total = sum(item['size'] for item in files)
        if shutil.disk_usage(DEST).free < total+2*1024**3:
            raise RuntimeError('Insufficient local space for a complete archive')
        names = [item['path'].casefold() for item in files]
        if len(names) != len(set(names)):
            raise ValueError('Windows case collision; archive needs explicit remapping')
        verified = {}
        done_bytes = 0
        last_update = 0
        with client.open_sftp() as sftp:
            for index,item in enumerate(files):
                local = local_path(item['path'])
                local.parent.mkdir(parents=True, exist_ok=True)
                partial = local.with_name(local.name+'.archive-part')
                source = TARGET+'/'+item['path']
                info = sftp.stat(source)
                if info.st_size != item['size'] or int(info.st_mtime) != item['mtime_ns']//1_000_000_000:
                    raise ValueError('Remote file changed before download: '+item['path'])
                def progress(transferred, size):
                    nonlocal last_update
                    now = time.monotonic()
                    if now-last_update > 5:
                        status('downloading', files_done=index, files_total=len(files), bytes_done=done_bytes+transferred,
                               bytes_total=total, current_file=item['path'])
                        last_update = now
                sftp.get(source, str(partial), callback=progress)
                if partial.stat().st_size != item['size']:
                    raise ValueError('Incomplete download')
                checksum = sha(partial)
                partial.replace(local)
                verified[item['path']] = checksum
                item['sha256'] = checksum
                done_bytes += item['size']
                if index%30==0:
                    (DEST/'downloaded_hashes.json').write_text(json.dumps(verified), encoding='utf-8')
        # 再次读取本地所有待删状态文件验证校验和，避免仅依赖下载时的检查。
        for item in files:
            if sha(local_path(item['path'])) != item['sha256']:
                raise ValueError('Local backup checksum changed: '+item['path'])
        inventory.update(local_verified=True, local_archive=str(DEST), verified_unix=time.time())
        (DEST/'verified_manifest.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
        status('publishing_github', files=len(files), bytes=total)
        inventory['git_commit'] = git_archive(inventory)
        (DEST/'verified_local_receipt.json').write_text(json.dumps(inventory,ensure_ascii=False),encoding='utf-8')
        with client.open_sftp() as sftp:
            sftp.put(str(DEST/'verified_local_receipt.json'), CONTROL+'/verified_local_receipt.json')
        status('verifying_server_before_cleanup', git_commit=inventory['git_commit'])
        # 使用同一lxy环境核验源文件SHA256，匹配后才逐项unlink。
        def update(line):
            try:
                value=json.loads(line)
            except json.JSONDecodeError:
                value={'message':line}
            status('server_cleanup', progress=value, git_commit=inventory['git_commit'])
        remote(client, PYTHON+' -u -B '+CONTROL+'/remote.py cleanup', sink=update)
        with client.open_sftp() as sftp:
            sftp.get(CONTROL+'/CLEANUP_COMPLETED.json',str(DEST/'CLEANUP_COMPLETED.json'))
        usage=remote(client,'du -x -B1 --max-depth=1 '+TARGET+' && du -x -s -B1 /xiliang/LXY')
        (DEST/'disk_usage_after.txt').write_text(usage,encoding='utf-8')
        status('complete', result=json.loads((DEST/'CLEANUP_COMPLETED.json').read_text()),disk_usage_after=usage)
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
