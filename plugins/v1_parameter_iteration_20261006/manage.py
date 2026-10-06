"""Deploy on network recovery and collect completed results without chat polling.

The SSH password exists only in memory and is supplied through an anonymous pipe.
"""
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

BASE = Path(sys.argv[1]).resolve()
DEST = BASE/'V1-P2参数实验_20261006'
REPO = BASE/'experiment_vcs/server_baseline'
REMOTE = '/xiliang/LXY/baseline_v1_p2_20261006'
PLUGIN = 'plugins/v1_parameter_iteration_20261006'
DEST.mkdir(exist_ok=True)


def status(value):
    temp = DEST/'自动任务状态.tmp'
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(DEST/'自动任务状态.json')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def quote(value):
    return "'" + value.replace("'", "'\\''") + "'"


def remote_run(client, command):
    _, out, err = client.exec_command(command)
    output = out.read().decode(errors='replace')
    error = err.read().decode(errors='replace')
    code = out.channel.recv_exit_status()
    if code:
        raise RuntimeError('Remote command failed: '+error+'\n'+output)
    return output


def connection(password, host):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.WarningPolicy())
    try:
        client.connect(host['hostname'], port=int(host['port']), username=host['user'],
            password=password, timeout=10, auth_timeout=15, banner_timeout=15,
            allow_agent=False, look_for_keys=False)
        client.get_transport().set_keepalive(20)
        return client
    except Exception:
        client.close()
        raise


def read_json(sftp, path):
    with sftp.open(path) as f:
        return json.loads(f.read().decode())


def main():
    config = paramiko.SSHConfig()
    with Path.home().joinpath('.ssh/config').open() as f:
        config.parse(f)
    host = config.lookup('Remote')
    password = sys.stdin.readline().rstrip('\r\n') if '--credential-stdin' in sys.argv else getpass.getpass('Lab password: ')
    deadline = time.monotonic() + 6*3600
    attempts = 0
    while True:
        attempts += 1
        try:
            client = connection(password, host)
            break
        except paramiko.AuthenticationException:
            status(dict(stage='authentication_failed', experiments_started=False))
            raise
        except (OSError, EOFError, paramiko.SSHException) as error:
            status(dict(stage='waiting_for_network', attempt=attempts, error=type(error).__name__,
                experiments_started=False, retry_seconds=60, maximum_initial_wait_hours=6,
                host=host['hostname'], port=int(host['port'])))
            if time.monotonic() >= deadline:
                raise RuntimeError('Initial SSH connection unavailable for six hours; no experiments started')
            time.sleep(60)
    package = BASE/'experiment_vcs/v1_p2_code_20261006.zip'
    metadata = json.loads((BASE/'experiment_vcs/v1_p2_package_metadata_20261006.json').read_text())
    if hashlib.sha256(package.read_bytes()).hexdigest() != metadata['package_sha256']:
        raise ValueError('Deployment archive changed after it was frozen')
    commit = metadata['code_commit']
    # The code archive is frozen beforehand and uploaded, never overwritten after launch.
    with client.open_sftp() as sftp:
        sftp.put(str(package), '/xiliang/LXY/v1_p2_code_20261006.zip')
        sftp.put(str(REPO/PLUGIN/'deploy.py'), '/xiliang/LXY/deploy_v1_p2_20261006.py')
    answer = remote_run(client, '/usr/bin/python3 /xiliang/LXY/deploy_v1_p2_20261006.py '+quote(commit))
    launch = json.loads(answer)
    status(dict(stage='launched', experiments_started=True, launch=launch))
    with client.open_sftp() as sftp:
        launch_record = read_json(sftp, REMOTE+'/V1_P2_LAUNCH.json')
    last = None
    while True:
        if client is None:
            try:
                client = connection(password, host)
            except (OSError, EOFError, paramiko.SSHException) as error:
                status(dict(stage='reconnecting', experiments_started=True, error=type(error).__name__))
                time.sleep(60)
                continue
        try:
            with client.open_sftp() as sftp:
                try:
                    completed = read_json(sftp, REMOTE+'/'+PLUGIN+'/outputs/FULL_COMPLETED.json')
                except IOError:
                    completed = None
                try:
                    progress = read_json(sftp, REMOTE+'/'+PLUGIN+'/outputs/STATUS.json')
                except IOError:
                    progress = {'stage': 'initializing'}
                try:
                    current = read_json(sftp, REMOTE+'/'+PLUGIN+'/outputs/CURRENT_FIT.json')
                except IOError:
                    current = None
            marker = (progress.get('stage'), progress.get('completed_fits'), progress.get('completed_test_records'))
            if marker != last:
                status(dict(stage='running', experiments_started=True, progress=progress, current_fit=current))
                last = marker
            if completed:
                if not completed['all_validation_stopped'] or not completed['all_selected_checkpoints_trained']:
                    raise ValueError('Completion marker violates training protocol')
                break
            process_check = 'from pathlib import Path; p=Path("/proc/%d/cmdline"); print(p.exists() and %r in p.read_bytes().decode().replace("\\0"," "))' % (launch_record['pid'], REMOTE+'/'+PLUGIN+'/run.py')
            alive = remote_run(client, '/usr/bin/python3 -c '+quote(process_check)).strip()
            if alive != 'True':
                failure = remote_run(client, 'tail -n 60 '+quote(REMOTE+'/V1_P2_LAUNCH.log'))
                (DEST/'服务器异常日志.txt').write_text(failure, encoding='utf-8')
                status(dict(stage='training_failed', experiments_started=True, progress=progress,
                    error='Own experiment process stopped before completion; preserve checkpoints and inspect error'))
                return
            time.sleep(60)
        except (OSError, EOFError, paramiko.SSHException) as error:
            status(dict(stage='reconnecting', experiments_started=True, error=type(error).__name__))
            client.close()
            client = None
            time.sleep(60)
    del password
    pack = f'''from pathlib import Path
import zipfile
r=Path({REMOTE!r})
o=r/{(PLUGIN+'/outputs')!r}
with zipfile.ZipFile(str(r/'completed_v1_p2_results.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in o.rglob('*'):
        if p.is_file() and p.name != 'resume.pt': z.write(str(p),str(p.relative_to(o)))
    for name in ['V1_P2_LAUNCH.json','V1_P2_LAUNCH.log','INPUT_DOWNLOAD_COMPLETE.json']:
        z.write(str(r/name),'provenance/'+name)
print('ARCHIVE_READY')
'''
    remote_run(client, '/usr/bin/python3 -c '+quote(pack))
    archive = DEST/'完成结果.zip'
    with client.open_sftp() as sftp:
        sftp.get(REMOTE+'/completed_v1_p2_results.zip', str(archive))
    client.close()
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if os.path.commonpath([str(DEST), str((DEST/name).resolve())]) != str(DEST):
                raise ValueError('Unsafe downloaded archive member')
        z.extractall(DEST)
    primary = json.loads((DEST/'RESULTS_180.json').read_text())
    rows = list(csv.DictReader((DEST/'test_seed_results.csv').open(encoding='utf-8-sig')))
    if len(primary)!=180 or len(rows)!=1080:
        raise ValueError('Incomplete downloaded results')
    import numpy as np
    frozen = BASE/'experiment_vcs/v1_frozen_inputs_20261006/plugins/adapters'
    verified = []
    for row in rows:
        d, h, m = row['domain'], row['horizon'], row['model']
        with np.load(frozen/m/d/h/'test.npz') as obj:
            base, target = obj['base_pred'].astype(np.float32), obj['target'].astype(np.float32)
        with np.load(DEST/'predictions'/row['version']/row['seed']/d/h/(m+'.npz')) as obj:
            pred = obj['pred']
        for k in ['mse','mae']:
            e = pred.astype(np.float64)-target.astype(np.float64)
            eb = base.astype(np.float64)-target.astype(np.float64)
            val = float(np.mean(e**2 if k=='mse' else np.abs(e)))
            original = float(np.mean(eb**2 if k=='mse' else np.abs(eb)))
            if not np.isclose(val, float(row[k]), rtol=1e-10, atol=1e-12):
                raise ValueError('Downloaded prediction does not match metric')
            if not np.isclose(original, float(row['baseline_'+k]), rtol=1e-10, atol=1e-12):
                raise ValueError('Baseline changed')
        verified.append(row)
    for row in primary:
        seeds = [r for r in verified if r['version']=='V1-P2' and
            (r['model'],r['domain'],int(r['horizon'])) == (row['model'],row['domain'],row['horizon'])]
        if len(seeds)!=3:
            raise ValueError('Missing primary seeds')
        for k in ['mse','mae']:
            if not np.isclose(row[k],np.mean([float(r[k]) for r in seeds]),rtol=1e-12):
                raise ValueError('Aggregate mismatch')
    status(dict(stage='downloaded_and_verified', experiments_started=True, completed=completed,
        verified_prediction_rows=len(verified), archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
    target = REPO/PLUGIN/'reports/20261006'
    target.mkdir(parents=True, exist_ok=True)
    for p in DEST.iterdir():
        if p.is_file() and p.suffix in ['.csv','.json','.md','.png']:
            shutil.copyfile(p, target/p.name)
    for p in DEST.rglob('fit_result.json'):
        t = target/p.relative_to(DEST)
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, t)
    relative = target.relative_to(REPO).as_posix()
    subprocess.run(['git','-c','core.autocrlf=false','add','--',relative],cwd=REPO,check=True)
    changes = subprocess.check_output(['git','diff','--cached','--name-only','--',relative],cwd=REPO,text=True)
    if changes.strip():
        subprocess.run(['git','-c','core.autocrlf=false','commit','--only','-m',
            'Archive V1-P2 trained-output hyperparameter experiment and original comparisons','--',relative],cwd=REPO,check=True)
        subprocess.run(['git','push','origin','HEAD:main'],cwd=REPO,check=True)
    status(dict(stage='complete_downloaded_verified_and_pushed', experiments_started=True, completed=completed))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status(dict(stage='manager_failed', error=type(error).__name__, message=str(error)))
        raise
