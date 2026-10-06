"""One-off local retrieval of this exact V1 rerun; password stays in memory."""
from pathlib import Path
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

BASE = next(p for p in Path(__file__).resolve().parents if (p / 'experiment_vcs/server_baseline/.git').exists())
REMOTE = '/xiliang/LXY/baseline_v1_no_variance_20261006'
OUTPUT = REMOTE + '/plugins/v1_no_variance_rerun_20261006/outputs'
DEST = BASE / 'GANF_V1_no_variance_rerun_20261006'
DEST.mkdir(exist_ok=True)
STATE = DEST / '自动下载状态.json'


def status(**fields):
    STATE.write_text(json.dumps(fields, ensure_ascii=False, indent=2), encoding='utf-8')


def archive(out):
    repo = BASE / 'experiment_vcs/server_baseline'
    relative = 'plugins/v1_no_variance_rerun_20261006/reports/20261006'
    dest = repo / relative;dest.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():
        if p.is_file() and p.suffix in ['.json', '.csv', '.md', '.png'] and p.name != STATE.name:
            shutil.copyfile(p, dest / p.name)
    subprocess.run(['git', 'add', '--', relative], cwd=repo, check=True)
    subprocess.run(['git', 'commit', '--only', '-m', 'Archive exact screenshot V1 rerun with three plugin seeds', '--', relative], cwd=repo, check=True)
    subprocess.run(['git', 'push', 'origin', 'HEAD:main'], cwd=repo, check=True)


def work(password):
    cfg = paramiko.SSHConfig()
    with Path.home().joinpath('.ssh/config').open() as f:cfg.parse(f)
    h = cfg.lookup('Remote')
    while True:
        c = paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.WarningPolicy())
        try:
            c.connect(h['hostname'], port=int(h['port']), username=h['user'], password=password,
                      timeout=15, auth_timeout=15, allow_agent=False, look_for_keys=False)
            with c.open_sftp() as s:
                def read(name):
                    try:
                        with s.open(OUTPUT + '/' + name) as f:return json.loads(f.read())
                    except OSError:return None
                completed = read('FULL_COMPLETED.json')
                if not completed:
                    failure = read('INCOMPLETE.json')
                    if failure:
                        status(stage='requires_attention', failure=failure, local_pid=os.getpid());return
                    status(stage='waiting', progress=read('STATUS.json'), local_pid=os.getpid())
                else:
                    assert completed['final_fits'] == 108 and completed['test_rows'] == 540
                    script = '''from pathlib import Path
import zipfile
r=Path('/xiliang/LXY/baseline_v1_no_variance_20261006')
o=r/'plugins/v1_no_variance_rerun_20261006/outputs'
with zipfile.ZipFile(r/'completed_v1_rerun_results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in o.rglob('*'):
  if p.is_file() and p.suffix in ['.json','.csv','.md','.png','.npz','.pt','.log']:z.write(p,p.relative_to(o))
 for name in ['INPUT_DOWNLOAD_COMPLETE.json','V1_LAUNCH.json','V1_LAUNCH.log']:
  if (r/name).exists():z.write(r/name,name)
'''
                    status(stage='packing', completed=completed, local_pid=os.getpid())
                    _, stdout, stderr = c.exec_command('/usr/bin/python3 -c ' + shlex.quote(script))
                    if stdout.channel.recv_exit_status():raise RuntimeError(stderr.read().decode(errors='replace'))
                    status(stage='downloading', local_pid=os.getpid())
                    archive_file = DEST.with_suffix('.zip')
                    s.get(REMOTE + '/completed_v1_rerun_results.zip', str(archive_file))
                    with zipfile.ZipFile(archive_file) as z:
                        for n in z.namelist():
                            if os.path.commonpath([str(DEST.resolve()), str((DEST / n).resolve())]) != str(DEST.resolve()):
                                raise ValueError('Unsafe archive member')
                        z.extractall(DEST)
                    rows = json.loads((DEST / 'RESULTS_180.json').read_text(encoding='utf-8'))
                    assert len(rows) == 180
                    try:archive(DEST);git = 'pushed_private_repository'
                    except Exception as e:git = 'results_downloaded_git_needs_attention: ' + str(e)
                    status(stage='complete', model_tasks=180, test_rows=540, local_output=str(DEST), git=git,
                           all_validation_plateau=completed['all_validation_plateau'])
                    return
        except Exception as error:
            status(stage='retrying_connection_or_download', error=str(error), local_pid=os.getpid())
        finally:c.close()
        time.sleep(300)


if __name__ == '__main__':
    if '--worker' in sys.argv:work(json.loads(sys.stdin.readline())['password'])
    else:
        password = getpass.getpass('Lab SSH password for V1 one-off retrieval: ')
        with (DEST / '自动下载日志.log').open('a', encoding='utf-8') as f:
            p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'], cwd=BASE,
                                 stdin=subprocess.PIPE, stdout=f, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0, text=True)
            p.stdin.write(json.dumps(dict(password=password)) + '\n');p.stdin.close()
        del password
        print('V1_RESULT_RETRIEVAL_STARTED', p.pid)
