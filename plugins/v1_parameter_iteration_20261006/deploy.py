"""Safe isolated bootstrap. Run with system Python; training uses lxy only."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

ALLOWED = Path('/xiliang/LXY')
ROOT = ALLOWED/'baseline_v1_p2_20261006'
PREVIOUS = ALLOWED/'baseline_v1_parameter_20261006'
PLUGIN_NAME = 'v1_parameter_iteration_20261006'
PLUGIN = ROOT/'plugins'/PLUGIN_NAME
PACKAGE = ALLOWED/'v1_p2_code_20261006.zip'


def inside(path):
    if os.path.commonpath([str(path.resolve()), str(ALLOWED)]) != str(ALLOWED):
        raise ValueError('Path outside /xiliang/LXY')


def main():
    commit = sys.argv[1]
    inside(ROOT)
    ROOT.mkdir(exist_ok=True)
    record_path = ROOT/'V1_P2_LAUNCH.json'
    if record_path.exists():
        launch = json.loads(record_path.read_text())
        proc = Path('/proc/%d' % launch['pid'])
        if proc.exists() and str(PLUGIN/'run.py') in (proc/'cmdline').read_bytes().decode().replace('\0', ' '):
            print(json.dumps(dict(already_running=True, launch=launch)))
            return
        completed = PLUGIN/'outputs/FULL_COMPLETED.json'
        if completed.exists():
            print(json.dumps(dict(already_completed=True, launch=launch)))
            return
        raise RuntimeError('Previous own job stopped before completion; inspect logs before resuming')
    if not (PREVIOUS/'INPUT_DOWNLOAD_COMPLETE.json').exists():
        raise RuntimeError('Frozen original inputs not found')
    marker = json.loads((PREVIOUS/'INPUT_DOWNLOAD_COMPLETE.json').read_text())
    for r in marker['records']:
        src, dst = PREVIOUS/r['path'], ROOT/r['path']
        inside(src)
        inside(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            try:
                os.link(str(src), str(dst))
            except OSError:
                shutil.copyfile(str(src), str(dst))
        if hashlib.sha256(dst.read_bytes()).hexdigest() != r['sha256']:
            raise ValueError('Frozen input mismatch: '+r['path'])
    shutil.copyfile(str(PREVIOUS/'INPUT_DOWNLOAD_COMPLETE.json'), str(ROOT/'INPUT_DOWNLOAD_COMPLETE.json'))
    with zipfile.ZipFile(str(PACKAGE)) as z:
        for name in z.namelist():
            member = ROOT/name
            inside(member)
            if os.path.commonpath([str(member.resolve()), str(ROOT)]) != str(ROOT):
                raise ValueError('Unsafe archive member')
        z.extractall(str(ROOT))
    memory = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total',
        '--format=csv,noheader,nounits'], universal_newlines=True).splitlines()
    memory = [tuple(map(int, v.split(','))) for v in memory]
    idle = [i for i,(used,total) in enumerate(memory) if used < 100]
    eligible = [i for i,(used,total) in enumerate(memory) if total-used >= 8192]
    if not eligible:
        raise RuntimeError('Less than 8GB free on each GPU; leave other processes alone')
    gpu = str(idle[0] if idle else max(eligible, key=lambda i: memory[i][1]-memory[i][0]))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, V1_P2_GPU=gpu,
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4',
        HF_HOME='/xiliang/LXY/.cache/huggingface', XDG_CACHE_HOME='/xiliang/LXY/.cache',
        TORCH_HOME='/xiliang/LXY/.cache/torch', MPLCONFIGDIR='/xiliang/LXY/.cache/matplotlib')
    log = (ROOT/'V1_P2_LAUNCH.log').open('a')
    process = subprocess.Popen(['/xiliang/LXY/envs/lxy/bin/python','-u',str(PLUGIN/'run.py')],
        cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(pid=process.pid, code_commit=commit, gpu=int(gpu),
        sharing=not bool(idle), torch_memory_fraction=.05, root=str(ROOT),
        gpu_initial_used_mib=memory[int(gpu)][0],
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    record_path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


if __name__ == '__main__':
    main()
