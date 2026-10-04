"""Wait for installation, then run experiments only with the lxy interpreter."""
from pathlib import Path
import json
import os
import subprocess
import time

ROOT=Path('/xiliang/LXY/baseline_v3_lab_20261004')
READY=ROOT/'ENV_READY'
while not READY.exists():
    pidfile=ROOT/'env_setup.pid'
    if pidfile.exists() and not Path('/proc',pidfile.read_text().strip()).exists():
        (ROOT/'LAUNCH_FAILED.json').write_text(json.dumps({'reason':'environment installation failed','log':'env_setup.log'}))
        raise RuntimeError('environment setup failed; no experiments started')
    time.sleep(15)
memory=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
if int(memory.strip())>100:
    (ROOT/'LAUNCH_FAILED.json').write_text(json.dumps({'reason':'GPU1 occupied; no processes touched'}))
    raise RuntimeError('GPU1 is occupied; refusing to compete with another job')
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
           XDG_CACHE_HOME='/xiliang/LXY/.cache',TMPDIR='/xiliang/LXY/tmp')
print('ENV_READY; starting real paired training',flush=True)
subprocess.run(['/xiliang/LXY/envs/lxy/bin/python','-u',
               str(ROOT/'plugins/internal_semflow_lab/run_pilot.py')],cwd=ROOT,env=env,check=True)
