"""Filesystem bootstrap only; experiments run exclusively with lxy Python."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT=Path('/xiliang/LXY/baseline_v1_parameter_20261006')
PLUGIN=ROOT/'plugins/v1_parameter_experiment_20261006'
OUT=PLUGIN/'outputs'
commit=sys.argv[1]
assert os.path.commonpath([str(ROOT.resolve()),'/xiliang/LXY'])=='/xiliang/LXY'
active=json.loads((ROOT/'V1_P1_LAUNCH.json').read_text())
assert not Path('/proc/%d'%active['pid']).exists(), 'Previous own task still active'
memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],universal_newlines=True).splitlines()
assert int(memory[2])<100, 'GPU2 is no longer idle; do not touch other processes'
history=ROOT/'source_before_convergence_v3'
if not history.exists():
    shutil.copytree(str(PLUGIN),str(history),ignore=shutil.ignore_patterns('outputs','__pycache__'))
if not (OUT/'PLAN_initial_budget80.json').exists():
    shutil.copyfile(str(OUT/'PLAN.json'),str(OUT/'PLAN_initial_budget80.json'))
if not (OUT/'PLAN_convergence_v2.json').exists():
    shutil.copyfile(str(OUT/'PLAN.json'),str(OUT/'PLAN_convergence_v2.json'))
if not (ROOT/'V1_P1_PRE_CONVERGENCE_LAUNCH.json').exists():
    shutil.copyfile(str(ROOT/'V1_P1_LAUNCH.json'),str(ROOT/'V1_P1_PRE_CONVERGENCE_LAUNCH.json'))
with zipfile.ZipFile('/xiliang/LXY/v1_parameter_code_20261006.zip') as z:
    for name in z.namelist():
        assert os.path.commonpath([str((ROOT/name).resolve()),str(ROOT)])==str(ROOT)
    z.extractall(str(ROOT))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    HF_HOME='/xiliang/LXY/.cache/huggingface',XDG_CACHE_HOME='/xiliang/LXY/.cache',
    TORCH_HOME='/xiliang/LXY/.cache/torch',MPLCONFIGDIR='/xiliang/LXY/.cache/matplotlib')
log=(ROOT/'V1_P1_LAUNCH.log').open('a')
p=subprocess.Popen(['/xiliang/LXY/envs/lxy/bin/python','-u',str(PLUGIN/'run.py')],
    cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,code_commit=commit,protocol='v3_raw_five_model_plateau',
    gpu=2,root=str(ROOT),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
(ROOT/'V1_P1_LAUNCH.json').write_text(json.dumps(record,indent=2))
(ROOT/'V1_P1_CONVERGENCE_V3_LAUNCH.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
