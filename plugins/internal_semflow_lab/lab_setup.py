"""Create exactly one isolated lab environment; every writable path is under LXY."""
import os
from pathlib import Path
import subprocess

BASE = Path('/xiliang/LXY')
ENV = BASE/'envs/lxy'
TOOLS = BASE/'baseline_tools'
for p in [BASE/'.cache/pip',BASE/'.cache/uv',BASE/'.cache/huggingface',BASE/'.cache/matplotlib',
          BASE/'tmp',TOOLS,BASE/'python_runtime']:
    p.mkdir(parents=True,exist_ok=True)
os.environ.update(PIP_CACHE_DIR=str(BASE/'.cache/pip'),UV_CACHE_DIR=str(BASE/'.cache/uv'),
                  UV_PYTHON_INSTALL_DIR=str(BASE/'python_runtime'),TMPDIR=str(BASE/'tmp'),
                  XDG_CACHE_HOME=str(BASE/'.cache'),HF_HOME=str(BASE/'.cache/huggingface'),
                  MPLCONFIGDIR=str(BASE/'.cache/matplotlib'),PYTHONDONTWRITEBYTECODE='1',
                  PYTHONNOUSERSITE='1',UV_PYTHON_BIN_DIR=str(TOOLS/'python_bin'),
                  UV_HTTP_TIMEOUT='120',PIP_CONFIG_FILE='/dev/null',PIP_EXTRA_INDEX_URL='',
                  PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_DEFAULT_TIMEOUT='60')


def run(cmd):
    print('RUN',cmd,flush=True)
    subprocess.run(cmd,env=os.environ,check=True)


if not (TOOLS/'bin/uv').exists():
    run(['/opt/conda/bin/python','-m','pip','install','--target',str(TOOLS),
         '--index-url','https://pypi.org/simple','uv==0.8.22'])
if not (ENV/'bin/python').exists():
    runtime=BASE/'python_runtime/python/bin/python3.10'
    run([str(TOOLS/'bin/uv'),'venv','--python',str(runtime) if runtime.exists() else '3.10','--seed',str(ENV)])
run([str(ENV/'bin/python'),'-m','pip','install','torch==2.5.1','torchvision==0.20.1',
     '--index-url','https://download.pytorch.org/whl/cu121'])
run([str(ENV/'bin/python'),'-m','pip','install','--index-url','https://pypi.org/simple',
     'numpy==1.26.4','pandas==2.2.3','scipy==1.14.1','scikit-learn==1.5.2',
     'matplotlib==3.9.2','transformers==4.40.2','einops==0.8.0','sentencepiece==0.2.0'])
run([str(ENV/'bin/python'),'-c',
     "import sys,torch,transformers; print(sys.prefix,torch.__version__,transformers.__version__); assert sys.prefix=='/xiliang/LXY/envs/lxy'; assert torch.cuda.is_available()"])
(BASE/'baseline_v3_lab_20261004/ENV_READY').write_text('lxy\n')
print('LXY_ENVIRONMENT_READY',flush=True)
