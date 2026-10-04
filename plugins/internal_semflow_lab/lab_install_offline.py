from pathlib import Path
import os
import subprocess

ROOT=Path('/xiliang/LXY/baseline_v3_lab_20261004')
PY='/xiliang/LXY/envs/lxy/bin/python'
os.environ.update(PIP_CACHE_DIR='/xiliang/LXY/.cache/pip',PIP_CONFIG_FILE='/dev/null',
    PIP_EXTRA_INDEX_URL='',PIP_DISABLE_PIP_VERSION_CHECK='1',TMPDIR='/xiliang/LXY/tmp',
    XDG_CACHE_HOME='/xiliang/LXY/.cache',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
    CUDA_VISIBLE_DEVICES='1')
cmd=[PY,'-m','pip','install','--no-index','--find-links',str(ROOT/'offline_wheels'),
    'torch==2.5.1','torchvision==0.20.1','numpy==1.26.4','pandas==2.2.3','scipy==1.14.1',
    'scikit-learn==1.5.2','matplotlib==3.9.2','transformers==4.40.2','einops==0.8.0',
    'sentencepiece==0.2.0','networkx==3.4.2']
print(cmd,flush=True)
subprocess.run(cmd,check=True)
subprocess.run([PY,'-c',"import sys,torch,transformers; print(sys.prefix,torch.__version__,transformers.__version__); assert sys.prefix=='/xiliang/LXY/envs/lxy'; assert torch.cuda.is_available()"],check=True)
(ROOT/'ENV_READY').write_text('lxy offline setup ready\n')
print('ENV_READY',flush=True)
