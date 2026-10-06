"""Package the isolated, already reviewed source without overwriting its edits."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import sys
import zipfile

BASE=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent.parent
REPO=BASE/'experiment_vcs/server_baseline'
SOURCE=REPO/'plugins/v1_no_variance_rerun_20261006'
DEST=REPO/'plugins/v1_parameter_experiment_20261006'
for p in DEST.glob('*.py'):
    ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
manifest=dict(source_commit='b5e4d91',backup_tag='pre_v1_hparam_20261006',
              convergence_backup_tag='pre_v1_p1_convergence_20261006',
              original_sources=json.loads((SOURCE/'SOURCE_SNAPSHOT.json').read_text()),
              modification='Teacher margin; matched formal convergence, raw five-model plateau and LR schedule; screening unchanged',
              files={name:hashlib.sha256((DEST/name).read_bytes()).hexdigest() for name in ['run.py','experiment.py','model.py','V1参数实验.md']})
(DEST/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
local_doc=BASE/'V1参数实验_20261006/V1参数实验.md'
local_doc.parent.mkdir(exist_ok=True)
shutil.copyfile(DEST/'V1参数实验.md',local_doc)
archive=BASE/'experiment_vcs/v1_parameter_code_20261006.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in DEST.iterdir():
        if p.is_file():z.write(p,'plugins/v1_parameter_experiment_20261006/'+p.name)
    for name in ['__init__.py','data.py','model.py']:
        z.write(REPO/'plugins/semflow'/name,'plugins/semflow/'+name)
print('PREPARED',str(archive))
