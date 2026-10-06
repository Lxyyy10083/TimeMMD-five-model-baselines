"""Freeze experiment source and package only the new plugin plus data helpers."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
base = Path(sys.argv[1]).resolve()
files = ['run.py', 'model.py', 'deploy.py', 'V1-P2参数实验.md']
manifest = dict(version='V1-P2', backup_tag='pre_v1_p2_iteration_20261006',
    backup_commit='d40d592', files={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in files},
    untouched_baselines=True, original_v1_source='plugins/v1_parameter_experiment_20261006/model.py')
(HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
package = base/'experiment_vcs/v1_p2_code_20261006.zip'
with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as z:
    for name in [*files, 'SOURCE_MANIFEST.json']:
        p = HERE/name
        z.write(p, p.relative_to(ROOT).as_posix())
    for name in ['__init__.py', 'data.py', 'model.py']:
        p = HERE.parent/'semflow'/name
        z.write(p, p.relative_to(ROOT).as_posix())
commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
(base/'experiment_vcs/v1_p2_package_metadata_20261006.json').write_text(json.dumps(
    dict(code_commit=commit, package_sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
         version='V1-P2'), indent=2), encoding='utf-8')
print(str(package))
