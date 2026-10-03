"""Resume only Aurora and combine with four preserved, completed models."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
PLUGIN=ROOT/'plugins/internal_semflow'
ORIGINAL=PLUGIN/'outputs_full_20261003'
AURORA=PLUGIN/'outputs_full_aurora_v2_20261004'


def main():
    AURORA.mkdir(exist_ok=True)
    (AURORA/'DESIGN_LOCKED.json').write_text((ORIGINAL/'DESIGN_LOCKED.json').read_text(),encoding='utf-8')
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg')
    for stage in ['train','evaluate']:
        command=[sys.executable,'-u',str(PLUGIN/'launch.py'),'--suite','full','--models','Aurora',
                 '--variants','control','internal','--stage',stage,'--output',str(AURORA)]
        with (AURORA/f'aurora_{stage}_driver.log').open('a',encoding='utf-8') as log:
            code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:
            (AURORA/'RESUME_FAILED.json').write_text(json.dumps(dict(stage=stage,code=code)),encoding='utf-8')
            return code
    code=subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('report.py')),
                         '--directory',str(ORIGINAL),'--aurora-directory',str(AURORA)],
                         cwd=ROOT,env=env).returncode
    if code:
        return code
    (ORIGINAL/'FULL_COMPLETED.json').write_text(json.dumps(dict(models=5,tasks=180,train_jobs=360,
        evaluation_records=360,aurora_directory=str(AURORA),
        note='Four original completed models retained; Aurora config fix only. Historical failure markers retained for audit.')),encoding='utf-8')
    print('ALL_FIVE_COMPLETED',flush=True)
    return 0


if __name__=='__main__':
    sys.exit(main())

