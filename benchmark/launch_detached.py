"""Run the resumable benchmark independently of the SSH session."""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / 'runs' / 'driver.log'
LOG.parent.mkdir(exist_ok=True)
cmd = [sys.executable, '-u', str(HERE / 'run_all.py'), '--retry-errors',
       '--epochs', '100', '--workers', '3']
with LOG.open('a', encoding='utf-8') as output:
    proc = subprocess.Popen(cmd, cwd=HERE.parent, stdin=subprocess.DEVNULL,
                            stdout=output, stderr=subprocess.STDOUT,
                            start_new_session=True, close_fds=True,
                            env=os.environ.copy())
(HERE / 'runs' / 'driver.pid').write_text(str(proc.pid), encoding='ascii')
print(f'Launched benchmark process {proc.pid}')
