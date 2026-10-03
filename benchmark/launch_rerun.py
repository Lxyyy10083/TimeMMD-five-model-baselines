"""Keep convergence retries running after the SSH session closes."""
import os
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
runs = here / 'runs'
runs.mkdir(exist_ok=True)
with (runs / 'rerun.log').open('a', encoding='utf-8') as output:
    process = subprocess.Popen([sys.executable, '-u', str(here / 'rerun_unconverged.py')],
                               cwd=here.parent, stdin=subprocess.DEVNULL,
                               stdout=output, stderr=subprocess.STDOUT,
                               start_new_session=True, close_fds=True,
                               env=os.environ.copy())
(runs / 'rerun.pid').write_text(str(process.pid), encoding='ascii')
print(f'Launched convergence retries {process.pid}')
