"""Launch the final aligned Readgpt benchmark independently of SSH."""
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
here = root / 'benchmark'
runs = here / 'readgpt_runs'
runs.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env['BENCHMARK_DATA_DIR'] = str(here / 'readgpt_data')
env['BENCHMARK_RUNS_DIR'] = str(runs)
cmd = [sys.executable, '-u', str(here / 'run_all.py'), '--retry-errors',
       '--epochs', '200', '--workers', '3']
with (runs / 'driver.log').open('a', encoding='utf-8') as stream:
    proc = subprocess.Popen(cmd, cwd=root, stdin=subprocess.DEVNULL,
                            stdout=stream, stderr=subprocess.STDOUT,
                            start_new_session=True, close_fds=True, env=env)
(runs / 'driver.pid').write_text(str(proc.pid), encoding='ascii')
print(f'Launched aligned Readgpt benchmark process {proc.pid}')
