"""Start the local deployment/collection manager with no credential file."""
from pathlib import Path
import getpass
import json
import subprocess
import sys
import time

base = Path(sys.argv[1]).resolve()
dest = base/'V1-P2参数实验_20261006'
dest.mkdir(exist_ok=True)
password = getpass.getpass('Lab password for V1-P2 background job: ')
with (dest/'自动任务.log').open('a', encoding='utf-8') as log:
    worker = subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('manage.py')),
        str(base),'--credential-stdin'], stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
        creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW)
    worker.stdin.write((password+'\n').encode())
    worker.stdin.close()
del password
record = dict(pid=worker.pid, started_local=time.strftime('%Y-%m-%d %H:%M:%S'),
    script='manage.py', initial_network_retry_hours=6, poll_seconds=60,
    credentials='memory and anonymous pipe only')
(dest/'自动任务进程.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(record, ensure_ascii=False), flush=True)
