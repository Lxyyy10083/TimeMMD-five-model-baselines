"""Detach result collection; pass the SSH credential only through an anonymous pipe."""
from pathlib import Path
import getpass
import json
import subprocess
import sys
import time

base=Path(sys.argv[1]).resolve()
dest=base/'V1参数实验_20261006'
dest.mkdir(exist_ok=True)
password=getpass.getpass('Lab password for background result collection: ')
with (dest/'自动收集.log').open('a',encoding='utf-8') as log:
    worker=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('download_results.py')),
        str(base),'--watch','--credential-stdin'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,
        creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW)
    worker.stdin.write((password+'\n').encode());worker.stdin.close()
del password
record=dict(pid=worker.pid,started_local=time.strftime('%Y-%m-%d %H:%M:%S'),
    script='download_results.py',poll_interval_seconds=50,credentials='memory and anonymous pipe only')
(dest/'自动收集进程.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False),flush=True)
