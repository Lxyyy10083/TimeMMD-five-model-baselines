"""Download completed server summaries to the local baseline folder.

Run locally in an interactive terminal. Password stays in process memory;
the script writes no credential or private key to disk.
"""
import getpass
import subprocess
import time
from pathlib import Path

import paramiko

HOST = 'connect.nmb1.seetacloud.com'
PORT = 19361
FINGERPRINT = 'd1061d72b331386b67deb57583b5e6c1'
REMOTE = '/root/autodl-tmp/carma_workspace/plugins'
LOCAL = Path(r'C:\Users\32113\OneDrive\Desktop\baseline\CARMA_results')


def connect(password):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username='root', password=password,
                   timeout=30, auth_timeout=30)
    actual = client.get_transport().get_remote_server_key().get_fingerprint().hex()
    if actual != FINGERPRINT:
        client.close()
        raise RuntimeError('SSH host key changed')
    return client


def main():
    password = getpass.getpass('SSH password for result monitor: ')
    operation_marked = False
    while True:
        try:
            with connect(password) as client:
                with client.open_sftp() as sftp:
                    try:
                        sftp.stat(f'{REMOTE}/runs/summary_finished.txt')
                    except FileNotFoundError:
                        print('Experiment still running', flush=True)
                    else:
                        LOCAL.mkdir(parents=True, exist_ok=True)
                        files = {'results_180.csv': 'results_180.csv',
                                 'SUMMARY.md': 'SUMMARY.md',
                                 'EXPERIMENT_LOG.md': 'EXPERIMENT_LOG.md',
                                 'runs/carma_status.csv': 'carma_status.csv',
                                 'runs/full_launcher.log': 'full_launcher.log'}
                        for remote, local in files.items():
                            sftp.get(f'{REMOTE}/{remote}', str(LOCAL / local))
                        marker = Path(r'C:\Users\32113\.codex\plugins\cache\openai-primary-runtime\spreadsheets\26.909.11814\skills\spreadsheets\container_tools\mark_artifact_operation_started.mjs')
                        builder = Path(__file__).with_name('build_result_workbook.mjs')
                        if not operation_marked:
                            subprocess.run(['node', str(marker), '--operation-kind', 'create',
                                            '--expected-output-count', '1', '--output-format', 'xlsx'],
                                           cwd=builder.parent, check=True)
                            operation_marked = True
                        subprocess.run(['node', str(builder)], cwd=builder.parent, check=True)
                        print(f'Downloaded final reports to {LOCAL}', flush=True)
                        return
        except Exception as exc:
            print(f'Monitor retry: {type(exc).__name__}: {exc}', flush=True)
        time.sleep(45)


if __name__ == '__main__':
    main()
