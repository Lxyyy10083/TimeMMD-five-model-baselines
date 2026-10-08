"""只核验/清理已备份的旧V3工作目录；模型资产和本轮r3不在删除范围。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import stat
import sys
import time

ALLOWED = Path('/xiliang/LXY')
TARGET = ALLOWED / 'baseline_v3_lab_20261004'
CONTROL = ALLOWED / 'maintenance_archive_20261008'


def checked_target():
    if TARGET.is_symlink() or TARGET.resolve(strict=True) != TARGET:
        raise ValueError('Target must be the exact authorized directory, without symlinks')
    if TARGET.parent != ALLOWED or TARGET == ALLOWED or not TARGET.is_dir():
        raise ValueError('Invalid cleanup target')
    return TARGET


def candidate(relative):
    p = Path(relative)
    # 重要的预训练资产、数据、源码和指标保留；只去掉旧训练状态和已安装的wheel副本。
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Invalid relative path')
    return (p.parts[0] == 'plugins' and p.name in ('checkpoint.pt', 'resume.pt', 'resume.tmp')) or (
        p.parts[0] == 'offline_wheels' and p.suffix == '.whl')


def active_legacy_processes():
    hits = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            cwd = os.readlink(proc/'cwd')
            opened_state = False
            try:
                for fd in (proc/'fd').iterdir():
                    try:
                        name = os.readlink(fd)
                        if name.startswith(str(TARGET)+'/') and candidate(name[len(str(TARGET))+1:]):
                            opened_state = True
                            break
                    except OSError:
                        continue
            except (OSError, PermissionError):
                pass
            if str(TARGET) in args or cwd == str(TARGET) or cwd.startswith(str(TARGET)+'/') or opened_state:
                hits.append(int(proc.name))
        except (OSError, PermissionError):
            continue
    return hits


def inventory():
    root = checked_target()
    active = active_legacy_processes()
    if active:
        raise RuntimeError('Legacy processes still active: ' + str(active))
    entries = []
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs.sort(); files.sort()
        for name in dirs + files:
            path = Path(base)/name
            info = path.lstat()
            item = dict(path=path.relative_to(root).as_posix(), mode=info.st_mode,
                        mtime_ns=info.st_mtime_ns, inode=info.st_ino, device=info.st_dev,
                        nlink=info.st_nlink)
            if stat.S_ISLNK(info.st_mode):
                item.update(kind='symlink', link=os.readlink(path), size=0, cleanup=False)
            elif stat.S_ISDIR(info.st_mode):
                item.update(kind='directory', size=0, cleanup=False)
            elif stat.S_ISREG(info.st_mode):
                item.update(kind='file', size=info.st_size, cleanup=candidate(item['path']))
            else:
                raise ValueError('Unsupported special file: ' + str(path))
            entries.append(item)
    return dict(root=str(root), created_unix=time.time(), entries=entries)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            block = f.read(8*1024*1024)
            if not block:
                break
            value.update(block)
    return value.hexdigest()


def check_entry(root, item):
    relative = Path(item['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Manifest path traversal')
    path = root/relative
    if path.is_symlink() or root not in path.resolve(strict=True).parents:
        raise ValueError('Unsafe candidate path: ' + item['path'])
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or (info.st_size, info.st_mtime_ns, info.st_ino, info.st_dev) != (
        item['size'], item['mtime_ns'], item['inode'], item['device']):
        raise ValueError('File changed after backup: ' + item['path'])
    if not candidate(item['path']) or not item.get('cleanup'):
        raise ValueError('File is not in the cleanup allowlist')
    return path


def verify_and_cleanup(receipt, execute=False):
    root = checked_target()
    if receipt.get('root') != str(root) or not receipt.get('local_verified') or not receipt.get('git_commit'):
        raise ValueError('Verified local archive and Git receipt are required')
    if active_legacy_processes():
        raise RuntimeError('An old experiment is active; cleanup refused')
    selected = [item for item in receipt['entries'] if item.get('cleanup')]
    # 先核验所有待删文件，再删除；任何一项校验失败都会保留所有尚未删除的文件。
    for index, item in enumerate(selected):
        path = check_entry(root, item)
        if digest(path) != item['sha256']:
            raise ValueError('Source checksum differs from local backup: ' + item['path'])
        if index % 20 == 0:
            print(json.dumps(dict(stage='verifying', index=index, total=len(selected))), flush=True)
    if active_legacy_processes():
        raise RuntimeError('An old experiment became active; cleanup refused')
    removed_bytes = 0
    deleted = []
    if execute:
        for item in selected:
            path = check_entry(root, item)
            # unlink只针对逐项核验过的普通文件，不递归删除目录，不跟随符号链接。
            path.unlink()
            removed_bytes += item['size']
            deleted.append(item['path'])
            (CONTROL/'deleted_files.json').write_text(json.dumps(deleted), encoding='utf-8')
    result = dict(executed=execute, candidates=len(selected), deleted=len(deleted),
                  removed_file_bytes=removed_bytes, git_commit=receipt['git_commit'],
                  local_archive=receipt['local_archive'], kept=['models', 'benchmark', 'source', 'metrics', 'predictions'],
                  finished_unix=time.time())
    (CONTROL/('CLEANUP_COMPLETED.json' if execute else 'VERIFIED.json')).write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    return result


def main():
    if sys.prefix != '/xiliang/LXY/envs/lxy':
        raise RuntimeError('Only the lxy environment is authorized')
    os.chdir(ALLOWED)
    CONTROL.mkdir(exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['inventory', 'verify', 'cleanup'])
    args = p.parse_args()
    if args.action == 'inventory':
        data = inventory()
        (CONTROL/'inventory.json').write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        files = [r for r in data['entries'] if r['kind'] == 'file']
        print(json.dumps(dict(files=len(files), bytes=sum(r['size'] for r in files),
            candidates=sum(r['cleanup'] for r in files), cleanup_file_bytes=sum(r['size'] for r in files if r['cleanup']))))
    else:
        receipt = json.loads((CONTROL/'verified_local_receipt.json').read_text(encoding='utf-8'))
        verify_and_cleanup(receipt, execute=args.action == 'cleanup')


if __name__ == '__main__':
    main()
