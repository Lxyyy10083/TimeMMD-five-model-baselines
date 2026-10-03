"""One-time migration after fixing validation checkpoint consistency."""
import csv
import shutil
from pathlib import Path

path = Path(__file__).resolve().parent / 'runs' / 'results.csv'
backup = path.with_name('results_pre_checkpoint_fix.csv')
if path.exists() and not backup.exists():
    shutil.copy2(path, backup)
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
        fields = list(rows[0]) if rows else []
    keep = [row for row in rows if row['model'] == 'Aurora' and row['status'] == 'ok']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(keep)
    print(f'Kept {len(keep)} Aurora results; archived {len(rows) - len(keep)} older rows.')
