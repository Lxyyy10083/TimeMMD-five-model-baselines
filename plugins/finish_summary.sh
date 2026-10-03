#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/carma_workspace
while ! grep -q '^ATTEMPTED 180$' plugins/runs/full_launcher.log 2>/dev/null; do
    sleep 30
done
/root/miniconda3/bin/python plugins/summarize_results.py
date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' > plugins/runs/summary_finished.txt
