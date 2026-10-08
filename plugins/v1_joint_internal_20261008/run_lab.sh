#!/usr/bin/env bash
# V1-JOINT-r2：仅在LXY下运行、仅使用lxy环境；日志与结果按上海日期/迭代版本归档。
set -Eeuo pipefail
umask 077

readonly LXY_ALLOWED_ROOT='/xiliang/LXY'
readonly LXY_ENV_ROOT='/xiliang/LXY/envs/lxy'
readonly LXY_ITERATION='V1-JOINT-r2'
LXY_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LXY_INPUTS_ROOT="${LXY_JOINT_INPUTS_ROOT:-/xiliang/LXY/baseline_v1_parameter_20261006}"
LXY_MODELS_ROOT="${LXY_JOINT_MODELS_ROOT:-/xiliang/LXY/baseline_v3_lab_20261004/models}"
LXY_RUN_DATE="${LXY_JOINT_RUN_DATE:-$(TZ=Asia/Shanghai date +%Y%m%d)}"

lxy_require_inside() {
    local lxy_resolved
    lxy_resolved="$(readlink -m -- "$1")"
    case "$lxy_resolved" in
        "$LXY_ALLOWED_ROOT"|"$LXY_ALLOWED_ROOT"/*) ;;
        *) printf 'Refusing path outside LXY: %s\n' "$lxy_resolved" >&2; exit 2 ;;
    esac
}

for LXY_PATH in "$LXY_REPO_ROOT" "$LXY_ENV_ROOT" "$LXY_INPUTS_ROOT" "$LXY_MODELS_ROOT" "$LXY_ALLOWED_ROOT/log"; do
    lxy_require_inside "$LXY_PATH"
done
[[ "$LXY_RUN_DATE" =~ ^[0-9]{8}$ ]] || { printf 'Invalid run date\n' >&2; exit 2; }
[[ -f "$LXY_ENV_ROOT/bin/activate" && -x "$LXY_ENV_ROOT/bin/python" ]] || {
    printf 'The authorized lxy environment is unavailable\n' >&2; exit 2;
}
source "$LXY_ENV_ROOT/bin/activate"
readonly LXY_PYTHON="$LXY_ENV_ROOT/bin/python"
"$LXY_PYTHON" -c 'import sys; assert sys.prefix == "/xiliang/LXY/envs/lxy", sys.prefix'

LXY_CODE_REVISION="${LXY_JOINT_CODE_REVISION:-}"
if [[ -z "$LXY_CODE_REVISION" && -f "$LXY_REPO_ROOT/CODE_VERSION.json" ]]; then
    LXY_CODE_REVISION="$("$LXY_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["code_commit"])' "$LXY_REPO_ROOT/CODE_VERSION.json")"
elif [[ -z "$LXY_CODE_REVISION" ]]; then
    LXY_CODE_REVISION="$(git -C "$LXY_REPO_ROOT" rev-parse HEAD)"
fi
[[ "$LXY_CODE_REVISION" =~ ^[0-9a-f]{12,40}$ ]] || { printf 'Missing/invalid code revision\n' >&2; exit 2; }
LXY_LOG_ROOT="$LXY_ALLOWED_ROOT/log/${LXY_RUN_DATE}_${LXY_ITERATION}_${LXY_CODE_REVISION:0:12}"
lxy_require_inside "$LXY_LOG_ROOT"
mkdir -p -- "$LXY_LOG_ROOT"
exec 9>"$LXY_LOG_ROOT/RUN.lock"
flock -n 9 || { printf 'This experiment is already running; leaving it untouched\n' >&2; exit 3; }
exec > >(tee -a "$LXY_LOG_ROOT/launcher.log") 2>&1
trap 'LXY_EXIT_CODE=$?; printf "{\"exit_code\":%s,\"finished_at\":\"%s\"}\n" "$LXY_EXIT_CODE" "$(TZ=Asia/Shanghai date --iso-8601=seconds)" > "$LXY_LOG_ROOT/SHELL_STATUS.json"' EXIT

export TZ=Asia/Shanghai
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export XDG_CACHE_HOME="$LXY_LOG_ROOT/cache"
export HF_HOME="$LXY_LOG_ROOT/cache/huggingface"
export TORCH_HOME="$LXY_LOG_ROOT/cache/torch"
export MPLCONFIGDIR="$LXY_LOG_ROOT/cache/matplotlib"
export CUDA_CACHE_PATH="$LXY_LOG_ROOT/cache/cuda"
export TORCH_EXTENSIONS_DIR="$LXY_LOG_ROOT/cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$LXY_LOG_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$LXY_LOG_ROOT/cache/triton"
export TMPDIR="$LXY_LOG_ROOT/tmp" TMP="$LXY_LOG_ROOT/tmp" TEMP="$LXY_LOG_ROOT/tmp"
mkdir -p -- "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" \
    "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"

# 所有资产仅从LXY读取。缺少models链接时只在本轮代码目录创建新链接，不覆盖已有路径。
"$LXY_PYTHON" - "$LXY_REPO_ROOT" "$LXY_INPUTS_ROOT" "$LXY_MODELS_ROOT" <<'PY'
from pathlib import Path
import sys
allowed = Path('/xiliang/LXY').resolve()
def inside(path):
    value = Path(path).resolve()
    if value != allowed and allowed not in value.parents:
        raise RuntimeError('outside LXY: ' + str(value))
    return value
repo, inputs, assets = [inside(p) for p in sys.argv[1:]]
if not (inputs/'INPUT_DOWNLOAD_COMPLETE.json').is_file():
    raise RuntimeError('Original frozen inputs are not complete')
for relative in ['benchmark/readgpt_data', 'plugins/adapters', 'plugins/semflow/cache']:
    if not inside(inputs/relative).is_dir():
        raise RuntimeError('missing frozen input directory: ' + relative)
link = repo/'models'
if not link.exists() and not link.is_symlink():
    if not assets.is_dir():
        raise RuntimeError('authorized model asset directory missing')
    link.symlink_to(assets, target_is_directory=True)
for relative in ['models/aurora', 'models/bert-base-uncased']:
    if not inside(repo/relative).is_dir():
        raise RuntimeError('missing local pretrained assets: ' + relative)
PY

# 只选择当前没有计算进程且至少有8GB空余的GPU；不结束或修改其他进程。
LXY_SELECTED_GPU="$("$LXY_PYTHON" - "${LXY_JOINT_GPU:-}" <<'PY'
import subprocess, sys
query = ['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free', '--format=csv,noheader,nounits']
rows = [tuple(s.strip() for s in row.split(',')) for row in subprocess.check_output(query, text=True).splitlines()]
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader,nounits'], text=True)
occupied = set(apps.splitlines())
eligible = [r for r in rows if r[1] not in occupied and int(r[2]) < 512 and int(r[3]) >= 8192]
requested = sys.argv[1]
if requested:
    eligible = [r for r in eligible if r[0] == requested]
if not eligible:
    raise SystemExit('No idle eligible GPU; other jobs are left untouched')
print(max(eligible, key=lambda r: int(r[3]))[0])
PY
)"
export CUDA_VISIBLE_DEVICES="$LXY_SELECTED_GPU"
cd -- "$LXY_REPO_ROOT"
"$LXY_PYTHON" - "$LXY_LOG_ROOT" "$LXY_CODE_REVISION" "$LXY_ITERATION" "$LXY_SELECTED_GPU" <<'PY'
from pathlib import Path
import datetime, json, os, sys, torch
if sys.prefix != '/xiliang/LXY/envs/lxy' or not torch.cuda.is_available():
    raise RuntimeError('lxy CUDA runtime required')
record = dict(code_commit=sys.argv[2], iteration=sys.argv[3], gpu=int(sys.argv[4]),
    started_at=datetime.datetime.now().astimezone().isoformat(), launcher_pid=os.getppid(),
    python=sys.executable, environment=sys.prefix, torch=torch.__version__,
    domains=9, models=5, horizons_per_domain=4, tasks=180, seeds=[2026,2027,2028],
    joint_fits=540, initial_safety_epochs=2000, epoch_extension=2000,
    cuda_memory_fraction=0.8, comparison='original frozen 180 experiments')
(Path(sys.argv[1])/'RUNTIME.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
PY

# 540次联合拟合全部满足验证平台期后才进行测试；达到预算则保留状态自动续训。
"$LXY_PYTHON" -u plugins/v1_joint_internal_20261008/run_all.py \
    --inputs-root "$LXY_INPUTS_ROOT" \
    --bert "$LXY_REPO_ROOT/models/bert-base-uncased" \
    --output "$LXY_LOG_ROOT" \
    --seeds 2026 2027 2028 \
    --epochs 2000 --epoch-extension 2000 --max-safety-epochs 0 \
    --cuda-memory-fraction 0.8 --device cuda

"$LXY_PYTHON" - "$LXY_LOG_ROOT" <<'PY'
from pathlib import Path
import csv, json, sys
root = Path(sys.argv[1])
record = json.loads((root/'FULL_COMPLETED.json').read_text())
rows = list(csv.DictReader((root/'results_180.csv').open(encoding='utf-8-sig')))
audit = list(csv.DictReader((root/'convergence_audit.csv').open(encoding='utf-8-sig')))
assert record['tasks'] == len(rows) == 180
assert record['fits'] == len(audit) == 540
assert record['all_validation_plateau'] and all(row['converged'] == 'True' for row in audit)
assert len({row['model'] for row in rows}) == 5 and len({row['domain'] for row in rows}) == 9
assert all(sum(row['model'] == m and row['domain'] == d for row in rows) == 4
           for m in {r['model'] for r in rows} for d in {r['domain'] for r in rows})
print('COMPLETE: 180 task results, 540 independently converged fits; ' + str(root))
PY
