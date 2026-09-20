#!/bin/bash
# 无卡前置：环境、ClimbMix 分片、tokenizer。不调用 torchrun，不检查 GPU。
#
# AutoDL 无卡开机后，在仓库根目录：
#   export NANOCHAT_BASE_DIR=/root/autodl-tmp/nanochat-d16
#   bash runs/rund16_prep.sh
# 换 4 卡后 NANOCHAT_BASE_DIR 必须和 rund16.sh 相同。

set -euo pipefail

if [ -z "${NANOCHAT_BASE_DIR:-}" ]; then
    if [ -d /root/autodl-tmp ]; then
        NANOCHAT_BASE_DIR=/root/autodl-tmp/nanochat-d16
    else
        NANOCHAT_BASE_DIR="$HOME/.cache/nanochat-d16"
    fi
fi
export NANOCHAT_BASE_DIR
# defaults: hf-mirror + tuna; sourced file may set HF_ENDPOINT again
source "$(dirname "$0")/autodl_env.sh"
trap restore_uv_lock EXIT
export OMP_NUM_THREADS=1
mkdir -p "$NANOCHAT_BASE_DIR"

echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "UV_DEFAULT_INDEX=$UV_DEFAULT_INDEX"
echo "PYTORCH_WHEEL_MIRROR=$PYTORCH_WHEEL_MIRROR"
echo "git commit: $(git rev-parse HEAD)"
git rev-parse HEAD > "$NANOCHAT_BASE_DIR/git_commit.txt"
git status --short > "$NANOCHAT_BASE_DIR/git_status.txt" || true
date -u +"prep_start_utc=%Y-%m-%dT%H:%M:%SZ" | tee "$NANOCHAT_BASE_DIR/prep_meta.txt"

command -v uv &> /dev/null || true
setup_python_env

python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

python -m scripts.tok_train
python -m scripts.tok_eval

echo "Waiting for remaining shards..."
wait "$DATASET_DOWNLOAD_PID"

python - <<'PY'
import os
from pathlib import Path
base = Path(os.environ["NANOCHAT_BASE_DIR"])
shards = sorted((base / "base_data_climbmix").glob("shard_*.parquet"))
tok = base / "tokenizer" / "tokenizer.pkl"
print(f"shards: {len(shards)}")
print(f"tokenizer: {tok} exists={tok.is_file()} size={tok.stat().st_size if tok.is_file() else 0}")
assert tok.is_file(), "tokenizer.pkl missing"
assert len(shards) >= 9, f"expected >=9 parquet files (8 train + val), got {len(shards)}"
print("prep OK")
PY

date -u +"prep_end_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$NANOCHAT_BASE_DIR/prep_meta.txt"
echo "PREP DONE. Then: export NANOCHAT_BASE_DIR=... && bash runs/rund16.sh"
