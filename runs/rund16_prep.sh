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
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS=1
mkdir -p "$NANOCHAT_BASE_DIR"

echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "git commit: $(git rev-parse HEAD)"
git rev-parse HEAD > "$NANOCHAT_BASE_DIR/git_commit.txt"
git status --short > "$NANOCHAT_BASE_DIR/git_status.txt" || true
date -u +"prep_start_utc=%Y-%m-%dT%H:%M:%SZ" | tee "$NANOCHAT_BASE_DIR/prep_meta.txt"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA built:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("prep allows 0 GPUs; rund16.sh will assert 4")
PY

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
echo "PREP DONE. Boot 4 GPUs and run bash runs/rund16.sh"
