#!/bin/bash
# d16 + SFT + GSM8K RL on 4x RTX PRO 6000 96GB.
# First formal run is BF16 (no --fp8). Do not auto-fallback if FP8 is added later.
#
# bash runs/rund16.sh
# screen -L -Logfile runs/rund16.log -S d16 bash runs/rund16.sh
# WANDB_RUN=d16-4x6000 bash runs/rund16.sh

set -euo pipefail

if [ -z "${NANOCHAT_BASE_DIR:-}" ]; then
    if [ -d /root/autodl-tmp ]; then
        NANOCHAT_BASE_DIR=/root/autodl-tmp/nanochat-d16
    else
        NANOCHAT_BASE_DIR="$HOME/.cache/nanochat-d16"
    fi
fi
export NANOCHAT_BASE_DIR
source "$(dirname "$0")/autodl_env.sh"
trap restore_uv_lock EXIT
export OMP_NUM_THREADS=1
mkdir -p "$NANOCHAT_BASE_DIR"

NPROC=4

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
rewrite_uv_lock_to_mirrors
uv sync --extra gpu
restore_uv_lock
source .venv/bin/activate

if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi

GIT_SHA="$(git rev-parse HEAD)"
echo "git commit: ${GIT_SHA}"
echo "${GIT_SHA}" > "$NANOCHAT_BASE_DIR/git_commit.txt"
git status --short > "$NANOCHAT_BASE_DIR/git_status.txt" || true
date -u +"start_utc=%Y-%m-%dT%H:%M:%SZ" | tee "$NANOCHAT_BASE_DIR/run_meta.txt"
echo "git_commit=${GIT_SHA}" >> "$NANOCHAT_BASE_DIR/run_meta.txt"
echo "wandb_run=${WANDB_RUN}" >> "$NANOCHAT_BASE_DIR/run_meta.txt"

python - <<'PY'
import torch
n = torch.cuda.device_count()
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU count:", n)
assert n == 4, f"Expected 4 GPUs, got {n}"
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(
        i,
        torch.cuda.get_device_name(i),
        f"{p.total_memory / 1024**3:.1f} GiB",
        torch.cuda.get_device_capability(i),
    )
PY
nvidia-smi topo -m

python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

TOK="$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
if [ ! -f "$TOK" ]; then
    python -m scripts.tok_train
    python -m scripts.tok_eval
else
    echo "skip tok_train, found $TOK"
fi

echo "Waiting for dataset download to complete..."
wait "$DATASET_DOWNLOAD_PID"

# d16 / BF16. window-pattern=L because Blackwell usually has no FA3.
# target-param-data-ratio=12 is the current master default, pinned so it cannot drift.
# Fewer GPUs keep the same token budget; gradient accumulation fills the global batch.
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    --depth=16 \
    --device-batch-size=32 \
    --window-pattern=L \
    --target-param-data-ratio=12 \
    --run="$WANDB_RUN"

torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_eval -- \
    --device-batch-size=16

torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_sft -- \
    --run="$WANDB_RUN"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_eval -- -i sft

torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_rl -- \
    --run="$WANDB_RUN"
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_eval -- -i rl -a GSM8K

echo "ALL DONE"
date -u +"end_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$NANOCHAT_BASE_DIR/run_meta.txt"
# python -m scripts.chat_cli -i rl -p "What is 15 * 17?"

# 整条成功才关机，省 AutoDL 计费。set -e 下中途失败不会走到这里。
# 跑完还要留机看日志：NOSHUTDOWN=1 bash runs/rund16.sh
if [ "${NOSHUTDOWN:-0}" != "1" ]; then
    echo "shutting down"
    shutdown -h now
fi
