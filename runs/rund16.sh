#!/bin/bash
# d16 + SFT + GSM8K RL. Default 2 GPUs (override with NPROC=4).
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
export PYTHONUNBUFFERED=1
mkdir -p "$NANOCHAT_BASE_DIR"

NPROC="${NPROC:-2}"
export NPROC
echo "NPROC=$NPROC"

setup_python_env

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
echo "nproc=${NPROC}" >> "$NANOCHAT_BASE_DIR/run_meta.txt"

python - <<'PY'
import os
import torch
n = torch.cuda.device_count()
want = int(os.environ.get("NPROC", "2"))
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU count:", n)
assert n == want, f"Expected {want} GPUs, got {n}"
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
# Use `python -m torch.distributed.run`, not PATH torchrun.
# AutoDL's torchrun is /root/miniconda3/bin/torchrun and launches conda python,
# which cannot see venv packages (wandb, rustbpe, tiktoken, pyarrow).
# Do not pass "--run=..." : torch 2.12 treats it as --run-path.
# Do not insert "--" after -m: argparse in the child rejects it.
# Wandb name comes from WANDB_RUN (see scripts).
LOGDIR="$NANOCHAT_BASE_DIR/torchrun_logs"
mkdir -p "$LOGDIR"
TORCHRUN=(python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" --tee 3 --log-dir "$LOGDIR")
# device-batch-size=32: after compile this d16 run uses ~50GB/96GB at 100% util.
# 64 would drop grad accum to 1 but roughly doubles activations and may OOM.
"${TORCHRUN[@]}" -m scripts.base_train \
    --depth=16 \
    --device-batch-size=32 \
    --window-pattern=L \
    --target-param-data-ratio=12

"${TORCHRUN[@]}" -m scripts.base_eval \
    --device-batch-size=16

"${TORCHRUN[@]}" -m scripts.chat_sft
"${TORCHRUN[@]}" -m scripts.chat_eval -i sft

"${TORCHRUN[@]}" -m scripts.chat_rl
"${TORCHRUN[@]}" -m scripts.chat_eval -i rl -a GSM8K

echo "ALL DONE"
date -u +"end_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$NANOCHAT_BASE_DIR/run_meta.txt"
# python -m scripts.chat_cli -i rl -p "What is 15 * 17?"

# 整条成功才关机，省 AutoDL 计费。set -e 下中途失败不会走到这里。
# 跑完还要留机看日志：NOSHUTDOWN=1 bash runs/rund16.sh
if [ "${NOSHUTDOWN:-0}" != "1" ]; then
    echo "shutting down"
    shutdown -h now
fi
