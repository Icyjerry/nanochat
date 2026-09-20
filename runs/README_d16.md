# d16 AutoDL recipe (4x RTX PRO 6000)

Python is unchanged from upstream nanochat. These scripts only change launch flags.

Cache directory: `NANOCHAT_BASE_DIR` (AutoDL data disk `/root/autodl-tmp/nanochat-d16` if that path exists, else `~/.cache/nanochat-d16`). Use the **same** value for prep and training.

## No-GPU machine

```bash
git clone -b d16-autodl https://github.com/Icyjerry/nanochat.git
cd nanochat
export NANOCHAT_BASE_DIR=/root/autodl-tmp/nanochat-d16
bash runs/rund16_prep.sh
```

Installs the CUDA PyTorch extra (does not need a GPU to install), downloads ClimbMix shards, trains the 32k tokenizer.

`dataset.py` talks to `huggingface.co` directly. `HF_ENDPOINT` may not rewrite that URL.

## 4-GPU machine

Optional short smoke tests (separate tags, does not write into `d16`):

```bash
source .venv/bin/activate
export NANOCHAT_BASE_DIR=/root/autodl-tmp/nanochat-d16
export OMP_NUM_THREADS=1

torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=16 --device-batch-size=32 --window-pattern=L \
    --target-param-data-ratio=12 \
    --num-iterations=20 --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
    --model-tag=d16_smoke_bf16 --run=dummy
```

Add `--fp8` and `--model-tag=d16_smoke_fp8` only as a second smoke. If it fails, stop; do not auto-retry as BF16 inside `rund16.sh`.

Formal run (BF16, no `--fp8`):

```bash
export NANOCHAT_BASE_DIR=/root/autodl-tmp/nanochat-d16
screen -L -Logfile runs/rund16.log -S d16 bash runs/rund16.sh
```

A successful run ends with `shutdown -h now` so AutoDL stops billing. Failures exit earlier (`set -e`) and leave the machine up. To keep the box after success: `NOSHUTDOWN=1 bash runs/rund16.sh`. Do not add shutdown to the no-GPU prep script.

Tokenizer is skipped if `tokenizer.pkl` already exists. Shards that already exist are skipped.

## Outputs

| Stage | Path under `$NANOCHAT_BASE_DIR` |
|-------|----------------------------------|
| tokenizer | `tokenizer/` |
| pretrain | `base_checkpoints/d16/` |
| SFT | `chatsft_checkpoints/d16/` |
| RL | `chatrl_checkpoints/d16/` |
| git SHA | `git_commit.txt` |

Chat after RL: `python -m scripts.chat_cli -i rl`
