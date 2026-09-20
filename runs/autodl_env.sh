# Shared AutoDL mirror env. Sourced by rund16_prep.sh and rund16.sh.
# Torch wheels are pinned in uv.lock to download.pytorch.org; PyPI index env vars
# do not rewrite those URLs. We rewrite the lockfile copies before uv sync.

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PYTORCH_WHEEL_MIRROR="${PYTORCH_WHEEL_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels}"
export PYPI_FILE_MIRROR="${PYPI_FILE_MIRROR:-https://pypi.tuna.tsinghua.edu.cn}"

rewrite_uv_lock_to_mirrors() {
    python3 - <<'PY'
from pathlib import Path
p = Path("uv.lock")
text = p.read_text()
import os
torch_mirror = os.environ["PYTORCH_WHEEL_MIRROR"].rstrip("/") + "/"
pypi_mirror = os.environ["PYPI_FILE_MIRROR"].rstrip("/") + "/"
text = text.replace("https://download-r2.pytorch.org/whl/", torch_mirror)
text = text.replace("https://download.pytorch.org/whl/", torch_mirror)
text = text.replace("https://files.pythonhosted.org/", pypi_mirror)
p.write_text(text)
print(f"uv.lock wheels -> {torch_mirror} and {pypi_mirror}")
PY
}

restore_uv_lock() {
    git checkout -- uv.lock 2>/dev/null || true
}

NANOCHAT_LIGHT_DEPS=(
    "filelock>=3.19.0"
    "kernels>=0.11.7"
    "numpy>=1.26.0"
    "psutil>=7.1.0"
    "pyarrow>=21.0.0"
    "rustbpe>=0.1.0"
    "tiktoken>=0.11.0"
    "wandb>=0.21.3"
)

find_cuda_python() {
    local cand
    for cand in python /root/miniconda3/bin/python /root/miniconda3/bin/python3 python3; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import torch; assert torch.version.cuda is not None" >/dev/null 2>&1; then
            command -v "$cand"
            return 0
        fi
    done
    return 1
}

system_torch_ok() {
    find_cuda_python >/dev/null
}

use_system_torch() {
    case "${NANOCHAT_USE_SYSTEM_TORCH:-auto}" in
        1|true|yes) return 0 ;;
        0|false|no) return 1 ;;
        *) system_torch_ok ;;
    esac
}

setup_python_env() {
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    if use_system_torch; then
        local py
        py="$(find_cuda_python)" || {
            echo "No CUDA-built torch on PATH (need torch.version.cuda, GPUs not required for prep)."
            echo "Check: /root/miniconda3/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda)'"
            exit 1
        }
        echo "Using system/conda CUDA torch via $py (skip uv GPU torch wheels)"
        "$py" - <<'PY'
import torch
print("system torch:", torch.__version__, "cuda_build:", torch.version.cuda, "available:", torch.cuda.is_available(), "n:", torch.cuda.device_count())
PY
        if [ ! -f .venv/.system-torch ]; then
            rm -rf .venv
            uv venv --python "$py" --system-site-packages
            touch .venv/.system-torch
        fi
        # shellcheck disable=SC1091
        source .venv/bin/activate
        uv pip install "${NANOCHAT_LIGHT_DEPS[@]}"
        python - <<'PY'
import torch
print("venv torch:", torch.__version__, "cuda_build:", torch.version.cuda, "available:", torch.cuda.is_available(), "n:", torch.cuda.device_count())
assert torch.version.cuda is not None, "venv is not seeing a CUDA-built torch"
print("GPU count:", torch.cuda.device_count(), "(0 is ok on a no-GPU AutoDL box)")
PY
    else
        echo "Installing lockfile torch via uv sync --extra gpu"
        [ -d ".venv" ] || uv venv
        rewrite_uv_lock_to_mirrors
        uv sync --extra gpu
        restore_uv_lock
        # shellcheck disable=SC1091
        source .venv/bin/activate
        python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA built:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
PY
    fi
}

