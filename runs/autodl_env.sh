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
