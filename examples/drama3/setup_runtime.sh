#!/usr/bin/env bash
set -euo pipefail
# Use an empty Python environment. Never install the full Omni optional extras here.
venv=${1:?usage: setup_runtime.sh /absolute/path/to/new-venv}
omni=${2:?usage: setup_runtime.sh /absolute/path/to/new-venv /absolute/path/to/sglang-omni}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-python3.12}
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
test ! -e "$venv"
test "$(git -C "$omni" rev-parse HEAD)" = 2cfb3b67becb437b70751defe37e63d3af4eedb0
"$python_bin" -m venv "$venv"
python="$venv/bin/python"
"$python" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 'torch==2.13.0+cu130'
# This exact pair is used by the pinned reference. Do not let another resolver replace Torch.
"$python" -m pip install --no-cache-dir --no-deps --index-url https://download.pytorch.org/whl/cu130 \
    'torchaudio==2.11.0+cu130' 'torchvision==0.28.0+cu130' 'torchcodec==0.16.0+cu130'
"$python" -m pip install --no-cache-dir --no-deps -r "$script_dir/requirements-runtime.lock"
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$omni" \
    "$python" "$script_dir/check_runtime.py"
"$venv/bin/hf" --help >/dev/null
"$python" -m pip freeze > "$venv/drama3-installed.txt"
printf 'Runtime ready. Use %s with PYTHONPATH=%s\n' "$python" "$omni"
