#!/usr/bin/env bash
set -Eeuo pipefail

# Run this from Windows WSL2 Ubuntu, not from native Git Bash.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-dfn3-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}" #changed from 3.12
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/dataset}"
HF_REPO_ID="${HF_REPO_ID:-Panav-Payappagoudar/sih-26-processed-audio}"
SPLITS_DIR="${SPLITS_DIR:-$ROOT_DIR/data/mvp/splits}"
DFN_ROOT="${DFN_ROOT:-$ROOT_DIR/dfn3-model-files/deepfilternet}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/mvp/dfn3}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/data/mvp/finetune/dfn3-custom}"
MODEL_ARCHIVE="${MODEL_ARCHIVE:-$DFN_ROOT/models/DeepFilterNet3.zip}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PREPARE_WORKERS="${PREPARE_WORKERS:-3}"
TRAIN_WORKERS="${TRAIN_WORKERS:-3}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}" #change to -1 later

die() { echo "ERROR: $*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || die "Use Windows WSL2 Ubuntu. Native Windows Git Bash is not supported by DFN3 training."

if [[ "$EUID" -eq 0 ]]; then
  APT=(apt-get)
else
  command -v sudo >/dev/null || die "sudo is required to install WSL system packages."
  APT=(sudo apt-get)
fi

if [[ "$INSTALL_SYSTEM_DEPS" == "1" ]]; then
  command -v apt-get >/dev/null || die "This launcher requires an Ubuntu/Debian WSL distribution with apt-get."
  "${APT[@]}" update
  if ! command -v "$PYTHON_BIN" >/dev/null; then
    if apt-cache show python3.12 >/dev/null 2>&1; then
      PYTHON_BIN="python3.12"
    elif apt-cache show python3.11 >/dev/null 2>&1; then
      echo "python3.12 is unavailable; falling back to Python 3.11."
      PYTHON_BIN="python3.11"
    elif apt-cache show python3 >/dev/null 2>&1; then
      echo "Versioned Python packages are unavailable; using the distro Python."
      PYTHON_BIN="python3"
    else
      die "No Python package is available. Check that this is Ubuntu/Debian WSL and that apt sources are configured."
    fi
  fi
  if [[ "$PYTHON_BIN" == "python3" ]]; then
    PYTHON_VENV_PACKAGE="python3-venv"
  else
    PYTHON_VENV_PACKAGE="$PYTHON_BIN-venv"
  fi
  "${APT[@]}" install -y \
    "$PYTHON_BIN" "$PYTHON_VENV_PACKAGE" \
    build-essential pkg-config libhdf5-dev \
    cargo rustc git ca-certificates curl
fi

command -v "$PYTHON_BIN" >/dev/null || die "$PYTHON_BIN was not found after apt installation."
command -v git >/dev/null || die "git was not found after system dependency installation."
if [[ -f "$ROOT_DIR/.gitmodules" ]] && command -v git >/dev/null; then
  git submodule update --init --recursive
fi
[[ -f "$DFN_ROOT/DeepFilterNet/df/train.py" ]] || die "DFN3 source not found at $DFN_ROOT."
[[ -f "$MODEL_ARCHIVE" ]] || die "DFN3 model archive not found at $MODEL_ARCHIVE. Set MODEL_ARCHIVE to its location."

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# WSL may expose nvidia-smi outside the normal PATH.
export PATH="/usr/lib/wsl/lib:$PATH"
command -v nvidia-smi >/dev/null || die "NVIDIA WSL driver not available; nvidia-smi was not found. Install/update the Windows NVIDIA driver."

# CUDA wheels are installed from the official PyTorch CUDA index.
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchaudio==2.5.1
python -m pip install -r "$ROOT_DIR/src/drdo_anc/fine_tune/requirements-wsl-cuda.txt"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the NVIDIA CUDA device."
print("CUDA device:", torch.cuda.get_device_name(0))
PY

if [[ ! -f "$DATASET_DIR/metadata.csv" ]]; then
  python src/drdo_anc/fine_tune/download_local.py \
    --repo-id "$HF_REPO_ID" \
    --output-dir "$DATASET_DIR"
fi
[[ -f "$DATASET_DIR/metadata.csv" ]] || die "Dataset download did not produce $DATASET_DIR/metadata.csv."

python src/drdo_anc/fine_tune/scan_filter.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$SPLITS_DIR"

python src/drdo_anc/fine_tune/prepare_dfn3_dataset.py \
  --dataset-dir "$DATASET_DIR" \
  --splits-dir "$SPLITS_DIR" \
  --output-dir "$DATA_DIR" \
  --dfn-root "$DFN_ROOT" \
  --num-workers "$PREPARE_WORKERS"

python src/drdo_anc/fine_tune/prepare_training_run.py \
  --model-archive "$MODEL_ARCHIVE" \
  --run-dir "$RUN_DIR" \
  --max-epochs "$MAX_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --train-workers "$TRAIN_WORKERS" \
  --device cuda

export PYTHONPATH="$DFN_ROOT/DeepFilterNet${PYTHONPATH:+:$PYTHONPATH}"
python "$DFN_ROOT/DeepFilterNet/df/train.py" \
  "$DATA_DIR/dataset.cfg" \
  "$DATA_DIR" \
  "$RUN_DIR"

echo "Fine-tuning completed. Checkpoints and logs: $RUN_DIR"
