#!/usr/bin/env bash
# Portable environment setup for the UNITE NSDI artifact.
#
# Preferred install path for AE reviewers (do not rely on `conda env create -f
# environment.yml` alone — that file only bootstraps Python; this script pins
# torch/CUDA and installs flash-attn with hard failure on missing imports).
#
# Usage:
#   bash scripts/setup_env.sh
#   ENV_NAME=unite bash scripts/setup_env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ENV_NAME="${ENV_NAME:-unite}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"

echo "[setup] repo root: ${ROOT}"
echo "[setup] env name : ${ENV_NAME}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[setup] ERROR: conda not found. Install Miniconda/Anaconda first, or create"
  echo "        a Python ${PYTHON_VERSION} venv and run the equivalent of:"
  echo "          python -m pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_INDEX_URL}"
  echo "          python -m pip install -r requirements.txt"
  echo "          PIP_NO_CACHE_DIR=1 python -m pip install flash-attn --no-build-isolation --no-cache-dir"
  echo "          python -m pip install -e . --no-deps"
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] conda env '${ENV_NAME}' already exists; activating."
else
  echo "[setup] creating conda env '${ENV_NAME}' (python=${PYTHON_VERSION})"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -x "${CONDA_PREFIX}/bin/python" ]; then
  echo "[setup] ERROR: conda activate failed (CONDA_PREFIX unset or python missing)."
  exit 1
fi

# Always use the env interpreter. Bare `pip` / `~/.local/bin/pip` often pollutes
# the user site and breaks PYTHONNOUSERSITE isolation.
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONNOUSERSITE=1
PYTHON="${CONDA_PREFIX}/bin/python"
PIP=("${PYTHON}" -m pip)

echo "[setup] python: ${PYTHON} ($("${PYTHON}" -V 2>&1))"
echo "[setup] pip   : $("${PIP[@]}" -V 2>&1)"

echo "[setup] upgrading pip/setuptools/wheel inside the env"
"${PIP[@]}" install --upgrade pip setuptools wheel

echo "[setup] installing PyTorch ${TORCH_VERSION} + torchvision ${TORCHVISION_VERSION}"
echo "        index: ${TORCH_INDEX_URL}"
"${PIP[@]}" install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"

echo "[setup] installing Python deps from requirements.txt"
"${PIP[@]}" install -r requirements.txt

echo "[setup] installing flash-attn (may take several minutes; no pip cache to avoid cross-device link errors)"
if ! PIP_NO_CACHE_DIR=1 "${PIP[@]}" install flash-attn --no-build-isolation --no-cache-dir; then
  echo "[setup] ERROR: flash-attn install failed."
  echo "        Retry options:"
  echo "          1) Install a prebuilt wheel matching torch ${TORCH_VERSION} / your CUDA, e.g."
  echo "               ${PYTHON} -m pip install <flash_attn-...whl> --no-cache-dir"
  echo "          2) Ensure a matching CUDA toolkit/nvcc is available, then rerun this script."
  echo "        Do not continue AE without a working flash_attn import."
  exit 1
fi

echo "[setup] installing package in editable mode (no dependency reshuffle)"
"${PIP[@]}" install -e . --no-deps

# Activation hooks: isolate from ~/.local and keep conda bin first on PATH.
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
cat <<'EOF' > "${CONDA_PREFIX}/etc/conda/activate.d/unite_env.sh"
# Prefer the active conda env over ~/.local (bin + site-packages).
export PYTHONNOUSERSITE=1
if [ -n "${CONDA_PREFIX:-}" ]; then
  case ":${PATH}:" in
    *":${CONDA_PREFIX}/bin:"*) ;;
    *) export PATH="${CONDA_PREFIX}/bin:${PATH}" ;;
  esac
fi
EOF

echo
echo "[setup] verifying critical imports ..."
"${PYTHON}" - <<'PY'
import sys
errors = []
try:
    import torch
    print(f"  torch={torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
except Exception as e:
    errors.append(f"torch: {e}")
try:
    import flash_attn
    print(f"  flash_attn={getattr(flash_attn, '__version__', 'ok')}")
except Exception as e:
    errors.append(f"flash_attn: {e}")
try:
    import xfuser
    print("  xfuser=ok")
except Exception as e:
    errors.append(f"xfuser: {e}")
try:
    import wan
    print("  wan=ok")
except Exception as e:
    errors.append(f"wan: {e}")
if errors:
    print("VERIFY FAILED:", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    sys.exit(1)
print("  all critical imports OK")
PY

echo
echo "[setup] done. Next:"
echo "  conda activate ${ENV_NAME}"
echo "  bash scripts/download_model.sh   # if weights are not local yet"
echo "  bash scripts/getting_started.sh  # smoke test (UNITE)"
echo "  bash scripts/run_nsdi_reproduce.sh  # default AE: 4-GPU UNITE vs TP"
