#!/usr/bin/env bash
# Download Wan2.1-T2V-14B weights into ./Wan2.1-T2V-14B (or $CKPT_DIR).
# Prefers ModelScope; falls back to Hugging Face instructions.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${CKPT_DIR:-${ROOT}/Wan2.1-T2V-14B}"

echo "[download] target: ${CKPT_DIR}"

if [ -d "${CKPT_DIR}" ] && [ -f "${CKPT_DIR}/config.json" ]; then
  echo "[download] checkpoint already present at ${CKPT_DIR}"
  exit 0
fi

mkdir -p "$(dirname "${CKPT_DIR}")"

if command -v modelscope >/dev/null 2>&1; then
  echo "[download] using ModelScope CLI"
  modelscope download --model Wan-AI/Wan2.1-T2V-14B --local_dir "${CKPT_DIR}"
elif python -c "import modelscope" >/dev/null 2>&1; then
  echo "[download] using modelscope Python API"
  python - <<PY
from modelscope import snapshot_download
snapshot_download('Wan-AI/Wan2.1-T2V-14B', local_dir='${CKPT_DIR}')
PY
else
  cat <<EOF
[download] ModelScope is not installed.

Option A (ModelScope):
  python -m pip install modelscope
  modelscope download --model Wan-AI/Wan2.1-T2V-14B --local_dir ${CKPT_DIR}

Option B (Hugging Face):
  python -m pip install "huggingface_hub[cli]"
  huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ${CKPT_DIR}

Then set:
  export CKPT_DIR=${CKPT_DIR}
EOF
  exit 1
fi

echo "[download] done: ${CKPT_DIR}"
echo "export CKPT_DIR=${CKPT_DIR}"
