#!/usr/bin/env bash
# Getting Started (kick-the-tires): run UNITE / Latent Parallelism on a small but real config.
#
# What this script actually runs:
#   - Model: Wan2.1-T2V-14B
#   - Resolution: 832*480 (480p)
#   - Frames: 17 (4n+1)
#   - Steps: 50, UniPC, guide_scale=5.0, shift=5.0, seed=42
#   - Method: --latent_parallel --lp_overlap_ratio 0.4  (paper r_max)
#   - GPUs: NPROC (default: all visible GPUs, typically 4)
#   - Outputs: ae_outputs/getting_started/unite.mp4 + generation.log
#   - Interconnect: NCCL_NVLS_ENABLE=0 (PCIe only; ignore NVLink if present)
#
# Wall-clock with weights already local and multi-GPU (>=2x 48GB): typically ~8-20 min
# (includes multi-minute 14B weight load). Denoising alone is much shorter (~2-3 min
# for 17 frames on 4xA6000). Logs may pause on model load — that is normal.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# Avoid ~/.local site-packages shadowing the active conda/venv (common footgun).
export PYTHONNOUSERSITE=1

# Match paper/AE communication setting: use PCIe even on NVLink machines.
# Override with ALLOW_NVLINK=1 if you intentionally want NVLink.
if [[ "${ALLOW_NVLINK:-0}" != "1" ]]; then
  export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
fi

CKPT_DIR="${CKPT_DIR:-${ROOT}/Wan2.1-T2V-14B}"
OUT_DIR="${OUT_DIR:-${ROOT}/ae_outputs/getting_started}"
NPROC="${NPROC:-}"
MASTER_PORT="${MASTER_PORT:-29601}"
PROMPT="${PROMPT:-A timelapse of a puddle drying up and disappearing on a hot day}"
TORCHRUN="${TORCHRUN:-python -m torch.distributed.run}"
if [ ! -f "${CKPT_DIR}/config.json" ]; then
  echo "[getting_started] ERROR: checkpoint not found at ${CKPT_DIR}"
  echo "  Run: bash scripts/download_model.sh"
  echo "  Or:  export CKPT_DIR=/path/to/Wan2.1-T2V-14B"
  exit 1
fi

if [ -z "${NPROC}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC="$(nvidia-smi -L | wc -l)"
  else
    NPROC=1
  fi
fi
if [ "${NPROC}" -lt 2 ]; then
  echo "[getting_started] ERROR: UNITE requires >=2 GPUs (got NPROC=${NPROC})."
  exit 1
fi

mkdir -p "${OUT_DIR}"
VIDEO="${OUT_DIR}/unite.mp4"
LOG="${OUT_DIR}/generation.log"

echo "[getting_started] CKPT_DIR=${CKPT_DIR}"
echo "[getting_started] NPROC=${NPROC}  size=832*480  frames=17  steps=50  r_max=0.4"
echo "[getting_started] output=${VIDEO}"

${TORCHRUN} --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  generate.py \
    --task t2v-14B \
    --size "832*480" \
    --frame_num 17 \
    --ckpt_dir "${CKPT_DIR}" \
    --offload_model False \
    --t5_fsdp \
    --sample_solver unipc \
    --sample_steps 50 \
    --sample_shift 5.0 \
    --sample_guide_scale 5.0 \
    --base_seed 42 \
    --prompt "${PROMPT}" \
    --latent_parallel \
    --lp_overlap_ratio 0.4 \
    --save_file "${VIDEO}" \
  2>&1 | tee "${LOG}"

echo
if [ -f "${VIDEO}" ]; then
  echo "[getting_started] SUCCESS"
  echo "  Video : ${VIDEO}"
  echo "  Log   : ${LOG}"
  echo "  Look for 'UNITE Latent Parallelism Denoising' / 'Total denoising time' in the log."
  grep -E "Total denoising time|UNITE Latent Parallelism|Finished" "${LOG}" || true
else
  echo "[getting_started] FAILED: video not found. See ${LOG}"
  exit 1
fi
