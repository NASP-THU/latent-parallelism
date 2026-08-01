#!/usr/bin/env bash
# NSDI AE reproduction (default): Wan2.1-T2V-14B on 4 GPUs, UNITE vs Tensor Parallel only.
#
# This is the recommended path for artifact reviewers. Absolute latency numbers vary
# by GPU / interconnect; success is relative ordering (UNITE substantially faster than TP)
# with plausible generated videos — not bit-exact match to paper tables.
#
# What this script runs by default:
#   - Model: Wan2.1-T2V-14B
#   - GPUs: 4 (override with NPROC=...)
#   - Resolution: 832*480
#   - Frames: 37
#   - Steps: 50, UniPC, guide_scale=5.0, shift=5.0, seed=42, r_max=0.4
#   - Strategies: unite, tp
#
# Optional full baseline set (slower):
#   STRATEGIES=unite,ulysses,ring,fsdp,tp bash scripts/run_nsdi_reproduce.sh
#
# Outputs: ae_outputs/nsdi_reproduce/<strategy>/{*.mp4,generation.log}
#          ae_outputs/nsdi_reproduce/latency_table.{csv,json} + summary.txt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PYTHONNOUSERSITE=1

if [[ "${ALLOW_NVLINK:-0}" != "1" ]]; then
  export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
fi

CKPT_DIR="${CKPT_DIR:-${ROOT}/Wan2.1-T2V-14B}"
OUT_DIR="${OUT_DIR:-${ROOT}/ae_outputs/nsdi_reproduce}"
NPROC="${NPROC:-4}"
BASE_PORT="${BASE_PORT:-29800}"
FRAME_NUM="${FRAME_NUM:-37}"
SIZE="${SIZE:-832*480}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
LP_OVERLAP_RATIO="${LP_OVERLAP_RATIO:-0.4}"
PROMPT="${PROMPT:-A timelapse of a puddle drying up and disappearing on a hot day}"
# Default AE claim path: UNITE vs Tensor Parallelism only.
STRATEGIES="${STRATEGIES:-unite,tp}"
TORCHRUN="${TORCHRUN:-python -m torch.distributed.run}"

if [ ! -f "${CKPT_DIR}/config.json" ]; then
  echo "[nsdi] ERROR: checkpoint not found at ${CKPT_DIR}"
  echo "  Run: bash scripts/download_model.sh"
  echo "  Or:  export CKPT_DIR=/path/to/Wan2.1-T2V-14B"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[nsdi] ERROR: nvidia-smi not found."
  exit 1
fi

VISIBLE="$(nvidia-smi -L | wc -l)"
if [ "${NPROC}" -lt 2 ]; then
  echo "[nsdi] ERROR: need >=2 GPUs (got NPROC=${NPROC})."
  exit 1
fi
if [ "${NPROC}" -gt "${VISIBLE}" ]; then
  echo "[nsdi] ERROR: NPROC=${NPROC} but only ${VISIBLE} GPUs are visible."
  exit 1
fi
if [ "${NPROC}" -ne 4 ]; then
  echo "[nsdi] NOTE: default AE path uses 4 GPUs; running with NPROC=${NPROC}."
fi

mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.txt"
: > "${SUMMARY}"

COMMON=(
  --task t2v-14B
  --size "${SIZE}"
  --frame_num "${FRAME_NUM}"
  --ckpt_dir "${CKPT_DIR}"
  --offload_model False
  --sample_solver unipc
  --sample_steps "${SAMPLE_STEPS}"
  --sample_shift 5.0
  --sample_guide_scale 5.0
  --base_seed 42
  --prompt "${PROMPT}"
)

strategy_args() {
  local name="$1"
  case "${name}" in
    unite)
      echo --t5_fsdp --latent_parallel --lp_overlap_ratio "${LP_OVERLAP_RATIO}"
      ;;
    tp)
      echo --t5_fsdp --tensor_parallel_size "${NPROC}"
      ;;
    ulysses)
      echo --t5_fsdp --ulysses_size "${NPROC}" --ring_size 1
      ;;
    ring)
      echo --t5_fsdp --ulysses_size 1 --ring_size "${NPROC}"
      ;;
    fsdp)
      echo --t5_fsdp --dit_fsdp
      ;;
    *)
      echo "[nsdi] ERROR: unknown strategy '${name}' (allowed: unite,tp,ulysses,ring,fsdp)" >&2
      return 1
      ;;
  esac
}

run_one() {
  local name="$1"
  local port="$2"
  local extra
  # shellcheck disable=SC2207
  extra=($(strategy_args "${name}"))
  local sdir="${OUT_DIR}/${name}"
  mkdir -p "${sdir}"
  local video="${sdir}/${name}.mp4"
  local log="${sdir}/generation.log"

  echo "============================================================" | tee -a "${SUMMARY}"
  echo "[nsdi] strategy=${name}  nproc=${NPROC}  extra=${extra[*]}" | tee -a "${SUMMARY}"
  echo "[nsdi] log=${log}" | tee -a "${SUMMARY}"

  set +e
  ${TORCHRUN} --nproc_per_node="${NPROC}" --master_port="${port}" \
    generate.py \
      "${COMMON[@]}" \
      --save_file "${video}" \
      "${extra[@]}" \
    2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  set -e

  if [ ${rc} -eq 0 ] && [ -f "${video}" ]; then
    echo "[nsdi] ${name}: OK" | tee -a "${SUMMARY}"
  else
    echo "[nsdi] ${name}: FAILED (exit=${rc})" | tee -a "${SUMMARY}"
    echo "[nsdi] continuing to next strategy; see ${log}" | tee -a "${SUMMARY}"
  fi

  sleep "${STRATEGY_PAUSE_SEC:-5}"
}

IFS=',' read -r -a STRATEGY_LIST <<< "${STRATEGIES}"

echo "[nsdi] CKPT_DIR=${CKPT_DIR}"
echo "[nsdi] NPROC=${NPROC}  size=${SIZE}  frames=${FRAME_NUM}  steps=${SAMPLE_STEPS}"
echo "[nsdi] strategies=${STRATEGIES}  lp_overlap_ratio=${LP_OVERLAP_RATIO}"
echo "[nsdi] Default path is UNITE vs TP only (~20-40 min wall-clock with 2x weight load)."
echo "[nsdi] Primary metric: denoising_s (not end-to-end wall-clock)."
echo "[nsdi] Logs may sit on weight load for several minutes — that is normal."

port_offset=0
for name in "${STRATEGY_LIST[@]}"; do
  name="$(echo "${name}" | xargs)"
  [ -z "${name}" ] && continue
  run_one "${name}" $((BASE_PORT + port_offset))
  port_offset=$((port_offset + 1))
done

echo
echo "[nsdi] parsing results ..."
python "${ROOT}/scripts/parse_latency.py" \
  --input_dir "${OUT_DIR}" \
  --csv "${OUT_DIR}/latency_table.csv" \
  --json "${OUT_DIR}/latency_table.json" \
  | tee -a "${SUMMARY}"

echo "[nsdi] done. Table: ${OUT_DIR}/latency_table.csv"
echo "[nsdi] Expect UNITE denoising_s << TP denoising_s on PCIe multi-GPU."
