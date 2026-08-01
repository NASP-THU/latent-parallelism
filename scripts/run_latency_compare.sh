#!/usr/bin/env bash
# Latency comparison across parallel strategies for Wan2.1-T2V-14B.
#
# Default (NSDI R1 path): UNITE vs Tensor Parallel only.
# Full baselines (optional):
#   STRATEGIES=unite,ulysses,ring,fsdp,tp bash scripts/run_latency_compare.sh
#
# What this script runs:
#   - Model: Wan2.1-T2V-14B
#   - Resolution: 832*480
#   - Frames: 37
#   - Steps: 50, UniPC, guide_scale=5.0, shift=5.0, seed=42
#   - nproc: NPROC (default: all visible GPUs)
#
# Outputs under ae_outputs/latency_compare/:
#   <strategy>/{video.mp4,generation.log}
#   latency_table.csv / latency_table.json / summary.txt
#
# Absolute times vary by GPU / interconnect; expect UNITE lower communication-bound
# latency than TP on PCIe multi-GPU.
# NCCL_NVLS_ENABLE=0 forces PCIe even when NVLink is available (paper/AE default).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PYTHONNOUSERSITE=1

if [[ "${ALLOW_NVLINK:-0}" != "1" ]]; then
  export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
fi

CKPT_DIR="${CKPT_DIR:-${ROOT}/Wan2.1-T2V-14B}"
OUT_DIR="${OUT_DIR:-${ROOT}/ae_outputs/latency_compare}"
NPROC="${NPROC:-}"
BASE_PORT="${BASE_PORT:-29700}"
FRAME_NUM="${FRAME_NUM:-37}"
SIZE="${SIZE:-832*480}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
LP_OVERLAP_RATIO="${LP_OVERLAP_RATIO:-0.4}"
PROMPT="${PROMPT:-A timelapse of a puddle drying up and disappearing on a hot day}"
# Default AE path: only UNITE vs TP. Override for full suite.
STRATEGIES="${STRATEGIES:-unite,tp}"
TORCHRUN="${TORCHRUN:-python -m torch.distributed.run}"

if [ ! -f "${CKPT_DIR}/config.json" ]; then
  echo "[latency] ERROR: checkpoint not found at ${CKPT_DIR}"
  exit 1
fi

if [ -z "${NPROC}" ]; then
  NPROC="$(nvidia-smi -L | wc -l)"
fi
if [ "${NPROC}" -lt 2 ]; then
  echo "[latency] ERROR: need >=2 GPUs (got ${NPROC})."
  exit 1
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
      echo "[latency] ERROR: unknown strategy '${name}'" >&2
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
  echo "[latency] strategy=${name}  nproc=${NPROC}  extra=${extra[*]}" | tee -a "${SUMMARY}"
  echo "[latency] log=${log}" | tee -a "${SUMMARY}"

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
    echo "[latency] ${name}: OK" | tee -a "${SUMMARY}"
  else
    echo "[latency] ${name}: FAILED (exit=${rc})" | tee -a "${SUMMARY}"
    echo "[latency] continuing to next strategy; see ${log}" | tee -a "${SUMMARY}"
  fi

  sleep "${STRATEGY_PAUSE_SEC:-5}"
}

IFS=',' read -r -a STRATEGY_LIST <<< "${STRATEGIES}"

echo "[latency] CKPT_DIR=${CKPT_DIR}  NPROC=${NPROC}  size=${SIZE}  frames=${FRAME_NUM}"
echo "[latency] strategies=${STRATEGIES}  lp_overlap_ratio=${LP_OVERLAP_RATIO}"
echo "[latency] Default is UNITE vs TP. For full suite: STRATEGIES=unite,ulysses,ring,fsdp,tp"
echo "[latency] Primary metric: denoising_s. Weight load may take several minutes per strategy."

port_offset=0
for name in "${STRATEGY_LIST[@]}"; do
  name="$(echo "${name}" | xargs)"
  [ -z "${name}" ] && continue
  run_one "${name}" $((BASE_PORT + port_offset))
  port_offset=$((port_offset + 1))
done

echo
echo "[latency] parsing results ..."
python "${ROOT}/scripts/parse_latency.py" \
  --input_dir "${OUT_DIR}" \
  --csv "${OUT_DIR}/latency_table.csv" \
  --json "${OUT_DIR}/latency_table.json" \
  | tee -a "${SUMMARY}"

echo "[latency] done. Table: ${OUT_DIR}/latency_table.csv"
