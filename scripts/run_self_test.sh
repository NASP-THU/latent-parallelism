#!/usr/bin/env bash
# Local self-test: Wan2.1-T2V-14B quality + latency vs Ulysses SP on 2 and 4 GPUs.
#
# Runs UNITE and Ulysses (and optionally TP) under the same prompt/size/steps.
# Default uses FRAME_NUM=17 for faster iteration while still producing real videos.
#
# Usage:
#   bash scripts/run_self_test.sh
#   NPROC_LIST=2,4 STRATEGIES=unite,ulysses,tp FRAME_NUM=37 bash scripts/run_self_test.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PYTHONNOUSERSITE=1
if [[ "${ALLOW_NVLINK:-0}" != "1" ]]; then
  export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
fi

CKPT_DIR="${CKPT_DIR:-${ROOT}/Wan2.1-T2V-14B}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/ae_outputs/self_test}"
NPROC_LIST="${NPROC_LIST:-2,4}"
STRATEGIES="${STRATEGIES:-unite,ulysses}"
FRAME_NUM="${FRAME_NUM:-17}"
SIZE="${SIZE:-832*480}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
LP_OVERLAP_RATIO="${LP_OVERLAP_RATIO:-0.4}"
BASE_PORT="${BASE_PORT:-29900}"
PROMPT="${PROMPT:-A timelapse of a puddle drying up and disappearing on a hot day}"
TORCHRUN="${TORCHRUN:-python -m torch.distributed.run}"

if [ ! -f "${CKPT_DIR}/config.json" ]; then
  echo "[self_test] ERROR: checkpoint not found at ${CKPT_DIR}"
  exit 1
fi

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
  local nproc="$2"
  case "${name}" in
    unite) echo --t5_fsdp --latent_parallel --lp_overlap_ratio "${LP_OVERLAP_RATIO}" ;;
    ulysses) echo --t5_fsdp --ulysses_size "${nproc}" --ring_size 1 ;;
    tp) echo --t5_fsdp --tensor_parallel_size "${nproc}" ;;
    *) echo "[self_test] unknown strategy ${name}" >&2; return 1 ;;
  esac
}

run_one() {
  local nproc="$1"
  local name="$2"
  local port="$3"
  local out_dir="${OUT_ROOT}/nproc${nproc}"
  mkdir -p "${out_dir}/${name}"
  local video="${out_dir}/${name}/${name}.mp4"
  local log="${out_dir}/${name}/generation.log"
  local extra
  # shellcheck disable=SC2207
  extra=($(strategy_args "${name}" "${nproc}"))

  echo "============================================================"
  echo "[self_test] nproc=${nproc} strategy=${name} overlap=${LP_OVERLAP_RATIO}"
  echo "[self_test] log=${log}"

  set +e
  ${TORCHRUN} --nproc_per_node="${nproc}" --master_port="${port}" \
    generate.py \
      "${COMMON[@]}" \
      --save_file "${video}" \
      "${extra[@]}" \
    2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  set -e

  if [ ${rc} -eq 0 ] && [ -f "${video}" ]; then
    echo "[self_test] ${name}@${nproc}: OK ($(du -h "${video}" | awk '{print $1}'))"
  else
    echo "[self_test] ${name}@${nproc}: FAILED (exit=${rc})"
  fi
  sleep "${STRATEGY_PAUSE_SEC:-5}"
}

mkdir -p "${OUT_ROOT}"
SUMMARY="${OUT_ROOT}/summary.txt"
: > "${SUMMARY}"

echo "[self_test] CKPT_DIR=${CKPT_DIR}" | tee -a "${SUMMARY}"
echo "[self_test] NPROC_LIST=${NPROC_LIST} STRATEGIES=${STRATEGIES}" | tee -a "${SUMMARY}"
echo "[self_test] size=${SIZE} frames=${FRAME_NUM} steps=${SAMPLE_STEPS} r_max=${LP_OVERLAP_RATIO}" | tee -a "${SUMMARY}"

port=${BASE_PORT}
IFS=',' read -r -a NPROCS <<< "${NPROC_LIST}"
IFS=',' read -r -a STRATS <<< "${STRATEGIES}"

for nproc in "${NPROCS[@]}"; do
  nproc="$(echo "${nproc}" | xargs)"
  [ -z "${nproc}" ] && continue
  for name in "${STRATS[@]}"; do
    name="$(echo "${name}" | xargs)"
    [ -z "${name}" ] && continue
    run_one "${nproc}" "${name}" "${port}"
    port=$((port + 1))
  done
  echo
  echo "[self_test] parsing nproc=${nproc} ..." | tee -a "${SUMMARY}"
  python "${ROOT}/scripts/parse_latency.py" \
    --input_dir "${OUT_ROOT}/nproc${nproc}" \
    --csv "${OUT_ROOT}/nproc${nproc}/latency_table.csv" \
    --json "${OUT_ROOT}/nproc${nproc}/latency_table.json" \
    | tee -a "${SUMMARY}"
done

echo "[self_test] done. Root: ${OUT_ROOT}"
