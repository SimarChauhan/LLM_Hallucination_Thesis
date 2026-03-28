#!/bin/bash
# Submit one Nibi white-box probe job per target model discovered in INPUT_FILE.
# Uses fixed encoder defaults:
#   RESPONSE_HF=Qwen/Qwen3.5-4B
#   VERIFIER_HF=microsoft/Phi-4-mini-instruct

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SINGLE_JOB_SCRIPT="${PROJECT_ROOT}/scripts/slurm_nibi_wb_probe.sh"

if [[ ! -f "${SINGLE_JOB_SCRIPT}" ]]; then
  echo "ERROR: missing script: ${SINGLE_JOB_SCRIPT}" >&2
  exit 1
fi

INPUT_FILE="${INPUT_FILE:-$PROJECT_ROOT/data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl}"
ACCOUNT="${ACCOUNT:-def-<PI_USERNAME>}"
RESPONSE_HF="${RESPONSE_HF:-Qwen/Qwen3.5-4B}"
VERIFIER_HF="${VERIFIER_HF:-microsoft/Phi-4-mini-instruct}"
EXPECTED_TARGET_COUNT="${EXPECTED_TARGET_COUNT:-6}"
DRY_RUN="${DRY_RUN:-0}"
SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SBATCH_GPUS_PER_NODE="${SBATCH_GPUS_PER_NODE:-}"

SUBSET="${SUBSET:-both}"
LAYER_CANDIDATES="${LAYER_CANDIDATES:-last8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
ENCODER_DEVICE_MAP="${ENCODER_DEVICE_MAP:-auto}"
SEQUENTIAL_ENCODERS="${SEQUENTIAL_ENCODERS:-1}"

if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "ERROR: input file not found: ${INPUT_FILE}" >&2
  exit 1
fi

if [[ ! "${DRY_RUN}" =~ ^[01]$ ]]; then
  echo "ERROR: DRY_RUN must be 0 or 1 (got: ${DRY_RUN})" >&2
  exit 1
fi

if [[ "${DRY_RUN}" == "0" && "${ACCOUNT}" == *"<PI_USERNAME>"* ]]; then
  echo "ERROR: ACCOUNT is still placeholder (${ACCOUNT}). Set ACCOUNT=def-YOUR_ACCOUNT." >&2
  exit 1
fi

TARGETS=()
while IFS= read -r target; do
  TARGETS+=("${target}")
done < <(python3 - "${INPUT_FILE}" <<'PY'
import json
import sys

path = sys.argv[1]
models = set()
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        model = str(row.get("model", "")).strip()
        if model:
            models.add(model)
for m in sorted(models):
    print(m)
PY
)

if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  echo "ERROR: no target models discovered from INPUT_FILE." >&2
  exit 1
fi

if [[ -n "${EXPECTED_TARGET_COUNT}" && "${EXPECTED_TARGET_COUNT}" != "0" ]]; then
  if ! [[ "${EXPECTED_TARGET_COUNT}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: EXPECTED_TARGET_COUNT must be numeric (or 0 to disable)." >&2
    exit 1
  fi
  if [[ "${#TARGETS[@]}" -ne "${EXPECTED_TARGET_COUNT}" ]]; then
    echo "ERROR: discovered ${#TARGETS[@]} targets, expected ${EXPECTED_TARGET_COUNT}." >&2
    echo "Set EXPECTED_TARGET_COUNT=0 to disable this guard." >&2
    exit 1
  fi
fi

for t in "${TARGETS[@]}"; do
  if [[ "${t}" == *","* ]]; then
    echo "ERROR: target contains comma, unsupported for sbatch --export parsing: ${t}" >&2
    exit 1
  fi
done

echo "=============================================="
echo "Nibi WB submission (fixed encoders)"
echo "Input:    ${INPUT_FILE}"
echo "Response: ${RESPONSE_HF}"
echo "Verifier: ${VERIFIER_HF}"
echo "Targets:  ${#TARGETS[@]}"
echo "Dry-run:  ${DRY_RUN}"
if [[ -n "${SBATCH_PARTITION}" ]]; then
  echo "Partition: ${SBATCH_PARTITION}"
fi
if [[ -n "${SBATCH_GPUS_PER_NODE}" ]]; then
  echo "GPUs/node: ${SBATCH_GPUS_PER_NODE}"
fi
echo "=============================================="
for t in "${TARGETS[@]}"; do
  echo "  - ${t}"
done
echo

declare -a JOB_TARGETS=()
declare -a JOB_IDS=()

for target in "${TARGETS[@]}"; do
  export_vars="ALL,PROJECT_ROOT=${PROJECT_ROOT},TARGET_NAME=${target},INPUT_FILE=${INPUT_FILE},RESPONSE_HF=${RESPONSE_HF},VERIFIER_HF=${VERIFIER_HF},SUBSET=${SUBSET},LAYER_CANDIDATES=${LAYER_CANDIDATES},BATCH_SIZE=${BATCH_SIZE},TORCH_DTYPE=${TORCH_DTYPE},ENCODER_DEVICE_MAP=${ENCODER_DEVICE_MAP},SEQUENTIAL_ENCODERS=${SEQUENTIAL_ENCODERS}"
  if [[ -n "${OUT_BASE:-}" ]]; then
    export_vars="${export_vars},OUT_BASE=${OUT_BASE}"
  fi

  cmd=(sbatch --parsable -A "${ACCOUNT}")
  if [[ -n "${SBATCH_PARTITION}" ]]; then
    cmd+=(--partition "${SBATCH_PARTITION}")
  fi
  if [[ -n "${SBATCH_GPUS_PER_NODE}" ]]; then
    cmd+=(--gpus-per-node "${SBATCH_GPUS_PER_NODE}")
  fi
  cmd+=(--export="${export_vars}" "${SINGLE_JOB_SCRIPT}")

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf "[DRY_RUN] "
    printf "%q " "${cmd[@]}"
    printf "\n"
    job_id="DRY_RUN"
  else
    job_id="$("${cmd[@]}")"
    echo "[submit] ${target} -> ${job_id}"
  fi

  JOB_TARGETS+=("${target}")
  JOB_IDS+=("${job_id}")
done

echo
printf "%-42s -> %s\n" "TARGET_NAME" "JOB_ID"
for i in "${!JOB_TARGETS[@]}"; do
  printf "%-42s -> %s\n" "${JOB_TARGETS[$i]}" "${JOB_IDS[$i]}"
done
