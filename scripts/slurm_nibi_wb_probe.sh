#!/bin/bash
#SBATCH --job-name=wb-probe-nibi
#SBATCH --account=def-<PI_USERNAME>      # Replace (or pass via: sbatch -A ...)
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Nibi template: probe-only white-box run on existing evaluated JSONL.
# Fixed encoder defaults:
#   RESPONSE_HF=Qwen/Qwen3.5-4B
#   VERIFIER_HF=microsoft/Phi-4-mini-instruct
#
# TARGET_NAME is required and must match the JSONL `model` field exactly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

RUNNER_PY="${PROJECT_ROOT}/scripts/run_wb_cross_model_probe_emnlp2025.py"
if [[ ! -f "${RUNNER_PY}" ]]; then
  echo "ERROR: runner not found: ${RUNNER_PY}" >&2
  echo "Set PROJECT_ROOT to your repo root if needed." >&2
  exit 1
fi

if [[ -f "${PROJECT_ROOT}/venv/bin/activate" ]]; then
  source "${PROJECT_ROOT}/venv/bin/activate"
elif [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  source "${PROJECT_ROOT}/.venv/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate wb_probe || conda activate base || true
fi

if [[ -z "${HF_TOKEN:-}" && -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  source "${PROJECT_ROOT}/.env"
  set +a
fi

INPUT_FILE="${INPUT_FILE:-$PROJECT_ROOT/data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl}"

TARGET_NAME="${TARGET_NAME:-}"
if [[ -z "${TARGET_NAME}" ]]; then
  echo "ERROR: TARGET_NAME is required and must exactly match a JSONL 'model' value." >&2
  exit 2
fi

RESPONSE_HF="${RESPONSE_HF:-Qwen/Qwen3.5-4B}"
VERIFIER_HF="${VERIFIER_HF:-microsoft/Phi-4-mini-instruct}"

SUBSET="${SUBSET:-both}"
LAYER_CANDIDATES="${LAYER_CANDIDATES:-last8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
ENCODER_DEVICE_MAP="${ENCODER_DEVICE_MAP:-auto}"
SEQUENTIAL_ENCODERS="${SEQUENTIAL_ENCODERS:-1}"

TARGET_TAG="$(echo "${TARGET_NAME}" | tr ' /()' '_' | tr -s '_')"
VERIFIER_TAG="$(echo "${VERIFIER_HF}" | tr '[:upper:]' '[:lower:]' | tr ' /()' '_' | tr -s '_')"
if [[ -n "${SCRATCH:-}" ]]; then
  OUT_BASE="${OUT_BASE:-$SCRATCH/$(basename "$PROJECT_ROOT")/whitebox/wb_cross_model_probe_emnlp2025_nibi}"
else
  OUT_BASE="${OUT_BASE:-$PROJECT_ROOT/data/results/analysis/final_analysis_ready/whitebox/wb_cross_model_probe_emnlp2025_nibi}"
fi
OUT_DIR="${OUT_BASE}/${TARGET_TAG}__${VERIFIER_TAG}"
mkdir -p "${OUT_DIR}"

echo "=== Nibi White-Box Probe (probe-only) ==="
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Node:     $(hostname)"
echo "Input:    ${INPUT_FILE}"
echo "Target:   ${TARGET_NAME}"
echo "Response: ${RESPONSE_HF}"
echo "Verifier: ${VERIFIER_HF}"
echo "Out:      ${OUT_DIR}"
echo

python - <<'PY'
import sys
mods = ["torch", "transformers", "pandas", "numpy", "dotenv"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception:
        missing.append(m)
if missing:
    print("Missing Python packages:", ", ".join(missing), file=sys.stderr)
    sys.exit(1)
PY

if [[ "${ENCODER_DEVICE_MAP}" == "auto" ]]; then
python - <<'PY'
import sys
try:
    import accelerate  # noqa: F401
except Exception:
    print("Missing Python package: accelerate (required for --encoder-device-map auto).", file=sys.stderr)
    sys.exit(1)
PY
fi

python - "${INPUT_FILE}" "${TARGET_NAME}" <<'PY'
import json
import sys

path, target = sys.argv[1], sys.argv[2]
found = False
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if str(row.get("model", "")) == target:
            found = True
            break
if not found:
    print(
        f"Target model not found in input JSONL: '{target}'. "
        "Use an exact model string from the input file.",
        file=sys.stderr,
    )
    sys.exit(1)
PY

python - "${RESPONSE_HF}" "${VERIFIER_HF}" <<'PY'
import sys
from transformers import AutoConfig

response, verifier = sys.argv[1], sys.argv[2]
errors = []

def check(model_id: str, role: str) -> None:
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(f"[preflight] {role} config ok: {model_id} -> {type(cfg).__name__}")
    except Exception as exc:
        msg = f"{role} config load failed for {model_id}: {type(exc).__name__}: {exc}"
        errors.append(msg)

check(response, "response")
check(verifier, "verifier")

if errors:
    print("[preflight] one or more encoder checks failed:", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    print(
        "Fixes: ensure HF_TOKEN can access gated repos and upgrade transformers "
        "if architecture support is missing.",
        file=sys.stderr,
    )
    sys.exit(1)
PY

CMD=(
  python "${RUNNER_PY}"
  --input "${INPUT_FILE}"
  --output-dir "${OUT_DIR}"
  --target-model-name "${TARGET_NAME}"
  --subset "${SUBSET}"
  --response-model-path-or-hf-id "${RESPONSE_HF}"
  --verifier-model-path-or-hf-id "${VERIFIER_HF}"
  --layer-candidates "${LAYER_CANDIDATES}"
  --torch-dtype "${TORCH_DTYPE}"
  --encoder-device-map "${ENCODER_DEVICE_MAP}"
  --encoder-device cuda
  --probe-device cuda
  --batch-size "${BATCH_SIZE}"
)

if [[ "${SEQUENTIAL_ENCODERS}" == "1" ]]; then
  CMD+=(--sequential-encoders)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo
echo "Done. Outputs:"
echo "  ${OUT_DIR}"
