#!/bin/bash
#SBATCH --job-name=wb-replay-match
#SBATCH --account=def-<PI_USERNAME>      # Replace (or pass via: sbatch -A ...)
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gpus-per-node=4
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Trillium template: "replay-matched" probe-only run (closest to true WB without regeneration).
#
# What this does:
# - Reuses an existing evaluated JSONL (API answers + labels).
# - Selects rows for TARGET_MODEL_NAME from the JSONL.
# - Extracts hidden states with a LOCAL response encoder you provide (RESPONSE_HF),
#   ideally the same checkpoint family/version as the target's open-weight model.
# - Optionally uses a separate verifier encoder (VERIFIER_HF) for cross-model fusion.
#
# Why this is "closer" to true WB than the default probe wrapper:
# - It removes the small-proxy default and forces an explicitly matched response encoder.
# - It can use the same local model for response+verifier (VERIFIER_HF defaults to RESPONSE_HF),
#   though cross-model fusion is less informative in that case.
#
# This is still NOT strict true white-box E2E because answers/labels are reused from the input JSONL.
#
# Required overrides (at minimum):
#   TARGET_MODEL_NAME   exact `model` string from input JSONL
#   RESPONSE_HF         local path or HF ID for matched response encoder
#
# Example (Qwen target, matched local encoder):
#   sbatch -A def-<acct> \
#     --export=ALL,TARGET_MODEL_NAME='Qwen3 Next 80B (OpenRouter)',RESPONSE_HF='Qwen/<your-exact-qwen-checkpoint>',VERIFIER_HF='Qwen/<your-second-verifier-or-same>' \
#     scripts/slurm_trillium_wb_probe_replay_matched.sh
#
# Example (same response+verifier for maximum closeness, no real cross-model signal):
#   sbatch -A def-<acct> \
#     --export=ALL,TARGET_MODEL_NAME='DeepSeek V3.2 (DeepSeek)',RESPONSE_HF='deepseek-ai/<exact-checkpoint>' \
#     scripts/slurm_trillium_wb_probe_replay_matched.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Optional Alliance module setup (uncomment if needed).
# module purge
# module load StdEnv/2023 python/3.11 cuda

if [[ -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
  source "$PROJECT_ROOT/venv/bin/activate"
elif [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate wb_probe || conda activate base || true
fi

if [[ -z "${HF_TOKEN:-}" && -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi

INPUT_FILE="${INPUT_FILE:-$PROJECT_ROOT/data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl}"

# Exact model string from input JSONL.
TARGET_MODEL_NAME="${TARGET_MODEL_NAME:-}"

# REQUIRED: set this to the local/HF model that best matches the target's open model.
RESPONSE_HF="${RESPONSE_HF:-}"

# Optional: if omitted, defaults to response encoder for maximum closeness (but fusion becomes redundant).
VERIFIER_HF="${VERIFIER_HF:-}"
if [[ -z "$VERIFIER_HF" ]]; then
  VERIFIER_HF="$RESPONSE_HF"
  VERIFIER_MODE_NOTE="same_as_response"
else
  VERIFIER_MODE_NOTE="custom"
fi

SUBSET="${SUBSET:-both}"                     # ce | ie | both
LAYER_CANDIDATES="${LAYER_CANDIDATES:-last8}"
BATCH_SIZE="${BATCH_SIZE:-1}"                # very large models (80B/17B-MoE verifiers) often need 1
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
SEQUENTIAL_ENCODERS="${SEQUENTIAL_ENCODERS:-1}"
ENCODER_DEVICE_MAP="${ENCODER_DEVICE_MAP:-auto}"

if [[ -z "$TARGET_MODEL_NAME" ]]; then
  echo "ERROR: TARGET_MODEL_NAME is required (exact `model` string from input JSONL)." >&2
  exit 2
fi
if [[ -z "$RESPONSE_HF" ]]; then
  echo "ERROR: RESPONSE_HF is required in replay-matched mode." >&2
  exit 2
fi

# Quick check that target exists in the input file.
python - "$INPUT_FILE" "$TARGET_MODEL_NAME" <<'PY'
import json, sys
path, target = sys.argv[1], sys.argv[2]
found = False
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        if str(r.get("model", "")) == target:
            found = True
            break
if not found:
    print(f"Target model not found in input JSONL: {target}", file=sys.stderr)
    sys.exit(3)
PY

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

if [[ "$ENCODER_DEVICE_MAP" == "auto" ]]; then
python - <<'PY'
import sys
try:
    import accelerate  # noqa: F401
except Exception:
    print("Missing Python package: accelerate (required for --encoder-device-map auto)", file=sys.stderr)
    sys.exit(1)
PY
fi

RUN_TAG_BASE="${RUN_TAG:-$(echo "$TARGET_MODEL_NAME" | tr ' /()' '_' | tr -s '_')}"
RUN_TAG="${RUN_TAG_BASE}_replay_matched_${VERIFIER_MODE_NOTE}"

if [[ -n "${SCRATCH:-}" ]]; then
  OUT_BASE="${OUT_BASE:-$SCRATCH/$(basename "$PROJECT_ROOT")/whitebox/wb_cross_model_probe_emnlp2025_replay_matched}"
else
  OUT_BASE="${OUT_BASE:-$PROJECT_ROOT/data/results/analysis/final_analysis_ready/whitebox/wb_cross_model_probe_emnlp2025_replay_matched}"
fi
OUT_DIR="$OUT_BASE/$RUN_TAG"
mkdir -p "$OUT_DIR"

echo "=== Trillium White-Box Probe (replay-matched) ==="
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Node:     $(hostname)"
echo "Input:    $INPUT_FILE"
echo "Target:   $TARGET_MODEL_NAME"
echo "Response: $RESPONSE_HF"
echo "Verifier: $VERIFIER_HF (${VERIFIER_MODE_NOTE})"
echo "Out:      $OUT_DIR"
echo "Note: replay-matched proxy mode (reuses stored answers/labels; no regeneration)"
echo

CMD=(
  python scripts/run_wb_cross_model_probe_emnlp2025.py
  --input "$INPUT_FILE"
  --output-dir "$OUT_DIR"
  --target-model-name "$TARGET_MODEL_NAME"
  --subset "$SUBSET"
  --response-model-path-or-hf-id "$RESPONSE_HF"
  --verifier-model-path-or-hf-id "$VERIFIER_HF"
  --layer-candidates "$LAYER_CANDIDATES"
  --torch-dtype "$TORCH_DTYPE"
  --encoder-device-map "$ENCODER_DEVICE_MAP"
  --encoder-device cuda
  --probe-device cuda
  --batch-size "$BATCH_SIZE"
)

if [[ "$SEQUENTIAL_ENCODERS" == "1" ]]; then
  CMD+=(--sequential-encoders)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo
echo "Done. Outputs:"
echo "  $OUT_DIR"
