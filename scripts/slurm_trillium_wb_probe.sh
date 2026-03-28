#!/bin/bash
#SBATCH --job-name=wb-probe-tri
#SBATCH --account=def-<PI_USERNAME>      # Replace (or pass via: sbatch -A ...)
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Trillium template: probe-only white-box run on existing evaluated JSONL.
# This uses local proxy encoders to extract hidden states over question+greedy_answer
# from an existing analysis-ready file (e.g., your black-box pipeline outputs).
#
# Usage:
#   sbatch scripts/slurm_trillium_wb_probe.sh
#
# Override defaults:
#   sbatch --export=ALL,TARGET_MODEL_NAME='Qwen3 Next 80B (OpenRouter)',RESPONSE_HF='Qwen/Qwen2.5-1.5B-Instruct',VERIFIER_HF='Qwen/Qwen2.5-0.5B-Instruct' scripts/slurm_trillium_wb_probe.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs/slurm

# Optional Alliance module setup (uncomment if you use modules on Trillium).
# module purge
# module load StdEnv/2023 python/3.11 cuda

# Activate an environment if present.
if [[ -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
  source "$PROJECT_ROOT/venv/bin/activate"
elif [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate wb_probe || conda activate base || true
fi

# Load secrets from .env if available (HF_TOKEN for gated models like Llama).
if [[ -z "${HF_TOKEN:-}" && -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi

INPUT_FILE="${INPUT_FILE:-$PROJECT_ROOT/data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl}"

# This target must match the exact `model` string in the input JSONL.
TARGET_MODEL_NAME="${TARGET_MODEL_NAME:-Qwen3 Next 80B (OpenRouter)}"

# Local/HF proxy encoders used for hidden-state extraction (white-box part).
RESPONSE_HF="${RESPONSE_HF:-Qwen/Qwen2.5-1.5B-Instruct}"
VERIFIER_HF="${VERIFIER_HF:-Qwen/Qwen2.5-0.5B-Instruct}"

SUBSET="${SUBSET:-both}"                    # ce | ie | both
LAYER_CANDIDATES="${LAYER_CANDIDATES:-last8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
SEQUENTIAL_ENCODERS="${SEQUENTIAL_ENCODERS:-1}"  # 1 reduces peak memory

RUN_TAG="${RUN_TAG:-$(echo "$TARGET_MODEL_NAME" | tr ' /()' '_' | tr -s '_')}"

if [[ -n "${SCRATCH:-}" ]]; then
  OUT_BASE="${OUT_BASE:-$SCRATCH/$(basename "$PROJECT_ROOT")/whitebox/wb_cross_model_probe_emnlp2025_trillium}"
else
  OUT_BASE="${OUT_BASE:-$PROJECT_ROOT/data/results/analysis/final_analysis_ready/whitebox/wb_cross_model_probe_emnlp2025_trillium}"
fi
OUT_DIR="$OUT_BASE/$RUN_TAG"
mkdir -p "$OUT_DIR"

echo "=== Trillium White-Box Probe (probe-only) ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Input:  $INPUT_FILE"
echo "Target: $TARGET_MODEL_NAME"
echo "RespHF: $RESPONSE_HF"
echo "VerHF:  $VERIFIER_HF"
echo "Out:    $OUT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
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
