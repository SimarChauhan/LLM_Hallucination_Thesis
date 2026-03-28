#!/bin/bash
#SBATCH --job-name=wb-true-tri
#SBATCH --account=def-<PI_USERNAME>      # Replace (or pass via: sbatch -A ...)
#SBATCH --time=0-24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Trillium template: true white-box run (local open-model generation + labeling)
# followed by a probe run on the generated artifact.
#
# Why this uses two commands instead of `--run-probe` in one Python script:
# `run_open_whitebox_end2end.py` keeps the generation model loaded when it launches
# its probe subprocess, which can cause avoidable GPU memory pressure. This script
# runs generation/labeling first, then the probe as a separate command.
#
# Usage:
#   sbatch scripts/slurm_trillium_true_whitebox.sh
#
# Override defaults:
#   sbatch --export=ALL,TARGET_MODEL_NAME='Qwen2.5-7B',RESPONSE_HF='Qwen/Qwen2.5-7B-Instruct',VERIFIER_HF='meta-llama/Llama-3.1-8B-Instruct',MAX_QUESTIONS=100 scripts/slurm_trillium_true_whitebox.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs/slurm

# Optional Alliance module setup (uncomment if needed on Trillium).
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

INPUT_FILE="${INPUT_FILE:-$PROJECT_ROOT/data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl}"

TARGET_MODEL_NAME="${TARGET_MODEL_NAME:-Qwen2.5-7B}"
RESPONSE_HF="${RESPONSE_HF:-Qwen/Qwen2.5-7B-Instruct}"
VERIFIER_HF="${VERIFIER_HF:-meta-llama/Llama-3.1-8B-Instruct}"

MAX_QUESTIONS="${MAX_QUESTIONS:-50}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
NLI_DEVICE="${NLI_DEVICE:-cpu}"          # cpu is safer for GPU memory; set cuda for speed
PROBE_SEQUENTIAL_ENCODERS="${PROBE_SEQUENTIAL_ENCODERS:-1}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-4}"
PROBE_LAYER_CANDIDATES="${PROBE_LAYER_CANDIDATES:-last8}"

RUN_NAME="${RUN_NAME:-$(echo "$TARGET_MODEL_NAME" | tr ' /()' '_' | tr -s '_' )_trillium_${SLURM_JOB_ID:-manual}}"

if [[ -n "${SCRATCH:-}" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/$(basename "$PROJECT_ROOT")/whitebox/open_e2e}"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/results/analysis/final_analysis_ready/whitebox/open_e2e}"
fi
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR"

echo "=== Trillium True White-Box (2-stage) ==="
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "Input:      $INPUT_FILE"
echo "Target:     $TARGET_MODEL_NAME"
echo "ResponseHF: $RESPONSE_HF"
echo "VerifierHF: $VERIFIER_HF"
echo "Run dir:    $RUN_DIR"
echo "Max Q:      $MAX_QUESTIONS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo

# Stage 1: local open-model generation + labeling (no probe subprocess).
GEN_CMD=(
  python scripts/run_open_whitebox_end2end.py
  --input "$INPUT_FILE"
  --output-root "$OUTPUT_ROOT"
  --run-name "$RUN_NAME"
  --max-questions "$MAX_QUESTIONS"
  --target-model-name "$TARGET_MODEL_NAME"
  --response-model-path-or-hf-id "$RESPONSE_HF"
  --verifier-model-path-or-hf-id "$VERIFIER_HF"
  --device "$DEVICE"
  --torch-dtype "$TORCH_DTYPE"
  --nli-device "$NLI_DEVICE"
  --probe-batch-size "$PROBE_BATCH_SIZE"
  --probe-layer-candidates "$PROBE_LAYER_CANDIDATES"
)

echo "Stage 1 command: ${GEN_CMD[*]}"
"${GEN_CMD[@]}"

ANALYSIS_JSONL="$RUN_DIR/open_whitebox_e2e.analysis_ready.jsonl"
PROBE_DIR="$RUN_DIR/probe_emnlp2025"

if [[ ! -f "$ANALYSIS_JSONL" ]]; then
  echo "Expected stage-1 artifact not found: $ANALYSIS_JSONL" >&2
  exit 1
fi

# Stage 2: white-box probe on the newly generated local-open-model outputs.
PROBE_CMD=(
  python scripts/run_wb_cross_model_probe_emnlp2025.py
  --input "$ANALYSIS_JSONL"
  --output-dir "$PROBE_DIR"
  --target-model-name "$TARGET_MODEL_NAME"
  --subset both
  --response-model-path-or-hf-id "$RESPONSE_HF"
  --verifier-model-path-or-hf-id "$VERIFIER_HF"
  --layer-candidates "$PROBE_LAYER_CANDIDATES"
  --torch-dtype "$TORCH_DTYPE"
  --encoder-device cuda
  --probe-device cuda
  --batch-size "$PROBE_BATCH_SIZE"
)

if [[ "$PROBE_SEQUENTIAL_ENCODERS" == "1" ]]; then
  PROBE_CMD+=(--sequential-encoders)
fi

echo
echo "Stage 2 command: ${PROBE_CMD[*]}"
"${PROBE_CMD[@]}"

echo
echo "Done. Outputs:"
echo "  $RUN_DIR"
echo "  $PROBE_DIR"
