#!/bin/bash
#SBATCH --job-name=wb_probe
#SBATCH --account=def-ilie         # Replace with your CCDB account (e.g. rpp-<alliance_username>)
#SBATCH --time=1-12:00:00                    # Killarney Performance Compute runs can be long for 80B+400B replay probes
#SBATCH --cpus-per-task=24
#SBATCH --mem=512G                           # Performance nodes have 2TB RAM; increase if CPU offload occurs
#SBATCH --gres=gpu:8                         # Killarney Performance Compute: 8 x H100 SXM 80GB
#SBATCH --output=slurm_wb_probe_%j.out
#SBATCH --error=slurm_wb_probe_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ssimran5@uwo.ca

# -----------------------------------------------------------------------------
# White-box cross-model probe on Alliance Canada (Killarney Performance Compute)
# Run from repo root:  sbatch scripts/slurm_run_wb_probe.sh
# Or:  sbatch --export=TARGET=DeepSeek,RESPONSE=...,VERIFIER=... scripts/slurm_run_wb_probe.sh
# -----------------------------------------------------------------------------

set -euo pipefail

# Project root: prefer the directory where `sbatch` was submitted from.
# Slurm copies the script to a spool dir, so BASH_SOURCE[0] is not reliable there.
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
RUNNER_PY="$PROJECT_ROOT/scripts/run_wb_cross_model_probe_emnlp2025.py"
if [[ ! -f "$RUNNER_PY" ]]; then
  echo "ERROR: runner not found: $RUNNER_PY" >&2
  echo "Set PROJECT_ROOT to your repo root in sbatch --export, e.g. PROJECT_ROOT=/home/ssimran5/LLM_Hallucination_Measure" >&2
  exit 1
fi

# Which (target, response, verifier) to run. Override via sbatch --export=...
# TARGET_NAME must match the "model" field in your JSONL exactly.
# Defaults below are set for the user's replay-matched Qwen3 Next + Llama4 Maverick verifier run.
TARGET_NAME="${TARGET_NAME:-Qwen3 Next 80B (OpenRouter)}"
RESPONSE_HF="${RESPONSE_HF:-Qwen/Qwen3-Next-80B-A3B-Instruct}"
VERIFIER_HF="${VERIFIER_HF:-meta-llama/Llama-4-Maverick-17B-128E-Instruct}"
SUBSET="${SUBSET:-both}"
LAYER_CANDIDATES="${LAYER_CANDIDATES:-last8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"           # Do not force bf16 for Maverick; preserve checkpoint dtype if possible
ENCODER_DEVICE_MAP="${ENCODER_DEVICE_MAP:-auto}"
SEQUENTIAL_ENCODERS="${SEQUENTIAL_ENCODERS:-1}"

# Paths (defaults work if you run from repo with data in place)
INPUT_FILE="${INPUT_FILE:-$PROJECT_ROOT/data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl}"
OUT_BASE="${OUT_BASE:-$PROJECT_ROOT/data/results/analysis/final_analysis_ready/whitebox/wb_cross_model_probe_emnlp2025}"

# Use scratch if available (faster I/O on many clusters)
if [ -n "$SCRATCH" ]; then
  OUT_BASE="${SCRATCH}/$(basename "$PROJECT_ROOT")/wb_probe_out"
  mkdir -p "$OUT_BASE"
fi

# HuggingFace token (for gated models e.g. Llama). Set in env or .env.
if [[ -z "${HF_TOKEN:-}" && -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi

# Python: use conda if available, else system
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate base
  if conda env list | grep -q "wb_probe"; then
    conda activate wb_probe
  fi
fi

# Optional: limit to one GPU for debugging (e.g. CUDA_VISIBLE_DEVICES=0)
# export CUDA_VISIBLE_DEVICES=0

# Reduce allocator fragmentation for large-model sequential loading.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

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

python - <<'PY'
import importlib.util
import os
import sys
from transformers import AutoConfig

response = os.environ["RESPONSE_HF"]
verifier = os.environ["VERIFIER_HF"]
errors = []

def check_config(model_id: str, role: str) -> None:
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(f"[preflight] {role} config ok: {model_id} -> {type(cfg).__name__}")
    except Exception as exc:
        errors.append(f"{role} config load failed for {model_id}: {type(exc).__name__}: {exc}")

check_config(response, "response")
check_config(verifier, "verifier")

if "glm-5" in response.lower() or "glm-5" in verifier.lower():
    if importlib.util.find_spec("triton") is None:
        errors.append(
            "GLM-5 requested but Python package 'triton' is missing. "
            "Install triton in the runtime environment or use a non-GLM verifier."
        )

if errors:
    print("[preflight] one or more checks failed:", file=sys.stderr)
    for item in errors:
        print(f"  - {item}", file=sys.stderr)
    sys.exit(1)
PY

echo "=============================================="
echo "White-box probe: target=$TARGET_NAME"
echo "  response=$RESPONSE_HF  verifier=$VERIFIER_HF"
echo "  input=$INPUT_FILE"
echo "  output=$OUT_BASE"
echo "  subset=$SUBSET layers=$LAYER_CANDIDATES batch_size=$BATCH_SIZE torch_dtype=$TORCH_DTYPE"
echo "  encoder_device_map=$ENCODER_DEVICE_MAP sequential_encoders=$SEQUENTIAL_ENCODERS"
echo "=============================================="

CMD=(
  python "$RUNNER_PY"
  --input "$INPUT_FILE"
  --output-dir "$OUT_BASE/${TARGET_NAME// /_}_$(basename "$VERIFIER_HF" | tr '/' '_')"
  --target-model-name "$TARGET_NAME"
  --response-model-path-or-hf-id "$RESPONSE_HF"
  --verifier-model-path-or-hf-id "$VERIFIER_HF"
  --layer-candidates "$LAYER_CANDIDATES"
  --torch-dtype "$TORCH_DTYPE"
  --encoder-device-map "$ENCODER_DEVICE_MAP"
  --encoder-device cuda
  --probe-device cuda
  --batch-size "$BATCH_SIZE"
  --subset "$SUBSET"
)

if [[ "$SEQUENTIAL_ENCODERS" == "1" ]]; then
  CMD+=(--sequential-encoders)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Done. Check output dir and slurm_wb_probe_${SLURM_JOB_ID}.out"
