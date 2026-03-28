#!/usr/bin/env bash
# Run True White-Box (Tan et al.) pipeline in full precision (bfloat16).
#
# Requirements:
#   - Python env with torch, transformers (activate venv/conda first)
#   - HF_TOKEN for Llama (meta-llama) if not already logged in
#   - ~18GB+ memory: use --probe-sequential-encoders on M3 Pro 18GB
#
# Usage:
#   source .venv/bin/activate   # or: conda activate your_env
#   ./scripts/run_true_whitebox_full_precision.sh
#
# Or with explicit python:
#   python scripts/run_open_whitebox_end2end.py [options below]

set -e
cd "$(dirname "$0")/.."

MAX_QUESTIONS="${MAX_QUESTIONS:-50}"
SEQUENTIAL="${SEQUENTIAL:-}"

CMD=(
  python3 scripts/run_open_whitebox_end2end.py
  --target-model-name "Qwen2.5-7B"
  --response-model-path-or-hf-id "Qwen/Qwen2.5-7B-Instruct"
  --verifier-model-path-or-hf-id "meta-llama/Llama-3.1-8B-Instruct"
  --max-questions "$MAX_QUESTIONS"
  --torch-dtype bfloat16
  --run-probe
)

if [[ -n "$SEQUENTIAL" ]] || [[ "$SEQUENTIAL" == "1" ]]; then
  CMD+=(--probe-sequential-encoders)
  echo "Using sequential encoders (low memory mode for ~18GB machines)"
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
