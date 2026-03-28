#!/bin/bash
# =============================================================================
# Submit all 6 white-box probe jobs to the Digital Alliance (Killarney) cluster.
# Run from the repo root:  bash scripts/submit_all_wb_probes.sh
# Default profile avoids known-failing model combos in this environment.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SCRIPT="${PROJECT_ROOT}/scripts/slurm_run_wb_probe.sh"
ACCOUNT="def-ilie"
PARTITION="gpubase_bynode_b4"
GPUS="gpu:h100:8"
CPUS=24
HF_CACHE="/scratch/ssimran5/hf_cache"
RUN_PROFILE="${RUN_PROFILE:-compatible}"    # compatible | full
DEEPSEEK_PROXY_RESPONSE="${DEEPSEEK_PROXY_RESPONSE:-Qwen/Qwen3-Next-80B-A3B-Instruct}"
SAFE_VERIFIER_A="${SAFE_VERIFIER_A:-meta-llama/Llama-3.1-8B-Instruct}"
SAFE_VERIFIER_B="${SAFE_VERIFIER_B:-Qwen/Qwen2.5-0.5B-Instruct}"

if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: Slurm launcher not found at: $SCRIPT" >&2
  exit 1
fi

COMMON_EXPORT="PROJECT_ROOT=${PROJECT_ROOT},SUBSET=both,LAYER_CANDIDATES=last8,BATCH_SIZE=1,TORCH_DTYPE=auto,ENCODER_DEVICE_MAP=auto,SEQUENTIAL_ENCODERS=1,HF_HOME=${HF_CACHE},TRANSFORMERS_CACHE=${HF_CACHE}/transformers"

echo "=============================================="
echo "Submitting 6 white-box probe jobs (profile: ${RUN_PROFILE})..."
echo "=============================================="

if [[ "$RUN_PROFILE" == "compatible" ]]; then
  echo "Using compatible profile:"
  echo "  - avoids DeepSeek-V3.2 local encoder load (uses proxy: ${DEEPSEEK_PROXY_RESPONSE})"
  echo "  - avoids GLM-5-FP8 verifier in current env"
  echo "  - uses verifiers: ${SAFE_VERIFIER_A} and ${SAFE_VERIFIER_B}"

  # ----------------------------------------------------------------------------
  # COMPATIBLE JOBS (512G RAM, 1-12h)
  # ----------------------------------------------------------------------------
  JID1=$(sbatch --parsable \
    -A "$ACCOUNT" -p "$PARTITION" --gres="$GPUS" \
    -t 1-12:00:00 --cpus-per-task="$CPUS" --mem=512G \
    --export=ALL,TARGET_NAME='Qwen3 Next 80B (OpenRouter)',RESPONSE_HF='Qwen/Qwen3-Next-80B-A3B-Instruct',VERIFIER_HF="${SAFE_VERIFIER_A}",$COMMON_EXPORT \
    "$SCRIPT")
  echo "[1/6] Qwen->VerifierA    job ID: $JID1"

  JID2=$(sbatch --parsable \
    -A "$ACCOUNT" -p "$PARTITION" --gres="$GPUS" \
    -t 1-12:00:00 --cpus-per-task="$CPUS" --mem=512G \
    --export=ALL,TARGET_NAME='Qwen3 Next 80B (OpenRouter)',RESPONSE_HF='Qwen/Qwen3-Next-80B-A3B-Instruct',VERIFIER_HF="${SAFE_VERIFIER_B}",$COMMON_EXPORT \
    "$SCRIPT")
  echo "[2/6] Qwen->VerifierB    job ID: $JID2"

  JID3=$(sbatch --parsable \
    -A "$ACCOUNT" -p "$PARTITION" --gres="$GPUS" \
    -t 1-12:00:00 --cpus-per-task="$CPUS" --mem=512G \
    --export=ALL,TARGET_NAME='Llama 4 Maverick 17B (Groq)',RESPONSE_HF='meta-llama/Llama-4-Maverick-17B-128E-Instruct',VERIFIER_HF="${SAFE_VERIFIER_A}",$COMMON_EXPORT \
    "$SCRIPT")
  echo "[3/6] Llama->VerifierA   job ID: $JID3"

  JID4=$(sbatch --parsable \
    -A "$ACCOUNT" -p "$PARTITION" --gres="$GPUS" \
    -t 1-12:00:00 --cpus-per-task="$CPUS" --mem=512G \
    --export=ALL,TARGET_NAME='Llama 4 Maverick 17B (Groq)',RESPONSE_HF='meta-llama/Llama-4-Maverick-17B-128E-Instruct',VERIFIER_HF="${SAFE_VERIFIER_B}",$COMMON_EXPORT \
    "$SCRIPT")
  echo "[4/6] Llama->VerifierB   job ID: $JID4"

  JID5=$(sbatch --parsable \
    -A "$ACCOUNT" -p "$PARTITION" --gres="$GPUS" \
    -t 1-12:00:00 --cpus-per-task="$CPUS" --mem=512G \
    --export=ALL,TARGET_NAME='DeepSeek V3.2 (DeepSeek)',RESPONSE_HF="${DEEPSEEK_PROXY_RESPONSE}",VERIFIER_HF="${SAFE_VERIFIER_A}",$COMMON_EXPORT \
    "$SCRIPT")
  echo "[5/6] DeepSeek->VerifierA job ID: $JID5"

  JID6=$(sbatch --parsable \
    -A "$ACCOUNT" -p "$PARTITION" --gres="$GPUS" \
    -t 1-12:00:00 --cpus-per-task="$CPUS" --mem=512G \
    --export=ALL,TARGET_NAME='DeepSeek V3.2 (DeepSeek)',RESPONSE_HF="${DEEPSEEK_PROXY_RESPONSE}",VERIFIER_HF="${SAFE_VERIFIER_B}",$COMMON_EXPORT \
    "$SCRIPT")
  echo "[6/6] DeepSeek->VerifierB job ID: $JID6"

elif [[ "$RUN_PROFILE" == "full" ]]; then
  echo "Using full profile (includes DeepSeek-V3.2 + GLM-5-FP8; known to fail in current env if deps/support are missing)."

  # ---------------------------------------------------------------------------
  # NON-GLM JOBS  (512G RAM, 1-12h)
  # ---------------------------------------------------------------------------
  JID1=$(sbatch --parsable \
    -A "$ACCOUNT" -p "$PARTITION" --gres="$GPUS" \
    -t 1-12:00:00 --cpus-per-task="$CPUS" --mem=512G \
    --export=ALL,TARGET_NAME='Qwen3 Next 80B (OpenRouter)',RESPONSE_HF='Qwen/Qwen3-Next-80B-A3B-Instruct',VERIFIER_HF='meta-llama/Llama-4-Maverick-17B-128E-Instruct',$COMMON_EXPORT \
    "$SCRIPT")
  echo "[1/6] Qwen->Llama        job ID: $JID1"

  JID2=$(sbatch --parsable \
    -A "$ACCOUNT" -p "$PARTITION" --gres="$GPUS" \
    -t 1-12:00:00 --cpus-per-task="$CPUS" --mem=512G \
    --export=ALL,TARGET_NAME='Llama 4 Maverick 17B (Groq)',RESPONSE_HF='meta-llama/Llama-4-Maverick-17B-128E-Instruct',VERIFIER_HF='Qwen/Qwen3-Next-80B-A3B-Instruct',$COMMON_EXPORT \
    "$SCRIPT")
  echo "[2/6] Llama->Qwen        job ID: $JID2"

  JID3=$(sbatch --parsable \
    -A "$ACCOUNT" -p "$PARTITION" --gres="$GPUS" \
    -t 1-12:00:00 --cpus-per-task="$CPUS" --mem=1T \
    --export=ALL,TARGET_NAME='DeepSeek V3.2 (DeepSeek)',RESPONSE_HF='deepseek-ai/DeepSeek-V3.2',VERIFIER_HF='meta-llama/Llama-4-Maverick-17B-128E-Instruct',$COMMON_EXPORT \
    "$SCRIPT")
  echo "[3/6] DeepSeek->Llama    job ID: $JID3"

  # ---------------------------------------------------------------------------
  # GLM JOBS  (1T RAM, 2-00h)
  # ---------------------------------------------------------------------------
  JID4=$(sbatch --parsable \
    -A "$ACCOUNT" -p "$PARTITION" --gres="$GPUS" \
    -t 2-00:00:00 --cpus-per-task="$CPUS" --mem=1T \
    --export=ALL,TARGET_NAME='Qwen3 Next 80B (OpenRouter)',RESPONSE_HF='Qwen/Qwen3-Next-80B-A3B-Instruct',VERIFIER_HF='zai-org/GLM-5-FP8',$COMMON_EXPORT \
    "$SCRIPT")
  echo "[4/6] Qwen->GLM-5        job ID: $JID4"

  JID5=$(sbatch --parsable \
    -A "$ACCOUNT" -p "$PARTITION" --gres="$GPUS" \
    -t 2-00:00:00 --cpus-per-task="$CPUS" --mem=1T \
    --export=ALL,TARGET_NAME='Llama 4 Maverick 17B (Groq)',RESPONSE_HF='meta-llama/Llama-4-Maverick-17B-128E-Instruct',VERIFIER_HF='zai-org/GLM-5-FP8',$COMMON_EXPORT \
    "$SCRIPT")
  echo "[5/6] Llama->GLM-5       job ID: $JID5"

  JID6=$(sbatch --parsable \
    -A "$ACCOUNT" -p "$PARTITION" --gres="$GPUS" \
    -t 2-00:00:00 --cpus-per-task="$CPUS" --mem=1T \
    --export=ALL,TARGET_NAME='DeepSeek V3.2 (DeepSeek)',RESPONSE_HF='deepseek-ai/DeepSeek-V3.2',VERIFIER_HF='zai-org/GLM-5-FP8',$COMMON_EXPORT \
    "$SCRIPT")
  echo "[6/6] DeepSeek->GLM-5    job ID: $JID6"
else
  echo "ERROR: RUN_PROFILE must be 'compatible' or 'full' (got: ${RUN_PROFILE})" >&2
  exit 1
fi

echo "=============================================="
echo "All 6 jobs submitted. Check queue with:"
echo "  squeue -u ssimran5"
echo "=============================================="
