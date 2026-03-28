#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH="$1"
INPUT_PATH="$2"
OUTPUT_PATH="$3"
RUN_ID="$4"
module load python/3.11 gcc arrow/19.0.1
source /home/ssimran5/LLM_Hallucination_Measure/.venv_nibi311/bin/activate
cd /home/ssimran5/LLM_Hallucination_Measure
export RESULTS_DIR_ABSOLUTE=/home/ssimran5/LLM_Hallucination_Measure/data/results
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
python -u scripts/tmp_nibi/reeval_equiv_only.py \
  --config "$CONFIG_PATH" \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_PATH" \
  --run-id "$RUN_ID" \
  --hybrid-calibration-file /home/ssimran5/LLM_Hallucination_Measure/data/calibration/hybrid_thresholds_frozen.json
