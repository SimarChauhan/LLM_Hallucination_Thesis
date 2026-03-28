#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-configs/version_evolution_claude_grok_chatgpt.yaml}"
RUN_ID="${2:-run_version_evolution_triplet_$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PHASE2="${SKIP_PHASE2:-0}"
FAMILIES="${FAMILIES:-OpenAI,Claude,Grok}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not executable: $PYTHON_BIN" >&2
  exit 1
fi

COMMON_ARGS=(--config "$CONFIG_PATH" --run-id "$RUN_ID")
if [[ "$DRY_RUN" == "1" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

echo "============================================================"
echo "Version evolution run: Claude + Grok + ChatGPT"
echo "Config:  $CONFIG_PATH"
echo "Run ID:  $RUN_ID"
echo "Python:  $PYTHON_BIN"
echo "Dry run: $DRY_RUN"
echo "Families: $FAMILIES"
echo "============================================================"

IFS=',' read -r -a FAMILY_ARRAY <<< "$FAMILIES"
phase_total=$(( ${#FAMILY_ARRAY[@]} + 1 ))
phase_idx=1
for family in "${FAMILY_ARRAY[@]}"; do
  family_trimmed="$(echo "$family" | xargs)"
  if [[ -z "$family_trimmed" ]]; then
    continue
  fi
  echo ""
  echo "[Phase ${phase_idx}/${phase_total}] ${family_trimmed} pair"
  "$PYTHON_BIN" scripts/run_pipeline.py "${COMMON_ARGS[@]}" --models "$family_trimmed"
  phase_idx=$((phase_idx + 1))
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo ""
  echo "Dry-run complete. No API calls were executed."
  exit 0
fi

if [[ "$SKIP_PHASE2" == "1" ]]; then
  echo ""
  echo "Phase 2 re-evaluation skipped (SKIP_PHASE2=1)."
  exit 0
fi

echo ""
echo "[Phase ${phase_idx}/${phase_total}] Re-evaluation (strict comparability)"
"$PYTHON_BIN" scripts/reeval_results.py \
  --config "$CONFIG_PATH" \
  --run-id "$RUN_ID" \
  --strict-comparability \
  --force-recompute \
  --use-llm-judge

echo ""
echo "Completed run_id: $RUN_ID"
echo "Raw output:       data/results/raw/$RUN_ID/results_version_evolution_triplet.jsonl"
echo "Evaluated output: data/results/evaluated/$RUN_ID/results_version_evolution_triplet_eval.jsonl"
