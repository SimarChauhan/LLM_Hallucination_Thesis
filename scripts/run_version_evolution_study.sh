#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-configs/version_evolution_full_timeline.yaml}"
RUN_ID="${2:-run_version_evolution_full_$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"

# Safety-first defaults: dry-run + smoke on, full run off.
DO_DRY_RUN="${DO_DRY_RUN:-1}"
DO_SMOKE="${DO_SMOKE:-1}"
DO_FULL="${DO_FULL:-0}"
DO_PHASE2="${DO_PHASE2:-1}"
DO_ANALYSIS="${DO_ANALYSIS:-1}"
SMOKE_MAX_QUESTIONS="${SMOKE_MAX_QUESTIONS:-5}"

# Optional comma-separated track filter, e.g. TRACKS=qwen-frontier or TRACKS=openai-capability,qwen-small
TRACKS="${TRACKS:-}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not executable: $PYTHON_BIN" >&2
  exit 1
fi

MODEL_ARGS=()
if [[ -n "$TRACKS" ]]; then
  TRACK_MODEL_NAMES=()
  while IFS= read -r line; do
    if [[ -n "$line" ]]; then
      TRACK_MODEL_NAMES+=("$line")
    fi
  done < <("$PYTHON_BIN" - "$CONFIG_PATH" "$TRACKS" <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
tracks = {t.strip() for t in sys.argv[2].split(",") if t.strip()}
models = []
for key in ("commercial_models", "opensource_models"):
    for m in cfg.get(key, []) or []:
        if m.get("track") in tracks:
            models.append(str(m.get("name") or f"{m.get('provider')}/{m.get('model')}"))
for name in models:
    print(name)
PY
  )
  if [[ "${#TRACK_MODEL_NAMES[@]}" -eq 0 ]]; then
    echo "No models matched TRACKS=$TRACKS in $CONFIG_PATH" >&2
    exit 1
  fi
  MODEL_ARGS+=(--models)
  MODEL_ARGS+=("${TRACK_MODEL_NAMES[@]}")
fi

echo "============================================================"
echo "Version evolution study orchestrator"
echo "Config:           $CONFIG_PATH"
echo "Run ID:           $RUN_ID"
echo "Python:           $PYTHON_BIN"
echo "Tracks filter:    ${TRACKS:-<none>}"
echo "Dry run:          $DO_DRY_RUN"
echo "Smoke run:        $DO_SMOKE (max_questions=$SMOKE_MAX_QUESTIONS)"
echo "Full run:         $DO_FULL"
echo "Phase 2 enabled:  $DO_PHASE2"
echo "Trend analysis:   $DO_ANALYSIS"
echo "============================================================"

if [[ "$DO_DRY_RUN" == "1" ]]; then
  echo ""
  echo "[Step 1] Dry-run validation"
  DRY_CMD=("$PYTHON_BIN" scripts/run_pipeline.py --config "$CONFIG_PATH" --dry-run)
  if [[ "${#MODEL_ARGS[@]}" -gt 0 ]]; then
    DRY_CMD+=("${MODEL_ARGS[@]}")
  fi
  "${DRY_CMD[@]}"
fi

SMOKE_CONFIG_PATH=""
if [[ "$DO_SMOKE" == "1" ]]; then
  echo ""
  echo "[Step 2] Smoke run (${SMOKE_MAX_QUESTIONS} questions/model)"
  SMOKE_CONFIG_PATH="$(mktemp "/tmp/version_evolution_smoke.XXXXXX")"
  "$PYTHON_BIN" - "$CONFIG_PATH" "$SMOKE_CONFIG_PATH" "$SMOKE_MAX_QUESTIONS" <<'PY'
import sys, yaml
from pathlib import Path
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
max_q = int(sys.argv[3])
cfg = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
dataset = cfg.setdefault("dataset", {})
dataset["max_questions"] = max_q
for bench in dataset.get("benchmarks", []) or []:
    bench["max_questions"] = max_q
out = cfg.setdefault("output", {})
results_file = str(out.get("results_file", "results.jsonl"))
eval_file = str(out.get("evaluated_file", "results_eval.jsonl"))
parquet_file = str(out.get("parquet_file", "results.parquet"))
suffix_jsonl = f"_smoke{max_q}.jsonl"
suffix_parquet = f"_smoke{max_q}.parquet"
if not results_file.endswith(suffix_jsonl):
    out["results_file"] = results_file.replace(".jsonl", suffix_jsonl)
if not eval_file.endswith(suffix_jsonl):
    out["evaluated_file"] = eval_file.replace(".jsonl", suffix_jsonl)
if not parquet_file.endswith(suffix_parquet):
    out["parquet_file"] = parquet_file.replace(".parquet", suffix_parquet)
dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(dst)
PY

  SMOKE_RUN_ID="${RUN_ID}_smoke${SMOKE_MAX_QUESTIONS}"
  SMOKE_CMD=("$PYTHON_BIN" scripts/run_pipeline.py --config "$SMOKE_CONFIG_PATH" --run-id "$SMOKE_RUN_ID")
  if [[ "${#MODEL_ARGS[@]}" -gt 0 ]]; then
    SMOKE_CMD+=("${MODEL_ARGS[@]}")
  fi
  "${SMOKE_CMD[@]}"
  if [[ "$DO_PHASE2" == "1" ]]; then
    REEVAL_SMOKE_CMD=("$PYTHON_BIN" scripts/reeval_results.py \
      --config "$SMOKE_CONFIG_PATH" \
      --run-id "$SMOKE_RUN_ID" \
      --strict-comparability \
      --force-recompute \
      --use-llm-judge)
    if [[ "${#MODEL_ARGS[@]}" -gt 0 ]]; then
      REEVAL_SMOKE_CMD+=("${MODEL_ARGS[@]}")
    fi
    "${REEVAL_SMOKE_CMD[@]}"
  fi
  if [[ "$DO_ANALYSIS" == "1" ]]; then
    "$PYTHON_BIN" scripts/analyze_version_evolution.py \
      --config "$SMOKE_CONFIG_PATH" \
      --run-id "$SMOKE_RUN_ID"
  fi
fi

if [[ "$DO_FULL" == "1" ]]; then
  echo ""
  echo "[Step 3] Full run"
  FULL_CMD=("$PYTHON_BIN" scripts/run_pipeline.py --config "$CONFIG_PATH" --run-id "$RUN_ID")
  if [[ "${#MODEL_ARGS[@]}" -gt 0 ]]; then
    FULL_CMD+=("${MODEL_ARGS[@]}")
  fi
  "${FULL_CMD[@]}"
  if [[ "$DO_PHASE2" == "1" ]]; then
    REEVAL_FULL_CMD=("$PYTHON_BIN" scripts/reeval_results.py \
      --config "$CONFIG_PATH" \
      --run-id "$RUN_ID" \
      --strict-comparability \
      --force-recompute \
      --use-llm-judge)
    if [[ "${#MODEL_ARGS[@]}" -gt 0 ]]; then
      REEVAL_FULL_CMD+=("${MODEL_ARGS[@]}")
    fi
    "${REEVAL_FULL_CMD[@]}"
  fi
  if [[ "$DO_ANALYSIS" == "1" ]]; then
    "$PYTHON_BIN" scripts/analyze_version_evolution.py \
      --config "$CONFIG_PATH" \
      --run-id "$RUN_ID"
  fi
fi

if [[ -n "$SMOKE_CONFIG_PATH" && -f "$SMOKE_CONFIG_PATH" ]]; then
  rm -f "$SMOKE_CONFIG_PATH"
fi

echo ""
echo "Completed."
echo "Main run id: $RUN_ID"
