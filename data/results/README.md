# Results directory layout

## Top level

- **`raw/`**: Phase 1 generation outputs.
- **`evaluated/`**: Phase 2 judged/equivalence outputs.
- **`analysis/`**: analysis artifacts and thesis reports.

## `evaluated/` organization

Root (kept for compatibility with existing scripts):
- `results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl` (primary v2 thesis file)
- `results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.jsonl` (reference analysis-ready file)
- `results_v2_phase2_eval_no_gemini_4842.final.jsonl` (final evaluated file)
- `retry_queue.jsonl` (active retry queue)
- `results_v2.parquet`

Subfolders:
- **`summaries/`**: markdown/json summary exports.
- **`hardcases/`**: hard-case subsets and rerun outputs.
- **`retry_queues/`**: derived retry queue snapshots.
- **`snapshots/`**: historical/pre-sync/offline-repair snapshots.
- **`canonical/`**: reserved for future canonicalized copies.

## `analysis/` organization

- **`final_analysis_ready/`**: prior deep analysis and LaTeX report artifacts.
- **`v2_thesis/`**: thesis-grade analysis package (metrics, stats, ablations, audit, figures, writeup).

The repo root path `analysis/v2_thesis` is kept as a symlink to `data/results/analysis/v2_thesis` for backward compatibility.

## Script defaults

- `scripts/run_pipeline.py` writes raw outputs to `data/results/raw`.
- `scripts/reeval_results.py` reads raw and writes evaluated outputs to `data/results/evaluated`.
- `scripts/analyze_results.py` reads evaluated outputs and writes analysis outputs under `data/results/analysis`.
