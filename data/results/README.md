# Results Artifacts (Curated)

This directory contains only the result artifacts needed to audit and reproduce thesis numbers.

## Layout

- `evaluated/`: evaluated JSONL outputs used for O1 and as inputs to overlap/version analyses
- `analysis/`: precomputed analysis outputs used directly in thesis tables and checks
- `whitebox/`: synced white-box probe outputs used for O2 summaries

## Main Inputs Kept

- `evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl`
  - primary O1 evaluated dataset (`4,842` rows)

- O3 family rerun evaluated outputs:
  - `evaluated/run_qwen_new_only_807_full_retry2_20260315T193059Z/*.jsonl`
  - `evaluated/run_llama_new_only_807_p1_20260315T222326Z/*.jsonl`
  - `evaluated/run_grok_new_only_807_p1_xai_20260315T224013Z/*.jsonl`

## Analysis Outputs Kept

- `analysis/cross_model_ce_overlap_t1p0_20260327/`
  - strict CE overlap package used for `720` total overlap and `529` same-wrong totals

- `analysis/version_evolution_equiv_only_20260319/`
  - O3 summaries, deltas, trends, and validation manifest

## Regeneration

Use scripts in `scripts/` and write regenerated outputs to a new timestamped folder.

## Notes

- Large transient artifacts and generated media were intentionally removed.
- This curated layout is designed for reproducibility verification and thesis audit.
