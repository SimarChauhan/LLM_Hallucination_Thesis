# Results directory layout (curated thesis bundle)

This repository keeps only the artifacts needed to audit and reproduce thesis numbers.

## Top level

- `evaluated/`: phase-2 evaluated JSONL files used for O1 and as inputs to overlap/version analyses.
- `analysis/`: precomputed analysis outputs used directly in thesis tables.
- `whitebox/`: synced white-box probe outputs used for O2 summaries.

## Kept evaluated files

- `evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl`:
  main 4,842-row evaluated dataset for thesis number verification.
- `evaluated/run_qwen_new_only_807_full_retry2_20260315T193059Z/*.jsonl`:
  O3 Qwen-family rerun output.
- `evaluated/run_llama_new_only_807_p1_20260315T222326Z/*.jsonl`:
  O3 Llama-family rerun output.
- `evaluated/run_grok_new_only_807_p1_xai_20260315T224013Z/*.jsonl`:
  O3 Grok-family rerun output.

## Kept analysis outputs

- `analysis/cross_model_ce_overlap_t1p0_20260327/`:
  strict cross-model CE overlap outputs (including same-wrong totals used in thesis text).
- `analysis/version_evolution_equiv_only_20260319/`:
  O3 model summaries, pairwise deltas, trend tables, and validation manifest.

## Notes

- Large transient artifacts and generated media were removed intentionally.
- If you need to regenerate derived outputs, use scripts in `scripts/` and write results to a new output folder.
