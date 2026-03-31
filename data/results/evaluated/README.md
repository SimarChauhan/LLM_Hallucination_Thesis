# Evaluated Outputs (Curated)

This folder contains only evaluated JSONL files required for thesis reproducibility.

## Primary O1 Dataset

- `results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl`
  - `4,842` rows
  - `807 questions x 6 models`
  - used by `scripts/validate_report_numbers.py` and overlap analysis

## O3 Evaluated Rerun Datasets

- `run_qwen_new_only_807_full_retry2_20260315T193059Z/results_version_evolution_qwen_new_only_eval.equiv_only_20260319.jsonl`
- `run_llama_new_only_807_p1_20260315T222326Z/results_version_evolution_llama_scale_version_807_eval.equiv_only_20260319.jsonl`
- `run_grok_new_only_807_p1_xai_20260315T224013Z/results_version_evolution_grok_new_only_807_eval.equiv_only_20260319.jsonl`

Each file has `2,421` rows (`807 x 3`). Combined with the shared/base version source used in O3 packaging, this yields the thesis total of `9,684` rows across `12` model versions.
