# Cross-Model CE Overlap Package (Strict t=1.0)

This folder contains strict CE overlap artifacts generated on `2026-03-27`.

## Configuration

- Source data:
  `data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl`
- CE threshold: `1.0` (`error_label_1.0`)
- Equivalence method: `nli_hybrid`
- Borderline LLM fallback in this run: `false`

## Key Totals

- Model pairs: `15`
- Total CE overlap: `720`
- Total same-wrong: `529`
- Total unclear: `35`
- Overall same-wrong rate: `73.5%`
- Jaccard range: `0.219` to `0.360` (mean `0.277`)

## Files

- `cross_model_ce_overlap_semantic_nlihybrid_t1p0.csv`
- `cross_model_ce_overlap_semantic.csv` (canonical strict copy)
- `summary_t1p0_nlihybrid.json`

Note: `.tex` table exports are intentionally not tracked in this curated repository.

## Reproduction Command

```bash
python scripts/compute_shared_ce_analysis.py \
  --ce-thresholds 1.0 \
  --equivalence-method nli_hybrid \
  --nli-batch-size 16 \
  --write-canonical-1p0 \
  --output-dir data/results/analysis/cross_model_ce_overlap_t1p0_20260327
```
