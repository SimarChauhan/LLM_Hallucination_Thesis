# Reproducibility Guide

This guide covers the minimum steps to verify the thesis results from the curated repository contents.

## 1) Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Bundle Integrity Check 

```bash
python scripts/verify_reproducibility_bundle.py
```

Expected core checks include:
- main evaluated rows: `4842` (`807 x 6`)
- cross-model CE overlap totals (strict `t=1.0`): overlap=`720`, same-wrong=`529`, same-wrong%=`73.5`
- Jaccard range: `0.219` to `0.360`
- O3 summary rows across 12 model versions: `9684`
- white-box run reports present: `18`

## 3) O1 Thesis Number Verification 

```bash
python scripts/validate_report_numbers.py
```

This re-computes and validates thesis-reported values across:
- dataset integrity
- overall rates
- per-model metrics
- threshold sensitivity
- category partitions
- overlap counts

## 4) Recompute Cross-Model CE Overlap (O1 overlap analysis)

```bash
python scripts/compute_shared_ce_analysis.py \
  --ce-thresholds 1.0 \
  --equivalence-method nli_hybrid \
  --nli-batch-size 16 \
  --write-canonical-1p0 \
  --output-dir data/results/analysis/cross_model_ce_overlap_t1p0_20260327
```

Note: this uses local NLI inference and may download the NLI model on first run.

## 5) Rebuild O3 Version-Evolution Package

```bash
python scripts/rebuild_version_evolution_package.py
```

This regenerates:
- `data/results/analysis/version_evolution_equiv_only_20260319/model_summary_*.csv`
- `pairwise_deltas_*.csv`
- `trend_tests_*.csv`
- `validation_checks.json`
- `report.md`

## 6) Aggregate O2 White-Box Probe Summaries

```bash
python scripts/summarize_synced_wb_probe_runs.py \
  --artifacts-root data/results/whitebox/nibi_sync_2026-03-15/live_pull \
  --output-dir data/results/analysis/whitebox_repro_summary
```
