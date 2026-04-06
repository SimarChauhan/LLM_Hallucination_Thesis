# LLM Hallucination Thesis Reproducibility Repository

This repository contains the code, configurations, and curated artifacts used in the thesis:

**Measuring Self-Consistent Errors in Large Language Models and Detecting Them via Proxy Cross-Model Probing**

The manuscript/PDF is submitted separately. This repository is for **analysis and reproducibility**.

## Thesis Objectives Covered

1. **O1 (Black-box CE measurement):** quantify self-consistent errors (CE) and inconsistent errors (IE) on TruthfulQA.
2. **O2 (White-box probing):** detect CE using cross-model probing with proxy encoders.
3. **O3 (Version evolution):** track CE behavior across model generations.

## Final Report Snapshot (Verified)

These headline numbers are verified by scripts in this repository:

- Main evaluated rows: `4,842` (`807 questions x 6 models`)
- Overall counts: `Correct=3,063`, `Incorrect=1,560`, `Not Attempted=219`
- CE at `t=1.0`: `656` rows (`13.5%` of all rows, `42.1%` of incorrect rows)
- Cross-model overlap (`t=1.0`): `total overlap=720`, `same wrong=529`, `same-wrong=73.5%`
- Jaccard range (pairwise overlap): `0.219` to `0.360`
- White-box run reports present: `18`
- Version-evolution rows: `9,684` (`12 models x 807`)

## Repository Structure

- `src/`: core pipeline modules (inference, labeling, equivalence, reliability, storage)
- `scripts/`: experiment and analysis scripts
- `configs/`: model and run configurations
- `data/`: curated input and result artifacts used by thesis analyses
- `tests/`: automated checks for key components
- `docs/`: reproducibility and supporting notes

Additional local readmes:
- `data/results/README.md`
- `data/results/evaluated/README.md`
- `data/results/analysis/cross_model_ce_overlap_t1p0_20260327/README.md`

## Quick Start

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment

```bash
cp .env.example .env
```

Add API credentials in `.env` if you plan to run API-backed collection or reruns.

## Reproducibility Commands

### A) Verify bundle integrity and key totals

```bash
python scripts/verify_reproducibility_bundle.py
```

### B) Validate thesis-reported numbers

```bash
python scripts/validate_report_numbers.py
```

### C) Recompute selected analyses

```bash
# O1 cross-model CE overlap (strict t=1.0)
python scripts/compute_shared_ce_analysis.py \
  --ce-thresholds 1.0 \
  --equivalence-method nli_hybrid \
  --nli-batch-size 16 \
  --write-canonical-1p0 \
  --output-dir data/results/analysis/cross_model_ce_overlap_t1p0_20260327

# O3 version-evolution package
python scripts/rebuild_version_evolution_package.py

# O2 white-box probe summary aggregation
python scripts/summarize_synced_wb_probe_runs.py \
  --artifacts-root data/results/whitebox/nibi_sync_2026-03-15/live_pull \
  --output-dir data/results/analysis/whitebox_repro_summary
```

Detailed runbook: `docs/reproducibility.md`

### D) Optional: sync white-box probe artifacts from a cluster

If you maintain probe outputs on a remote host (e.g. Alliance Canada), you can pull logs and `wb_probe_out` locally and refresh summary tables:

```bash
bash scripts/sync_nibi_wb_probe_artifacts.sh
```

By default this also runs `scripts/summarize_synced_wb_probe_runs.py` into `<dest>/summaries`. Use `--skip-summary` to disable. (There is no bundled post-sync comparison report script in this repository.)

## Notes

- Generated PDFs, LaTeX build artifacts, caches, and temporary files are intentionally excluded.
- Artifacts needed to audit and reproduce reported numbers are retained.
- Full API reruns may require credentials, compute time, and external model availability.

## License

See `LICENSE`.
