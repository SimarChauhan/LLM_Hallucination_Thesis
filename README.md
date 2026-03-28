# Thesis Repository: Self-Consistent Error Detection in LLMs

This repository is a curated, thesis-only version of the research workspace for:

**Self-Consistent Error Detection in Closed and Open Large Language Models: A Black-Box and White-Box Comparative Study**

It includes the manuscript source, analysis code, experiment configuration, and data artifacts required to inspect and reproduce the thesis results. Generated reports, temporary files, local environments, and non-essential outputs were removed.

## Scope

The repository covers three components:

1. Black-box measurement of self-consistent errors (CE) and inconsistent errors (IE) on TruthfulQA.
2. White-box probing experiments for CE detection.
3. Version-evolution analysis of CE behavior across model generations.

## Repository Structure

- `thesis.tex`, `references.bib`: thesis manuscript source.
- `src/`: core pipeline modules (inference, labeling, equivalence, reliability, storage).
- `scripts/`: runnable experiment and analysis scripts.
- `configs/`: model and run configurations.
- `data/`: curated input and result artifacts used by analyses in this thesis.
- `tests/`: automated checks for key pipeline components.
- `docs/`: project notes and reproducibility guides.

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Add API credentials as needed in `.env`.

### 3. Run the pipeline

```bash
python scripts/run_pipeline.py --help
python scripts/reeval_results.py --help
python scripts/analyze_results.py --help
```

### 4. Verify reproducibility from included artifacts

```bash
python scripts/verify_reproducibility_bundle.py
python scripts/validate_report_numbers.py
```

Detailed runbook: `docs/reproducibility.md`

### 5. Build the thesis PDF locally

```bash
latexmk -pdf thesis.tex
```

## Notes

- This repository intentionally excludes generated PDFs, figure images, LaTeX build artifacts, cache files, and temporary folders.
- Data files are retained in a form suitable for result verification and audit.

## License

See `LICENSE`.
