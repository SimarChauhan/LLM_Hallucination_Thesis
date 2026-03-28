#!/usr/bin/env python3
"""Generate a repository file index with practical "where-is-what" guidance."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "FILE_INDEX.md"

EXCLUDE_DIRS = {
    ".git",
    "venv",
    "__pycache__",
    ".cache",
    ".claude",
    ".cursor",
    ".mplconfig",
}

TOP_LEVEL_DESCRIPTIONS = {
    "archive": "Archived legacy/generated artifacts kept out of active workspace.",
    "analysis": "Scratch/working analysis workspace (non-core runtime).",
    "configs": "Versioned experiment configurations (YAML).",
    "data": "Calibrations, raw/evaluated results, and derived analysis outputs.",
    "docs": "Documentation and report sources/artifacts.",
    "downloads": "Synced artifacts from remote clusters (e.g., Nibi).",
    "output": "Locally generated report outputs (PDF and build files).",
    "progress_report": "Progress snapshots and notes.",
    "scripts": "Executable pipeline, analysis, SLURM, and utility scripts.",
    "src": "Core Python modules used by the pipeline.",
    "tests": "Unit and regression tests.",
    "tmp": "Temporary work products and transient build intermediates.",
}

KEY_DATA_PATHS = {
    "data/calibration": "Frozen calibration artifacts (e.g., hybrid thresholds).",
    "data/results/raw": "Phase 1 generation outputs (greedy + stochastic answers).",
    "data/results/evaluated": "Phase 2 judged/labelled outputs.",
    "data/results/analysis": "Post-processing tables/figures/reports.",
    "data/results/whitebox": "White-box probe outputs and sync snapshots.",
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def list_files(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted([p for p in path.iterdir() if p.is_file()], key=lambda p: p.name.lower())


def list_dirs(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted(
        [p for p in path.iterdir() if p.is_dir() and p.name not in EXCLUDE_DIRS],
        key=lambda p: p.name.lower(),
    )


def classify_script(name: str) -> str:
    if name.startswith("run_"):
        return "Run/Orchestration"
    if name.startswith("slurm_"):
        return "SLURM Job Script"
    if name.startswith("submit_"):
        return "Batch Submission"
    if name.startswith("sync_"):
        return "Sync/Transfer"
    if name.startswith("analyze_"):
        return "Analysis"
    if name.startswith("generate_"):
        return "Report/Figure Generation"
    if name.startswith("create_"):
        return "Dataset/Sample Builder"
    if name.startswith("calibrate_"):
        return "Calibration"
    if name.startswith("repair_") or name.startswith("backfill_"):
        return "Data Repair"
    if name.startswith("compare_") or name.startswith("compute_"):
        return "Comparison/Stats"
    if name.startswith("summarize_") or name.startswith("validate_"):
        return "Validation/Summary"
    return "Utility"


def classify_config(name: str) -> str:
    lname = name.lower()
    if "qwen" in lname:
        return "Qwen timeline/track"
    if "claude" in lname or "grok" in lname or "chatgpt" in lname:
        return "Closed-model timeline"
    if "full_timeline" in lname:
        return "Cross-family full timeline"
    return "General"


def classify_src(name: str) -> str:
    stem = Path(name).stem
    mapping = {
        "providers": "Model API/provider adapters",
        "truthfulqa": "TruthfulQA loading/parsing",
        "dataset": "General dataset loaders",
        "schemas": "Data models and serialization",
        "storage": "JSONL/parquet persistence",
        "correctness": "Correctness judging logic",
        "semantic": "Semantic equivalence logic",
        "semantic_entropy": "Semantic entropy calculations",
        "hybrid_judging": "Hybrid NLI+LLM judgment policy",
        "nli_judge": "NLI judge wrapper",
        "labeling": "Error label derivation",
        "contamination": "Prompt overlap/contamination checks",
        "inference": "Inference helper utilities",
        "annotation": "Annotation sample utilities",
        "reliability": "Reliability/diagnostics helpers",
    }
    return mapping.get(stem, "Core module")


def gather_top_level() -> tuple[list[Path], list[Path]]:
    files = sorted([p for p in ROOT.iterdir() if p.is_file()], key=lambda p: p.name.lower())
    dirs = sorted(
        [p for p in ROOT.iterdir() if p.is_dir() and p.name not in EXCLUDE_DIRS],
        key=lambda p: p.name.lower(),
    )
    return files, dirs


def render() -> str:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    top_files, top_dirs = gather_top_level()

    configs = list_files(ROOT / "configs")
    scripts = list_files(ROOT / "scripts")
    src_files = [p for p in list_files(ROOT / "src") if p.suffix == ".py"]
    tests = [p for p in list_files(ROOT / "tests") if p.suffix == ".py"]
    archive_dirs = list_dirs(ROOT / "archive")

    lines: list[str] = []
    lines.append("# File Index")
    lines.append("")
    lines.append(f"Generated: `{now}`")
    lines.append("")
    lines.append("This index is the single map of where things live in this repository.")
    lines.append("")

    lines.append("## Top-Level Directories")
    lines.append("")
    lines.append("| Path | Purpose |")
    lines.append("|---|---|")
    for d in top_dirs:
        desc = TOP_LEVEL_DESCRIPTIONS.get(d.name, "Project directory")
        lines.append(f"| `{rel(d)}` | {desc} |")
    lines.append("")

    lines.append("## Top-Level Files")
    lines.append("")
    for f in top_files:
        lines.append(f"- `{rel(f)}`")
    lines.append("")

    lines.append("## Data Layout (Authoritative)")
    lines.append("")
    for p, desc in KEY_DATA_PATHS.items():
        lines.append(f"- `{p}`: {desc}")
    lines.append("")

    lines.append("## Archive Layout")
    lines.append("")
    if archive_dirs:
        for d in archive_dirs:
            lines.append(f"- `{rel(d)}`")
            for child in list_dirs(d):
                lines.append(f"- `{rel(child)}`")
    else:
        lines.append("- _No archive directories yet._")
    lines.append("")

    lines.append("## Config Registry")
    lines.append("")
    lines.append("| Config | Type |")
    lines.append("|---|---|")
    for c in configs:
        lines.append(f"| `{rel(c)}` | {classify_config(c.name)} |")
    lines.append("")

    lines.append("## Script Registry")
    lines.append("")
    lines.append("| Script | Category |")
    lines.append("|---|---|")
    for s in scripts:
        lines.append(f"| `{rel(s)}` | {classify_script(s.name)} |")
    lines.append("")

    lines.append("## Source Module Registry")
    lines.append("")
    lines.append("| Module | Responsibility |")
    lines.append("|---|---|")
    for s in src_files:
        lines.append(f"| `{rel(s)}` | {classify_src(s.name)} |")
    lines.append("")

    lines.append("## Test Registry")
    lines.append("")
    for t in tests:
        lines.append(f"- `{rel(t)}`")
    lines.append("")

    lines.append("## High-Use Entry Points")
    lines.append("")
    lines.append("- `scripts/run_pipeline.py`: Phase 1 raw generation.")
    lines.append("- `scripts/reeval_results.py`: Phase 2 re-eval + labeling.")
    lines.append("- `scripts/run_version_evolution_study.sh`: staged evolution orchestrator.")
    lines.append("- `scripts/analyze_version_evolution.py`: pairwise/trend analysis.")
    lines.append("- `configs/version_evolution_qwen_frontier_plus_7b.yaml`: current Qwen study config.")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `venv/` is local environment state; not part of source organization.")
    lines.append("- `tmp/`, `output/`, and `downloads/` contain generated or synced artifacts.")
    lines.append("- For tracked experiment outputs, use `data/results/*` paths above.")
    lines.append("")

    lines.append("## Refresh This Index")
    lines.append("")
    lines.append("```bash")
    lines.append("./venv/bin/python scripts/generate_file_index.py")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    OUT.write_text(render(), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
