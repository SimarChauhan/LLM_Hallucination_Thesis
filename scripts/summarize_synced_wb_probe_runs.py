#!/usr/bin/env python3
"""
Aggregate synced white-box probe run summaries into local leaderboards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SUMMARY_NAME = "wb_cross_model_probe_emnlp2025_metrics_summary.csv"
RUN_REPORT_NAME = "wb_cross_model_probe_emnlp2025_run_report.json"


def _load_run_report(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / RUN_REPORT_NAME
    if not report_path.exists():
        return {}
    try:
        with report_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _discover_summary_files(root: Path) -> list[Path]:
    all_files = sorted(root.rglob(SUMMARY_NAME))
    if not all_files:
        return []

    scoped: list[Path] = []
    for path in all_files:
        rel_parts = set(path.relative_to(root).parts)
        if "wb_probe_out" in rel_parts or "wb_probe_out_fallback" in rel_parts:
            scoped.append(path)

    # If run-scoped files exist, ignore ad-hoc top-level copies.
    if scoped:
        return sorted(scoped)
    return all_files


def _build_combined_frame(files: list[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for summary_path in files:
        run_dir = summary_path.parent
        run_report = _load_run_report(run_dir)

        try:
            frame = pd.read_csv(summary_path)
        except Exception:
            continue

        frame["run_dir"] = str(run_dir)
        frame["run_name"] = run_dir.name
        frame["summary_file"] = str(summary_path)

        frame["target_model_name"] = run_report.get("target_model_name")
        frame["response_model_path_or_hf_id"] = run_report.get("response_model_path_or_hf_id")
        frame["verifier_model_path_or_hf_id"] = run_report.get("verifier_model_path_or_hf_id")
        frame["subset_mode"] = run_report.get("subset_mode")
        frame["probe_seeds"] = ",".join(str(s) for s in run_report.get("probe_seeds", [])) if run_report else None
        frame["lambda_step"] = run_report.get("lambda_step")
        frame["ce_threshold"] = run_report.get("ce_threshold")
        frame["input"] = run_report.get("input")
        frame["elapsed_seconds"] = run_report.get("elapsed_seconds")

        rows.append(frame)

    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    return combined


def _write_outputs(combined: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = output_dir / "wb_probe_combined_metrics_summary.csv"
    combined.to_csv(combined_path, index=False)

    rank_cols = ["run_name", "subset", "auroc_mean", "prauc_mean", "accuracy_at_0_5_mean"]
    best_by_run_subset = (
        combined.sort_values(rank_cols, ascending=[True, True, False, False, False])
        .drop_duplicates(subset=["run_name", "subset"], keep="first")
        .reset_index(drop=True)
    )
    best_by_run_subset.to_csv(output_dir / "wb_probe_best_by_run_subset.csv", index=False)

    best_overall_by_subset = (
        combined.sort_values(["subset", "auroc_mean", "prauc_mean"], ascending=[True, False, False])
        .drop_duplicates(subset=["subset"], keep="first")
        .reset_index(drop=True)
    )
    best_overall_by_subset.to_csv(output_dir / "wb_probe_best_overall_by_subset.csv", index=False)

    run_inventory_cols = [
        "run_name",
        "run_dir",
        "target_model_name",
        "response_model_path_or_hf_id",
        "verifier_model_path_or_hf_id",
        "subset_mode",
        "probe_seeds",
        "lambda_step",
        "ce_threshold",
        "elapsed_seconds",
        "input",
    ]
    run_inventory = combined[run_inventory_cols].drop_duplicates().sort_values("run_name").reset_index(drop=True)
    run_inventory.to_csv(output_dir / "wb_probe_run_inventory.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize synced white-box probe artifacts.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        required=True,
        help="Root folder containing synced artifacts (e.g., downloads/nibi_wb_probe).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where aggregated summary CSVs will be written.",
    )
    args = parser.parse_args()

    root = args.artifacts_root
    if not root.exists():
        raise SystemExit(f"Artifacts root does not exist: {root}")

    files = _discover_summary_files(root)
    if not files:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(args.output_dir / "wb_probe_combined_metrics_summary.csv", index=False)
        print("No metrics summary files found.")
        return 0

    combined = _build_combined_frame(files)
    if combined.empty:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(args.output_dir / "wb_probe_combined_metrics_summary.csv", index=False)
        print("Summary files were found but none could be parsed.")
        return 0

    _write_outputs(combined, args.output_dir)

    n_runs = int(combined["run_name"].nunique())
    n_rows = int(len(combined))
    print(f"Found {len(files)} summary files.")
    print(f"Aggregated {n_rows} rows across {n_runs} runs.")
    print(f"Wrote summaries to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
