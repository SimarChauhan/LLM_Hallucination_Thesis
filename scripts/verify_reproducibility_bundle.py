#!/usr/bin/env python3
"""
Quick integrity audit for the curated thesis reproducibility bundle.

This script validates that the key files and headline numbers needed to
reproduce thesis analyses are present and internally consistent.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _approx_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def _check_equal(name: str, got: object, expected: object) -> CheckResult:
    ok = got == expected
    return CheckResult(name=name, passed=ok, detail=f"got={got}, expected={expected}")


def _check_float(name: str, got: float, expected: float, tol: float) -> CheckResult:
    ok = _approx_equal(got, expected, tol=tol)
    return CheckResult(
        name=name,
        passed=ok,
        detail=f"got={got:.6f}, expected={expected:.6f}, tol={tol}",
    )


def _must_exist(path: Path) -> CheckResult:
    return CheckResult(
        name=f"exists: {path}",
        passed=path.exists(),
        detail="present" if path.exists() else "missing",
    )


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def run_checks(repo_root: Path) -> list[CheckResult]:
    checks: list[CheckResult] = []

    main_jsonl = repo_root / "data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl"
    overlap_csv = repo_root / "data/results/analysis/cross_model_ce_overlap_t1p0_20260327/cross_model_ce_overlap_semantic_nlihybrid_t1p0.csv"
    version_summary = repo_root / "data/results/analysis/version_evolution_equiv_only_20260319/model_summary_t0p9.csv"
    version_manifest = repo_root / "data/results/analysis/version_evolution_equiv_only_20260319/analysis_manifest.json"
    whitebox_root = repo_root / "data/results/whitebox"

    required_paths = [
        main_jsonl,
        overlap_csv,
        version_summary,
        version_manifest,
        repo_root / "scripts/validate_report_numbers.py",
        repo_root / "scripts/compute_shared_ce_analysis.py",
        repo_root / "scripts/analyze_version_evolution.py",
        repo_root / "scripts/rebuild_version_evolution_package.py",
        repo_root / "scripts/summarize_synced_wb_probe_runs.py",
    ]
    checks.extend(_must_exist(path) for path in required_paths)

    if not all(c.passed for c in checks):
        return checks

    main_df = pd.read_json(main_jsonl, lines=True)
    checks.append(_check_equal("main rows", len(main_df), 4842))
    checks.append(_check_equal("main unique questions", int(main_df["question_id"].nunique()), 807))
    checks.append(_check_equal("main unique models", int(main_df["model"].nunique()), 6))
    checks.append(
        _check_equal(
            "main duplicate (question_id, model)",
            int(main_df.duplicated(subset=["question_id", "model"]).sum()),
            0,
        )
    )

    overlap_df = pd.read_csv(overlap_csv)
    total_overlap = int(overlap_df["both_ce_overlap"].sum())
    total_same = int(overlap_df["same_wrong_answer"].sum())
    total_unclear = int(overlap_df["unclear_equivalence"].sum())
    same_pct = round(100.0 * total_same / total_overlap, 1) if total_overlap else 0.0
    jaccard_min = float(overlap_df["jaccard"].min())
    jaccard_max = float(overlap_df["jaccard"].max())

    checks.append(_check_equal("cross-model pair rows", len(overlap_df), 15))
    checks.append(_check_equal("cross-model total overlap", total_overlap, 720))
    checks.append(_check_equal("cross-model total same-wrong", total_same, 529))
    checks.append(_check_equal("cross-model total unclear", total_unclear, 35))
    checks.append(_check_float("cross-model same-wrong pct", same_pct, 73.5, tol=1e-9))
    checks.append(_check_float("cross-model jaccard min", round(jaccard_min, 3), 0.219, tol=1e-9))
    checks.append(_check_float("cross-model jaccard max", round(jaccard_max, 3), 0.360, tol=1e-9))

    v_df = pd.read_csv(version_summary)
    checks.append(_check_equal("version summary models", len(v_df), 12))
    checks.append(_check_equal("version summary unique models", int(v_df["model"].nunique()), 12))
    checks.append(_check_equal("version summary total rows", int(v_df["n_rows"].sum()), 9684))

    run_files = sorted((repo_root / "data/results/evaluated").glob("run_*/*.jsonl"))
    checks.append(_check_equal("version run file count", len(run_files), 3))
    for run_file in run_files:
        checks.append(_check_equal(f"rows in {run_file.name}", _line_count(run_file), 2421))

    wb_reports = list(whitebox_root.glob("**/wb_cross_model_probe_emnlp2025_run_report.json"))
    wb_summaries = list(whitebox_root.glob("**/wb_cross_model_probe_emnlp2025_metrics_summary.csv"))
    checks.append(_check_equal("whitebox run reports", len(wb_reports), 18))
    checks.append(_check_equal("whitebox metrics summaries", len(wb_summaries), 18))

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify curated thesis reproducibility bundle.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to parent of scripts/).",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    checks = run_checks(repo_root)

    passed = 0
    failed = 0
    for check in checks:
        mark = "PASS" if check.passed else "FAIL"
        print(f"[{mark}] {check.name}: {check.detail}")
        if check.passed:
            passed += 1
        else:
            failed += 1

    print("")
    print(f"Total checks: {len(checks)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
