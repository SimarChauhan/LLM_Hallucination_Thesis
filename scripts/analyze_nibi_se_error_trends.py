#!/usr/bin/env python3
"""
Analyze SE (self-consistent) error trends using:
1) Historical evaluated version-evolution outputs already in the repo.
2) Nibi whitebox sync artifacts (run reports + test scores).

Outputs:
- historical_se_model_summary.csv
- nibi_se_sync_summary.csv
- nibi_duplicate_consistency_checks.csv
- figures/*.png
- se_error_trend_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_HISTORICAL_DIR = Path("data/results/evaluated")
DEFAULT_NIBI_DIR = Path("data/results/whitebox")
DEFAULT_LEGACY_NIBI_DIR = Path("downloads/nibi_wb_probe/wb_probe_out")
DEFAULT_OUTPUT_ROOT = Path("data/results/analysis/se_error_trends")
DEFAULT_FALLBACK_INPUT = Path(
    "data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SE error trends from Nibi + existing data.")
    parser.add_argument("--historical-dir", type=Path, default=DEFAULT_HISTORICAL_DIR)
    parser.add_argument("--nibi-dir", type=Path, default=DEFAULT_NIBI_DIR)
    parser.add_argument("--legacy-nibi-dir", type=Path, default=DEFAULT_LEGACY_NIBI_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--historical-threshold", type=str, default="0.9", choices=["1.0", "0.9", "0.8", "0.7"])
    parser.add_argument("--nibi-threshold", type=float, default=1.0)
    return parser.parse_args()


def _short_model_name(name: str) -> str:
    cleaned = str(name)
    for token in [" (OpenRouter)", " (Anthropic)", " (OpenAI)", " (Groq)", " (DeepSeek)", " (xAI)"]:
        cleaned = cleaned.replace(token, "")
    return cleaned


def _se_binomial(p: float, n: int) -> float:
    if n <= 0 or not np.isfinite(p):
        return float("nan")
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def _resolve_input_path(raw_path: Optional[str]) -> Optional[Path]:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    marker = "data/results/"
    text = str(raw_path)
    if marker in text:
        rel = Path(text[text.index(marker) :])
        if rel.exists():
            return rel
    if DEFAULT_FALLBACK_INPUT.exists():
        return DEFAULT_FALLBACK_INPUT
    return None


def _historical_input_files(historical_dir: Path) -> List[Path]:
    files = sorted(historical_dir.glob("run_*/*.jsonl"))
    selected: List[Path] = []
    for path in files:
        name = path.name.lower()
        if "version_evolution" not in name and "eval" not in name:
            continue
        selected.append(path)
    return selected


def load_historical_summary(historical_dir: Path, threshold: str) -> pd.DataFrame:
    error_col = f"error_label_{threshold}"
    rows: List[pd.DataFrame] = []

    for path in _historical_input_files(historical_dir):
        try:
            frame = pd.read_json(path, lines=True)
        except ValueError:
            continue
        if frame.empty or error_col not in frame.columns:
            continue
        if "model" not in frame.columns or "question_id" not in frame.columns:
            continue

        part = frame.copy()
        part["source_file"] = str(path)
        part["run_date"] = pd.to_datetime(part.get("run_date"), errors="coerce")
        part["model_release_date"] = pd.to_datetime(part.get("model_release_date"), errors="coerce")
        part["model_version_index"] = pd.to_numeric(part.get("model_version_index"), errors="coerce")
        part["model_track"] = part.get("model_track")
        part["model_family"] = part.get("model_family")
        part["is_se_error"] = part[error_col].astype(str).eq("self_consistent_error")
        rows.append(part)

    if not rows:
        return pd.DataFrame()

    joined = pd.concat(rows, ignore_index=True)
    group_cols = [
        "source_file",
        "run_date",
        "model_track",
        "model_family",
        "model",
        "model_release_date",
        "model_version_index",
    ]
    summary = (
        joined.groupby(group_cols, dropna=False)
        .agg(
            n_rows=("question_id", "size"),
            n_questions=("question_id", "nunique"),
            se_error_rate=("is_se_error", "mean"),
        )
        .reset_index()
    )
    summary["se_error_rate_pct"] = 100.0 * summary["se_error_rate"]
    summary["se_error_rate_se"] = summary.apply(
        lambda r: _se_binomial(float(r["se_error_rate"]), int(r["n_rows"])),
        axis=1,
    )
    summary["model_short"] = summary["model"].map(_short_model_name)
    summary = summary.sort_values(
        by=["model_track", "model_release_date", "model_version_index", "model"],
        na_position="last",
    ).reset_index(drop=True)
    return summary


@dataclass
class InputCoverage:
    n_questions: int
    n_se_errors: int


def load_nibi_summary(
    nibi_dir: Path, nibi_threshold: float, legacy_nibi_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sync_pattern = "nibi_sync_*/**/wb_cross_model_probe_emnlp2025_run_report.json"
    report_paths = list(sorted(nibi_dir.glob(sync_pattern)))
    if legacy_nibi_dir and legacy_nibi_dir.exists():
        report_paths.extend(sorted(legacy_nibi_dir.glob("**/wb_cross_model_probe_emnlp2025_run_report.json")))

    legacy_default_date = None
    if legacy_nibi_dir and legacy_nibi_dir.exists():
        legacy_summary = legacy_nibi_dir.parent / "summaries" / "wb_probe_combined_metrics_summary.csv"
        if legacy_summary.exists():
            legacy_default_date = pd.to_datetime(
                datetime.fromtimestamp(legacy_summary.stat().st_mtime, tz=timezone.utc).date()
            )

    if not report_paths:
        return pd.DataFrame(), pd.DataFrame()

    eval_cache: Dict[str, pd.DataFrame] = {}
    coverage_cache: Dict[Tuple[str, str], InputCoverage] = {}
    rows: List[Dict[str, object]] = []

    for report_path in report_paths:
        match = re.search(r"nibi_sync_(\d{4}-\d{2}-\d{2})", str(report_path))
        if match:
            sync_date = pd.to_datetime(match.group(1))
            source_tag = "nibi_sync"
        else:
            if legacy_default_date is not None:
                sync_date = legacy_default_date
            else:
                sync_date = pd.to_datetime(datetime.fromtimestamp(report_path.stat().st_mtime, tz=timezone.utc).date())
            source_tag = "legacy_nibi_download"

        with report_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        target_model = str(payload.get("target_model_name", ""))
        subset_counts = payload.get("subset_counts", {}) or {}
        ce_counts = subset_counts.get("ce", {}) or {}
        ie_counts = subset_counts.get("ie", {}) or {}
        ce_neg_error = ce_counts.get("negative_error")
        ce_pos_correct = ce_counts.get("positive_correct")
        ie_neg_error = ie_counts.get("negative_error")
        ie_pos_correct = ie_counts.get("positive_correct")

        input_path = _resolve_input_path(payload.get("input"))
        input_path_str = str(input_path) if input_path else None

        n_questions = None
        n_se_errors_from_input = None
        if input_path_str:
            if input_path_str not in eval_cache:
                eval_cache[input_path_str] = pd.read_json(input_path_str, lines=True)
            eval_frame = eval_cache[input_path_str]
            key = (input_path_str, target_model)
            if key not in coverage_cache:
                model_frame = eval_frame[eval_frame["model"].astype(str) == target_model].copy()
                model_questions = int(model_frame["question_id"].nunique())
                label_col = f"error_label_{nibi_threshold:.1f}"
                if label_col in model_frame.columns:
                    se_count = int(model_frame[label_col].astype(str).eq("self_consistent_error").sum())
                else:
                    se_count = -1
                coverage_cache[key] = InputCoverage(n_questions=model_questions, n_se_errors=se_count)
            coverage = coverage_cache[key]
            n_questions = coverage.n_questions
            n_se_errors_from_input = coverage.n_se_errors

        test_scores_path = report_path.with_name("wb_cross_model_probe_emnlp2025_test_scores.csv")
        test_error_rate_all = float("nan")
        test_error_rate_ce = float("nan")
        test_error_rate_ie = float("nan")
        n_test_rows = 0
        if test_scores_path.exists():
            score_frame = pd.read_csv(test_scores_path)
            n_test_rows = int(len(score_frame))
            if n_test_rows > 0 and "y_true" in score_frame.columns:
                y = pd.to_numeric(score_frame["y_true"], errors="coerce")
                test_error_rate_all = float(1.0 - y.mean())
                if "subset" in score_frame.columns:
                    ce_subset = score_frame[score_frame["subset"] == "ce"]
                    ie_subset = score_frame[score_frame["subset"] == "ie"]
                    if not ce_subset.empty:
                        test_error_rate_ce = float(1.0 - pd.to_numeric(ce_subset["y_true"], errors="coerce").mean())
                    if not ie_subset.empty:
                        test_error_rate_ie = float(1.0 - pd.to_numeric(ie_subset["y_true"], errors="coerce").mean())

        se_error_rate = float("nan")
        if n_questions and n_questions > 0 and ce_neg_error is not None:
            se_error_rate = float(ce_neg_error) / float(n_questions)

        rows.append(
            {
                "sync_date": sync_date,
                "target_model_name": target_model,
                "target_model_short": _short_model_name(target_model),
                "run_report_path": str(report_path),
                "source_tag": source_tag,
                "input_path_resolved": input_path_str,
                "ce_threshold": payload.get("ce_threshold"),
                "subset_mode": payload.get("subset_mode"),
                "ce_negative_error": ce_neg_error,
                "ce_positive_correct": ce_pos_correct,
                "ie_negative_error": ie_neg_error,
                "ie_positive_correct": ie_pos_correct,
                "n_questions_total": n_questions,
                "n_se_errors_from_input": n_se_errors_from_input,
                "se_error_rate_from_ce_count": se_error_rate,
                "se_error_rate_from_ce_count_pct": 100.0 * se_error_rate if np.isfinite(se_error_rate) else float("nan"),
                "se_error_rate_from_ce_count_se": _se_binomial(se_error_rate, int(n_questions))
                if np.isfinite(se_error_rate) and n_questions
                else float("nan"),
                "test_rows": n_test_rows,
                "test_error_rate_all": test_error_rate_all,
                "test_error_rate_ce_subset": test_error_rate_ce,
                "test_error_rate_ie_subset": test_error_rate_ie,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    check_cols = ["ce_negative_error", "ce_positive_correct", "ie_negative_error", "ie_positive_correct"]
    consistency = (
        frame.groupby(["sync_date", "target_model_name"], dropna=False)[check_cols]
        .nunique(dropna=False)
        .reset_index()
    )
    consistency["is_consistent_across_duplicates"] = consistency[check_cols].max(axis=1).eq(1)

    dedup = (
        frame.sort_values("run_report_path")
        .drop_duplicates(subset=["sync_date", "target_model_name"], keep="first")
        .reset_index(drop=True)
    )
    dedup = dedup.sort_values(["target_model_name", "sync_date"]).reset_index(drop=True)
    return dedup, consistency


def plot_historical_timeline(historical: pd.DataFrame, output_path: Path) -> None:
    if historical.empty:
        return

    subset = historical.dropna(subset=["model_release_date"]).copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for track, track_df in subset.groupby("model_track", dropna=False):
        if track_df.empty:
            continue
        ordered = track_df.sort_values(["model_release_date", "model_version_index", "model"])
        label = str(track) if pd.notna(track) else "unknown_track"
        ax.plot(
            ordered["model_release_date"],
            ordered["se_error_rate_pct"],
            marker="o",
            linewidth=2,
            label=label,
            alpha=0.9,
        )

    ax.set_title("Historical SE Error Rate Over Model Release Timeline")
    ax.set_xlabel("Model release date")
    ax.set_ylabel("SE error rate (%)")
    ax.grid(alpha=0.25)
    ax.legend(title="Model track", loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_nibi_sync_timeline(nibi: pd.DataFrame, output_path: Path) -> None:
    if nibi.empty:
        return
    subset = nibi.dropna(subset=["sync_date", "se_error_rate_from_ce_count_pct"]).copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for model, model_df in subset.groupby("target_model_short", dropna=False):
        ordered = model_df.sort_values("sync_date")
        ax.plot(
            ordered["sync_date"],
            ordered["se_error_rate_from_ce_count_pct"],
            marker="o",
            linewidth=2,
            label=str(model),
            alpha=0.9,
        )

    ax.set_title("Nibi Sync: SE Error Rate Over Collection Date")
    ax.set_xlabel("Nibi sync date")
    ax.set_ylabel("SE error rate (%)")
    ax.grid(alpha=0.25)
    ax.legend(title="Target model", loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_nibi_delta_bar(nibi: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    if nibi.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for model, model_df in nibi.groupby("target_model_short", dropna=False):
        sub = model_df.dropna(subset=["sync_date", "se_error_rate_from_ce_count_pct"]).sort_values("sync_date")
        if len(sub) < 2:
            continue
        first = sub.iloc[0]
        last = sub.iloc[-1]
        rows.append(
            {
                "target_model_short": model,
                "first_date": first["sync_date"],
                "last_date": last["sync_date"],
                "delta_pp": float(last["se_error_rate_from_ce_count_pct"] - first["se_error_rate_from_ce_count_pct"]),
                "n_dates": int(sub["sync_date"].nunique()),
            }
        )
    delta = pd.DataFrame(rows)
    if delta.empty:
        return delta
    delta = delta.sort_values("delta_pp", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(delta["target_model_short"], delta["delta_pp"], color="#1f77b4", alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title("Nibi SE Error Change (First Available Date -> Latest Date)")
    ax.set_xlabel("Change in SE error rate (percentage points)")
    ax.set_ylabel("Target model")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return delta


def _track_slopes(historical: pd.DataFrame) -> pd.DataFrame:
    if historical.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for track, track_df in historical.groupby("model_track", dropna=False):
        sub = track_df.dropna(subset=["model_version_index", "se_error_rate"]).copy()
        if sub["model_version_index"].nunique() < 2:
            continue
        x = sub["model_version_index"].to_numpy(dtype=float)
        y = sub["se_error_rate"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        rows.append(
            {
                "model_track": track,
                "n_points": int(len(sub)),
                "slope_per_version": float(slope),
                "slope_per_version_pp": 100.0 * float(slope),
                "intercept": float(intercept),
            }
        )
    return pd.DataFrame(rows).sort_values("slope_per_version_pp", ascending=False).reset_index(drop=True)


def write_report(
    output_dir: Path,
    historical: pd.DataFrame,
    nibi: pd.DataFrame,
    consistency: pd.DataFrame,
    delta: pd.DataFrame,
    track_slopes: pd.DataFrame,
    historical_threshold: str,
    nibi_threshold: float,
) -> None:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report_path = output_dir / "se_error_trend_report.md"

    historical_mean = float(historical["se_error_rate"].mean()) if not historical.empty else float("nan")
    nibi_mean = float(nibi["se_error_rate_from_ce_count"].mean()) if not nibi.empty else float("nan")
    max_abs_delta = float(delta["delta_pp"].abs().max()) if not delta.empty else float("nan")
    consistency_pass_rate = (
        float(consistency["is_consistent_across_duplicates"].mean()) if not consistency.empty else float("nan")
    )

    lines: List[str] = []
    lines.append("# SE Error Trend Analysis (Nibi + Existing Data)")
    lines.append("")
    lines.append(f"- Generated: `{now_utc}`")
    lines.append(f"- Historical SE label threshold: `{historical_threshold}`")
    lines.append(f"- Nibi CE threshold used for SE counts: `{nibi_threshold:.1f}`")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"- Historical mean SE error rate (threshold {historical_threshold}): `{100.0 * historical_mean:.2f}%`."
        if np.isfinite(historical_mean)
        else "- Historical mean SE error rate: `N/A`."
    )
    lines.append(
        f"- Nibi mean SE error rate (from CE counts): `{100.0 * nibi_mean:.2f}%`."
        if np.isfinite(nibi_mean)
        else "- Nibi mean SE error rate: `N/A`."
    )
    lines.append(
        f"- Maximum model-level Nibi change across sync dates: `{max_abs_delta:.3f}` percentage points."
        if np.isfinite(max_abs_delta)
        else "- Nibi date-over-date change could not be computed."
    )
    lines.append(
        f"- Duplicate-run consistency check pass rate: `{100.0 * consistency_pass_rate:.1f}%`."
        if np.isfinite(consistency_pass_rate)
        else "- Duplicate-run consistency checks unavailable."
    )
    lines.append("")
    lines.append("## Main Findings")
    lines.append("")
    if not delta.empty:
        unchanged = int((delta["delta_pp"].abs() < 1e-9).sum())
        lines.append(
            f"- Across `{len(delta)}` target models with at least two Nibi snapshots, `{unchanged}` models show exactly zero SE-rate drift."
        )
    if not track_slopes.empty:
        top = track_slopes.iloc[0]
        lines.append(
            f"- Largest historical slope: `{top['model_track']}` at `{top['slope_per_version_pp']:.2f}` pp/version "
            f"(linear fit on available version-index points)."
        )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `historical_se_model_summary.csv`")
    lines.append("- `nibi_se_sync_summary.csv`")
    lines.append("- `nibi_duplicate_consistency_checks.csv`")
    lines.append("- `nibi_se_delta_pp.csv`")
    lines.append("- `track_slope_summary.csv`")
    lines.append("- `figures/01_historical_release_timeline.png`")
    lines.append("- `figures/02_nibi_sync_timeline.png`")
    lines.append("- `figures/03_nibi_delta_pp.png`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Historical and Nibi series are complementary but not identical: "
        "historical uses evaluated row labels directly, while Nibi rates are reconstructed from CE subset counts."
    )
    unique_dates = sorted(pd.to_datetime(nibi["sync_date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique().tolist())
    if unique_dates:
        lines.append(
            "- Nibi snapshot dates detected in this run: `"
            + "`, `".join(unique_dates)
            + "`."
        )
    lines.append("- If more Nibi sync drops are added later, rerun this script to extend the timeline.")

    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"run_{timestamp}")
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    historical = load_historical_summary(args.historical_dir, args.historical_threshold)
    nibi, consistency = load_nibi_summary(
        args.nibi_dir,
        args.nibi_threshold,
        legacy_nibi_dir=args.legacy_nibi_dir,
    )
    slopes = _track_slopes(historical)

    plot_historical_timeline(historical, figures_dir / "01_historical_release_timeline.png")
    plot_nibi_sync_timeline(nibi, figures_dir / "02_nibi_sync_timeline.png")
    delta = plot_nibi_delta_bar(nibi, figures_dir / "03_nibi_delta_pp.png")

    historical.to_csv(output_dir / "historical_se_model_summary.csv", index=False)
    nibi.to_csv(output_dir / "nibi_se_sync_summary.csv", index=False)
    consistency.to_csv(output_dir / "nibi_duplicate_consistency_checks.csv", index=False)
    delta.to_csv(output_dir / "nibi_se_delta_pp.csv", index=False)
    slopes.to_csv(output_dir / "track_slope_summary.csv", index=False)

    write_report(
        output_dir=output_dir,
        historical=historical,
        nibi=nibi,
        consistency=consistency,
        delta=delta,
        track_slopes=slopes,
        historical_threshold=args.historical_threshold,
        nibi_threshold=args.nibi_threshold,
    )

    print(f"[done] Output directory: {output_dir}")
    print(f"[done] Historical rows: {len(historical)}")
    print(f"[done] Nibi rows: {len(nibi)}")
    if not delta.empty:
        print(f"[done] Max abs Nibi delta (pp): {delta['delta_pp'].abs().max():.6f}")


if __name__ == "__main__":
    main()
