#!/usr/bin/env python3
"""
Compare self-consistent error overlap across all model pairs.

Outputs:
- model_error_overlap.txt (human-readable summary)
- model_error_overlap.csv (pairwise summary table)
- model_error_overlap_details.csv (question-level categories by pair)
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def short_model_name(model: str) -> str:
    return str(model).split("/")[-1]


def resolve_input_path(project_root: Path, provided: Optional[str]) -> Path:
    if provided:
        path = Path(provided)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    evaluated_dir = project_root / "data" / "results" / "evaluated"
    if evaluated_dir.exists():
        jsonl_candidates = sorted(
            evaluated_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if jsonl_candidates:
            return jsonl_candidates[0]

    parquet_fallback = project_root / "data" / "results" / "results.parquet"
    if parquet_fallback.exists():
        return parquet_fallback

    raise FileNotFoundError(
        "Could not find an input file automatically. "
        "Provide --input pointing to an analysis JSONL or parquet file."
    )


def load_results_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def hypergeom_overlap_p_value(observed_overlap: int, population_size: int, set_a: int, set_b: int) -> float:
    """
    One-sided hypergeometric tail:
    P[X >= observed_overlap], X ~ Hypergeom(N=population_size, K=set_a, n=set_b)
    """
    if population_size <= 0:
        return float("nan")
    if set_a < 0 or set_b < 0 or set_a > population_size or set_b > population_size:
        return float("nan")

    min_x = max(0, set_b - (population_size - set_a))
    max_x = min(set_a, set_b)
    if observed_overlap <= min_x:
        return 1.0
    if observed_overlap > max_x:
        return 0.0

    denominator = math.comb(population_size, set_b)
    if denominator == 0:
        return float("nan")

    p_tail = 0.0
    for x in range(observed_overlap, max_x + 1):
        numerator = math.comb(set_a, x) * math.comb(population_size - set_a, set_b - x)
        p_tail += numerator / denominator
    return min(1.0, max(0.0, p_tail))


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Compare self-consistent error overlap across model pairs.")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help=(
            "Input .jsonl/.parquet file. If omitted, the newest JSONL in "
            "data/results/evaluated is used; falls back to data/results/results.parquet."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.9, help="Error-label threshold to use (e.g. 0.9).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "data" / "results"),
        help="Directory for text and CSV outputs.",
    )
    parser.add_argument(
        "--max-detail-rows",
        type=int,
        default=0,
        help="Optional cap on question-level detail rows in details CSV (0 = no cap).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    input_path = resolve_input_path(project_root, args.input if args.input else None)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_txt = output_dir / "model_error_overlap.txt"
    output_csv = output_dir / "model_error_overlap.csv"
    output_details_csv = output_dir / "model_error_overlap_details.csv"

    lines: List[str] = []

    def log(text: str = "") -> None:
        print(text)
        lines.append(text)

    log("=" * 90)
    log("SELF-CONSISTENT ERROR OVERLAP BETWEEN MODELS (ALL PAIRS)")
    log("=" * 90)
    log(f"Input: {input_path}")

    df = load_results_frame(input_path)
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    threshold_key = f"error_label_{args.threshold:.1f}"
    if threshold_key not in df.columns:
        available = [col for col in df.columns if str(col).startswith("error_label_")]
        raise ValueError(
            f"Requested column '{threshold_key}' is missing. "
            f"Available error label columns: {sorted(available)}"
        )

    required = ["question_id", "model", threshold_key]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["question_id", "model"]).copy()
    df["question_id"] = df["question_id"].astype(str)
    df["model"] = df["model"].astype(str)
    df = df.drop_duplicates(subset=["question_id", "model"], keep="first")

    models = sorted(df["model"].unique().tolist())
    if len(models) < 2:
        raise ValueError("Need at least two models to compare.")

    log(f"Loaded {len(df)} rows across {len(models)} models")
    log(f"Threshold column: {threshold_key}")
    log()

    model_all_questions: Dict[str, Set[str]] = {}
    model_sc_errors: Dict[str, Set[str]] = {}
    for model in models:
        sub = df[df["model"] == model]
        qids = set(sub["question_id"].tolist())
        model_all_questions[model] = qids
        model_sc_errors[model] = set(sub[sub[threshold_key] == "self_consistent_error"]["question_id"].tolist())
        log(
            f"{short_model_name(model):<26} "
            f"questions={len(qids):>4}  self_consistent_errors={len(model_sc_errors[model]):>4}"
        )

    question_lookup_cols = [c for c in ["question", "ground_truth"] if c in df.columns]
    question_lookup = (
        df[["question_id", *question_lookup_cols]].drop_duplicates(subset=["question_id"], keep="first")
        if question_lookup_cols
        else pd.DataFrame({"question_id": sorted(df["question_id"].unique().tolist())})
    )
    question_lookup = question_lookup.set_index("question_id", drop=False)

    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    log()
    log("PAIRWISE SUMMARY")
    log("-" * 90)
    header = (
        f"{'Model A':<24} {'Model B':<24} {'CommonQ':>7} {'A_SC':>6} {'B_SC':>6} "
        f"{'Overlap':>8} {'Jaccard':>8} {'p(>=k)':>10}"
    )
    log(header)
    log("-" * 90)

    for model_a, model_b in itertools.combinations(models, 2):
        common_q = model_all_questions[model_a] & model_all_questions[model_b]
        errors_a = model_sc_errors[model_a] & common_q
        errors_b = model_sc_errors[model_b] & common_q
        overlap = errors_a & errors_b
        only_a = errors_a - errors_b
        only_b = errors_b - errors_a

        n_common = len(common_q)
        n_a = len(errors_a)
        n_b = len(errors_b)
        n_overlap = len(overlap)
        union = len(errors_a | errors_b)
        jaccard = (n_overlap / union) if union > 0 else float("nan")
        expected = (n_a * n_b / n_common) if n_common > 0 else float("nan")
        p_value = hypergeom_overlap_p_value(n_overlap, n_common, n_a, n_b)

        summary_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "model_a_short": short_model_name(model_a),
                "model_b_short": short_model_name(model_b),
                "n_common_questions": n_common,
                "n_sc_errors_model_a": n_a,
                "n_sc_errors_model_b": n_b,
                "n_overlap": n_overlap,
                "n_only_model_a": len(only_a),
                "n_only_model_b": len(only_b),
                "overlap_rate_given_a": (n_overlap / n_a) if n_a else float("nan"),
                "overlap_rate_given_b": (n_overlap / n_b) if n_b else float("nan"),
                "jaccard": jaccard,
                "expected_overlap_random": expected,
                "hypergeom_p_greater_equal_overlap": p_value,
                "threshold_column": threshold_key,
                "input_file": str(input_path),
            }
        )

        log(
            f"{short_model_name(model_a):<24} {short_model_name(model_b):<24} "
            f"{n_common:>7} {n_a:>6} {n_b:>6} {n_overlap:>8} {jaccard:>8.3f} {p_value:>10.3g}"
        )

        for qid in sorted(overlap):
            row = {"pair_model_a": model_a, "pair_model_b": model_b, "question_id": qid, "category": "both"}
            if qid in question_lookup.index:
                for col in question_lookup_cols:
                    row[col] = question_lookup.at[qid, col]
            detail_rows.append(row)

        for qid in sorted(only_a):
            row = {
                "pair_model_a": model_a,
                "pair_model_b": model_b,
                "question_id": qid,
                "category": f"only_{short_model_name(model_a)}",
            }
            if qid in question_lookup.index:
                for col in question_lookup_cols:
                    row[col] = question_lookup.at[qid, col]
            detail_rows.append(row)

        for qid in sorted(only_b):
            row = {
                "pair_model_a": model_a,
                "pair_model_b": model_b,
                "question_id": qid,
                "category": f"only_{short_model_name(model_b)}",
            }
            if qid in question_lookup.index:
                for col in question_lookup_cols:
                    row[col] = question_lookup.at[qid, col]
            detail_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["hypergeom_p_greater_equal_overlap", "n_overlap"],
        ascending=[True, False],
    )

    details_df = pd.DataFrame(detail_rows)
    if args.max_detail_rows > 0 and len(details_df) > args.max_detail_rows:
        details_df = details_df.head(args.max_detail_rows).copy()

    summary_df.to_csv(output_csv, index=False)
    details_df.to_csv(output_details_csv, index=False)
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log("-" * 90)
    log(f"Saved summary CSV: {output_csv}")
    log(f"Saved details CSV: {output_details_csv}")
    log(f"Saved text report: {output_txt}")

    if not summary_df.empty:
        top = summary_df.iloc[0]
        log()
        log(
            "Top overlap pair by smallest p-value: "
            f"{short_model_name(str(top['model_a']))} vs {short_model_name(str(top['model_b']))}, "
            f"overlap={int(top['n_overlap'])}, p={float(top['hypergeom_p_greater_equal_overlap']):.3g}"
        )


if __name__ == "__main__":
    main()
