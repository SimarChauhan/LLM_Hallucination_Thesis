#!/usr/bin/env python3
"""
Analysis script for LLM Self-Consistent Error Measurement results.

High-rigor upgrades:
- Optional hard-fail on mixed protocol versions
- Excludes incomplete rows from primary headline metrics
- Optional bootstrap 95% confidence intervals
- Optional paired significance tests (exact McNemar)
- Reliability panel with fail-fast checks in strict mode
"""

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reliability import compute_ensemble_reliability
from src.storage import ResultStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return False


def _format_pct(numer: float, denom: float) -> str:
    if denom <= 0:
        return "N/A"
    return f"{100.0 * numer / denom:.1f}%"


def _bootstrap_proportion_ci(
    values: np.ndarray,
    num_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        v = float(values[0])
        return (v, v)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(num_bootstrap, values.size))
    samples = values[idx].mean(axis=1)
    return (
        float(np.quantile(samples, alpha / 2)),
        float(np.quantile(samples, 1 - alpha / 2)),
    )


def _exact_binomial_two_sided(k: int, n: int) -> float:
    """Two-sided exact p-value for Binomial(n, 0.5) with observed min-tail k."""
    if n <= 0:
        return 1.0
    tail_prob = 0.0
    for i in range(0, k + 1):
        tail_prob += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * tail_prob)


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Exact McNemar p-value from discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    return _exact_binomial_two_sided(min(b, c), n)


def _cohen_kappa(system_labels: List[str], human_labels: List[str]) -> Optional[float]:
    if len(system_labels) != len(human_labels) or len(system_labels) < 2:
        return None

    labels = sorted(set(system_labels) | set(human_labels))
    if not labels:
        return None

    n = len(system_labels)
    observed = sum(1 for s, h in zip(system_labels, human_labels) if s == h) / n

    p_sys: Dict[str, float] = {}
    p_hum: Dict[str, float] = {}
    for label in labels:
        p_sys[label] = sum(1 for s in system_labels if s == label) / n
        p_hum[label] = sum(1 for h in human_labels if h == label) / n

    expected = sum(p_sys[l] * p_hum[l] for l in labels)
    if abs(1.0 - expected) < 1e-12:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def load_data(results_dir: str, results_file: str = "results.jsonl") -> pd.DataFrame:
    """Load evaluated records into a DataFrame."""
    storage = ResultStorage(results_dir, results_file)
    df = storage.to_dataframe()

    if df.empty:
        logger.warning("No data found in results file")
        return df

    logger.info("Loaded %d records from %s/%s", len(df), results_dir, results_file)
    return df


def load_analysis_defaults(config_path: str) -> Dict[str, Any]:
    """Load optional analysis defaults from config.yaml."""
    try:
        import yaml
    except ImportError:
        return {}

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        return {}

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return dict(config.get("analysis", {}))


def validate_uniform_protocol(df: pd.DataFrame, require_uniform_protocol: bool) -> Optional[str]:
    """Return protocol version if uniform; raise in strict mode on mixed/missing values."""
    if "protocol_version" not in df.columns:
        if require_uniform_protocol:
            raise ValueError("Missing required field: protocol_version")
        return None

    non_null = df["protocol_version"].dropna().astype(str)
    if non_null.empty:
        if require_uniform_protocol:
            raise ValueError("protocol_version is required but all rows are null")
        return None

    values = sorted(set(non_null.tolist()))
    if require_uniform_protocol and len(values) != 1:
        raise ValueError(f"Mixed protocol_version values detected: {values}")

    return values[0] if len(values) == 1 else None


def primary_analysis_slice(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Exclude incomplete rows from primary headline metrics."""
    if "is_incomplete" not in df.columns:
        return df.copy(), 0

    incomplete_mask = df["is_incomplete"].map(_to_bool)
    excluded = int(incomplete_mask.sum())
    return df[~incomplete_mask].copy(), excluded


def table_1_error_breakdown(df: pd.DataFrame, with_ci: bool = False) -> pd.DataFrame:
    """Table 1: model-level error breakdown with optional 95% CIs."""
    if df.empty:
        return pd.DataFrame()

    results: List[Dict[str, Any]] = []
    for model in sorted(df["model"].dropna().unique()):
        model_df = df[df["model"] == model].copy()
        if model_df.empty:
            continue

        greedy_correct = model_df["greedy_correct"].map(_to_bool)
        incorrect_mask = ~greedy_correct
        incorrect_df = model_df[incorrect_mask]

        total = int(len(model_df))
        correct = int(greedy_correct.sum())
        incorrect = int(len(incorrect_df))

        label_col = "error_label_0.9"
        self_consistent = int((incorrect_df.get(label_col) == "self_consistent_error").sum()) if label_col in incorrect_df.columns else 0
        inconsistent = int((incorrect_df.get(label_col) == "inconsistent_error").sum()) if label_col in incorrect_df.columns else 0

        not_attempted = int((model_df.get("correctness_grade") == "NOT_ATTEMPTED").sum()) if "correctness_grade" in model_df.columns else 0
        unclear = int(model_df["correctness_unclear"].map(_to_bool).sum()) if "correctness_unclear" in model_df.columns else 0

        row: Dict[str, Any] = {
            "Model": model.split("/")[-1],
            "Total": total,
            "Correct": correct,
            "Incorrect": incorrect,
            "Accuracy": _format_pct(correct, total),
            "Not Attempted": not_attempted,
            "Judge Unclear": unclear,
            "Self-Consistent Errors": self_consistent,
            "Inconsistent Errors": inconsistent,
            "% Self-Consistent (of errors)": _format_pct(self_consistent, incorrect),
        }

        if with_ci:
            acc_values = greedy_correct.astype(int).to_numpy()
            acc_lo, acc_hi = _bootstrap_proportion_ci(acc_values)
            row["Accuracy 95% CI"] = f"[{100*acc_lo:.1f}, {100*acc_hi:.1f}]%"

            sc_values = (model_df.get(label_col) == "self_consistent_error").astype(int).to_numpy() if label_col in model_df.columns else np.array([])
            sc_lo, sc_hi = _bootstrap_proportion_ci(sc_values)
            row["SC Error Rate 95% CI"] = f"[{100*sc_lo:.1f}, {100*sc_hi:.1f}]%" if sc_values.size else "N/A"

        results.append(row)

    return pd.DataFrame(results)


def table_3_threshold_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Table 3: threshold sensitivity of self-consistent error rates."""
    if df.empty:
        return pd.DataFrame()

    incorrect_df = df[~df["greedy_correct"].map(_to_bool)]
    if incorrect_df.empty:
        return pd.DataFrame()

    thresholds = ["1.0", "0.9", "0.8", "0.7"]
    results = []
    for model in sorted(df["model"].dropna().unique()):
        model_incorrect = incorrect_df[incorrect_df["model"] == model]
        total_errors = len(model_incorrect)

        row: Dict[str, Any] = {"Model": model.split("/")[-1]}
        for threshold in thresholds:
            col = f"error_label_{threshold}"
            if col not in model_incorrect.columns or total_errors == 0:
                row[f"Threshold {threshold}"] = "N/A"
                continue
            sc = int((model_incorrect[col] == "self_consistent_error").sum())
            row[f"Threshold {threshold}"] = _format_pct(sc, total_errors)

        results.append(row)

    return pd.DataFrame(results)


def table_4_unclear_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Table 4: semantic judge unclear-rate summary."""
    if df.empty:
        return pd.DataFrame()

    required = {"equiv_total", "equiv_num_unclear", "equiv_num_same", "equiv_num_different"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    incorrect_df = df[~df["greedy_correct"].map(_to_bool)]
    if incorrect_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for model in sorted(df["model"].dropna().unique()):
        model_incorrect = incorrect_df[incorrect_df["model"] == model]
        if model_incorrect.empty:
            continue

        total_judgments = int(model_incorrect["equiv_total"].fillna(0).sum())
        unclear_judgments = int(model_incorrect["equiv_num_unclear"].fillna(0).sum())
        same_judgments = int(model_incorrect["equiv_num_same"].fillna(0).sum())
        different_judgments = int(model_incorrect["equiv_num_different"].fillna(0).sum())

        rows.append(
            {
                "Model": model.split("/")[-1],
                "Total Judgments": total_judgments,
                "Same": same_judgments,
                "Different": different_judgments,
                "Unclear": unclear_judgments,
                "Unclear Rate": _format_pct(unclear_judgments, total_judgments),
            }
        )

    return pd.DataFrame(rows)


def _human_label_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in [
        "human_correctness_grade",
        "human_grade",
        "human_label",
        "human_correct",
    ]:
        if candidate in df.columns:
            return candidate
    return None


def table_5_reliability_panel(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Table 5: reliability panel by model with strict fail-fast option."""
    if df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    missing_by_model: Dict[str, List[str]] = {}

    human_col = _human_label_column(df)

    for model in sorted(df["model"].dropna().unique()):
        model_df = df[df["model"] == model].copy()
        if model_df.empty:
            continue

        missing: List[str] = []

        alpha_val: Optional[float] = None
        pairwise_val: Optional[float] = None
        n_items = 0
        n_judges = 0

        if "correctness_judge_grades" in model_df.columns:
            grades = [
                g for g in model_df["correctness_judge_grades"].tolist()
                if isinstance(g, list) and len(g) >= 2
            ]
            if grades:
                rel = compute_ensemble_reliability(grades)
                alpha_val = float(rel["krippendorff_alpha"])
                pairwise_val = float(rel["pairwise_agreement"])
                n_items = int(rel["n_items"])
                n_judges = int(rel["n_raters"])

        if alpha_val is None and "inter_rater_alpha" in model_df.columns:
            alpha_series = model_df["inter_rater_alpha"].dropna()
            if not alpha_series.empty:
                alpha_val = float(alpha_series.mean())

        if alpha_val is None:
            missing.append("inter_rater_alpha")

        repeat_mean: Optional[float] = None
        repeat_n = 0
        if "judge_repeat_consistency" in model_df.columns:
            repeat_series = pd.to_numeric(model_df["judge_repeat_consistency"], errors="coerce").dropna()
            repeat_n = int(len(repeat_series))
            if repeat_n > 0:
                repeat_mean = float(repeat_series.mean())
        if repeat_mean is None:
            missing.append("judge_repeat_consistency")

        escalation_rate: Optional[float] = None
        if "escalated_to_human" in model_df.columns:
            escalation_rate = float(model_df["escalated_to_human"].map(_to_bool).mean())
        else:
            missing.append("escalated_to_human")

        kappa_val: Optional[float] = None
        if human_col and "correctness_grade" in model_df.columns:
            paired = model_df[[human_col, "correctness_grade"]].dropna()
            if len(paired) >= 2:
                sys_labels = paired["correctness_grade"].astype(str).tolist()
                hum_labels = paired[human_col].astype(str).tolist()
                kappa_val = _cohen_kappa(sys_labels, hum_labels)

        rows.append(
            {
                "Model": model.split("/")[-1],
                "Inter-rater Alpha": f"{alpha_val:.4f}" if alpha_val is not None else "MISSING",
                "Pairwise Agreement": f"{pairwise_val:.1%}" if pairwise_val is not None else "N/A",
                "Repeat Consistency": f"{repeat_mean:.3f}" if repeat_mean is not None else "MISSING",
                "Repeat N": repeat_n,
                "Escalation Rate": f"{escalation_rate:.1%}" if escalation_rate is not None else "MISSING",
                "Human-System Kappa": f"{kappa_val:.4f}" if kappa_val is not None else "N/A",
                "Judge Items": n_items,
                "Judges": n_judges,
            }
        )

        if missing:
            missing_by_model[model] = missing

    if strict and missing_by_model:
        parts = [f"{k}: {', '.join(v)}" for k, v in sorted(missing_by_model.items())]
        raise ValueError("Missing required reliability metrics for strict analysis: " + " | ".join(parts))

    return pd.DataFrame(rows)


def table_6_nli_confidence_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Table 6: NLI probability distribution by judgment."""
    if df.empty or "nli_equiv_probs" not in df.columns:
        return pd.DataFrame()

    all_probs: Dict[str, List[Tuple[float, float]]] = {"same": [], "different": [], "unclear": []}
    for probs_list in df["nli_equiv_probs"].dropna().tolist():
        if not isinstance(probs_list, list):
            continue
        for p in probs_list:
            if isinstance(p, dict) and p.get("judgment") in all_probs:
                all_probs[p["judgment"]].append((float(p.get("forward", 0.0)), float(p.get("reverse", 0.0))))

    if not any(all_probs.values()):
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for judgment in ["same", "different", "unclear"]:
        pairs = all_probs[judgment]
        if not pairs:
            rows.append(
                {
                    "Judgment": judgment,
                    "Count": 0,
                    "Mean P(fwd)": "N/A",
                    "Std P(fwd)": "N/A",
                    "Mean P(rev)": "N/A",
                    "Std P(rev)": "N/A",
                }
            )
            continue
        fwds = np.array([p[0] for p in pairs], dtype=float)
        revs = np.array([p[1] for p in pairs], dtype=float)
        rows.append(
            {
                "Judgment": judgment,
                "Count": len(pairs),
                "Mean P(fwd)": f"{float(np.mean(fwds)):.3f}",
                "Std P(fwd)": f"{float(np.std(fwds)):.3f}",
                "Mean P(rev)": f"{float(np.mean(revs)):.3f}",
                "Std P(rev)": f"{float(np.std(revs)):.3f}",
            }
        )

    return pd.DataFrame(rows)


def table_7_not_attempted_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Table 7: NOT_ATTEMPTED breakdown by model."""
    if df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for model in sorted(df["model"].dropna().unique()):
        model_df = df[df["model"] == model]
        total = int(len(model_df))

        na_grade = int((model_df.get("correctness_grade") == "NOT_ATTEMPTED").sum()) if "correctness_grade" in model_df.columns else 0
        na_label = int((model_df.get("error_label_0.9") == "not_attempted").sum()) if "error_label_0.9" in model_df.columns else 0

        rows.append(
            {
                "Model": model.split("/")[-1],
                "Total": total,
                "Not Attempted (grade)": na_grade,
                "Not Attempted (label)": na_label,
                "% Not Attempted": _format_pct(na_grade, total),
            }
        )

    return pd.DataFrame(rows)


def table_8_benchmark_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Table 8: benchmark-isolated rates and pooled stratified rates."""
    required = {"dataset_name", "dataset_split"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for model in sorted(df["model"].dropna().unique()):
        model_df = df[df["model"] == model]
        if model_df.empty:
            continue

        group_rows = []
        for (dname, dsplit), sub in model_df.groupby(["dataset_name", "dataset_split"], dropna=False):
            total = len(sub)
            if total == 0:
                continue
            correct = int(sub["greedy_correct"].map(_to_bool).sum())
            sc_err = int((sub.get("error_label_0.9") == "self_consistent_error").sum()) if "error_label_0.9" in sub.columns else 0
            group_rows.append(
                {
                    "dataset_name": str(dname),
                    "dataset_split": str(dsplit),
                    "n": total,
                    "acc": correct / total,
                    "sc": sc_err / total,
                }
            )
            rows.append(
                {
                    "Model": model.split("/")[-1],
                    "Dataset": f"{dname}:{dsplit}",
                    "N": total,
                    "Accuracy": f"{100 * (correct / total):.1f}%",
                    "SC Error Rate": f"{100 * (sc_err / total):.1f}%",
                }
            )

        if group_rows:
            total_n = sum(r["n"] for r in group_rows)
            pooled_acc = sum(r["acc"] * r["n"] for r in group_rows) / total_n
            pooled_sc = sum(r["sc"] * r["n"] for r in group_rows) / total_n
            rows.append(
                {
                    "Model": model.split("/")[-1],
                    "Dataset": "POOLED(stratified)",
                    "N": total_n,
                    "Accuracy": f"{100 * pooled_acc:.1f}%",
                    "SC Error Rate": f"{100 * pooled_sc:.1f}%",
                }
            )

    return pd.DataFrame(rows)


def table_9_contamination_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Table 9: sensitivity with and without contamination-flagged rows."""
    if df.empty or "contamination_flag" not in df.columns:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for model in sorted(df["model"].dropna().unique()):
        sub = df[df["model"] == model]
        if sub.empty:
            continue

        all_n = len(sub)
        all_acc = sub["greedy_correct"].map(_to_bool).mean() if all_n else float("nan")

        clean = sub[~sub["contamination_flag"].map(_to_bool)]
        clean_n = len(clean)
        clean_acc = clean["greedy_correct"].map(_to_bool).mean() if clean_n else float("nan")

        flagged_n = int(sub["contamination_flag"].map(_to_bool).sum())

        rows.append(
            {
                "Model": model.split("/")[-1],
                "All N": all_n,
                "Flagged N": flagged_n,
                "Clean N": clean_n,
                "Accuracy (all)": f"{100 * all_acc:.1f}%" if all_n else "N/A",
                "Accuracy (clean)": f"{100 * clean_acc:.1f}%" if clean_n else "N/A",
            }
        )

    return pd.DataFrame(rows)


def table_10_paired_significance(df: pd.DataFrame) -> pd.DataFrame:
    """Table 10: paired model delta significance using exact McNemar tests."""
    if df.empty:
        return pd.DataFrame()

    models = sorted(df["model"].dropna().unique())
    if len(models) < 2:
        return pd.DataFrame()

    join_keys = ["question_id"]
    if "dataset_name" in df.columns:
        join_keys.append("dataset_name")
    if "dataset_split" in df.columns:
        join_keys.append("dataset_split")

    rows: List[Dict[str, Any]] = []

    for model_a, model_b in combinations(models, 2):
        a_df = df[df["model"] == model_a].copy()
        b_df = df[df["model"] == model_b].copy()

        keep_cols = list(set(join_keys + [
            "greedy_correct",
            "error_label_0.9",
            "correctness_grade",
            "escalated_to_human",
        ]) & set(a_df.columns) & set(b_df.columns))

        merged = a_df[keep_cols].merge(
            b_df[keep_cols],
            on=join_keys,
            how="inner",
            suffixes=("_a", "_b"),
        )

        if merged.empty:
            continue

        metrics = {
            "accuracy": (
                merged["greedy_correct_a"].map(_to_bool).astype(int),
                merged["greedy_correct_b"].map(_to_bool).astype(int),
            ) if "greedy_correct_a" in merged.columns else None,
            "self_consistent_error": (
                (merged["error_label_0.9_a"] == "self_consistent_error").astype(int),
                (merged["error_label_0.9_b"] == "self_consistent_error").astype(int),
            ) if "error_label_0.9_a" in merged.columns else None,
            "not_attempted": (
                (merged["correctness_grade_a"] == "NOT_ATTEMPTED").astype(int),
                (merged["correctness_grade_b"] == "NOT_ATTEMPTED").astype(int),
            ) if "correctness_grade_a" in merged.columns else None,
            "escalated": (
                merged["escalated_to_human_a"].map(_to_bool).astype(int),
                merged["escalated_to_human_b"].map(_to_bool).astype(int),
            ) if "escalated_to_human_a" in merged.columns else None,
        }

        for metric_name, metric_vals in metrics.items():
            if metric_vals is None:
                continue
            x, y = metric_vals
            b = int(((x == 1) & (y == 0)).sum())
            c = int(((x == 0) & (y == 1)).sum())
            p_value = _mcnemar_exact_p(b, c)
            delta = float(x.mean() - y.mean())

            rows.append(
                {
                    "Model A": model_a.split("/")[-1],
                    "Model B": model_b.split("/")[-1],
                    "Metric": metric_name,
                    "N paired": int(len(merged)),
                    "Delta (A-B)": round(delta, 4),
                    "Discordant A_only": b,
                    "Discordant B_only": c,
                    "p_value": round(p_value, 6),
                }
            )

    return pd.DataFrame(rows)


def plot_1_model_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """Plot 1: model-wise self-consistent vs inconsistent error counts."""
    if df.empty:
        return

    incorrect_df = df[~df["greedy_correct"].map(_to_bool)]
    if incorrect_df.empty or "error_label_0.9" not in incorrect_df.columns:
        logger.warning("No incorrect answers available for plot_1")
        return

    plot_data = []
    for model in sorted(df["model"].dropna().unique()):
        sub = incorrect_df[incorrect_df["model"] == model]
        short = model.split("/")[-1]

        sc = int((sub["error_label_0.9"] == "self_consistent_error").sum())
        inc = int((sub["error_label_0.9"] == "inconsistent_error").sum())

        plot_data.append({"Model": short, "Type": "Self-Consistent", "Count": sc})
        plot_data.append({"Model": short, "Type": "Inconsistent", "Count": inc})

    plot_df = pd.DataFrame(plot_data)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x="Model", y="Count", hue="Type", ax=ax)
    ax.set_title("Error Type Breakdown by Model (threshold=0.9)", fontsize=14)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Number of Errors", fontsize=12)
    ax.legend(title="Error Type")
    plt.tight_layout()

    output_path = Path(output_dir) / "plot_1_model_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved plot to %s", output_path)


def plot_3_equivalence_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """Plot 3: distribution of equivalence ratios for incorrect answers."""
    if df.empty or "equivalence_ratio" not in df.columns:
        return

    incorrect_df = df[~df["greedy_correct"].map(_to_bool)]
    ratios = pd.to_numeric(incorrect_df["equivalence_ratio"], errors="coerce").dropna()
    if ratios.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ratios, bins=20, edgecolor="black", alpha=0.7)

    for threshold, color, label in [(0.9, "red", "0.9"), (0.8, "orange", "0.8"), (0.7, "green", "0.7")]:
        ax.axvline(x=threshold, color=color, linestyle="--", linewidth=2, label=f"Threshold {label}")

    ax.set_title("Distribution of Equivalence Ratios (Incorrect Answers)", fontsize=14)
    ax.set_xlabel("Equivalence Ratio", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend()
    plt.tight_layout()

    output_path = Path(output_dir) / "plot_3_equivalence_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved plot to %s", output_path)


def find_common_self_consistent_errors(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top repeated self-consistent error patterns."""
    if df.empty or "error_label_0.9" not in df.columns:
        return pd.DataFrame()

    sc_errors = df[df["error_label_0.9"] == "self_consistent_error"]
    if sc_errors.empty:
        return pd.DataFrame()

    error_counts: Counter = Counter()
    examples: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for _, row in sc_errors.iterrows():
        question = str(row.get("question", ""))
        wrong_answer = str(row.get("greedy_answer", ""))
        key = (question[:100], wrong_answer)
        error_counts[key] += 1

        model_short = str(row.get("model", "")).split("/")[-1]
        if key not in examples:
            gt = row.get("ground_truth", [])
            examples[key] = {
                "Question": (question[:150] + "...") if len(question) > 150 else question,
                "Wrong Answer": wrong_answer,
                "Correct Answer(s)": str(gt),
                "Model(s)": model_short,
                "Equiv Ratio": row.get("equivalence_ratio", "N/A"),
            }
        else:
            existing = examples[key]["Model(s)"]
            if model_short and model_short not in existing:
                examples[key]["Model(s)"] = f"{existing}, {model_short}"

    rows: List[Dict[str, Any]] = []
    for key, count in error_counts.most_common(n):
        item = dict(examples[key])
        item["Occurrence"] = int(count)
        rows.append(item)

    return pd.DataFrame(rows)


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    records = []
    for _, row in df.iterrows():
        out: Dict[str, Any] = {}
        for k, v in row.to_dict().items():
            if isinstance(v, np.generic):
                out[k] = v.item()
            else:
                out[k] = v
        records.append(out)
    return records


def build_eval_summary(
    full_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    excluded_incomplete: int,
    protocol_version: Optional[str],
    tables: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Build machine-readable summary."""
    return {
        "dataset_summary": {
            "total_records": int(len(full_df)),
            "primary_records": int(len(primary_df)),
            "excluded_incomplete": int(excluded_incomplete),
            "unique_questions": int(full_df["question_id"].nunique()) if "question_id" in full_df.columns else 0,
            "models_tested": int(full_df["model"].nunique()) if "model" in full_df.columns else 0,
            "models": [m.split("/")[-1] for m in sorted(full_df["model"].dropna().unique())] if "model" in full_df.columns else [],
            "protocol_version": protocol_version,
        },
        "tables": {name: _df_to_records(df) for name, df in tables.items()},
    }


def generate_report(
    full_df: pd.DataFrame,
    output_dir: str,
    require_uniform_protocol: bool = False,
    with_ci: bool = False,
    with_significance: bool = False,
) -> str:
    """Generate report, tables, and optional significance outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    protocol_version = validate_uniform_protocol(full_df, require_uniform_protocol)
    primary_df, excluded_incomplete = primary_analysis_slice(full_df)

    table1 = table_1_error_breakdown(primary_df, with_ci=with_ci)
    table3 = table_3_threshold_sensitivity(primary_df)
    table4 = table_4_unclear_rate(primary_df)
    table5 = table_5_reliability_panel(primary_df, strict=require_uniform_protocol)
    table6 = table_6_nli_confidence_distribution(primary_df)
    table7 = table_7_not_attempted_breakdown(primary_df)
    table8 = table_8_benchmark_breakdown(primary_df)
    table9 = table_9_contamination_sensitivity(primary_df)
    table10 = table_10_paired_significance(primary_df) if with_significance else pd.DataFrame()
    examples = find_common_self_consistent_errors(primary_df, n=10)

    report_lines: List[str] = []
    report_lines.append("=" * 78)
    report_lines.append("LLM SELF-CONSISTENT ERROR MEASUREMENT - ANALYSIS REPORT")
    report_lines.append("=" * 78)
    report_lines.append("")

    report_lines.append("DATASET SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Total records loaded: {len(full_df)}")
    report_lines.append(f"Primary records analyzed (excluding incomplete): {len(primary_df)}")
    report_lines.append(f"Excluded incomplete records: {excluded_incomplete}")
    report_lines.append(f"Unique questions: {primary_df['question_id'].nunique() if 'question_id' in primary_df.columns else 0}")
    report_lines.append(f"Models tested: {primary_df['model'].nunique() if 'model' in primary_df.columns else 0}")
    if "model" in primary_df.columns:
        models = ", ".join(m.split("/")[-1] for m in sorted(primary_df["model"].dropna().unique()))
        report_lines.append(f"Models: {models}")
    if protocol_version:
        report_lines.append(f"Protocol version: {protocol_version}")
    report_lines.append("")

    sections = [
        ("TABLE 1: ERROR BREAKDOWN BY MODEL", table1, "table_1_error_breakdown.csv"),
        ("TABLE 3: THRESHOLD SENSITIVITY", table3, "table_3_threshold_sensitivity.csv"),
        ("TABLE 4: SEMANTIC JUDGE RELIABILITY (UNCLEAR RATE)", table4, "table_4_unclear_rate.csv"),
        ("TABLE 5: RELIABILITY PANEL", table5, "table_5_reliability_panel.csv"),
        ("TABLE 6: NLI CONFIDENCE DISTRIBUTION", table6, "table_6_nli_confidence_distribution.csv"),
        ("TABLE 7: NOT_ATTEMPTED BREAKDOWN", table7, "table_7_not_attempted_breakdown.csv"),
        ("TABLE 8: BENCHMARK ISOLATION + STRATIFIED POOL", table8, "table_8_benchmark_breakdown.csv"),
        ("TABLE 9: CONTAMINATION SENSITIVITY", table9, "table_9_contamination_sensitivity.csv"),
    ]

    if with_significance:
        sections.append(("TABLE 10: PAIRED SIGNIFICANCE (EXACT MCNEMAR)", table10, "table_10_significance.csv"))

    for title, table_df, filename in sections:
        report_lines.append(title)
        report_lines.append("-" * 40)
        if not table_df.empty:
            report_lines.append(table_df.to_string(index=False))
            table_df.to_csv(output_path / filename, index=False)
        else:
            report_lines.append("No data available")
        report_lines.append("")

    report_lines.append("TOP SELF-CONSISTENT ERRORS")
    report_lines.append("-" * 40)
    if not examples.empty:
        for i, row in examples.iterrows():
            report_lines.append(f"{i+1}. Question: {row['Question']}")
            report_lines.append(f"   Wrong Answer: {row['Wrong Answer']}")
            report_lines.append(f"   Correct: {row['Correct Answer(s)']}")
            report_lines.append(f"   Model(s): {row['Model(s)']}")
            report_lines.append(f"   Equivalence Ratio: {row['Equiv Ratio']}")
            report_lines.append("")
        examples.to_csv(output_path / "top_self_consistent_errors.csv", index=False)
    else:
        report_lines.append("No self-consistent errors found")

    report_text = "\n".join(report_lines)
    report_file = output_path / "analysis_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    tables = {
        "table_1": table1,
        "table_3": table3,
        "table_4": table4,
        "table_5": table5,
        "table_6": table6,
        "table_7": table7,
        "table_8": table8,
        "table_9": table9,
        "table_10": table10,
        "examples": examples,
    }
    eval_dict = build_eval_summary(
        full_df=full_df,
        primary_df=primary_df,
        excluded_incomplete=excluded_incomplete,
        protocol_version=protocol_version,
        tables=tables,
    )
    with open(output_path / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_dict, f, indent=2)

    logger.info("Saved report to %s", report_file)
    logger.info("Saved evaluation summary to %s", output_path / "eval_results.json")

    print("\n" + report_text)
    return report_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze LLM Self-Consistent Error Measurement results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Optional config file used for analysis defaults (default: config.yaml).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/results/evaluated",
        help="Directory containing evaluated results (default: data/results/evaluated)",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="results_v2_eval.jsonl",
        help="Results file name (default: results_v2_eval.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results/analysis",
        help="Directory for tables/plots/report (default: data/results/analysis)",
    )
    parser.add_argument(
        "--require-uniform-protocol",
        action="store_true",
        default=None,
        help="Hard-fail if protocol_version is missing or mixed across rows.",
    )
    parser.add_argument(
        "--with-ci",
        action="store_true",
        help="Include 95% bootstrap confidence intervals in key metrics.",
    )
    parser.add_argument(
        "--with-significance",
        action="store_true",
        help="Compute paired significance tests for model deltas (exact McNemar).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(script_dir / config_path)
    analysis_defaults = load_analysis_defaults(config_path)
    require_uniform_protocol = (
        bool(analysis_defaults.get("require_uniform_protocol", False))
        if args.require_uniform_protocol is None
        else bool(args.require_uniform_protocol)
    )

    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = str(script_dir / results_dir)

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = str(script_dir / output_dir)

    df = load_data(results_dir, args.results_file)
    if df.empty:
        logger.error("No data to analyze. Run the pipeline first.")
        sys.exit(1)

    generate_report(
        full_df=df,
        output_dir=output_dir,
        require_uniform_protocol=require_uniform_protocol,
        with_ci=args.with_ci,
        with_significance=args.with_significance,
    )

    if not args.no_plots:
        logger.info("Generating plots...")
        primary_df, _ = primary_analysis_slice(df)
        plot_1_model_comparison(primary_df, output_dir)
        plot_3_equivalence_distribution(primary_df, output_dir)

    logger.info("Analysis complete. Output saved to %s", output_dir)


if __name__ == "__main__":
    main()
