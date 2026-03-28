#!/usr/bin/env python3
"""Build thesis-grade analysis artifacts for v2 evaluated JSONL data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REQUIRED_COLUMNS: Tuple[str, ...] = (
    "question_id",
    "model",
    "correctness_grade",
    "error_label_1.0",
    "greedy_correct",
    "equivalence_ratio",
    "stochastic_actual_n",
)

CORRECTNESS_DOMAIN: Tuple[str, ...] = ("CORRECT", "INCORRECT", "NOT_ATTEMPTED")
ERROR_LABEL_DOMAIN: Tuple[str, ...] = (
    "reliably_correct",
    "fragile_correct",
    "self_consistent_error",
    "inconsistent_error",
    "not_attempted",
)


@dataclass
class MetricEstimate:
    value: float
    ci_low: float
    ci_high: float
    n: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return False


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def normalize_minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_val = float(np.nanmin(values))
    max_val = float(np.nanmax(values))
    if not math.isfinite(min_val) or not math.isfinite(max_val):
        return np.full(values.shape, np.nan, dtype=float)
    span = max_val - min_val
    if span <= 1e-12:
        return np.zeros(values.shape, dtype=float)
    return (values - min_val) / span


def wilcoxon_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    n = len(y_true)
    if n == 0:
        return float("nan")
    y_true = y_true.astype(int)
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.empty(n, dtype=float)

    index = 0
    while index < n:
        end = index
        while end + 1 < n and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        average_rank = (index + end + 2) / 2.0
        ranks[order[index : end + 1]] = average_rank
        index = end + 1

    rank_sum_positive = float(ranks[y_true == 1].sum())
    auc = (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    y_true = y_true.astype(int)
    positive_count = int((y_true == 1).sum())
    if positive_count == 0:
        return float("nan")

    order = np.argsort(-y_score)
    labels = y_true[order]
    tp = 0
    fp = 0
    ap = 0.0
    prev_recall = 0.0

    for label in labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / positive_count
        if label == 1:
            ap += (recall - prev_recall) * precision
            prev_recall = recall

    return float(ap)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


def accuracy_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    if len(y_true) == 0:
        return float("nan")
    predictions = (y_prob >= threshold).astype(int)
    return float(np.mean(predictions == y_true))


def bootstrap_ci_scalar(
    values: np.ndarray,
    metric_fn,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        value = float(metric_fn(values))
        return value, value
    rng = np.random.default_rng(seed)
    estimates: List[float] = []
    n = len(values)
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, n)
        estimate = float(metric_fn(values[sample_idx]))
        if math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return float("nan"), float("nan")
    low = float(np.quantile(estimates, alpha / 2.0))
    high = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    return low, high


def bootstrap_ci_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    if len(y_true) == 0:
        return float("nan"), float("nan")
    if len(y_true) == 1:
        value = float(metric_fn(y_true, y_score))
        return value, value
    rng = np.random.default_rng(seed)
    estimates: List[float] = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, n)
        estimate = float(metric_fn(y_true[sample_idx], y_score[sample_idx]))
        if math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return float("nan"), float("nan")
    low = float(np.quantile(estimates, alpha / 2.0))
    high = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    return low, high


def bootstrap_ci_difference(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    if len(a) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(a)
    estimates: List[float] = []
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, n)
        estimates.append(float(np.mean(a[sample_idx] - b[sample_idx])))
    low = float(np.quantile(estimates, alpha / 2.0))
    high = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    return low, high


def quantile_bins(values: pd.Series, n_bins: int = 5) -> pd.Series:
    non_null = values.dropna()
    if non_null.nunique() <= 1:
        return pd.Series(["all"] * len(values), index=values.index)
    labels = [f"Q{i+1}" for i in range(n_bins)]
    binned = pd.qcut(values.rank(method="first"), q=n_bins, labels=labels)
    return binned.astype(str)


def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = y_true.astype(int)
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(-y_score)
    labels = y_true[order]
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    tpr = np.concatenate(([0.0], tp / positives, [1.0]))
    fpr = np.concatenate(([0.0], fp / negatives, [1.0]))
    return fpr, tpr


def pr_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = y_true.astype(int)
    positives = int((y_true == 1).sum())
    if positives == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0])
    order = np.argsort(-y_score)
    labels = y_true[order]
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return recall, precision


def exact_binomial_two_sided(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    tail = 0.0
    for value in range(0, k + 1):
        tail += math.comb(n, value) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return exact_binomial_two_sided(min(b, c), n)


def holm_adjust(p_values: Sequence[float], alpha: float = 0.05) -> Tuple[List[float], List[bool], List[float]]:
    count = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * count
    alphas = [alpha] * count

    for rank, (original_idx, p_val) in enumerate(indexed):
        multiplier = count - rank
        adjusted[original_idx] = min(1.0, p_val * multiplier)
        alphas[original_idx] = alpha / multiplier

    running_max = 0.0
    for original_idx, _ in indexed:
        running_max = max(running_max, adjusted[original_idx])
        adjusted[original_idx] = running_max

    significant = [adj <= alpha for adj in adjusted]
    return adjusted, significant, alphas


def validate_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("Input JSONL is empty")

    frame = pd.DataFrame(records)
    missing_required = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    correctness_values = sorted(set(frame["correctness_grade"].dropna().astype(str).tolist()))
    invalid_correctness = [value for value in correctness_values if value not in CORRECTNESS_DOMAIN]

    error_values = sorted(set(frame["error_label_1.0"].dropna().astype(str).tolist()))
    invalid_error_labels = [value for value in error_values if value not in ERROR_LABEL_DOMAIN]

    key_null_rates = {
        column: float(frame[column].isna().mean()) for column in REQUIRED_COLUMNS
    }

    duplicate_rows = int(frame.duplicated(subset=["question_id", "model"], keep=False).sum())

    return {
        "row_count": int(len(frame)),
        "question_count": int(frame["question_id"].nunique()),
        "model_count": int(frame["model"].nunique()),
        "invalid_correctness_values": invalid_correctness,
        "invalid_error_label_values": invalid_error_labels,
        "null_rates": key_null_rates,
        "duplicate_question_model_rows": duplicate_rows,
    }


def metric_row(
    metric_name: str,
    subset: str,
    value: float,
    ci_low: float,
    ci_high: float,
    n: int,
    notes: str,
) -> Dict[str, Any]:
    return {
        "metric_name": metric_name,
        "subset": subset,
        "value": value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n,
        "notes": notes,
    }


def estimate_binary_metric(
    values: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> MetricEstimate:
    mean_value = float(np.mean(values)) if len(values) else float("nan")
    ci_low, ci_high = bootstrap_ci_scalar(
        values,
        metric_fn=lambda sampled: np.mean(sampled),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return MetricEstimate(value=mean_value, ci_low=ci_low, ci_high=ci_high, n=int(len(values)))


def score_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int,
    seed: int,
    with_ci: bool = True,
) -> Dict[str, MetricEstimate]:
    metrics: Dict[str, MetricEstimate] = {}

    auroc = wilcoxon_auc(y_true, y_prob)
    auroc_ci = (
        bootstrap_ci_metric(y_true, y_prob, wilcoxon_auc, n_bootstrap, seed)
        if with_ci
        else (float("nan"), float("nan"))
    )
    metrics["auroc"] = MetricEstimate(auroc, auroc_ci[0], auroc_ci[1], int(len(y_true)))

    pr_auc = average_precision(y_true, y_prob)
    pr_auc_ci = (
        bootstrap_ci_metric(y_true, y_prob, average_precision, n_bootstrap, seed + 1)
        if with_ci
        else (float("nan"), float("nan"))
    )
    metrics["pr_auc"] = MetricEstimate(pr_auc, pr_auc_ci[0], pr_auc_ci[1], int(len(y_true)))

    brier = brier_score(y_true, y_prob)
    brier_ci = (
        bootstrap_ci_metric(y_true, y_prob, brier_score, n_bootstrap, seed + 2)
        if with_ci
        else (float("nan"), float("nan"))
    )
    metrics["brier"] = MetricEstimate(brier, brier_ci[0], brier_ci[1], int(len(y_true)))

    acc_05 = accuracy_at_threshold(y_true, y_prob, 0.5)
    acc_05_ci = (
        bootstrap_ci_metric(
            y_true,
            y_prob,
            lambda yt, yp: accuracy_at_threshold(yt, yp, 0.5),
            n_bootstrap,
            seed + 3,
        )
        if with_ci
        else (float("nan"), float("nan"))
    )
    metrics["accuracy_at_0.5"] = MetricEstimate(acc_05, acc_05_ci[0], acc_05_ci[1], int(len(y_true)))
    return metrics


def compute_aggregate_metrics(
    frame: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []

    total = len(frame)
    correct_mask = frame["correctness_grade"] == "CORRECT"
    incorrect_mask = frame["correctness_grade"] == "INCORRECT"
    not_attempted_mask = frame["correctness_grade"] == "NOT_ATTEMPTED"

    binary_specs = {
        "accuracy": correct_mask.to_numpy(dtype=float),
        "incorrect_rate": incorrect_mask.to_numpy(dtype=float),
        "not_attempted_rate": not_attempted_mask.to_numpy(dtype=float),
        "ce_rate_all": (frame["error_label_1.0"] == "self_consistent_error").to_numpy(dtype=float),
        "ie_rate_all": (frame["error_label_1.0"] == "inconsistent_error").to_numpy(dtype=float),
        "unclear_rate": frame["correctness_unclear"].apply(to_bool).to_numpy(dtype=float)
        if "correctness_unclear" in frame.columns
        else np.zeros(total, dtype=float),
    }

    summary_json: Dict[str, Any] = {
        "bootstrap": {
            "seed": seed,
            "n_bootstrap": n_bootstrap,
            "alpha": 0.05,
        },
        "overall": {},
        "scores": {},
        "per_model": {},
    }

    for idx, (name, values) in enumerate(binary_specs.items()):
        estimate = estimate_binary_metric(values, n_bootstrap=n_bootstrap, seed=seed + idx)
        rows.append(
            metric_row(
                metric_name=name,
                subset="all_rows",
                value=estimate.value,
                ci_low=estimate.ci_low,
                ci_high=estimate.ci_high,
                n=estimate.n,
                notes="Bootstrap 95% CI",
            )
        )
        summary_json["overall"][name] = {
            "estimate": estimate.value,
            "ci_low": estimate.ci_low,
            "ci_high": estimate.ci_high,
            "n": estimate.n,
        }

    incorrect_frame = frame[incorrect_mask].copy()
    if len(incorrect_frame) > 0:
        ce_in_incorrect = (incorrect_frame["error_label_1.0"] == "self_consistent_error").to_numpy(dtype=float)
        ie_in_incorrect = (incorrect_frame["error_label_1.0"] == "inconsistent_error").to_numpy(dtype=float)

        ce_estimate = estimate_binary_metric(ce_in_incorrect, n_bootstrap=n_bootstrap, seed=seed + 100)
        ie_estimate = estimate_binary_metric(ie_in_incorrect, n_bootstrap=n_bootstrap, seed=seed + 101)

        rows.append(
            metric_row(
                metric_name="ce_share_among_incorrect",
                subset="incorrect_only",
                value=ce_estimate.value,
                ci_low=ce_estimate.ci_low,
                ci_high=ce_estimate.ci_high,
                n=ce_estimate.n,
                notes="Bootstrap 95% CI",
            )
        )
        rows.append(
            metric_row(
                metric_name="ie_share_among_incorrect",
                subset="incorrect_only",
                value=ie_estimate.value,
                ci_low=ie_estimate.ci_low,
                ci_high=ie_estimate.ci_high,
                n=ie_estimate.n,
                notes="Bootstrap 95% CI",
            )
        )

        summary_json["overall"]["ce_share_among_incorrect"] = {
            "estimate": ce_estimate.value,
            "ci_low": ce_estimate.ci_low,
            "ci_high": ce_estimate.ci_high,
            "n": ce_estimate.n,
        }
        summary_json["overall"]["ie_share_among_incorrect"] = {
            "estimate": ie_estimate.value,
            "ci_low": ie_estimate.ci_low,
            "ci_high": ie_estimate.ci_high,
            "n": ie_estimate.n,
        }

    answered = frame[frame["correctness_grade"].isin(["CORRECT", "INCORRECT"])].copy()
    answered["y_error"] = (answered["correctness_grade"] == "INCORRECT").astype(int)
    answered["score_disagreement"] = (1.0 - answered["equivalence_ratio"].astype(float)).clip(lower=0.0, upper=1.0)

    entropy_values = answered["semantic_entropy"].astype(float)
    entropy_values = entropy_values.where(np.isfinite(entropy_values), np.nan)
    answered["score_entropy"] = normalize_minmax(entropy_values.to_numpy(dtype=float))

    score_inputs = {
        "disagreement_prob": answered[["y_error", "score_disagreement"]].dropna(),
        "semantic_entropy_prob": answered[["y_error", "score_entropy"]].dropna(),
    }

    for index, (score_name, score_frame) in enumerate(score_inputs.items()):
        if len(score_frame) == 0:
            continue
        y_true = score_frame["y_error"].to_numpy(dtype=int)
        y_prob = score_frame.iloc[:, 1].to_numpy(dtype=float)
        estimates = score_metrics(y_true, y_prob, n_bootstrap=n_bootstrap, seed=seed + 200 + index * 20)

        summary_json["scores"][score_name] = {}
        for metric_name, estimate in estimates.items():
            rows.append(
                metric_row(
                    metric_name=f"{score_name}_{metric_name}",
                    subset="answered_rows",
                    value=estimate.value,
                    ci_low=estimate.ci_low,
                    ci_high=estimate.ci_high,
                    n=estimate.n,
                    notes="Bootstrap 95% CI",
                )
            )
            summary_json["scores"][score_name][metric_name] = {
                "estimate": estimate.value,
                "ci_low": estimate.ci_low,
                "ci_high": estimate.ci_high,
                "n": estimate.n,
            }

    return pd.DataFrame(rows), summary_json, answered


def compute_group_metrics(
    frame: pd.DataFrame,
    answered_frame: pd.DataFrame,
    group_bootstrap_iters: int,
    seed: int,
    summary_json: Dict[str, Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def group_record(group_type: str, model: str, dataset_name: str, dataset_split: str, group_frame: pd.DataFrame, group_answered: pd.DataFrame, base_seed: int) -> Dict[str, Any]:
        n_rows = len(group_frame)
        n_questions = int(group_frame["question_id"].nunique())

        accuracy_values = (group_frame["correctness_grade"] == "CORRECT").to_numpy(dtype=float)
        incorrect_values = (group_frame["correctness_grade"] == "INCORRECT").to_numpy(dtype=float)
        not_attempted_values = (group_frame["correctness_grade"] == "NOT_ATTEMPTED").to_numpy(dtype=float)

        accuracy = estimate_binary_metric(accuracy_values, group_bootstrap_iters, base_seed)
        incorrect_rate = estimate_binary_metric(incorrect_values, group_bootstrap_iters, base_seed + 1)
        not_attempted_rate = estimate_binary_metric(not_attempted_values, group_bootstrap_iters, base_seed + 2)

        incorrect_group = group_frame[group_frame["correctness_grade"] == "INCORRECT"]
        if len(incorrect_group) > 0:
            ce_share = estimate_binary_metric(
                (incorrect_group["error_label_1.0"] == "self_consistent_error").to_numpy(dtype=float),
                group_bootstrap_iters,
                base_seed + 3,
            )
            ie_share = estimate_binary_metric(
                (incorrect_group["error_label_1.0"] == "inconsistent_error").to_numpy(dtype=float),
                group_bootstrap_iters,
                base_seed + 4,
            )
        else:
            ce_share = MetricEstimate(float("nan"), float("nan"), float("nan"), 0)
            ie_share = MetricEstimate(float("nan"), float("nan"), float("nan"), 0)

        output: Dict[str, Any] = {
            "group_type": group_type,
            "model": model,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "n_rows": n_rows,
            "n_questions": n_questions,
            "accuracy": accuracy.value,
            "accuracy_ci_low": accuracy.ci_low,
            "accuracy_ci_high": accuracy.ci_high,
            "incorrect_rate": incorrect_rate.value,
            "incorrect_rate_ci_low": incorrect_rate.ci_low,
            "incorrect_rate_ci_high": incorrect_rate.ci_high,
            "not_attempted_rate": not_attempted_rate.value,
            "not_attempted_rate_ci_low": not_attempted_rate.ci_low,
            "not_attempted_rate_ci_high": not_attempted_rate.ci_high,
            "ce_share_among_incorrect": ce_share.value,
            "ce_share_among_incorrect_ci_low": ce_share.ci_low,
            "ce_share_among_incorrect_ci_high": ce_share.ci_high,
            "ie_share_among_incorrect": ie_share.value,
            "ie_share_among_incorrect_ci_low": ie_share.ci_low,
            "ie_share_among_incorrect_ci_high": ie_share.ci_high,
        }

        if len(group_answered) > 1 and group_answered["y_error"].nunique() > 1:
            score_specs = {
                "disagreement": group_answered["score_disagreement"].to_numpy(dtype=float),
            }
            entropy_non_null = group_answered.dropna(subset=["score_entropy"])
            score_specs["entropy"] = entropy_non_null["score_entropy"].to_numpy(dtype=float)

            y_disagreement = group_answered["y_error"].to_numpy(dtype=int)
            disagreement_metrics = score_metrics(
                y_disagreement,
                score_specs["disagreement"],
                group_bootstrap_iters,
                base_seed + 10,
                with_ci=False,
            )
            output.update(
                {
                    "disagreement_auroc": disagreement_metrics["auroc"].value,
                    "disagreement_pr_auc": disagreement_metrics["pr_auc"].value,
                    "disagreement_brier": disagreement_metrics["brier"].value,
                    "disagreement_accuracy_at_0.5": disagreement_metrics["accuracy_at_0.5"].value,
                }
            )

            if len(entropy_non_null) > 1 and entropy_non_null["y_error"].nunique() > 1:
                y_entropy = entropy_non_null["y_error"].to_numpy(dtype=int)
                entropy_metrics = score_metrics(
                    y_entropy,
                    score_specs["entropy"],
                    group_bootstrap_iters,
                    base_seed + 30,
                    with_ci=False,
                )
                output.update(
                    {
                        "entropy_auroc": entropy_metrics["auroc"].value,
                        "entropy_pr_auc": entropy_metrics["pr_auc"].value,
                        "entropy_brier": entropy_metrics["brier"].value,
                        "entropy_accuracy_at_0.5": entropy_metrics["accuracy_at_0.5"].value,
                    }
                )
            else:
                output.update(
                    {
                        "entropy_auroc": float("nan"),
                        "entropy_pr_auc": float("nan"),
                        "entropy_brier": float("nan"),
                        "entropy_accuracy_at_0.5": float("nan"),
                    }
                )
        else:
            output.update(
                {
                    "disagreement_auroc": float("nan"),
                    "disagreement_pr_auc": float("nan"),
                    "disagreement_brier": float("nan"),
                    "disagreement_accuracy_at_0.5": float("nan"),
                    "entropy_auroc": float("nan"),
                    "entropy_pr_auc": float("nan"),
                    "entropy_brier": float("nan"),
                    "entropy_accuracy_at_0.5": float("nan"),
                }
            )

        return output

    model_names = sorted(frame["model"].dropna().unique().tolist())
    for model_index, model_name in enumerate(model_names):
        model_frame = frame[frame["model"] == model_name].copy()
        model_answered = answered_frame[answered_frame["model"] == model_name].copy()
        record = group_record(
            "model",
            model_name,
            "ALL",
            "ALL",
            model_frame,
            model_answered,
            seed + 1000 + model_index * 100,
        )
        rows.append(record)

        summary_json["per_model"][model_name] = {
            key: value for key, value in record.items() if key not in {"group_type", "model", "dataset_name", "dataset_split"}
        }

    dataset_grouped = frame.groupby(["model", "dataset_name", "dataset_split"], dropna=False)
    answered_norm = answered_frame.copy()
    answered_norm["dataset_name_norm"] = answered_norm["dataset_name"].fillna("<NA>").astype(str)
    answered_norm["dataset_split_norm"] = answered_norm["dataset_split"].fillna("<NA>").astype(str)
    for group_index, ((model, dataset_name, dataset_split), group_frame) in enumerate(dataset_grouped):
        dataset_name_norm = "<NA>" if pd.isna(dataset_name) else str(dataset_name)
        dataset_split_norm = "<NA>" if pd.isna(dataset_split) else str(dataset_split)
        subset_answered = answered_norm[
            (answered_norm["model"] == model)
            & (answered_norm["dataset_name_norm"] == dataset_name_norm)
            & (answered_norm["dataset_split_norm"] == dataset_split_norm)
        ]
        rows.append(
            group_record(
                "model_dataset",
                str(model),
                str(dataset_name),
                str(dataset_split),
                group_frame,
                subset_answered,
                seed + 3000 + group_index * 10,
            )
        )

    return pd.DataFrame(rows)


def compute_pairwise_hypothesis_tests(
    frame: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    alpha: float,
) -> pd.DataFrame:
    pivot = frame.copy()
    pivot["is_correct"] = (pivot["correctness_grade"] == "CORRECT").astype(int)
    question_model = pivot.pivot_table(
        index="question_id",
        columns="model",
        values="is_correct",
        aggfunc="first",
    )

    models = sorted(question_model.columns.tolist())
    rows: List[Dict[str, Any]] = []

    pair_index = 0
    for left_index in range(len(models)):
        for right_index in range(left_index + 1, len(models)):
            model_a = models[left_index]
            model_b = models[right_index]
            paired = question_model[[model_a, model_b]].dropna()
            if paired.empty:
                continue

            a_values = paired[model_a].to_numpy(dtype=int)
            b_values = paired[model_b].to_numpy(dtype=int)

            b_count = int(np.sum((a_values == 1) & (b_values == 0)))
            c_count = int(np.sum((a_values == 0) & (b_values == 1)))
            p_value = mcnemar_exact_p(b_count, c_count)

            diff = float(np.mean(a_values - b_values))
            ci_low, ci_high = bootstrap_ci_difference(
                a_values,
                b_values,
                n_bootstrap=n_bootstrap,
                seed=seed + pair_index,
            )

            rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_pairs": int(len(paired)),
                    "accuracy_a": float(np.mean(a_values)),
                    "accuracy_b": float(np.mean(b_values)),
                    "accuracy_diff_a_minus_b": diff,
                    "accuracy_diff_ci_low": ci_low,
                    "accuracy_diff_ci_high": ci_high,
                    "mcnemar_b": b_count,
                    "mcnemar_c": c_count,
                    "mcnemar_p_value": p_value,
                }
            )
            pair_index += 1

    if not rows:
        return pd.DataFrame()

    p_values = [row["mcnemar_p_value"] for row in rows]
    adjusted, significant, alpha_thresholds = holm_adjust(p_values, alpha=alpha)

    for index, row in enumerate(rows):
        row["holm_adjusted_p"] = adjusted[index]
        row["holm_alpha_threshold"] = alpha_thresholds[index]
        row["significant_after_holm"] = bool(significant[index])

    return pd.DataFrame(rows).sort_values(by="holm_adjusted_p", ascending=True)


def compute_threshold_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    answered = frame[frame["correctness_grade"].isin(["CORRECT", "INCORRECT"])].copy()
    has_strict_reference = "error_label_1.0" in frame.columns

    for threshold in (0.7, 0.8, 0.9, 1.0):
        key = f"error_label_{threshold:.1f}"
        if key not in frame.columns:
            continue
        cols = ["correctness_grade", key]
        if has_strict_reference and key != "error_label_1.0":
            cols.append("error_label_1.0")
        threshold_frame = frame[cols].copy()
        threshold_frame = threshold_frame[threshold_frame["correctness_grade"].isin(["CORRECT", "INCORRECT"])]
        if threshold_frame.empty:
            continue

        incorrect_only = threshold_frame[threshold_frame["correctness_grade"] == "INCORRECT"]
        incorrect_labels = incorrect_only[key].astype(str)
        ce_count = int((incorrect_labels == "self_consistent_error").sum())
        ie_count = int((incorrect_labels == "inconsistent_error").sum())

        # Meaningful threshold metric: how well thresholded CE labels match the
        # strict reference (error_label_1.0) among incorrect responses.
        precision = float("nan")
        recall = float("nan")
        f1 = float("nan")
        if has_strict_reference and not incorrect_only.empty:
            pred_ce = (incorrect_only[key].astype(str) == "self_consistent_error").astype(int)
            ref_ce = (incorrect_only["error_label_1.0"].astype(str) == "self_consistent_error").astype(int)

            tp = int(np.sum((pred_ce == 1) & (ref_ce == 1)))
            fp = int(np.sum((pred_ce == 1) & (ref_ce == 0)))
            fn = int(np.sum((pred_ce == 0) & (ref_ce == 1)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if math.isfinite(precision) and math.isfinite(recall) and (precision + recall) > 0
                else float("nan")
            )

        rows.append(
            {
                "analysis_type": "equivalence_threshold",
                "dimension": "threshold",
                "value": threshold,
                "n": int(len(threshold_frame)),
                "n_incorrect": int(len(incorrect_only)),
                "ce_count": ce_count,
                "ie_count": ie_count,
                "ce_share_among_incorrect": ce_count / len(incorrect_only) if len(incorrect_only) else float("nan"),
                "ie_share_among_incorrect": ie_count / len(incorrect_only) if len(incorrect_only) else float("nan"),
                "precision_ce_vs_strict": precision,
                "recall_ce_vs_strict": recall,
                "f1_ce_vs_strict": f1,
                # Backward-compatible aliases retained for downstream consumers.
                "precision_incorrect": precision,
                "recall_incorrect": recall,
                "f1_incorrect": f1,
            }
        )

    if "semantic_entropy" in answered.columns:
        entropy_non_null = answered.dropna(subset=["semantic_entropy"]).copy()
        if not entropy_non_null.empty:
            entropy_non_null["entropy_bin"] = quantile_bins(entropy_non_null["semantic_entropy"], n_bins=5)
            for entropy_bin, group in entropy_non_null.groupby("entropy_bin"):
                rows.append(
                    {
                        "analysis_type": "semantic_entropy_bin",
                        "dimension": "entropy_bin",
                        "value": str(entropy_bin),
                        "n": int(len(group)),
                        "n_incorrect": int((group["correctness_grade"] == "INCORRECT").sum()),
                        "ce_count": int((group["error_label_1.0"] == "self_consistent_error").sum()),
                        "ie_count": int((group["error_label_1.0"] == "inconsistent_error").sum()),
                        "ce_share_among_incorrect": float(
                            ((group["error_label_1.0"] == "self_consistent_error").sum())
                            / max((group["correctness_grade"] == "INCORRECT").sum(), 1)
                        ),
                        "ie_share_among_incorrect": float(
                            ((group["error_label_1.0"] == "inconsistent_error").sum())
                            / max((group["correctness_grade"] == "INCORRECT").sum(), 1)
                        ),
                        "precision_ce_vs_strict": float("nan"),
                        "recall_ce_vs_strict": float("nan"),
                        "f1_ce_vs_strict": float("nan"),
                        "precision_incorrect": float("nan"),
                        "recall_incorrect": float("nan"),
                        "f1_incorrect": float("nan"),
                    }
                )

    if "stochastic_actual_n" in answered.columns:
        for sample_n, group in answered.groupby("stochastic_actual_n"):
            rows.append(
                {
                    "analysis_type": "sampling_depth",
                    "dimension": "stochastic_actual_n",
                    "value": int(sample_n),
                    "n": int(len(group)),
                    "n_incorrect": int((group["correctness_grade"] == "INCORRECT").sum()),
                    "ce_count": int((group["error_label_1.0"] == "self_consistent_error").sum()),
                    "ie_count": int((group["error_label_1.0"] == "inconsistent_error").sum()),
                    "ce_share_among_incorrect": float(
                        ((group["error_label_1.0"] == "self_consistent_error").sum())
                        / max((group["correctness_grade"] == "INCORRECT").sum(), 1)
                    ),
                    "ie_share_among_incorrect": float(
                        ((group["error_label_1.0"] == "inconsistent_error").sum())
                        / max((group["correctness_grade"] == "INCORRECT").sum(), 1)
                    ),
                    "precision_ce_vs_strict": float("nan"),
                    "recall_ce_vs_strict": float("nan"),
                    "f1_ce_vs_strict": float("nan"),
                    "precision_incorrect": float("nan"),
                    "recall_incorrect": float("nan"),
                    "f1_incorrect": float("nan"),
                }
            )

    for (dataset_name, dataset_split), group in answered.groupby(["dataset_name", "dataset_split"], dropna=False):
        rows.append(
            {
                "analysis_type": "dataset_stratified",
                "dimension": "dataset_name_split",
                "value": f"{dataset_name}::{dataset_split}",
                "n": int(len(group)),
                "n_incorrect": int((group["correctness_grade"] == "INCORRECT").sum()),
                "ce_count": int((group["error_label_1.0"] == "self_consistent_error").sum()),
                "ie_count": int((group["error_label_1.0"] == "inconsistent_error").sum()),
                "ce_share_among_incorrect": float(
                    ((group["error_label_1.0"] == "self_consistent_error").sum())
                    / max((group["correctness_grade"] == "INCORRECT").sum(), 1)
                ),
                "ie_share_among_incorrect": float(
                    ((group["error_label_1.0"] == "inconsistent_error").sum())
                    / max((group["correctness_grade"] == "INCORRECT").sum(), 1)
                ),
                "precision_ce_vs_strict": float("nan"),
                "recall_ce_vs_strict": float("nan"),
                "f1_ce_vs_strict": float("nan"),
                "precision_incorrect": float("nan"),
                "recall_incorrect": float("nan"),
                "f1_incorrect": float("nan"),
            }
        )

    return pd.DataFrame(rows)


def assign_audit_stratum(frame: pd.DataFrame) -> pd.Series:
    strata = pd.Series(["other"] * len(frame), index=frame.index)

    ce_mask = (frame["correctness_grade"] == "INCORRECT") & (frame["error_label_1.0"] == "self_consistent_error")
    ie_mask = (frame["correctness_grade"] == "INCORRECT") & (frame["error_label_1.0"] == "inconsistent_error")

    uncertain_mask = (
        frame["correctness_unclear"].apply(to_bool)
        | (frame["correctness_grade"] == "NOT_ATTEMPTED")
        | (frame["error_label_1.0"] == "fragile_correct")
        | (
            (frame["correctness_grade"] == "INCORRECT")
            & frame["equivalence_ratio"].astype(float).between(0.90, 1.0, inclusive="left")
        )
    )

    correct_mask = frame["correctness_grade"] == "CORRECT"

    strata.loc[correct_mask] = "correct"
    strata.loc[uncertain_mask] = "uncertain_edge"
    strata.loc[ie_mask] = "ie_like_incorrect"
    strata.loc[ce_mask] = "ce_like_incorrect"

    return strata


def build_audit_sample(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sample_frame = frame.copy()
    sample_frame["audit_stratum"] = assign_audit_stratum(sample_frame)

    target_order = ["correct", "ce_like_incorrect", "ie_like_incorrect", "uncertain_edge"]
    target_base = sample_size // len(target_order)
    target_counts = {stratum: target_base for stratum in target_order}
    remainder = sample_size - target_base * len(target_order)
    for index in range(remainder):
        target_counts[target_order[index]] += 1

    selected_indices: List[int] = []

    for stratum in target_order:
        pool = sample_frame[sample_frame["audit_stratum"] == stratum].index.to_numpy()
        take = min(len(pool), target_counts[stratum])
        if take > 0:
            chosen = rng.choice(pool, size=take, replace=False)
            selected_indices.extend(chosen.tolist())

    if len(selected_indices) < sample_size:
        remaining = sample_frame.loc[~sample_frame.index.isin(selected_indices)].index.to_numpy()
        need = min(sample_size - len(selected_indices), len(remaining))
        if need > 0:
            chosen = rng.choice(remaining, size=need, replace=False)
            selected_indices.extend(chosen.tolist())

    sampled = sample_frame.loc[selected_indices].copy()
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sampled.insert(0, "audit_id", [f"AUDIT_{i+1:03d}" for i in range(len(sampled))])
    sampled["system_binary_correct"] = (sampled["correctness_grade"] == "CORRECT").map({True: "CORRECT", False: "INCORRECT"})

    keep_columns = [
        "audit_id",
        "audit_stratum",
        "question_id",
        "model",
        "dataset_name",
        "dataset_split",
        "question",
        "ground_truth",
        "greedy_answer",
        "correctness_grade",
        "error_label_1.0",
        "equivalence_ratio",
        "semantic_entropy",
        "semantic_entropy_label",
        "stochastic_actual_n",
        "system_binary_correct",
    ]

    for column in keep_columns:
        if column not in sampled.columns:
            sampled[column] = np.nan

    return sampled[keep_columns]


def build_annotation_sheet(audit_sample: pd.DataFrame) -> pd.DataFrame:
    annotations = audit_sample.copy()
    annotations["annotator_a_label"] = ""
    annotations["annotator_a_notes"] = ""
    annotations["annotator_b_label"] = ""
    annotations["annotator_b_notes"] = ""
    annotations["adjudicated_label"] = ""
    annotations["adjudicator_notes"] = ""
    annotations["error_taxonomy"] = ""
    annotations["final_include_for_metrics"] = ""
    return annotations


def normalize_human_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"", "NONE", "NAN"}:
        return None
    if text in {"CORRECT", "TRUE", "1", "YES"}:
        return "CORRECT"
    if text in {"INCORRECT", "FALSE", "0", "NO"}:
        return "INCORRECT"
    if text in {"UNCLEAR", "UNSURE", "AMBIGUOUS"}:
        return "UNCLEAR"
    return text


def cohen_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    if len(labels_a) != len(labels_b) or len(labels_a) < 2:
        return float("nan")
    classes = sorted(set(labels_a) | set(labels_b))
    n = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    p_a = {label: sum(1 for value in labels_a if value == label) / n for label in classes}
    p_b = {label: sum(1 for value in labels_b if value == label) / n for label in classes}
    expected = sum(p_a[label] * p_b[label] for label in classes)

    if abs(1.0 - expected) < 1e-12:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def create_audit_report(annotation_frame: pd.DataFrame, output_path: Path) -> None:
    lines: List[str] = []

    frame = annotation_frame.copy()
    frame["annotator_a_norm"] = frame["annotator_a_label"].map(normalize_human_label)
    frame["annotator_b_norm"] = frame["annotator_b_label"].map(normalize_human_label)
    frame["adjudicated_norm"] = frame["adjudicated_label"].map(normalize_human_label)

    both = frame.dropna(subset=["annotator_a_norm", "annotator_b_norm"])
    both_non_unclear = both[
        both["annotator_a_norm"].isin(["CORRECT", "INCORRECT"])
        & both["annotator_b_norm"].isin(["CORRECT", "INCORRECT"])
    ]

    lines.append("# Human Audit Agreement Report")
    lines.append("")
    lines.append(f"Generated: {utc_now_iso()}")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- Total audit rows: {len(frame)}")
    lines.append(f"- Double-labeled rows (A + B): {len(both)}")
    lines.append(f"- Double-labeled rows usable for binary kappa: {len(both_non_unclear)}")

    if len(both_non_unclear) >= 2:
        agreement = float(np.mean(both_non_unclear["annotator_a_norm"] == both_non_unclear["annotator_b_norm"]))
        kappa = cohen_kappa(
            both_non_unclear["annotator_a_norm"].tolist(),
            both_non_unclear["annotator_b_norm"].tolist(),
        )
        lines.append("")
        lines.append("## Inter-Annotator Reliability")
        lines.append(f"- Raw agreement: {agreement:.4f}")
        lines.append(f"- Cohen's kappa (binary): {kappa:.4f}")
    else:
        lines.append("")
        lines.append("## Inter-Annotator Reliability")
        lines.append("- Pending: not enough completed double annotations for kappa.")

    adjudicated = frame.dropna(subset=["adjudicated_norm"])
    adjudicated_binary = adjudicated[adjudicated["adjudicated_norm"].isin(["CORRECT", "INCORRECT"])]
    if len(adjudicated_binary) > 0:
        system_binary = adjudicated_binary["system_binary_correct"].map(normalize_human_label)
        human_binary = adjudicated_binary["adjudicated_norm"]
        tp = int(np.sum((system_binary == "INCORRECT") & (human_binary == "INCORRECT")))
        fp = int(np.sum((system_binary == "INCORRECT") & (human_binary == "CORRECT")))
        tn = int(np.sum((system_binary == "CORRECT") & (human_binary == "CORRECT")))
        fn = int(np.sum((system_binary == "CORRECT") & (human_binary == "INCORRECT")))
        lines.append("")
        lines.append("## System vs Adjudicated Confusion")
        lines.append(f"- Rows with adjudicated binary label: {len(adjudicated_binary)}")
        lines.append(f"- TP: {tp}")
        lines.append(f"- FP: {fp}")
        lines.append(f"- TN: {tn}")
        lines.append(f"- FN: {fn}")
    else:
        lines.append("")
        lines.append("## System vs Adjudicated Confusion")
        lines.append("- Pending: no adjudicated labels yet.")

    taxonomy_counts = (
        frame["error_taxonomy"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .value_counts()
    )
    lines.append("")
    lines.append("## Error Taxonomy Counts")
    if taxonomy_counts.empty:
        lines.append("- Pending: no taxonomy tags yet.")
    else:
        for label, count in taxonomy_counts.items():
            lines.append(f"- {label}: {int(count)}")

    lines.append("")
    lines.append("## Annotation Rubric")
    lines.append("- CORRECT: answer is fully supported by ground truth; no material factual error.")
    lines.append("- INCORRECT: answer contains factual error, contradiction, or unsupported claim.")
    lines.append("- UNCLEAR: insufficient information to decide or ambiguous ground truth.")
    lines.append("- `error_taxonomy` examples: fabrication, contradiction, omission, entity_swap, numeric_error, reasoning_error.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_figures(
    frame: pd.DataFrame,
    group_metrics: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    answered_frame: pd.DataFrame,
    figures_dir: Path,
) -> None:
    ensure_dir(figures_dir)

    def font(size: int):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size=size)
        except OSError:
            return ImageFont.load_default()

    def canvas(title: str, width: int = 1600, height: int = 900) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((40, 30), title, fill=(20, 20, 20), font=font(36))
        return image, draw

    def draw_plot_box(draw: ImageDraw.ImageDraw, left: int, top: int, right: int, bottom: int) -> None:
        draw.rectangle([left, top, right, bottom], outline=(80, 80, 80), width=2)

    def line_points(xs: Sequence[float], ys: Sequence[float], left: int, top: int, right: int, bottom: int) -> List[Tuple[float, float]]:
        if not xs or not ys:
            return []
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if abs(x_max - x_min) <= 1e-12:
            x_max = x_min + 1.0
        if abs(y_max - y_min) <= 1e-12:
            y_max = y_min + 1.0
        mapped: List[Tuple[float, float]] = []
        for x_value, y_value in zip(xs, ys):
            x_ratio = (x_value - x_min) / (x_max - x_min)
            y_ratio = (y_value - y_min) / (y_max - y_min)
            x_pixel = left + x_ratio * (right - left)
            y_pixel = bottom - y_ratio * (bottom - top)
            mapped.append((x_pixel, y_pixel))
        return mapped

    model_rows = group_metrics[group_metrics["group_type"] == "model"].copy().sort_values(by="accuracy", ascending=False)
    if not model_rows.empty:
        image, draw = canvas("Model Accuracy with 95% Bootstrap CI")
        left, top, right, bottom = 100, 140, 1520, 760
        draw_plot_box(draw, left, top, right, bottom)
        model_names = model_rows["model"].tolist()
        values = model_rows["accuracy"].to_numpy(dtype=float)
        lows = model_rows["accuracy_ci_low"].to_numpy(dtype=float)
        highs = model_rows["accuracy_ci_high"].to_numpy(dtype=float)
        count = len(model_names)
        bar_width = (right - left - 40) / max(count, 1)
        for index, (name, value, low, high) in enumerate(zip(model_names, values, lows, highs)):
            x0 = left + 20 + index * bar_width
            x1 = x0 + bar_width * 0.7
            y1 = bottom
            y0 = bottom - value * (bottom - top)
            draw.rectangle([x0, y0, x1, y1], fill=(76, 120, 168), outline=(45, 72, 100))
            ci_low_y = bottom - low * (bottom - top)
            ci_high_y = bottom - high * (bottom - top)
            x_mid = (x0 + x1) / 2.0
            draw.line([x_mid, ci_low_y, x_mid, ci_high_y], fill=(20, 20, 20), width=3)
            draw.line([x_mid - 6, ci_low_y, x_mid + 6, ci_low_y], fill=(20, 20, 20), width=3)
            draw.line([x_mid - 6, ci_high_y, x_mid + 6, ci_high_y], fill=(20, 20, 20), width=3)
            short_name = name if len(name) <= 20 else name[:20] + "…"
            draw.text((x0, bottom + 12), short_name, fill=(20, 20, 20), font=font(16))
            draw.text((x0, y0 - 28), f"{value:.3f}", fill=(20, 20, 20), font=font(15))
        image.save(figures_dir / "model_accuracy_with_ci.png")

    label_counts = frame.groupby(["model", "error_label_1.0"]).size().reset_index(name="count")
    if not label_counts.empty:
        image, draw = canvas("Error Composition by Model (Threshold 1.0)")
        left, top, right, bottom = 100, 140, 1520, 760
        draw_plot_box(draw, left, top, right, bottom)
        pivot = label_counts.pivot(index="model", columns="error_label_1.0", values="count").fillna(0.0)
        proportions = pivot.div(pivot.sum(axis=1), axis=0)
        order = [
            "reliably_correct",
            "fragile_correct",
            "self_consistent_error",
            "inconsistent_error",
            "not_attempted",
        ]
        palette = {
            "reliably_correct": (84, 162, 75),
            "fragile_correct": (242, 185, 65),
            "self_consistent_error": (76, 120, 168),
            "inconsistent_error": (229, 87, 86),
            "not_attempted": (153, 153, 153),
        }
        model_names = proportions.index.tolist()
        bar_width = (right - left - 40) / max(len(model_names), 1)
        for index, model_name in enumerate(model_names):
            x0 = left + 20 + index * bar_width
            x1 = x0 + bar_width * 0.7
            y_cursor = bottom
            for label in order:
                if label not in proportions.columns:
                    continue
                share = float(proportions.loc[model_name, label])
                height = share * (bottom - top)
                y_next = y_cursor - height
                draw.rectangle([x0, y_next, x1, y_cursor], fill=palette[label], outline=(255, 255, 255))
                y_cursor = y_next
            short_name = model_name if len(model_name) <= 20 else model_name[:20] + "…"
            draw.text((x0, bottom + 12), short_name, fill=(20, 20, 20), font=font(16))
        legend_x = 1120
        legend_y = 170
        for index, label in enumerate(order):
            draw.rectangle([legend_x, legend_y + index * 36, legend_x + 24, legend_y + index * 36 + 24], fill=palette[label])
            draw.text((legend_x + 34, legend_y + index * 36), label, fill=(20, 20, 20), font=font(18))
        image.save(figures_dir / "error_composition_by_model.png")

    entropy_rows = answered_frame.dropna(subset=["semantic_entropy"]).copy()
    if not entropy_rows.empty:
        image, draw = canvas("Semantic Entropy Distribution by Outcome")
        left, top, right, bottom = 100, 140, 1520, 760
        draw_plot_box(draw, left, top, right, bottom)
        correct_values = entropy_rows[entropy_rows["y_error"] == 0]["semantic_entropy"].to_numpy(dtype=float)
        incorrect_values = entropy_rows[entropy_rows["y_error"] == 1]["semantic_entropy"].to_numpy(dtype=float)
        all_values = np.concatenate([correct_values, incorrect_values]) if len(incorrect_values) else correct_values
        if len(all_values) > 0:
            bins = np.linspace(float(np.min(all_values)), float(np.max(all_values)), 25)
            if len(np.unique(bins)) > 1:
                correct_hist, edges = np.histogram(correct_values, bins=bins, density=True)
                incorrect_hist, _ = np.histogram(incorrect_values, bins=bins, density=True)
                centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()
                max_y = max(float(np.max(correct_hist)) if len(correct_hist) else 0.0, float(np.max(incorrect_hist)) if len(incorrect_hist) else 0.0, 1e-6)
                c_points = line_points(centers, (correct_hist / max_y).tolist(), left, top, right, bottom)
                i_points = line_points(centers, (incorrect_hist / max_y).tolist(), left, top, right, bottom)
                if len(c_points) >= 2:
                    draw.line(c_points, fill=(84, 162, 75), width=4)
                if len(i_points) >= 2:
                    draw.line(i_points, fill=(229, 87, 86), width=4)
        draw.text((120, 780), "Green: CORRECT density (normalized)", fill=(84, 162, 75), font=font(20))
        draw.text((620, 780), "Red: INCORRECT density (normalized)", fill=(229, 87, 86), font=font(20))
        image.save(figures_dir / "semantic_entropy_distribution.png")

    threshold_rows = threshold_sensitivity[threshold_sensitivity["analysis_type"] == "equivalence_threshold"].copy()
    if not threshold_rows.empty:
        threshold_rows = threshold_rows.sort_values(by="value")
        image, draw = canvas("Threshold Sensitivity of CE/IE Split")
        left, top, right, bottom = 100, 140, 1520, 760
        draw_plot_box(draw, left, top, right, bottom)
        xs = threshold_rows["value"].astype(float).tolist()
        ce_values = threshold_rows["ce_share_among_incorrect"].astype(float).tolist()
        ie_values = threshold_rows["ie_share_among_incorrect"].astype(float).tolist()
        ce_points = line_points(xs, ce_values, left, top, right, bottom)
        ie_points = line_points(xs, ie_values, left, top, right, bottom)
        if len(ce_points) >= 2:
            draw.line(ce_points, fill=(76, 120, 168), width=4)
        if len(ie_points) >= 2:
            draw.line(ie_points, fill=(245, 133, 24), width=4)
        for point, threshold in zip(ce_points, xs):
            draw.ellipse([point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5], fill=(76, 120, 168))
            draw.text((point[0] - 12, bottom + 14), f"{threshold:.1f}", fill=(20, 20, 20), font=font(16))
        for point in ie_points:
            draw.ellipse([point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5], fill=(245, 133, 24))
        draw.text((120, 780), "Blue: CE share among incorrect", fill=(76, 120, 168), font=font(20))
        draw.text((620, 780), "Orange: IE share among incorrect", fill=(245, 133, 24), font=font(20))
        image.save(figures_dir / "threshold_sensitivity_ce_ie.png")

    if len(answered_frame) > 0:
        image, draw = canvas("ROC and PR Curves: Disagreement vs Semantic Entropy")
        left_roc, top_roc, right_roc, bottom_roc = 80, 140, 770, 760
        left_pr, top_pr, right_pr, bottom_pr = 850, 140, 1540, 760
        draw_plot_box(draw, left_roc, top_roc, right_roc, bottom_roc)
        draw_plot_box(draw, left_pr, top_pr, right_pr, bottom_pr)
        y_true = answered_frame["y_error"].to_numpy(dtype=int)
        disagreement = answered_frame["score_disagreement"].to_numpy(dtype=float)
        entropy_df = answered_frame.dropna(subset=["score_entropy"])

        d_fpr, d_tpr = roc_curve_points(y_true, disagreement)
        d_roc_points = line_points(d_fpr.tolist(), d_tpr.tolist(), left_roc, top_roc, right_roc, bottom_roc)
        if len(d_roc_points) >= 2:
            draw.line(d_roc_points, fill=(76, 120, 168), width=4)
        draw.line([(left_roc, bottom_roc), (right_roc, top_roc)], fill=(160, 160, 160), width=2)

        d_recall, d_precision = pr_curve_points(y_true, disagreement)
        d_pr_points = line_points(d_recall.tolist(), d_precision.tolist(), left_pr, top_pr, right_pr, bottom_pr)
        if len(d_pr_points) >= 2:
            draw.line(d_pr_points, fill=(76, 120, 168), width=4)

        legend_lines = [
            f"Disagreement AUROC={wilcoxon_auc(y_true, disagreement):.3f}",
            f"Disagreement PR-AUC={average_precision(y_true, disagreement):.3f}",
        ]

        if len(entropy_df) > 0 and entropy_df["y_error"].nunique() > 1:
            entropy_y = entropy_df["y_error"].to_numpy(dtype=int)
            entropy_s = entropy_df["score_entropy"].to_numpy(dtype=float)
            e_fpr, e_tpr = roc_curve_points(entropy_y, entropy_s)
            e_roc_points = line_points(e_fpr.tolist(), e_tpr.tolist(), left_roc, top_roc, right_roc, bottom_roc)
            if len(e_roc_points) >= 2:
                draw.line(e_roc_points, fill=(229, 87, 86), width=4)
            e_recall, e_precision = pr_curve_points(entropy_y, entropy_s)
            e_pr_points = line_points(e_recall.tolist(), e_precision.tolist(), left_pr, top_pr, right_pr, bottom_pr)
            if len(e_pr_points) >= 2:
                draw.line(e_pr_points, fill=(229, 87, 86), width=4)
            legend_lines.append(f"Entropy AUROC={wilcoxon_auc(entropy_y, entropy_s):.3f}")
            legend_lines.append(f"Entropy PR-AUC={average_precision(entropy_y, entropy_s):.3f}")

        draw.text((110, 90), "ROC (left)", fill=(20, 20, 20), font=font(24))
        draw.text((880, 90), "PR (right)", fill=(20, 20, 20), font=font(24))
        for index, text in enumerate(legend_lines):
            draw.text((90, 790 + index * 26), text, fill=(20, 20, 20), font=font(20))
        image.save(figures_dir / "roc_pr_comparison.png")


def build_writeup(
    output_path: Path,
    manifest: Dict[str, Any],
    aggregate_metrics: pd.DataFrame,
    group_metrics: pd.DataFrame,
    hypothesis_tests: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    audit_sample: pd.DataFrame,
) -> None:
    def get_metric(name: str, subset: str = "all_rows") -> Optional[Mapping[str, Any]]:
        rows = aggregate_metrics[(aggregate_metrics["metric_name"] == name) & (aggregate_metrics["subset"] == subset)]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    accuracy = get_metric("accuracy")
    incorrect_rate = get_metric("incorrect_rate")
    ce_share = get_metric("ce_share_among_incorrect", subset="incorrect_only")
    ie_share = get_metric("ie_share_among_incorrect", subset="incorrect_only")
    disagreement_auc = get_metric("disagreement_prob_auroc", subset="answered_rows")
    entropy_auc = get_metric("semantic_entropy_prob_auroc", subset="answered_rows")

    model_table = group_metrics[group_metrics["group_type"] == "model"].sort_values(by="accuracy", ascending=False)

    lines: List[str] = []
    lines.append("# Thesis Results Writeup (v2 Hybrid Semantic Evaluation)")
    lines.append("")
    lines.append("## Data Freeze and Reproducibility")
    lines.append(f"- Input file: `{manifest['input_file']}`")
    lines.append(f"- SHA256: `{manifest['sha256']}`")
    lines.append(f"- Rows: {manifest['validation']['row_count']}")
    lines.append(f"- Questions: {manifest['validation']['question_count']}")
    lines.append(f"- Models: {manifest['validation']['model_count']}")
    lines.append(f"- Generated (UTC): {manifest['generated_at_utc']}")

    lines.append("")
    lines.append("## RQ1 — Prevalence of Hallucination/Error Types")
    if accuracy and incorrect_rate and ce_share and ie_share:
        lines.append(
            f"- Overall accuracy: {accuracy['value']:.3f} (95% CI {accuracy['ci_low']:.3f}–{accuracy['ci_high']:.3f})."
        )
        lines.append(
            f"- Overall incorrect rate: {incorrect_rate['value']:.3f} (95% CI {incorrect_rate['ci_low']:.3f}–{incorrect_rate['ci_high']:.3f})."
        )
        lines.append(
            f"- Among incorrect responses: CE share {ce_share['value']:.3f} (95% CI {ce_share['ci_low']:.3f}–{ce_share['ci_high']:.3f}), IE share {ie_share['value']:.3f} (95% CI {ie_share['ci_low']:.3f}–{ie_share['ci_high']:.3f})."
        )

    if not model_table.empty:
        lines.append("- Model-level accuracy ranking:")
        for _, row in model_table.iterrows():
            lines.append(
                f"  - {row['model']}: {row['accuracy']:.3f} (95% CI {row['accuracy_ci_low']:.3f}–{row['accuracy_ci_high']:.3f})"
            )

    lines.append("")
    lines.append("## RQ2 — Black-Box Detectability via Consistency/Entropy")
    if disagreement_auc:
        lines.append(
            f"- Disagreement-derived score AUROC: {disagreement_auc['value']:.3f} (95% CI {disagreement_auc['ci_low']:.3f}–{disagreement_auc['ci_high']:.3f})."
        )
    if entropy_auc:
        lines.append(
            f"- Semantic-entropy score AUROC: {entropy_auc['value']:.3f} (95% CI {entropy_auc['ci_low']:.3f}–{entropy_auc['ci_high']:.3f})."
        )
    lines.append("- PR-AUC is computed with step-wise average precision (non-inflated interpolation).")

    lines.append("")
    lines.append("## RQ3 — Cross-Model Comparative Inference")
    if not hypothesis_tests.empty:
        significant = hypothesis_tests[hypothesis_tests["significant_after_holm"] == True]
        lines.append(
            f"- Pairwise model comparisons run with exact McNemar and Holm-Bonferroni correction ({len(hypothesis_tests)} pairs)."
        )
        lines.append(f"- Significant pairs after correction: {len(significant)}.")
        if len(significant) > 0:
            top = significant.head(5)
            lines.append("- Top significant differences (accuracy diff A−B):")
            for _, row in top.iterrows():
                lines.append(
                    f"  - {row['model_a']} vs {row['model_b']}: {row['accuracy_diff_a_minus_b']:.3f} (95% CI {row['accuracy_diff_ci_low']:.3f}–{row['accuracy_diff_ci_high']:.3f}), adj-p={row['holm_adjusted_p']:.4g}."
                )

    lines.append("")
    lines.append("## RQ4 — Pattern and Robustness Analysis")
    threshold_rows = threshold_sensitivity[threshold_sensitivity["analysis_type"] == "equivalence_threshold"]
    if not threshold_rows.empty:
        lines.append("- Threshold sensitivity (0.7→1.0) was computed for CE/IE split and CE-detection fidelity vs strict CE labels (`error_label_1.0`).")
        best_f1 = threshold_rows.sort_values(by="f1_ce_vs_strict", ascending=False).head(1)
        if not best_f1.empty:
            row = best_f1.iloc[0]
            lines.append(
                f"- Best CE-detection F1 in sweep: threshold {row['value']} with F1={row['f1_ce_vs_strict']:.3f} (precision={row['precision_ce_vs_strict']:.3f}, recall={row['recall_ce_vs_strict']:.3f})."
            )

    entropy_bins = threshold_sensitivity[threshold_sensitivity["analysis_type"] == "semantic_entropy_bin"]
    if not entropy_bins.empty:
        lines.append("- Entropy-bin stratification indicates how incorrect-rate concentration shifts across uncertainty quantiles.")

    lines.append("")
    lines.append("## Human Audit Design and Status")
    lines.append(f"- Audit sample size: {len(audit_sample)} rows (target 150–200; configured to 200).")
    stratum_counts = audit_sample["audit_stratum"].value_counts().to_dict()
    for stratum, count in stratum_counts.items():
        lines.append(f"- {stratum}: {count}")
    lines.append("- Double annotation + adjudication template is ready in `audit_annotations.csv`.")
    lines.append("- Agreement report auto-updates from annotation sheet once labels are filled.")

    lines.append("")
    lines.append("## Threats to Validity")
    lines.append("- Dataset metadata sparsity: `category`/`task_type` are missing, so stratification uses available proxies.")
    lines.append("- Sample depth is fixed at `stochastic_actual_n=10`, limiting depth sensitivity conclusions.")
    lines.append("- LLM-as-judge outputs can carry bias; audit is used as a calibration check.")

    lines.append("")
    lines.append("## Reproducibility Checklist")
    lines.append("- Input file hash is recorded in `manifest.json`.")
    lines.append("- Bootstrap seed and iteration count are recorded in `bootstrap_ci.json`.")
    lines.append("- Deterministic audit sampling uses a fixed RNG seed.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(output_path: Path, input_path: Path, validation: Dict[str, Any]) -> Dict[str, Any]:
    manifest = {
        "generated_at_utc": utc_now_iso(),
        "input_file": str(input_path.resolve()),
        "sha256": sha256_of_file(input_path),
        "validation": validation,
    }
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_or_create_annotations(audit_annotations_path: Path, audit_template: pd.DataFrame) -> pd.DataFrame:
    if audit_annotations_path.exists():
        return pd.read_csv(audit_annotations_path)
    return audit_template


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_root = Path(args.output_dir)
    metrics_dir = output_root / "metrics"
    stats_dir = output_root / "stats"
    ablation_dir = output_root / "ablation"
    audit_dir = output_root / "audit"
    figures_dir = output_root / "figures"

    for directory in [output_root, metrics_dir, stats_dir, ablation_dir, audit_dir, figures_dir]:
        ensure_dir(directory)

    records = read_jsonl(input_path)
    validation = validate_dataset(records)
    if validation["invalid_correctness_values"]:
        raise ValueError(f"Invalid correctness_grade values: {validation['invalid_correctness_values']}")
    if validation["invalid_error_label_values"]:
        raise ValueError(f"Invalid error_label_1.0 values: {validation['invalid_error_label_values']}")
    if validation["duplicate_question_model_rows"] > 0:
        raise ValueError(
            "Duplicate (question_id, model) rows detected: "
            f"{validation['duplicate_question_model_rows']}"
        )
    frame = pd.DataFrame(records)

    for required in ["correctness_unclear", "semantic_entropy", "dataset_name", "dataset_split"]:
        if required not in frame.columns:
            frame[required] = np.nan

    manifest = write_manifest(output_root / "manifest.json", input_path, validation)

    aggregate_metrics, bootstrap_json, answered = compute_aggregate_metrics(
        frame,
        n_bootstrap=args.bootstrap_iters,
        seed=args.seed,
    )

    group_metrics = compute_group_metrics(
        frame,
        answered,
        group_bootstrap_iters=args.group_bootstrap_iters,
        seed=args.seed,
        summary_json=bootstrap_json,
    )

    hypothesis_tests = compute_pairwise_hypothesis_tests(
        frame,
        n_bootstrap=args.bootstrap_iters,
        seed=args.seed + 4000,
        alpha=args.alpha,
    )

    threshold_sensitivity = compute_threshold_sensitivity(frame)

    audit_sample = build_audit_sample(frame, sample_size=args.audit_size, seed=args.seed + 5000)
    audit_template = build_annotation_sheet(audit_sample)

    aggregate_metrics.to_csv(metrics_dir / "aggregate_metrics.csv", index=False)
    group_metrics.to_csv(metrics_dir / "group_metrics_by_model.csv", index=False)
    hypothesis_tests.to_csv(stats_dir / "hypothesis_tests.csv", index=False)
    (stats_dir / "bootstrap_ci.json").write_text(json.dumps(bootstrap_json, indent=2) + "\n", encoding="utf-8")
    threshold_sensitivity.to_csv(ablation_dir / "threshold_sensitivity.csv", index=False)
    audit_sample.to_csv(audit_dir / "audit_sample_200.csv", index=False)

    existing_or_template = load_or_create_annotations(audit_dir / "audit_annotations.csv", audit_template)
    existing_or_template.to_csv(audit_dir / "audit_annotations.csv", index=False)

    create_audit_report(existing_or_template, audit_dir / "audit_agreement_report.md")

    generate_figures(frame, group_metrics, threshold_sensitivity, answered, figures_dir)

    build_writeup(
        output_root / "thesis_results_writeup.md",
        manifest,
        aggregate_metrics,
        group_metrics,
        hypothesis_tests,
        threshold_sensitivity,
        audit_sample,
    )

    summary = {
        "generated_at_utc": utc_now_iso(),
        "output_root": str(output_root.resolve()),
        "files": [
            str((metrics_dir / "aggregate_metrics.csv").resolve()),
            str((metrics_dir / "group_metrics_by_model.csv").resolve()),
            str((stats_dir / "hypothesis_tests.csv").resolve()),
            str((stats_dir / "bootstrap_ci.json").resolve()),
            str((ablation_dir / "threshold_sensitivity.csv").resolve()),
            str((audit_dir / "audit_sample_200.csv").resolve()),
            str((audit_dir / "audit_annotations.csv").resolve()),
            str((audit_dir / "audit_agreement_report.md").resolve()),
            str((output_root / "thesis_results_writeup.md").resolve()),
        ],
    }
    (output_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build thesis-grade analysis package for evaluated JSONL data")
    parser.add_argument(
        "--input",
        type=str,
        default="data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl",
        help="Path to evaluated analysis-ready JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/v2_thesis",
        help="Output directory root for thesis artifacts",
    )
    parser.add_argument("--bootstrap-iters", type=int, default=2000, help="Bootstrap iterations for CIs")
    parser.add_argument(
        "--group-bootstrap-iters",
        type=int,
        default=400,
        help="Bootstrap iterations for grouped rate confidence intervals",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha for significance correction")
    parser.add_argument("--audit-size", type=int, default=200, help="Audit sample size (150-200 recommended)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
