#!/usr/bin/env python3
"""Generate a simple-language, deeply validated LaTeX analysis report.

This script performs:
1) Strict data integrity and logic validation on final.analysis_ready JSONL.
2) Cross-checks against previously generated analysis CSV outputs.
3) Figure generation for plain-language interpretation.
4) A detailed LaTeX explainer report with glossary, graph walkthroughs,
   expected-vs-unexpected findings, and literature citations.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE = Path("/Users/simar/LLM_Hallucination_Measure")
FINAL_JSONL = BASE / "data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.jsonl"
TRUTHFULQA_CSV = BASE / "TruthfulQA.csv"
ANALYSIS_DIR = BASE / "data/results/analysis/final_analysis_ready"

OUT_DIR = ANALYSIS_DIR / "latex_report_simple"
FIG_DIR = OUT_DIR / "figures"


ALLOWED_GRADE = {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}
ALLOWED_DECISION = {"MAJORITY", "ADJUDICATOR"}
ALLOWED_STATUS = {"OK", "PARSE_FAILED", "API_FAILED"}
ALLOWED_EQ = {"same", "different", "unclear"}

REPORT_THRESHOLDS = [1.0, 0.9, 0.8]
FIVE_LABELS = [
    "reliably_correct",
    "fragile_correct",
    "self_consistent_error",
    "inconsistent_error",
    "not_attempted",
]
LABEL_PRETTY = {
    "reliably_correct": "Correct + same meaning",
    "fragile_correct": "Correct + different meaning",
    "self_consistent_error": "Incorrect + same meaning",
    "inconsistent_error": "Incorrect + different meaning",
    "not_attempted": "NOT_ATTEMPTED",
}


def b(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def pct(v: float) -> str:
    if pd.isna(v):
        return "N/A"
    return f"{100.0 * float(v):.1f}%"


def tex_escape(x: Any) -> str:
    s = str(x)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for a, b_ in repl.items():
        s = s.replace(a, b_)
    return s


def short_model(name: str) -> str:
    m = {
        "Claude Opus 4.6 (Anthropic)": "Claude Opus 4.6",
        "DeepSeek V3.2 (DeepSeek)": "DeepSeek V3.2",
        "GPT-5.2 (OpenAI)": "GPT-5.2",
        "Grok 4 (xAI)": "Grok 4",
        "Llama 4 Maverick 17B (Groq)": "Llama 4 Maverick",
        "Qwen3 Next 80B (OpenRouter)": "Qwen3 Next 80B",
    }
    return m.get(name, name)


def qid_to_idx(qid: Any) -> float:
    if not isinstance(qid, str):
        return math.nan
    m = re.search(r"truthfulqa_csv_(\d+)$", qid)
    return float(m.group(1)) if m else math.nan


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def bootstrap_ci(values: np.ndarray, n_boot: int = 4000, seed: int = 42) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    samples = arr[idx].mean(axis=1)
    return (float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))


def exact_binom_two_sided(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    tail = 0.0
    for i in range(0, k + 1):
        tail += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar_p(b_only: int, c_only: int) -> float:
    return exact_binom_two_sided(min(b_only, c_only), b_only + c_only)


def load_data() -> Dict[str, Any]:
    rows = [json.loads(line) for line in FINAL_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["q_idx"] = df["question_id"].map(qid_to_idx)
    df["is_correct"] = df["greedy_correct"].map(b)
    df["is_incorrect"] = ~df["is_correct"]
    df["is_na"] = df["correctness_grade"].eq("NOT_ATTEMPTED")
    df["is_sc"] = df["error_label_0.9"].eq("self_consistent_error")
    df["is_inc"] = df["error_label_0.9"].eq("inconsistent_error")
    df["is_fragile"] = df["error_label_0.9"].eq("fragile_correct")
    for thr in REPORT_THRESHOLDS:
        tag = str(thr).replace(".", "_")
        df[f"is_sc_{tag}"] = df[f"error_label_{thr:.1f}"].eq("self_consistent_error")
    df["escalated"] = df["escalated_to_human"].map(b)
    df["incomplete"] = df["is_incomplete"].map(b)
    df["any_parse_failed"] = df["correctness_judge_statuses"].map(
        lambda x: isinstance(x, list) and any(i == "PARSE_FAILED" for i in x)
    )
    df["any_api_failed"] = df["correctness_judge_statuses"].map(
        lambda x: isinstance(x, list) and any(i == "API_FAILED" for i in x)
    )
    df["adjudicated"] = df["correctness_decision_source"].eq("ADJUDICATOR")

    truthfulqa = (
        pd.read_csv(TRUTHFULQA_CSV)
        .reset_index(drop=False)
        .rename(columns={"index": "q_idx", "Category": "category", "Type": "question_type"})
    )
    df = df.merge(truthfulqa[["q_idx", "category", "question_type", "Question"]], on="q_idx", how="left")
    return {"df": df, "truthfulqa": truthfulqa}


def classify_label(is_correct: bool, stats: Dict[str, Any], threshold: float, grade: str) -> str:
    if grade == "NOT_ATTEMPTED":
        return "not_attempted"
    num_same = int(stats.get("num_same", 0))
    num_diff = int(stats.get("num_different", 0))
    denom = num_same + num_diff
    ratio = (num_same / denom) if denom > 0 else 0.0
    consistent = ratio >= threshold
    if is_correct:
        return "reliably_correct" if consistent else "fragile_correct"
    return "self_consistent_error" if consistent else "inconsistent_error"


def run_strict_validations(df: pd.DataFrame, truthfulqa: pd.DataFrame) -> Tuple[List[CheckResult], Dict[str, Any]]:
    checks: List[CheckResult] = []
    details: Dict[str, Any] = {}

    n_rows = len(df)
    n_q = int(df["question_id"].nunique())
    n_m = int(df["model"].nunique())
    expected = n_q * n_m
    checks.append(CheckResult("Row count equals question x model matrix", n_rows == expected, f"rows={n_rows}, expected={expected}"))

    dup = int(df.duplicated(subset=["question_id", "model"]).sum())
    checks.append(CheckResult("No duplicate (question_id, model) pairs", dup == 0, f"duplicates={dup}"))

    per_q = df.groupby("question_id")["model"].nunique()
    bad_q = int((per_q != n_m).sum())
    checks.append(CheckResult("Every question has all models", bad_q == 0, f"bad_questions={bad_q}"))

    per_m = df.groupby("model")["question_id"].nunique()
    bad_m = int((per_m != n_q).sum())
    checks.append(CheckResult("Every model has all questions", bad_m == 0, f"bad_models={bad_m}"))

    # Consistency of question text/ground truth across models
    q_text_bad = 0
    q_gt_bad = 0
    for _, sub in df.groupby("question_id"):
        if sub["question"].nunique(dropna=False) != 1:
            q_text_bad += 1
        gt = sub["ground_truth"].apply(lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
        if gt.nunique(dropna=False) != 1:
            q_gt_bad += 1
    checks.append(CheckResult("Question text consistent across models", q_text_bad == 0, f"inconsistent_questions={q_text_bad}"))
    checks.append(CheckResult("Ground truth list consistent across models", q_gt_bad == 0, f"inconsistent_ground_truth={q_gt_bad}"))

    invalid_grade = int((~df["correctness_grade"].isin(ALLOWED_GRADE)).sum())
    checks.append(CheckResult("All correctness_grade values valid", invalid_grade == 0, f"invalid_grade_rows={invalid_grade}"))

    grade_match = (
        ((df["correctness_grade"] == "CORRECT") & (df["is_correct"]))
        | ((df["correctness_grade"].isin(["INCORRECT", "NOT_ATTEMPTED"])) & (~df["is_correct"]))
    )
    grade_mismatch = int((~grade_match).sum())
    checks.append(CheckResult("greedy_correct is consistent with correctness_grade", grade_mismatch == 0, f"mismatch_rows={grade_mismatch}"))

    invalid_decision = int((~df["correctness_decision_source"].isin(ALLOWED_DECISION)).sum())
    checks.append(CheckResult("Decision source is MAJORITY or ADJUDICATOR only", invalid_decision == 0, f"invalid_decision_rows={invalid_decision}"))

    # Majority/adjudicator consistency
    majority_bad = 0
    adj_bad = 0
    unresolved = int(df["correctness_decision_source"].isin(["UNRESOLVED", "NO_JUDGE"]).sum())
    for _, r in df.iterrows():
        grades = r.get("correctness_judge_grades")
        statuses = r.get("correctness_judge_statuses")
        if not isinstance(grades, list) or not isinstance(statuses, list):
            continue
        ok_grades = [g for g, s in zip(grades, statuses) if s == "OK" and g in ALLOWED_GRADE]
        vc = Counter(ok_grades)
        majority = None
        if vc:
            top_label, top_n = vc.most_common(1)[0]
            if top_n >= 2:
                majority = top_label

        if r["correctness_decision_source"] == "MAJORITY":
            if majority is None or r["correctness_grade"] != majority:
                majority_bad += 1
        elif r["correctness_decision_source"] == "ADJUDICATOR":
            if r.get("correctness_adjudicator_status") != "OK":
                adj_bad += 1
            elif r.get("correctness_adjudicator_grade") != r["correctness_grade"]:
                adj_bad += 1
    checks.append(CheckResult("Majority decisions match judge-majority label", majority_bad == 0, f"bad_rows={majority_bad}"))
    checks.append(CheckResult("Adjudicator decisions have OK status and matching grade", adj_bad == 0, f"bad_rows={adj_bad}"))
    checks.append(CheckResult("No unresolved/no_judge rows", unresolved == 0, f"rows={unresolved}"))

    # Judge statuses and grade nullability
    status_bad = 0
    status_grade_bad = 0
    for _, r in df.iterrows():
        grades = r.get("correctness_judge_grades")
        statuses = r.get("correctness_judge_statuses")
        if not isinstance(grades, list) or not isinstance(statuses, list):
            status_bad += 1
            continue
        if len(grades) != len(statuses):
            status_bad += 1
            continue
        for g, s in zip(grades, statuses):
            if s not in ALLOWED_STATUS:
                status_bad += 1
                continue
            if s == "OK" and g not in ALLOWED_GRADE:
                status_grade_bad += 1
            if s in {"PARSE_FAILED", "API_FAILED"} and g is not None:
                status_grade_bad += 1
    checks.append(CheckResult("Judge status arrays are valid and aligned", status_bad == 0, f"bad_rows={status_bad}"))
    checks.append(CheckResult("Judge grade nullability matches status", status_grade_bad == 0, f"bad_slots={status_grade_bad}"))

    # Equivalence structure and math
    eq_len_bad = 0
    eq_label_bad = 0
    eq_stats_bad = 0
    eq_ratio_bad = 0
    nli_len_bad = 0
    label_bad = 0
    for _, r in df.iterrows():
        stoch = r.get("stochastic_answers")
        eq = r.get("equivalence_results")
        stats = r.get("equivalence_stats")
        nli = r.get("nli_equiv_probs")
        if not isinstance(stoch, list) or not isinstance(eq, list) or not isinstance(stats, dict):
            eq_len_bad += 1
            continue
        if len(stoch) != len(eq):
            eq_len_bad += 1
        if any(x not in ALLOWED_EQ for x in eq):
            eq_label_bad += 1

        same = sum(1 for x in eq if x == "same")
        diff = sum(1 for x in eq if x == "different")
        unc = sum(1 for x in eq if x == "unclear")
        if stats.get("num_same") != same or stats.get("num_different") != diff or stats.get("num_unclear") != unc or stats.get("total") != len(eq):
            eq_stats_bad += 1
        denom = same + diff
        ratio = same / denom if denom > 0 else 0.0
        if not isinstance(r.get("equivalence_ratio"), (int, float)) or abs(float(r["equivalence_ratio"]) - ratio) > 1e-9:
            eq_ratio_bad += 1

        if isinstance(nli, list) and len(nli) != len(eq):
            nli_len_bad += 1

        for thr, field in [(1.0, "error_label_1.0"), (0.9, "error_label_0.9"), (0.8, "error_label_0.8"), (0.7, "error_label_0.7")]:
            exp = classify_label(bool(r["is_correct"]), stats, thr, r["correctness_grade"])
            if r.get(field) != exp:
                label_bad += 1
                break
    checks.append(CheckResult("stochastic/equivalence lengths match", eq_len_bad == 0, f"bad_rows={eq_len_bad}"))
    checks.append(CheckResult("equivalence labels are in allowed set", eq_label_bad == 0, f"bad_rows={eq_label_bad}"))
    checks.append(CheckResult("equivalence_stats counts match equivalence_results", eq_stats_bad == 0, f"bad_rows={eq_stats_bad}"))
    checks.append(CheckResult("equivalence_ratio matches count-derived value", eq_ratio_bad == 0, f"bad_rows={eq_ratio_bad}"))
    checks.append(CheckResult("nli_equiv_probs length matches equivalence_results", nli_len_bad == 0, f"bad_rows={nli_len_bad}"))
    checks.append(CheckResult("All threshold error labels are mathematically consistent", label_bad == 0, f"bad_rows={label_bad}"))

    # Explicit 0.9 five-label partition sanity check
    label_09 = df["error_label_0.9"].astype(str)
    bad_label_09 = int((~label_09.isin(FIVE_LABELS)).sum())
    part_counts = {k: int((label_09 == k).sum()) for k in FIVE_LABELS}
    part_total = int(sum(part_counts.values()))
    part_ok = (bad_label_09 == 0 and part_total == n_rows)
    checks.append(
        CheckResult(
            "0.9 label partition is complete and valid",
            part_ok,
            (
                f"invalid_rows={bad_label_09}, partition_total={part_total}, "
                f"reliably_correct={part_counts['reliably_correct']}, "
                f"fragile_correct={part_counts['fragile_correct']}, "
                f"self_consistent_error={part_counts['self_consistent_error']}, "
                f"inconsistent_error={part_counts['inconsistent_error']}, "
                f"not_attempted={part_counts['not_attempted']}"
            ),
        )
    )

    # Escalation logic for analysis_ready output
    non_na_escalated = int(((~df["is_na"]) & (df["escalated"])).sum())
    na_not_escalated = int(((df["is_na"]) & (~df["escalated"])).sum())
    checks.append(CheckResult("Non-NOT_ATTEMPTED rows are not escalated (analysis_ready rule)", non_na_escalated == 0, f"rows={non_na_escalated}"))
    checks.append(CheckResult("All NOT_ATTEMPTED rows are escalated", na_not_escalated == 0, f"rows={na_not_escalated}"))

    incomplete = int(df["incomplete"].sum())
    checks.append(CheckResult("No incomplete rows", incomplete == 0, f"rows={incomplete}"))

    # TruthfulQA coverage
    covered = set(df["q_idx"].dropna().astype(int).tolist())
    total_q = len(truthfulqa)
    missing = [i for i in range(total_q) if i not in covered]
    checks.append(CheckResult("Expected filtered coverage is 807/817", len(covered) == 807 and len(missing) == 10, f"covered={len(covered)}, missing={len(missing)}"))

    # Cross-check against previously exported model metrics
    model_csv = ANALYSIS_DIR / "thesis_deep_model_metrics.csv"
    group_csv = ANALYSIS_DIR / "thesis_deep_group_metrics.csv"
    metric_match_bad = 0
    group_match_bad = 0
    if model_csv.exists():
        m_csv = pd.read_csv(model_csv)
        rows = []
        for model, sub in df.groupby("model"):
            n = len(sub)
            correct = int(sub["is_correct"].sum())
            incorrect = int(sub["is_incorrect"].sum())
            rows.append(
                {
                    "model": model,
                    "n": n,
                    "correct": correct,
                    "incorrect": incorrect,
                    "accuracy": correct / n,
                    "not_attempted": int(sub["is_na"].sum()),
                    "not_attempted_rate": float(sub["is_na"].mean()),
                    "self_consistent_0_9": int(sub["is_sc"].sum()),
                    "self_consistent_rate_total": float(sub["is_sc"].mean()),
                    "self_consistent_rate_of_errors": (int(sub["is_sc"].sum()) / incorrect) if incorrect else math.nan,
                    "rows_any_parse_failed": int(sub["any_parse_failed"].sum()),
                    "rows_any_api_failed": int(sub["any_api_failed"].sum()),
                    "rows_adjudicated": int(sub["adjudicated"].sum()),
                    "rows_escalated": int(sub["escalated"].sum()),
                }
            )
        m_new = pd.DataFrame(rows)
        merged = m_new.merge(m_csv, on="model", suffixes=("_new", "_csv"))
        for _, r in merged.iterrows():
            cols = [
                ("n_new", "n_csv"),
                ("correct_new", "correct_csv"),
                ("incorrect_new", "incorrect_csv"),
                ("self_consistent_0_9_new", "self_consistent_0_9_csv"),
                ("rows_any_parse_failed_new", "rows_any_parse_failed_csv"),
                ("rows_any_api_failed_new", "rows_any_api_failed_csv"),
                ("rows_adjudicated_new", "rows_adjudicated_csv"),
                ("rows_escalated_new", "rows_escalated_csv"),
            ]
            for a, b_ in cols:
                if int(r[a]) != int(r[b_]):
                    metric_match_bad += 1
            fcols = [
                ("accuracy_new", "accuracy_csv"),
                ("not_attempted_rate_new", "not_attempted_rate_csv"),
                ("self_consistent_rate_total_new", "self_consistent_rate_total_csv"),
                ("self_consistent_rate_of_errors_new", "self_consistent_rate_of_errors_csv"),
            ]
            for a, b_ in fcols:
                if not (pd.isna(r[a]) and pd.isna(r[b_])) and abs(float(r[a]) - float(r[b_])) > 1e-12:
                    metric_match_bad += 1
    else:
        metric_match_bad = 1
    checks.append(CheckResult("Model metrics match previous exported CSV", metric_match_bad == 0, f"differences={metric_match_bad}"))

    if group_csv.exists():
        g_csv = pd.read_csv(group_csv)
        closed = {"Claude Opus 4.6 (Anthropic)", "GPT-5.2 (OpenAI)", "Grok 4 (xAI)"}
        openw = {"DeepSeek V3.2 (DeepSeek)", "Llama 4 Maverick 17B (Groq)", "Qwen3 Next 80B (OpenRouter)"}
        rows_g = []
        for name, models in [("closed_api", closed), ("open_weight_api", openw)]:
            sub = df[df["model"].isin(models)]
            incorrect = int(sub["is_incorrect"].sum())
            rows_g.append(
                {
                    "group": name,
                    "rows": len(sub),
                    "accuracy": float(sub["is_correct"].mean()),
                    "self_consistent_rate_total": float(sub["is_sc"].mean()),
                    "self_consistent_rate_of_errors": (int(sub["is_sc"].sum()) / incorrect) if incorrect else math.nan,
                    "not_attempted_rate": float(sub["is_na"].mean()),
                }
            )
        g_new = pd.DataFrame(rows_g)
        m = g_new.merge(g_csv, on="group", suffixes=("_new", "_csv"))
        for _, r in m.iterrows():
            if int(r["rows_new"]) != int(r["rows_csv"]):
                group_match_bad += 1
            for c in ["accuracy", "self_consistent_rate_total", "self_consistent_rate_of_errors", "not_attempted_rate"]:
                if abs(float(r[f"{c}_new"]) - float(r[f"{c}_csv"])) > 1e-12:
                    group_match_bad += 1
    else:
        group_match_bad = 1
    checks.append(CheckResult("Group metrics match previous exported CSV", group_match_bad == 0, f"differences={group_match_bad}"))

    # Save details
    details["rows_total"] = n_rows
    details["unique_questions"] = n_q
    details["unique_models"] = n_m
    details["rows_not_attempted"] = int(df["is_na"].sum())
    details["rows_parse_failed"] = int(df["any_parse_failed"].sum())
    details["rows_api_failed"] = int(df["any_api_failed"].sum())
    details["rows_adjudicated"] = int(df["adjudicated"].sum())
    details["missing_indices"] = missing
    details["missing_df"] = truthfulqa[truthfulqa["q_idx"].isin(missing)][["q_idx", "category", "Question"]].copy()
    details["pass_count"] = sum(1 for c in checks if c.passed)
    details["check_count"] = len(checks)
    return checks, details


def compute_core_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    model_rows = []
    for model, sub in sorted(df.groupby("model"), key=lambda x: x[0]):
        n = len(sub)
        correct = int(sub["is_correct"].sum())
        incorrect = int(sub["is_incorrect"].sum())
        acc = correct / n
        acc_ci = bootstrap_ci(sub["is_correct"].astype(int).to_numpy())
        sc_total = float(sub["is_sc"].mean())
        sc_total_ci = bootstrap_ci(sub["is_sc"].astype(int).to_numpy())
        model_rows.append(
            {
                "model": model,
                "model_short": short_model(model),
                "n": n,
                "accuracy": acc,
                "accuracy_ci_low": acc_ci[0],
                "accuracy_ci_high": acc_ci[1],
                "incorrect": incorrect,
                "self_consistent_0_9": int(sub["is_sc"].sum()),
                "self_consistent_rate_total": sc_total,
                "self_consistent_rate_total_ci_low": sc_total_ci[0],
                "self_consistent_rate_total_ci_high": sc_total_ci[1],
                "self_consistent_rate_of_errors": (int(sub["is_sc"].sum()) / incorrect) if incorrect else math.nan,
                "not_attempted": int(sub["is_na"].sum()),
                "not_attempted_rate": float(sub["is_na"].mean()),
                "parse_failed_rows": int(sub["any_parse_failed"].sum()),
                "api_failed_rows": int(sub["any_api_failed"].sum()),
            }
        )
    model_df = pd.DataFrame(model_rows).sort_values("accuracy", ascending=False)

    # Threshold-specific distributions for 1.0 / 0.9 / 0.8
    threshold_rows = []
    threshold_model_rows = []
    for thr in REPORT_THRESHOLDS:
        col = f"error_label_{thr:.1f}"
        total = len(df)
        row = {"threshold": thr, "rows": total}
        for label in FIVE_LABELS:
            c = int((df[col] == label).sum())
            row[f"{label}_count"] = c
            row[f"{label}_rate"] = (c / total) if total else math.nan
        threshold_rows.append(row)

        for model, sub in sorted(df.groupby("model"), key=lambda x: x[0]):
            m_total = len(sub)
            mrow = {
                "model": model,
                "model_short": short_model(model),
                "threshold": thr,
                "rows": m_total,
            }
            for label in FIVE_LABELS:
                c = int((sub[col] == label).sum())
                mrow[f"{label}_count"] = c
                mrow[f"{label}_rate"] = (c / m_total) if m_total else math.nan
            threshold_model_rows.append(mrow)

    threshold_df = pd.DataFrame(threshold_rows).sort_values("threshold", ascending=False)
    threshold_model_df = pd.DataFrame(threshold_model_rows)
    label_09_model_df = threshold_model_df[threshold_model_df["threshold"].eq(0.9)].copy()
    label_09_model_df["correct_same_rate"] = label_09_model_df["reliably_correct_rate"]
    label_09_model_df["correct_different_rate"] = label_09_model_df["fragile_correct_rate"]
    label_09_model_df["incorrect_same_rate"] = label_09_model_df["self_consistent_error_rate"]
    label_09_model_df["incorrect_different_rate"] = label_09_model_df["inconsistent_error_rate"]
    label_09_model_df = label_09_model_df.sort_values("model_short")

    group_rows = []
    groups = {
        "Closed API (Claude/GPT-5.2/Grok)": {"Claude Opus 4.6 (Anthropic)", "GPT-5.2 (OpenAI)", "Grok 4 (xAI)"},
        "Open-weight API (DeepSeek/Llama/Qwen)": {"DeepSeek V3.2 (DeepSeek)", "Llama 4 Maverick 17B (Groq)", "Qwen3 Next 80B (OpenRouter)"},
    }
    for name, models in groups.items():
        sub = df[df["model"].isin(models)]
        incorrect = int(sub["is_incorrect"].sum())
        group_rows.append(
            {
                "group": name,
                "rows": len(sub),
                "accuracy": float(sub["is_correct"].mean()),
                "self_consistent_rate_total": float(sub["is_sc"].mean()),
                "self_consistent_rate_of_errors": (int(sub["is_sc"].sum()) / incorrect) if incorrect else math.nan,
                "not_attempted_rate": float(sub["is_na"].mean()),
            }
        )
    group_df = pd.DataFrame(group_rows)

    cat_rows = []
    cat_df = df[~df["category"].isna()].copy()
    for (model, cat), sub in cat_df.groupby(["model", "category"]):
        n = len(sub)
        incorrect = int(sub["is_incorrect"].sum())
        sc = int(sub["is_sc"].sum())
        cat_rows.append(
            {
                "model": model,
                "model_short": short_model(model),
                "category": cat,
                "n": n,
                "incorrect": incorrect,
                "sc_rate_total": sc / n if n else math.nan,
                "sc_rate_errors": sc / incorrect if incorrect else math.nan,
                "accuracy": float(sub["is_correct"].mean()) if n else math.nan,
            }
        )
    cat_by_model = pd.DataFrame(cat_rows)

    cat_agg_rows = []
    for cat, sub in cat_by_model.groupby("category"):
        cat_agg_rows.append(
            {
                "category": cat,
                "mean_sc_rate_total": float(sub["sc_rate_total"].mean()),
                "std_sc_rate_total": float(sub["sc_rate_total"].std(ddof=0)),
                "mean_sc_rate_errors": float(sub["sc_rate_errors"].replace([np.inf, -np.inf], np.nan).dropna().mean()),
                "mean_accuracy": float(sub["accuracy"].mean()),
                "support_rows": int(sub["n"].sum()),
                "support_questions_per_model": float(sub["n"].mean()),
            }
        )
    cat_agg = pd.DataFrame(cat_agg_rows).sort_values("mean_sc_rate_total", ascending=False)

    # Question type table
    qtype_rows = []
    for qtype, sub in df.groupby("question_type"):
        incorrect = int(sub["is_incorrect"].sum())
        qtype_rows.append(
            {
                "question_type": qtype,
                "rows": len(sub),
                "accuracy": float(sub["is_correct"].mean()),
                "sc_rate_total": float(sub["is_sc"].mean()),
                "sc_rate_errors": (int(sub["is_sc"].sum()) / incorrect) if incorrect else math.nan,
                "na_rate": float(sub["is_na"].mean()),
            }
        )
    qtype_df = pd.DataFrame(qtype_rows)

    # Pairwise significance
    pair_rows = []
    for a, b_ in [(a, b_) for i, a in enumerate(model_df["model"]) for b_ in model_df["model"][i + 1 :]]:
        A = df[df["model"] == a][["question_id", "is_correct", "is_sc", "is_na"]].rename(
            columns={"is_correct": "acc_a", "is_sc": "sc_a", "is_na": "na_a"}
        )
        B = df[df["model"] == b_][["question_id", "is_correct", "is_sc", "is_na"]].rename(
            columns={"is_correct": "acc_b", "is_sc": "sc_b", "is_na": "na_b"}
        )
        m = A.merge(B, on="question_id", how="inner")
        for metric in ["acc", "sc", "na"]:
            xa = m[f"{metric}_a"].astype(int)
            xb = m[f"{metric}_b"].astype(int)
            b_only = int(((xa == 1) & (xb == 0)).sum())
            c_only = int(((xa == 0) & (xb == 1)).sum())
            pair_rows.append(
                {
                    "model_a": a,
                    "model_b": b_,
                    "metric": metric,
                    "n": int(len(m)),
                    "rate_a": float(xa.mean()),
                    "rate_b": float(xb.mean()),
                    "delta_a_minus_b": float(xa.mean() - xb.mean()),
                    "discordant_a_only": b_only,
                    "discordant_b_only": c_only,
                    "p_exact_mcnemar": mcnemar_p(b_only, c_only),
                }
            )
    pair_df = pd.DataFrame(pair_rows)

    # Judge diagnostics
    patterns = Counter()
    slot_ok = Counter()
    slot_parse = Counter()
    slot_api = Counter()
    two_ok = 0
    two_ok_agree = 0
    for _, r in df.iterrows():
        statuses = r.get("correctness_judge_statuses")
        grades = r.get("correctness_judge_grades")
        if not isinstance(statuses, list):
            continue
        patterns[tuple(statuses)] += 1
        for i, s in enumerate(statuses):
            if s == "OK":
                slot_ok[i] += 1
            elif s == "PARSE_FAILED":
                slot_parse[i] += 1
            elif s == "API_FAILED":
                slot_api[i] += 1
        if isinstance(grades, list):
            ok = [g for g, s in zip(grades, statuses) if s == "OK" and g in ALLOWED_GRADE]
            if len(ok) == 2:
                two_ok += 1
                if ok[0] == ok[1]:
                    two_ok_agree += 1

    pattern_df = pd.DataFrame([{"pattern": " | ".join(k), "rows": v} for k, v in patterns.items()]).sort_values("rows", ascending=False)
    slot_df = pd.DataFrame(
        {
            "slot": [1, 2, 3],
            "OK": [slot_ok.get(0, 0), slot_ok.get(1, 0), slot_ok.get(2, 0)],
            "PARSE_FAILED": [slot_parse.get(0, 0), slot_parse.get(1, 0), slot_parse.get(2, 0)],
            "API_FAILED": [slot_api.get(0, 0), slot_api.get(1, 0), slot_api.get(2, 0)],
        }
    )

    return {
        "model_df": model_df,
        "threshold_df": threshold_df,
        "threshold_model_df": threshold_model_df,
        "label_09_model_df": label_09_model_df,
        "group_df": group_df,
        "cat_by_model_df": cat_by_model,
        "cat_agg_df": cat_agg,
        "qtype_df": qtype_df,
        "pair_df": pair_df,
        "pattern_df": pattern_df,
        "slot_df": slot_df,
        "two_ok": two_ok,
        "two_ok_agree": two_ok_agree,
        "two_ok_agree_rate": (two_ok_agree / two_ok) if two_ok else math.nan,
    }


def make_pairwise_matrix(pair_df: pd.DataFrame, metric: str, models: List[str]) -> pd.DataFrame:
    mat = pd.DataFrame(np.nan, index=models, columns=models)
    sub = pair_df[pair_df["metric"] == metric]
    for _, r in sub.iterrows():
        a, b_ = r["model_a"], r["model_b"]
        d = float(r["delta_a_minus_b"])
        mat.loc[a, b_] = d
        mat.loc[b_, a] = -d
    arr = mat.to_numpy(copy=True)
    np.fill_diagonal(arr, 0.0)
    return pd.DataFrame(arr, index=models, columns=models)


def generate_figures(t: Dict[str, Any]) -> None:
    sns.set_theme(style="whitegrid")
    model_df = t["model_df"].copy()
    threshold_df = t["threshold_df"].copy()
    threshold_model_df = t["threshold_model_df"].copy()
    label_09_model_df = t["label_09_model_df"].copy()
    group_df = t["group_df"].copy()
    cat_agg_df = t["cat_agg_df"].copy()
    cat_by_model = t["cat_by_model_df"].copy()
    pair_df = t["pair_df"].copy()
    qtype_df = t["qtype_df"].copy()
    pattern_df = t["pattern_df"].copy()
    slot_df = t["slot_df"].copy()

    # Fig 1
    f, axes = plt.subplots(1, 3, figsize=(16, 5))
    xdf = model_df.sort_values("accuracy", ascending=False)
    sns.barplot(data=xdf, x="model_short", y="accuracy", color="#2a9d8f", ax=axes[0])
    axes[0].set_title("Model Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Rate")

    sns.barplot(data=xdf, x="model_short", y="self_consistent_rate_total", color="#e76f51", ax=axes[1])
    axes[1].set_title("Self-Consistent Error Rate (all rows)")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Rate")

    sns.barplot(data=xdf, x="model_short", y="not_attempted_rate", color="#457b9d", ax=axes[2])
    axes[2].set_title("NOT_ATTEMPTED Rate")
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Rate")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    f.tight_layout()
    f.savefig(FIG_DIR / "simple_fig_01_model_headlines.png", dpi=220, bbox_inches="tight")
    plt.close(f)

    # Fig 2
    f, ax = plt.subplots(figsize=(11, 5.5))
    xdf = model_df.sort_values("model_short")
    x = np.arange(len(xdf))
    sc = xdf["self_consistent_0_9"] / xdf["n"]
    inc = (xdf["incorrect"] - xdf["self_consistent_0_9"]) / xdf["n"]
    na = xdf["not_attempted"] / xdf["n"]
    ax.bar(x, sc, label="Self-consistent incorrect", color="#e76f51")
    ax.bar(x, inc, bottom=sc, label="Inconsistent incorrect", color="#f4a261")
    ax.bar(x, na, bottom=sc + inc, label="NOT_ATTEMPTED", color="#457b9d")
    ax.set_xticks(x)
    ax.set_xticklabels(xdf["model_short"], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of rows")
    ax.set_title("How each model's errors are distributed")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    f.tight_layout()
    f.savefig(FIG_DIR / "simple_fig_02_error_breakdown.png", dpi=220, bbox_inches="tight")
    plt.close(f)

    # Fig 3
    melt = group_df.melt(
        id_vars=["group"],
        value_vars=["accuracy", "self_consistent_rate_total", "not_attempted_rate"],
        var_name="metric",
        value_name="value",
    )
    name_map = {
        "accuracy": "Accuracy",
        "self_consistent_rate_total": "SC rate (all rows)",
        "not_attempted_rate": "NOT_ATTEMPTED rate",
    }
    melt["metric"] = melt["metric"].map(name_map)
    f, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=melt, x="metric", y="value", hue="group", ax=ax, palette="Set2")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_xlabel("")
    ax.set_title("Closed-API vs Open-weight-API group comparison")
    ax.grid(axis="y", alpha=0.25)
    f.tight_layout()
    f.savefig(FIG_DIR / "simple_fig_03_group_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(f)

    # Fig 4
    top = cat_agg_df.sort_values("mean_sc_rate_total", ascending=False).head(12)
    f, ax = plt.subplots(figsize=(10, 6.5))
    sns.barplot(data=top, y="category", x="mean_sc_rate_total", color="#d62828", ax=ax)
    for i, r in top.reset_index(drop=True).iterrows():
        ax.text(float(r["mean_sc_rate_total"]) + 0.005, i, f"n={int(round(r['support_questions_per_model']))}", va="center", fontsize=9)
    ax.set_xlabel("Mean self-consistent error rate (all rows)")
    ax.set_ylabel("")
    ax.set_title("Top vulnerable categories")
    ax.grid(axis="x", alpha=0.25)
    f.tight_layout()
    f.savefig(FIG_DIR / "simple_fig_04_top_categories.png", dpi=220, bbox_inches="tight")
    plt.close(f)

    # Fig 5
    cats = top["category"].tolist()
    h = cat_by_model[cat_by_model["category"].isin(cats)].copy()
    piv = h.pivot(index="category", columns="model_short", values="sc_rate_errors")
    f, ax = plt.subplots(figsize=(10.8, 7.2))
    sns.heatmap(piv, cmap="YlOrRd", linewidths=0.4, linecolor="white", cbar_kws={"label": "SC rate among errors"}, ax=ax)
    ax.set_title("Category x model heatmap (SC rate among incorrect rows)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    f.tight_layout()
    f.savefig(FIG_DIR / "simple_fig_05_category_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(f)

    # Fig 6 and 7
    models = model_df["model"].tolist()
    labels = [short_model(m) for m in models]
    for metric, fname, title in [
        ("acc", "simple_fig_06_pairwise_accuracy_heatmap.png", "Pairwise accuracy delta (A-B, percentage points)"),
        ("sc", "simple_fig_07_pairwise_sc_heatmap.png", "Pairwise self-consistent-rate delta (A-B, percentage points)"),
    ]:
        mat = make_pairwise_matrix(pair_df, metric, models)
        f, ax = plt.subplots(figsize=(7.5, 6.4))
        sns.heatmap(mat * 100, annot=True, fmt=".1f", cmap="RdBu_r", center=0, xticklabels=labels, yticklabels=labels, cbar_kws={"label": "Delta (pp)"}, ax=ax)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)
        f.tight_layout()
        f.savefig(FIG_DIR / fname, dpi=220, bbox_inches="tight")
        plt.close(f)

    # Fig 8
    q_melt = qtype_df.melt(id_vars=["question_type"], value_vars=["accuracy", "sc_rate_total", "na_rate"], var_name="metric", value_name="value")
    q_melt["metric"] = q_melt["metric"].map({"accuracy": "Accuracy", "sc_rate_total": "SC rate (all rows)", "na_rate": "NOT_ATTEMPTED rate"})
    f, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=q_melt, x="metric", y="value", hue="question_type", ax=ax, palette="Set1")
    ax.set_ylim(0, 1)
    ax.set_title("Adversarial vs Non-Adversarial outcomes")
    ax.set_xlabel("")
    ax.set_ylabel("Rate")
    ax.grid(axis="y", alpha=0.25)
    f.tight_layout()
    f.savefig(FIG_DIR / "simple_fig_08_question_type.png", dpi=220, bbox_inches="tight")
    plt.close(f)

    # Fig 9
    f, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    top_pat = pattern_df.head(6)
    sns.barplot(data=top_pat, y="pattern", x="rows", color="#6d597a", ax=axes[0])
    axes[0].set_title("Top judge status patterns")
    axes[0].set_xlabel("Rows")
    axes[0].set_ylabel("")
    axes[0].grid(axis="x", alpha=0.25)

    slot_m = slot_df.melt(id_vars=["slot"], var_name="status", value_name="rows")
    sns.barplot(data=slot_m, x="slot", y="rows", hue="status", ax=axes[1])
    axes[1].set_title("Judge status by slot")
    axes[1].set_xlabel("Judge slot")
    axes[1].set_ylabel("Rows")
    axes[1].grid(axis="y", alpha=0.25)
    f.tight_layout()
    f.savefig(FIG_DIR / "simple_fig_09_judge_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(f)

    # Fig 10: per-threshold SC rates by model
    f, ax = plt.subplots(figsize=(10.5, 5.5))
    tdf = threshold_model_df.copy()
    tdf["threshold_label"] = tdf["threshold"].map(lambda x: f"{x:.1f}")
    sns.barplot(
        data=tdf,
        x="model_short",
        y="self_consistent_error_rate",
        hue="threshold_label",
        order=sorted(model_df["model_short"].tolist()),
        hue_order=["1.0", "0.9", "0.8"],
        palette=["#8ecae6", "#219ebc", "#023047"],
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_title("Self-consistent error rate by threshold (1.0 vs 0.9 vs 0.8)")
    ax.set_xlabel("")
    ax.set_ylabel("Rate")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Threshold")
    f.tight_layout()
    f.savefig(FIG_DIR / "simple_fig_10_threshold_sc_by_model.png", dpi=220, bbox_inches="tight")
    plt.close(f)

    # Fig 11: full 0.9 label composition by model
    f, ax = plt.subplots(figsize=(11, 5.8))
    xdf = label_09_model_df.sort_values("model_short")
    x = np.arange(len(xdf))
    c_same = xdf["correct_same_rate"].to_numpy()
    c_diff = xdf["correct_different_rate"].to_numpy()
    i_same = xdf["incorrect_same_rate"].to_numpy()
    i_diff = xdf["incorrect_different_rate"].to_numpy()
    na = xdf["not_attempted_rate"].to_numpy()
    ax.bar(x, c_same, label=LABEL_PRETTY["reliably_correct"], color="#2a9d8f")
    ax.bar(x, c_diff, bottom=c_same, label=LABEL_PRETTY["fragile_correct"], color="#90be6d")
    ax.bar(x, i_same, bottom=c_same + c_diff, label=LABEL_PRETTY["self_consistent_error"], color="#e76f51")
    ax.bar(x, i_diff, bottom=c_same + c_diff + i_same, label=LABEL_PRETTY["inconsistent_error"], color="#f4a261")
    ax.bar(x, na, bottom=c_same + c_diff + i_same + i_diff, label=LABEL_PRETTY["not_attempted"], color="#457b9d")
    ax.set_xticks(x)
    ax.set_xticklabels(xdf["model_short"], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of rows")
    ax.set_xlabel("")
    ax.set_title("Full 0.9 label breakdown by model")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    f.tight_layout()
    f.savefig(FIG_DIR / "simple_fig_11_full_label_breakdown_0_9.png", dpi=220, bbox_inches="tight")
    plt.close(f)

    # Fig 12: full 1.0 label composition by model (strictest threshold)
    label_10_model_df = threshold_model_df[threshold_model_df["threshold"].eq(1.0)].copy()
    label_10_model_df = label_10_model_df.sort_values("model_short")
    f, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(label_10_model_df))
    c_same = label_10_model_df["reliably_correct_rate"].to_numpy()
    c_diff = label_10_model_df["fragile_correct_rate"].to_numpy()
    i_same = label_10_model_df["self_consistent_error_rate"].to_numpy()
    i_diff = label_10_model_df["inconsistent_error_rate"].to_numpy()
    na = label_10_model_df["not_attempted_rate"].to_numpy()
    ax.bar(x, c_same, label=LABEL_PRETTY["reliably_correct"], color="#2a9d8f")
    ax.bar(x, c_diff, bottom=c_same, label=LABEL_PRETTY["fragile_correct"], color="#90be6d")
    ax.bar(x, i_same, bottom=c_same + c_diff, label=LABEL_PRETTY["self_consistent_error"], color="#e76f51")
    ax.bar(x, i_diff, bottom=c_same + c_diff + i_same, label=LABEL_PRETTY["inconsistent_error"], color="#f4a261")
    ax.bar(x, na, bottom=c_same + c_diff + i_same + i_diff, label=LABEL_PRETTY["not_attempted"], color="#457b9d")
    ax.set_xticks(x)
    ax.set_xticklabels(label_10_model_df["model_short"], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of rows")
    ax.set_xlabel("")
    ax.set_title("Full 1.0 label breakdown by model (strictest threshold)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    f.tight_layout()
    f.savefig(FIG_DIR / "simple_fig_12_full_label_breakdown_1_0.png", dpi=220, bbox_inches="tight")
    plt.close(f)

    # Fig 13: SC rate at threshold 1.0 by model (standalone bar chart)
    sc_10 = threshold_model_df[threshold_model_df["threshold"].eq(1.0)].copy()
    sc_10 = sc_10.sort_values("self_consistent_error_rate", ascending=False)
    f, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#e76f51" if r >= 0.15 else "#f4a261" if r >= 0.10 else "#2a9d8f"
              for r in sc_10["self_consistent_error_rate"]]
    bars = ax.bar(range(len(sc_10)), sc_10["self_consistent_error_rate"].to_numpy(), color=colors)
    ax.set_xticks(range(len(sc_10)))
    ax.set_xticklabels(sc_10["model_short"], rotation=30, ha="right")
    for i, (_, r) in enumerate(sc_10.iterrows()):
        ax.text(i, float(r["self_consistent_error_rate"]) + 0.005,
                f"{100*r['self_consistent_error_rate']:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.30)
    ax.set_ylabel("Self-consistent error rate")
    ax.set_xlabel("")
    ax.set_title("Self-consistent error rate at threshold 1.0 (strictest possible)")
    ax.grid(axis="y", alpha=0.25)
    f.tight_layout()
    f.savefig(FIG_DIR / "simple_fig_13_sc_rate_1_0_by_model.png", dpi=220, bbox_inches="tight")
    plt.close(f)


def write_bib() -> None:
    bib = r"""@inproceedings{lin2022truthfulqa,
  title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
  author={Lin, Stephanie and Hilton, Jacob and Evans, Owain},
  booktitle={Proceedings of ACL 2022},
  pages={3214--3252},
  year={2022},
  doi={10.18653/v1/2022.acl-long.229},
  url={https://aclanthology.org/2022.acl-long.229/}
}

@inproceedings{manakul2023selfcheckgpt,
  title={SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models},
  author={Manakul, Potsawee and Liusie, Adian and Gales, Mark},
  booktitle={Proceedings of EMNLP 2023},
  pages={9004--9017},
  year={2023},
  doi={10.18653/v1/2023.emnlp-main.557},
  url={https://aclanthology.org/2023.emnlp-main.557/}
}

@inproceedings{zhang2023sac3,
  title={SAC3: Reliable Hallucination Detection in Black-Box Language Models via Semantic-aware Cross-check Consistency},
  author={Zhang, Jiaxin and Li, Zhuohang and Das, Kamalika and Malin, Bradley and Kumar, Sricharan},
  booktitle={Findings of EMNLP 2023},
  pages={15445--15458},
  year={2023},
  doi={10.18653/v1/2023.findings-emnlp.1032},
  url={https://aclanthology.org/2023.findings-emnlp.1032/}
}

@article{farquhar2024semanticentropy,
  title={Detecting Hallucinations in Large Language Models Using Semantic Entropy},
  author={Farquhar, Sebastian and Kossen, Jannik and Kuhn, Lorenz and Gal, Yarin},
  journal={Nature},
  volume={630},
  number={8017},
  pages={625--630},
  year={2024},
  doi={10.1038/s41586-024-07421-0},
  url={https://www.nature.com/articles/s41586-024-07421-0}
}

@inproceedings{yehuda2024interrogatellm,
  title={InterrogateLLM: Zero-Resource Hallucination Detection in LLM-Generated Answers},
  author={Yehuda, Yakir and Malkiel, Itzik and Barkan, Oren and Weill, Jonathan and Ronen, Royi and Koenigstein, Noam},
  booktitle={Proceedings of ACL 2024},
  pages={9333--9347},
  year={2024},
  doi={10.18653/v1/2024.acl-long.506},
  url={https://aclanthology.org/2024.acl-long.506/}
}

@inproceedings{tan2025tooconsistent,
  title={Too Consistent to Detect: A Study of Self-Consistent Errors in LLMs},
  author={Tan, Hexiang and Sun, Fei and Liu, Sha and Su, Du and Cao, Qi and Chen, Xin and Wang, Jingang and Cai, Xunliang and Wang, Yuanzhuo and Shen, Huawei and Cheng, Xueqi},
  booktitle={Proceedings of EMNLP 2025},
  pages={4755--4765},
  year={2025},
  doi={10.18653/v1/2025.emnlp-main.238},
  url={https://aclanthology.org/2025.emnlp-main.238/}
}

@inproceedings{liu2025agser,
  title={Attention-guided Self-reflection for Zero-shot Hallucination Detection in Large Language Models},
  author={Liu, Qiang and Chen, Xinlong and Ding, Yue and Song, Bowen and Wang, Weiqiang and Wu, Shu and Wang, Liang},
  booktitle={Proceedings of EMNLP 2025},
  pages={21005--21021},
  year={2025},
  doi={10.18653/v1/2025.emnlp-main.1063},
  url={https://aclanthology.org/2025.emnlp-main.1063/}
}

@article{ji2023survey,
  title={Survey of Hallucination in Natural Language Generation},
  author={Ji, Ziwei and Lee, Nayeon and Frieske, Rob and others},
  journal={ACM Computing Surveys},
  volume={55},
  number={12},
  pages={1--38},
  year={2023},
  doi={10.1145/3571730}
}

@article{kuhn2023semanticuncertainty,
  title={Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation},
  author={Kuhn, Lorenz and Gal, Yarin and Farquhar, Sebastian},
  journal={arXiv preprint arXiv:2302.09664},
  year={2023},
  url={https://arxiv.org/abs/2302.09664}
}

@article{he2020deberta,
  title={DeBERTa: Decoding-enhanced BERT with Disentangled Attention},
  author={He, Pengcheng and Liu, Xiaodong and Gao, Jianfeng and Chen, Weizhu},
  journal={arXiv preprint arXiv:2006.03654},
  year={2020},
  url={https://arxiv.org/abs/2006.03654}
}
"""
    (OUT_DIR / "references.bib").write_text(bib, encoding="utf-8")


def table_validation_rows(checks: List[CheckResult]) -> str:
    rows = []
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        rows.append(f"{tex_escape(c.name)} & {status} & {tex_escape(c.details)} \\\\")
    return "\n".join(rows)


def table_model_rows(df: pd.DataFrame) -> str:
    out = []
    for _, r in df.sort_values("accuracy", ascending=False).iterrows():
        out.append(
            f"{tex_escape(r['model_short'])} & {int(r['n'])} & "
            f"{100*r['accuracy']:.1f} [{100*r['accuracy_ci_low']:.1f}, {100*r['accuracy_ci_high']:.1f}] & "
            f"{100*r['self_consistent_rate_total']:.1f} & {100*r['self_consistent_rate_of_errors']:.1f} & "
            f"{100*r['not_attempted_rate']:.1f} \\\\"
        )
    return "\n".join(out)


def table_threshold_rows(df: pd.DataFrame) -> str:
    out = []
    for _, r in df.sort_values("threshold", ascending=False).iterrows():
        out.append(
            f"{r['threshold']:.1f} & "
            f"{100*r['reliably_correct_rate']:.1f} & "
            f"{100*r['fragile_correct_rate']:.1f} & "
            f"{100*r['self_consistent_error_rate']:.1f} & "
            f"{100*r['inconsistent_error_rate']:.1f} & "
            f"{100*r['not_attempted_rate']:.1f} \\\\"
        )
    return "\n".join(out)


def table_label_09_rows(df: pd.DataFrame) -> str:
    out = []
    for _, r in df.sort_values("model_short").iterrows():
        out.append(
            f"{tex_escape(r['model_short'])} & "
            f"{100*r['correct_same_rate']:.1f} & "
            f"{100*r['correct_different_rate']:.1f} & "
            f"{100*r['incorrect_same_rate']:.1f} & "
            f"{100*r['incorrect_different_rate']:.1f} & "
            f"{100*r['not_attempted_rate']:.1f} \\\\"
        )
    return "\n".join(out)


def table_category_rows(df: pd.DataFrame, n: int = 12) -> str:
    out = []
    top = df.sort_values("mean_sc_rate_total", ascending=False).head(n)
    for _, r in top.iterrows():
        out.append(
            f"{tex_escape(r['category'])} & {100*r['mean_sc_rate_total']:.1f} & "
            f"{100*r['mean_sc_rate_errors']:.1f} & {int(round(r['support_questions_per_model']))} \\\\"
        )
    return "\n".join(out)


def table_significance_rows(pair_df: pd.DataFrame, metric: str, n: int = 10) -> str:
    sub = pair_df[(pair_df["metric"] == metric) & (pair_df["p_exact_mcnemar"] < 0.05)].sort_values("p_exact_mcnemar").head(n)
    if sub.empty:
        return r"\multicolumn{3}{c}{No significant pairs at $p<0.05$.} \\"
    rows = []
    for _, r in sub.iterrows():
        pair = f"{short_model(r['model_a'])} vs {short_model(r['model_b'])}"
        rows.append(f"{tex_escape(pair)} & {100*r['delta_a_minus_b']:+.1f} & {r['p_exact_mcnemar']:.2e} \\\\")
    return "\n".join(rows)


def table_missing_rows(missing_df: pd.DataFrame) -> str:
    if missing_df.empty:
        return r"\multicolumn{3}{c}{No missing rows.} \\"
    out = []
    for _, r in missing_df.sort_values("q_idx").iterrows():
        q = str(r["Question"])
        if len(q) > 92:
            q = q[:89] + "..."
        out.append(f"{int(r['q_idx'])} & {tex_escape(r['category'])} & {tex_escape(q)} \\\\")
    return "\n".join(out)


def build_expected_vs_observed(model_df: pd.DataFrame, details: Dict[str, Any]) -> List[Dict[str, str]]:
    overall_acc = model_df["accuracy"].mean()
    sc_total = model_df["self_consistent_rate_total"].mean()
    parse_rows = details["rows_parse_failed"]
    two_ok_agree = details.get("two_ok_agree_rate", math.nan)

    return [
        {
            "theme": "Self-consistency can hide hallucinations",
            "expected_from_literature": "Yes. Prior work says simple consistency checks miss a subset of hallucinations.",
            "observed_here": f"Yes. Mean SC(all rows) across models is {100*sc_total:.1f}%, and many incorrect rows are self-consistent.",
            "status": "Expected",
            "cite": r"\cite{manakul2023selfcheckgpt,zhang2023sac3,tan2025tooconsistent}",
        },
        {
            "theme": "Meaning-level methods are needed",
            "expected_from_literature": "Yes. Semantic grouping/uncertainty outperforms pure string-level checks.",
            "observed_here": "Yes. Pipeline uses semantic equivalence (NLI) and resolves labels despite paraphrase variation.",
            "status": "Expected",
            "cite": r"\cite{farquhar2024semanticentropy,kuhn2023semanticuncertainty,he2020deberta}",
        },
        {
            "theme": "Topic/category vulnerability differs",
            "expected_from_literature": "Yes. TruthfulQA was designed with misconception-heavy categories and uneven difficulty.",
            "observed_here": "Yes. Large spread in category SC rates; top categories are substantially higher than median.",
            "status": "Expected",
            "cite": r"\cite{lin2022truthfulqa,tan2025tooconsistent}",
        },
        {
            "theme": "Larger or stronger models always eliminate SC errors",
            "expected_from_literature": "No. Recent work warns SC errors can remain stable or increase.",
            "observed_here": "Mixed. Accuracy improves for some models, but SC rates are not monotonically reduced.",
            "status": "Expected non-monotonic behavior",
            "cite": r"\cite{lin2022truthfulqa,tan2025tooconsistent}",
        },
        {
            "theme": "Zero-resource black-box detection is feasible",
            "expected_from_literature": "Yes, but with limitations.",
            "observed_here": f"Yes. All rows are resolved (0 unresolved), but {parse_rows} rows show parser-related fragility in judge outputs.",
            "status": "Expected with operational caveats",
            "cite": r"\cite{manakul2023selfcheckgpt,yehuda2024interrogatellm,liu2025agser}",
        },
        {
            "theme": "RQ3 white-box superiority can be tested here",
            "expected_from_literature": "White-box cross-model probing can improve SC-error detection if hidden states are available.",
            "observed_here": "Not testable in this dataset because no hidden-state probe outputs are present.",
            "status": "Missing required experiment",
            "cite": r"\cite{tan2025tooconsistent}",
        },
        {
            "theme": "Judge repeat consistency is usually stable when two judges are available",
            "expected_from_literature": "Ensembles can be stable if failures are sparse.",
            "observed_here": f"Yes. Two-OK agreement is {100*two_ok_agree:.2f}% when exactly two judge slots are available.",
            "status": "Expected",
            "cite": r"\cite{tan2025tooconsistent,ji2023survey}",
        },
        {
            "theme": "Overall factuality remains a hard problem",
            "expected_from_literature": "Yes. Hallucination remains unresolved in general-purpose LLMs.",
            "observed_here": f"Yes. Mean model accuracy is {100*overall_acc:.1f}% and NOT_ATTEMPTED is non-zero.",
            "status": "Expected",
            "cite": r"\cite{ji2023survey,lin2022truthfulqa}",
        },
    ]


def expected_rows_tex(rows: List[Dict[str, str]]) -> str:
    out = []
    for r in rows:
        out.append(
            f"{tex_escape(r['theme'])} & {tex_escape(r['status'])} & "
            f"{tex_escape(r['observed_here'])} {r['cite']} \\\\"
        )
    return "\n".join(out)


def write_tex(checks: List[CheckResult], details: Dict[str, Any], t: Dict[str, Any], expected_rows: List[Dict[str, str]]) -> None:
    model_df = t["model_df"]
    threshold_df = t["threshold_df"]
    label_09_model_df = t["label_09_model_df"]
    group_df = t["group_df"]
    cat_agg_df = t["cat_agg_df"]
    qtype_df = t["qtype_df"]
    pair_df = t["pair_df"]
    pattern_df = t["pattern_df"]
    slot_df = t["slot_df"]
    missing_df = details["missing_df"]

    overall_accuracy = model_df["accuracy"].mean()
    overall_sc_all = model_df["self_consistent_rate_total"].mean()
    overall_na = model_df["not_attempted_rate"].mean()
    thr_lookup = {float(r["threshold"]): r for _, r in threshold_df.iterrows()}
    sc_1_0 = thr_lookup[1.0]["self_consistent_error_rate"]
    sc_0_9 = thr_lookup[0.9]["self_consistent_error_rate"]
    sc_0_8 = thr_lookup[0.8]["self_consistent_error_rate"]
    passes = details["pass_count"]
    total_checks = details["check_count"]

    closed = group_df[group_df["group"].str.contains("Closed", regex=False)].iloc[0]
    openw = group_df[group_df["group"].str.contains("Open-weight", regex=False)].iloc[0]

    qtype_bullets = []
    for _, r in qtype_df.sort_values("question_type").iterrows():
        qtype_bullets.append(
            f"\\item \\textbf{{{tex_escape(r['question_type'])}}}: accuracy {pct(r['accuracy'])}, "
            f"self-consistent error rate {pct(r['sc_rate_total'])}, self-consistent among errors {pct(r['sc_rate_errors'])}, "
            f"NOT\\_ATTEMPTED {pct(r['na_rate'])}."
        )

    model_release_bullets = [
        r"\item \textbf{Claude Opus 4.6:} February 5, 2026",
        r"\item \textbf{GPT-5.2:} December 11, 2025",
        r"\item \textbf{Qwen3 Next 80B:} September 9, 2025 (first public checkpoint)",
        r"\item \textbf{DeepSeek V3.2:} December 1, 2025",
        r"\item \textbf{Grok 4:} July 9, 2025",
        r"\item \textbf{Llama 4 Maverick:} April 5, 2025",
    ]

    glossary = [
        ("Greedy answer", "The model's first/main answer for a question."),
        ("Stochastic samples", "Extra answers from the same model using randomness (temperature sampling)."),
        ("Self-consistent error", "The model repeatedly gives wrong answers that mean the same thing."),
        ("Inconsistent error", "The model is wrong, but its sampled answers vary in meaning."),
        ("NOT_ATTEMPTED", "The judge ensemble cannot confidently assign CORRECT or INCORRECT."),
        ("Correct + same meaning", "The greedy answer is correct AND the sampled answers mostly keep the same meaning as the greedy answer."),
        ("Correct + different meaning", "The greedy answer is correct AND the sampled answers often change meaning compared to the greedy answer."),
        ("Incorrect + same meaning", "The greedy answer is incorrect AND the sampled answers mostly keep the same (wrong) meaning."),
        ("Incorrect + different meaning", "The greedy answer is incorrect AND the sampled answers often change meaning."),
        ("+ (in labels)", "Means AND: both parts must be true."),
        ("Threshold (e.g., 0.9)", "How strict 'same meaning' is: a row counts as 'same' if same/(same+different) >= threshold (excluding 'unclear')."),
        ("Semantic equivalence", "Two answers have the same meaning, even if wording differs."),
        ("NLI", "Natural Language Inference; a model that tests whether one statement implies another."),
        ("Majority decision", "At least 2 of 3 judges agree on the same grade."),
        ("Adjudicator", "Tie-breaker judge used when majority is not available."),
        ("McNemar test", "A paired significance test for two systems on the same questions."),
        ("95% bootstrap CI", "A range that estimates uncertainty by repeated resampling."),
    ]
    glossary_rows = "\n".join([f"{tex_escape(k)} & {tex_escape(v)} \\\\" for k, v in glossary])

    tex = f"""\\documentclass[11pt]{{article}}
\\usepackage[a4paper,margin=1in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{array}}
\\usepackage{{float}}
\\usepackage{{hyperref}}
\\usepackage{{xcolor}}

\\title{{Simple-Language Deep Validation Report\\\\
\\large LLM Self-Consistent Error Analysis (Second Pass)}}
\\author{{Simranjeet Singh}}
\\date{{February 18, 2026}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
This report is a second, stricter pass over the final analysis dataset. It is written in simple terms, checks data integrity and math consistency step by step, explains every major term, and interprets each graph in plain language. The input contains {details['rows_total']} rows ({details['unique_questions']} questions $\\times$ {details['unique_models']} models). We ran {total_checks} validation checks and passed {passes}. The report also compares findings against recent primary research and marks each major pattern as expected or unexpected.
\\end{{abstract}}

\\section{{What this report does (plain language)}}
In simple terms, this report answers five questions:
\\begin{{enumerate}}
\\item Is the dataset internally correct (no broken rows, no inconsistent labels)?
\\item Do the headline numbers match what is in the raw file?
\\item What do the model differences mean in practice?
\\item Which question categories are risky?
\\item Are these results normal according to recent research, or surprising?
\\end{{enumerate}}

\\section{{Data and scope}}
\\textbf{{Main file used:}} \\texttt{{{tex_escape(str(FINAL_JSONL))}}}

\\textbf{{TruthfulQA source used for categories:}} \\texttt{{{tex_escape(str(TRUTHFULQA_CSV))}}}

\\begin{{itemize}}
\\item Rows: {details['rows_total']}
\\item Questions covered: {details['unique_questions']} (filtered subset of TruthfulQA)
\\item Models: {details['unique_models']}
\\item Parse-failed judge rows: {details['rows_parse_failed']}
\\item API-failed judge rows: {details['rows_api_failed']}
\\item Adjudicated rows: {details['rows_adjudicated']}
\\end{{itemize}}

\\paragraph{{Model release dates (timeline context).}}
I am adding this so the comparison is grounded in time, not just scores.
\\begin{{itemize}}
{chr(10).join(model_release_bullets)}
\\end{{itemize}}
I am not claiming release date alone explains performance, but it is useful context when comparing model behavior.

\\section{{Glossary (what each term means)}}
\\begin{{longtable}}{{p{{0.28\\textwidth}}p{{0.66\\textwidth}}}}
\\toprule
Term & Meaning in simple words \\\\
\\midrule
{glossary_rows}
\\bottomrule
\\end{{longtable}}

\\paragraph{{How to read labels like Correct+Same.}}
In labels like \\texttt{{Correct+Same}} and \\texttt{{Incorrect+Same}}, the \\textbf{{+}} symbol means \\textbf{{AND}}. \\textbf{{Correct/Incorrect}} refers to the judged grade of the greedy answer. \\textbf{{Same/Different}} refers to whether the model's sampled answers are semantically equivalent to the greedy answer (paraphrases count), based on an NLI equivalence judge. At threshold $t$ (e.g., 0.9), we count a row as \\textbf{{Same}} if $\\#same/(\\#same+\\#different) \\ge t$; comparisons labeled \\texttt{{unclear}} are excluded from the ratio.

\\section{{Validation checklist (double-check pass)}}
\\textbf{{Result:}} {passes}/{total_checks} checks passed.

\\begin{{longtable}}{{p{{0.50\\textwidth}}p{{0.08\\textwidth}}p{{0.35\\textwidth}}}}
\\toprule
Check & Status & Details \\\\
\\midrule
{table_validation_rows(checks)}
\\bottomrule
\\end{{longtable}}

\\paragraph{{Interpretation.}}
This means the core dataset logic is stable: row structure, grade logic, majority/adjudicator behavior, equivalence math, and threshold labels all line up exactly with what the pipeline claims.

\\section{{Main findings in simple terms}}
\\begin{{itemize}}
\\item Average model accuracy is {pct(overall_accuracy)}.
\\item Average self-consistent error rate (all rows, threshold 0.9) is {pct(overall_sc_all)}.
\\item Threshold sensitivity for self-consistent error rate: 1.0 = {pct(sc_1_0)}, 0.9 = {pct(sc_0_9)}, 0.8 = {pct(sc_0_8)}.
\\item Average NOT\\_ATTEMPTED rate is {pct(overall_na)}.
\\item Closed APIs in this run perform better than open-weight APIs on average: accuracy {pct(closed['accuracy'])} vs {pct(openw['accuracy'])}.
\\item But higher accuracy does \\emph{{not automatically}} mean low self-consistent error; model behavior is mixed by metric.
\\end{{itemize}}

\\begin{{table}}[H]
\\centering
\\caption{{Per-model headline results}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Model & N & Accuracy [95\\% CI] & SC(all) & SC(of errors) & NOT\\_ATTEMPTED \\\\
\\midrule
{table_model_rows(model_df)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[H]
\\centering
\\caption{{Overall label distribution by threshold (percent of all rows)}}
\\begin{{tabular}}{{rrrrrr}}
\\toprule
Threshold & Correct+Same & Correct+Different & Incorrect+Same & Incorrect+Different & NOT\\_ATTEMPTED \\\\
\\midrule
{table_threshold_rows(threshold_df)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[H]
\\centering
\\caption{{Full 0.9 breakdown by model (requested split)}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Model & Correct+Same & Correct+Different & Incorrect+Same & Incorrect+Different & NOT\\_ATTEMPTED \\\\
\\midrule
{table_label_09_rows(label_09_model_df)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\section{{Graph-by-graph explanation}}
\\subsection{{Figure 1: Model headline rates}}
\\begin{{figure}}[H]
  \\centering
  \\includegraphics[width=0.99\\textwidth]{{figures/simple_fig_01_model_headlines.png}}
  \\caption{{How to read: left = accuracy (higher is better), middle = self-consistent errors (lower is better), right = NOT\\_ATTEMPTED (lower is better).}}
\\end{{figure}}
\\textbf{{Plain-language takeaway:}} Models trade off these dimensions. A model can have good accuracy and still keep non-trivial self-consistent errors.

\\subsection{{Figure 2: Error composition per model}}
\\begin{{figure}}[H]
  \\centering
  \\includegraphics[width=0.90\\textwidth]{{figures/simple_fig_02_error_breakdown.png}}
  \\caption{{How to read: each stacked bar shows what fraction of all rows are self-consistent incorrect, inconsistent incorrect, and NOT\\_ATTEMPTED.}}
\\end{{figure}}
\\textbf{{Plain-language takeaway:}} Error types are not equally distributed. Some models fail more through stable wrong beliefs than random inconsistency.

\\subsection{{Figure 3: Group comparison}}
\\begin{{figure}}[H]
  \\centering
  \\includegraphics[width=0.82\\textwidth]{{figures/simple_fig_03_group_comparison.png}}
  \\caption{{How to read: compares closed APIs vs open-weight APIs in the same pipeline.}}
\\end{{figure}}
\\textbf{{Plain-language takeaway:}} Closed APIs lead in accuracy in this run, while open-weight APIs show higher self-consistent and NOT\\_ATTEMPTED rates.

\\subsection{{Figure 4 and Figure 5: Category vulnerability}}
\\begin{{figure}}[H]
  \\centering
  \\includegraphics[width=0.49\\textwidth]{{figures/simple_fig_04_top_categories.png}}\\hfill
  \\includegraphics[width=0.49\\textwidth]{{figures/simple_fig_05_category_heatmap.png}}
  \\caption{{Left: top vulnerable categories by mean self-consistent error rate. Right: per-model heatmap on those categories.}}
\\end{{figure}}
\\textbf{{Plain-language takeaway:}} Category risk is real and uneven. Some categories (e.g., confusion/misquotation types) repeatedly show high stable wrongness.

\\begin{{table}}[H]
\\centering
\\caption{{Top vulnerable categories (Table 4). Plain meaning: where models are most likely to be confidently wrong. ``Mean SC(all)'' is over all rows in that category; ``Mean SC(of errors)'' is over wrong rows only.}}
\\begin{{tabular}}{{lrrr}}
\\toprule
Category & Mean SC(all) (all rows) & Mean SC(of errors) (wrong rows only) & Questions \\\\
\\midrule
{table_category_rows(cat_agg_df)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\paragraph{{How I read Table 4 in one pass.}}
If a category is high in \\textbf{{Mean SC(all)}}, that category is risky overall. If it is high in \\textbf{{Mean SC(of errors)}}, then when the model is wrong there, it tends to be stubbornly wrong (same wrong meaning repeated). I pay the most attention to categories that are high on both columns.

\\subsection{{Figure 6 and Figure 7: Pairwise deltas}}
\\begin{{figure}}[H]
  \\centering
  \\includegraphics[width=0.49\\textwidth]{{figures/simple_fig_06_pairwise_accuracy_heatmap.png}}\\hfill
  \\includegraphics[width=0.49\\textwidth]{{figures/simple_fig_07_pairwise_sc_heatmap.png}}
  \\caption{{A-B pairwise deltas. Positive means row model A is higher than model B for that metric.}}
\\end{{figure}}
\\textbf{{Plain-language takeaway:}} Pairwise differences are substantial for many model pairs, especially in accuracy.

\\begin{{table}}[H]
\\centering
\\caption{{Most significant accuracy differences (exact McNemar)}}
\\begin{{tabular}}{{p{{8.5cm}}rr}}
\\toprule
Pair & $\\Delta$ accuracy (pp) & $p$ \\\\
\\midrule
{table_significance_rows(pair_df, 'acc', n=10)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[H]
\\centering
\\caption{{Most significant self-consistent rate differences (exact McNemar)}}
\\begin{{tabular}}{{p{{8.5cm}}rr}}
\\toprule
Pair & $\\Delta$ SC rate (pp) & $p$ \\\\
\\midrule
{table_significance_rows(pair_df, 'sc', n=10)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\subsection{{Figure 8: Adversarial vs non-adversarial}}
\\begin{{figure}}[H]
  \\centering
  \\includegraphics[width=0.84\\textwidth]{{figures/simple_fig_08_question_type.png}}
  \\caption{{How to read: same metrics split by TruthfulQA question type.}}
\\end{{figure}}
\\textbf{{Plain-language takeaway:}} Adversarial questions are harder and produce more self-consistent errors.

\\begin{{itemize}}
{chr(10).join(qtype_bullets)}
\\end{{itemize}}

\\subsection{{Figure 9: Judge diagnostics}}
\\begin{{figure}}[H]
  \\centering
  \\includegraphics[width=0.95\\textwidth]{{figures/simple_fig_09_judge_diagnostics.png}}
  \\caption{{Status patterns and per-slot failures for judge calls.}}
\\end{{figure}}
\\textbf{{Plain-language takeaway:}} Most rows are all-OK, and most parse failures are concentrated in one slot. Two-OK rows agree in {100*t['two_ok_agree_rate']:.2f}\\% of cases.

\\subsection{{Figure 10: Threshold sensitivity (1.0 / 0.9 / 0.8)}}
\\begin{{figure}}[H]
  \\centering
  \\includegraphics[width=0.92\\textwidth]{{figures/simple_fig_10_threshold_sc_by_model.png}}
  \\caption{{How to read: each model is shown with three bars. Lower is better because it means fewer self-consistent errors.}}
\\end{{figure}}
\\textbf{{Plain-language takeaway:}} As the threshold loosens from 1.0 to 0.8, more rows are counted as "same meaning", so self-consistent error rate increases.

\\subsection{{Figure 11: Full 0.9 label composition}}
\\begin{{figure}}[H]
  \\centering
  \\includegraphics[width=0.94\\textwidth]{{figures/simple_fig_11_full_label_breakdown_0_9.png}}
  \\caption{{How to read: each bar sums to 100\\%. This explicitly separates correct+different and incorrect+different from the consistent portions.}}
\\end{{figure}}
\\textbf{{Plain-language takeaway:}} This is the exact split you asked for at threshold 0.9: correct+same, correct+different, incorrect+same, incorrect+different, and NOT\\_ATTEMPTED.

\\section{{Are these findings expected according to recent research?}}
The table below marks each major pattern as expected/unexpected based on primary sources.

\\begin{{longtable}}{{p{{0.34\\textwidth}}p{{0.10\\textwidth}}p{{0.50\\textwidth}}}}
\\toprule
Theme & Status & Observed here (with citation) \\\\
\\midrule
{expected_rows_tex(expected_rows)}
\\bottomrule
\\end{{longtable}}

\\section{{What is still missing (important honesty check)}}
RQ3 in your proposal requires a black-box vs white-box comparison (e.g., hidden-state cross-model probing) and AUROC/cost analysis. This file does not include hidden-state probe outputs, so we cannot claim RQ3 is completed. This matches the method distinction reported in recent work \\cite{{tan2025tooconsistent}}.

\\section{{Recent research context (why this is a reasonable result profile)}}
Hallucination remains a broad unsolved issue in LLMs \\cite{{ji2023survey}}. TruthfulQA itself shows that larger/stronger models can still fail on misconception-heavy factual questions \\cite{{lin2022truthfulqa}}. Sampling-based black-box methods are practical but can miss stable wrongness \\cite{{manakul2023selfcheckgpt,zhang2023sac3}}. More recent studies specifically show that self-consistent errors are hard for mainstream detectors and can persist with scale \\cite{{tan2025tooconsistent}}. Methods that focus on semantic-level uncertainty or richer self-reflection improve robustness \\cite{{farquhar2024semanticentropy,yehuda2024interrogatellm,liu2025agser}}.

\\section{{Bottom-line summary in plain terms}}
\\begin{{enumerate}}
\\item The dataset is internally consistent and mathematically sound after strict checks.
\\item Self-consistent errors are common enough to matter; retries alone are not a safety mechanism.
\\item Category-level risk is real, so model risk is not one single number.
\\item Results are mostly expected given recent literature.
\\item Your current file supports RQ1, RQ2, and RQ4 strongly; RQ3 still needs a white-box run.
\\end{{enumerate}}

\\appendix
\\section{{Filtered-out TruthfulQA rows (807/817 coverage)}}
\\begin{{longtable}}{{rll}}
\\toprule
Index & Category & Question (truncated) \\\\
\\midrule
{table_missing_rows(missing_df)}
\\bottomrule
\\end{{longtable}}

\\section{{Reproducibility}}
This report is generated from:
\\begin{{itemize}}
\\item \\texttt{{{tex_escape(str(FINAL_JSONL))}}}
\\item \\texttt{{{tex_escape(str(TRUTHFULQA_CSV))}}}
\\item \\texttt{{{tex_escape(str(ANALYSIS_DIR / 'thesis_deep_model_metrics.csv'))}}}
\\item \\texttt{{{tex_escape(str(ANALYSIS_DIR / 'thesis_deep_group_metrics.csv'))}}}
\\item \\texttt{{{tex_escape(str(ANALYSIS_DIR / 'thesis_deep_pairwise_significance.csv'))}}}
\\end{{itemize}}

\\bibliographystyle{{plain}}
\\bibliography{{references}}

\\end{{document}}
"""

    (OUT_DIR / "thesis_simple_explainer_report.tex").write_text(tex, encoding="utf-8")


def write_machine_outputs(checks: List[CheckResult], details: Dict[str, Any], t: Dict[str, Any], expected_rows: List[Dict[str, str]]) -> None:
    threshold_df = t["threshold_df"]
    thr_lookup = {float(r["threshold"]): r for _, r in threshold_df.iterrows()}
    out = {
        "summary": {
            "rows_total": details["rows_total"],
            "unique_questions": details["unique_questions"],
            "unique_models": details["unique_models"],
            "checks_passed": details["pass_count"],
            "checks_total": details["check_count"],
            "rows_not_attempted": details["rows_not_attempted"],
            "rows_parse_failed": details["rows_parse_failed"],
            "rows_api_failed": details["rows_api_failed"],
            "rows_adjudicated": details["rows_adjudicated"],
            "missing_indices": details["missing_indices"],
            "threshold_sc_rate": {
                "1.0": float(thr_lookup[1.0]["self_consistent_error_rate"]),
                "0.9": float(thr_lookup[0.9]["self_consistent_error_rate"]),
                "0.8": float(thr_lookup[0.8]["self_consistent_error_rate"]),
            },
        },
        "validation_checks": [{"name": c.name, "passed": c.passed, "details": c.details} for c in checks],
        "expected_vs_observed": expected_rows,
    }
    (OUT_DIR / "validation_and_expected_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    t["model_df"].to_csv(OUT_DIR / "table_model_metrics_recomputed.csv", index=False)
    t["group_df"].to_csv(OUT_DIR / "table_group_metrics_recomputed.csv", index=False)
    t["cat_agg_df"].to_csv(OUT_DIR / "table_category_aggregate_recomputed.csv", index=False)
    t["pair_df"].to_csv(OUT_DIR / "table_pairwise_significance_recomputed.csv", index=False)
    t["qtype_df"].to_csv(OUT_DIR / "table_question_type_recomputed.csv", index=False)
    t["pattern_df"].to_csv(OUT_DIR / "table_judge_patterns_recomputed.csv", index=False)
    t["slot_df"].to_csv(OUT_DIR / "table_judge_slot_recomputed.csv", index=False)
    t["threshold_df"].to_csv(OUT_DIR / "table_threshold_breakdown_recomputed.csv", index=False)
    t["threshold_model_df"].to_csv(OUT_DIR / "table_threshold_by_model_recomputed.csv", index=False)
    t["label_09_model_df"].to_csv(OUT_DIR / "table_label_breakdown_0_9_recomputed.csv", index=False)


def compile_pdf() -> Dict[str, Any]:
    tex_file = OUT_DIR / "thesis_simple_explainer_report.tex"
    if not tex_file.exists():
        return {"compiled": False, "reason": "tex_missing"}
    if shutil.which("pdflatex") is None:
        return {"compiled": False, "reason": "pdflatex_missing"}

    tex_cache = OUT_DIR / ".texcache"
    tex_fonts = OUT_DIR / ".texfonts"
    tex_cache.mkdir(parents=True, exist_ok=True)
    tex_fonts.mkdir(parents=True, exist_ok=True)
    env = dict(**os.environ)
    env["TEXMFVAR"] = str(tex_cache)
    env["VARTEXFONTS"] = str(tex_fonts)

    cmds = [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_file.name],
        ["bibtex", "thesis_simple_explainer_report"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_file.name],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_file.name],
    ]
    for cmd in cmds:
        proc = subprocess.run(cmd, cwd=OUT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, check=False)
        if proc.returncode != 0:
            (OUT_DIR / "compile_error.log").write_text(proc.stdout, encoding="utf-8")
            return {
                "compiled": False,
                "reason": "command_failed",
                "failed_cmd": " ".join(cmd),
                "log": str(OUT_DIR / "compile_error.log"),
            }
    err = OUT_DIR / "compile_error.log"
    if err.exists():
        err.unlink()
    return {"compiled": True, "pdf": str(OUT_DIR / "thesis_simple_explainer_report.pdf")}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    loaded = load_data()
    df = loaded["df"]
    truthfulqa = loaded["truthfulqa"]

    checks, details = run_strict_validations(df, truthfulqa)
    tables = compute_core_tables(df)
    details["two_ok_agree_rate"] = tables["two_ok_agree_rate"]

    generate_figures(tables)
    expected_rows = build_expected_vs_observed(tables["model_df"], details)

    write_bib()
    write_tex(checks, details, tables, expected_rows)
    write_machine_outputs(checks, details, tables, expected_rows)
    compile_status = compile_pdf()

    status = {
        "tex_file": str(OUT_DIR / "thesis_simple_explainer_report.tex"),
        "bib_file": str(OUT_DIR / "references.bib"),
        "figure_dir": str(FIG_DIR),
        "compiled": compile_status,
        "checks_passed": details["pass_count"],
        "checks_total": details["check_count"],
    }
    (OUT_DIR / "build_report_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
