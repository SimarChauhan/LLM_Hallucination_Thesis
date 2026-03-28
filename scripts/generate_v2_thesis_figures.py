#!/usr/bin/env python3
"""Generate publication-quality figures and tables for v2 thesis LaTeX report.

Loads v2 evaluated JSONL + TruthfulQA.csv, joins for category/question_type,
produces matplotlib figures and CSVs for category, adversarial, pairwise SC,
label breakdown, judge diagnostics. Writes to data/results/analysis/v2_thesis/.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSONL = (
    PROJECT_ROOT
    / "data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl"
)
DEFAULT_TQA = PROJECT_ROOT / "TruthfulQA.csv"
DEFAULT_OUT = PROJECT_ROOT / "data/results/analysis/v2_thesis"

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

CLOSED = {"Claude Opus 4.6 (Anthropic)", "GPT-5.2 (OpenAI)", "Grok 4 (xAI)"}
OPENW = {"DeepSeek V3.2 (DeepSeek)", "Llama 4 Maverick 17B (Groq)", "Qwen3 Next 80B (OpenRouter)"}

MODEL_SHORT = {
    "Claude Opus 4.6 (Anthropic)": "Claude Opus 4.6",
    "DeepSeek V3.2 (DeepSeek)": "DeepSeek V3.2",
    "GPT-5.2 (OpenAI)": "GPT-5.2",
    "Grok 4 (xAI)": "Grok 4",
    "Llama 4 Maverick 17B (Groq)": "Llama 4 Maverick",
    "Qwen3 Next 80B (OpenRouter)": "Qwen3 Next 80B",
}


def qid_to_idx(qid: Any) -> Optional[float]:
    if not isinstance(qid, str):
        return None
    m = re.search(r"truthfulqa_csv_(\d+)$", str(qid))
    return float(m.group(1)) if m else None


def mcnemar_p(b_only: int, c_only: int) -> float:
    n = b_only + c_only
    if n <= 0:
        return 1.0
    tail = 0.0
    for i in range(0, min(b_only, c_only) + 1):
        tail += math.comb(n, i) * (0.5**n)
    return min(1.0, 2.0 * tail)


def load_and_merge(
    jsonl_path: Path,
    tqa_path: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["q_idx"] = df["question_id"].map(qid_to_idx)

    tqa = pd.read_csv(tqa_path).reset_index(drop=False).rename(
        columns={"index": "q_idx", "Category": "category", "Type": "question_type"}
    )
    df = df.merge(tqa[["q_idx", "category", "question_type"]], on="q_idx", how="left")

    df["is_correct"] = (df["correctness_grade"] == "CORRECT").astype(int)
    df["is_incorrect"] = (df["correctness_grade"] == "INCORRECT").astype(int)
    df["is_na"] = (df["correctness_grade"] == "NOT_ATTEMPTED").astype(int)
    df["is_sc_09"] = (df["error_label_0.9"] == "self_consistent_error").astype(int)
    df["is_sc_10"] = (df["error_label_1.0"] == "self_consistent_error").astype(int)

    def any_status(r, status: str) -> bool:
        s = r.get("correctness_judge_statuses")
        if not isinstance(s, list):
            return False
        return any(x == status for x in s)

    df["any_parse_failed"] = df.apply(lambda r: any_status(r, "PARSE_FAILED"), axis=1)
    df["any_api_failed"] = df.apply(lambda r: any_status(r, "API_FAILED"), axis=1)
    df["adjudicated"] = (df["correctness_decision_source"] == "ADJUDICATOR").astype(int)

    df["model_short"] = df["model"].map(lambda m: MODEL_SHORT.get(str(m), str(m)))
    return df


def compute_tables(df: pd.DataFrame) -> Dict[str, Any]:
    models = sorted(df["model"].unique().tolist())
    model_rows = []
    for model, sub in df.groupby("model"):
        n = len(sub)
        correct = int(sub["is_correct"].sum())
        incorrect = int(sub["is_incorrect"].sum())
        sc09 = int(sub["is_sc_09"].sum())
        sc10 = int(sub["is_sc_10"].sum())
        model_rows.append({
            "model": model,
            "model_short": MODEL_SHORT.get(model, model),
            "n": n,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": correct / n if n else 0,
            "not_attempted": int(sub["is_na"].sum()),
            "not_attempted_rate": float(sub["is_na"].mean()),
            "self_consistent_0_9": sc09,
            "self_consistent_1_0": sc10,
            "self_consistent_rate_total": sc09 / n if n else 0,
            "self_consistent_rate_of_errors": (sc09 / incorrect) if incorrect else float("nan"),
        })
    model_df = pd.DataFrame(model_rows).sort_values("accuracy", ascending=False)

    # Label breakdown 0.9 and 1.0
    label_09_rows = []
    label_10_rows = []
    for thr, col in [(0.9, "error_label_0.9"), (1.0, "error_label_1.0")]:
        for model, sub in df.groupby("model"):
            n = len(sub)
            row = {"model": model, "model_short": MODEL_SHORT.get(model, model), "n": n}
            for label in FIVE_LABELS:
                c = int((sub[col] == label).sum())
                row[f"{label}_count"] = c
                row[f"{label}_rate"] = c / n if n else 0
            row["correct_same_rate"] = row["reliably_correct_rate"]
            row["correct_different_rate"] = row["fragile_correct_rate"]
            row["incorrect_same_rate"] = row["self_consistent_error_rate"]
            row["incorrect_different_rate"] = row["inconsistent_error_rate"]
            if thr == 0.9:
                label_09_rows.append(row)
            else:
                label_10_rows.append(row)
    label_09_df = pd.DataFrame(label_09_rows)
    label_10_df = pd.DataFrame(label_10_rows)

    # Group metrics
    group_rows = []
    for name, model_set in [("Closed API", CLOSED), ("Open-weight API", OPENW)]:
        sub = df[df["model"].isin(model_set)]
        incorrect = int(sub["is_incorrect"].sum())
        group_rows.append({
            "group": name,
            "rows": len(sub),
            "accuracy": float(sub["is_correct"].mean()),
            "self_consistent_rate_total": float(sub["is_sc_09"].mean()),
            "self_consistent_rate_of_errors": (int(sub["is_sc_09"].sum()) / incorrect) if incorrect else float("nan"),
            "not_attempted_rate": float(sub["is_na"].mean()),
        })
    group_df = pd.DataFrame(group_rows)

    # Category
    cat_rows = []
    cat_df = df[df["category"].notna()].copy()
    for (model, cat), sub in cat_df.groupby(["model", "category"]):
        n = len(sub)
        incorrect = int(sub["is_incorrect"].sum())
        sc = int(sub["is_sc_09"].sum())
        cat_rows.append({
            "model": model,
            "model_short": MODEL_SHORT.get(model, model),
            "category": cat,
            "n": n,
            "incorrect": incorrect,
            "sc_rate_total": sc / n if n else float("nan"),
            "sc_rate_errors": sc / incorrect if incorrect else float("nan"),
            "accuracy": float(sub["is_correct"].mean()),
        })
    cat_by_model = pd.DataFrame(cat_rows) if cat_rows else pd.DataFrame()

    cat_agg_rows = []
    if not cat_by_model.empty:
        for cat, sub in cat_by_model.groupby("category"):
            sc_errors = sub["sc_rate_errors"].replace([np.inf, -np.inf], np.nan).dropna()
            cat_agg_rows.append({
                "category": cat,
                "mean_sc_rate_total": float(sub["sc_rate_total"].mean()),
                "std_sc_rate_total": float(sub["sc_rate_total"].std(ddof=0)) if len(sub) > 1 else 0,
                "mean_sc_rate_errors": float(sc_errors.mean()) if len(sc_errors) else float("nan"),
                "mean_accuracy": float(sub["accuracy"].mean()),
                "support_rows": int(sub["n"].sum()),
                "support_questions_per_model": float(sub["n"].mean()),
            })
    cat_agg_df = pd.DataFrame(cat_agg_rows).sort_values("mean_sc_rate_total", ascending=False) if cat_agg_rows else pd.DataFrame()

    # Adversarial split
    qtype_rows = []
    for qtype, sub in df.groupby("question_type"):
        if pd.isna(qtype) or str(qtype).strip() == "":
            continue
        incorrect = int(sub["is_incorrect"].sum())
        qtype_rows.append({
            "question_type": str(qtype),
            "rows": len(sub),
            "accuracy": float(sub["is_correct"].mean()),
            "sc_rate_total": float(sub["is_sc_09"].mean()),
            "sc_rate_errors": (int(sub["is_sc_09"].sum()) / incorrect) if incorrect else float("nan"),
            "na_rate": float(sub["is_na"].mean()),
        })
    qtype_df = pd.DataFrame(qtype_rows) if qtype_rows else pd.DataFrame()

    # Pairwise (accuracy + SC)
    pair_rows = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            A = df[df["model"] == a][["question_id", "is_correct", "is_sc_09", "is_na"]].rename(
                columns={"is_correct": "acc_a", "is_sc_09": "sc_a", "is_na": "na_a"}
            )
            B = df[df["model"] == b][["question_id", "is_correct", "is_sc_09", "is_na"]].rename(
                columns={"is_correct": "acc_b", "is_sc_09": "sc_b", "is_na": "na_b"}
            )
            m = A.merge(B, on="question_id", how="inner")
            for metric, xa_name, xb_name in [
                ("acc", "acc_a", "acc_b"),
                ("sc", "sc_a", "sc_b"),
                ("na", "na_a", "na_b"),
            ]:
                xa = m[xa_name].astype(int)
                xb = m[xb_name].astype(int)
                b_only = int(((xa == 1) & (xb == 0)).sum())
                c_only = int(((xa == 0) & (xb == 1)).sum())
                pair_rows.append({
                    "model_a": a,
                    "model_b": b,
                    "metric": metric,
                    "n": int(len(m)),
                    "rate_a": float(xa.mean()),
                    "rate_b": float(xb.mean()),
                    "delta_a_minus_b": float(xa.mean() - xb.mean()),
                    "discordant_a_only": b_only,
                    "discordant_b_only": c_only,
                    "p_exact_mcnemar": mcnemar_p(b_only, c_only),
                })
    pair_df = pd.DataFrame(pair_rows) if pair_rows else pd.DataFrame()

    # Judge diagnostics
    from collections import Counter
    patterns: Counter = Counter()
    slot_ok: Dict[int, int] = {}
    slot_parse: Dict[int, int] = {}
    slot_api: Dict[int, int] = {}
    for _, r in df.iterrows():
        statuses = r.get("correctness_judge_statuses")
        if not isinstance(statuses, list):
            continue
        patterns[tuple(statuses)] += 1
        for i, s in enumerate(statuses):
            if s == "OK":
                slot_ok[i] = slot_ok.get(i, 0) + 1
            elif s == "PARSE_FAILED":
                slot_parse[i] = slot_parse.get(i, 0) + 1
            elif s == "API_FAILED":
                slot_api[i] = slot_api.get(i, 0) + 1

    slot_df = pd.DataFrame({
        "slot": [1, 2, 3],
        "OK": [slot_ok.get(0, 0), slot_ok.get(1, 0), slot_ok.get(2, 0)],
        "PARSE_FAILED": [slot_parse.get(0, 0), slot_parse.get(1, 0), slot_parse.get(2, 0)],
        "API_FAILED": [slot_api.get(0, 0), slot_api.get(1, 0), slot_api.get(2, 0)],
    })
    pattern_df = pd.DataFrame(
        [{"pattern": " | ".join(str(x) for x in k), "rows": v} for k, v in patterns.most_common(10)]
    )

    # Threshold by model (1.0, 0.9, 0.8)
    thresh_model_rows = []
    for thr in [1.0, 0.9, 0.8]:
        col = f"error_label_{thr:.1f}"
        if col not in df.columns:
            continue
        for model, sub in df.groupby("model"):
            n = len(sub)
            sc = int((sub[col] == "self_consistent_error").sum())
            thresh_model_rows.append({
                "model": model,
                "model_short": MODEL_SHORT.get(model, model),
                "threshold": thr,
                "self_consistent_error_rate": sc / n if n else 0,
            })
    thresh_model_df = pd.DataFrame(thresh_model_rows) if thresh_model_rows else pd.DataFrame()

    return {
        "model_df": model_df,
        "label_09_df": label_09_df,
        "label_10_df": label_10_df,
        "group_df": group_df,
        "cat_by_model_df": cat_by_model,
        "cat_agg_df": cat_agg_df,
        "qtype_df": qtype_df,
        "pair_df": pair_df,
        "pattern_df": pattern_df,
        "slot_df": slot_df,
        "thresh_model_df": thresh_model_df,
        "rows_parse_failed": int(df["any_parse_failed"].sum()),
        "rows_api_failed": int(df["any_api_failed"].sum()),
        "rows_adjudicated": int(df["adjudicated"].sum()),
    }


def make_pairwise_matrix(pair_df: pd.DataFrame, metric: str, models: List[str]) -> pd.DataFrame:
    mat = pd.DataFrame(np.nan, index=models, columns=models)
    sub = pair_df[pair_df["metric"] == metric]
    for _, r in sub.iterrows():
        a, b = r["model_a"], r["model_b"]
        d = float(r["delta_a_minus_b"])
        mat.loc[a, b] = d
        mat.loc[b, a] = -d
    arr = mat.to_numpy(copy=True)
    np.fill_diagonal(arr, 0.0)
    labels = [MODEL_SHORT.get(m, m) for m in models]
    return pd.DataFrame(arr, index=labels, columns=labels)


def generate_figures(t: Dict[str, Any], out_dir: Path) -> None:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    model_df = t["model_df"]
    label_09_df = t["label_09_df"]
    label_10_df = t["label_10_df"]
    group_df = t["group_df"]
    cat_agg_df = t["cat_agg_df"]
    cat_by_model = t["cat_by_model_df"]
    pair_df = t["pair_df"]
    qtype_df = t["qtype_df"]
    pattern_df = t["pattern_df"]
    slot_df = t["slot_df"]
    thresh_model_df = t["thresh_model_df"]

    # Fig 1: Model headlines
    if not model_df.empty:
        f, axes = plt.subplots(1, 3, figsize=(16, 5))
        xdf = model_df.sort_values("accuracy", ascending=False)
        sns.barplot(data=xdf, x="model_short", y="accuracy", color="#2a9d8f", ax=axes[0])
        axes[0].set_title("Model Accuracy")
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis="x", rotation=35)
        axes[0].set_xlabel("")
        sns.barplot(data=xdf, x="model_short", y="self_consistent_rate_total", color="#e76f51", ax=axes[1])
        axes[1].set_title("Self-Consistent Error Rate (all rows)")
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis="x", rotation=35)
        axes[1].set_xlabel("")
        sns.barplot(data=xdf, x="model_short", y="not_attempted_rate", color="#457b9d", ax=axes[2])
        axes[2].set_title("NOT_ATTEMPTED Rate")
        axes[2].set_ylim(0, 1)
        axes[2].tick_params(axis="x", rotation=35)
        axes[2].set_xlabel("")
        for ax in axes:
            ax.grid(axis="y", alpha=0.25)
        f.tight_layout()
        f.savefig(figures_dir / "fig_01_model_headlines.png", dpi=220, bbox_inches="tight")
        plt.close(f)

    # Fig 2: Error composition
    if not model_df.empty:
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
        ax.set_title("Error composition by model")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.25)
        f.tight_layout()
        f.savefig(figures_dir / "fig_02_error_breakdown.png", dpi=220, bbox_inches="tight")
        plt.close(f)

    # Fig 3: Group comparison
    if not group_df.empty:
        melt = group_df.melt(
            id_vars=["group"],
            value_vars=["accuracy", "self_consistent_rate_total", "not_attempted_rate"],
            var_name="metric",
            value_name="value",
        )
        melt["metric"] = melt["metric"].map({
            "accuracy": "Accuracy",
            "self_consistent_rate_total": "SC rate (all rows)",
            "not_attempted_rate": "NOT_ATTEMPTED rate",
        })
        f, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=melt, x="metric", y="value", hue="group", ax=ax, palette="Set2")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Rate")
        ax.set_title("Closed-API vs Open-weight-API")
        ax.grid(axis="y", alpha=0.25)
        f.tight_layout()
        f.savefig(figures_dir / "fig_03_group_comparison.png", dpi=220, bbox_inches="tight")
        plt.close(f)

    # Fig 4-5: Category
    if not cat_agg_df.empty:
        top = cat_agg_df.head(12)
        f, ax = plt.subplots(figsize=(10, 6.5))
        sns.barplot(data=top, y="category", x="mean_sc_rate_total", color="#d62828", ax=ax)
        for i, r in top.reset_index(drop=True).iterrows():
            ax.text(float(r["mean_sc_rate_total"]) + 0.005, i, f"n={int(round(r['support_questions_per_model']))}", va="center", fontsize=9)
        ax.set_xlabel("Mean self-consistent error rate (all rows)")
        ax.set_title("Top vulnerable categories")
        ax.grid(axis="x", alpha=0.25)
        f.tight_layout()
        f.savefig(figures_dir / "fig_04_top_categories.png", dpi=220, bbox_inches="tight")
        plt.close(f)

        if not cat_by_model.empty:
            cats = top["category"].tolist()
            h = cat_by_model[cat_by_model["category"].isin(cats)]
            if not h.empty:
                piv = h.pivot(index="category", columns="model_short", values="sc_rate_errors")
                f, ax = plt.subplots(figsize=(10.8, 7.2))
                sns.heatmap(piv, cmap="YlOrRd", linewidths=0.4, linecolor="white", cbar_kws={"label": "SC rate among errors"}, ax=ax)
                ax.set_title("Category x model heatmap")
                f.tight_layout()
                f.savefig(figures_dir / "fig_05_category_heatmap.png", dpi=220, bbox_inches="tight")
                plt.close(f)

    # Fig 6-7: Pairwise heatmaps
    if not pair_df.empty and not model_df.empty:
        models = model_df["model"].tolist()
        for metric, fname, title in [
            ("acc", "fig_06_pairwise_accuracy_heatmap.png", "Pairwise accuracy delta (A-B, pp)"),
            ("sc", "fig_07_pairwise_sc_heatmap.png", "Pairwise self-consistent rate delta (A-B, pp)"),
        ]:
            mat = make_pairwise_matrix(pair_df, metric, models)
            f, ax = plt.subplots(figsize=(7.5, 6.4))
            sns.heatmap(mat * 100, annot=True, fmt=".1f", cmap="RdBu_r", center=0, cbar_kws={"label": "Delta (pp)"}, ax=ax)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=35)
            f.tight_layout()
            f.savefig(figures_dir / fname, dpi=220, bbox_inches="tight")
            plt.close(f)

    # Fig 8: Adversarial
    if not qtype_df.empty:
        q_melt = qtype_df.melt(
            id_vars=["question_type"],
            value_vars=["accuracy", "sc_rate_total", "na_rate"],
            var_name="metric",
            value_name="value",
        )
        q_melt["metric"] = q_melt["metric"].map({
            "accuracy": "Accuracy",
            "sc_rate_total": "SC rate (all rows)",
            "na_rate": "NOT_ATTEMPTED rate",
        })
        f, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=q_melt, x="metric", y="value", hue="question_type", ax=ax, palette="Set1")
        ax.set_ylim(0, 1)
        ax.set_title("Adversarial vs Non-Adversarial")
        ax.grid(axis="y", alpha=0.25)
        f.tight_layout()
        f.savefig(figures_dir / "fig_08_question_type.png", dpi=220, bbox_inches="tight")
        plt.close(f)

    # Fig 9: Judge diagnostics
    f, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    top_pat = pattern_df.head(6)
    if not top_pat.empty:
        sns.barplot(data=top_pat, y="pattern", x="rows", color="#6d597a", ax=axes[0])
    axes[0].set_title("Top judge status patterns")
    axes[0].set_xlabel("Rows")
    slot_m = slot_df.melt(id_vars=["slot"], var_name="status", value_name="rows")
    sns.barplot(data=slot_m, x="slot", y="rows", hue="status", ax=axes[1])
    axes[1].set_title("Judge status by slot")
    axes[1].set_xlabel("Judge slot")
    f.tight_layout()
    f.savefig(figures_dir / "fig_09_judge_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(f)

    # Fig 10: Threshold by model
    if not thresh_model_df.empty:
        f, ax = plt.subplots(figsize=(10.5, 5.5))
        tdf = thresh_model_df.copy()
        tdf["threshold_label"] = tdf["threshold"].map(lambda x: f"{x:.1f}")
        sns.barplot(
            data=tdf,
            x="model_short",
            y="self_consistent_error_rate",
            hue="threshold_label",
            order=sorted(model_df["model_short"].tolist()) if not model_df.empty else None,
            hue_order=["1.0", "0.9", "0.8"],
            palette=["#8ecae6", "#219ebc", "#023047"],
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_title("Self-consistent error rate by threshold")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
        f.tight_layout()
        f.savefig(figures_dir / "fig_10_threshold_sc_by_model.png", dpi=220, bbox_inches="tight")
        plt.close(f)

    # Fig 11: Full label breakdown 0.9
    if not label_09_df.empty:
        xdf = label_09_df.sort_values("model_short")
        f, ax = plt.subplots(figsize=(11, 5.8))
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
        ax.set_title("Full 0.9 label breakdown by model")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        f.tight_layout()
        f.savefig(figures_dir / "fig_11_full_label_breakdown_0_9.png", dpi=220, bbox_inches="tight")
        plt.close(f)

    # Fig 12: Full label breakdown 1.0
    if not label_10_df.empty:
        xdf = label_10_df.sort_values("model_short")
        f, ax = plt.subplots(figsize=(11, 5.8))
        x = np.arange(len(xdf))
        c_same = xdf["reliably_correct_rate"].to_numpy()
        c_diff = xdf["fragile_correct_rate"].to_numpy()
        i_same = xdf["self_consistent_error_rate"].to_numpy()
        i_diff = xdf["inconsistent_error_rate"].to_numpy()
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
        ax.set_title("Full 1.0 label breakdown by model")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        f.tight_layout()
        f.savefig(figures_dir / "fig_12_full_label_breakdown_1_0.png", dpi=220, bbox_inches="tight")
        plt.close(f)

    # Copy pipeline flowchart from simple report if exists
    src_flowchart = PROJECT_ROOT / "data/results/analysis/final_analysis_ready/latex_report_simple/figures/simple_fig_14_pipeline_flowchart.png"
    if src_flowchart.exists():
        import shutil
        shutil.copy(src_flowchart, figures_dir / "fig_14_pipeline_flowchart.png")


def run_validation(df: pd.DataFrame) -> Dict[str, Any]:
    checks = {}
    n = len(df)
    n_q = int(df["question_id"].nunique())
    n_m = int(df["model"].nunique())
    checks["row_count_equals_q_x_m"] = n == n_q * n_m
    checks["no_duplicates"] = int(df.duplicated(subset=["question_id", "model"]).sum()) == 0
    label_09 = df["error_label_0.9"].astype(str)
    part = {k: int((label_09 == k).sum()) for k in FIVE_LABELS}
    checks["label_partition_complete"] = sum(part.values()) == n
    checks["rows_total"] = n
    checks["unique_questions"] = n_q
    checks["unique_models"] = n_m
    checks["partition_counts"] = part
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_JSONL))
    parser.add_argument("--truthfulqa", type=str, default=str(DEFAULT_TQA))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    jsonl_path = Path(args.input)
    tqa_path = Path(args.truthfulqa)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    if not tqa_path.exists():
        raise FileNotFoundError(f"TruthfulQA not found: {tqa_path}")

    df = load_and_merge(jsonl_path, tqa_path)
    t = compute_tables(df)

    # Write CSVs
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if not t["cat_agg_df"].empty:
        t["cat_agg_df"].to_csv(tables_dir / "category_aggregate.csv", index=False)
    if not t["qtype_df"].empty:
        t["qtype_df"].to_csv(tables_dir / "adversarial_split.csv", index=False)
    if not t["pair_df"].empty:
        sc_pairs = t["pair_df"][t["pair_df"]["metric"] == "sc"]
        sc_pairs.to_csv(tables_dir / "pairwise_sc.csv", index=False)
    if not t["label_09_df"].empty:
        t["label_09_df"].to_csv(tables_dir / "label_breakdown_0_9.csv", index=False)
    if not t["label_10_df"].empty:
        t["label_10_df"].to_csv(tables_dir / "label_breakdown_1_0.csv", index=False)

    judge_diag = {
        "rows_parse_failed": t["rows_parse_failed"],
        "rows_api_failed": t["rows_api_failed"],
        "rows_adjudicated": t["rows_adjudicated"],
    }
    (out_dir / "judge_diagnostics.json").write_text(json.dumps(judge_diag, indent=2) + "\n", encoding="utf-8")
    t["slot_df"].to_csv(tables_dir / "judge_slot_counts.csv", index=False)

    # Validation
    validation = run_validation(df)
    (out_dir / "validation_checks.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")

    # Figures
    generate_figures(t, out_dir)

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
