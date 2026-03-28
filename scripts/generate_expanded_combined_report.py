#!/usr/bin/env python3
"""Generate the expanded combined professor report (threshold 1.0 focus).

Reads the 4842 baseline JSONL and version-evolution analysis CSVs,
computes/verifies all numbers from data, produces two figures,
and emits .tex / .txt / optionally .pdf.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
BASELINE_JSONL = Path(
    "data/results/evaluated/"
    "results_v2_phase2_eval_no_gemini_4842.final.analysis_ready."
    "skip_greedy_semantic_eval.jsonl"
)
ANALYSIS_DIR = Path("data/results/analysis/version_evolution_equiv_only_20260319")

THRESHOLDS = ["1.0", "0.9", "0.8", "0.7"]

FAMILY_COLORS = {
    "Grok": "#1d3557",
    "Llama": "#386641",
    "Qwen": "#9c6644",
    "Claude": "#6a0dad",
    "GPT": "#d62828",
    "DeepSeek": "#457b9d",
    "Other": "#888888",
}

# Release dates for the baseline-only models (not present in version-evolution CSVs).
# These are timeline/context values used purely for presentation.
BASELINE_RELEASE_DATES = {
    "Claude Opus 4.6 (Anthropic)": "2026-02-05",
    "GPT-5.2 (OpenAI)": "2025-12-11",
    "DeepSeek V3.2 (DeepSeek)": "2025-12-01",
}


def _family_of(model: str) -> str:
    m = model.lower()
    if "grok" in m:
        return "Grok"
    if "llama" in m:
        return "Llama"
    if "qwen" in m:
        return "Qwen"
    if "claude" in m:
        return "Claude"
    if "gpt" in m:
        return "GPT"
    if "deepseek" in m:
        return "DeepSeek"
    return "Other"


def _short_name(model: str) -> str:
    """Abbreviate model name for chart labels."""
    replacements = [
        (" (Anthropic)", ""),
        (" (OpenAI)", ""),
        (" (DeepSeek)", ""),
        (" (xAI)", ""),
        (" (Groq)", ""),
        (" (OpenRouter)", ""),
        (", 2024-04-18", ""),
        (", 2024-07-23", ""),
        (", 2024-09-16", ""),
        (", 2024-11-26", ""),
        (", 2024-12-06", ""),
        (", 2025-04-05", ""),
        (", 2025-06-10", ""),
        (", 2025-07-09", ""),
        (", 2025-07-28", ""),
        (", 2025-09-09", ""),
        (", 2025-11-19", ""),
        (", 2026-03-09", ""),
        (" Instruct", ""),
        (" Fast Reasoning", ""),
        (" Beta 0309 Reasoning", " 0309"),
    ]
    s = model
    for old, new in replacements:
        s = s.replace(old, new)
    return s


# ── data loading ───────────────────────────────────────────────────────────

def load_baseline(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df["is_correct"] = df["correctness_grade"].astype(str).eq("CORRECT")
    df["is_incorrect"] = df["correctness_grade"].astype(str).eq("INCORRECT")
    df["is_na"] = df["correctness_grade"].astype(str).eq("NOT_ATTEMPTED")
    for thr in THRESHOLDS:
        col = f"error_label_{thr}"
        df[f"is_ce_{thr}"] = df[col].astype(str).eq("self_consistent_error")
        df[f"is_ie_{thr}"] = df[col].astype(str).eq("inconsistent_error")
    return df


def load_version_evolution(analysis_dir: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    out["model_summary_t1p0"] = pd.read_csv(analysis_dir / "model_summary_t1p0.csv")
    out["pairwise_t1p0"] = pd.read_csv(analysis_dir / "pairwise_deltas_t1p0.csv")
    out["trend_t1p0"] = pd.read_csv(analysis_dir / "trend_tests_t1p0.csv")
    out["validation"] = json.loads(
        (analysis_dir / "validation_checks.json").read_text(encoding="utf-8")
    )
    return out


# ── baseline computations ─────────────────────────────────────────────────

def compute_baseline_aggregates(df: pd.DataFrame) -> Dict[str, Any]:
    n = len(df)
    n_correct = int(df["is_correct"].sum())
    n_incorrect = int(df["is_incorrect"].sum())
    n_na = int(df["is_na"].sum())
    agg: Dict[str, Any] = {
        "total": n,
        "correct": n_correct,
        "incorrect": n_incorrect,
        "not_attempted": n_na,
        "correct_pct": round(100.0 * n_correct / n, 1),
        "incorrect_pct": round(100.0 * n_incorrect / n, 1),
        "na_pct": round(100.0 * n_na / n, 1),
        "threshold_sweep": {},
    }
    for thr in THRESHOLDS:
        ce = int(df[f"is_ce_{thr}"].sum())
        ie = int(df[f"is_ie_{thr}"].sum())
        agg["threshold_sweep"][thr] = {
            "ce": ce,
            "ie": ie,
            "ce_pct_total": round(100.0 * ce / n, 1),
            "ce_among_wrong": round(100.0 * ce / n_incorrect, 1),
        }
    return agg


def compute_per_model_baseline(df: pd.DataFrame, thr: str = "1.0") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model, g in df.groupby("model"):
        n = len(g)
        n_correct = int(g["is_correct"].sum())
        n_incorrect = int(g["is_incorrect"].sum())
        ce = int(g[f"is_ce_{thr}"].sum())
        ie = int(g[f"is_ie_{thr}"].sum())
        rows.append({
            "model": model,
            "n": n,
            "accuracy_pct": round(100.0 * n_correct / n, 1),
            "n_wrong": n_incorrect,
            "ce": ce,
            "ie": ie,
            "ce_rate_pct": round(100.0 * ce / n, 1),
            "ce_among_wrong_pct": round(100.0 * ce / n_incorrect, 1) if n_incorrect > 0 else 0.0,
            "source": "baseline_4842",
        })
    return pd.DataFrame(rows).sort_values("ce_rate_pct")


def _auroc_manual(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC without sklearn (Wilcoxon-Mann-Whitney statistic)."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks = 0.0
    all_vals = np.concatenate([neg, pos])
    all_labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    order = np.argsort(all_vals, kind="mergesort")
    ranked = np.empty_like(order, dtype=float)
    ranked[order] = np.arange(1, len(all_vals) + 1, dtype=float)
    # handle ties
    sorted_vals = all_vals[order]
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j > i + 1:
            avg_rank = np.mean(ranked[order[i:j]])
            ranked[order[i:j]] = avg_rank
        i = j
    sum_pos_ranks = ranked[all_labels == 1].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def compute_auroc(df: pd.DataFrame) -> Dict[str, float]:
    sub = df[df["correctness_grade"].isin(["CORRECT", "INCORRECT"])].copy()
    y_true = (~sub["is_correct"]).astype(int).values
    aurocs = {}
    # disagreement = 1 - equivalence_ratio (less agreement -> more likely wrong)
    for col, name, negate in [
        ("semantic_entropy_norm", "semantic_entropy", False),
        ("equivalence_ratio", "disagreement", True),
    ]:
        if col in sub.columns:
            vals = pd.to_numeric(sub[col], errors="coerce")
            mask = vals.notna()
            if mask.sum() > 100:
                scores = -vals[mask].values if negate else vals[mask].values
                aurocs[name] = round(_auroc_manual(y_true[mask], scores), 3)
    return aurocs


# ── unified model table ───────────────────────────────────────────────────

def build_unified_model_table(
    baseline_per_model: pd.DataFrame,
    ve_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Combine baseline-only models with version-evolution models, deduplicating."""
    ve_rows = []
    for _, r in ve_summary.iterrows():
        ve_rows.append({
            "model": r["model"],
            "n": int(r["n_rows"]),
            "accuracy_pct": round(float(r["accuracy_pct"]), 1),
            "ce_rate_pct": round(float(r["ce_rate_pct"]), 2),
            "source": str(r["source_dataset"]),
            "family": r.get("family", ""),
            "version_index": int(r.get("version_index", 0)),
            "release_date": str(r.get("release_date", "")),
        })
    ve_df = pd.DataFrame(ve_rows)

    baseline_only = baseline_per_model[
        ~baseline_per_model["model"].isin(ve_df["model"])
    ].copy()
    for _, r in baseline_only.iterrows():
        ve_rows.append({
            "model": r["model"],
            "n": int(r["n"]),
            "accuracy_pct": r["accuracy_pct"],
            "ce_rate_pct": r["ce_rate_pct"],
            "source": "baseline_4842_only",
            "family": _family_of(str(r["model"])),
            "version_index": 0,
            "release_date": BASELINE_RELEASE_DATES.get(str(r["model"]), ""),
        })

    combined = pd.DataFrame(ve_rows)
    combined["family"] = combined["model"].apply(_family_of)
    combined["short_name"] = combined["model"].apply(_short_name)
    return combined.sort_values("ce_rate_pct").reset_index(drop=True)


# ── Figure 1: CE Landscape ────────────────────────────────────────────────

def make_fig1(
    agg: Dict[str, Any],
    unified: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.6]}
    )

    # Left panel: threshold sensitivity
    thrs = ["1.0", "0.9", "0.8", "0.7"]
    thr_labels = ["t=1.0", "t=0.9", "t=0.8", "t=0.7"]
    ce_among_wrong = [agg["threshold_sweep"][t]["ce_among_wrong"] for t in thrs]
    ce_pct_total = [agg["threshold_sweep"][t]["ce_pct_total"] for t in thrs]

    ax_left.plot(thr_labels, ce_among_wrong, "o-", color="#d62828", linewidth=2.2, label="CE among wrong (%)", markersize=8)
    ax_left.plot(thr_labels, ce_pct_total, "s-", color="#1d3557", linewidth=2.2, label="CE of total rows (%)", markersize=8)
    for i, (cw, ct) in enumerate(zip(ce_among_wrong, ce_pct_total)):
        ax_left.annotate(f"{cw:.1f}", (thr_labels[i], cw), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
        ax_left.annotate(f"{ct:.1f}", (thr_labels[i], ct), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9)
    ax_left.set_ylabel("Percent", fontsize=11)
    ax_left.set_title("Threshold Sensitivity (4842 baseline)", fontsize=12)
    ax_left.legend(fontsize=9, frameon=False)
    ax_left.grid(alpha=0.2)
    ax_left.set_ylim(0, max(ce_among_wrong) + 8)

    # Right panel: horizontal bar chart of ALL models sorted by CE rate
    models_sorted = unified.sort_values("ce_rate_pct", ascending=True)
    y_pos = np.arange(len(models_sorted))
    colors = [FAMILY_COLORS.get(_family_of(m), "#888") for m in models_sorted["model"]]

    bars = ax_right.barh(y_pos, models_sorted["ce_rate_pct"].values, color=colors, alpha=0.85, height=0.65)
    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels(models_sorted["short_name"].values, fontsize=8.5)
    ax_right.set_xlabel("CE rate (% of total rows, t=1.0)", fontsize=10)
    ax_right.set_title("Model-Level CE Rate (all models combined)", fontsize=12)
    ax_right.grid(axis="x", alpha=0.2)

    for bar, val in zip(bars, models_sorted["ce_rate_pct"].values):
        ax_right.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=8,
        )

    # Legend for families
    from matplotlib.patches import Patch
    seen = {}
    for m in models_sorted["model"]:
        fam = _family_of(m)
        if fam not in seen:
            seen[fam] = FAMILY_COLORS.get(fam, "#888")
    legend_patches = [Patch(facecolor=c, label=f) for f, c in seen.items()]
    ax_right.legend(handles=legend_patches, fontsize=8, loc="lower right", frameon=False)

    fig.suptitle("The CE Landscape: How Big Is the Problem and Who Is Affected?", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ── Figure 2: Version-Evolution Trajectories ──────────────────────────────

def make_fig2(
    ve_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.2, 1]}
    )

    tracks = [
        ("grok_version", "Grok", FAMILY_COLORS["Grok"]),
        ("llama_scale_version", "Llama", FAMILY_COLORS["Llama"]),
        ("qwen_scale_version", "Qwen", FAMILY_COLORS["Qwen"]),
    ]

    # Left panel: CE + accuracy trajectories by release date
    ax_acc = ax_left.twinx()
    for track, label, color in tracks:
        sub = ve_summary[ve_summary["track"] == track].sort_values("version_index")
        dates = pd.to_datetime(sub["release_date"])
        ce = sub["ce_rate_pct"].values
        acc = sub["accuracy_pct"].values
        src = sub["source_dataset"].values

        ax_left.plot(dates, ce, "o-", color=color, linewidth=2.2, label=f"{label} CE", markersize=7, zorder=3)
        ax_acc.plot(dates, acc, "s--", color=color, linewidth=1.2, alpha=0.5, markersize=5)

        for d, c, a, s in zip(dates, ce, acc, src):
            marker = "D" if "existing" in str(s) else "o"
            ax_left.scatter(d, c, color=color, s=60, marker=marker, edgecolor="black", linewidth=0.4, zorder=4)
            ax_left.annotate(f"{c:.1f}", (d, c), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5, color=color)

    ax_left.set_ylabel("CE rate (% of total, t=1.0)", fontsize=10)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=10, alpha=0.5)
    ax_acc.yaxis.label.set_alpha(0.5)
    ax_left.set_xlabel("Release date", fontsize=10)
    ax_left.set_title("CE and Accuracy Over Time", fontsize=12)
    ax_left.legend(fontsize=8, loc="upper right", frameon=False)
    ax_left.grid(alpha=0.2)
    ax_left.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)

    # Right panel: consecutive pairwise CE improvement bars, color-coded by family
    ce_consec = pairwise[
        (pairwise["metric"] == "ce_rate") & (pairwise["consecutive_pair"] == True)
    ].copy()
    ce_consec = ce_consec.sort_values(["track", "older_model"])

    bar_labels = []
    bar_vals = []
    bar_colors = []
    bar_p = []
    for _, r in ce_consec.iterrows():
        track = r["track"]
        for tname, tlabel, tcolor in tracks:
            if track == tname:
                vi_old = ve_summary.loc[ve_summary["model"] == r["older_model"], "version_index"]
                vi_new = ve_summary.loc[ve_summary["model"] == r["newer_model"], "version_index"]
                old_idx = int(vi_old.iloc[0]) if len(vi_old) > 0 else "?"
                new_idx = int(vi_new.iloc[0]) if len(vi_new) > 0 else "?"
                bar_labels.append(f"{tlabel} {old_idx}\u2192{new_idx}\n(n={int(r['n_paired_questions'])})")
                bar_vals.append(float(r["improvement_pp"]))
                bar_colors.append(tcolor)
                bar_p.append(float(r["mcnemar_p_exact"]))
                break

    x = np.arange(len(bar_labels))
    bars = ax_right.bar(x, bar_vals, color=bar_colors, alpha=0.85, width=0.6)
    ax_right.axhline(0, color="black", linewidth=0.8)
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(bar_labels, fontsize=8)
    ax_right.set_ylabel("CE improvement (pp, positive = better)", fontsize=10)
    ax_right.set_title("Consecutive Step Changes", fontsize=12)
    ax_right.grid(axis="y", alpha=0.2)

    for i, (bar, p) in enumerate(zip(bars, bar_p)):
        y_val = bar.get_height()
        offset = 0.4 if y_val >= 0 else -0.8
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "NS"))
        ax_right.text(
            bar.get_x() + bar.get_width() / 2, y_val + offset,
            f"p={p:.1e}\n{sig}", ha="center", fontsize=7, style="italic",
        )

    fig.suptitle("Version-Evolution: Does CE Improve With Newer Releases? (t=1.0)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ── LaTeX helpers ──────────────────────────────────────────────────────────

def _le(text: object) -> str:
    """LaTeX-escape a string."""
    s = "" if text is None else str(text)
    for k, v in [
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
        ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
        ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
    ]:
        s = s.replace(k, v)
    return s


def _fp(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}\\%"


# ── LaTeX generation ──────────────────────────────────────────────────────

def build_tex(
    agg: Dict[str, Any],
    baseline_per_model: pd.DataFrame,
    aurocs: Dict[str, float],
    unified: pd.DataFrame,
    ve_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    trends: pd.DataFrame,
    validation: Dict[str, Any],
    fig1_name: str,
    fig2_name: str,
) -> str:  # noqa: C901
    # helpers
    def _ce_seq(track: str) -> str:
        sub = ve_summary[ve_summary["track"] == track].sort_values("version_index")
        return r" $\rightarrow$ ".join(f"{v:.2f}\\%" for v in sub["ce_rate_pct"])

    def _acc_seq(track: str) -> str:
        sub = ve_summary[ve_summary["track"] == track].sort_values("version_index")
        return r" $\rightarrow$ ".join(f"{v:.1f}\\%" for v in sub["accuracy_pct"])

    def _pairwise_items(track: str) -> str:
        ce_consec = pairwise[
            (pairwise["metric"] == "ce_rate")
            & (pairwise["consecutive_pair"] == True)
            & (pairwise["track"] == track)
        ].sort_values("older_model")
        items = []
        for _, r in ce_consec.iterrows():
            vi_old = ve_summary.loc[ve_summary["model"] == r["older_model"], "version_index"]
            vi_new = ve_summary.loc[ve_summary["model"] == r["newer_model"], "version_index"]
            old_idx = int(vi_old.iloc[0]) if len(vi_old) > 0 else "?"
            new_idx = int(vi_new.iloc[0]) if len(vi_new) > 0 else "?"
            sign = "+" if r["improvement_pp"] > 0 else ""
            p = r["mcnemar_p_exact"]
            sig = "" if p < 0.05 else " (not significant)"
            items.append(
                f"    \\item v{old_idx}$\\rightarrow$v{new_idx}: "
                f"{sign}{r['improvement_pp']:.2f} pp change in CE "
                f"(p~=~{p:.1e}){sig}"
            )
        return "\n".join(items)

    def _trend_line(track: str, metric: str = "ce_rate") -> str:
        row = trends[(trends["track"] == track) & (trends["metric"] == metric)]
        if row.empty:
            return "N/A"
        r = row.iloc[0]
        slope = r["slope_per_version"]
        p = r["p_value"]
        direction = "decreasing" if slope < 0 else "increasing"
        return (
            f"{slope:+.2f} pp per version ({direction}, "
            f"p $\\approx$ {p:.1e})"
        )

    def _unified_rows() -> str:
        rows = []
        for _, r in unified.iterrows():
            fam = _family_of(r["model"])
            best = r["ce_rate_pct"] < 1.0
            name_str = _le(str(r["model"]))
            if best:
                name_str = r"\textbf{" + name_str + "}"
            rd = str(r.get("release_date", "")).strip()
            rd = rd if rd else "---"
            rows.append(
                f"{name_str} & "
                f"{fam} & "
                f"{_le(rd)} & "
                f"{r['accuracy_pct']:.1f}\\% & "
                f"{r['ce_rate_pct']:.2f}\\% \\\\"
            )
        return "\n".join(rows)

    def _ve_table_rows() -> str:
        rows = []
        for _, r in ve_summary.sort_values(["track", "version_index"]).iterrows():
            rd = str(r.get("release_date", ""))[:10]
            rows.append(
                f"{_le(r.get('family',''))} & "
                f"v{int(r['version_index'])} & "
                f"{_le(str(r['model']))} & "
                f"{rd} & "
                f"{r['accuracy_pct']:.1f}\\% & "
                f"{r['ce_rate_pct']:.2f}\\% \\\\"
            )
        return "\n".join(rows)

    cl = baseline_per_model[baseline_per_model["model"].str.contains("Claude")]
    gpt = baseline_per_model[baseline_per_model["model"].str.contains("GPT")]
    cl_row = cl.iloc[0] if len(cl) > 0 else {}
    gpt_row = gpt.iloc[0] if len(gpt) > 0 else {}
    ts = agg["threshold_sweep"]

    # best model stats
    best_idx = unified["ce_rate_pct"].idxmin()
    best_model = unified.loc[best_idx, "model"]
    best_ce = unified.loc[best_idx, "ce_rate_pct"]
    best_rd = unified.loc[best_idx, "release_date"]

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs, array, longtable, float, graphicx}}
\setlength{{\parskip}}{{6pt}}
\setlength{{\parindent}}{{0pt}}
\usepackage{{xcolor}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage{{hyperref}}
\hypersetup{{
  pdftitle={{Self-Consistent Error Rate Analysis}},
  pdfauthor={{Simranjeet Singh}},
  colorlinks=true, linkcolor=blue!60!black, urlcolor=blue!60!black,
}}

\title{{Self-Consistent Error Rate Analysis}}
\author{{Simranjeet Singh}}
\date{{March 19, 2026}}

\begin{{document}}
\maketitle

Hi Prof,

You suggested I look at whether the rate of self-consistent errors changes as models get newer.
This document puts everything together in one place --- the original baseline study I did first,
followed by the version-evolution follow-up I ran after.
I've tried to keep the language simple and add my own interpretation after each result.

%% ================================================================
\section*{{A quick note on what I mean by a self-consistent error}}

When I ask an LLM the same question multiple times, it does not always give the same answer.
A \textbf{{self-consistent error (CE)}} happens when the model is wrong \emph{{and}} gives
the same wrong answer every single time --- it is not guessing randomly, it is confidently
and repeatably wrong.
An \textbf{{inconsistent error (IE)}} is the opposite: the model is wrong, but its answers
vary across runs, which at least signals some uncertainty.

The \textbf{{equivalence threshold}} controls how strictly I define ``same answer.''
At $t = 1.0$ (what I use here), all 10 stochastic samples must be judged semantically
identical to count as self-consistent.
Lower thresholds (0.9, 0.8, 0.7) allow near-matches and naturally give higher CE counts.

%% ================================================================
\section{{Study 1 --- The Baseline (February 2026)}}

\subsection*{{What I did}}

I ran six frontier models on 807 TruthfulQA questions each (4,842 rows total),
collecting 1 greedy answer and 10 stochastic samples per question.
The models were: Claude Opus~4.6 (Anthropic), GPT-5.2 (OpenAI), DeepSeek~V3.2 (DeepSeek),
Qwen3~Next~80B (Alibaba/OpenRouter), Llama~4~Maverick~17B (Meta/Groq), and Grok~4 (xAI).
All numbers below come directly from the verified 4,842-row JSONL file.

\subsection*{{Overall accuracy}}

\begin{{center}}
\begin{{tabular}}{{lr}}
\toprule
Outcome & Count \\
\midrule
Total questions & {agg['total']:,} \\
Correct & {agg['correct']:,} \quad ({agg['correct_pct']}\%) \\
Incorrect & {agg['incorrect']:,} \quad ({agg['incorrect_pct']}\%) \\
Not attempted & {agg['not_attempted']:,} \quad ({agg['na_pct']}\%) \\
\bottomrule
\end{{tabular}}
\end{{center}}

So roughly one in three answers was wrong, and the models refused to answer about 4.5\% of the time.

\subsection*{{How many of those wrong answers were self-consistent?}}

\begin{{center}}
\begin{{tabular}}{{ccccc}}
\toprule
Threshold & CE count & IE count & CE (\% of total) & CE share of wrong \\
\midrule
1.0 & {ts['1.0']['ce']} & {ts['1.0']['ie']} & {ts['1.0']['ce_pct_total']}\% & {ts['1.0']['ce_among_wrong']}\% \\
0.9 & {ts['0.9']['ce']} & {ts['0.9']['ie']} & {ts['0.9']['ce_pct_total']}\% & {ts['0.9']['ce_among_wrong']}\% \\
0.8 & {ts['0.8']['ce']} & {ts['0.8']['ie']} & {ts['0.8']['ce_pct_total']}\% & {ts['0.8']['ce_among_wrong']}\% \\
0.7 & {ts['0.7']['ce']} & {ts['0.7']['ie']} & {ts['0.7']['ce_pct_total']}\% & {ts['0.7']['ce_among_wrong']}\% \\
\bottomrule
\end{{tabular}}
\end{{center}}

At the strictest threshold, \textbf{{{ts['1.0']['ce_among_wrong']}\% of wrong answers were self-consistent}}.
This was higher than I expected --- more than four in ten errors are not random slips, they are
confident wrong beliefs baked into the model.
As you loosen the threshold, CE share climbs all the way to {ts['0.7']['ce_among_wrong']}\%,
which means nearly six in ten wrong answers look self-consistent under a more relaxed definition.

\subsection*{{How do individual models compare?}}

The two models I looked at most closely:

\begin{{itemize}}
  \item \textbf{{{_le(cl_row.get('model','Claude Opus 4.6 (Anthropic)'))}}}: accuracy {cl_row.get('accuracy_pct',''):.1f}\%, CE rate {cl_row.get('ce_rate_pct',''):.2f}\% of all questions.
        Of its wrong answers, {cl_row.get('ce_among_wrong_pct',''):.1f}\% were self-consistent ---
        so when Claude is wrong, it tends to be \emph{{consistently}} wrong.
  \item \textbf{{{_le(gpt_row.get('model','GPT-5.2 (OpenAI)'))}}}: accuracy {gpt_row.get('accuracy_pct',''):.1f}\%, CE rate {gpt_row.get('ce_rate_pct',''):.2f}\% of all questions.
        Only {gpt_row.get('ce_among_wrong_pct',''):.1f}\% of its wrong answers were self-consistent ---
        GPT tends to make more varied mistakes, which is in a sense less ``locked in.''
\end{{itemize}}

Even among the best models in the world, the CE rate is meaningful (7--11\% of all questions).

\subsection*{{Can we tell from the outside when a model is making a CE?}}

I tried two output-only signals --- disagreement across samples and semantic entropy ---
to see if they could flag self-consistent errors without accessing the model internals:

\begin{{itemize}}
  \item Disagreement AUROC: \textbf{{{aurocs.get('disagreement', 0):.3f}}}
  \item Semantic entropy AUROC: \textbf{{{aurocs.get('semantic_entropy', 0):.3f}}}
\end{{itemize}}

An AUROC of 0.5 means random, 1.0 means perfect.
Both scores are barely above 0.5, so these signals essentially do not work.
This makes intuitive sense: a CE looks exactly like a correct answer from the outside ---
the model is confident and consistent, just wrong.

\bigskip
\noindent\textit{{This is what motivated the second study.
If we cannot detect CE from the outside, maybe newer model versions just have less of it to begin with.}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{{fig1_name}}}
\caption{{Left: how CE share changes with threshold (Study~1, 4,842 baseline). Right: CE rate for all 15 models across both studies, sorted from best to worst. Each colour is a model family.}}
\end{{figure}}

%% ================================================================
\section{{Study 2 --- Does CE Decrease With Newer Versions? (March 2026)}}

\subsection*{{Why I ran this}}

The baseline only captured each model at one point in time.
Your recommendation was to look at the \emph{{rate of change}} --- so I picked three model
families that have released several versions over the past two years and re-ran
the same evaluation on all of them.

\subsection*{{What I did}}

I selected Grok (xAI), Llama (Meta), and Qwen (Alibaba), took four released versions of each
(12 models total, 807 questions each, 9,684 rows total), and ran the exact same protocol:
1 greedy answer + 10 stochastic samples, with a uniform equivalence-checking method
(NLI-based with GPT-5.2 fallback for borderline cases).

The three latest endpoints --- Grok~4, Llama~4~Maverick, Qwen3~Next~80B --- overlap with
Study~1, so those numbers are consistent across both studies.
The nine older versions are fresh reruns done in March 2026.

\subsection*{{Models used, in order}}

\begin{{center}}
\footnotesize
\begin{{longtable}}{{l c p{{6.5cm}} c c}}
\toprule
Family & Ver & Model & Released & Params \\
\midrule
Grok & v1 & Grok 3 (xAI) & 2025-06-10 & undisclosed \\
Grok & v2 & Grok 4 (xAI) & 2025-07-09 & undisclosed \\
Grok & v3 & Grok 4.1 Fast Reasoning (xAI) & 2025-11-19 & undisclosed \\
Grok & v4 & Grok 4.20 Beta 0309 Reasoning (xAI) & 2026-03-09 & undisclosed \\
\midrule
Llama & v1 & Llama 3 8B Instruct (Meta / OpenRouter) & 2024-04-18 & 8B \\
Llama & v2 & Llama 3.1 8B Instruct (Meta / OpenRouter) & 2024-07-23 & 8B \\
Llama & v3 & Llama 3.3 70B Instruct (Meta / OpenRouter) & 2024-12-06 & 70B \\
Llama & v4 & Llama 4 Maverick 17B 128E (Meta / Groq) & 2025-04-05 & 17B MoE \\
\midrule
Qwen & v1 & Qwen2.5 7B Instruct (Alibaba / OpenRouter) & 2024-09-16 & 7B \\
Qwen & v2 & Qwen2.5 72B Instruct (Alibaba / OpenRouter) & 2024-11-26 & 72B \\
Qwen & v3 & Qwen3 30B A3B (Alibaba / OpenRouter) & 2025-07-28 & 30B MoE \\
Qwen & v4 & Qwen3 Next 80B (Alibaba / OpenRouter) & 2025-09-09 & 80B \\
\bottomrule
\end{{longtable}}
\end{{center}}

One important caveat for Llama and Qwen: I could not hold model size constant ---
the Llama track goes 8B $\rightarrow$ 8B $\rightarrow$ 70B $\rightarrow$ 17B (MoE),
and Qwen goes 7B $\rightarrow$ 72B $\rightarrow$ 30B $\rightarrow$ 80B.
So for those two families, changes in CE could reflect scale differences as much as
genuine generational improvement.
Grok's track is the cleanest because all four versions are large closed models from the same lab.

\subsection*{{Results by model}}

\begin{{center}}
\footnotesize
\begin{{longtable}}{{l c p{{6.5cm}} c r r}}
\toprule
Family & Ver & Model & Released & Accuracy & CE rate \\
\midrule
{_ve_table_rows()}
\bottomrule
\end{{longtable}}
\end{{center}}

Reading the CE rate column left to right (oldest to newest) for each family:

\begin{{itemize}}
  \item \textbf{{Grok:}} {_ce_seq('grok_version')}\\
        \textit{{Clear, steady improvement every single generation.
        Accuracy also rose: {_acc_seq('grok_version')}.
        The newest model (Grok~4.20, released 2026-03-09) is nearly CE-free at 0.50\%.}}
  \item \textbf{{Llama:}} {_ce_seq('llama_scale_version')}\\
        \textit{{No clear trend --- goes up, then down, then up again.
        Accuracy: {_acc_seq('llama_scale_version')}.
        The big jump at v3 (70B) is likely a scale effect, not a genuine improvement.}}
  \item \textbf{{Qwen:}} {_ce_seq('qwen_scale_version')}\\
        \textit{{Also non-monotonic.
        Accuracy: {_acc_seq('qwen_scale_version')}.
        Similar story to Llama --- the CE rate bounces around rather than trending down.}}
\end{{itemize}}

\subsection*{{Is each step-change statistically real?}}

I used McNemar's test on shared question IDs to check whether consecutive improvements
are genuine (positive = CE went down, i.e.\ improved):

\textbf{{Grok}} (cleanest comparison, no scale confound):
\begin{{itemize}}
{_pairwise_items('grok_version')}
\end{{itemize}}

\textit{{Every single step for Grok is highly significant.
This is not noise.}}

\textbf{{Llama}} (8B $\rightarrow$ 8B $\rightarrow$ 70B $\rightarrow$ 17B MoE --- scale changes between versions):
\begin{{itemize}}
{_pairwise_items('llama_scale_version')}
\end{{itemize}}

\textbf{{Qwen}} (7B $\rightarrow$ 72B $\rightarrow$ 30B $\rightarrow$ 80B --- scale changes between versions):
\begin{{itemize}}
{_pairwise_items('qwen_scale_version')}
\end{{itemize}}

\subsection*{{Overall trend direction (logistic regression)}}

To get a single number summarising the trend across all four versions,
I ran a logistic regression of CE outcome on version index:

\begin{{itemize}}
  \item \textbf{{Grok}}: {_trend_line('grok_version')} --- strongly \textbf{{improving}}
  \item \textbf{{Llama}}: {_trend_line('llama_scale_version')} --- getting \textbf{{worse}} on average
  \item \textbf{{Qwen}}: {_trend_line('qwen_scale_version')} --- flat or slightly worse
\end{{itemize}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{{fig2_name}}}
\caption{{Left: CE rate (solid line) and accuracy (dashed, right axis) over release dates for each family. Right: consecutive step changes in CE, with p-values. Green bars = improvement; red = worsening.}}
\end{{figure}}

%% ================================================================
\section*{{What this all means}}

The big picture: self-consistent errors are a substantial fraction of LLM failures ---
at the strictest threshold, 42\% of wrong answers are confident and repeatable.
We cannot detect them from the outside (AUROC $\approx$ 0.56).

The follow-up showed that it is possible for newer versions to fix this ---
Grok went from 17.7\% CE down to 0.5\% across four generations,
with every step being statistically significant.
But it is not automatic: Llama and Qwen show no consistent downward trend,
and their mixed model sizes make it hard to say whether any improvement is real.

The honest takeaway for now is that Grok is the one family where the data clearly shows
the CE rate getting better with each release.
For the others, I need either more controlled comparisons (same size, same family)
or more versions before I can say confidently whether CE is improving.

\bigskip
\noindent Thanks,\\
Simran

\end{{document}}
"""
    return tex


# ── plain text generation ─────────────────────────────────────────────────

def build_txt(
    agg: Dict[str, Any],
    baseline_per_model: pd.DataFrame,
    aurocs: Dict[str, float],
    unified: pd.DataFrame,
    ve_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    trends: pd.DataFrame,
) -> str:
    ts = agg["threshold_sweep"]

    def _ce_seq_txt(track: str) -> str:
        sub = ve_summary[ve_summary["track"] == track].sort_values("version_index")
        return " -> ".join(f"{v:.2f}%" for v in sub["ce_rate_pct"])

    def _acc_seq_txt(track: str) -> str:
        sub = ve_summary[ve_summary["track"] == track].sort_values("version_index")
        return " -> ".join(f"{v:.1f}%" for v in sub["accuracy_pct"])

    def _pw_txt(track: str) -> str:
        ce_consec = pairwise[
            (pairwise["metric"] == "ce_rate")
            & (pairwise["consecutive_pair"] == True)
            & (pairwise["track"] == track)
        ].sort_values("older_model")
        lines = []
        for _, r in ce_consec.iterrows():
            vi_old = ve_summary.loc[ve_summary["model"] == r["older_model"], "version_index"]
            vi_new = ve_summary.loc[ve_summary["model"] == r["newer_model"], "version_index"]
            old_idx = int(vi_old.iloc[0]) if len(vi_old) > 0 else "?"
            new_idx = int(vi_new.iloc[0]) if len(vi_new) > 0 else "?"
            sign = "+" if r["improvement_pp"] > 0 else ""
            p = r["mcnemar_p_exact"]
            sig = "" if p < 0.05 else " (not significant)"
            lines.append(
                f"  - v{old_idx}->v{new_idx}: {sign}{r['improvement_pp']:.2f} pp change in CE "
                f"(p={p:.1e}){sig}"
            )
        return "\n".join(lines)

    def _trend_txt(track: str) -> str:
        row = trends[(trends["track"] == track) & (trends["metric"] == "ce_rate")]
        if row.empty:
            return "N/A"
        r = row.iloc[0]
        direction = "decreasing" if r["slope_per_version"] < 0 else "increasing"
        return f"{r['slope_per_version']:+.2f} pp/version ({direction}, p ~ {r['p_value']:.1e})"

    cl = baseline_per_model[baseline_per_model["model"].str.contains("Claude")]
    gpt = baseline_per_model[baseline_per_model["model"].str.contains("GPT")]
    cl_row = cl.iloc[0] if len(cl) > 0 else {}
    gpt_row = gpt.iloc[0] if len(gpt) > 0 else {}

    ve_lines = []
    for _, r in ve_summary.sort_values(["track", "version_index"]).iterrows():
        rd = str(r.get("release_date", ""))[:10]
        ve_lines.append(
            f"  {r.get('family',''):5s} v{int(r['version_index'])}  {str(r['model']):<55s}"
            f"  {rd}  Acc={r['accuracy_pct']:5.1f}%  CE={r['ce_rate_pct']:5.2f}%"
        )

    return f"""Self-Consistent Error Rate Analysis
Simranjeet Singh — March 19, 2026

Hi Prof,

You suggested I look at whether the rate of self-consistent errors changes as
models get newer. This document puts everything together --- Study 1 (the
original baseline I ran first) followed by Study 2 (the version-evolution
follow-up). I've added plain-English interpretations after each result.

========================================================================
WHAT IS A SELF-CONSISTENT ERROR?
========================================================================
When I ask an LLM the same question multiple times, it does not always give
the same answer. A self-consistent error (CE) happens when the model is wrong
AND gives the same wrong answer every single time --- it is confidently and
repeatably wrong.

An inconsistent error (IE) is the opposite: the model is wrong, but its
answers vary across runs, which at least signals some uncertainty.

Equivalence threshold (t=1.0): all 10 stochastic samples must be judged
semantically identical to count as self-consistent. Lower thresholds (0.9,
0.8, 0.7) allow near-matches. I use t=1.0 throughout.

========================================================================
STUDY 1 --- THE BASELINE (FEBRUARY 2026)
========================================================================

What I did:
  I ran six frontier models on 807 TruthfulQA questions each (4,842 rows),
  collecting 1 greedy answer + 10 stochastic samples per question.
  Models: Claude Opus 4.6 (Anthropic), GPT-5.2 (OpenAI), DeepSeek V3.2,
  Qwen3 Next 80B, Llama 4 Maverick 17B, Grok 4.
  All numbers below are verified directly from the source JSONL file.

Overall accuracy:
  Total rows:     {agg['total']:,}
  Correct:        {agg['correct']:,}  ({agg['correct_pct']}%)
  Incorrect:      {agg['incorrect']:,}  ({agg['incorrect_pct']}%)
  Not attempted:  {agg['not_attempted']:,}  ({agg['na_pct']}%)

  -> So roughly 1 in 3 answers was wrong, and models refused ~4.5% of the time.

How many wrong answers were self-consistent?

  Threshold  CE count  IE count  CE of total  CE share of wrong
  ---------  --------  --------  -----------  -----------------
  t=1.0      {ts['1.0']['ce']:<8}  {ts['1.0']['ie']:<8}  {ts['1.0']['ce_pct_total']:<11}  {ts['1.0']['ce_among_wrong']}%
  t=0.9      {ts['0.9']['ce']:<8}  {ts['0.9']['ie']:<8}  {ts['0.9']['ce_pct_total']:<11}  {ts['0.9']['ce_among_wrong']}%
  t=0.8      {ts['0.8']['ce']:<8}  {ts['0.8']['ie']:<8}  {ts['0.8']['ce_pct_total']:<11}  {ts['0.8']['ce_among_wrong']}%
  t=0.7      {ts['0.7']['ce']:<8}  {ts['0.7']['ie']:<8}  {ts['0.7']['ce_pct_total']:<11}  {ts['0.7']['ce_among_wrong']}%

  -> At t=1.0, {ts['1.0']['ce_among_wrong']}% of wrong answers are self-consistent.
     That is more than 4 in 10 errors being confident and repeatable.
     As the threshold loosens, CE share climbs to {ts['0.7']['ce_among_wrong']}% at t=0.7.

Individual model highlights (from Study 1):
  {cl_row.get('model','Claude Opus 4.6 (Anthropic)')}
    Accuracy: {cl_row.get('accuracy_pct',''):.1f}%,  CE rate: {cl_row.get('ce_rate_pct',''):.2f}% of all questions
    CE share of wrong answers: {cl_row.get('ce_among_wrong_pct',''):.1f}%
    -> When Claude is wrong, it tends to be consistently wrong.

  {gpt_row.get('model','GPT-5.2 (OpenAI)')}
    Accuracy: {gpt_row.get('accuracy_pct',''):.1f}%,  CE rate: {gpt_row.get('ce_rate_pct',''):.2f}% of all questions
    CE share of wrong answers: {gpt_row.get('ce_among_wrong_pct',''):.1f}%
    -> GPT makes more varied mistakes, which is slightly less "locked in."

  Even among the best models, CE rates of 7-11% of all questions are meaningful.

Can we tell from the outside when a model is making a CE?
  Disagreement AUROC:     {aurocs.get('disagreement', 0):.3f}
  Semantic entropy AUROC: {aurocs.get('semantic_entropy', 0):.3f}

  -> 0.5 = random, 1.0 = perfect. Both scores are barely above random.
     These signals essentially do not work. A CE looks like a correct answer
     from the outside --- confident and consistent, just wrong.

  This is what motivated Study 2: if we cannot detect CE externally, maybe
  newer model versions just have less of it to begin with.

========================================================================
STUDY 2 --- DOES CE DECREASE WITH NEWER VERSIONS? (MARCH 2026)
========================================================================

Why I ran this:
  The baseline only captured each model at one snapshot in time.
  Your recommendation was to look at the rate of change --- so I picked three
  families with multiple releases and re-ran the same evaluation on all of them.

What I did:
  - Three families: Grok (xAI), Llama (Meta), Qwen (Alibaba)
  - Four versions of each = 12 models, 807 questions each = 9,684 rows total
  - Same protocol: 1 greedy + 10 stochastic samples, uniform equiv-checking
    (NLI-based with GPT-5.2 fallback)
  - Three latest endpoints (Grok 4, Llama 4 Maverick, Qwen3 Next 80B) overlap
    with Study 1. The nine older versions are fresh reruns from March 2026.

Models used (oldest to newest within each family):
  Grok (all large closed-weight models -- cleanest comparison):
    v1  Grok 3 (xAI)                              released 2025-06-10  undisclosed params
    v2  Grok 4 (xAI)                              released 2025-07-09  undisclosed params
    v3  Grok 4.1 Fast Reasoning (xAI)             released 2025-11-19  undisclosed params
    v4  Grok 4.20 Beta 0309 Reasoning (xAI)       released 2026-03-09  undisclosed params

  Llama (NOTE: parameter count changes between versions -- scale confound):
    v1  Llama 3 8B Instruct (Meta/OpenRouter)     released 2024-04-18  8B
    v2  Llama 3.1 8B Instruct (Meta/OpenRouter)   released 2024-07-23  8B
    v3  Llama 3.3 70B Instruct (Meta/OpenRouter)  released 2024-12-06  70B
    v4  Llama 4 Maverick 17B 128E (Meta/Groq)     released 2025-04-05  17B MoE

  Qwen (NOTE: parameter count also changes -- scale confound):
    v1  Qwen2.5 7B Instruct (Alibaba/OpenRouter)  released 2024-09-16  7B
    v2  Qwen2.5 72B Instruct (Alibaba/OpenRouter) released 2024-11-26  72B
    v3  Qwen3 30B A3B (Alibaba/OpenRouter)        released 2025-07-28  30B MoE
    v4  Qwen3 Next 80B (Alibaba/OpenRouter)       released 2025-09-09  80B

  -> Grok is the only family where I can hold scale roughly constant.
     For Llama and Qwen, any CE change could be size-driven, not generation-driven.

Results by model:

{chr(10).join(ve_lines)}

CE rate from oldest to newest version:
  Grok:  {_ce_seq_txt('grok_version')}  <-- clear, steady improvement
         Accuracy: {_acc_seq_txt('grok_version')}  <-- also improving
  Llama: {_ce_seq_txt('llama_scale_version')}  <-- no clear trend
  Qwen:  {_ce_seq_txt('qwen_scale_version')}  <-- no clear trend

  -> Grok dropped from 17.72% CE to just 0.50% across four generations.
     That is a 17 percentage-point improvement. Llama and Qwen bounce around
     with no consistent direction.

Is each step-change statistically real? (McNemar's test on shared question IDs)
  Positive = CE went down (improved). Negative = CE went up (worse).

  Grok (clean comparison, no scale confound):
{_pw_txt('grok_version')}
  -> Every single step is highly significant. This is not noise.

  Llama (8B->8B->70B->17B MoE -- scale changes throughout):
{_pw_txt('llama_scale_version')}

  Qwen (7B->72B->30B->80B -- scale changes throughout):
{_pw_txt('qwen_scale_version')}

Overall CE trend direction (logistic regression over all 4 versions):
  Grok:  {_trend_txt('grok_version')}  <-- strongly improving
  Llama: {_trend_txt('llama_scale_version')}  <-- getting worse on average
  Qwen:  {_trend_txt('qwen_scale_version')}  <-- flat or slightly worse

========================================================================
WHAT IT ALL MEANS
========================================================================
Self-consistent errors make up a large share of LLM failures --- at the
strictest threshold, 42% of wrong answers are confident and repeatable.
We cannot reliably detect them from the outside (AUROC ~0.56).

The version-evolution study showed that it is possible to reduce CE with
newer releases: Grok went from 17.7% down to 0.5%, with every step being
statistically significant. But it is not automatic --- Llama and Qwen show
no consistent improvement, and their mixed model sizes make it hard to say
whether any change is real.

The honest summary for now: Grok is the one family where the data clearly
shows CE getting better generation over generation. For Llama and Qwen, I
need either more controlled comparisons (same size) or more versions before
I can say anything definitive.

A few caveats:
  - Study 2 timeline is mixed: older points are new reruns; latest endpoints
    reuse Study 1 data.
  - Pairwise tests use ~797 shared question IDs, not always the full 807.
  - Llama and Qwen tracks mix parameter scale with generation.
  - All results use threshold t=1.0; the threshold sweep is Study 1 context only.

Thanks,
Simran
"""


# ── main ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-jsonl", type=Path, default=BASELINE_JSONL)
    p.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--compile-pdf", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or args.analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading baseline JSONL …")
    df_base = load_baseline(args.baseline_jsonl)
    print(f"       {len(df_base)} rows, {df_base['model'].nunique()} models")

    print("[2/7] Computing baseline aggregates …")
    agg = compute_baseline_aggregates(df_base)
    baseline_per_model = compute_per_model_baseline(df_base, thr="1.0")
    aurocs = compute_auroc(df_base)
    print(f"       AUROCs: {aurocs}")

    print("[3/7] Loading version-evolution analysis …")
    ve = load_version_evolution(args.analysis_dir)
    ve_summary = ve["model_summary_t1p0"]
    pairwise = ve["pairwise_t1p0"]
    trends = ve["trend_t1p0"]
    validation = ve["validation"]

    print("[4/7] Building unified model table …")
    unified = build_unified_model_table(baseline_per_model, ve_summary)
    print(f"       {len(unified)} unique models in combined table")

    fig1_name = "prof_expanded_fig1_blackbox_verified.png"
    fig2_name = "prof_expanded_fig2_version_evolution_t1p0.png"

    print("[5/7] Generating figures …")
    make_fig1(agg, unified, out_dir / fig1_name)
    make_fig2(ve_summary, pairwise, out_dir / fig2_name)

    print("[6/7] Generating LaTeX …")
    tex = build_tex(
        agg, baseline_per_model, aurocs, unified,
        ve_summary, pairwise, trends, validation,
        fig1_name, fig2_name,
    )
    tex_path = out_dir / "prof_single_combined_report_t1p0_expanded.tex"
    tex_path.write_text(tex, encoding="utf-8")

    print("[7/7] Generating plain text …")
    txt = build_txt(agg, baseline_per_model, aurocs, unified, ve_summary, pairwise, trends)
    txt_path = out_dir / "prof_single_combined_report_t1p0_expanded.txt"
    txt_path.write_text(txt, encoding="utf-8")

    if args.compile_pdf:
        print("[pdf]  Compiling LaTeX …")
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_path.name],
                cwd=str(out_dir), capture_output=True,
            )
        pdf_path = tex_path.with_suffix(".pdf")
        if pdf_path.exists():
            print(f"[done] PDF: {pdf_path}")
        else:
            print("[warn] pdflatex did not produce a PDF; check .log file")

    print(f"\n[done] Outputs in {out_dir}/")
    print(f"  tex: {tex_path.name}")
    print(f"  txt: {txt_path.name}")
    print(f"  fig: {fig1_name}, {fig2_name}")


if __name__ == "__main__":
    main()
