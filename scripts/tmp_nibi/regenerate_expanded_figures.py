#!/usr/bin/env python3
"""Regenerate the two figures for the expanded combined professor report (t=1.0)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("data/results/analysis/version_evolution_equiv_only_20260319")

BASELINE_JSONL = Path(
    "data/results/evaluated/"
    "results_v2_phase2_eval_no_gemini_4842.final.analysis_ready."
    "skip_greedy_semantic_eval.jsonl"
)

FAMILY_COLORS = {
    "Grok": "#1d3557",
    "Llama": "#386641",
    "Qwen": "#9c6644",
    "GPT": "#c1121f",
    "Claude": "#7b2d8e",
    "DeepSeek": "#457b9d",
}


def _family_of(model: str) -> str:
    m = model.lower()
    if "grok" in m:
        return "Grok"
    if "llama" in m or "maverick" in m:
        return "Llama"
    if "qwen" in m:
        return "Qwen"
    if "gpt" in m or "chatgpt" in m:
        return "GPT"
    if "claude" in m:
        return "Claude"
    if "deepseek" in m:
        return "DeepSeek"
    return "Other"


def fig1_blackbox(baseline_df: pd.DataFrame, summary_t1: pd.DataFrame) -> Path:
    """Left: threshold sensitivity (Study 1).  Right: all-model CE bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.4]})

    # --- LEFT: threshold sweep from baseline ---
    thresholds = [1.0, 0.9, 0.8, 0.7]
    ce_share_wrong = [42.1, 48.9, 53.9, 58.9]
    ce_total = [13.5, 15.8, 17.4, 19.0]
    labels = [f"t={t}" for t in thresholds]

    ax1.plot(labels, ce_share_wrong, "o-", color="#c1121f", linewidth=2, markersize=7, label="CE among wrong (%)")
    ax1.plot(labels, ce_total, "s-", color="#1d3557", linewidth=2, markersize=7, label="CE of total rows (%)")
    for i, (cw, ct) in enumerate(zip(ce_share_wrong, ce_total)):
        ax1.annotate(f"{cw}", (labels[i], cw), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
        ax1.annotate(f"{ct}", (labels[i], ct), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9)
    ax1.set_ylabel("Percent", fontsize=11)
    ax1.set_title("Threshold Sensitivity (4842 baseline)", fontsize=11, fontweight="bold")
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    ax1.set_ylim(0, 68)
    ax1.grid(alpha=0.2)

    # --- RIGHT: all-model CE rate bar chart (t=1.0) ---
    model_ce = summary_t1[["model", "ce_rate_pct"]].copy()

    # also add Study 1 models not in summary_t1 (Claude, GPT, DeepSeek)
    if BASELINE_JSONL.exists():
        bdf = baseline_df.copy()
        bdf["is_ce"] = bdf["error_label_1.0"].astype(str).eq("self_consistent_error")
        b_agg = bdf.groupby("model").agg(ce_rate=("is_ce", "mean")).reset_index()
        b_agg["ce_rate_pct"] = 100.0 * b_agg["ce_rate"]
        missing = b_agg[~b_agg["model"].isin(model_ce["model"])]
        model_ce = pd.concat([model_ce, missing[["model", "ce_rate_pct"]]], ignore_index=True)

    model_ce = model_ce.sort_values("ce_rate_pct", ascending=True).reset_index(drop=True)
    model_ce["family"] = model_ce["model"].apply(_family_of)
    colors = [FAMILY_COLORS.get(f, "#888888") for f in model_ce["family"]]

    y_pos = np.arange(len(model_ce))
    ax2.barh(y_pos, model_ce["ce_rate_pct"], color=colors, edgecolor="white", linewidth=0.3, height=0.72)
    ax2.set_yticks(y_pos)

    short_names = []
    for m in model_ce["model"]:
        s = str(m)
        for remove in ["(OpenRouter)", "(Groq)", "(xAI)", "(Alibaba)", "(Meta)"]:
            s = s.replace(remove, "").strip()
        for remove in [", 2024-04-18", ", 2024-07-23", ", 2024-09-16", ", 2024-11-26",
                       ", 2024-12-06", ", 2025-06-10", ", 2025-07-28", ", 2025-11-19",
                       ", 2026-03-09"]:
            s = s.replace(remove, "")
        short_names.append(s.strip().rstrip(","))
    ax2.set_yticklabels(short_names, fontsize=8.5)

    for i, v in enumerate(model_ce["ce_rate_pct"]):
        ax2.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8)

    ax2.set_xlabel("CE rate (% of total rows, t=1.0)", fontsize=10)
    ax2.set_title("Model-Level CE Rate (all models combined)", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, model_ce["ce_rate_pct"].max() * 1.15)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=FAMILY_COLORS[f]) for f in ["Grok", "Llama", "GPT", "Claude", "Qwen", "DeepSeek"]]
    ax2.legend(handles, ["Grok", "Llama", "GPT", "Claude", "Qwen", "DeepSeek"],
               fontsize=8, frameon=False, loc="lower right", ncol=2)

    fig.suptitle("The CE Landscape: How Big Is the Problem and Who Is Affected?", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    p = OUT_DIR / "prof_expanded_fig1_blackbox_verified.png"
    fig.savefig(p, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return p


def fig2_version_evolution(summary_t1: pd.DataFrame, pairwise_t1: pd.DataFrame) -> Path:
    """Left: CE + accuracy timeline.  Right: consecutive step-change bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})

    track_info = {
        "grok_version": ("Grok CE", "Grok Acc", "#1d3557"),
        "llama_scale_version": ("Llama CE", "Llama Acc", "#386641"),
        "qwen_scale_version": ("Qwen CE", "Qwen Acc", "#9c6644"),
    }

    ax1r = ax1.twinx()
    for track, (ce_label, acc_label, color) in track_info.items():
        sub = summary_t1[summary_t1["track"] == track].sort_values("release_date").copy()
        sub["release_dt"] = pd.to_datetime(sub["release_date"])
        ax1.plot(sub["release_dt"], sub["ce_rate_pct"], "o-", color=color, linewidth=2, markersize=7, label=ce_label)
        ax1r.plot(sub["release_dt"], sub["accuracy_pct"], "s--", color=color, linewidth=1.2, markersize=5, alpha=0.5, label=acc_label)
        for _, r in sub.iterrows():
            ax1.annotate(f"{r['ce_rate_pct']:.1f}", (r["release_dt"], r["ce_rate_pct"]),
                         textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=color)

    ax1.set_ylabel("CE rate (% of total, t=1.0)", fontsize=10)
    ax1r.set_ylabel("Accuracy (%)", fontsize=10, alpha=0.6)
    ax1.set_xlabel("Release date", fontsize=10)
    ax1.set_title("CE and Accuracy Over Time", fontsize=11, fontweight="bold")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1r.legend(frameon=False, fontsize=7, loc="center right")
    ax1.grid(alpha=0.2)

    # --- RIGHT: consecutive step-change bars ---
    ce_consec = pairwise_t1[
        (pairwise_t1["metric"] == "ce_rate") & (pairwise_t1["consecutive_pair"])
    ].copy()
    ce_consec = ce_consec.sort_values(["track", "older_model"])

    track_label_map = {"grok_version": "Grok", "llama_scale_version": "Llama", "qwen_scale_version": "Qwen"}

    label_parts = []
    for _, r in ce_consec.iterrows():
        fam = track_label_map.get(r["track"], r["track"])
        n = int(r["n_paired_questions"])
        older_idx = r["older_model"]
        newer_idx = r["newer_model"]
        # extract version indices from the models in the track
        sub_sum = summary_t1[summary_t1["track"] == r["track"]].sort_values("version_index")
        models = sub_sum["model"].tolist()
        try:
            oi = models.index(older_idx) + 1
            ni = models.index(newer_idx) + 1
        except ValueError:
            oi, ni = "?", "?"
        label_parts.append(f"{fam} {oi}\u2192{ni}\n(n={n})")

    x = np.arange(len(ce_consec))
    bar_colors = []
    for _, r in ce_consec.iterrows():
        c = {"grok_version": "#1d3557", "llama_scale_version": "#386641", "qwen_scale_version": "#9c6644"}.get(r["track"], "#888")
        bar_colors.append(c)

    improvements = ce_consec["improvement_pp"].values
    ax2.bar(x, improvements, color=bar_colors, edgecolor="white", linewidth=0.3, width=0.65)
    ax2.axhline(0, color="black", linewidth=0.8)

    sig_labels = {True: "***", False: ""}
    for i, (_, r) in enumerate(ce_consec.iterrows()):
        p = r["mcnemar_p_exact"]
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
        else:
            stars = "NS"
        val = r["improvement_pp"]
        offset = 0.3 if val >= 0 else -0.3
        ax2.text(i, val + offset, f"p={p:.1e}\n{stars}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(label_parts, fontsize=8)
    ax2.set_ylabel("CE improvement (pp, positive = better)", fontsize=10)
    ax2.set_title("Consecutive Step Changes", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.2)

    fig.suptitle("Version-Evolution: Does CE Improve With Newer Releases? (t=1.0)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    p = OUT_DIR / "prof_expanded_fig2_version_evolution_t1p0.png"
    fig.savefig(p, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> None:
    summary_t1 = pd.read_csv(OUT_DIR / "model_summary_t1p0.csv")
    pairwise_t1 = pd.read_csv(OUT_DIR / "pairwise_deltas_t1p0.csv")

    baseline_df = pd.DataFrame()
    if BASELINE_JSONL.exists():
        baseline_df = pd.read_json(BASELINE_JSONL, lines=True)

    p1 = fig1_blackbox(baseline_df, summary_t1)
    print(f"[done] {p1}")

    p2 = fig2_version_evolution(summary_t1, pairwise_t1)
    print(f"[done] {p2}")


if __name__ == "__main__":
    main()
