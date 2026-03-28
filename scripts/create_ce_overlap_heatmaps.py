#!/usr/bin/env python3
"""
Create pairwise heatmaps for cross-model CE overlap analysis:
1) overlap count (both CE)
2) same-wrong percentage
3) combined panel
4) same-wrong count with percentage annotations
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_DATA_FILE = Path("data/results/analysis/v2_thesis/tables/cross_model_ce_overlap_semantic.csv")
DEFAULT_OUTPUT_DIR = Path("data/results/analysis/v2_thesis/figures")

# Short model names for display
MODEL_SHORT_NAMES = {
    "Claude Opus 4.6 (Anthropic)": "Claude",
    "DeepSeek V3.2 (DeepSeek)": "DeepSeek",
    "GPT-5.2 (OpenAI)": "GPT-5.2",
    "Grok 4 (xAI)": "Grok",
    "Llama 4 Maverick 17B (Groq)": "Llama",
    "Qwen3 Next 80B (OpenRouter)": "Qwen3",
}

MODEL_ORDER = ["Claude", "GPT-5.2", "Grok", "DeepSeek", "Llama", "Qwen3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CE overlap heatmaps from pairwise CSV.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix tag for output filenames, e.g. nlihybrid_t1p0.",
    )
    parser.add_argument(
        "--title-suffix",
        type=str,
        default="",
        help="Optional text appended to plot titles.",
    )
    return parser.parse_args()


def output_name(base: str, tag: str) -> str:
    if not tag:
        return base
    stem, ext = base.rsplit(".", 1)
    return f"{stem}_{tag}.{ext}"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"model_a", "model_b", "both_ce_overlap", "same_wrong_answer", "same_wrong_pct"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df["model_a_short"] = df["model_a"].map(MODEL_SHORT_NAMES)
    df["model_b_short"] = df["model_b"].map(MODEL_SHORT_NAMES)
    return df


def create_symmetric_matrix(df: pd.DataFrame, value_col: str, models: list[str]) -> np.ndarray:
    n = len(models)
    matrix = np.zeros((n, n))
    model_to_idx = {m: i for i, m in enumerate(models)}

    for _, row in df.iterrows():
        i = model_to_idx.get(row["model_a_short"])
        j = model_to_idx.get(row["model_b_short"])
        if i is None or j is None:
            continue
        matrix[i, j] = row[value_col]
        matrix[j, i] = row[value_col]

    np.fill_diagonal(matrix, np.nan)
    return matrix


def main() -> None:
    args = parse_args()
    print("Loading data...")
    df = load_data(args.data_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    title_suffix = f"\n{args.title_suffix}" if args.title_suffix else ""

    overlap_matrix = create_symmetric_matrix(df, "both_ce_overlap", MODEL_ORDER)
    same_wrong_matrix = create_symmetric_matrix(df, "same_wrong_pct", MODEL_ORDER)
    same_wrong_count_matrix = create_symmetric_matrix(df, "same_wrong_answer", MODEL_ORDER)
    mask = np.eye(len(MODEL_ORDER), dtype=bool)

    # Use data-driven scales for portability across thresholds.
    overlap_vals = df["both_ce_overlap"].to_numpy(dtype=float)
    same_wrong_vals = df["same_wrong_pct"].to_numpy(dtype=float)
    same_wrong_count_vals = df["same_wrong_answer"].to_numpy(dtype=float)

    ov_min, ov_max = float(np.nanmin(overlap_vals)), float(np.nanmax(overlap_vals))
    sw_min, sw_max = float(np.nanmin(same_wrong_vals)), float(np.nanmax(same_wrong_vals))
    swc_min, swc_max = float(np.nanmin(same_wrong_count_vals)), float(np.nanmax(same_wrong_count_vals))

    # Figure 1: overlap count.
    fig1, ax1 = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        overlap_matrix,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        xticklabels=MODEL_ORDER,
        yticklabels=MODEL_ORDER,
        mask=mask,
        cbar_kws={"label": "Questions with CE in both models"},
        ax=ax1,
        vmin=ov_min,
        vmax=ov_max,
    )
    ax1.set_title(
        "Cross-Model CE Overlap\n(Questions where both models are confidently wrong)" + title_suffix,
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    plt.tight_layout()
    out1 = args.output_dir / output_name("fig_ce_overlap_heatmap.png", args.tag)
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved: {out1}")

    # Figure 2: same wrong percentage.
    fig2, ax2 = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        same_wrong_matrix,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        xticklabels=MODEL_ORDER,
        yticklabels=MODEL_ORDER,
        mask=mask,
        cbar_kws={"label": "% with same wrong answer"},
        ax=ax2,
        vmin=sw_min,
        vmax=sw_max,
    )
    ax2.set_title(
        "Same Wrong Answer Rate\n(Among overlapping CE errors, % with semantically equivalent wrong answers)"
        + title_suffix,
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    plt.tight_layout()
    out2 = args.output_dir / output_name("fig_same_wrong_answer_heatmap.png", args.tag)
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")

    # Figure 3: combined panel.
    fig3, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(
        overlap_matrix,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        xticklabels=MODEL_ORDER,
        yticklabels=MODEL_ORDER,
        mask=mask,
        cbar_kws={"label": "Count"},
        ax=axes[0],
        vmin=ov_min,
        vmax=ov_max,
    )
    axes[0].set_title("A) CE Overlap Count\n(Both models confidently wrong)" + title_suffix, fontsize=11, fontweight="bold")
    sns.heatmap(
        same_wrong_matrix,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        xticklabels=MODEL_ORDER,
        yticklabels=MODEL_ORDER,
        mask=mask,
        cbar_kws={"label": "%"},
        ax=axes[1],
        vmin=sw_min,
        vmax=sw_max,
    )
    axes[1].set_title("B) Same Wrong Answer %\n(Among overlaps, % with equivalent wrong answer)" + title_suffix, fontsize=11, fontweight="bold")
    plt.tight_layout()
    out3 = args.output_dir / output_name("fig_ce_overlap_combined_heatmap.png", args.tag)
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved: {out3}")

    # Figure 4: same wrong counts + percentages.
    fig4, ax4 = plt.subplots(figsize=(8, 6.5))
    annot_matrix = np.empty_like(same_wrong_count_matrix, dtype=object)
    for i in range(len(MODEL_ORDER)):
        for j in range(len(MODEL_ORDER)):
            if i == j:
                annot_matrix[i, j] = ""
            else:
                count = same_wrong_count_matrix[i, j]
                pct = same_wrong_matrix[i, j]
                annot_matrix[i, j] = f"{int(count)}\n({pct:.0f}%)"
    sns.heatmap(
        same_wrong_count_matrix,
        annot=annot_matrix,
        fmt="",
        cmap="Purples",
        xticklabels=MODEL_ORDER,
        yticklabels=MODEL_ORDER,
        mask=mask,
        cbar_kws={"label": "Questions with same wrong answer"},
        ax=ax4,
        vmin=swc_min,
        vmax=swc_max,
    )
    ax4.set_title("Shared Misconceptions\n(Count and % of CE overlaps with same wrong answer)" + title_suffix, fontsize=12, fontweight="bold")
    ax4.set_xlabel("")
    ax4.set_ylabel("")
    plt.tight_layout()
    out4 = args.output_dir / output_name("fig_shared_misconceptions_heatmap.png", args.tag)
    fig4.savefig(out4, dpi=150, bbox_inches="tight")
    print(f"Saved: {out4}")

    plt.close("all")

    total_overlap = int(df["both_ce_overlap"].sum())
    total_same = int(df["same_wrong_answer"].sum())
    overall = (100.0 * total_same / total_overlap) if total_overlap else 0.0
    print("\nAll heatmaps generated successfully!")
    print("Summary statistics:")
    print(f"  Total CE overlaps: {total_overlap}")
    print(f"  Total same wrong answer: {total_same}")
    print(f"  Overall same wrong %: {overall:.1f}%")


if __name__ == "__main__":
    main()
