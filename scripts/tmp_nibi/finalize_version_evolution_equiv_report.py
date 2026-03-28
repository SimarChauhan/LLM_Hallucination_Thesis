#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("data/results/analysis/version_evolution_equiv_only_20260319")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_PATH = OUT_DIR / "combined_version_evolution_equiv_only.jsonl"
NEW_FILES = {
    "qwen": Path(
        "data/results/evaluated/run_qwen_new_only_807_full_retry2_20260315T193059Z/"
        "results_version_evolution_qwen_new_only_eval.equiv_only_20260319.jsonl"
    ),
    "llama": Path(
        "data/results/evaluated/run_llama_new_only_807_p1_20260315T222326Z/"
        "results_version_evolution_llama_scale_version_807_eval.equiv_only_20260319.jsonl"
    ),
    "grok": Path(
        "data/results/evaluated/run_grok_new_only_807_p1_xai_20260315T224013Z/"
        "results_version_evolution_grok_new_only_807_eval.equiv_only_20260319.jsonl"
    ),
}

MAIN_THRESHOLD = "0.9"
THRESHOLDS = ["1.0", "0.9", "0.8", "0.7"]

TRACK_LABELS = {
    "qwen_scale_version": "Qwen",
    "llama_scale_version": "Llama",
    "grok_version": "Grok",
}
COLORS = {
    "qwen_scale_version": "#9c6644",
    "llama_scale_version": "#386641",
    "grok_version": "#1d3557",
}


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


def load_combined_df() -> pd.DataFrame:
    if not COMBINED_PATH.exists():
        raise FileNotFoundError(f"Missing combined file: {COMBINED_PATH}")
    df = pd.read_json(COMBINED_PATH, lines=True)
    if df.empty:
        raise ValueError("Combined dataset is empty.")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["version_index"] = pd.to_numeric(df["version_index"], errors="coerce")
    return df


def bootstrap_mean_ci(values: np.ndarray, num_bootstrap: int = 600, seed: int = 42) -> Tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        v = float(values[0])
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(num_bootstrap, values.size))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def exact_binomial_two_sided(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    tail_prob = 0.0
    for i in range(0, k + 1):
        tail_prob += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * tail_prob)


def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return exact_binomial_two_sided(min(b, c), n)


def compute_model_summary(df: pd.DataFrame, threshold: str) -> pd.DataFrame:
    out = df.copy()
    out["is_correct"] = out["greedy_correct"].map(_to_bool)
    out["is_ce"] = out[f"error_label_{threshold}"].astype(str).eq("self_consistent_error")
    out["is_ie"] = out[f"error_label_{threshold}"].astype(str).eq("inconsistent_error")
    gcols = ["track", "family", "version_index", "release_date", "model", "source_dataset", "protocol_group"]
    res = (
        out.groupby(gcols, dropna=False)
        .agg(
            n_rows=("question_id", "size"),
            n_questions=("question_id", "nunique"),
            accuracy=("is_correct", "mean"),
            ce_rate=("is_ce", "mean"),
            ie_rate=("is_ie", "mean"),
        )
        .reset_index()
        .sort_values(["track", "version_index", "model"])
    )
    for c in ["accuracy", "ce_rate", "ie_rate"]:
        res[f"{c}_pct"] = 100.0 * res[c]
    res["threshold"] = threshold
    return res


def compute_pairwise(df: pd.DataFrame, threshold: str) -> pd.DataFrame:
    out = df.copy()
    out["is_correct"] = out["greedy_correct"].map(_to_bool).astype(int)
    out["is_ce"] = out[f"error_label_{threshold}"].astype(str).eq("self_consistent_error").astype(int)

    rows: List[Dict[str, Any]] = []
    for track, tdf in out.groupby("track", dropna=False):
        ordered = (
            tdf[["model", "version_index"]]
            .drop_duplicates()
            .sort_values(["version_index", "model"])
            ["model"]
            .tolist()
        )
        for i, older in enumerate(ordered):
            for j in range(i + 1, len(ordered)):
                newer = ordered[j]
                left = tdf[tdf["model"] == older][["question_id", "is_correct", "is_ce"]]
                right = tdf[tdf["model"] == newer][["question_id", "is_correct", "is_ce"]]
                merged = left.merge(right, on="question_id", suffixes=("_old", "_new"))
                if merged.empty:
                    continue
                for metric, col_old, col_new, higher_is_better in [
                    ("accuracy", "is_correct_old", "is_correct_new", True),
                    ("ce_rate", "is_ce_old", "is_ce_new", False),
                ]:
                    old_vals = merged[col_old].to_numpy(dtype=float)
                    new_vals = merged[col_new].to_numpy(dtype=float)
                    diffs = new_vals - old_vals
                    ci_lo, ci_hi = bootstrap_mean_ci(
                        diffs,
                        num_bootstrap=600,
                        seed=42 + i * 101 + j * 17 + (0 if metric == "accuracy" else 1),
                    )
                    b = int(((old_vals == 1) & (new_vals == 0)).sum())
                    c = int(((old_vals == 0) & (new_vals == 1)).sum())
                    delta = float(diffs.mean())
                    improvement = delta if higher_is_better else -delta
                    rows.append(
                        {
                            "track": track,
                            "older_model": older,
                            "newer_model": newer,
                            "consecutive_pair": bool(j == i + 1),
                            "n_paired_questions": int(len(merged)),
                            "metric": metric,
                            "older_rate": float(old_vals.mean()),
                            "newer_rate": float(new_vals.mean()),
                            "delta_new_minus_old": delta,
                            "delta_new_minus_old_pp": 100.0 * delta,
                            "improvement_pp": 100.0 * improvement,
                            "bootstrap_ci_low_pp": 100.0 * ci_lo,
                            "bootstrap_ci_high_pp": 100.0 * ci_hi,
                            "mcnemar_b_old1_new0": b,
                            "mcnemar_c_old0_new1": c,
                            "mcnemar_p_exact": mcnemar_exact_p(b, c),
                            "threshold": threshold,
                        }
                    )
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    return res.sort_values(["track", "metric", "older_model", "newer_model"]).reset_index(drop=True)


def compute_light_trends(summary_main: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for track, tdf in summary_main.groupby("track", dropna=False):
        tdf = tdf.sort_values("version_index")
        if tdf["version_index"].nunique() < 2:
            continue
        x = tdf["version_index"].to_numpy(dtype=float)
        for metric in ["accuracy_pct", "ce_rate_pct"]:
            y = tdf[metric].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            rows.append(
                {
                    "track": track,
                    "metric": metric,
                    "n_versions": int(len(tdf)),
                    "slope_pp_per_version": float(slope),
                    "intercept": float(intercept),
                }
            )
    return pd.DataFrame(rows).sort_values(["track", "metric"]).reset_index(drop=True)


def validate_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    checks["combined_rows"] = int(len(df))
    checks["expected_combined_rows"] = 9684
    checks["combined_rows_match_expected"] = bool(len(df) == 9684)
    keys = df[["model", "question_id"]].drop_duplicates()
    checks["unique_model_question_keys"] = int(len(keys))
    checks["keys_match_rows"] = bool(len(keys) == len(df))

    per_model = df.groupby("model")["question_id"].size().to_dict()
    checks["rows_per_model"] = {str(k): int(v) for k, v in per_model.items()}
    checks["all_models_have_807_rows"] = bool(all(v == 807 for v in per_model.values()))
    checks["n_models"] = int(df["model"].nunique())
    checks["n_questions"] = int(df["question_id"].nunique())

    protocol = {}
    fallback = {}
    one_failed_proxy = {}
    for fam, path in NEW_FILES.items():
        dfi = pd.read_json(path, lines=True)
        protocol[fam] = {
            "rows": int(len(dfi)),
            "keys": int(dfi[["model", "question_id"]].drop_duplicates().shape[0]),
            "judge_protocol_nunique": int(dfi["judge_protocol"].nunique()) if "judge_protocol" in dfi.columns else 0,
            "judge_protocol_values": sorted(dfi["judge_protocol"].dropna().astype(str).unique().tolist())
            if "judge_protocol" in dfi.columns
            else [],
            "hybrid_enabled_values": sorted(dfi["hybrid_enabled"].dropna().astype(str).unique().tolist())
            if "hybrid_enabled" in dfi.columns
            else [],
            "equivalence_only_eval_values": sorted(dfi["equivalence_only_eval"].dropna().astype(str).unique().tolist())
            if "equivalence_only_eval" in dfi.columns
            else [],
        }
        if "equivalence_decision_source" in dfi.columns:
            nli_count = 0
            llm_count = 0
            other_count = 0
            for item in dfi["equivalence_decision_source"].tolist():
                if isinstance(item, list):
                    for v in item:
                        tag = str(v).upper()
                        if tag == "NLI":
                            nli_count += 1
                        elif tag == "LLM":
                            llm_count += 1
                        else:
                            other_count += 1
                else:
                    tag = str(item).upper()
                    if tag == "NLI":
                        nli_count += 1
                    elif tag == "LLM":
                        llm_count += 1
                    else:
                        other_count += 1
            fallback[fam] = {"NLI": int(nli_count), "LLM": int(llm_count), "OTHER": int(other_count)}
        else:
            fallback[fam] = {}

        one_failed = 0
        mattered_proxy = 0
        if "correctness_judge_statuses" in dfi.columns and "correctness_judge_grades" in dfi.columns:
            for _, row in dfi.iterrows():
                sts = row.get("correctness_judge_statuses")
                grs = row.get("correctness_judge_grades")
                if not isinstance(sts, list) or not isinstance(grs, list) or len(sts) != len(grs):
                    continue
                bad_idx = [i for i, s in enumerate(sts) if str(s) != "OK"]
                if len(bad_idx) != 1:
                    continue
                one_failed += 1
                ok_grades = [str(grs[i]) for i, s in enumerate(sts) if str(s) == "OK" and grs[i] is not None]
                if len(ok_grades) == 2 and ok_grades[0] != ok_grades[1]:
                    mattered_proxy += 1
        one_failed_proxy[fam] = {"one_failed_rows": one_failed, "mattered_proxy_rows": mattered_proxy}

    checks["new_file_protocol_checks"] = protocol
    checks["equivalence_fallback_counts"] = fallback
    checks["one_failed_mattered_proxy_from_equiv_only"] = one_failed_proxy

    track_overlap: Dict[str, int] = {}
    for track, tdf in df.groupby("track", dropna=False):
        model_sets = []
        for _, mdf in tdf.groupby("model", dropna=False):
            model_sets.append(set(mdf["question_id"].astype(str).tolist()))
        if not model_sets:
            continue
        inter = set.intersection(*model_sets)
        track_overlap[str(track)] = int(len(inter))
    checks["common_question_ids_by_track"] = track_overlap
    return checks


def make_figures(summary_main: pd.DataFrame, summary_all: pd.DataFrame, pairwise_main: pd.DataFrame) -> List[Path]:
    out_paths: List[Path] = []

    # CE timeline t=0.9
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
    for ax, track in zip(axes, ["qwen_scale_version", "llama_scale_version", "grok_version"]):
        sub = summary_main[summary_main["track"] == track].sort_values("version_index")
        ax.plot(sub["version_index"], sub["ce_rate_pct"], marker="o", color=COLORS[track], linewidth=2.2)
        for _, r in sub.iterrows():
            marker = "D" if str(r["source_dataset"]) == "existing_4842_hybrid" else "o"
            ax.scatter(
                r["version_index"],
                r["ce_rate_pct"],
                color=COLORS[track],
                s=68,
                marker=marker,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
            ax.text(r["version_index"], r["ce_rate_pct"] + 0.8, f"{r['ce_rate_pct']:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(TRACK_LABELS[track])
        ax.set_xlabel("Version index")
        ax.grid(alpha=0.25)
        ax.set_xticks(sub["version_index"].tolist())
    axes[0].set_ylabel("SE/CE rate (%) at t=0.9")
    fig.suptitle("Self-Consistent Error Rate Over Time", y=1.03, fontsize=14)
    fig.tight_layout()
    p1 = OUT_DIR / "ce_rate_over_time_t0p9.png"
    fig.savefig(p1, dpi=220, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(p1)

    # Threshold sensitivity
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
    tcolors = {"1.0": "#6c757d", "0.9": "#d62828", "0.8": "#f77f00", "0.7": "#2a9d8f"}
    for ax, track in zip(axes, ["qwen_scale_version", "llama_scale_version", "grok_version"]):
        sub = summary_all[summary_all["track"] == track].sort_values(["threshold", "version_index"])
        for thr in THRESHOLDS:
            ss = sub[sub["threshold"] == thr].sort_values("version_index")
            ax.plot(ss["version_index"], ss["ce_rate_pct"], marker="o", linewidth=1.8, color=tcolors[thr], label=f"t={thr}")
        ax.set_title(TRACK_LABELS[track])
        ax.set_xlabel("Version index")
        ax.grid(alpha=0.25)
        ax.set_xticks(sorted(sub["version_index"].unique()))
    axes[0].set_ylabel("SE/CE rate (%)")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Threshold Sensitivity (SE/CE)", y=1.03, fontsize=14)
    fig.tight_layout()
    p2 = OUT_DIR / "ce_rate_threshold_sensitivity.png"
    fig.savefig(p2, dpi=220, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(p2)

    # Consecutive pairwise CE improvements
    ce_consec = pairwise_main[(pairwise_main["metric"] == "ce_rate") & (pairwise_main["consecutive_pair"])].copy()
    ce_consec = ce_consec.sort_values(["track", "improvement_pp"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [f"{TRACK_LABELS.get(t, t)}: {o} -> {n}" for t, o, n in zip(ce_consec["track"], ce_consec["older_model"], ce_consec["newer_model"])]
    x = np.arange(len(ce_consec))
    y_lo = np.maximum(ce_consec["improvement_pp"] - ce_consec["bootstrap_ci_low_pp"], 0.0)
    y_hi = np.maximum(ce_consec["bootstrap_ci_high_pp"] - ce_consec["improvement_pp"], 0.0)
    ax.bar(x, ce_consec["improvement_pp"], color="#457b9d", alpha=0.9)
    ax.errorbar(
        x,
        ce_consec["improvement_pp"],
        yerr=[y_lo, y_hi],
        fmt="none",
        ecolor="black",
        elinewidth=0.9,
        capsize=3,
    )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("CE improvement (pp, positive is better)")
    ax.set_title("Consecutive Version Deltas (t=0.9)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p3 = OUT_DIR / "pairwise_ce_deltas_consecutive_t0p9.png"
    fig.savefig(p3, dpi=220, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(p3)
    return out_paths


def write_report(
    df: pd.DataFrame,
    summary_main: pd.DataFrame,
    summary_all: pd.DataFrame,
    pairwise_main: pd.DataFrame,
    light_trends: pd.DataFrame,
    checks: Dict[str, Any],
    figure_paths: List[Path],
) -> Path:
    qwen = summary_main[summary_main["track"] == "qwen_scale_version"].sort_values("version_index")
    llama = summary_main[summary_main["track"] == "llama_scale_version"].sort_values("version_index")
    grok = summary_main[summary_main["track"] == "grok_version"].sort_values("version_index")

    def _seq_text(x: pd.DataFrame) -> str:
        return " -> ".join(f"{v:.4f}" for v in x["ce_rate"].tolist())

    lines: List[str] = []

    def _table_text(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "(no rows)"
        return frame.to_string(index=False)
    lines.append("# Version-Evolution Report: Self-Consistent Errors Over Time")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Families analyzed: Qwen, Llama, Grok.")
    lines.append("- Primary metric: `self_consistent_error` rate at threshold 0.9 (`error_label_0.9`).")
    lines.append("- Sensitivity included for thresholds 1.0 / 0.9 / 0.8 / 0.7.")
    lines.append("")
    lines.append("## Integrity Gates")
    lines.append(f"- Combined rows: `{checks['combined_rows']}` (expected `9684`) -> `{checks['combined_rows_match_expected']}`")
    lines.append(f"- Unique `(model, question_id)` keys: `{checks['unique_model_question_keys']}` (match rows: `{checks['keys_match_rows']}`)")
    lines.append(f"- All models have 807 rows: `{checks['all_models_have_807_rows']}`")
    lines.append(f"- Models in combined set: `{checks['n_models']}`; questions: `{checks['n_questions']}`")
    lines.append("")
    lines.append("### Protocol Checks (new equiv_only files)")
    for fam, payload in checks["new_file_protocol_checks"].items():
        lines.append(
            f"- {fam}: rows={payload['rows']}, keys={payload['keys']}, "
            f"judge_protocol_nunique={payload['judge_protocol_nunique']}, "
            f"hybrid_enabled={payload['hybrid_enabled_values']}, "
            f"equivalence_only_eval={payload['equivalence_only_eval_values']}"
        )
    lines.append("")
    lines.append("### Fallback Usage (equivalence decisions)")
    for fam, payload in checks["equivalence_fallback_counts"].items():
        lines.append(f"- {fam}: {payload}")
    lines.append("")

    lines.append("### Common Question-ID Overlap by Track")
    for track, n in checks.get("common_question_ids_by_track", {}).items():
        lines.append(f"- {track}: common question_id count across all 4 versions = {n}")
    lines.append("")

    lines.append("### One-Failed/Mattered Proxy from equiv_only files")
    for fam, payload in checks["one_failed_mattered_proxy_from_equiv_only"].items():
        lines.append(f"- {fam}: one_failed={payload['one_failed_rows']}, mattered_proxy={payload['mattered_proxy_rows']}")
    lines.append("")
    lines.append("## Quick Trend Snapshot (t=0.9)")
    lines.append(f"- Qwen CE sequence: `{_seq_text(qwen)}` (non-monotonic)")
    lines.append(f"- Llama CE sequence: `{_seq_text(llama)}` (non-monotonic)")
    lines.append(f"- Grok CE sequence: `{_seq_text(grok)}` (clear downward trend)")
    lines.append("")
    lines.append("## CE@0.9 Summary Table")
    lines.append("```")
    lines.append(
        _table_text(
            summary_main[
                ["track", "version_index", "release_date", "model", "source_dataset", "accuracy_pct", "ce_rate_pct", "ie_rate_pct"]
            ]
        )
    )
    lines.append("```")
    lines.append("")
    lines.append("## Light Trend Slopes (model-level OLS, pp/version)")
    lines.append("```")
    lines.append(_table_text(light_trends))
    lines.append("```")
    lines.append("")
    lines.append("## Consecutive Pairwise Deltas (CE@0.9)")
    ce_consec = pairwise_main[(pairwise_main["metric"] == "ce_rate") & (pairwise_main["consecutive_pair"])].copy()
    lines.append("```")
    lines.append(
        _table_text(
            ce_consec[
                ["track", "older_model", "newer_model", "improvement_pp", "bootstrap_ci_low_pp", "bootstrap_ci_high_pp", "mcnemar_p_exact"]
            ]
        )
    )
    lines.append("```")
    lines.append("")
    lines.append("## Caveats")
    lines.append("- Latest endpoints are reused from existing dataset; older versions are from new equiv_only reruns.")
    lines.append("- Mixed source origin means absolute levels should be interpreted with caution; trend direction is stronger evidence.")
    lines.append("- Pairwise comparisons are computed on question_id intersections; some family pairs align on 797 shared IDs rather than full 807.")
    lines.append("")
    lines.append("## Figures")
    for p in figure_paths:
        lines.append(f"- {p}")
    lines.append("")

    report_path = OUT_DIR / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    df = load_combined_df()
    checks = validate_integrity(df)
    (OUT_DIR / "validation_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    summaries: List[pd.DataFrame] = []
    pairwise_rows: List[pd.DataFrame] = []
    for thr in THRESHOLDS:
        sm = compute_model_summary(df, thr)
        summaries.append(sm)
        sm.to_csv(OUT_DIR / f"model_summary_t{thr.replace('.', 'p')}.csv", index=False)
        pw = compute_pairwise(df, thr)
        pairwise_rows.append(pw)
        pw.to_csv(OUT_DIR / f"pairwise_deltas_t{thr.replace('.', 'p')}.csv", index=False)

    summary_all = pd.concat(summaries, ignore_index=True)
    pairwise_all = pd.concat(pairwise_rows, ignore_index=True)
    summary_all.to_csv(OUT_DIR / "model_summary_all_thresholds.csv", index=False)
    pairwise_all.to_csv(OUT_DIR / "pairwise_deltas_all_thresholds.csv", index=False)

    summary_main = summary_all[summary_all["threshold"] == MAIN_THRESHOLD].copy()
    pairwise_main = pairwise_all[pairwise_all["threshold"] == MAIN_THRESHOLD].copy()
    light_trends = compute_light_trends(summary_main)
    light_trends.to_csv(OUT_DIR / "trend_tests_light_t0p9.csv", index=False)

    figure_paths = make_figures(summary_main, summary_all, pairwise_main)
    report_path = write_report(df, summary_main, summary_all, pairwise_main, light_trends, checks, figure_paths)

    manifest = {
        "out_dir": str(OUT_DIR),
        "combined_rows": int(len(df)),
        "models": int(df["model"].nunique()),
        "questions": int(df["question_id"].nunique()),
        "report": str(report_path),
        "figures": [str(p) for p in figure_paths],
        "validation": str(OUT_DIR / "validation_checks.json"),
        "trend_light": str(OUT_DIR / "trend_tests_light_t0p9.csv"),
    }
    (OUT_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
