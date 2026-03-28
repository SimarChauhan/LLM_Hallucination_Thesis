#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


DEFAULT_ANALYSIS_DIR = Path("data/results/analysis/version_evolution_equiv_only_20260319")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX report for version-evolution equiv_only analysis."
    )
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-tex", type=Path, default=None)
    return parser.parse_args()


def latex_escape(text: object) -> str:
    s = "" if text is None else str(text)
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
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def fmt_pct(x: float, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x:.{digits}f}\\%"


def fmt_float(x: float, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x:.{digits}f}"


def rows_model_summary(summary_t09: pd.DataFrame) -> Iterable[str]:
    if summary_t09.empty:
        return [r"\multicolumn{7}{c}{No rows found} \\"]
    out: List[str] = []
    ordered = summary_t09.sort_values(["track", "version_index", "release_date", "model"])
    for _, r in ordered.iterrows():
        out.append(
            " & ".join(
                [
                    latex_escape(r.get("track")),
                    str(int(r.get("version_index", 0))),
                    latex_escape(pd.to_datetime(r.get("release_date"), errors="coerce").strftime("%Y-%m-%d")),
                    latex_escape(r.get("model")),
                    fmt_pct(float(r.get("accuracy_pct", np.nan))),
                    fmt_pct(float(r.get("ce_rate_pct", np.nan))),
                    latex_escape(r.get("source_dataset")),
                ]
            )
            + r" \\"
        )
    return out


def rows_pairwise(pairwise_t09: pd.DataFrame) -> Iterable[str]:
    if pairwise_t09.empty:
        return [r"\multicolumn{8}{c}{No consecutive pair rows found} \\"]
    out: List[str] = []
    ce = pairwise_t09[
        (pairwise_t09["metric"].astype(str) == "ce_rate")
        & (pairwise_t09["consecutive_pair"].astype(bool))
    ].copy()
    ce = ce.sort_values(["track", "older_model", "newer_model"])
    for _, r in ce.iterrows():
        out.append(
            " & ".join(
                [
                    latex_escape(r.get("track")),
                    latex_escape(r.get("older_model")),
                    latex_escape(r.get("newer_model")),
                    str(int(r.get("n_paired_questions", 0))),
                    fmt_float(float(r.get("improvement_pp", np.nan)), 2),
                    fmt_float(float(r.get("bootstrap_ci_low_pp", np.nan)), 2),
                    fmt_float(float(r.get("bootstrap_ci_high_pp", np.nan)), 2),
                    fmt_float(float(r.get("mcnemar_p_exact", np.nan)), 4),
                ]
            )
            + r" \\"
        )
    return out


def rows_light_trends(trends: pd.DataFrame) -> Iterable[str]:
    if trends.empty:
        return [r"\multicolumn{5}{c}{No trend rows found} \\"]
    out: List[str] = []
    ordered = trends.sort_values(["track", "metric"])
    for _, r in ordered.iterrows():
        out.append(
            " & ".join(
                [
                    latex_escape(r.get("track")),
                    latex_escape(r.get("metric")),
                    str(int(r.get("n_versions", 0))),
                    fmt_float(float(r.get("slope_pp_per_version", np.nan)), 3),
                    fmt_float(float(r.get("intercept", np.nan)), 3),
                ]
            )
            + r" \\"
        )
    return out


def build_tex(
    analysis_dir: Path,
    summary_t09: pd.DataFrame,
    pairwise_t09: pd.DataFrame,
    trend_light: pd.DataFrame,
    checks: dict,
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    model_rows = list(rows_model_summary(summary_t09))
    pair_rows = list(rows_pairwise(pairwise_t09))
    trend_rows = list(rows_light_trends(trend_light))

    n_models = int(checks.get("n_models", 0))
    n_questions = int(checks.get("n_questions", 0))
    row_ok = bool(checks.get("all_models_have_807_rows", False))
    overlap = checks.get("common_question_ids_by_track", {}) or {}
    fallback = checks.get("equivalence_fallback_counts", {}) or {}

    qwen_seq = summary_t09[summary_t09["track"] == "qwen_scale_version"].sort_values("version_index")["ce_rate"].tolist()
    llama_seq = summary_t09[summary_t09["track"] == "llama_scale_version"].sort_values("version_index")["ce_rate"].tolist()
    grok_seq = summary_t09[summary_t09["track"] == "grok_version"].sort_values("version_index")["ce_rate"].tolist()
    seq = lambda xs: " -> ".join(f"{v:.4f}" for v in xs) if xs else "N/A"

    def fallback_line(fam: str) -> str:
        vals = fallback.get(fam, {})
        return f"NLI={int(vals.get('NLI', 0))}, LLM={int(vals.get('LLM', 0))}, OTHER={int(vals.get('OTHER', 0))}"

    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{float}}
\usepackage{{hyperref}}

\title{{Version-Evolution SE/CE Report (Equiv-Only)}}
\author{{Simranjeet Singh}}
\date{{Generated on {latex_escape(generated)}}}

\begin{{document}}
\maketitle

\section{{Scope}}
Families analyzed: Qwen, Llama, Grok. Primary metric is self-consistent error rate at threshold 0.9 (\texttt{{error\_label\_0.9}}). Threshold sensitivity is included for 1.0 / 0.9 / 0.8 / 0.7.

\section{{Integrity Gates}}
\begin{{itemize}}
  \item Combined rows: \textbf{{{int(checks.get("combined_rows", 0))}}} (expected 9684; match={latex_escape(checks.get("combined_rows_match_expected"))})
  \item Unique (model, question\_id) keys: \textbf{{{int(checks.get("unique_model_question_keys", 0))}}} (keys\_match\_rows={latex_escape(checks.get("keys_match_rows"))})
  \item Models in combined set: \textbf{{{n_models}}}; distinct question\_ids observed: \textbf{{{n_questions}}}
  \item All models have 807 rows: \textbf{{{latex_escape(row_ok)}}}
\end{{itemize}}

\noindent Common question-id overlap across all 4 versions:
\begin{{itemize}}
  \item qwen\_scale\_version: {int(overlap.get("qwen_scale_version", 0))}
  \item llama\_scale\_version: {int(overlap.get("llama_scale_version", 0))}
  \item grok\_version: {int(overlap.get("grok_version", 0))}
\end{{itemize}}

\noindent Equivalence fallback totals:
\begin{{itemize}}
  \item Qwen: {fallback_line("qwen")}
  \item Llama: {fallback_line("llama")}
  \item Grok: {fallback_line("grok")}
\end{{itemize}}

\section{{Quick Snapshot (CE@0.9)}}
\begin{{itemize}}
  \item Qwen CE sequence: \texttt{{{seq(qwen_seq)}}}
  \item Llama CE sequence: \texttt{{{seq(llama_seq)}}}
  \item Grok CE sequence: \texttt{{{seq(grok_seq)}}}
\end{{itemize}}

\section{{Model Summary (CE@0.9)}}
\small
\begin{{longtable}}{{p{{2.6cm}} c p{{1.9cm}} p{{4.8cm}} r r p{{2.6cm}}}}
\toprule
Track & Ver & Date & Model & Acc. & CE & Source \\
\midrule
{chr(10).join(model_rows)}
\bottomrule
\end{{longtable}}
\normalsize

\section{{Consecutive Pairwise CE Deltas (t=0.9)}}
\small
\begin{{longtable}}{{p{{2.2cm}} p{{3.1cm}} p{{3.1cm}} r r r r r}}
\toprule
Track & Older & Newer & n paired & Improve pp & CI low & CI high & McNemar p \\
\midrule
{chr(10).join(pair_rows)}
\bottomrule
\end{{longtable}}
\normalsize

\section{{Light Trend Slopes}}
\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{p{{3.2cm}} p{{2.3cm}} c r r}}
\toprule
Track & Metric & n versions & Slope (pp/ver) & Intercept \\
\midrule
{chr(10).join(trend_rows)}
\bottomrule
\end{{tabular}}
\caption{{Model-level OLS slopes over version index.}}
\end{{table}}

\section{{Figures}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{ce_rate_over_time_t0p9.png}}
\caption{{Self-consistent error rate over time (threshold 0.9).}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{ce_rate_threshold_sensitivity.png}}
\caption{{Threshold sensitivity by track (1.0/0.9/0.8/0.7).}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{pairwise_ce_deltas_consecutive_t0p9.png}}
\caption{{Consecutive pairwise CE improvements with bootstrap CIs.}}
\end{{figure}}

\section{{Caveats}}
\begin{{itemize}}
  \item Latest endpoints are reused from existing 4842-hybrid data; older versions are from new equiv-only reruns.
  \item Mixed source origin can shift absolute levels; trend direction is the more reliable signal.
  \item Pairwise comparisons are computed on intersected question IDs; some family intersections are 797 rather than 807.
\end{{itemize}}

\section{{Reproducibility}}
\texttt{{{latex_escape(str(analysis_dir.resolve()))}}}

\end{{document}}
"""


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")

    summary_t09 = pd.read_csv(analysis_dir / "model_summary_t0p9.csv")
    pairwise_t09 = pd.read_csv(analysis_dir / "pairwise_deltas_t0p9.csv")
    trend_light = pd.read_csv(analysis_dir / "trend_tests_light_t0p9.csv")
    checks = json.loads((analysis_dir / "validation_checks.json").read_text(encoding="utf-8"))

    tex = build_tex(analysis_dir, summary_t09, pairwise_t09, trend_light, checks)
    out_tex = args.output_tex or (analysis_dir / "version_evolution_equiv_only_report.tex")
    out_tex.write_text(tex, encoding="utf-8")
    print(f"[done] wrote {out_tex}")


if __name__ == "__main__":
    main()
