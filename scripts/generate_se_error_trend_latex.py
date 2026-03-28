#!/usr/bin/env python3
"""
Render a LaTeX report for SE-error trend analysis outputs.

Expected input directory contents:
- historical_se_model_summary.csv
- nibi_se_sync_summary.csv
- nibi_se_delta_pp.csv
- track_slope_summary.csv
- figures/01_historical_release_timeline.png
- figures/02_nibi_sync_timeline.png
- figures/03_nibi_delta_pp.png
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("data/results/analysis/se_error_trends")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LaTeX report for SE-error trends.")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Path to one run_* analysis folder. Defaults to latest run under data/results/analysis/se_error_trends.",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=None,
        help="Output .tex path. Defaults to <analysis-dir>/se_error_trend_report.tex",
    )
    return parser.parse_args()


def latest_run_dir(root: Path) -> Path:
    runs = sorted([p for p in root.glob("run_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run_* directories found under {root}")
    return runs[-1]


def latex_escape(text: object) -> str:
    s = "" if text is None else str(text)
    replacements = {
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
    for k, v in replacements.items():
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


def table_rows_historical_latest(df: pd.DataFrame) -> Iterable[str]:
    if df.empty:
        return []
    out = (
        df.sort_values(["model", "model_release_date", "run_date"])
        .drop_duplicates(subset=["model"], keep="last")
        .sort_values("se_error_rate_pct", ascending=False)
    )
    rows = []
    for _, r in out.iterrows():
        model = latex_escape(r.get("model"))
        track = latex_escape(r.get("model_track"))
        release_date = pd.to_datetime(r.get("model_release_date"), errors="coerce")
        release = release_date.strftime("%Y-%m-%d") if pd.notna(release_date) else "N/A"
        se_rate = fmt_pct(float(r.get("se_error_rate_pct", np.nan)))
        n_rows = int(r.get("n_rows", 0))
        rows.append(f"{model} & {track} & {release} & {se_rate} & {n_rows} \\\\")
    return rows


def table_rows_nibi_latest(df: pd.DataFrame) -> Iterable[str]:
    if df.empty:
        return []
    out = (
        df.sort_values(["target_model_name", "sync_date"])
        .drop_duplicates(subset=["target_model_name"], keep="last")
        .sort_values("se_error_rate_from_ce_count_pct", ascending=False)
    )
    rows = []
    for _, r in out.iterrows():
        model = latex_escape(r.get("target_model_name"))
        sync_date = pd.to_datetime(r.get("sync_date"), errors="coerce")
        date_text = sync_date.strftime("%Y-%m-%d") if pd.notna(sync_date) else "N/A"
        rate = fmt_pct(float(r.get("se_error_rate_from_ce_count_pct", np.nan)))
        ce_err = int(r.get("ce_negative_error", 0))
        n_questions = int(r.get("n_questions_total", 0))
        rows.append(f"{model} & {date_text} & {rate} & {ce_err} & {n_questions} \\\\")
    return rows


def table_rows_nibi_delta(df: pd.DataFrame) -> Iterable[str]:
    if df.empty:
        return []
    out = df.sort_values("target_model_short")
    rows = []
    for _, r in out.iterrows():
        model = latex_escape(r.get("target_model_short"))
        first_date = pd.to_datetime(r.get("first_date"), errors="coerce")
        last_date = pd.to_datetime(r.get("last_date"), errors="coerce")
        first_txt = first_date.strftime("%Y-%m-%d") if pd.notna(first_date) else "N/A"
        last_txt = last_date.strftime("%Y-%m-%d") if pd.notna(last_date) else "N/A"
        delta = fmt_float(float(r.get("delta_pp", np.nan)), digits=3)
        n_dates = int(r.get("n_dates", 0))
        rows.append(f"{model} & {first_txt} & {last_txt} & {delta} & {n_dates} \\\\")
    return rows


def build_tex(
    analysis_dir: Path,
    historical: pd.DataFrame,
    nibi: pd.DataFrame,
    delta: pd.DataFrame,
    slopes: pd.DataFrame,
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    hist_mean = float(historical["se_error_rate_pct"].mean()) if not historical.empty else np.nan
    nibi_mean = float(nibi["se_error_rate_from_ce_count_pct"].mean()) if not nibi.empty else np.nan
    max_delta = float(delta["delta_pp"].abs().max()) if not delta.empty else np.nan
    n_models_nibi = int(nibi["target_model_name"].nunique()) if not nibi.empty else 0

    nibi_dates = (
        sorted(pd.to_datetime(nibi["sync_date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique().tolist())
        if not nibi.empty
        else []
    )
    nibi_dates_txt = ", ".join(nibi_dates) if nibi_dates else "N/A"

    top_slope_track = "N/A"
    top_slope_pp = np.nan
    if not slopes.empty:
        top = slopes.iloc[0]
        top_slope_track = latex_escape(top.get("model_track"))
        top_slope_pp = float(top.get("slope_per_version_pp", np.nan))

    hist_rows = list(table_rows_historical_latest(historical))
    nibi_rows = list(table_rows_nibi_latest(nibi))
    delta_rows = list(table_rows_nibi_delta(delta))

    if not hist_rows:
        hist_rows = [r"\multicolumn{5}{c}{No historical rows found} \\"]
    if not nibi_rows:
        nibi_rows = [r"\multicolumn{5}{c}{No Nibi rows found} \\"]
    if not delta_rows:
        delta_rows = [r"\multicolumn{5}{c}{No per-model delta rows found} \\"]

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\usepackage{{float}}
\usepackage{{longtable}}

\title{{SE Error Trend Report (LaTeX)}}
\author{{Simranjeet Singh}}
\date{{Generated on {latex_escape(generated)}}}

\begin{{document}}
\maketitle

\section{{Overview}}
This report summarizes self-consistent error (SE) trends using:
\begin{{itemize}}
  \item Historical version-evolution evaluated outputs already in the repository.
  \item Nibi white-box sync snapshots and legacy Nibi probe exports.
\end{{itemize}}

\section{{Executive Summary}}
\begin{{itemize}}
  \item Historical mean SE error rate: \textbf{{{fmt_pct(hist_mean)}}}
  \item Nibi mean SE error rate: \textbf{{{fmt_pct(nibi_mean)}}}
  \item Max model-level Nibi drift (percentage points): \textbf{{{fmt_float(max_delta)}}}
  \item Nibi models covered: \textbf{{{n_models_nibi}}}
  \item Nibi snapshot dates observed: \texttt{{{latex_escape(nibi_dates_txt)}}}
  \item Largest historical linear slope track: \texttt{{{top_slope_track}}} ({fmt_float(top_slope_pp, 2)} pp/version)
\end{{itemize}}

\section{{Historical SE Rates (Latest by Model)}}
\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{p{{5.2cm}} p{{3.2cm}} p{{2cm}} r r}}
\toprule
Model & Track & Release Date & SE Rate & n \\
\midrule
{chr(10).join(hist_rows)}
\bottomrule
\end{{tabular}}
\caption{{Historical SE error rate at threshold 0.9 (latest entry per model).}}
\end{{table}}

\section{{Nibi SE Rates (Latest by Model)}}
\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{p{{5.2cm}} p{{2cm}} r r r}}
\toprule
Target Model & Latest Date & SE Rate & CE Errors & Questions \\
\midrule
{chr(10).join(nibi_rows)}
\bottomrule
\end{{tabular}}
\caption{{Nibi SE error rate reconstructed from CE counts (CE threshold 1.0).}}
\end{{table}}

\section{{Nibi Date-Range Delta by Model}}
\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{p{{4cm}} p{{2cm}} p{{2cm}} r r}}
\toprule
Model & First Date & Last Date & Delta (pp) & \#Dates \\
\midrule
{chr(10).join(delta_rows)}
\bottomrule
\end{{tabular}}
\caption{{Change from first available to latest available Nibi snapshot per model.}}
\end{{table}}

\section{{Figures}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{figures/01_historical_release_timeline.png}}
\caption{{Historical SE error rate over model release timeline.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{figures/02_nibi_sync_timeline.png}}
\caption{{Nibi sync-date SE error trends by target model.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.90\textwidth]{{figures/03_nibi_delta_pp.png}}
\caption{{Nibi SE error delta (first available date to latest date) by model.}}
\end{{figure}}

\section{{Reproducibility}}
Source analysis folder:
\begin{{quote}}
\texttt{{{latex_escape(str(analysis_dir.resolve()))}}}
\end{{quote}}

\end{{document}}
"""
    return tex


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir or latest_run_dir(DEFAULT_ROOT)
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")

    historical = pd.read_csv(analysis_dir / "historical_se_model_summary.csv")
    nibi = pd.read_csv(analysis_dir / "nibi_se_sync_summary.csv")
    delta = pd.read_csv(analysis_dir / "nibi_se_delta_pp.csv")
    slopes = pd.read_csv(analysis_dir / "track_slope_summary.csv")

    tex = build_tex(analysis_dir, historical, nibi, delta, slopes)
    output_tex = args.output_tex or (analysis_dir / "se_error_trend_report.tex")
    output_tex.write_text(tex, encoding="utf-8")
    print(f"[done] Wrote LaTeX report: {output_tex}")


if __name__ == "__main__":
    main()
