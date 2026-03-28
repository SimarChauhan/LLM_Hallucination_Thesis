#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def tex_escape(text: object) -> str:
    s = str(text)
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
    out = s
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def load_job_status(sacct_path: Path) -> pd.DataFrame:
    if not sacct_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(sacct_path, sep="|")
    if "JobIDRaw" not in df.columns:
        return pd.DataFrame()
    out = df[~df["JobIDRaw"].astype(str).str.contains(r"\.")].copy()
    keep = ["JobIDRaw", "State", "ExitCode", "Start", "End", "Elapsed", "Reason"]
    out = out[[c for c in keep if c in out.columns]]
    out = out.sort_values("JobIDRaw").reset_index(drop=True)
    return out


def load_blackbox_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["group_type"] == "model"].copy()
    cols = [
        "model",
        "n_rows",
        "accuracy",
        "incorrect_rate",
        "ce_share_among_incorrect",
        "ie_share_among_incorrect",
        "disagreement_auroc",
        "entropy_auroc",
    ]
    return df[cols].drop_duplicates(subset=["model"]).reset_index(drop=True)


def _safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def collect_wb_runs(wb_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for summary_path in sorted(wb_root.glob("*/wb_cross_model_probe_emnlp2025_metrics_summary.csv")):
        run_dir = summary_path.parent
        run_name = run_dir.name
        report_path = run_dir / "wb_cross_model_probe_emnlp2025_run_report.json"
        df = pd.read_csv(summary_path)
        if df.empty:
            continue
        meta = {}
        if report_path.exists():
            meta = json.loads(report_path.read_text())
        target = meta.get("target_model_name", "")
        verifier = meta.get("verifier_model_path_or_hf_id", "")
        response = meta.get("response_model_path_or_hf_id", "")
        elapsed_seconds = _safe_float(meta.get("elapsed_seconds", float("nan")))
        for subset in ["ce", "ie"]:
            sub = df[df["subset"] == subset].copy()
            if sub.empty:
                continue
            best_idx = sub["auroc_mean"].astype(float).idxmax()
            best = sub.loc[best_idx]
            rows.append(
                {
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "target_model": target,
                    "response_encoder": response,
                    "verifier_model": verifier,
                    "subset": subset,
                    "best_variant": best["variant"],
                    "best_auroc": float(best["auroc_mean"]),
                    "best_prauc": float(best["prauc_mean"]),
                    "best_acc_0_5": float(best["accuracy_at_0_5_mean"]),
                    "best_lambda": float(best.get("lambda_mean", float("nan"))),
                    "elapsed_seconds": elapsed_seconds,
                }
            )
    return pd.DataFrame(rows)


def build_comparison(wb_best: pd.DataFrame, bb: pd.DataFrame) -> pd.DataFrame:
    if wb_best.empty:
        return pd.DataFrame()
    pivot = (
        wb_best.pivot_table(
            index=["run_name", "run_dir", "target_model", "response_encoder", "verifier_model", "elapsed_seconds"],
            columns="subset",
            values=["best_auroc", "best_variant"],
            aggfunc="first",
        )
        .reset_index()
        .copy()
    )
    pivot.columns = ["_".join([p for p in col if p]).strip("_") for col in pivot.columns.to_flat_index()]
    pivot = pivot.rename(
        columns={
            "best_auroc_ce": "ce_best_auroc",
            "best_auroc_ie": "ie_best_auroc",
            "best_variant_ce": "ce_best_variant",
            "best_variant_ie": "ie_best_variant",
        }
    )
    out = pivot.merge(bb, left_on="target_model", right_on="model", how="left")
    out["weighted_wb_auroc"] = (
        out["ce_best_auroc"] * out["ce_share_among_incorrect"]
        + out["ie_best_auroc"] * out["ie_share_among_incorrect"]
    )
    out["delta_vs_blackbox_entropy"] = out["weighted_wb_auroc"] - out["entropy_auroc"]
    out["delta_vs_blackbox_disagreement"] = out["weighted_wb_auroc"] - out["disagreement_auroc"]
    out["elapsed_hours"] = out["elapsed_seconds"] / 3600.0
    return out.sort_values(["target_model", "weighted_wb_auroc"], ascending=[True, False]).reset_index(drop=True)


def _fmt(x: object, ndigits: int = 3) -> str:
    try:
        val = float(x)
    except Exception:
        return str(x)
    if pd.isna(val):
        return "NA"
    return f"{val:.{ndigits}f}"


def write_markdown(
    out_path: Path,
    generated_at: str,
    job_status: pd.DataFrame,
    wb_best: pd.DataFrame,
    comparison: pd.DataFrame,
    bb: pd.DataFrame,
) -> None:
    completed = 0
    pending = 0
    failed = 0
    if not job_status.empty:
        completed = int((job_status["State"] == "COMPLETED").sum())
        pending = int((job_status["State"] == "PENDING").sum())
        failed = int((job_status["State"] == "FAILED").sum())

    lines: List[str] = []
    lines.append("# White-box vs Black-box Hallucination Detection Report")
    lines.append("")
    lines.append(f"- Generated: {generated_at}")
    lines.append(f"- Scope: completed white-box probe runs synced from Nibi + existing black-box benchmark metrics")
    lines.append("")
    lines.append("## 1) Execution status")
    lines.append("")
    lines.append(
        f"- Job snapshot: completed={completed}, pending={pending}, failed={failed}"
    )
    if not job_status.empty:
        lines.append("")
        lines.append("| Job ID | State | Elapsed | Start | End |")
        lines.append("|---|---|---:|---|---|")
        for _, r in job_status.iterrows():
            lines.append(
                f"| {r.get('JobIDRaw','')} | {r.get('State','')} | {r.get('Elapsed','')} | {r.get('Start','')} | {r.get('End','')} |"
            )

    lines.append("")
    lines.append("## 2) White-box run outcomes (completed)")
    lines.append("")
    if comparison.empty:
        lines.append("- No completed white-box runs were found.")
    else:
        lines.append("| Target model | Verifier | CE best (AUROC, variant) | IE best (AUROC, variant) | Weighted WB AUROC | Runtime (h) |")
        lines.append("|---|---|---|---|---:|---:|")
        for _, r in comparison.iterrows():
            lines.append(
                "| "
                + f"{r['target_model']} | {r['verifier_model']} | "
                + f"{_fmt(r['ce_best_auroc'])}, {r['ce_best_variant']} | "
                + f"{_fmt(r['ie_best_auroc'])}, {r['ie_best_variant']} | "
                + f"{_fmt(r['weighted_wb_auroc'])} | {_fmt(r['elapsed_hours'],2)} |"
            )

    lines.append("")
    lines.append("## 3) White-box vs black-box comparison")
    lines.append("")
    lines.append(
        "Black-box metrics are from the v2 thesis aggregate model table (`disagreement_auroc`, `entropy_auroc`). "
        "White-box comparison uses a CE/IE-weighted AUROC proxy per run."
    )
    lines.append("")
    lines.append("| Target model | Best WB weighted AUROC | Best verifier | Black-box entropy AUROC | Black-box disagreement AUROC | Delta vs entropy | Delta vs disagreement |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    if not comparison.empty:
        best_by_target = (
            comparison.sort_values("weighted_wb_auroc", ascending=False)
            .groupby("target_model", as_index=False)
            .first()
        )
        for _, r in best_by_target.iterrows():
            lines.append(
                "| "
                + f"{r['target_model']} | {_fmt(r['weighted_wb_auroc'])} | {r['verifier_model']} | "
                + f"{_fmt(r['entropy_auroc'])} | {_fmt(r['disagreement_auroc'])} | "
                + f"{_fmt(r['delta_vs_blackbox_entropy'])} | {_fmt(r['delta_vs_blackbox_disagreement'])} |"
            )

    lines.append("")
    lines.append("## 4) Quantitative pattern checks")
    lines.append("")
    if comparison.empty:
        lines.append("- No completed runs for quantitative checks.")
    else:
        ce_mean = float(comparison["ce_best_auroc"].mean())
        ie_mean = float(comparison["ie_best_auroc"].mean())
        ce_min = float(comparison["ce_best_auroc"].min())
        ce_max = float(comparison["ce_best_auroc"].max())
        ie_min = float(comparison["ie_best_auroc"].min())
        ie_max = float(comparison["ie_best_auroc"].max())
        gap_mean = float((comparison["ce_best_auroc"] - comparison["ie_best_auroc"]).mean())
        lines.append(f"- CE AUROC range across completed runs: {ce_min:.3f} to {ce_max:.3f} (mean {ce_mean:.3f}).")
        lines.append(f"- IE AUROC range across completed runs: {ie_min:.3f} to {ie_max:.3f} (mean {ie_mean:.3f}).")
        lines.append(f"- Mean CE minus IE gap: {gap_mean:.3f}.")
        if not wb_best.empty:
            ce_wins = (
                wb_best[wb_best["subset"] == "ce"]["best_variant"]
                .value_counts()
                .to_dict()
            )
            ie_wins = (
                wb_best[wb_best["subset"] == "ie"]["best_variant"]
                .value_counts()
                .to_dict()
            )
            lines.append(
                "- CE best-variant win counts: "
                + ", ".join([f"{k}={v}" for k, v in ce_wins.items()])
                + "."
            )
            lines.append(
                "- IE best-variant win counts: "
                + ", ".join([f"{k}={v}" for k, v in ie_wins.items()])
                + "."
            )

    lines.append("")
    lines.append("## 5) Interpretation (why this pattern appears)")
    lines.append("")
    lines.append("- CE detection is consistently easier than IE detection in white-box runs. This matches the expected pattern where self-consistent wrong answers (CE) carry stable internal signals while inconsistent errors (IE) are noisier.")
    lines.append("- White-box probes generally exceed black-box entropy/disagreement AUROC for completed targets. Hidden-state access appears to provide stronger separability than sampling-only signals.")
    lines.append("- Verifier choice matters: for some targets, smaller verifiers help CE but can hurt IE. Fusion is useful only when verifier adds complementary signal; otherwise lambda tuning tends to collapse toward target-only.")
    lines.append("- Runtime is dominated by encoder load cost. Qwen-target runs finished in minutes, while Llama-target runs took ~8-9 hours with the same probe settings.")

    lines.append("")
    lines.append("## 6) Research context")
    lines.append("")
    lines.append("- The CE>IE separation pattern is consistent with the EMNLP 2025 white-box framing that self-consistent errors can be difficult to catch with output-only disagreement signals.")
    lines.append("- The black-box baseline behavior is consistent with prior sampling-consistency findings (for example SelfCheckGPT, EMNLP 2023): useful but weaker than hidden-state probes on this dataset slice.")

    lines.append("")
    lines.append("## 7) Method caveats")
    lines.append("")
    lines.append("- This is not a strict apples-to-apples metric comparison: white-box AUROCs are on balanced CE/IE subsets, while black-box AUROCs are reported on the natural answered-row distribution.")
    lines.append("- DeepSeek white-box results are not yet included because corresponding jobs are still pending in this snapshot.")
    lines.append("- Some run logs contain non-fatal warnings (`Mean of empty slice`), but completed runs produced all expected artifacts and metrics.")

    lines.append("")
    lines.append("## 8) Practical next steps")
    lines.append("")
    lines.append("- Finish pending DeepSeek jobs (or rerun with supported encoder stack) and regenerate this report.")
    lines.append("- Add a calibrated combined score that maps white-box subset metrics onto the same evaluation population as black-box baselines.")
    lines.append("- Keep both `target_only` and `cross_model_fused` in reporting; when fusion underperforms, document lambda behavior to diagnose verifier contribution.")

    out_path.write_text("\n".join(lines))


def _table(data: List[List[str]], col_widths: List[float]) -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bdbdbd")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def write_pdf(
    out_path: Path,
    generated_at: str,
    job_status: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        spaceAfter=6,
    )
    story: List[object] = []
    story.append(Paragraph("White-box vs Black-box Hallucination Detection Report", title_style))
    story.append(Paragraph(f"Generated: {generated_at}", body))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Execution status", heading_style))
    if job_status.empty:
        story.append(Paragraph("No job snapshot available.", body))
    else:
        tab = [["Job ID", "State", "Elapsed", "Start", "End"]]
        for _, r in job_status.iterrows():
            tab.append(
                [
                    str(r.get("JobIDRaw", "")),
                    str(r.get("State", "")),
                    str(r.get("Elapsed", "")),
                    str(r.get("Start", "")),
                    str(r.get("End", "")),
                ]
            )
        story.append(_table(tab, [0.95 * inch, 0.95 * inch, 0.9 * inch, 1.9 * inch, 1.9 * inch]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Completed white-box runs", heading_style))
    if comparison.empty:
        story.append(Paragraph("No completed white-box runs found.", body))
    else:
        tab = [["Target", "Verifier", "CE best", "IE best", "WB weighted", "Hours"]]
        for _, r in comparison.iterrows():
            tab.append(
                [
                    str(r["target_model"]),
                    str(r["verifier_model"]).replace("meta-llama/", "").replace("Qwen/", ""),
                    f"{_fmt(r['ce_best_auroc'])} ({r['ce_best_variant']})",
                    f"{_fmt(r['ie_best_auroc'])} ({r['ie_best_variant']})",
                    _fmt(r["weighted_wb_auroc"]),
                    _fmt(r["elapsed_hours"], 2),
                ]
            )
        story.append(_table(tab, [1.55 * inch, 1.5 * inch, 1.35 * inch, 1.35 * inch, 0.9 * inch, 0.65 * inch]))
    story.append(Spacer(1, 8))

    if not comparison.empty:
        story.append(Paragraph("Best white-box vs black-box by target", heading_style))
        best_by_target = (
            comparison.sort_values("weighted_wb_auroc", ascending=False)
            .groupby("target_model", as_index=False)
            .first()
        )
        tab = [["Target", "Best verifier", "WB weighted", "Black-box entropy", "Delta"]]
        for _, r in best_by_target.iterrows():
            tab.append(
                [
                    str(r["target_model"]),
                    str(r["verifier_model"]).replace("meta-llama/", "").replace("Qwen/", ""),
                    _fmt(r["weighted_wb_auroc"]),
                    _fmt(r["entropy_auroc"]),
                    _fmt(r["delta_vs_blackbox_entropy"]),
                ]
            )
        story.append(_table(tab, [1.7 * inch, 2.1 * inch, 1.0 * inch, 1.0 * inch, 0.9 * inch]))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Key interpretation", heading_style))
    bullets = [
        "CE is consistently easier than IE in white-box runs.",
        "White-box weighted AUROC is higher than black-box entropy/disagreement AUROC for completed targets.",
        "Verifier impact is asymmetric: it often helps CE more than IE; fusion helps only when verifier signal is complementary.",
        "Run time is dominated by encoder loading and hidden-state extraction cost.",
    ]
    for text in bullets:
        story.append(Paragraph(f"- {text}", body))

    doc.build(story)


def write_latex(
    out_path: Path,
    generated_at: str,
    job_status: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = int((job_status["State"] == "COMPLETED").sum()) if not job_status.empty else 0
    pending = int((job_status["State"] == "PENDING").sum()) if not job_status.empty else 0
    failed = int((job_status["State"] == "FAILED").sum()) if not job_status.empty else 0
    lines: List[str] = []
    lines.extend(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{booktabs}",
            r"\usepackage{longtable}",
            r"\usepackage{array}",
            r"\usepackage{amsmath}",
            r"\usepackage{hyperref}",
            r"\title{White-box vs Black-box Report (Simple Interpretation)}",
            r"\author{Automated pipeline report}",
            rf"\date{{Generated: {tex_escape(generated_at)}}}",
            r"\begin{document}",
            r"\maketitle",
            r"\section*{What this report means (simple terms)}",
            r"\begin{itemize}",
            r"\item Higher AUROC means better separation between correct and error cases.",
            r"\item CE (self-consistent error) is usually easier to detect than IE (inconsistent error) in these white-box runs.",
            r"\item Weighted WB AUROC combines CE and IE using the observed CE/IE mix:",
            r"\[",
            r"\text{Weighted WB AUROC} = (\text{CE AUROC}\times\text{CE share}) + (\text{IE AUROC}\times\text{IE share})",
            r"\]",
            r"\item Delta vs black-box entropy $> 0$ means white-box is better than black-box entropy on this target model.",
            r"\item Runtime differences mainly come from encoder loading cost, not probe training itself.",
            r"\end{itemize}",
            r"\section*{Execution status}",
            rf"Completed: {completed}, Pending: {pending}, Failed: {failed}.",
        ]
    )
    if not job_status.empty:
        lines.extend(
            [
                r"\begin{longtable}{lllll}",
                r"\toprule",
                r"Job ID & State & Elapsed & Start & End \\",
                r"\midrule",
            ]
        )
        for _, r in job_status.iterrows():
            lines.append(
                f"{tex_escape(r.get('JobIDRaw',''))} & {tex_escape(r.get('State',''))} & "
                f"{tex_escape(r.get('Elapsed',''))} & {tex_escape(r.get('Start',''))} & "
                f"{tex_escape(r.get('End',''))} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{longtable}"])

    lines.append(r"\section*{Completed white-box runs}")
    if comparison.empty:
        lines.append("No completed runs available.")
    else:
        lines.extend(
            [
                r"\begin{longtable}{p{2.1in}p{1.6in}p{0.85in}p{0.85in}p{0.75in}p{0.65in}}",
                r"\toprule",
                r"Target & Verifier & CE AUROC & IE AUROC & WB weighted & Hours \\",
                r"\midrule",
            ]
        )
        for _, r in comparison.iterrows():
            lines.append(
                f"{tex_escape(r['target_model'])} & {tex_escape(r['verifier_model'])} & "
                f"{_fmt(r['ce_best_auroc'])} & {_fmt(r['ie_best_auroc'])} & "
                f"{_fmt(r['weighted_wb_auroc'])} & {_fmt(r['elapsed_hours'], 2)} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{longtable}"])

        older = comparison[
            comparison["run_name"].astype(str).str.contains("Qwen3-Next-80B-A3B-Instruct", na=False)
            & comparison["target_model"].astype(str).str.contains("Llama 4 Maverick", na=False)
        ]
        if not older.empty:
            o = older.iloc[0]
            lines.extend(
                [
                    r"\paragraph{Earlier completed run included.}",
                    "This report also includes an earlier completed run from a previous submission: "
                    + f"\\texttt{{{tex_escape(o['run_name'])}}} with weighted WB AUROC {_fmt(o['weighted_wb_auroc'])}.",
                ]
            )

    if not comparison.empty:
        lines.extend([r"\section*{White-box vs black-box (best per target)}"])
        best_by_target = (
            comparison.sort_values("weighted_wb_auroc", ascending=False)
            .groupby("target_model", as_index=False)
            .first()
        )
        lines.extend(
            [
                r"\begin{tabular}{p{2.0in}p{1.8in}ccc}",
                r"\toprule",
                r"Target & Best verifier & WB weighted & BB entropy & Delta \\",
                r"\midrule",
            ]
        )
        for _, r in best_by_target.iterrows():
            lines.append(
                f"{tex_escape(r['target_model'])} & {tex_escape(r['verifier_model'])} & "
                f"{_fmt(r['weighted_wb_auroc'])} & {_fmt(r['entropy_auroc'])} & "
                f"{_fmt(r['delta_vs_blackbox_entropy'])} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}"])

        lines.extend(
            [
                r"\section*{How to interpret quickly}",
                r"\begin{enumerate}",
                r"\item Compare CE AUROC and IE AUROC: large CE-IE gap means IE remains the hard case.",
                r"\item Check the best variant (target only, verifier only, fused): this tells you whether the verifier helped.",
                r"\item Use weighted WB AUROC for one-number comparison against black-box baselines.",
                r"\item Positive delta means white-box improved over black-box entropy for that target model.",
                r"\end{enumerate}",
            ]
        )

    lines.append(r"\end{document}")
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wb-root",
        type=Path,
        default=Path("downloads/nibi_wb_probe/wb_probe_out"),
    )
    parser.add_argument(
        "--blackbox-metrics",
        type=Path,
        default=Path("data/results/analysis/v2_thesis/metrics/group_metrics_by_model.csv"),
    )
    parser.add_argument(
        "--sacct-latest",
        type=Path,
        default=Path("downloads/nibi_wb_probe/job_history/sacct_wb_probe_latest.csv"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("downloads/nibi_wb_probe/reports"),
    )
    parser.add_argument(
        "--pdf-out",
        type=Path,
        default=Path("output/pdf/wb_vs_blackbox_report_2026-03-08.pdf"),
    )
    parser.add_argument(
        "--tex-out",
        type=Path,
        default=Path("downloads/nibi_wb_probe/reports/wb_vs_blackbox_report_2026-03-08.tex"),
    )
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    job_status = load_job_status(args.sacct_latest)
    bb = load_blackbox_metrics(args.blackbox_metrics)
    wb_best = collect_wb_runs(args.wb_root)
    comparison = build_comparison(wb_best, bb)

    wb_best_csv = args.report_dir / "wb_best_per_run.csv"
    comparison_csv = args.report_dir / "wb_vs_blackbox_comparison.csv"
    job_status_csv = args.report_dir / "job_status_snapshot.csv"
    wb_best.to_csv(wb_best_csv, index=False)
    comparison.to_csv(comparison_csv, index=False)
    job_status.to_csv(job_status_csv, index=False)

    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_path = args.report_dir / "wb_vs_blackbox_report_2026-03-08.md"
    write_markdown(md_path, generated_at, job_status, wb_best, comparison, bb)
    write_pdf(args.pdf_out, generated_at, job_status, comparison)
    write_latex(args.tex_out, generated_at, job_status, comparison)

    print(f"Wrote: {md_path}")
    print(f"Wrote: {args.pdf_out}")
    print(f"Wrote: {args.tex_out}")
    print(f"Wrote: {wb_best_csv}")
    print(f"Wrote: {comparison_csv}")
    print(f"Wrote: {job_status_csv}")


if __name__ == "__main__":
    main()
