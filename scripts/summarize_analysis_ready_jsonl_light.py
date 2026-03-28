#!/usr/bin/env python3
"""Lightweight summarizer for *.final.analysis_ready.jsonl outputs.

Uses only the Python standard library (no pandas/matplotlib), so it can run in
minimal environments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _parse_iso8601(s: Any) -> Optional[datetime]:
    if not isinstance(s, str) or not s:
        return None
    # Accept "...Z" timestamps.
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _quantile(xs: List[float], q: float) -> float:
    """Return quantile using linear interpolation (q in [0,1])."""
    if not xs:
        return float("nan")
    if q <= 0:
        return min(xs)
    if q >= 1:
        return max(xs)
    ys = sorted(xs)
    pos = (len(ys) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    frac = pos - lo
    return ys[lo] * (1 - frac) + ys[hi] * frac


def _pct(x: float) -> str:
    if x != x:  # NaN
        return "N/A"
    return f"{100.0 * x:.1f}%"


@dataclass
class ModelSummary:
    model: str
    n: int
    unique_questions: int
    accuracy: float
    correctness_grade: Dict[str, int]
    error_label_1_0: Dict[str, int]
    equivalence_ratio_mean: float
    equivalence_ratio_p10: float
    equivalence_ratio_p50: float
    equivalence_ratio_p90: float
    judge_status_counts: Dict[str, int]
    decision_source_counts: Dict[str, int]
    dataset_name_split_counts: Dict[str, int]

    @property
    def correct_reliably_share(self) -> float:
        correct = self.correctness_grade.get("CORRECT", 0)
        reliably = self.error_label_1_0.get("reliably_correct", 0)
        return reliably / correct if correct else float("nan")

    @property
    def incorrect_self_consistent_share(self) -> float:
        incorrect = self.correctness_grade.get("INCORRECT", 0)
        self_consistent = self.error_label_1_0.get("self_consistent_error", 0)
        return self_consistent / incorrect if incorrect else float("nan")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def summarize(path: Path) -> Dict[str, Any]:
    totals = Counter()
    grade_global = Counter()
    label_global = {thr: Counter() for thr in ("1.0", "0.9", "0.8", "0.7")}
    judge_status_global = Counter()
    decision_source_global = Counter()
    dataset_name_split_global = Counter()

    timestamps: List[datetime] = []
    question_ids: set[str] = set()

    model_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    model_ids_by_model: Dict[str, Counter] = defaultdict(Counter)

    for r in _iter_jsonl(path):
        totals["rows"] += 1

        model = r.get("model") or r.get("model_id") or "UNKNOWN_MODEL"
        model_rows[model].append(r)
        model_ids_by_model[model][str(r.get("model_id"))] += 1

        qid = r.get("question_id")
        if isinstance(qid, str):
            question_ids.add(qid)

        ts = _parse_iso8601(r.get("timestamp"))
        if ts is not None:
            timestamps.append(ts)

        g = r.get("correctness_grade")
        if isinstance(g, str):
            grade_global[g] += 1

        for thr in ("1.0", "0.9", "0.8", "0.7"):
            k = f"error_label_{thr}"
            v = r.get(k)
            if isinstance(v, str):
                label_global[thr][v] += 1

        statuses = r.get("correctness_judge_statuses")
        if isinstance(statuses, list):
            for s in statuses:
                if isinstance(s, str):
                    judge_status_global[s] += 1

        ds = r.get("correctness_decision_source")
        if isinstance(ds, str):
            decision_source_global[ds] += 1

        dn = r.get("dataset_name")
        sp = r.get("dataset_split")
        dataset_name_split_global[f"{dn}::{sp}"] += 1

        if r.get("contamination_flag") is True:
            totals["contamination_flag_true"] += 1
        if r.get("is_incomplete") is True:
            totals["is_incomplete_true"] += 1
        if r.get("correctness_unclear") is True:
            totals["correctness_unclear_true"] += 1
        if r.get("stochastic_actual_n") is not None:
            totals[f"stochastic_actual_n={r.get('stochastic_actual_n')}"] += 1
        if r.get("stochastic_target_n") is not None:
            totals[f"stochastic_target_n={r.get('stochastic_target_n')}"] += 1

    # Per-model summaries (grouped by `model`, since it is typically human-readable)
    model_summaries: List[ModelSummary] = []
    for model, rows in sorted(model_rows.items()):
        grades = Counter()
        labels_1_0 = Counter()
        ratios: List[float] = []
        statuses = Counter()
        decision_sources = Counter()
        dataset_name_split = Counter()
        qids: set[str] = set()

        correct = 0
        for r in rows:
            g = r.get("correctness_grade")
            if isinstance(g, str):
                grades[g] += 1
                if g == "CORRECT":
                    correct += 1

            l = r.get("error_label_1.0")
            if isinstance(l, str):
                labels_1_0[l] += 1

            eq = r.get("equivalence_ratio")
            if isinstance(eq, (int, float)):
                ratios.append(float(eq))

            st = r.get("correctness_judge_statuses")
            if isinstance(st, list):
                for s in st:
                    if isinstance(s, str):
                        statuses[s] += 1

            ds = r.get("correctness_decision_source")
            if isinstance(ds, str):
                decision_sources[ds] += 1

            dn = r.get("dataset_name")
            sp = r.get("dataset_split")
            dataset_name_split[f"{dn}::{sp}"] += 1

            qid = r.get("question_id")
            if isinstance(qid, str):
                qids.add(qid)

        n = len(rows)
        model_summaries.append(
            ModelSummary(
                model=model,
                n=n,
                unique_questions=len(qids),
                accuracy=(correct / n) if n else float("nan"),
                correctness_grade=dict(grades),
                error_label_1_0=dict(labels_1_0),
                equivalence_ratio_mean=_mean(ratios),
                equivalence_ratio_p10=_quantile(ratios, 0.10),
                equivalence_ratio_p50=_quantile(ratios, 0.50),
                equivalence_ratio_p90=_quantile(ratios, 0.90),
                judge_status_counts=dict(statuses),
                decision_source_counts=dict(decision_sources),
                dataset_name_split_counts=dict(dataset_name_split),
            )
        )

    timestamps_sorted = sorted(timestamps)
    ts_min = timestamps_sorted[0].isoformat() if timestamps_sorted else None
    ts_max = timestamps_sorted[-1].isoformat() if timestamps_sorted else None

    out = {
        "input_path": str(path),
        "file_size_bytes": path.stat().st_size if path.exists() else None,
        "rows": totals["rows"],
        "unique_question_ids": len(question_ids),
        "timestamp_min": ts_min,
        "timestamp_max": ts_max,
        "global": {
            "correctness_grade": dict(grade_global),
            "error_label": {thr: dict(c) for thr, c in label_global.items()},
            "judge_status_counts": dict(judge_status_global),
            "decision_source_counts": dict(decision_source_global),
            "dataset_name_split_counts": dict(dataset_name_split_global),
            "contamination_flag_true": totals["contamination_flag_true"],
            "is_incomplete_true": totals["is_incomplete_true"],
            "correctness_unclear_true": totals["correctness_unclear_true"],
            "stochastic_actual_n_counts": {
                k.split("=", 1)[1]: v for k, v in totals.items() if k.startswith("stochastic_actual_n=")
            },
            "stochastic_target_n_counts": {
                k.split("=", 1)[1]: v for k, v in totals.items() if k.startswith("stochastic_target_n=")
            },
        },
        "per_model": [
            {
                **asdict(ms),
                "correct_reliably_share": ms.correct_reliably_share,
                "incorrect_self_consistent_share": ms.incorrect_self_consistent_share,
                "model_id_variants": dict(model_ids_by_model.get(ms.model, Counter())),
            }
            for ms in model_summaries
        ],
    }
    return out


def to_markdown(summary: Dict[str, Any]) -> str:
    rows = summary["rows"]
    uq = summary["unique_question_ids"]
    ts_min = summary["timestamp_min"]
    ts_max = summary["timestamp_max"]
    fsz = summary.get("file_size_bytes")
    fsz_mb = (fsz / (1024 * 1024)) if isinstance(fsz, (int, float)) else float("nan")

    grade = summary["global"]["correctness_grade"]
    labels_1 = summary["global"]["error_label"]["1.0"]

    md: List[str] = []
    md.append(f"# Summary: {Path(summary['input_path']).name}")
    md.append("")
    md.append("## Dataset")
    md.append(f"- Rows: **{rows}**")
    md.append(f"- File size: **{fsz_mb:.1f} MB**" if fsz_mb == fsz_mb else "- File size: **N/A**")
    md.append(f"- Unique `question_id`: **{uq}**")
    md.append(f"- Timestamp range: **{ts_min}** → **{ts_max}**")
    md.append("")
    md.append("## Global correctness")
    md.append(f"- `correctness_grade`: {grade}")
    md.append(f"- `error_label_1.0`: {labels_1}")
    md.append(f"- Judge vote statuses (all models, all rows): {summary['global']['judge_status_counts']}")
    md.append(f"- Decision source: {summary['global']['decision_source_counts']}")
    md.append("")
    md.append("## Per-model (grouped by `model`)")
    md.append("")
    md.append("| Model | N | Acc | EqRatio mean | EqRatio p50 | Reliably-correct share (of correct) | Self-consistent-error share (of incorrect) | NOT_ATTEMPTED |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for ms in summary["per_model"]:
        model = ms["model"]
        n = ms["n"]
        acc = ms["accuracy"]
        eq_mean = ms["equivalence_ratio_mean"]
        eq_p50 = ms["equivalence_ratio_p50"]
        corr = ms["correctness_grade"].get("CORRECT", 0)
        inc = ms["correctness_grade"].get("INCORRECT", 0)
        na = ms["correctness_grade"].get("NOT_ATTEMPTED", 0)
        reliable_share = ms["correct_reliably_share"]
        self_cons_share = ms["incorrect_self_consistent_share"]
        md.append(
            f"| {model} | {n} | {_pct(acc)} | {eq_mean:.3f} | {eq_p50:.3f} | {_pct(reliable_share)} | {_pct(self_cons_share)} | {na} |"
        )
        _ = (corr, inc)  # keep locals for future extensions

    md.append("")
    md.append("## Metadata consistency notes")
    md.append("- `model_id` variants per `model` (helps catch accidental splits during analysis):")
    for ms in summary["per_model"]:
        variants = ms.get("model_id_variants", {})
        if len(variants) > 1:
            md.append(f"  - {ms['model']}: {variants}")
    md.append(f"- `dataset_name::dataset_split` distribution: {summary['global']['dataset_name_split_counts']}")
    md.append("")
    return "\n".join(md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="Path to *.analysis_ready.jsonl")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write machine-readable summary JSON to this path (optional)",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Write human-readable summary Markdown to this path (optional)",
    )
    args = ap.parse_args()

    summary = summarize(args.input)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(to_markdown(summary), encoding="utf-8")

    if args.out_json is None and args.out_md is None:
        # Default to stdout Markdown when no outputs are specified.
        print(to_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
