"""
Human annotation sample collection and management (Tier 2 - C3).

Provides tools to:
1. Select a stratified sample of evaluated records for human review
2. Export the sample in CSV/JSONL format suitable for annotation
3. Load completed human annotations
4. Compute calibration metrics (accuracy, Cohen's kappa) against system judgments

This enables NLI threshold calibration and overall evaluation trustworthiness
measurement as recommended by LLM-as-judge research.
"""

import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def select_annotation_sample(
    records: List[Dict[str, Any]],
    n: int = 50,
    stratify_by: str = "correctness_match_type",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Select a stratified sample of records for human annotation.

    Ensures representation across match types and models so calibration
    is not biased toward easy string-match cases.

    Args:
        records: All evaluated records (list of dicts).
        n: Total number of records to sample.
        stratify_by: Field to stratify on (default: correctness_match_type).
        seed: Random seed for reproducibility.

    Returns:
        List of sampled records (shallow copies).
    """
    rng = random.Random(seed)

    # Group by stratification key
    groups: Dict[Optional[str], List[Dict]] = defaultdict(list)
    for rec in records:
        key = rec.get(stratify_by) or "none"
        groups[key].append(rec)

    # Proportional allocation (at least 1 per group that exists)
    group_keys = sorted(groups.keys())
    total_available = sum(len(g) for g in groups.values())
    if total_available == 0:
        return []

    sampled: List[Dict] = []
    remaining = n

    for key in group_keys:
        pool = groups[key]
        # Proportional share, at least 1
        share = max(1, round(n * len(pool) / total_available))
        share = min(share, remaining, len(pool))
        sampled.extend(rng.sample(pool, share))
        remaining -= share
        if remaining <= 0:
            break

    # If we still have remaining budget, fill from largest groups
    if remaining > 0:
        all_remaining = [r for r in records if r not in sampled]
        rng.shuffle(all_remaining)
        sampled.extend(all_remaining[:remaining])

    rng.shuffle(sampled)
    logger.info(
        f"Selected {len(sampled)} records for annotation "
        f"(stratified by {stratify_by}, {len(groups)} groups)"
    )
    return sampled


def export_annotation_sheet(
    sample: List[Dict[str, Any]],
    output_path: str,
    fmt: str = "csv",
) -> str:
    """
    Export annotation sample in a format suitable for human review.

    Each record includes the question, ground truth, greedy answer,
    system judgment, and blank columns for human annotation.

    Args:
        sample: List of record dicts to export.
        output_path: File path to write to.
        fmt: "csv" or "jsonl".

    Returns:
        The output_path written.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        fieldnames = [
            "question_id",
            "question",
            "ground_truth",
            "greedy_answer",
            "system_correct",
            "system_match_type",
            "system_grade",
            "equivalence_ratio",
            "model",
            # Blank columns for annotator
            "human_correct",         # TRUE / FALSE / UNCLEAR
            "human_notes",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in sample:
                gt = rec.get("ground_truth", [])
                if isinstance(gt, list):
                    gt = "; ".join(gt)
                writer.writerow({
                    "question_id": rec.get("question_id", ""),
                    "question": rec.get("question", ""),
                    "ground_truth": gt,
                    "greedy_answer": rec.get("greedy_answer", ""),
                    "system_correct": rec.get("greedy_correct"),
                    "system_match_type": rec.get("correctness_match_type", ""),
                    "system_grade": rec.get("correctness_grade", ""),
                    "equivalence_ratio": rec.get("equivalence_ratio", ""),
                    "model": rec.get("model", ""),
                    "human_correct": "",
                    "human_notes": "",
                })
    else:  # jsonl
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in sample:
                row = {
                    "question_id": rec.get("question_id", ""),
                    "question": rec.get("question", ""),
                    "ground_truth": rec.get("ground_truth", []),
                    "greedy_answer": rec.get("greedy_answer", ""),
                    "system_correct": rec.get("greedy_correct"),
                    "system_match_type": rec.get("correctness_match_type", ""),
                    "system_grade": rec.get("correctness_grade", ""),
                    "equivalence_ratio": rec.get("equivalence_ratio"),
                    "model": rec.get("model", ""),
                    "human_correct": None,
                    "human_notes": "",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(f"Exported {len(sample)} annotation records to {output_path}")
    return output_path


def load_human_annotations(filepath: str) -> List[Dict[str, Any]]:
    """
    Load completed human annotations from CSV or JSONL.

    Expects a ``human_correct`` column/field with values:
    TRUE / FALSE / UNCLEAR (case-insensitive), or boolean.

    Returns:
        List of annotation dicts, each with at least:
        - question_id, human_correct (bool or None), human_notes
    """
    path = Path(filepath)
    annotations: List[Dict[str, Any]] = []

    def _parse_bool(val: Any) -> Optional[bool]:
        if isinstance(val, bool):
            return val
        if val is None or str(val).strip().upper() in ("", "UNCLEAR", "NONE"):
            return None
        return str(val).strip().upper() in ("TRUE", "YES", "1", "CORRECT")

    if path.suffix == ".csv":
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                annotations.append({
                    "question_id": row.get("question_id", ""),
                    "human_correct": _parse_bool(row.get("human_correct")),
                    "human_notes": row.get("human_notes", ""),
                    "system_correct": _parse_bool(row.get("system_correct")),
                    "system_match_type": row.get("system_match_type", ""),
                })
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                annotations.append({
                    "question_id": row.get("question_id", ""),
                    "human_correct": _parse_bool(row.get("human_correct")),
                    "human_notes": row.get("human_notes", ""),
                    "system_correct": _parse_bool(row.get("system_correct")),
                    "system_match_type": row.get("system_match_type", ""),
                })

    logger.info(f"Loaded {len(annotations)} human annotations from {filepath}")
    return annotations


def compute_calibration_metrics(
    annotations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compare system vs human judgments on the annotation sample.

    Args:
        annotations: List of annotation dicts with ``system_correct`` and
            ``human_correct`` (both bool or None).

    Returns:
        Dict with:
        - accuracy: fraction of agreement
        - cohens_kappa: chance-corrected agreement
        - n_total, n_valid (where both have judgments)
        - confusion_matrix: {TP, FP, TN, FN}
        - per_match_type: accuracy breakdown by system_match_type
    """
    # Filter to records where both system and human have judgments
    valid = [
        a for a in annotations
        if a.get("human_correct") is not None and a.get("system_correct") is not None
    ]

    if not valid:
        return {
            "accuracy": 0.0,
            "cohens_kappa": 0.0,
            "n_total": len(annotations),
            "n_valid": 0,
            "confusion_matrix": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
            "per_match_type": {},
        }

    # Confusion matrix
    tp = sum(1 for a in valid if a["system_correct"] and a["human_correct"])
    fp = sum(1 for a in valid if a["system_correct"] and not a["human_correct"])
    tn = sum(1 for a in valid if not a["system_correct"] and not a["human_correct"])
    fn = sum(1 for a in valid if not a["system_correct"] and a["human_correct"])

    n = len(valid)
    accuracy = (tp + tn) / n if n > 0 else 0.0

    # Cohen's kappa
    p_yes_sys = (tp + fp) / n
    p_yes_hum = (tp + fn) / n
    p_e = p_yes_sys * p_yes_hum + (1 - p_yes_sys) * (1 - p_yes_hum)
    kappa = (accuracy - p_e) / (1 - p_e) if (1 - p_e) > 0 else 1.0

    # Per match type
    type_groups: Dict[str, List[Dict]] = defaultdict(list)
    for a in valid:
        mt = a.get("system_match_type") or "none"
        type_groups[mt].append(a)

    per_type = {}
    for mt, group in sorted(type_groups.items()):
        correct = sum(1 for a in group if a["system_correct"] == a["human_correct"])
        per_type[mt] = {
            "accuracy": round(correct / len(group), 4) if group else 0.0,
            "n": len(group),
        }

    return {
        "accuracy": round(accuracy, 4),
        "cohens_kappa": round(kappa, 4),
        "n_total": len(annotations),
        "n_valid": n,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "per_match_type": per_type,
    }
