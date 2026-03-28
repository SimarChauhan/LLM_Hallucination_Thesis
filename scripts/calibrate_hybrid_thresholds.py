#!/usr/bin/env python3
"""Calibrate hybrid NLI thresholds for equivalence and sample correctness."""

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nli_judge import NLISemanticJudge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


EQUIV_LABELS = ("same", "different", "unclear")
CORR_LABELS = ("CORRECT", "INCORRECT", "NOT_ATTEMPTED")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line %d in %s: %s", line_num, path, exc)
    return records


def _macro_f1(y_true: List[str], y_pred: List[str], labels: Tuple[str, ...]) -> float:
    if not y_true:
        return 0.0
    per_label = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        denom = (2 * tp) + fp + fn
        per_label.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(sum(per_label) / len(per_label))


def _accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    return float(sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true))


def _normalize_equiv_label(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token in EQUIV_LABELS:
        return token
    if token in {"equivalent", "same_meaning"}:
        return "same"
    if token in {"conflict", "not_equivalent", "different_meaning"}:
        return "different"
    return "unclear"


def _normalize_corr_label(value: Any) -> str:
    token = str(value or "").strip().upper()
    if token in CORR_LABELS:
        return token
    if token in {"A", "TRUE", "YES", "1"}:
        return "CORRECT"
    if token in {"B", "FALSE", "NO", "0"}:
        return "INCORRECT"
    if token in {"C", "UNCLEAR", "UNKNOWN", "NA"}:
        return "NOT_ATTEMPTED"
    return "NOT_ATTEMPTED"


def _predict_equivalence(pf: float, pr: float, same_hi: float, diff_lo: float) -> str:
    if pf >= same_hi and pr >= same_hi:
        return "same"
    if pf <= diff_lo or pr <= diff_lo:
        return "different"
    return "unclear"


def _predict_correctness(p_max: float, corr_hi: float, corr_lo: float) -> str:
    if p_max >= corr_hi:
        return "CORRECT"
    if p_max <= corr_lo:
        return "INCORRECT"
    return "NOT_ATTEMPTED"


def _candidate_grid() -> List[Tuple[float, float]]:
    hi_candidates = [round(x, 2) for x in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]]
    lo_candidates = [round(x, 2) for x in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]]
    grid: List[Tuple[float, float]] = []
    for hi in hi_candidates:
        for lo in lo_candidates:
            if lo < hi:
                grid.append((hi, lo))
    return grid


def _calibrate_equivalence(
    rows: List[Dict[str, Any]],
    nli_judge: NLISemanticJudge,
    default_same_hi: float,
    default_diff_lo: float,
) -> Dict[str, Any]:
    prepared = []
    for row in rows:
        question = str(row.get("question", ""))
        answer_a = str(row.get("answer_a", row.get("greedy_answer", "")))
        answer_b = str(row.get("answer_b", row.get("sample_answer", "")))
        human_label = _normalize_equiv_label(row.get("human_label", row.get("label")))
        if not question or not answer_a or not answer_b:
            continue
        context_a = f"Question: {question} Answer: {answer_a}"
        context_b = f"Question: {question} Answer: {answer_b}"
        try:
            pf = float(nli_judge._get_entailment_prob(context_a, context_b))
            pr = float(nli_judge._get_entailment_prob(context_b, context_a))
        except Exception as exc:
            logger.warning("Skipping equivalence row due to NLI failure: %s", exc)
            continue
        prepared.append({"pf": pf, "pr": pr, "label": human_label})

    if not prepared:
        return {
            "eq_same_hi": default_same_hi,
            "eq_diff_lo": default_diff_lo,
            "objective_macro_f1": None,
            "accuracy": None,
            "n_rows": 0,
        }

    best = {
        "eq_same_hi": default_same_hi,
        "eq_diff_lo": default_diff_lo,
        "objective_macro_f1": -1.0,
        "accuracy": -1.0,
    }
    true_labels = [item["label"] for item in prepared]
    for same_hi, diff_lo in _candidate_grid():
        preds = [_predict_equivalence(item["pf"], item["pr"], same_hi, diff_lo) for item in prepared]
        macro_f1 = _macro_f1(true_labels, preds, EQUIV_LABELS)
        acc = _accuracy(true_labels, preds)
        if (macro_f1 > best["objective_macro_f1"]) or (
            math.isclose(macro_f1, best["objective_macro_f1"]) and acc > best["accuracy"]
        ):
            best = {
                "eq_same_hi": same_hi,
                "eq_diff_lo": diff_lo,
                "objective_macro_f1": macro_f1,
                "accuracy": acc,
            }

    best["n_rows"] = len(prepared)
    return best


def _calibrate_correctness(
    rows: List[Dict[str, Any]],
    nli_judge: NLISemanticJudge,
    default_corr_hi: float,
    default_corr_lo: float,
) -> Dict[str, Any]:
    prepared = []
    for row in rows:
        question = str(row.get("question", ""))
        answer = str(row.get("sample_answer", row.get("prediction", row.get("answer", ""))))
        human_grade = _normalize_corr_label(row.get("human_grade", row.get("grade")))
        gold = row.get("ground_truths", row.get("ground_truth", []))
        if isinstance(gold, str):
            gold_targets = [g.strip() for g in gold.split("||") if g.strip()]
        else:
            gold_targets = [str(g).strip() for g in (gold or []) if str(g).strip()]
        if not question or not answer or not gold_targets:
            continue
        context_answer = f"Question: {question} Answer: {answer}"
        best_prob = -1.0
        for target in gold_targets:
            context_gold = f"Question: {question} Answer: {target}"
            try:
                prob = float(nli_judge._get_entailment_prob(context_answer, context_gold))
            except Exception as exc:
                logger.warning("Skipping one gold target due to NLI failure: %s", exc)
                continue
            if prob > best_prob:
                best_prob = prob
        if best_prob < 0.0:
            continue
        prepared.append({"p_max": best_prob, "label": human_grade})

    if not prepared:
        return {
            "corr_hi": default_corr_hi,
            "corr_lo": default_corr_lo,
            "objective_macro_f1": None,
            "accuracy": None,
            "n_rows": 0,
        }

    best = {
        "corr_hi": default_corr_hi,
        "corr_lo": default_corr_lo,
        "objective_macro_f1": -1.0,
        "accuracy": -1.0,
    }
    true_labels = [item["label"] for item in prepared]
    for corr_hi, corr_lo in _candidate_grid():
        preds = [_predict_correctness(item["p_max"], corr_hi, corr_lo) for item in prepared]
        macro_f1 = _macro_f1(true_labels, preds, CORR_LABELS)
        acc = _accuracy(true_labels, preds)
        if (macro_f1 > best["objective_macro_f1"]) or (
            math.isclose(macro_f1, best["objective_macro_f1"]) and acc > best["accuracy"]
        ):
            best = {
                "corr_hi": corr_hi,
                "corr_lo": corr_lo,
                "objective_macro_f1": macro_f1,
                "accuracy": acc,
            }

    best["n_rows"] = len(prepared)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate hybrid NLI thresholds for re-evaluation.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--equivalence-dev",
        type=str,
        default=None,
        help="JSONL dev file for equivalence labels (question, answer_a, answer_b, human_label).",
    )
    parser.add_argument(
        "--correctness-dev",
        type=str,
        default=None,
        help="JSONL dev file for correctness labels (question, sample_answer, ground_truth(s), human_grade).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/calibration/hybrid_thresholds.json",
        help="Output calibration JSON path.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        fallback = Path(__file__).parent.parent / args.config
        if fallback.exists():
            config_path = fallback
        else:
            raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    judge_cfg = config.get("judge", {}) or {}
    nli_cfg = judge_cfg.get("nli", {}) or {}
    hybrid_cfg = config.get("hybrid", {}) or {}
    hybrid_threshold_cfg = hybrid_cfg.get("thresholds", {}) or {}

    defaults = {
        "eq_same_hi": float(hybrid_threshold_cfg.get("eq_same_hi", 0.70)),
        "eq_diff_lo": float(hybrid_threshold_cfg.get("eq_diff_lo", 0.30)),
        "corr_hi": float(hybrid_threshold_cfg.get("corr_hi", 0.70)),
        "corr_lo": float(hybrid_threshold_cfg.get("corr_lo", 0.30)),
    }

    nli_model_name = nli_cfg.get("model", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    nli_device = nli_cfg.get("device", None)
    logger.info("Loading NLI judge %s", nli_model_name)
    nli_judge = NLISemanticJudge(
        model_name=nli_model_name,
        device=nli_device,
        entailment_threshold=float(nli_cfg.get("entailment_threshold", 0.5)),
        different_threshold=float(nli_cfg.get("different_threshold", 0.3)),
    )

    equivalence_rows = _load_jsonl(args.equivalence_dev) if args.equivalence_dev else []
    correctness_rows = _load_jsonl(args.correctness_dev) if args.correctness_dev else []

    eq_result = _calibrate_equivalence(
        rows=equivalence_rows,
        nli_judge=nli_judge,
        default_same_hi=defaults["eq_same_hi"],
        default_diff_lo=defaults["eq_diff_lo"],
    )
    corr_result = _calibrate_correctness(
        rows=correctness_rows,
        nli_judge=nli_judge,
        default_corr_hi=defaults["corr_hi"],
        default_corr_lo=defaults["corr_lo"],
    )

    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    calibration_id = f"hybrid-{now}"
    payload = {
        "calibration_id": calibration_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "nli_model": nli_model_name,
        "eq_same_hi": eq_result["eq_same_hi"],
        "eq_diff_lo": eq_result["eq_diff_lo"],
        "corr_hi": corr_result["corr_hi"],
        "corr_lo": corr_result["corr_lo"],
        "equivalence_dev": {
            "n_rows": eq_result["n_rows"],
            "macro_f1": eq_result["objective_macro_f1"],
            "accuracy": eq_result["accuracy"],
            "path": args.equivalence_dev,
        },
        "correctness_dev": {
            "n_rows": corr_result["n_rows"],
            "macro_f1": corr_result["objective_macro_f1"],
            "accuracy": corr_result["accuracy"],
            "path": args.correctness_dev,
        },
        "defaults": defaults,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    logger.info("Saved hybrid calibration to %s", output_path)
    logger.info(
        "Frozen thresholds: eq_same_hi=%.2f eq_diff_lo=%.2f corr_hi=%.2f corr_lo=%.2f",
        payload["eq_same_hi"],
        payload["eq_diff_lo"],
        payload["corr_hi"],
        payload["corr_lo"],
    )


if __name__ == "__main__":
    main()
