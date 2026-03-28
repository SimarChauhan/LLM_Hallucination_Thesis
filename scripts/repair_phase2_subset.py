#!/usr/bin/env python3
"""
Cheap Phase-2 recovery workflow:
1) Offline repair parse-failed judge slots (no API calls).
2) Emit only unresolved hard cases for targeted paid re-evaluation.
3) Merge targeted rerun rows back into the full evaluated file.
4) Rebuild retry_queue.jsonl from merged results.
"""

import argparse
import json
import os
import sys
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Allow `from src...` imports when run from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional runtime dependency
    load_dotenv = None

from src.correctness import (
    _grade_letter_to_label,
    _parse_cot_grade,
    check_correctness_llm,
    check_correctness_llm_adjudicator,
)
from src.labeling import classify_at_multiple_thresholds, compute_equivalence_stats
from src.schemas import EquivalenceStats


VALID_GRADES = {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}
LOGGER = logging.getLogger(__name__)


@dataclass
class PrepareStats:
    total_rows: int = 0
    repaired_judge_slots: int = 0
    rows_with_repair: int = 0
    resolved_offline: int = 0
    unresolved_hard_cases: int = 0


@dataclass
class MissingJudgeRepairStats:
    total_rows: int = 0
    rows_with_missing_slots: int = 0
    missing_slots_targeted: int = 0
    judge_calls: int = 0
    repaired_slots_ok: int = 0
    repaired_slots_parse_failed: int = 0
    repaired_slots_api_failed: int = 0
    rows_resolved_majority: int = 0
    adjudicator_calls: int = 0
    rows_resolved_adjudicator: int = 0
    rows_unresolved: int = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def pair_key(rec: Dict[str, Any]) -> Tuple[str, str]:
    return (str(rec.get("question_id", "")), str(rec.get("model", "")))


def question_sort_key(question_id: str) -> Tuple[int, str]:
    digits = ""
    for ch in reversed(question_id):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    if digits:
        return (int(digits), question_id)
    return (10**9, question_id)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
    return rows


def write_jsonl_atomic(rows: Sequence[Dict[str, Any]], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, target)


def _ensure_len(values: Optional[List[Any]], n: int) -> List[Any]:
    out = list(values or [])
    while len(out) < n:
        out.append(None)
    return out


def _majority_grade(judge_grades: Iterable[Optional[str]]) -> Optional[str]:
    valid = [g for g in judge_grades if g in VALID_GRADES]
    if len(valid) < 2:
        return None
    counts = Counter(valid)
    top_grade, top_count = counts.most_common(1)[0]
    if top_count <= len(valid) / 2:
        return None
    return top_grade


def _parse_judge_specs(judges: Sequence[str]) -> List[Dict[str, str]]:
    parsed: List[Dict[str, str]] = []
    for spec in judges:
        if ":" not in spec:
            raise ValueError(f"Invalid judge spec '{spec}'. Expected format provider:model")
        provider, model = spec.split(":", 1)
        provider = provider.strip().lower()
        model = model.strip()
        if not provider or not model:
            raise ValueError(f"Invalid judge spec '{spec}'. Provider and model must be non-empty.")
        parsed.append({"provider": provider, "model": model})
    if len(parsed) != 3:
        raise ValueError(f"Exactly 3 judges required; got {len(parsed)}.")
    return parsed


def _as_text_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if item is None:
            out.append("")
        else:
            out.append(str(item))
    return out


def _get_equiv_stats(rec: Dict[str, Any]) -> EquivalenceStats:
    raw_stats = rec.get("equivalence_stats")
    if isinstance(raw_stats, dict):
        try:
            return EquivalenceStats.from_dict(raw_stats)
        except Exception:
            pass

    raw_judgments = rec.get("equivalence_results")
    if isinstance(raw_judgments, list):
        judgments = [j for j in raw_judgments if j in {"same", "different", "unclear"}]
        return compute_equivalence_stats(judgments)

    return EquivalenceStats(num_same=0, num_different=0, num_unclear=0, total=0)


def _recompute_labels_and_escalation(
    rec: Dict[str, Any], unclear_treatment: str
) -> None:
    equiv_stats = _get_equiv_stats(rec)
    grade = rec.get("correctness_grade")
    is_correct = bool(rec.get("greedy_correct"))
    labels = classify_at_multiple_thresholds(
        is_correct=is_correct,
        equivalence_stats=equiv_stats,
        thresholds=[1.0, 0.9, 0.8, 0.7],
        unclear_treatment=unclear_treatment,  # "exclude" or "count_as_different"
        grade=grade,
    )
    rec["error_label_1.0"] = labels[1.0]
    rec["error_label_0.9"] = labels[0.9]
    rec["error_label_0.8"] = labels[0.8]
    rec["error_label_0.7"] = labels[0.7]

    repeat = rec.get("judge_repeat_consistency")
    repeat_inconsistent = False
    if repeat is not None:
        try:
            repeat_inconsistent = float(repeat) < 1.0
        except Exception:
            repeat_inconsistent = False

    escalated = bool(rec.get("correctness_unclear"))
    escalated = escalated or repeat_inconsistent
    escalated = escalated or bool(rec.get("is_incomplete"))
    rec["escalated_to_human"] = escalated


def prepare(
    evaluated_in: str,
    raw_in: str,
    repaired_out: str,
    hardcases_eval_out: str,
    hardcases_raw_out: str,
    pairs_out: str,
    unclear_treatment: str,
) -> PrepareStats:
    stats = PrepareStats()
    rows = load_jsonl(evaluated_in)
    raw_rows = load_jsonl(raw_in)
    raw_index = {pair_key(r): r for r in raw_rows}

    repaired_rows: List[Dict[str, Any]] = []
    hard_pairs: List[Tuple[str, str]] = []

    for rec in rows:
        stats.total_rows += 1
        out = dict(rec)

        grades = list(out.get("correctness_judge_grades") or [])
        statuses = list(out.get("correctness_judge_statuses") or [])
        reasoning = list(out.get("correctness_judge_reasoning") or [])
        n = max(len(grades), len(statuses), len(reasoning), 3)
        grades = _ensure_len(grades, n)
        statuses = _ensure_len(statuses, n)
        reasoning = _ensure_len(reasoning, n)

        row_repaired = False
        for idx in range(n):
            if grades[idx] in VALID_GRADES:
                continue
            text = str(reasoning[idx] or "").strip()
            if not text:
                continue
            letter, _ = _parse_cot_grade(text)
            if not letter:
                continue
            grades[idx] = _grade_letter_to_label(letter)
            statuses[idx] = "OK"
            stats.repaired_judge_slots += 1
            row_repaired = True

        if row_repaired:
            stats.rows_with_repair += 1
            out["correctness_judge_grades"] = grades
            out["correctness_judge_statuses"] = statuses

        majority = _majority_grade(grades)
        if majority is None:
            stats.unresolved_hard_cases += 1
            hard_pairs.append(pair_key(out))
            # Preserve existing correctness fields; this row will be re-evaluated.
            _recompute_labels_and_escalation(out, unclear_treatment)
            repaired_rows.append(out)
            continue

        stats.resolved_offline += 1
        out["correctness_grade"] = majority
        out["correctness_unclear"] = (majority == "NOT_ATTEMPTED")
        out["greedy_correct"] = (majority == "CORRECT")
        out["correctness_match_type"] = "llm_judge_ensemble"
        out["correctness_decision_source"] = "MAJORITY"
        out["correctness_adjudicator_grade"] = None
        out["correctness_adjudicator_status"] = None
        out["correctness_adjudicator_reasoning"] = None
        _recompute_labels_and_escalation(out, unclear_treatment)
        repaired_rows.append(out)

    hard_pair_set = set(hard_pairs)
    hard_eval_rows = [r for r in repaired_rows if pair_key(r) in hard_pair_set]

    hard_raw_rows: List[Dict[str, Any]] = []
    missing_pairs: List[Tuple[str, str]] = []
    for p in hard_pairs:
        raw_row = raw_index.get(p)
        if raw_row is None:
            missing_pairs.append(p)
            continue
        hard_raw_rows.append(raw_row)
    if missing_pairs:
        preview = ", ".join([f"{qid}|{m}" for qid, m in missing_pairs[:5]])
        raise ValueError(
            f"{len(missing_pairs)} hard-case pairs were not found in raw input. "
            f"Examples: {preview}"
        )

    repaired_rows.sort(
        key=lambda r: (question_sort_key(str(r.get("question_id", ""))), str(r.get("model", "")))
    )
    hard_eval_rows.sort(
        key=lambda r: (question_sort_key(str(r.get("question_id", ""))), str(r.get("model", "")))
    )
    hard_raw_rows.sort(
        key=lambda r: (question_sort_key(str(r.get("question_id", ""))), str(r.get("model", "")))
    )

    pair_rows = [{"question_id": qid, "model": model} for qid, model in hard_pairs]

    write_jsonl_atomic(repaired_rows, repaired_out)
    write_jsonl_atomic(hard_eval_rows, hardcases_eval_out)
    write_jsonl_atomic(hard_raw_rows, hardcases_raw_out)
    write_jsonl_atomic(pair_rows, pairs_out)
    return stats


def merge(
    repaired_in: str,
    rerun_in: str,
    final_out: str,
    retry_queue_out: str,
    unclear_treatment: str,
) -> Dict[str, int]:
    repaired_rows = load_jsonl(repaired_in)
    rerun_rows = load_jsonl(rerun_in)

    rerun_index = {pair_key(r): r for r in rerun_rows}
    rerun_keys = set(rerun_index.keys())

    merged_rows: List[Dict[str, Any]] = []
    replaced = 0
    for rec in repaired_rows:
        key = pair_key(rec)
        if key in rerun_keys:
            merged_rows.append(rerun_index[key])
            replaced += 1
        else:
            merged_rows.append(rec)

    extra_rerun = [k for k in rerun_keys if k not in {pair_key(r) for r in repaired_rows}]
    if extra_rerun:
        for k in extra_rerun:
            merged_rows.append(rerun_index[k])

    # Recompute labels/escalation once more for consistency.
    for rec in merged_rows:
        _recompute_labels_and_escalation(rec, unclear_treatment)

    merged_rows.sort(
        key=lambda r: (question_sort_key(str(r.get("question_id", ""))), str(r.get("model", "")))
    )

    queue_rows: List[Dict[str, Any]] = []
    now = utc_now()
    for rec in merged_rows:
        if bool(rec.get("escalated_to_human")):
            queue_rows.append(
                {
                    "question_id": rec.get("question_id"),
                    "model": rec.get("model"),
                    "reason": "trust_or_escalate",
                    "grade": rec.get("correctness_grade"),
                    "timestamp": now,
                }
            )

    write_jsonl_atomic(merged_rows, final_out)
    write_jsonl_atomic(queue_rows, retry_queue_out)

    return {
        "merged_total_rows": len(merged_rows),
        "rerun_rows": len(rerun_rows),
        "rows_replaced": replaced,
        "retry_queue_rows": len(queue_rows),
    }


def rerun_missing_judges(
    evaluated_in: str,
    output_out: str,
    judges: Sequence[str],
    adjudicator: Optional[str],
    max_new_tokens: int,
    adjudicator_max_new_tokens: int,
    initial_delay: float,
    max_delay: float,
    backoff_factor: float,
    unclear_treatment: str,
    dry_run: bool,
) -> MissingJudgeRepairStats:
    stats = MissingJudgeRepairStats()
    rows = load_jsonl(evaluated_in)
    judge_cfg = _parse_judge_specs(judges)
    adjudicator_cfg: Optional[Dict[str, str]] = None
    if adjudicator:
        if ":" not in adjudicator:
            raise ValueError(
                f"Invalid adjudicator spec '{adjudicator}'. Expected provider:model"
            )
        provider, model = adjudicator.split(":", 1)
        adjudicator_cfg = {"provider": provider.strip().lower(), "model": model.strip()}

    if load_dotenv is not None:
        load_dotenv()
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

    client: Optional[Any] = None
    if not dry_run:
        from src.providers import MultiProviderClient
        client = MultiProviderClient(
            initial_delay=initial_delay,
            max_delay=max_delay,
            backoff_factor=backoff_factor,
        )

    repaired_rows: List[Dict[str, Any]] = []
    for rec in rows:
        stats.total_rows += 1
        out = dict(rec)

        grades = list(out.get("correctness_judge_grades") or [])
        statuses = _as_text_list(out.get("correctness_judge_statuses"))
        reasoning = _as_text_list(out.get("correctness_judge_reasoning"))
        n = max(len(judge_cfg), len(grades), len(statuses), len(reasoning))
        grades = _ensure_len(grades, n)
        statuses = _ensure_len(statuses, n)
        reasoning = _ensure_len(reasoning, n)

        # Target only missing/failed slots among the configured 3 judges.
        missing_slots: List[int] = []
        for idx in range(len(judge_cfg)):
            grade = grades[idx]
            status = str(statuses[idx] or "")
            if grade in VALID_GRADES and status == "OK":
                continue
            missing_slots.append(idx)

        if missing_slots:
            stats.rows_with_missing_slots += 1
            stats.missing_slots_targeted += len(missing_slots)

        if (not dry_run) and missing_slots:
            question = str(out.get("question", ""))
            prediction = str(out.get("greedy_answer", ""))
            ground_truths = [str(x) for x in (out.get("ground_truth") or [])]
            for idx in missing_slots:
                cfg = judge_cfg[idx]
                stats.judge_calls += 1
                try:
                    res = check_correctness_llm(
                        prediction=prediction,
                        ground_truths=ground_truths,
                        question=question,
                        inference_client=client,  # type: ignore[arg-type]
                        judge_provider=cfg["provider"],
                        judge_model=cfg["model"],
                        max_new_tokens=max_new_tokens,
                    )
                except Exception as exc:
                    # Keep explicit infra failure marker for this slot.
                    grades[idx] = None
                    statuses[idx] = "API_FAILED"
                    reasoning[idx] = f"Judge exception: {exc}"
                    stats.repaired_slots_api_failed += 1
                    continue

                if res.match_type == "llm_judge_failed":
                    grades[idx] = None
                    statuses[idx] = "API_FAILED"
                    reasoning[idx] = (res.judge_reasoning or ["Judge failure"])[0]
                    stats.repaired_slots_api_failed += 1
                elif res.match_type == "llm_judge_parse_failed":
                    grades[idx] = None
                    statuses[idx] = "PARSE_FAILED"
                    reasoning[idx] = (res.judge_reasoning or ["Judge parse failure"])[0]
                    stats.repaired_slots_parse_failed += 1
                else:
                    grades[idx] = res.grade
                    statuses[idx] = "OK"
                    reasoning[idx] = (res.judge_reasoning or [""])[0]
                    stats.repaired_slots_ok += 1

        out["correctness_judge_grades"] = grades[: len(judge_cfg)]
        out["correctness_judge_statuses"] = statuses[: len(judge_cfg)]
        out["correctness_judge_reasoning"] = reasoning[: len(judge_cfg)]

        majority = _majority_grade(out["correctness_judge_grades"])
        if majority is not None:
            out["correctness_grade"] = majority
            out["correctness_unclear"] = (majority == "NOT_ATTEMPTED")
            out["greedy_correct"] = (majority == "CORRECT")
            out["correctness_match_type"] = "llm_judge_ensemble"
            out["correctness_decision_source"] = "MAJORITY"
            out["correctness_adjudicator_grade"] = None
            out["correctness_adjudicator_status"] = None
            out["correctness_adjudicator_reasoning"] = None
            stats.rows_resolved_majority += 1
        else:
            if (not dry_run) and adjudicator_cfg is not None:
                stats.adjudicator_calls += 1
                grade_label, status, reason = check_correctness_llm_adjudicator(
                    prediction=str(out.get("greedy_answer", "")),
                    ground_truths=[str(x) for x in (out.get("ground_truth") or [])],
                    question=str(out.get("question", "")),
                    inference_client=client,  # type: ignore[arg-type]
                    phase1_grades=out["correctness_judge_grades"],
                    phase1_statuses=out["correctness_judge_statuses"],
                    phase1_reasoning=out["correctness_judge_reasoning"],
                    judge_provider=adjudicator_cfg["provider"],
                    judge_model=adjudicator_cfg["model"],
                    max_new_tokens=adjudicator_max_new_tokens,
                )
                out["correctness_adjudicator_status"] = status
                out["correctness_adjudicator_reasoning"] = reason
                if status == "OK" and grade_label in VALID_GRADES:
                    out["correctness_grade"] = grade_label
                    out["correctness_unclear"] = (grade_label == "NOT_ATTEMPTED")
                    out["greedy_correct"] = (grade_label == "CORRECT")
                    out["correctness_match_type"] = "llm_judge_ensemble"
                    out["correctness_decision_source"] = "ADJUDICATOR"
                    out["correctness_adjudicator_grade"] = grade_label
                    stats.rows_resolved_adjudicator += 1
                else:
                    out["correctness_grade"] = "NOT_ATTEMPTED"
                    out["correctness_unclear"] = True
                    out["greedy_correct"] = False
                    out["correctness_match_type"] = "llm_judge_ensemble"
                    out["correctness_decision_source"] = "UNRESOLVED"
                    out["correctness_adjudicator_grade"] = None
                    stats.rows_unresolved += 1
            else:
                out["correctness_grade"] = "NOT_ATTEMPTED"
                out["correctness_unclear"] = True
                out["greedy_correct"] = False
                out["correctness_match_type"] = "llm_judge_ensemble"
                out["correctness_decision_source"] = "UNRESOLVED"
                if "correctness_adjudicator_grade" not in out:
                    out["correctness_adjudicator_grade"] = None
                stats.rows_unresolved += 1

        _recompute_labels_and_escalation(out, unclear_treatment)
        repaired_rows.append(out)

    repaired_rows.sort(
        key=lambda r: (question_sort_key(str(r.get("question_id", ""))), str(r.get("model", "")))
    )

    if not dry_run:
        write_jsonl_atomic(repaired_rows, output_out)
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline repair + targeted Phase-2 rerun helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Offline repair and emit hard-case subsets.")
    p_prepare.add_argument("--evaluated-in", required=True, help="Existing evaluated JSONL.")
    p_prepare.add_argument("--raw-in", required=True, help="Phase-1 raw JSONL used for targeted rerun.")
    p_prepare.add_argument("--repaired-out", required=True, help="Full offline-repaired evaluated JSONL.")
    p_prepare.add_argument("--hardcases-eval-out", required=True, help="Hard-case subset from evaluated rows.")
    p_prepare.add_argument("--hardcases-raw-out", required=True, help="Hard-case subset from raw rows.")
    p_prepare.add_argument("--pairs-out", required=True, help="JSONL list of hard-case (question_id, model) pairs.")
    p_prepare.add_argument(
        "--unclear-treatment",
        default="exclude",
        choices=["exclude", "count_as_different"],
        help="Labeling policy for unclear equivalence judgments.",
    )

    p_merge = sub.add_parser("merge", help="Merge targeted rerun output and rebuild retry queue.")
    p_merge.add_argument("--repaired-in", required=True, help="Full offline-repaired evaluated JSONL.")
    p_merge.add_argument("--rerun-in", required=True, help="Evaluated JSONL produced from hard-case rerun.")
    p_merge.add_argument("--final-out", required=True, help="Final merged evaluated JSONL.")
    p_merge.add_argument("--retry-queue-out", required=True, help="Rebuilt retry_queue.jsonl output path.")
    p_merge.add_argument(
        "--unclear-treatment",
        default="exclude",
        choices=["exclude", "count_as_different"],
        help="Labeling policy for unclear equivalence judgments.",
    )

    p_rerun_missing = sub.add_parser(
        "rerun-missing-judges",
        help="Rerun only missing/failed judge slots, then recompute/adjudicate.",
    )
    p_rerun_missing.add_argument("--evaluated-in", required=True, help="Input evaluated JSONL subset.")
    p_rerun_missing.add_argument("--output-out", required=True, help="Output evaluated JSONL subset.")
    p_rerun_missing.add_argument(
        "--judges",
        nargs=3,
        metavar="PROVIDER:MODEL",
        default=[
            "openai:gpt-5.2",
            "anthropic:claude-sonnet-4-5",
            "xai:grok-4-1-fast-non-reasoning",
        ],
        help="Exactly 3 judge specs in ensemble order.",
    )
    p_rerun_missing.add_argument(
        "--adjudicator",
        default="openai:gpt-5.2",
        help="Adjudicator provider:model (set empty string to disable).",
    )
    p_rerun_missing.add_argument(
        "--max-new-tokens",
        type=int,
        default=320,
        help="Max tokens for missing-slot judge reruns.",
    )
    p_rerun_missing.add_argument(
        "--adjudicator-max-new-tokens",
        type=int,
        default=320,
        help="Max tokens for adjudicator calls.",
    )
    p_rerun_missing.add_argument("--initial-delay", type=float, default=2.0)
    p_rerun_missing.add_argument("--max-delay", type=float, default=60.0)
    p_rerun_missing.add_argument("--backoff-factor", type=float, default=2.0)
    p_rerun_missing.add_argument(
        "--unclear-treatment",
        default="exclude",
        choices=["exclude", "count_as_different"],
        help="Labeling policy for unclear equivalence judgments.",
    )
    p_rerun_missing.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and count calls without making API requests or writing output.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        stats = prepare(
            evaluated_in=args.evaluated_in,
            raw_in=args.raw_in,
            repaired_out=args.repaired_out,
            hardcases_eval_out=args.hardcases_eval_out,
            hardcases_raw_out=args.hardcases_raw_out,
            pairs_out=args.pairs_out,
            unclear_treatment=args.unclear_treatment,
        )
        print(
            json.dumps(
                {
                    "phase": "prepare",
                    "total_rows": stats.total_rows,
                    "repaired_judge_slots": stats.repaired_judge_slots,
                    "rows_with_repair": stats.rows_with_repair,
                    "resolved_offline": stats.resolved_offline,
                    "hard_cases": stats.unresolved_hard_cases,
                },
                indent=2,
            )
        )
        return

    if args.command == "merge":
        summary = merge(
            repaired_in=args.repaired_in,
            rerun_in=args.rerun_in,
            final_out=args.final_out,
            retry_queue_out=args.retry_queue_out,
            unclear_treatment=args.unclear_treatment,
        )
        payload = {"phase": "merge"}
        payload.update(summary)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "rerun-missing-judges":
        adjudicator = args.adjudicator.strip() if isinstance(args.adjudicator, str) else None
        if adjudicator == "":
            adjudicator = None
        stats = rerun_missing_judges(
            evaluated_in=args.evaluated_in,
            output_out=args.output_out,
            judges=args.judges,
            adjudicator=adjudicator,
            max_new_tokens=int(args.max_new_tokens),
            adjudicator_max_new_tokens=int(args.adjudicator_max_new_tokens),
            initial_delay=float(args.initial_delay),
            max_delay=float(args.max_delay),
            backoff_factor=float(args.backoff_factor),
            unclear_treatment=args.unclear_treatment,
            dry_run=bool(args.dry_run),
        )
        payload = {
            "phase": "rerun-missing-judges",
            "dry_run": bool(args.dry_run),
            "total_rows": stats.total_rows,
            "rows_with_missing_slots": stats.rows_with_missing_slots,
            "missing_slots_targeted": stats.missing_slots_targeted,
            "judge_calls": stats.judge_calls,
            "repaired_slots_ok": stats.repaired_slots_ok,
            "repaired_slots_parse_failed": stats.repaired_slots_parse_failed,
            "repaired_slots_api_failed": stats.repaired_slots_api_failed,
            "rows_resolved_majority": stats.rows_resolved_majority,
            "adjudicator_calls": stats.adjudicator_calls,
            "rows_resolved_adjudicator": stats.rows_resolved_adjudicator,
            "rows_unresolved": stats.rows_unresolved,
        }
        print(json.dumps(payload, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
