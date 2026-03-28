#!/usr/bin/env python3
"""
Compute cross-model CE overlap and same-wrong rates.

Supports:
1) Multiple CE thresholds (e.g., 0.8 / 0.9 / 1.0).
2) Two equivalence methods:
   - heuristic: string-based matcher (fast baseline)
   - nli_hybrid: bidirectional NLI with thresholds
     same if both directions >= eq_same_hi
     different if either direction <= eq_diff_lo
     otherwise unclear (or optional LLM fallback on borderline only)

The script writes one CSV and one LaTeX table per CE threshold.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


DEFAULT_DATA_FILE = Path(
    "data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("data/results/analysis/v2_thesis/tables")


@dataclass(frozen=True)
class EquivalenceDecision:
    label: str
    detail: str
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute cross-model CE overlap analysis.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--ce-thresholds",
        nargs="+",
        type=float,
        default=[1.0],
        help="CE label thresholds to evaluate (choices: 0.8 0.9 1.0).",
    )
    parser.add_argument(
        "--equivalence-method",
        choices=["heuristic", "nli_hybrid"],
        default="nli_hybrid",
    )
    parser.add_argument("--eq-same-hi", type=float, default=0.70)
    parser.add_argument("--eq-diff-lo", type=float, default=0.30)
    parser.add_argument("--nli-model", type=str, default=DEFAULT_NLI_MODEL)
    parser.add_argument("--nli-device", type=str, default=None)
    parser.add_argument("--nli-batch-size", type=int, default=8)
    parser.add_argument(
        "--llm-fallback-borderline",
        action="store_true",
        help="For nli_hybrid only: use LLM judge only for borderline NLI cases.",
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        default="openai",
        help="LLM provider for borderline fallback (e.g., openai, anthropic, google, xai, deepseek, groq, openrouter).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5.2",
        help="LLM model used for borderline fallback decisions.",
    )
    parser.add_argument(
        "--judge-max-new-tokens",
        type=int,
        default=220,
        help="Max new tokens for each borderline LLM equivalence call.",
    )
    parser.add_argument(
        "--write-canonical-1p0",
        action="store_true",
        help="Also write canonical files without suffix for CE threshold 1.0.",
    )
    args = parser.parse_args()

    valid = {0.8, 0.9, 1.0}
    invalid = [t for t in args.ce_thresholds if t not in valid]
    if invalid:
        raise ValueError(f"Unsupported CE thresholds: {invalid}. Use only 0.8/0.9/1.0.")
    if not (0.0 < args.eq_diff_lo < args.eq_same_hi < 1.0):
        raise ValueError("Require 0 < eq_diff_lo < eq_same_hi < 1.")
    if args.llm_fallback_borderline and args.equivalence_method != "nli_hybrid":
        raise ValueError("--llm-fallback-borderline is only valid with --equivalence-method nli_hybrid.")
    if args.judge_max_new_tokens <= 0:
        raise ValueError("--judge-max-new-tokens must be > 0.")

    args.ce_thresholds = sorted(set(args.ce_thresholds))
    return args


def threshold_to_label_col(threshold: float) -> str:
    if threshold == 1.0:
        return "error_label_1.0"
    if threshold == 0.9:
        return "error_label_0.9"
    if threshold == 0.8:
        return "error_label_0.8"
    raise ValueError(f"Unsupported threshold {threshold}")


def threshold_tag(threshold: float) -> str:
    return str(threshold).replace(".", "p")


def equivalence_method_tag(method: str, llm_fallback_borderline: bool) -> str:
    if method == "heuristic":
        return "heuristic"
    if llm_fallback_borderline:
        return "nlihybridllm"
    return "nlihybrid"


def load_data(path: Path) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    with path.open() as handle:
        for line in handle:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def normalize_answer(text: str) -> str:
    if not text:
        return ""
    out = text.lower().strip()
    out = re.sub(r"[^\w\s]", "", out)
    out = re.sub(r"\s+", " ", out)
    stop = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "it",
        "that",
        "this",
        "to",
        "of",
        "for",
        "in",
        "on",
        "with",
    }
    return " ".join(tok for tok in out.split() if tok not in stop)


def extract_core_answer(text: str) -> str:
    if not text:
        return ""
    sentences = text.replace("\n", " ").split(".")
    if sentences:
        return normalize_answer(sentences[0])
    return normalize_answer(text)


def heuristic_equivalence(answer_a: str, answer_b: str, seq_threshold: float = 0.6) -> EquivalenceDecision:
    if not answer_a or not answer_b:
        return EquivalenceDecision(label="unclear", detail="empty", score=0.0)

    norm_a = normalize_answer(answer_a)
    norm_b = normalize_answer(answer_b)
    if norm_a == norm_b:
        return EquivalenceDecision(label="same", detail="exact", score=1.0)
    if norm_a in norm_b or norm_b in norm_a:
        return EquivalenceDecision(label="same", detail="substring", score=0.9)

    core_a = extract_core_answer(answer_a)
    core_b = extract_core_answer(answer_b)
    if core_a and core_b and (core_a == core_b or core_a in core_b or core_b in core_a):
        return EquivalenceDecision(label="same", detail="core_match", score=0.85)

    words_a = set(norm_a.split())
    words_b = set(norm_b.split())
    jaccard = 0.0
    if words_a and words_b:
        inter = words_a & words_b
        union = words_a | words_b
        jaccard = len(inter) / len(union)
        if jaccard >= 0.7:
            return EquivalenceDecision(label="same", detail="word_overlap", score=jaccard)

    seq = SequenceMatcher(None, norm_a, norm_b).ratio()
    if seq >= seq_threshold:
        return EquivalenceDecision(label="same", detail="sequence", score=seq)
    return EquivalenceDecision(label="different", detail="different", score=max(seq, jaccard))


def nli_batch_pair_entailment(
    nli_judge: object,
    premises: List[str],
    hypotheses: List[str],
) -> List[float]:
    """
    Batch entailment for aligned premise/hypothesis pairs.
    """
    import torch

    if len(premises) != len(hypotheses):
        raise ValueError("Premises and hypotheses must have equal length.")
    if not premises:
        return []

    inputs = nli_judge.tokenizer(
        premises,
        hypotheses,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {k: v.to(nli_judge.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = nli_judge.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        entailment_probs = probs[:, nli_judge.entailment_idx].tolist()

    return entailment_probs


def build_question_model_lookup(
    frame: pd.DataFrame,
) -> Tuple[Dict[str, Dict[str, Dict[str, str]]], List[str]]:
    """
    Returns:
      question_lookup[question_id][model] = {'question': str, 'answer': str}
      model list in stable sorted order
    """
    q_lookup: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for _, row in frame.iterrows():
        qid = str(row["question_id"])
        model = str(row["model"])
        q_lookup[qid][model] = {
            "question": str(row.get("question", "")),
            "answer": str(row.get("greedy_answer", "") or ""),
        }
    models = sorted(frame["model"].dropna().astype(str).unique().tolist())
    return q_lookup, models


def build_ce_maps(
    frame: pd.DataFrame, thresholds: Iterable[float]
) -> Tuple[Dict[float, Dict[str, Dict[str, Dict[str, str]]]], Dict[float, Dict[str, int]]]:
    """
    Returns:
      ce_by_threshold[t][qid][model] = {'question': str, 'answer': str}
      ce_count_by_threshold[t][model] = CE count
    """
    ce_by_threshold: Dict[float, Dict[str, Dict[str, Dict[str, str]]]] = {}
    ce_count_by_threshold: Dict[float, Dict[str, int]] = {}
    for t in thresholds:
        col = threshold_to_label_col(t)
        subset = frame[frame[col] == "self_consistent_error"].copy()
        q_map: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
        counts: Dict[str, int] = defaultdict(int)
        for _, row in subset.iterrows():
            qid = str(row["question_id"])
            model = str(row["model"])
            q_map[qid][model] = {
                "question": str(row.get("question", "")),
                "answer": str(row.get("greedy_answer", "") or ""),
            }
            counts[model] += 1
        ce_by_threshold[t] = q_map
        ce_count_by_threshold[t] = counts
    return ce_by_threshold, ce_count_by_threshold


def compute_overlap_qids_for_pair(
    ce_q_map: Dict[str, Dict[str, Dict[str, str]]], model_a: str, model_b: str
) -> List[str]:
    overlap = []
    for qid, model_data in ce_q_map.items():
        if model_a in model_data and model_b in model_data:
            overlap.append(qid)
    return overlap


def format_latex_table(
    frame: pd.DataFrame,
    ce_threshold: float,
    method: str,
    llm_fallback_borderline: bool,
    judge_provider: str,
    judge_model: str,
    eq_same_hi: float,
    eq_diff_lo: float,
) -> str:
    if method == "heuristic":
        method_desc = "heuristic string-based equivalence"
    elif llm_fallback_borderline:
        method_desc = f"NLI-hybrid with LLM fallback on borderline ({judge_provider}/{judge_model})"
    else:
        method_desc = "NLI-hybrid equivalence (no LLM fallback)"
    caption = (
        "Cross-model self-consistent error overlap with semantic answer comparison "
        f"(CE threshold {ce_threshold:.1f}; {method_desc}; "
        f"eq\\_same\\_hi={eq_same_hi:.2f}, eq\\_diff\\_lo={eq_diff_lo:.2f})."
    )
    out = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{caption} "
        r"``Overlap'' = questions where both models have CE. "
        r"``Same Wrong'' = overlap questions where both models gave semantically equivalent wrong answers.}}",
        r"\small",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Model A & Model B & A CE & B CE & Overlap & Jaccard & Same Wrong & Unclear & \% \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        ma = str(row["model_a"]).replace("_", r"\_").split(" (")[0]
        mb = str(row["model_b"]).replace("_", r"\_").split(" (")[0]
        out.append(
            f"{ma} & {mb} & {int(row['ce_a'])} & {int(row['ce_b'])} & {int(row['both_ce_overlap'])} "
            f"& {float(row['jaccard']):.3f} & {int(row['same_wrong_answer'])} & {int(row['unclear_equivalence'])} "
            f"& {float(row['same_wrong_pct']):.1f}\\% \\\\"
        )
    out.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(out) + "\n"


def save_outputs(
    results: pd.DataFrame,
    output_dir: Path,
    method: str,
    llm_fallback_borderline: bool,
    judge_provider: str,
    judge_model: str,
    ce_threshold: float,
    eq_same_hi: float,
    eq_diff_lo: float,
    write_canonical_1p0: bool,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    method_tag = equivalence_method_tag(method, llm_fallback_borderline)
    t_tag = threshold_tag(ce_threshold)
    csv_path = output_dir / f"cross_model_ce_overlap_semantic_{method_tag}_t{t_tag}.csv"
    tex_path = output_dir / f"cross_model_ce_semantic_overlap_{method_tag}_t{t_tag}.tex"

    results.to_csv(csv_path, index=False)
    latex = format_latex_table(
        results,
        ce_threshold=ce_threshold,
        method=method,
        llm_fallback_borderline=llm_fallback_borderline,
        judge_provider=judge_provider,
        judge_model=judge_model,
        eq_same_hi=eq_same_hi,
        eq_diff_lo=eq_diff_lo,
    )
    tex_path.write_text(latex)

    # Maintain backwards-compatible canonical names for strict 1.0 if requested.
    if write_canonical_1p0 and ce_threshold == 1.0:
        canonical_csv = output_dir / "cross_model_ce_overlap_semantic.csv"
        canonical_tex = output_dir / "cross_model_ce_semantic_overlap.tex"
        results.to_csv(canonical_csv, index=False)
        canonical_tex.write_text(latex)

    return csv_path, tex_path


def main() -> None:
    args = parse_args()
    print("Loading data...")
    df = load_data(args.data_file)
    print(f"Loaded {len(df)} rows from {args.data_file}")

    q_lookup, models = build_question_model_lookup(df)
    ce_by_threshold, ce_counts = build_ce_maps(df, args.ce_thresholds)
    pairs = list(combinations(models, 2))
    print(f"Models ({len(models)}): {models}")
    print(f"Model pairs: {len(pairs)}")
    print(f"CE thresholds: {args.ce_thresholds}")
    print(f"Equivalence method: {args.equivalence_method}")
    print(f"LLM fallback on borderline: {args.llm_fallback_borderline}")

    nli_judge = None
    decide_equivalence_hybrid_fn = None
    inference_client = None
    if args.equivalence_method == "nli_hybrid":
        from src.nli_judge import NLISemanticJudge

        print(f"Loading NLI model: {args.nli_model}")
        nli_judge = NLISemanticJudge(
            model_name=args.nli_model,
            device=args.nli_device,
            batch_size=args.nli_batch_size,
        )
        if args.llm_fallback_borderline:
            from src.hybrid_judging import decide_equivalence_hybrid
            from src.providers import MultiProviderClient

            decide_equivalence_hybrid_fn = decide_equivalence_hybrid
            inference_client = MultiProviderClient()
            print(
                f"Borderline LLM judge enabled: provider={args.judge_provider}, model={args.judge_model}, "
                f"max_new_tokens={args.judge_max_new_tokens}"
            )

    # Pair-level overlap sets for each threshold.
    overlap_qids: Dict[Tuple[str, str], Dict[float, List[str]]] = {}
    for model_a, model_b in pairs:
        per_t: Dict[float, List[str]] = {}
        for t in args.ce_thresholds:
            per_t[t] = compute_overlap_qids_for_pair(ce_by_threshold[t], model_a, model_b)
        overlap_qids[(model_a, model_b)] = per_t

    # Cache equivalence decisions once per (pair, qid) across all requested thresholds.
    eq_cache: Dict[Tuple[str, str, str], EquivalenceDecision] = {}
    total_comparisons = 0
    llm_borderline_decisions = 0
    for pair_idx, (model_a, model_b) in enumerate(pairs, start=1):
        union_qids = sorted(set().union(*overlap_qids[(model_a, model_b)].values()))
        if args.equivalence_method == "heuristic":
            for qid in union_qids:
                key = (model_a, model_b, qid)
                if key in eq_cache:
                    continue
                if qid not in q_lookup or model_a not in q_lookup[qid] or model_b not in q_lookup[qid]:
                    eq_cache[key] = EquivalenceDecision(label="unclear", detail="missing_answer", score=0.0)
                    continue
                answer_a = q_lookup[qid][model_a]["answer"]
                answer_b = q_lookup[qid][model_b]["answer"]
                eq_cache[key] = heuristic_equivalence(answer_a, answer_b)
                total_comparisons += 1
        else:
            assert nli_judge is not None
            if args.llm_fallback_borderline:
                assert decide_equivalence_hybrid_fn is not None
                for qid in union_qids:
                    key = (model_a, model_b, qid)
                    if key in eq_cache:
                        continue
                    if qid not in q_lookup or model_a not in q_lookup[qid] or model_b not in q_lookup[qid]:
                        eq_cache[key] = EquivalenceDecision(label="unclear", detail="missing_answer", score=0.0)
                        continue

                    question = q_lookup[qid][model_a]["question"]
                    answer_a = q_lookup[qid][model_a]["answer"]
                    answer_b = q_lookup[qid][model_b]["answer"]
                    decision = decide_equivalence_hybrid_fn(
                        question=question,
                        answer_a=answer_a,
                        answer_b=answer_b,
                        nli_judge=nli_judge,
                        eq_same_hi=args.eq_same_hi,
                        eq_diff_lo=args.eq_diff_lo,
                        inference_client=inference_client,
                        judge_provider=args.judge_provider,
                        judge_model=args.judge_model,
                        max_new_tokens=args.judge_max_new_tokens,
                    )
                    score = min(float(decision.prob_forward or 0.0), float(decision.prob_reverse or 0.0))
                    eq_cache[key] = EquivalenceDecision(
                        label=decision.label,
                        detail=decision.source_detail,
                        score=score,
                    )
                    if decision.source == "LLM":
                        llm_borderline_decisions += 1
                    total_comparisons += 1
            else:
                valid_qids: List[str] = []
                context_a: List[str] = []
                context_b: List[str] = []
                for qid in union_qids:
                    key = (model_a, model_b, qid)
                    if key in eq_cache:
                        continue
                    if qid not in q_lookup or model_a not in q_lookup[qid] or model_b not in q_lookup[qid]:
                        eq_cache[key] = EquivalenceDecision(label="unclear", detail="missing_answer", score=0.0)
                        continue
                    question = q_lookup[qid][model_a]["question"]
                    answer_a = q_lookup[qid][model_a]["answer"]
                    answer_b = q_lookup[qid][model_b]["answer"]
                    valid_qids.append(qid)
                    context_a.append(f"Question: {question} Answer: {answer_a}")
                    context_b.append(f"Question: {question} Answer: {answer_b}")

                batch_size = max(1, int(args.nli_batch_size))
                for start in range(0, len(valid_qids), batch_size):
                    end = start + batch_size
                    qids_batch = valid_qids[start:end]
                    a_batch = context_a[start:end]
                    b_batch = context_b[start:end]
                    pf = nli_batch_pair_entailment(nli_judge, a_batch, b_batch)
                    pr = nli_batch_pair_entailment(nli_judge, b_batch, a_batch)
                    for idx, qid in enumerate(qids_batch):
                        p_forward = float(pf[idx])
                        p_reverse = float(pr[idx])
                        if p_forward >= args.eq_same_hi and p_reverse >= args.eq_same_hi:
                            label = "same"
                            detail = "NLI_HIGH_CONFIDENCE_SAME"
                        elif p_forward <= args.eq_diff_lo or p_reverse <= args.eq_diff_lo:
                            label = "different"
                            detail = "NLI_HIGH_CONFIDENCE_DIFFERENT"
                        else:
                            label = "unclear"
                            detail = "NLI_BORDERLINE_NO_LLM"
                        eq_cache[(model_a, model_b, qid)] = EquivalenceDecision(
                            label=label,
                            detail=detail,
                            score=min(p_forward, p_reverse),
                        )
                        total_comparisons += 1
        print(f"  Pair {pair_idx:02d}/{len(pairs)} done: {model_a[:16]} vs {model_b[:16]} ({len(union_qids)} overlaps)")

    print(f"Computed equivalence decisions: {total_comparisons}")
    if args.equivalence_method == "nli_hybrid" and args.llm_fallback_borderline:
        print(f"Borderline decisions sent to LLM: {llm_borderline_decisions}")

    for ce_t in args.ce_thresholds:
        rows: List[Dict[str, object]] = []
        print("\n" + "=" * 100)
        print(f"CE threshold {ce_t:.1f}")
        print("=" * 100)
        for model_a, model_b in pairs:
            overlap = overlap_qids[(model_a, model_b)][ce_t]
            overlap_count = len(overlap)
            same_count = 0
            unclear_count = 0
            for qid in overlap:
                decision = eq_cache[(model_a, model_b, qid)]
                if decision.label == "same":
                    same_count += 1
                elif decision.label == "unclear":
                    unclear_count += 1

            ce_a = int(ce_counts[ce_t].get(model_a, 0))
            ce_b = int(ce_counts[ce_t].get(model_b, 0))
            union = ce_a + ce_b - overlap_count
            jaccard = overlap_count / union if union else 0.0
            same_pct = 100.0 * same_count / overlap_count if overlap_count else 0.0
            rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "ce_a": ce_a,
                    "ce_b": ce_b,
                    "both_ce_overlap": overlap_count,
                    "jaccard": round(jaccard, 3),
                    "same_wrong_answer": same_count,
                    "unclear_equivalence": unclear_count,
                    "same_wrong_pct": round(same_pct, 1),
                }
            )

        results = pd.DataFrame(rows).sort_values("both_ce_overlap", ascending=False).reset_index(drop=True)
        csv_path, tex_path = save_outputs(
            results=results,
            output_dir=args.output_dir,
            method=args.equivalence_method,
            llm_fallback_borderline=args.llm_fallback_borderline,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            ce_threshold=ce_t,
            eq_same_hi=args.eq_same_hi,
            eq_diff_lo=args.eq_diff_lo,
            write_canonical_1p0=args.write_canonical_1p0,
        )
        total_overlap = int(results["both_ce_overlap"].sum())
        total_same = int(results["same_wrong_answer"].sum())
        total_unclear = int(results["unclear_equivalence"].sum())
        overall_pct = 100.0 * total_same / total_overlap if total_overlap else 0.0
        print(f"Saved CSV: {csv_path}")
        print(f"Saved TeX: {tex_path}")
        print(f"Total CE overlaps: {total_overlap}")
        print(f"Total same wrong: {total_same}")
        print(f"Total unclear: {total_unclear}")
        print(f"Overall same wrong %: {overall_pct:.1f}%")


if __name__ == "__main__":
    main()
