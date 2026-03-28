"""Hybrid NLI + LLM judging utilities for equivalence and correctness."""

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .correctness import check_correctness_llm

if TYPE_CHECKING:
    from .nli_judge import NLISemanticJudge
    from .providers import MultiProviderClient

logger = logging.getLogger(__name__)


EQUIVALENCE_BORDERLINE_PROMPT = """
You are grading semantic equivalence between two candidate answers to the SAME question.

Return exactly one label:
- "same": both answers express the same factual claim/truth conditions.
- "different": answers make materially different or conflicting factual claims.
- "unclear": insufficient specificity, ambiguity, refusal, or cannot decide.

Rules:
- Use ONLY the provided question and answers.
- Do NOT use outside/world knowledge.
- Ignore style, wording, and verbosity differences when meaning is the same.
- If one answer adds a conflicting factual claim, label "different".

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Output format requirements:
- Return exactly one valid JSON object and nothing else.
- JSON schema:
  {{"label":"same|different|unclear","reasoning":"<1-2 short sentences>"}}
""".strip()


@dataclass(frozen=True)
class HybridThresholds:
    """Threshold set used by hybrid judging."""

    eq_same_hi: float = 0.70
    eq_diff_lo: float = 0.30
    corr_hi: float = 0.70
    corr_lo: float = 0.30


@dataclass
class HybridEquivalenceDecision:
    """Final equivalence decision for one answer pair."""

    label: str
    source: str
    source_detail: str
    prob_forward: Optional[float] = None
    prob_reverse: Optional[float] = None
    llm_label_forward: Optional[str] = None
    llm_label_reverse: Optional[str] = None
    llm_reasoning_forward: Optional[str] = None
    llm_reasoning_reverse: Optional[str] = None


@dataclass
class HybridCorrectnessDecision:
    """Final correctness decision for one stochastic sample."""

    grade: str
    source: str
    source_detail: str
    p_max: Optional[float] = None
    matched_gold: Optional[str] = None
    matched_gold_index: Optional[int] = None
    judge_reasoning: Optional[str] = None


def _judge_response_format_for_provider(provider: str) -> Optional[Dict[str, Any]]:
    p = (provider or "").strip().lower()
    if p == "openai":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "equivalence_grade",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label": {"type": "string", "enum": ["same", "different", "unclear"]},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["label", "reasoning"],
                },
            },
        }
    if p in {"xai", "deepseek", "groq", "openrouter", "huggingface", "google"}:
        return {"type": "json_object"}
    return None


def _normalize_equivalence_label(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token in {"same", "different", "unclear"}:
        return token
    if token in {"equivalent", "equivalence"}:
        return "same"
    if token in {"not_equivalent", "not equivalent", "conflict", "contradiction"}:
        return "different"
    if token in {"unknown", "ambiguous", "cannot_decide", "cant_decide"}:
        return "unclear"
    return ""


def _parse_equivalence_response(raw_text: str) -> Tuple[str, str]:
    text = (raw_text or "").strip()
    if not text:
        return "", ""

    candidates: List[str] = [text]
    block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if block:
        candidates.append(block.group(1).strip())

    decoder = json.JSONDecoder()
    for idx in range(len(text) - 1, -1, -1):
        if text[idx] != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[idx:])
        except Exception:
            continue
        if isinstance(obj, dict):
            candidates.append(text[idx: idx + end].strip())

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            label = _normalize_equivalence_label(parsed.get("label", ""))
            reasoning = str(parsed.get("reasoning", "")).strip()
            if label:
                return label, reasoning

    lowered = text.lower()
    for label in ("same", "different", "unclear"):
        if re.search(rf"\b{label}\b", lowered):
            return label, text
    return "", text


def _equivalence_llm_call(
    question: str,
    answer_a: str,
    answer_b: str,
    inference_client: "MultiProviderClient",
    judge_provider: str,
    judge_model: str,
    max_new_tokens: int,
) -> Tuple[str, str]:
    prompt_text = EQUIVALENCE_BORDERLINE_PROMPT.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
    )
    response_format = _judge_response_format_for_provider(judge_provider)
    result = inference_client.generate_greedy(
        provider=judge_provider,
        model=judge_model,
        prompt=prompt_text,
        max_new_tokens=max_new_tokens,
        response_format=response_format,
    )
    return _parse_equivalence_response(result.text)


def decide_equivalence_hybrid(
    question: str,
    answer_a: str,
    answer_b: str,
    nli_judge: "NLISemanticJudge",
    eq_same_hi: float,
    eq_diff_lo: float,
    inference_client: Optional["MultiProviderClient"] = None,
    judge_provider: str = "openai",
    judge_model: str = "gpt-5.2",
    max_new_tokens: int = 220,
) -> HybridEquivalenceDecision:
    """
    Decide semantic equivalence with NLI first, GPT only on borderline cases.
    """
    text_a = (answer_a or "").strip()
    text_b = (answer_b or "").strip()
    if not text_a or not text_b:
        return HybridEquivalenceDecision(
            label="unclear",
            source="NLI",
            source_detail="NLI_EMPTY_INPUT",
            prob_forward=0.0,
            prob_reverse=0.0,
        )

    context_a = f"Question: {question} Answer: {text_a}"
    context_b = f"Question: {question} Answer: {text_b}"
    try:
        prob_forward = float(nli_judge._get_entailment_prob(context_a, context_b))
        prob_reverse = float(nli_judge._get_entailment_prob(context_b, context_a))
    except Exception as exc:
        logger.warning("NLI equivalence failed: %s", exc)
        return HybridEquivalenceDecision(
            label="unclear",
            source="NLI",
            source_detail="NLI_EXCEPTION",
            prob_forward=0.0,
            prob_reverse=0.0,
        )

    if prob_forward >= eq_same_hi and prob_reverse >= eq_same_hi:
        return HybridEquivalenceDecision(
            label="same",
            source="NLI",
            source_detail="NLI_HIGH_CONFIDENCE_SAME",
            prob_forward=prob_forward,
            prob_reverse=prob_reverse,
        )

    if prob_forward <= eq_diff_lo or prob_reverse <= eq_diff_lo:
        return HybridEquivalenceDecision(
            label="different",
            source="NLI",
            source_detail="NLI_HIGH_CONFIDENCE_DIFFERENT",
            prob_forward=prob_forward,
            prob_reverse=prob_reverse,
        )

    if inference_client is None:
        return HybridEquivalenceDecision(
            label="unclear",
            source="NLI",
            source_detail="NLI_BORDERLINE_NO_LLM",
            prob_forward=prob_forward,
            prob_reverse=prob_reverse,
        )

    try:
        label_forward, reasoning_forward = _equivalence_llm_call(
            question=question,
            answer_a=text_a,
            answer_b=text_b,
            inference_client=inference_client,
            judge_provider=judge_provider,
            judge_model=judge_model,
            max_new_tokens=max_new_tokens,
        )
        label_reverse, reasoning_reverse = _equivalence_llm_call(
            question=question,
            answer_a=text_b,
            answer_b=text_a,
            inference_client=inference_client,
            judge_provider=judge_provider,
            judge_model=judge_model,
            max_new_tokens=max_new_tokens,
        )
    except Exception as exc:
        logger.warning("LLM equivalence borderline call failed: %s", exc)
        return HybridEquivalenceDecision(
            label="unclear",
            source="LLM",
            source_detail="LLM_EXCEPTION",
            prob_forward=prob_forward,
            prob_reverse=prob_reverse,
        )

    if label_forward and label_forward == label_reverse and label_forward in {"same", "different", "unclear"}:
        final_label = label_forward
        detail = "LLM_BORDERLINE_AGREE"
    else:
        final_label = "unclear"
        detail = "LLM_BORDERLINE_DISAGREE"

    return HybridEquivalenceDecision(
        label=final_label,
        source="LLM",
        source_detail=detail,
        prob_forward=prob_forward,
        prob_reverse=prob_reverse,
        llm_label_forward=label_forward or None,
        llm_label_reverse=label_reverse or None,
        llm_reasoning_forward=reasoning_forward or None,
        llm_reasoning_reverse=reasoning_reverse or None,
    )


def grade_sample_correctness_hybrid(
    question: str,
    sample_answer: str,
    ground_truths: List[str],
    nli_judge: "NLISemanticJudge",
    corr_hi: float,
    corr_lo: float,
    inference_client: Optional["MultiProviderClient"] = None,
    judge_provider: str = "openai",
    judge_model: str = "gpt-5.2",
    max_new_tokens: int = 260,
) -> HybridCorrectnessDecision:
    """
    Grade one stochastic sample versus OR gold targets using hybrid logic.
    """
    answer = (sample_answer or "").strip()
    if not answer:
        return HybridCorrectnessDecision(
            grade="NOT_ATTEMPTED",
            source="NLI",
            source_detail="EMPTY_SAMPLE",
            p_max=0.0,
        )

    cleaned_gold = [g.strip() for g in (ground_truths or []) if isinstance(g, str) and g.strip()]
    if not cleaned_gold:
        return HybridCorrectnessDecision(
            grade="NOT_ATTEMPTED",
            source="NLI",
            source_detail="NO_GOLD_TARGET",
            p_max=0.0,
        )

    context_sample = f"Question: {question} Answer: {answer}"
    best_prob = -1.0
    best_idx = None
    best_gold = None

    for idx, gold in enumerate(cleaned_gold):
        context_gold = f"Question: {question} Answer: {gold}"
        try:
            prob = float(nli_judge._get_entailment_prob(context_sample, context_gold))
        except Exception as exc:
            logger.warning("NLI correctness failed for sample vs gold: %s", exc)
            continue
        if prob > best_prob:
            best_prob = prob
            best_idx = idx
            best_gold = gold

    if best_idx is None:
        return HybridCorrectnessDecision(
            grade="NOT_ATTEMPTED",
            source="NLI",
            source_detail="NLI_EXCEPTION",
            p_max=0.0,
        )

    if best_prob >= corr_hi:
        return HybridCorrectnessDecision(
            grade="CORRECT",
            source="NLI",
            source_detail="NLI_HIGH_CONFIDENCE_CORRECT",
            p_max=best_prob,
            matched_gold=best_gold,
            matched_gold_index=best_idx,
        )

    if best_prob <= corr_lo:
        return HybridCorrectnessDecision(
            grade="INCORRECT",
            source="NLI",
            source_detail="NLI_HIGH_CONFIDENCE_INCORRECT",
            p_max=best_prob,
            matched_gold=best_gold,
            matched_gold_index=best_idx,
        )

    if inference_client is None:
        return HybridCorrectnessDecision(
            grade="NOT_ATTEMPTED",
            source="NLI",
            source_detail="NLI_BORDERLINE_NO_LLM",
            p_max=best_prob,
            matched_gold=best_gold,
            matched_gold_index=best_idx,
        )

    llm_result = check_correctness_llm(
        prediction=answer,
        ground_truths=cleaned_gold,
        question=question,
        inference_client=inference_client,
        judge_provider=judge_provider,
        judge_model=judge_model,
        max_new_tokens=max_new_tokens,
    )
    return HybridCorrectnessDecision(
        grade=(llm_result.grade or "NOT_ATTEMPTED"),
        source="LLM",
        source_detail="LLM_BORDERLINE",
        p_max=best_prob,
        matched_gold=best_gold,
        matched_gold_index=best_idx,
        judge_reasoning=((llm_result.judge_reasoning or [""])[0] if llm_result.judge_reasoning else None),
    )


def compute_pairwise_hybrid_equivalence(
    question: str,
    sample_answers: List[str],
    nli_judge: "NLISemanticJudge",
    eq_same_hi: float,
    eq_diff_lo: float,
    inference_client: Optional["MultiProviderClient"] = None,
    judge_provider: str = "openai",
    judge_model: str = "gpt-5.2",
    max_new_tokens: int = 220,
) -> Dict[Tuple[int, int], HybridEquivalenceDecision]:
    """
    Compute hybrid equivalence decisions for all unordered sample pairs.
    """
    decisions: Dict[Tuple[int, int], HybridEquivalenceDecision] = {}
    total = len(sample_answers)
    for left in range(total):
        for right in range(left + 1, total):
            decisions[(left, right)] = decide_equivalence_hybrid(
                question=question,
                answer_a=sample_answers[left],
                answer_b=sample_answers[right],
                nli_judge=nli_judge,
                eq_same_hi=eq_same_hi,
                eq_diff_lo=eq_diff_lo,
                inference_client=inference_client,
                judge_provider=judge_provider,
                judge_model=judge_model,
                max_new_tokens=max_new_tokens,
            )
    return decisions


def compute_stochastic_correctness_metrics(
    equivalence_results: List[str],
    sample_grades: List[str],
) -> Dict[str, Optional[float]]:
    """
    Compute sample-level correctness metrics and the different-subset breakdown.
    """
    scored_grades = [g for g in sample_grades if g in {"CORRECT", "INCORRECT"}]
    scored_n = len(scored_grades)
    correct_n = sum(1 for g in scored_grades if g == "CORRECT")
    not_attempted_n = sum(1 for g in sample_grades if g == "NOT_ATTEMPTED")

    different_indices = [idx for idx, label in enumerate(equivalence_results) if label == "different"]
    different_scored_n = 0
    different_correct_n = 0
    for idx in different_indices:
        if idx >= len(sample_grades):
            continue
        grade = sample_grades[idx]
        if grade not in {"CORRECT", "INCORRECT"}:
            continue
        different_scored_n += 1
        if grade == "CORRECT":
            different_correct_n += 1

    stochastic_correct_rate = (correct_n / scored_n) if scored_n > 0 else None
    p_correct_given_different = (
        different_correct_n / different_scored_n
        if different_scored_n > 0
        else None
    )

    return {
        "stochastic_correct_rate": stochastic_correct_rate,
        "stochastic_scored_n": scored_n,
        "stochastic_not_attempted_n": not_attempted_n,
        "different_scored_n": different_scored_n,
        "different_correct_n": different_correct_n,
        "p_correct_given_different": p_correct_given_different,
    }
