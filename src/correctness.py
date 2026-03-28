"""
Correctness checking for answer verification.

Correctness uses LLM-as-judge grading as the primary decision method.
String/NLI helpers remain in this module for legacy compatibility and
calibration workflows, but are not used in the main correctness cascade.

Enhanced per research best-practices (Tier 1 + 2):
- Chain-of-Thought (CoT) reasoning before the grade letter
- NOT_ATTEMPTED tracked separately (not collapsed into INCORRECT)
- NLI probabilities stored for auditability
- Ensemble voting (majority)
- Failed judges skipped (not counted as INCORRECT)
- Phase-2 adjudicator fallback for unresolved ensemble decisions
"""

import re
import json
import string
import logging
from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING

from .schemas import CorrectnessResult

if TYPE_CHECKING:
    from .nli_judge import NLISemanticJudge
    from .providers import MultiProviderClient

logger = logging.getLogger(__name__)

# ===========================================================================
# SimpleQA-style LLM-as-judge grading prompt WITH Chain-of-Thought
# ===========================================================================

LLM_JUDGE_GRADING_PROMPT = """
Your job is to look at a question, one or more gold targets, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
```
These predicted answers are all CORRECT because:
 - They fully contain the important information in at least one gold target.
 - They do not contain any information that contradicts the gold target.
 - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
 - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
```
These predicted answers are all INCORRECT because:
 - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
 - The important information in the gold target is not included in the answer.
 - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k".
 - Predicted answers "120k", "124k", and "115k" are all CORRECT.
 - Predicted answers "100k" and "113k" are INCORRECT.
 - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
- If multiple gold targets are provided, they are alternatives (OR), not requirements to satisfy all at once.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
- Do not punish for typos in people's name if it's clearly the same name.
- Use only the provided question, gold targets, and predicted answer. Do not use outside/world knowledge.
- If the predicted answer contains both correct and incorrect claims, grade INCORRECT.
- If unsure between INCORRECT and NOT_ATTEMPTED, choose NOT_ATTEMPTED unless there is a clear contradiction.
- Do not overthink. Use only explicit evidence in the provided texts.

Here is a new example. Grade the predicted answer.
```
Question: {question}
Gold targets (each line is an alternative acceptable reference):
{targets_block}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Output format requirements:
- Return exactly one valid JSON object and nothing else.
- JSON schema:
  {{"reasoning":"<2-3 short sentences>","grade":"<A|B|C>"}}
""".strip()

LLM_JUDGE_ADJUDICATOR_PROMPT = """
You are the final adjudicator for answer correctness.

You will receive:
- A question
- One or more acceptable gold targets (alternatives, OR)
- A predicted answer
- Phase-1 panel outcomes from three judges

Labels:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Definitions:
- CORRECT: predicted answer includes the key information in at least one gold target, with no contradiction.
- INCORRECT: predicted answer contains a factual contradiction against the gold target(s).
- NOT_ATTEMPTED: predicted answer does not include key required information and does not clearly contradict.

Important adjudication rules:
- Use Phase-1 panel outputs as evidence, but make an independent final decision.
- If a Phase-1 judge has status API_FAILED or PARSE_FAILED, treat it as infrastructure failure, not a semantic vote.
- If panel votes conflict, resolve the conflict using question + gold targets + predicted answer.
- Use only the provided case information; do not use outside/world knowledge.
- If the predicted answer contains both correct and incorrect claims, choose INCORRECT.
- If unsure between INCORRECT and NOT_ATTEMPTED, choose NOT_ATTEMPTED unless contradiction is explicit.
- Do not overthink. Use only explicit evidence in the provided texts.

Now adjudicate this case:
Question: {question}
Gold targets (each line is an alternative acceptable reference):
{targets_block}
Predicted answer: {predicted_answer}

Phase-1 panel outputs:
{phase1_votes_block}

Output format requirements:
- Return exactly one valid JSON object and nothing else.
- JSON schema:
  {{"reasoning":"<2-3 short sentences>","grade":"<A|B|C>"}}
""".strip()

# Articles to strip
ARTICLES = {"a", "an", "the"}


# ===========================================================================
# Helpers
# ===========================================================================

def normalize(text: str, strip_articles: bool = True) -> str:
    """
    Normalize text for comparison.

    Operations:
    - Lowercase
    - Strip articles (the, a, an)
    - Remove punctuation
    - Remove extra whitespace
    - Handle "X (something)" formatting -> extract "X"

    Args:
        text: Input text to normalize
        strip_articles: Whether to remove articles

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower().strip()

    # Handle "X (something)" formatting - extract the part before parentheses
    # This handles cases like "Paris (city)" -> "Paris"
    paren_match = re.match(r'^([^(]+)\s*\(.*\)$', text)
    if paren_match:
        text = paren_match.group(1).strip()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Split into words for article removal
    words = text.split()

    # Strip articles if requested
    if strip_articles:
        words = [w for w in words if w not in ARTICLES]

    # Rejoin and normalize whitespace
    text = " ".join(words)

    return text


def _parse_cot_grade(raw_text: str) -> Tuple[str, str]:
    """Extract (grade_letter, reasoning) from a CoT response.

    The judge is asked to write reasoning first, then a final line with just
    the grade letter.  We search from the bottom up for a standalone A/B/C.

    Returns:
        (grade_letter, reasoning) where grade_letter is "A", "B", or "C"
        (or "" on parse failure).
    """
    raw_text = raw_text.strip()

    def _normalize_grade_token(value: Any) -> str:
        token = str(value).strip().upper()
        if token in ("A", "B", "C"):
            return token
        if token == "CORRECT":
            return "A"
        if token == "INCORRECT":
            return "B"
        if token == "NOT_ATTEMPTED":
            return "C"
        return ""

    # 1) Preferred strict format: JSON object
    json_candidates: List[str] = [raw_text]
    block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.IGNORECASE | re.DOTALL)
    if block:
        json_candidates.append(block.group(1).strip())

    # Anthropic and other non-schema-enforced providers may prepend prose
    # before a final JSON object. Recover by scanning for decodable objects.
    decoder = json.JSONDecoder()
    for idx in range(len(raw_text) - 1, -1, -1):
        if raw_text[idx] != "{":
            continue
        try:
            parsed_obj, end = decoder.raw_decode(raw_text[idx:])
        except Exception:
            continue
        if isinstance(parsed_obj, dict):
            json_candidates.append(raw_text[idx: idx + end].strip())

    seen = set()
    deduped_json_candidates: List[str] = []
    for candidate in json_candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            deduped_json_candidates.append(candidate)

    for candidate in deduped_json_candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            grade = _normalize_grade_token(parsed.get("grade", ""))
            reasoning = str(parsed.get("reasoning", "")).strip()
            if grade:
                return grade, reasoning

    # 2) Backward-compatible line parsing
    lines = raw_text.split("\n")
    grade_re = re.compile(r"^(?:FINAL\s+)?GRADE\s*:\s*([ABC])\b[.\s]*$", re.IGNORECASE)

    for idx in range(len(lines) - 1, -1, -1):
        stripped = lines[idx].strip().upper()
        if stripped in ("A", "B", "C"):
            reasoning = "\n".join(lines[:idx]).strip()
            return stripped, reasoning
        grade_match = grade_re.match(lines[idx].strip())
        if grade_match:
            reasoning = "\n".join(lines[:idx]).strip()
            return grade_match.group(1).upper(), reasoning
        for letter in ("A", "B", "C"):
            if stripped.startswith(f"{letter}:") or stripped.startswith(f"{letter} ") or stripped == f"GRADE: {letter}":
                reasoning = "\n".join(lines[:idx]).strip()
                return letter, reasoning

    # No valid grade line found.
    return "", raw_text


def _grade_letter_to_label(letter: str) -> str:
    """Map A/B/C to CORRECT/INCORRECT/NOT_ATTEMPTED."""
    return {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}.get(letter, "NOT_ATTEMPTED")

def _format_gold_targets_block(ground_truths: List[str]) -> str:
    """Format acceptable references as a newline bullet list for judge prompts."""
    cleaned = [g.strip() for g in ground_truths if isinstance(g, str) and g.strip()]
    if not cleaned:
        return "- (no reference provided)"
    return "\n".join(f"- {g}" for g in cleaned)


def _judge_response_format_for_provider(provider: str) -> Optional[Dict[str, Any]]:
    """Return API-level structured output mode for supported judge providers."""
    p = (provider or "").strip().lower()
    if p == "openai":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "judge_grade",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "reasoning": {"type": "string"},
                        "grade": {"type": "string", "enum": ["A", "B", "C"]},
                    },
                    "required": ["reasoning", "grade"],
                },
            },
        }
    if p in {"xai", "deepseek", "groq", "openrouter", "huggingface", "google"}:
        # OpenAI-compatible providers commonly support json_object mode.
        return {"type": "json_object"}
    return None


def _format_phase1_votes_block(
    judge_grades: Optional[List[Optional[str]]],
    judge_statuses: Optional[List[str]],
    judge_reasoning: Optional[List[str]],
) -> str:
    """Format Phase-1 panel metadata for adjudicator context."""
    grades = judge_grades or []
    statuses = judge_statuses or []
    reasoning = judge_reasoning or []
    n = max(len(grades), len(statuses), len(reasoning))
    if n == 0:
        return "- (no phase-1 judge outputs available)"

    lines: List[str] = []
    for idx in range(n):
        status = statuses[idx] if idx < len(statuses) else "UNKNOWN"
        grade = grades[idx] if idx < len(grades) else None
        grade_text = grade if grade is not None else "NO_SEMANTIC_GRADE"
        lines.append(f"- Judge {idx + 1}: status={status}, grade={grade_text}")
        if idx < len(reasoning) and reasoning[idx]:
            cleaned = " ".join(reasoning[idx].strip().split())
            if len(cleaned) > 400:
                cleaned = cleaned[:397] + "..."
            lines.append(f"  Reasoning: {cleaned}")
    return "\n".join(lines)


# ===========================================================================
# String matching
# ===========================================================================

def check_correctness_string(
    prediction: str,
    ground_truths: List[str],
    strip_articles: bool = True,
    max_length_ratio: float = 3.0
) -> CorrectnessResult:
    """
    Check if a prediction matches any ground truth using string matching.

    Matching rule:
    1. Exact match after normalization

    Args:
        prediction: The model's predicted answer
        ground_truths: List of acceptable ground truth answers
        strip_articles: Whether to strip articles during normalization
        max_length_ratio: Deprecated (kept for backward compatibility)

    Returns:
        CorrectnessResult with is_correct, match_type, and matched_answer
    """
    # Normalize prediction
    norm_pred = normalize(prediction, strip_articles)

    if not norm_pred:
        return CorrectnessResult(
            is_correct=False,
            match_type=None,
            matched_answer=None,
            grade="INCORRECT",
        )

    for gold in ground_truths:
        norm_gold = normalize(gold, strip_articles)

        if not norm_gold:
            continue

        # 1. Exact match
        if norm_pred == norm_gold:
            return CorrectnessResult(
                is_correct=True,
                match_type="exact",
                matched_answer=gold,
                grade="CORRECT",
            )

    # No match found
    return CorrectnessResult(
        is_correct=False,
        match_type=None,
        matched_answer=None,
    )


# ===========================================================================
# NLI matching (with probability storage)
# ===========================================================================

def check_correctness_nli(
    prediction: str,
    ground_truths: List[str],
    nli_judge: "NLISemanticJudge",
    question: str = "",
    entailment_threshold: float = 0.5
) -> CorrectnessResult:
    """
    Check if a prediction matches any ground truth using NLI.

    Now stores NLI probabilities in the result for auditability.
    """
    if not prediction or not prediction.strip():
        return CorrectnessResult(
            is_correct=False,
            match_type=None,
            matched_answer=None
        )

    # Add question context
    context_pred = f"Question: {question} Answer: {prediction}" if question else prediction

    for gold in ground_truths:
        if not gold or not gold.strip():
            continue

        context_gold = f"Question: {question} Answer: {gold}" if question else gold

        try:
            # 1. Check if prediction entails gold (prediction -> gold)
            prob = nli_judge._get_entailment_prob(context_pred, context_gold)
            # 2. Check reverse direction
            prob_reverse = nli_judge._get_entailment_prob(context_gold, context_pred)

            if prob >= entailment_threshold:
                logger.debug(f"NLI match: '{prediction}' -> '{gold}' (p={prob:.3f})")
                return CorrectnessResult(
                    is_correct=True,
                    match_type="nli_entailment",
                    matched_answer=gold,
                    grade="CORRECT",
                    nli_probs={"forward": round(prob, 4), "reverse": round(prob_reverse, 4)},
                )
            if prob_reverse >= entailment_threshold:
                logger.debug(f"NLI match (gold->pred): '{gold}' -> '{prediction}' (p={prob_reverse:.3f})")
                return CorrectnessResult(
                    is_correct=True,
                    match_type="nli_gold_entails_prediction",
                    matched_answer=gold,
                    grade="CORRECT",
                    nli_probs={"forward": round(prob, 4), "reverse": round(prob_reverse, 4)},
                )
        except Exception as e:
            logger.warning(f"NLI check failed for '{prediction}' vs '{gold}': {e}")
            continue

    return CorrectnessResult(
        is_correct=False,
        match_type=None,
        matched_answer=None
    )


# ===========================================================================
# LLM-as-judge (single judge, with CoT + position swap)
# ===========================================================================

def check_correctness_llm(
    prediction: str,
    ground_truths: List[str],
    question: str,
    inference_client: "MultiProviderClient",
    judge_provider: str = "openai",
    judge_model: str = "gpt-4o",
    max_new_tokens: int = 200,
) -> CorrectnessResult:
    """
    Check correctness using an LLM-as-judge with Chain-of-Thought grading.

    Sends the question, gold target, and predicted answer to a judge LLM.
    The judge explains its reasoning first, then outputs A/B/C.

    Args:
        prediction: The model's predicted answer
        ground_truths: List of acceptable ground truth answers
        question: The original question text
        inference_client: MultiProviderClient for making API calls
        judge_provider: Provider for the judge model
        judge_model: Model ID for the judge
        max_new_tokens: Max tokens for the judge response (increased for CoT)

    Returns:
        CorrectnessResult with grade and reasoning
    """
    if not prediction or not prediction.strip():
        return CorrectnessResult(is_correct=False, match_type=None, matched_answer=None, grade="INCORRECT")

    targets_block = _format_gold_targets_block(ground_truths)
    matched_reference = next((g for g in ground_truths if isinstance(g, str) and g.strip()), None)

    # --- Original-order call ---
    prompt_text = LLM_JUDGE_GRADING_PROMPT.format(
        question=question,
        targets_block=targets_block,
        predicted_answer=prediction,
    )

    try:
        response_format = _judge_response_format_for_provider(judge_provider)
        result = inference_client.generate_greedy(
            provider=judge_provider,
            model=judge_model,
            prompt=prompt_text,
            max_new_tokens=max_new_tokens,
            response_format=response_format,
        )
        grade_letter, reasoning = _parse_cot_grade(result.text)
    except Exception as e:
        logger.warning(f"LLM judge failed for '{prediction[:60]}...': {e}")
        return CorrectnessResult(
            is_correct=False,
            match_type="llm_judge_failed",
            matched_answer=None,
            is_unclear=True,
            grade="NOT_ATTEMPTED",
            judge_statuses=["API_FAILED"],
            judge_reasoning=[f"Judge exception: {e}"],
        )

    # Parse failure should never be converted into a guessed grade.
    if not grade_letter:
        return CorrectnessResult(
            is_correct=False,
            match_type="llm_judge_parse_failed",
            matched_answer=None,
            is_unclear=True,
            grade="NOT_ATTEMPTED",
            judge_statuses=["PARSE_FAILED"],
            judge_reasoning=[reasoning],
        )

    grade_label = _grade_letter_to_label(grade_letter)

    # Build result
    is_correct = (grade_label == "CORRECT")
    is_unclear = (grade_label == "NOT_ATTEMPTED")

    return CorrectnessResult(
        is_correct=is_correct,
        match_type="llm_judge" if is_correct else ("llm_judge_not_attempted" if is_unclear else None),
        matched_answer=matched_reference if is_correct else None,
        is_unclear=is_unclear,
        grade=grade_label,
        judge_statuses=["OK"],
        judge_reasoning=[reasoning],
    )


# ===========================================================================
# LLM-as-judge adjudicator (phase-2 fallback)
# ===========================================================================

def check_correctness_llm_adjudicator(
    prediction: str,
    ground_truths: List[str],
    question: str,
    inference_client: "MultiProviderClient",
    phase1_grades: Optional[List[Optional[str]]],
    phase1_statuses: Optional[List[str]],
    phase1_reasoning: Optional[List[str]],
    judge_provider: str = "openai",
    judge_model: str = "gpt-4o",
    max_new_tokens: int = 260,
) -> Tuple[Optional[str], str, str]:
    """
    Run a single adjudicator judge for unresolved ensemble outcomes.

    Returns:
        (grade_label, status, reasoning)
        - grade_label: CORRECT / INCORRECT / NOT_ATTEMPTED (or None on failure)
        - status: OK / API_FAILED / PARSE_FAILED
        - reasoning: adjudicator CoT text or error detail
    """
    targets_block = _format_gold_targets_block(ground_truths)
    phase1_votes_block = _format_phase1_votes_block(
        judge_grades=phase1_grades,
        judge_statuses=phase1_statuses,
        judge_reasoning=phase1_reasoning,
    )

    prompt_text = LLM_JUDGE_ADJUDICATOR_PROMPT.format(
        question=question,
        targets_block=targets_block,
        predicted_answer=prediction,
        phase1_votes_block=phase1_votes_block,
    )

    try:
        response_format = _judge_response_format_for_provider(judge_provider)
        result = inference_client.generate_greedy(
            provider=judge_provider,
            model=judge_model,
            prompt=prompt_text,
            max_new_tokens=max_new_tokens,
            response_format=response_format,
        )
        grade_letter, reasoning = _parse_cot_grade(result.text)
    except Exception as e:
        logger.warning(f"Adjudicator judge failed for '{prediction[:60]}...': {e}")
        return None, "API_FAILED", f"Adjudicator exception: {e}"

    if not grade_letter:
        return None, "PARSE_FAILED", reasoning

    return _grade_letter_to_label(grade_letter), "OK", reasoning


# ===========================================================================
# LLM-as-judge ensemble (multi-judge)
# ===========================================================================

def check_correctness_llm_ensemble(
    prediction: str,
    ground_truths: List[str],
    question: str,
    inference_client: "MultiProviderClient",
    judges: List[Dict[str, Any]],
    max_new_tokens: int = 200,
    failure_policy: str = "skip",
) -> CorrectnessResult:
    """
    Check correctness using multiple LLM judges with ensemble voting.

    Each judge grades with CoT reasoning. Failed judges are skipped by default
    instead of counting as INCORRECT votes.

    Args:
        prediction: The model's predicted answer
        ground_truths: List of acceptable ground truth answers
        question: The original question text
        inference_client: MultiProviderClient for API calls
        judges: List of {"provider": str, "model": str} per judge
        max_new_tokens: Max tokens for each judge response
        failure_policy: "skip" (drop failed judges) or "false" (count as INCORRECT)

    Returns:
        CorrectnessResult with ensemble-level grade and judge details
    """
    if not prediction or not prediction.strip():
        return CorrectnessResult(
            is_correct=False, match_type=None, matched_answer=None, judge_votes=None
        )
    if not judges:
        return CorrectnessResult(
            is_correct=False, match_type=None, matched_answer=None, judge_votes=None
        )

    matched_reference = next((g for g in ground_truths if isinstance(g, str) and g.strip()), None)

    votes: List[bool] = []
    vote_grades: List[str] = []
    all_grades: List[Optional[str]] = []
    all_statuses: List[str] = []
    all_reasoning: List[str] = []

    for i, j in enumerate(judges):
        provider = j.get("provider", "openai")
        model = j.get("model", "gpt-4o")
        try:
            res = check_correctness_llm(
                prediction=prediction,
                ground_truths=ground_truths,
                question=question,
                inference_client=inference_client,
                judge_provider=provider,
                judge_model=model,
                max_new_tokens=max_new_tokens,
            )
            if res.match_type == "llm_judge_failed":
                if failure_policy == "false":
                    vote_grades.append("INCORRECT")
                    votes.append(False)
                    all_grades.append(None)
                    all_statuses.append("API_FAILED")
                    all_reasoning.append((res.judge_reasoning or ["Judge failure"])[0])
                else:
                    all_grades.append(None)
                    all_statuses.append("API_FAILED")
                    all_reasoning.append((res.judge_reasoning or ["Judge failure"])[0])
                continue
            if res.match_type == "llm_judge_parse_failed":
                if failure_policy == "false":
                    vote_grades.append("INCORRECT")
                    votes.append(False)
                    all_grades.append(None)
                    all_statuses.append("PARSE_FAILED")
                    all_reasoning.append((res.judge_reasoning or ["Judge parse failure"])[0])
                else:
                    all_grades.append(None)
                    all_statuses.append("PARSE_FAILED")
                    all_reasoning.append((res.judge_reasoning or ["Judge parse failure"])[0])
                continue
            semantic_grade = res.grade or "NOT_ATTEMPTED"
            vote_grades.append(semantic_grade)
            votes.append(semantic_grade == "CORRECT")
            all_grades.append(semantic_grade)
            all_statuses.append("OK")
            all_reasoning.append((res.judge_reasoning or [""])[0])
        except Exception as e:
            logger.warning(f"Judge {i} ({provider}/{model}) failed: {e}")
            if failure_policy == "false":
                # Legacy behavior: count failure as INCORRECT
                vote_grades.append("INCORRECT")
                votes.append(False)
                all_grades.append(None)
                all_statuses.append("API_FAILED")
                all_reasoning.append(f"Exception: {e}")
            else:
                # "skip": exclude this judge from the vote entirely
                all_grades.append(None)
                all_statuses.append("API_FAILED")
                all_reasoning.append(f"Exception: {e}")
                continue

    # Strict-majority policy requires at least 2 valid semantic votes.
    if len(vote_grades) < 2:
        return CorrectnessResult(
            is_correct=False,
            match_type="llm_judge_ensemble",
            matched_answer=None,
            judge_votes=votes,
            is_unclear=True,
            grade="NOT_ATTEMPTED",
            judge_grades=all_grades,
            judge_statuses=all_statuses,
            judge_reasoning=all_reasoning,
            decision_source="UNRESOLVED",
        )

    grade_counts = {
        "CORRECT": vote_grades.count("CORRECT"),
        "INCORRECT": vote_grades.count("INCORRECT"),
        "NOT_ATTEMPTED": vote_grades.count("NOT_ATTEMPTED"),
    }
    majority_grade = max(grade_counts, key=grade_counts.get)
    majority_count = grade_counts[majority_grade]

    # Strict majority: winner must exceed half of all valid votes.
    if majority_count <= len(vote_grades) / 2:
        return CorrectnessResult(
            is_correct=False,
            match_type="llm_judge_ensemble",
            matched_answer=None,
            judge_votes=votes,
            is_unclear=True,
            grade="NOT_ATTEMPTED",
            judge_grades=all_grades,
            judge_statuses=all_statuses,
            judge_reasoning=all_reasoning,
            decision_source="UNRESOLVED",
        )

    is_correct = majority_grade == "CORRECT"
    is_unclear = majority_grade == "NOT_ATTEMPTED"

    return CorrectnessResult(
        is_correct=is_correct,
        match_type="llm_judge_ensemble",
        matched_answer=matched_reference if is_correct else None,
        judge_votes=votes,
        is_unclear=is_unclear,
        grade=majority_grade,
        judge_grades=all_grades,
        judge_statuses=all_statuses,
        judge_reasoning=all_reasoning,
        decision_source="MAJORITY",
    )


# ===========================================================================
# Main cascade
# ===========================================================================

def check_correctness(
    prediction: str,
    ground_truths: List[str],
    strip_articles: bool = True,
    max_length_ratio: float = 3.0,
    nli_judge: Optional["NLISemanticJudge"] = None,
    question: str = "",
    use_nli_fallback: bool = False,
    nli_entailment_threshold: float = 0.5,
    inference_client: Optional["MultiProviderClient"] = None,
    llm_judge_provider: str = "openai",
    llm_judge_model: str = "gpt-4o",
    use_llm_fallback: bool = False,
    llm_judge_ensemble: Optional[List[Dict[str, Any]]] = None,
    max_new_tokens: int = 200,
    failure_policy: str = "skip",
    adjudicator: Optional[Dict[str, Any]] = None,
    adjudicator_max_new_tokens: int = 260,
) -> CorrectnessResult:
    """
    Check if a prediction matches any ground truth (cascade approach).

    1. Uses LLM-as-judge grading when enabled.
    2. If LLM judging is disabled/unavailable, returns NOT_ATTEMPTED.

    Args:
        prediction: The model's predicted answer
        ground_truths: List of acceptable ground truth answers
        strip_articles: Deprecated for main correctness cascade (kept for API compatibility)
        max_length_ratio: Deprecated for main correctness cascade (kept for API compatibility)
        nli_judge: Deprecated for correctness cascade (kept for API compatibility)
        question: The original question (for LLM context)
        use_nli_fallback: Deprecated for correctness cascade (ignored)
        nli_entailment_threshold: Deprecated for correctness cascade (ignored)
        inference_client: Optional MultiProviderClient for LLM judge
        llm_judge_provider: Deprecated for main correctness cascade (kept for API compatibility)
        llm_judge_model: Deprecated for main correctness cascade (kept for API compatibility)
        use_llm_fallback: Whether to enable LLM-as-judge grading
        llm_judge_ensemble: Exactly 3 judge configs for ensemble
        max_new_tokens: Max tokens for LLM judge CoT
        failure_policy: "skip" or "false" for failed judges
        adjudicator: Optional {"provider": str, "model": str} for unresolved Phase-1 decisions
        adjudicator_max_new_tokens: Max tokens for adjudicator CoT

    Returns:
        CorrectnessResult with is_correct, match_type, grade, etc.
    """
    # NLI fallback for correctness is intentionally disabled. We keep the
    # parameters for backward compatibility with existing call sites.
    if use_nli_fallback and nli_judge is not None:
        logger.debug(
            "Correctness NLI fallback is deprecated and ignored; "
            "using LLM-judge-only correctness."
        )

    # Step 1: Try LLM-as-judge (most accurate, paid) if available
    if use_llm_fallback and inference_client is not None:
        if not llm_judge_ensemble:
            logger.warning("Correctness requires a 3-judge ensemble, but none was provided.")
            return CorrectnessResult(
                is_correct=False,
                match_type="no_correctness_judge",
                matched_answer=None,
                is_unclear=True,
                grade="NOT_ATTEMPTED",
                decision_source="NO_JUDGE",
            )
        if len(llm_judge_ensemble) != 3:
            logger.warning(
                "Correctness requires exactly 3 judges; got %d.",
                len(llm_judge_ensemble),
            )
            return CorrectnessResult(
                is_correct=False,
                match_type="no_correctness_judge",
                matched_answer=None,
                is_unclear=True,
                grade="NOT_ATTEMPTED",
                decision_source="NO_JUDGE",
            )
        ensemble_result = check_correctness_llm_ensemble(
            prediction=prediction,
            ground_truths=ground_truths,
            question=question,
            inference_client=inference_client,
            judges=llm_judge_ensemble,
            max_new_tokens=max_new_tokens,
            failure_policy=failure_policy,
        )
        if ensemble_result.decision_source != "UNRESOLVED":
            return ensemble_result

        if not adjudicator:
            return ensemble_result

        adjudicator_provider = adjudicator.get("provider", "openai")
        adjudicator_model = adjudicator.get("model", "gpt-4o")
        adjudicator_grade, adjudicator_status, adjudicator_reasoning = check_correctness_llm_adjudicator(
            prediction=prediction,
            ground_truths=ground_truths,
            question=question,
            inference_client=inference_client,
            phase1_grades=ensemble_result.judge_grades,
            phase1_statuses=ensemble_result.judge_statuses,
            phase1_reasoning=ensemble_result.judge_reasoning,
            judge_provider=adjudicator_provider,
            judge_model=adjudicator_model,
            max_new_tokens=adjudicator_max_new_tokens,
        )

        if adjudicator_status != "OK" or adjudicator_grade is None:
            ensemble_result.adjudicator_status = adjudicator_status
            ensemble_result.adjudicator_reasoning = adjudicator_reasoning
            return ensemble_result

        matched_reference = next((g for g in ground_truths if isinstance(g, str) and g.strip()), None)
        is_correct = adjudicator_grade == "CORRECT"
        is_unclear = adjudicator_grade == "NOT_ATTEMPTED"
        return CorrectnessResult(
            is_correct=is_correct,
            match_type="llm_judge_ensemble",
            matched_answer=matched_reference if is_correct else None,
            judge_votes=ensemble_result.judge_votes,
            is_unclear=is_unclear,
            grade=adjudicator_grade,
            judge_grades=ensemble_result.judge_grades,
            judge_statuses=ensemble_result.judge_statuses,
            judge_reasoning=ensemble_result.judge_reasoning,
            decision_source="ADJUDICATOR",
            adjudicator_grade=adjudicator_grade,
            adjudicator_status=adjudicator_status,
            adjudicator_reasoning=adjudicator_reasoning,
        )

    # No LLM judge available -> no correctness decision should be forced.
    return CorrectnessResult(
        is_correct=False,
        match_type="no_correctness_judge",
        matched_answer=None,
        is_unclear=True,
        grade="NOT_ATTEMPTED",
        decision_source="NO_JUDGE",
    )


# ===========================================================================
# Answer extraction helper
# ===========================================================================

def extract_answer_from_response(response: str) -> str:
    """
    Extract the answer from a model response.

    Handles common response patterns:
    - Direct answer
    - "The answer is X"
    - "X is the answer"
    - Responses with explanation after the answer

    Args:
        response: Raw model response

    Returns:
        Extracted answer string
    """
    if not response:
        return ""

    # Clean up the response
    text = response.strip()

    # Take first line if multi-line (often the answer is first)
    lines = text.split('\n')
    text = lines[0].strip()

    # Try to extract if wrapped in common patterns
    answer_patterns = [
        r"(?:the\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)",
        r"answer:\s*(.+?)(?:\.|$)",
        r"^(.+?)(?:\s+is the answer)",
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # If no pattern matched, take the text up to the first period
    if "." in text:
        text = text.split(".")[0].strip()

    return text


if __name__ == "__main__":
    # Test cases
    logging.basicConfig(level=logging.INFO)
    print("Testing correctness module...\n")

    test_cases = [
        # (prediction, ground_truths, expected_correct)
        ("Paris", ["Paris", "paris"], True),
        ("The answer is Paris", ["Paris"], True),
        ("Paris, France", ["Paris"], True),
        ("Lyon", ["Paris"], False),
        ("Barack Obama", ["Barack Hussein Obama", "Obama"], True),
        ("the Beatles", ["Beatles", "The Beatles"], True),
    ]

    print("String matching tests:")
    for pred, truths, expected in test_cases:
        result = check_correctness_string(pred, truths)
        status = "PASS" if result.is_correct == expected else "FAIL"
        print(f"  [{status}] '{pred}' vs {truths}: {result.is_correct} ({result.match_type})")

    print("\n" + "="*60)
    print("To test NLI-based correctness, run with --nli flag")
    print("="*60)
