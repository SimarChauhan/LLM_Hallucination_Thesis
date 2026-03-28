"""
Error classification with full distribution tracking for sensitivity analysis.

Five-category labeling scheme (expanded from four):
- reliably_correct:      greedy correct AND stochastic samples consistent
- fragile_correct:       greedy correct BUT stochastic samples inconsistent
- self_consistent_error: greedy incorrect AND stochastic samples consistent (hallucination)
- inconsistent_error:    greedy incorrect AND stochastic samples inconsistent
- not_attempted:         judge returned NOT_ATTEMPTED (tracked separately per research)
"""

from typing import List, Dict, Literal, Optional

from .schemas import EquivalenceStats, EquivalenceJudgment, ErrorLabel


def compute_equivalence_stats(
    judgments: List[EquivalenceJudgment]
) -> EquivalenceStats:
    """
    Compute statistics from a list of equivalence judgments.

    Args:
        judgments: List of "same", "different", "unclear" judgments

    Returns:
        EquivalenceStats with counts and total
    """
    num_same = sum(1 for j in judgments if j == "same")
    num_different = sum(1 for j in judgments if j == "different")
    num_unclear = sum(1 for j in judgments if j == "unclear")

    return EquivalenceStats(
        num_same=num_same,
        num_different=num_different,
        num_unclear=num_unclear,
        total=len(judgments)
    )


def classify_error(
    is_correct: bool,
    equivalence_stats: EquivalenceStats,
    threshold: float = 0.9,
    unclear_treatment: Literal["exclude", "count_as_different"] = "exclude",
    grade: Optional[str] = None,
) -> ErrorLabel:
    """
    Classify a result based on correctness and semantic equivalence.

    Five-category scheme:
    - correct + consistent     -> reliably_correct
    - correct + inconsistent   -> fragile_correct
    - incorrect + consistent   -> self_consistent_error
    - incorrect + inconsistent -> inconsistent_error
    - NOT_ATTEMPTED            -> not_attempted  (new: tracked separately)

    Args:
        is_correct: Whether the greedy answer was correct
        equivalence_stats: Statistics about equivalence across samples
        threshold: Minimum equivalence ratio for "consistent" classification
        unclear_treatment: How to handle "unclear" judgments:
            - "exclude": Don't count unclear in the ratio (default)
            - "count_as_different": Treat unclear as different
        grade: Optional raw grade from the judge ("CORRECT", "INCORRECT",
               "NOT_ATTEMPTED").  When grade is "NOT_ATTEMPTED", the record
               is labeled "not_attempted" regardless of correctness/consistency.

    Returns:
        ErrorLabel with classification and stats
    """
    # NEW: if the correctness judge said NOT_ATTEMPTED, label separately
    if grade == "NOT_ATTEMPTED":
        return ErrorLabel(
            label="not_attempted",
            equivalence_stats=equivalence_stats,
            threshold_used=threshold,
        )

    # Compute equivalence ratio based on treatment
    if unclear_treatment == "exclude":
        ratio = equivalence_stats.equivalence_ratio
    else:  # count_as_different
        ratio = equivalence_stats.equivalence_ratio_with_unclear

    consistent = ratio >= threshold

    if is_correct:
        label = "reliably_correct" if consistent else "fragile_correct"
    else:
        label = "self_consistent_error" if consistent else "inconsistent_error"

    return ErrorLabel(
        label=label,
        equivalence_stats=equivalence_stats,
        threshold_used=threshold
    )


def classify_at_multiple_thresholds(
    is_correct: bool,
    equivalence_stats: EquivalenceStats,
    thresholds: List[float] = [1.0, 0.9, 0.8, 0.7],
    unclear_treatment: Literal["exclude", "count_as_different"] = "exclude",
    grade: Optional[str] = None,
) -> Dict[float, str]:
    """
    Classify a result at multiple thresholds for sensitivity analysis.

    Args:
        is_correct: Whether the greedy answer was correct
        equivalence_stats: Statistics about equivalence across samples
        thresholds: List of thresholds to evaluate
        unclear_treatment: How to handle "unclear" judgments
        grade: Optional raw grade from the judge

    Returns:
        Dictionary mapping threshold to label
    """
    results = {}

    for threshold in thresholds:
        error_label = classify_error(
            is_correct,
            equivalence_stats,
            threshold,
            unclear_treatment,
            grade=grade,
        )
        results[threshold] = error_label.label

    return results


if __name__ == "__main__":
    # Test the labeling module
    print("Testing labeling module...\n")

    # Test cases with different distributions
    test_cases = [
        # (judgments, is_correct, grade, expected_label_at_0.9)
        (["same"] * 10, True, "CORRECT", "reliably_correct"),
        (["same"] * 10, False, "INCORRECT", "self_consistent_error"),
        (["same"] * 9 + ["different"], False, "INCORRECT", "self_consistent_error"),  # 90%
        (["same"] * 8 + ["different"] * 2, False, "INCORRECT", "inconsistent_error"),  # 80%
        (["same"] * 7 + ["unclear"] * 3, False, "INCORRECT", "self_consistent_error"),  # 100% excl unclear
        (["different"] * 10, False, "INCORRECT", "inconsistent_error"),
        (["same"] * 10, False, "NOT_ATTEMPTED", "not_attempted"),  # NEW: NOT_ATTEMPTED
        (["different"] * 10, False, "NOT_ATTEMPTED", "not_attempted"),  # NEW: NOT_ATTEMPTED overrides
    ]

    print("Classification tests (threshold=0.9, unclear=exclude):")
    for judgments, is_correct, grade, expected in test_cases:
        stats = compute_equivalence_stats(judgments)
        result = classify_error(is_correct, stats, threshold=0.9, grade=grade)
        status = "PASS" if result.label == expected else "FAIL"
        print(f"  [{status}] {stats.num_same}s/{stats.num_different}d/{stats.num_unclear}u, "
              f"correct={is_correct}, grade={grade} -> {result.label} "
              f"(ratio={stats.equivalence_ratio:.2f})")

    print("\nMulti-threshold test:")
    stats = compute_equivalence_stats(["same"] * 8 + ["different"] * 2)
    labels = classify_at_multiple_thresholds(False, stats, grade="INCORRECT")
    print(f"  Stats: {stats.num_same}s/{stats.num_different}d/{stats.num_unclear}u")
    for threshold, label in sorted(labels.items(), reverse=True):
        print(f"  threshold={threshold}: {label}")
