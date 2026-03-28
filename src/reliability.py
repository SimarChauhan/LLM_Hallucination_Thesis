"""
Inter-rater reliability metrics for LLM judge ensembles.

Computes Krippendorff's alpha (nominal) to quantify how well multiple
judges agree.  Recommended target: alpha >= 0.75 for production use.

References:
- Krippendorff, K. (2011). Computing Krippendorff's Alpha-Reliability.
- "Can You Trust LLM Judgments?" (arXiv 2412.12509)
"""

import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


def krippendorff_alpha_nominal(
    ratings_matrix: List[List[Optional[str]]],
) -> float:
    """
    Compute Krippendorff's alpha for nominal data.

    Args:
        ratings_matrix: list of items, each containing a list of rater judgments.
            Each inner list has one entry per rater.
            ``None`` = missing rating (rater was skipped / failed).
            String values like "CORRECT", "INCORRECT", "NOT_ATTEMPTED".

    Returns:
        Alpha coefficient (float).  1.0 = perfect agreement, 0.0 = chance,
        negative = worse than chance.  Returns 0.0 if there is insufficient data.
    """
    # Collect all observed values
    all_values = set()
    for item_ratings in ratings_matrix:
        for r in item_ratings:
            if r is not None:
                all_values.add(r)

    if len(all_values) <= 1:
        # If all ratings are the same value (or no data), alpha is undefined;
        # conventionally return 1.0 for perfect (trivial) agreement.
        return 1.0

    # Build coincidence matrix
    value_list = sorted(all_values)
    value_to_idx = {v: i for i, v in enumerate(value_list)}
    n_values = len(value_list)
    coincidence = [[0.0] * n_values for _ in range(n_values)]

    total_pairs = 0.0

    for item_ratings in ratings_matrix:
        # Get non-None ratings for this item
        observed = [r for r in item_ratings if r is not None]
        n_u = len(observed)
        if n_u < 2:
            continue  # Need at least 2 raters for this item

        # For each pair of ratings within this item
        for i in range(n_u):
            for j in range(n_u):
                if i == j:
                    continue
                ci = value_to_idx[observed[i]]
                cj = value_to_idx[observed[j]]
                coincidence[ci][cj] += 1.0 / (n_u - 1)
                total_pairs += 1.0 / (n_u - 1)

    if total_pairs == 0:
        return 0.0

    # Compute observed disagreement (D_o)
    d_o = 0.0
    for c in range(n_values):
        for k in range(n_values):
            if c != k:
                d_o += coincidence[c][k]
    d_o /= total_pairs

    # Compute expected disagreement (D_e)
    # Marginal frequencies
    n_c = [sum(coincidence[c][k] for k in range(n_values)) for c in range(n_values)]
    n_total = sum(n_c)

    if n_total <= 1:
        return 0.0

    d_e = 0.0
    for c in range(n_values):
        for k in range(n_values):
            if c != k:
                d_e += n_c[c] * n_c[k]
    d_e /= (n_total * (n_total - 1))

    if d_e == 0:
        return 1.0  # Perfect agreement

    alpha = 1.0 - (d_o / d_e)
    return alpha


def compute_ensemble_reliability(
    all_judge_grades: List[List[Optional[str]]],
) -> Dict[str, Any]:
    """
    Compute reliability metrics across all ensemble judgments in a re-eval run.

    Args:
        all_judge_grades: List of per-item grade lists.
            Each inner list has one grade per judge (str or None).
            Example: [["CORRECT", "CORRECT", "INCORRECT"], ["INCORRECT", None, "INCORRECT"], ...]
            ``None`` should be used for infrastructure failures (API/parse).

    Returns:
        {
            "krippendorff_alpha": float,
            "n_items": int,
            "n_raters": int,
            "pairwise_agreement": float,   # fraction of items where all raters agree
            "grade_distribution": dict,     # count per grade value
        }
    """
    if not all_judge_grades:
        return {
            "krippendorff_alpha": 0.0,
            "n_items": 0,
            "n_raters": 0,
            "pairwise_agreement": 0.0,
            "grade_distribution": {},
        }

    # Compute alpha
    alpha = krippendorff_alpha_nominal(all_judge_grades)

    # Number of raters (max across items)
    n_raters = max(len(g) for g in all_judge_grades) if all_judge_grades else 0

    # Pairwise agreement: fraction of items where all non-None ratings are the same
    n_agree = 0
    n_with_ratings = 0
    grade_counts: Dict[str, int] = {}

    for item_grades in all_judge_grades:
        # "FAILED" is kept for backward compatibility with older records.
        observed = [g for g in item_grades if g is not None and g != "FAILED"]
        for g in observed:
            grade_counts[g] = grade_counts.get(g, 0) + 1
        if len(observed) >= 2:
            n_with_ratings += 1
            if len(set(observed)) == 1:
                n_agree += 1

    pairwise_agreement = n_agree / n_with_ratings if n_with_ratings > 0 else 0.0

    return {
        "krippendorff_alpha": round(alpha, 4),
        "n_items": len(all_judge_grades),
        "n_raters": n_raters,
        "pairwise_agreement": round(pairwise_agreement, 4),
        "grade_distribution": grade_counts,
    }


if __name__ == "__main__":
    # Quick sanity tests
    print("Testing reliability module...\n")

    # Perfect agreement
    perfect = [["CORRECT", "CORRECT", "CORRECT"]] * 10
    r = compute_ensemble_reliability(perfect)
    print(f"Perfect agreement: alpha={r['krippendorff_alpha']}, agreement={r['pairwise_agreement']}")

    # No agreement
    no_agree = [
        ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"],
        ["INCORRECT", "CORRECT", "NOT_ATTEMPTED"],
        ["NOT_ATTEMPTED", "INCORRECT", "CORRECT"],
    ] * 10
    r = compute_ensemble_reliability(no_agree)
    print(f"No agreement: alpha={r['krippendorff_alpha']}, agreement={r['pairwise_agreement']}")

    # Partial agreement
    partial = [["CORRECT", "CORRECT", "INCORRECT"]] * 10
    r = compute_ensemble_reliability(partial)
    print(f"Partial agreement: alpha={r['krippendorff_alpha']}, agreement={r['pairwise_agreement']}")

    # With missing ratings
    with_missing = [["CORRECT", None, "CORRECT"], ["INCORRECT", "INCORRECT", None]] * 5
    r = compute_ensemble_reliability(with_missing)
    print(f"With missing: alpha={r['krippendorff_alpha']}, agreement={r['pairwise_agreement']}")
