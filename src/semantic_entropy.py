"""Semantic-entropy computation over stochastic QA samples."""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .nli_judge import NLISemanticJudge


@dataclass
class SemanticEntropyResult:
    """Discrete semantic-entropy summary for one question's samples."""

    entropy: float
    entropy_norm: float
    n_clusters: int
    cluster_ids: List[int]
    cluster_sizes: List[int]
    n_samples: int


def _find(parent: List[int], idx: int) -> int:
    while parent[idx] != idx:
        parent[idx] = parent[parent[idx]]
        idx = parent[idx]
    return idx


def _union(parent: List[int], rank: List[int], a: int, b: int) -> None:
    root_a = _find(parent, a)
    root_b = _find(parent, b)
    if root_a == root_b:
        return
    if rank[root_a] < rank[root_b]:
        parent[root_a] = root_b
    elif rank[root_a] > rank[root_b]:
        parent[root_b] = root_a
    else:
        parent[root_b] = root_a
        rank[root_a] += 1


def _extract_pair_label(value: Any) -> str:
    if isinstance(value, str):
        label = value
    elif isinstance(value, dict):
        label = value.get("label") or value.get("judgment") or ""
    else:
        label = getattr(value, "label", None) or getattr(value, "judgment", "") or ""
    label = str(label).strip().lower()
    if label in {"same", "different", "unclear"}:
        return label
    return "unclear"


def compute_semantic_entropy(
    question: str,
    sample_answers: List[str],
    nli_judge: Optional["NLISemanticJudge"] = None,
    pair_judgments: Optional[Dict[Tuple[int, int], Any]] = None,
) -> SemanticEntropyResult:
    """
    Compute discrete semantic entropy over stochastic samples.

    Samples are clustered by pairwise bidirectional-NLI equivalence
    (judgment == "same"), and entropy is computed over cluster frequencies.
    """
    n_samples = len(sample_answers)
    if n_samples == 0:
        return SemanticEntropyResult(
            entropy=0.0,
            entropy_norm=0.0,
            n_clusters=0,
            cluster_ids=[],
            cluster_sizes=[],
            n_samples=0,
        )

    parent = list(range(n_samples))
    rank = [0] * n_samples

    for left in range(n_samples):
        ans_left = (sample_answers[left] or "").strip()
        if not ans_left:
            continue
        for right in range(left + 1, n_samples):
            ans_right = (sample_answers[right] or "").strip()
            if not ans_right:
                continue
            if pair_judgments is not None:
                pair_value = pair_judgments.get((left, right), pair_judgments.get((right, left)))
                judgment = _extract_pair_label(pair_value)
            else:
                if nli_judge is None:
                    raise ValueError("compute_semantic_entropy requires nli_judge when pair_judgments is not provided.")
                pair_result = nli_judge.judge_equivalence(question, ans_left, ans_right)
                judgment = _extract_pair_label(pair_result)
            if judgment == "same":
                _union(parent, rank, left, right)

    root_to_cluster: Dict[int, int] = {}
    cluster_ids: List[int] = []
    cluster_sizes_dict: Dict[int, int] = {}

    for idx in range(n_samples):
        root = _find(parent, idx)
        cluster_id = root_to_cluster.setdefault(root, len(root_to_cluster))
        cluster_ids.append(cluster_id)
        cluster_sizes_dict[cluster_id] = cluster_sizes_dict.get(cluster_id, 0) + 1

    n_clusters = len(root_to_cluster)
    cluster_sizes = [cluster_sizes_dict[cid] for cid in range(n_clusters)]
    probabilities = [size / n_samples for size in cluster_sizes if size > 0]
    entropy = float(-sum(prob * math.log(prob) for prob in probabilities))
    entropy_norm = float(entropy / math.log(n_samples)) if n_samples > 1 else 0.0

    return SemanticEntropyResult(
        entropy=entropy,
        entropy_norm=entropy_norm,
        n_clusters=n_clusters,
        cluster_ids=cluster_ids,
        cluster_sizes=cluster_sizes,
        n_samples=n_samples,
    )


def classify_error_by_entropy(
    is_correct: bool,
    entropy: Optional[float],
    entropy_norm: Optional[float],
    entropy_threshold: float,
    use_normalized_entropy: bool = True,
    grade: Optional[str] = None,
) -> Optional[str]:
    """
    Map correctness + semantic entropy to the existing 5-category labels.
    """
    if grade == "NOT_ATTEMPTED":
        return "not_attempted"

    score = entropy_norm if use_normalized_entropy else entropy
    if score is None:
        return None

    is_consistent = score <= entropy_threshold

    if is_correct:
        return "reliably_correct" if is_consistent else "fragile_correct"
    return "self_consistent_error" if is_consistent else "inconsistent_error"
