"""Central dataclasses for type consistency across all modules."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime, timezone


@dataclass
class Question:
    """Represents a question from the dataset."""
    id: str
    text: str
    ground_truths: List[str]  # All acceptable answers
    category: Optional[str] = None


@dataclass
class GenerationParams:
    """Parameters used for text generation."""
    do_sample: bool
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    max_new_tokens: int


@dataclass
class GenerationResult:
    """Result of a single generation."""
    text: str
    params: GenerationParams
    logprobs: Optional[List[float]] = None  # Optional, not all APIs provide
    request_meta: Optional[Dict[str, Any]] = None  # latency, retries, finish reason, truncation, etc.


@dataclass
class CorrectnessResult:
    """Result of correctness checking.

    Enhanced per research best practices:
    - ``grade`` preserves the raw judge verdict (CORRECT / INCORRECT / NOT_ATTEMPTED)
      instead of collapsing NOT_ATTEMPTED into ``is_correct=False``.
    - ``judge_grades`` / ``judge_reasoning`` store per-judge Chain-of-Thought
      outputs for the ensemble, enabling transparency and auditing.
    - ``nli_probs`` captures NLI entailment probabilities when the NLI
      cascade step was used, so thresholds can be recalibrated without re-running.
    """
    is_correct: bool
    match_type: Optional[str] = None
    # "exact", "prediction_contains_gold", "gold_contains_prediction",
    # "nli_entailment", "nli_gold_entails_prediction",
    # "llm_judge", "llm_judge_ensemble", "llm_judge_not_attempted"
    matched_answer: Optional[str] = None
    judge_votes: Optional[List[bool]] = None  # ensemble: one bool per judge
    is_unclear: bool = False  # True when judge(s) returned NOT_ATTEMPTED

    # --- New fields (Tier 1 + 2) ---
    grade: Optional[str] = None  # "CORRECT", "INCORRECT", "NOT_ATTEMPTED"
    judge_grades: Optional[List[Optional[str]]] = None  # per-judge semantic grade (None for infra failures)
    judge_statuses: Optional[List[str]] = None  # per-judge status: OK/API_FAILED/PARSE_FAILED
    judge_reasoning: Optional[List[str]] = None  # per-judge CoT text (ensemble)
    decision_source: Optional[str] = None  # "MAJORITY", "ADJUDICATOR", "UNRESOLVED", "NO_JUDGE"
    adjudicator_grade: Optional[str] = None  # semantic grade returned by adjudicator (if used)
    adjudicator_status: Optional[str] = None  # "OK", "API_FAILED", "PARSE_FAILED", or None
    adjudicator_reasoning: Optional[str] = None  # CoT text from adjudicator
    nli_probs: Optional[Dict[str, float]] = None  # {"forward": float, "reverse": float}
    position_consistent: Optional[bool] = None  # True if both orders agree


@dataclass
class NLIEquivalenceResult:
    """Rich result from a single NLI equivalence comparison.

    Stores the bidirectional entailment probabilities alongside the
    categorical judgment so downstream consumers can re-threshold,
    calibrate, or audit without re-running NLI.
    """
    judgment: str  # "same", "different", "unclear"
    prob_forward: float  # P(A -> B)
    prob_reverse: float  # P(B -> A)


@dataclass
class EquivalenceStats:
    """Statistics about semantic equivalence across samples."""
    num_same: int
    num_different: int
    num_unclear: int
    total: int
    nli_probs: Optional[List[Dict[str, Any]]] = None  # per-sample probs for auditing

    @property
    def equivalence_ratio(self) -> float:
        """Ratio of 'same' judgments, excluding unclear (default behavior)."""
        denominator = self.num_same + self.num_different
        return self.num_same / denominator if denominator > 0 else 0.0

    @property
    def equivalence_ratio_with_unclear(self) -> float:
        """Ratio treating unclear as different."""
        return self.num_same / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            "num_same": self.num_same,
            "num_different": self.num_different,
            "num_unclear": self.num_unclear,
            "total": self.total,
        }
        if self.nli_probs is not None:
            d["nli_probs"] = self.nli_probs
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "EquivalenceStats":
        """Create from dictionary (backward compatible)."""
        return cls(
            num_same=data["num_same"],
            num_different=data["num_different"],
            num_unclear=data["num_unclear"],
            total=data["total"],
            nli_probs=data.get("nli_probs"),
        )


EquivalenceJudgment = Literal["same", "different", "unclear"]


@dataclass
class ErrorLabel:
    """Classification of a result (5-category scheme)."""
    label: Literal[
        "reliably_correct", "fragile_correct",
        "self_consistent_error", "inconsistent_error",
        "not_attempted",
        # Backward compatibility
        "correct",
    ]
    equivalence_stats: EquivalenceStats
    threshold_used: float


@dataclass
class ResultRecord:
    """Complete result record for a question-model pair."""
    question_id: str
    question: str
    ground_truth: List[str]
    model: str
    greedy_answer: str
    greedy_correct: Optional[bool]  # None means not yet evaluated (Phase 1 raw data)
    correctness_match_type: Optional[str]
    stochastic_answers: Optional[List[str]]
    equivalence_results: Optional[List[str]]  # "same", "different", "unclear"
    equivalence_stats: Optional[EquivalenceStats]
    equivalence_ratio: Optional[float]
    # Labels at different thresholds for sensitivity analysis
    error_label_1_0: Optional[str] = None
    error_label_0_9: Optional[str] = None
    error_label_0_8: Optional[str] = None
    error_label_0_7: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    # --- New fields (Tier 1 + 2) ---
    correctness_unclear: Optional[bool] = None
    correctness_grade: Optional[str] = None           # "CORRECT"/"INCORRECT"/"NOT_ATTEMPTED"
    correctness_judge_grades: Optional[List[Optional[str]]] = None  # per-judge semantic grades
    correctness_judge_statuses: Optional[List[str]] = None  # per-judge status: OK/API_FAILED/PARSE_FAILED
    correctness_judge_reasoning: Optional[List[str]] = None  # CoT text from judges
    correctness_decision_source: Optional[str] = None  # "MAJORITY", "ADJUDICATOR", "UNRESOLVED", "NO_JUDGE"
    correctness_adjudicator_grade: Optional[str] = None  # semantic adjudicator grade
    correctness_adjudicator_status: Optional[str] = None  # adjudicator status code
    correctness_adjudicator_reasoning: Optional[str] = None  # adjudicator reasoning text
    correctness_nli_probs: Optional[Dict[str, float]] = None  # NLI probs used in correctness
    correctness_position_consistent: Optional[bool] = None
    greedy_correctness_rejudged: Optional[bool] = None  # whether greedy correctness was re-judged in this run
    greedy_correctness_source: Optional[str] = None  # JUDGED / PRECOMPUTED / MISSING
    nli_equiv_probs: Optional[List[Dict[str, Any]]] = None  # per-sample NLI probs for equivalence
    equivalence_decision_source: Optional[List[str]] = None  # per-sample decision source (NLI/LLM)
    equivalence_decision_source_detail: Optional[List[str]] = None  # per-sample detailed source reason
    equivalence_results_nli: Optional[List[str]] = None  # legacy NLI-only baseline judgments
    equivalence_stats_nli: Optional[Dict[str, Any]] = None  # legacy NLI-only baseline stats
    equivalence_ratio_nli: Optional[float] = None  # legacy NLI-only ratio
    nli_equiv_probs_nli: Optional[List[Dict[str, Any]]] = None  # legacy NLI-only per-sample probs
    error_label_nli_1_0: Optional[str] = None  # baseline label at threshold 1.0 (NLI-only)
    error_label_nli_0_9: Optional[str] = None  # baseline label at threshold 0.9 (NLI-only)
    error_label_nli_0_8: Optional[str] = None  # baseline label at threshold 0.8 (NLI-only)
    error_label_nli_0_7: Optional[str] = None  # baseline label at threshold 0.7 (NLI-only)
    stochastic_sample_grades: Optional[List[str]] = None  # CORRECT/INCORRECT/NOT_ATTEMPTED per sample
    stochastic_sample_grade_source: Optional[List[str]] = None  # NLI/LLM per sample
    stochastic_sample_grade_source_detail: Optional[List[str]] = None  # detailed source reason per sample
    stochastic_sample_grade_confidence: Optional[List[Dict[str, Any]]] = None  # p_max + matched target metadata
    stochastic_correct_rate: Optional[float] = None
    stochastic_scored_n: Optional[int] = None
    stochastic_not_attempted_n: Optional[int] = None
    different_scored_n: Optional[int] = None
    different_correct_n: Optional[int] = None
    p_correct_given_different: Optional[float] = None
    semantic_pair_decisions: Optional[List[Dict[str, Any]]] = None  # sample-sample hybrid pair labels/sources
    semantic_entropy: Optional[float] = None  # Shannon entropy (nats) over semantic clusters
    semantic_entropy_norm: Optional[float] = None  # entropy normalized by log(n_samples)
    n_semantic_clusters: Optional[int] = None
    semantic_cluster_ids: Optional[List[int]] = None  # per-sample cluster id
    semantic_cluster_sizes: Optional[List[int]] = None  # one size per cluster id
    semantic_entropy_label: Optional[str] = None  # 5-way label using entropy threshold
    hybrid_enabled: Optional[bool] = None
    hybrid_thresholds: Optional[Dict[str, float]] = None
    hybrid_calibration_id: Optional[str] = None
    hybrid_judge_model: Optional[str] = None
    inter_rater_alpha: Optional[float] = None          # Krippendorff's alpha (for ensemble)
    # --- High-rigor protocol/provenance fields (v3) ---
    run_id: Optional[str] = None
    protocol_version: Optional[str] = None
    config_hash: Optional[str] = None
    prompt_version: Optional[str] = None
    run_date: Optional[str] = None  # UTC date (YYYY-MM-DD) when the run started
    dataset_name: Optional[str] = None
    dataset_split: Optional[str] = None
    dataset_item_hash: Optional[str] = None
    model_provider: Optional[str] = None
    model_id: Optional[str] = None
    model_snapshot_id: Optional[str] = None  # Dated or immutable snapshot identifier
    model_release_date: Optional[str] = None  # YYYY-MM-DD (model release date)
    model_track: Optional[str] = None  # version-evolution track name
    model_family: Optional[str] = None  # model family (openai/anthropic/xai/qwen/...)
    model_version_index: Optional[int] = None  # ordered index within model_track
    stochastic_target_n: Optional[int] = None
    stochastic_actual_n: Optional[int] = None
    is_incomplete: Optional[bool] = None
    sample_metadata: Optional[List[Dict[str, Any]]] = None  # one dict per stochastic sample
    judge_protocol: Optional[str] = None
    judge_repeat_consistency: Optional[float] = None
    escalated_to_human: Optional[bool] = None
    contamination_flag: Optional[bool] = None
    contamination_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "question_id": self.question_id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "model": self.model,
            "greedy_answer": self.greedy_answer,
            "greedy_correct": self.greedy_correct,
            "correctness_match_type": self.correctness_match_type,
            "stochastic_answers": self.stochastic_answers,
            "equivalence_results": self.equivalence_results,
            "equivalence_stats": self.equivalence_stats.to_dict() if self.equivalence_stats else None,
            "equivalence_ratio": self.equivalence_ratio,
            "error_label_1.0": self.error_label_1_0,
            "error_label_0.9": self.error_label_0_9,
            "error_label_0.8": self.error_label_0_8,
            "error_label_0.7": self.error_label_0_7,
            "timestamp": self.timestamp
        }
        # Include new fields only when populated (avoids bloating old records)
        _optional = {
            "correctness_unclear": self.correctness_unclear,
            "correctness_grade": self.correctness_grade,
            "correctness_judge_grades": self.correctness_judge_grades,
            "correctness_judge_statuses": self.correctness_judge_statuses,
            "correctness_judge_reasoning": self.correctness_judge_reasoning,
            "correctness_decision_source": self.correctness_decision_source,
            "correctness_adjudicator_grade": self.correctness_adjudicator_grade,
            "correctness_adjudicator_status": self.correctness_adjudicator_status,
            "correctness_adjudicator_reasoning": self.correctness_adjudicator_reasoning,
            "correctness_nli_probs": self.correctness_nli_probs,
            "correctness_position_consistent": self.correctness_position_consistent,
            "greedy_correctness_rejudged": self.greedy_correctness_rejudged,
            "greedy_correctness_source": self.greedy_correctness_source,
            "nli_equiv_probs": self.nli_equiv_probs,
            "equivalence_decision_source": self.equivalence_decision_source,
            "equivalence_decision_source_detail": self.equivalence_decision_source_detail,
            "equivalence_results_nli": self.equivalence_results_nli,
            "equivalence_stats_nli": self.equivalence_stats_nli,
            "equivalence_ratio_nli": self.equivalence_ratio_nli,
            "nli_equiv_probs_nli": self.nli_equiv_probs_nli,
            "error_label_nli_1.0": self.error_label_nli_1_0,
            "error_label_nli_0.9": self.error_label_nli_0_9,
            "error_label_nli_0.8": self.error_label_nli_0_8,
            "error_label_nli_0.7": self.error_label_nli_0_7,
            "stochastic_sample_grades": self.stochastic_sample_grades,
            "stochastic_sample_grade_source": self.stochastic_sample_grade_source,
            "stochastic_sample_grade_source_detail": self.stochastic_sample_grade_source_detail,
            "stochastic_sample_grade_confidence": self.stochastic_sample_grade_confidence,
            "stochastic_correct_rate": self.stochastic_correct_rate,
            "stochastic_scored_n": self.stochastic_scored_n,
            "stochastic_not_attempted_n": self.stochastic_not_attempted_n,
            "different_scored_n": self.different_scored_n,
            "different_correct_n": self.different_correct_n,
            "p_correct_given_different": self.p_correct_given_different,
            "semantic_pair_decisions": self.semantic_pair_decisions,
            "semantic_entropy": self.semantic_entropy,
            "semantic_entropy_norm": self.semantic_entropy_norm,
            "n_semantic_clusters": self.n_semantic_clusters,
            "semantic_cluster_ids": self.semantic_cluster_ids,
            "semantic_cluster_sizes": self.semantic_cluster_sizes,
            "semantic_entropy_label": self.semantic_entropy_label,
            "hybrid_enabled": self.hybrid_enabled,
            "hybrid_thresholds": self.hybrid_thresholds,
            "hybrid_calibration_id": self.hybrid_calibration_id,
            "hybrid_judge_model": self.hybrid_judge_model,
            "inter_rater_alpha": self.inter_rater_alpha,
            "run_id": self.run_id,
            "protocol_version": self.protocol_version,
            "config_hash": self.config_hash,
            "prompt_version": self.prompt_version,
            "run_date": self.run_date,
            "dataset_name": self.dataset_name,
            "dataset_split": self.dataset_split,
            "dataset_item_hash": self.dataset_item_hash,
            "model_provider": self.model_provider,
            "model_id": self.model_id,
            "model_snapshot_id": self.model_snapshot_id,
            "model_release_date": self.model_release_date,
            "model_track": self.model_track,
            "model_family": self.model_family,
            "model_version_index": self.model_version_index,
            "stochastic_target_n": self.stochastic_target_n,
            "stochastic_actual_n": self.stochastic_actual_n,
            "is_incomplete": self.is_incomplete,
            "sample_metadata": self.sample_metadata,
            "judge_protocol": self.judge_protocol,
            "judge_repeat_consistency": self.judge_repeat_consistency,
            "escalated_to_human": self.escalated_to_human,
            "contamination_flag": self.contamination_flag,
            "contamination_reason": self.contamination_reason,
        }
        for k, v in _optional.items():
            if v is not None:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ResultRecord":
        """Create from dictionary (backward compatible)."""
        equiv_stats = None
        if data.get("equivalence_stats"):
            equiv_stats = EquivalenceStats.from_dict(data["equivalence_stats"])

        return cls(
            question_id=data["question_id"],
            question=data["question"],
            ground_truth=data["ground_truth"],
            model=data["model"],
            greedy_answer=data["greedy_answer"],
            greedy_correct=data["greedy_correct"],
            correctness_match_type=data.get("correctness_match_type"),
            stochastic_answers=data.get("stochastic_answers"),
            equivalence_results=data.get("equivalence_results"),
            equivalence_stats=equiv_stats,
            equivalence_ratio=data.get("equivalence_ratio"),
            error_label_1_0=data.get("error_label_1.0"),
            error_label_0_9=data.get("error_label_0.9"),
            error_label_0_8=data.get("error_label_0.8"),
            error_label_0_7=data.get("error_label_0.7"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
            # New fields — .get() defaults to None for backward compat
            correctness_unclear=data.get("correctness_unclear"),
            correctness_grade=data.get("correctness_grade"),
            correctness_judge_grades=data.get("correctness_judge_grades"),
            correctness_judge_statuses=data.get("correctness_judge_statuses"),
            correctness_judge_reasoning=data.get("correctness_judge_reasoning"),
            correctness_decision_source=data.get("correctness_decision_source"),
            correctness_adjudicator_grade=data.get("correctness_adjudicator_grade"),
            correctness_adjudicator_status=data.get("correctness_adjudicator_status"),
            correctness_adjudicator_reasoning=data.get("correctness_adjudicator_reasoning"),
            correctness_nli_probs=data.get("correctness_nli_probs"),
            correctness_position_consistent=data.get("correctness_position_consistent"),
            greedy_correctness_rejudged=data.get("greedy_correctness_rejudged"),
            greedy_correctness_source=data.get("greedy_correctness_source"),
            nli_equiv_probs=data.get("nli_equiv_probs"),
            equivalence_decision_source=data.get("equivalence_decision_source"),
            equivalence_decision_source_detail=data.get("equivalence_decision_source_detail"),
            equivalence_results_nli=data.get("equivalence_results_nli"),
            equivalence_stats_nli=data.get("equivalence_stats_nli"),
            equivalence_ratio_nli=data.get("equivalence_ratio_nli"),
            nli_equiv_probs_nli=data.get("nli_equiv_probs_nli"),
            error_label_nli_1_0=data.get("error_label_nli_1.0"),
            error_label_nli_0_9=data.get("error_label_nli_0.9"),
            error_label_nli_0_8=data.get("error_label_nli_0.8"),
            error_label_nli_0_7=data.get("error_label_nli_0.7"),
            stochastic_sample_grades=data.get("stochastic_sample_grades"),
            stochastic_sample_grade_source=data.get("stochastic_sample_grade_source"),
            stochastic_sample_grade_source_detail=data.get("stochastic_sample_grade_source_detail"),
            stochastic_sample_grade_confidence=data.get("stochastic_sample_grade_confidence"),
            stochastic_correct_rate=data.get("stochastic_correct_rate"),
            stochastic_scored_n=data.get("stochastic_scored_n"),
            stochastic_not_attempted_n=data.get("stochastic_not_attempted_n"),
            different_scored_n=data.get("different_scored_n"),
            different_correct_n=data.get("different_correct_n"),
            p_correct_given_different=data.get("p_correct_given_different"),
            semantic_pair_decisions=data.get("semantic_pair_decisions"),
            semantic_entropy=data.get("semantic_entropy"),
            semantic_entropy_norm=data.get("semantic_entropy_norm"),
            n_semantic_clusters=data.get("n_semantic_clusters"),
            semantic_cluster_ids=data.get("semantic_cluster_ids"),
            semantic_cluster_sizes=data.get("semantic_cluster_sizes"),
            semantic_entropy_label=data.get("semantic_entropy_label"),
            hybrid_enabled=data.get("hybrid_enabled"),
            hybrid_thresholds=data.get("hybrid_thresholds"),
            hybrid_calibration_id=data.get("hybrid_calibration_id"),
            hybrid_judge_model=data.get("hybrid_judge_model"),
            inter_rater_alpha=data.get("inter_rater_alpha"),
            run_id=data.get("run_id"),
            protocol_version=data.get("protocol_version"),
            config_hash=data.get("config_hash"),
            prompt_version=data.get("prompt_version"),
            run_date=data.get("run_date"),
            dataset_name=data.get("dataset_name"),
            dataset_split=data.get("dataset_split"),
            dataset_item_hash=data.get("dataset_item_hash"),
            model_provider=data.get("model_provider"),
            model_id=data.get("model_id"),
            model_snapshot_id=data.get("model_snapshot_id"),
            model_release_date=data.get("model_release_date"),
            model_track=data.get("model_track"),
            model_family=data.get("model_family"),
            model_version_index=data.get("model_version_index"),
            stochastic_target_n=data.get("stochastic_target_n"),
            stochastic_actual_n=data.get("stochastic_actual_n"),
            is_incomplete=data.get("is_incomplete"),
            sample_metadata=data.get("sample_metadata"),
            judge_protocol=data.get("judge_protocol"),
            judge_repeat_consistency=data.get("judge_repeat_consistency"),
            escalated_to_human=data.get("escalated_to_human"),
            contamination_flag=data.get("contamination_flag"),
            contamination_reason=data.get("contamination_reason"),
        )
