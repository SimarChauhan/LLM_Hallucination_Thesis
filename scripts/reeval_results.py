#!/usr/bin/env python3
"""
Phase 2: Re-evaluation Pipeline for LLM Self-Consistent Error Measurement.

Reads raw data collected by run_pipeline.py (Phase 1) and performs:
  1. Correctness evaluation  (LLM-as-judge)
  2. Semantic equivalence  (hybrid NLI + borderline GPT judging)
  2.5 Sample correctness grading for stochastic answers (hybrid)
  3. Five-category labeling at multiple thresholds
     - reliably_correct, fragile_correct,
       self_consistent_error, inconsistent_error,
       not_attempted

This script is cheap and fast to re-run because it does NOT call the
model-under-test. Only judge-side API calls are made
(correctness + borderline hybrid decisions), and the NLI model runs locally.

Enhanced per research best-practices (Tier 1 + 2):
  - Chain-of-Thought reasoning stored per judge
  - NLI probabilities stored for equivalence judgments
  - NOT_ATTEMPTED tracked as a separate category
  - Ensemble voting (majority)
  - Inter-rater reliability (Krippendorff's alpha) computed and logged
  - Annotation sample export (--export-annotation-sample N)

Usage:
    # Full re-evaluation
    python scripts/reeval_results.py

    # Re-eval only missing/partial records
    python scripts/reeval_results.py --only-missing

    # Recompute semantic equivalence + semantic entropy only (no greedy re-judging)
    python scripts/reeval_results.py --semantic-only --force-recompute --no-llm-judge

    # Skip greedy re-judge only; keep hybrid equivalence + sample correctness + entropy
    python scripts/reeval_results.py --skip-greedy-correctness --force-recompute

    # Re-eval a specific model
    python scripts/reeval_results.py --models "Claude Opus"

    # LLM-as-judge is enabled by default (costs money)
    python scripts/reeval_results.py --use-llm-judge

    # Export annotation sample after re-eval
    python scripts/reeval_results.py --export-annotation-sample 50

    # Custom input/output
    python scripts/reeval_results.py --input data/results/results.jsonl --output data/results/results_eval.jsonl
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Any, Optional, Tuple

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
REPO_ROOT = Path(__file__).resolve().parents[1]

from src.correctness import check_correctness
from src.hybrid_judging import (
    compute_pairwise_hybrid_equivalence,
    compute_stochastic_correctness_metrics,
    decide_equivalence_hybrid,
    grade_sample_correctness_hybrid,
)
from src.nli_judge import NLISemanticJudge
from src.labeling import compute_equivalence_stats, classify_at_multiple_thresholds
from src.semantic_entropy import compute_semantic_entropy, classify_error_by_entropy
from src.schemas import ResultRecord, EquivalenceStats
from src.storage import ResultStorage

# Load environment variables (CWD first, then repository root).
load_dotenv()
load_dotenv(REPO_ROOT / ".env", override=False)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("reeval.log"),
    ],
)
logger = logging.getLogger(__name__)


def to_bool(value: Any) -> bool:
    """Robust bool parsing for mixed legacy JSON types."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return False


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Return stable hash of loaded config."""
    payload = yaml.safe_dump(config, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def infer_model_family(record: Dict[str, Any]) -> str:
    """Infer model family/provider from record fields."""
    provider = (record.get("model_provider") or "").strip().lower()
    if provider:
        return provider
    model_name = (record.get("model") or "").lower()
    if "openai" in model_name or "gpt" in model_name:
        return "openai"
    if "anthropic" in model_name or "claude" in model_name:
        return "anthropic"
    if "google" in model_name or "gemini" in model_name:
        return "google"
    if "xai" in model_name or "grok" in model_name:
        return "xai"
    if "deepseek" in model_name:
        return "deepseek"
    if "qwen" in model_name or "llama" in model_name:
        return "huggingface"
    return "unknown"


def describe_judge_protocol(
    use_llm_judge: bool,
    llm_judge_ensemble: Optional[List[Dict[str, Any]]],
    adjudicator: Optional[Dict[str, Any]] = None,
) -> str:
    """Human-readable judge protocol descriptor."""
    if not use_llm_judge:
        return "no_correctness_judge"
    if llm_judge_ensemble:
        if adjudicator:
            return (
                f"llm_ensemble(n={len(llm_judge_ensemble)})"
                f"+adjudicator({adjudicator.get('provider')}/{adjudicator.get('model')})"
            )
        return f"llm_ensemble(n={len(llm_judge_ensemble)})"
    return "no_correctness_judge"


def should_repeat_judge(record: Dict[str, Any], fraction: float) -> bool:
    """Deterministically pick records for repeat-judge consistency measurement."""
    if fraction <= 0:
        return False
    key = f"{record.get('question_id','')}|{record.get('model','')}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF
    return score < fraction


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_records(filepath: str) -> List[Dict[str, Any]]:
    """Load raw JSON records from a JSONL file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing line {line_num}: {e}")
    logger.info(f"Loaded {len(records)} raw records from {filepath}")
    return records


def save_records(records: List[Dict[str, Any]], filepath: str) -> None:
    """Write evaluated records to a JSONL file atomically (overwrites)."""
    ResultStorage.write_jsonl_atomic(records, filepath)
    logger.info(f"Saved {len(records)} evaluated records to {filepath}")


def is_phase2_record_complete(record: Dict[str, Any]) -> bool:
    """
    Return True only when core Phase-2 outputs are present.

    This is stricter than checking `greedy_correct` alone and prevents
    interrupted/partially-written rows from being skipped by `--only-missing`.
    """
    if record.get("greedy_correct") is None:
        return False
    if record.get("error_label_1.0") is None:
        return False
    if record.get("error_label_0.9") is None:
        return False
    if record.get("error_label_0.8") is None:
        return False
    if record.get("error_label_0.7") is None:
        return False

    stochastic_answers = record.get("stochastic_answers") or []
    if stochastic_answers:
        if record.get("equivalence_results") is None:
            return False
        if record.get("equivalence_stats") is None:
            return False
    return True


def _round_optional(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _build_skip_greedy_output_path(input_path: str) -> str:
    """Derive non-overwriting output path for skip-greedy mode."""
    path = Path(input_path)
    suffix = ".skip_greedy_semantic_eval"
    if path.suffix:
        return str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))
    return str(path.with_name(f"{path.name}{suffix}.jsonl"))


def _ensure_skip_mode_non_overwrite(input_path: str, output_path: str, skip_greedy_correctness: bool) -> None:
    """Fail fast if skip-greedy mode would overwrite input data."""
    if not skip_greedy_correctness:
        return
    if Path(input_path).expanduser().resolve() == Path(output_path).expanduser().resolve():
        raise ValueError(
            "skip-greedy-correctness mode requires a distinct output file; "
            "refusing to overwrite input."
        )


def _build_precomputed_correctness_result(rec: Dict[str, Any]) -> Tuple[SimpleNamespace, bool]:
    """
    Reuse precomputed greedy correctness fields without re-judging.

    Returns:
        (correctness_result, is_missing_precomputed)
    """
    existing_grade = rec.get("correctness_grade")
    existing_bool = rec.get("greedy_correct")
    existing_unclear = rec.get("correctness_unclear")

    valid_grade = existing_grade in {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}
    valid_bool = existing_bool in {True, False}
    if not valid_grade and not valid_bool:
        result = SimpleNamespace(
            is_correct=False,
            match_type=rec.get("correctness_match_type"),
            is_unclear=True,
            grade="NOT_ATTEMPTED",
            judge_votes=rec.get("correctness_judge_votes"),
            judge_grades=rec.get("correctness_judge_grades"),
            judge_statuses=rec.get("correctness_judge_statuses"),
            judge_reasoning=rec.get("correctness_judge_reasoning"),
            decision_source="PRECOMPUTED_MISSING",
            adjudicator_grade=rec.get("correctness_adjudicator_grade"),
            adjudicator_status=rec.get("correctness_adjudicator_status"),
            adjudicator_reasoning=rec.get("correctness_adjudicator_reasoning"),
            nli_probs=rec.get("correctness_nli_probs"),
        )
        return result, True

    if not valid_grade:
        existing_grade = "CORRECT" if existing_bool is True else "INCORRECT"
    if not valid_bool:
        existing_bool = existing_grade == "CORRECT"

    inferred_unclear = bool(existing_unclear) or existing_grade == "NOT_ATTEMPTED"
    result = SimpleNamespace(
        is_correct=bool(existing_bool),
        match_type=rec.get("correctness_match_type"),
        is_unclear=inferred_unclear,
        grade=existing_grade,
        judge_votes=rec.get("correctness_judge_votes"),
        judge_grades=rec.get("correctness_judge_grades"),
        judge_statuses=rec.get("correctness_judge_statuses"),
        judge_reasoning=rec.get("correctness_judge_reasoning"),
        decision_source=rec.get("correctness_decision_source") or "PRECOMPUTED",
        adjudicator_grade=rec.get("correctness_adjudicator_grade"),
        adjudicator_status=rec.get("correctness_adjudicator_status"),
        adjudicator_reasoning=rec.get("correctness_adjudicator_reasoning"),
        nli_probs=rec.get("correctness_nli_probs"),
    )
    return result, False


def _load_hybrid_thresholds(
    config: Dict[str, Any],
    override_file: Optional[str],
) -> Dict[str, Any]:
    """
    Load hybrid thresholds from calibration artifact if available, else defaults.
    """
    defaults = {
        "eq_same_hi": 0.70,
        "eq_diff_lo": 0.30,
        "corr_hi": 0.70,
        "corr_lo": 0.30,
    }
    hybrid_cfg = config.get("hybrid", {}) or {}
    threshold_cfg = hybrid_cfg.get("thresholds", {}) or {}
    thresholds = {
        "eq_same_hi": float(threshold_cfg.get("eq_same_hi", defaults["eq_same_hi"])),
        "eq_diff_lo": float(threshold_cfg.get("eq_diff_lo", defaults["eq_diff_lo"])),
        "corr_hi": float(threshold_cfg.get("corr_hi", defaults["corr_hi"])),
        "corr_lo": float(threshold_cfg.get("corr_lo", defaults["corr_lo"])),
    }

    calibration_path = (
        override_file
        or hybrid_cfg.get("calibration_file")
        or (config.get("calibration", {}) or {}).get("hybrid_calibration_file")
    )
    calibration_id = None
    if calibration_path:
        path = Path(calibration_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            thresholds["eq_same_hi"] = float(payload.get("eq_same_hi", thresholds["eq_same_hi"]))
            thresholds["eq_diff_lo"] = float(payload.get("eq_diff_lo", thresholds["eq_diff_lo"]))
            thresholds["corr_hi"] = float(payload.get("corr_hi", thresholds["corr_hi"]))
            thresholds["corr_lo"] = float(payload.get("corr_lo", thresholds["corr_lo"]))
            calibration_id = str(payload.get("calibration_id") or path.stem)
            logger.info("Loaded hybrid calibration from %s", path)
        else:
            logger.warning("Hybrid calibration file not found: %s (using defaults/config thresholds)", path)

    if thresholds["eq_diff_lo"] >= thresholds["eq_same_hi"]:
        raise ValueError(
            f"Invalid hybrid thresholds: eq_diff_lo ({thresholds['eq_diff_lo']}) must be < eq_same_hi ({thresholds['eq_same_hi']})."
        )
    if thresholds["corr_lo"] >= thresholds["corr_hi"]:
        raise ValueError(
            f"Invalid hybrid thresholds: corr_lo ({thresholds['corr_lo']}) must be < corr_hi ({thresholds['corr_hi']})."
        )

    return {
        "thresholds": thresholds,
        "calibration_id": calibration_id,
    }


def run_reeval(
    config_path: str,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    only_missing: bool = False,
    force_recompute: bool = False,
    strict_comparability: bool = False,
    protocol_version: Optional[str] = None,
    run_id: Optional[str] = None,
    models_filter: Optional[List[str]] = None,
    use_llm_judge: bool = True,
    export_annotation_sample: int = 0,
    nli_calibration_file: Optional[str] = None,
    hybrid_calibration_file: Optional[str] = None,
    skip_greedy_correctness: bool = False,
    semantic_only: bool = False,
):
    """
    Phase 2: Evaluate all collected records.

    Args:
        config_path: Path to config.yaml
        input_file: Override input JSONL path
        output_file: Override output JSONL path
        only_missing: If True, only evaluate records that haven't been evaluated yet
        force_recompute: If True, recompute every selected record regardless of existing fields
        strict_comparability: If True, force uniform protocol and fail on mixed settings
        protocol_version: Protocol version tag to write into all output records
        run_id: Optional run id to resolve immutable raw/evaluated run directories
        models_filter: Optional model name filters (partial match)
        use_llm_judge: Whether to enable LLM-as-judge in correctness evaluation
        export_annotation_sample: If > 0, export N records for human annotation
        nli_calibration_file: Path to NLI calibration JSON (from human annotations)
        hybrid_calibration_file: Path to frozen hybrid threshold calibration artifact
        skip_greedy_correctness: If True, reuse precomputed greedy correctness and do not re-judge
        semantic_only: If True, skip greedy correctness re-judging and run semantic steps only
    """
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    protocol_config = config.get("protocol", {})
    high_rigor = bool(protocol_config.get("high_rigor", True))
    if protocol_version is None:
        protocol_version = str(protocol_config.get("version", "v3"))
    prompt_version = str(protocol_config.get("prompt_version", "qa-short-v1"))
    config_hash = compute_config_hash(config)
    collection_config = config.get("collection", {})
    inference_config = config.get("inference", {})
    stochastic_config = inference_config.get("stochastic", {})
    required_samples = int(collection_config.get("required_samples", stochastic_config.get("num_samples", 10)))

    if strict_comparability and only_missing and not force_recompute:
        raise ValueError("strict comparability requires full recompute; disable --only-missing or pass --force-recompute.")
    if strict_comparability:
        force_recompute = True
        only_missing = False
    if skip_greedy_correctness and semantic_only:
        raise ValueError("Use either --skip-greedy-correctness or --semantic-only, not both.")
    if not use_llm_judge and not semantic_only:
        raise ValueError(
            "Correctness now requires LLM-as-judge. "
            "Run with --use-llm-judge, or pass --semantic-only."
        )
    if semantic_only:
        logger.info("Semantic-only mode enabled: skipping greedy correctness re-judging.")
    if skip_greedy_correctness:
        logger.info("Skip-greedy mode enabled: reusing precomputed greedy correctness.")

    # Paths (supports immutable run directories)
    output_config = config.get("output", {})
    base_dir = os.environ.get("RESULTS_DIR_ABSOLUTE") or output_config.get("results_dir", "data/results")
    base_dir = str(Path(base_dir).expanduser().resolve())
    raw_dir = output_config.get("raw_dir", "raw")
    evaluated_dir = output_config.get("evaluated_dir", "evaluated")
    immutable_runs = bool(output_config.get("immutable_runs", high_rigor))
    raw_results_file = output_config.get("results_file", "results.jsonl")
    evaluated_results_file = output_config.get("evaluated_file", "results_v2_eval.jsonl")

    if input_file is None:
        if skip_greedy_correctness:
            if run_id and immutable_runs:
                run_evaluated_input = Path(base_dir) / evaluated_dir / run_id / evaluated_results_file
                if run_evaluated_input.exists():
                    input_file = str(run_evaluated_input)
                else:
                    input_file = str(Path(base_dir) / evaluated_dir / evaluated_results_file)
            else:
                input_file = str(Path(base_dir) / evaluated_dir / evaluated_results_file)
        else:
            if run_id:
                run_input = Path(base_dir) / raw_dir / run_id / raw_results_file
                if run_input.exists():
                    input_file = str(run_input)
                else:
                    input_file = str(Path(base_dir) / raw_dir / raw_results_file)
            else:
                input_file = str(Path(base_dir) / raw_dir / raw_results_file)
    if output_file is None:
        if skip_greedy_correctness:
            output_file = _build_skip_greedy_output_path(input_file)
        else:
            if run_id and immutable_runs:
                output_file = str(Path(base_dir) / evaluated_dir / run_id / evaluated_results_file)
            else:
                output_file = str(Path(base_dir) / evaluated_dir / evaluated_results_file)

    _ensure_skip_mode_non_overwrite(
        input_path=input_file,
        output_path=output_file,
        skip_greedy_correctness=skip_greedy_correctness,
    )

    results_dir = str(Path(output_file).parent)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    storage_for_retry = ResultStorage(
        results_dir=results_dir,
        results_file=Path(output_file).name,
    )

    all_records = load_raw_records(input_file)
    if models_filter:
        records = [
            r for r in all_records
            if any(f.lower() in r.get("model", "").lower() for f in models_filter)
        ]
        logger.info(f"Filtered to {len(records)} records matching models: {models_filter}")
    else:
        records = all_records
    if not records:
        logger.warning("No records to evaluate.")
        return

    judge_config = config.get("judge", {})
    enforce_cross_family = bool(judge_config.get("enforce_cross_family", high_rigor))
    repeat_eval_fraction = float(judge_config.get("repeat_eval_fraction", 0.1 if high_rigor else 0.0))

    # ---- Initialize NLI judge for correctness + equivalence ----
    nli_config = judge_config.get("nli", {})
    nli_model_name = nli_config.get(
        "model", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    )
    entailment_threshold = nli_config.get("entailment_threshold", 0.5)
    different_threshold = nli_config.get("different_threshold", 0.3)
    device = nli_config.get("device", None)

    logger.info(f"Initializing NLI judge: {nli_model_name}")
    nli_judge = NLISemanticJudge(
        model_name=nli_model_name,
        device=device,
        entailment_threshold=entailment_threshold,
        different_threshold=different_threshold,
    )

    cal_file = nli_calibration_file or config.get("calibration", {}).get("nli_calibration_file")
    if cal_file and Path(cal_file).exists():
        nli_judge.load_calibration(cal_file)

    # ---- Optionally initialize LLM-as-judge ----
    inference_client = None
    llm_judge_ensemble = None
    adjudicator_config: Optional[Dict[str, Any]] = None
    adjudicator_max_new_tokens = 260
    if use_llm_judge:
        from src.providers import MultiProviderClient

        rate_config = config.get("rate_limit", {})
        inference_client = MultiProviderClient(
            initial_delay=rate_config.get("initial_delay", 2.0),
            max_delay=rate_config.get("max_delay", 60.0),
            backoff_factor=rate_config.get("backoff_factor", 2.0),
        )
        llm_judge_ensemble = judge_config.get("llm_judge_ensemble") or []
        if llm_judge_ensemble:
            if len(llm_judge_ensemble) != 3:
                raise ValueError(
                    "LLM judging requires exactly 3 judges in `judge.llm_judge_ensemble`; "
                    f"got {len(llm_judge_ensemble)}."
                )
            logger.info("LLM-as-judge ensemble: 3 judges")
            for i, j in enumerate(llm_judge_ensemble):
                logger.info(f"  Judge {i}: {j.get('provider')} / {j.get('model')}")
        else:
            raise ValueError(
                "LLM judging is enabled but `judge.llm_judge_ensemble` is empty/missing. "
                "Populate `judge.llm_judge_ensemble` in config.yaml."
            )

        raw_adjudicator_cfg = judge_config.get("adjudicator", {}) or {}
        if bool(raw_adjudicator_cfg.get("enabled", False)):
            adjudicator_config = {
                "provider": raw_adjudicator_cfg.get(
                    "provider",
                    judge_config.get("llm_judge", {}).get("provider", "openai"),
                ),
                "model": raw_adjudicator_cfg.get(
                    "model",
                    judge_config.get("llm_judge", {}).get("model", "gpt-4o"),
                ),
            }
            adjudicator_max_new_tokens = int(
                raw_adjudicator_cfg.get(
                    "max_new_tokens",
                    judge_config.get("llm_judge", {}).get("max_new_tokens", 260),
                )
            )
            logger.info(
                "Phase-2 adjudicator enabled: %s / %s",
                adjudicator_config["provider"],
                adjudicator_config["model"],
            )

    ensemble_config = judge_config.get("ensemble", {})
    failure_policy = ensemble_config.get("failure_policy", "skip")
    max_new_tokens = judge_config.get("llm_judge", {}).get("max_new_tokens", 200)
    judge_protocol = describe_judge_protocol(
        use_llm_judge=use_llm_judge,
        llm_judge_ensemble=llm_judge_ensemble,
        adjudicator=adjudicator_config,
    )

    # ---- Correctness settings ----
    correctness_config = config.get("correctness", {})
    semantic_config = config.get("semantic", {})
    unclear_treatment = semantic_config.get("unclear_treatment", "exclude")
    semantic_entropy_config = config.get("semantic_entropy", {}) or {}
    use_normalized_semantic_entropy = bool(semantic_entropy_config.get("use_normalized", True))
    semantic_entropy_threshold = float(semantic_entropy_config.get("threshold", 0.35))
    hybrid_cfg = config.get("hybrid", {}) or {}
    hybrid_enabled = bool(hybrid_cfg.get("enabled", True))
    borderline_cfg = hybrid_cfg.get("borderline_judge", {}) or {}
    hybrid_judge_provider = str(borderline_cfg.get("provider", "openai"))
    hybrid_judge_model = str(borderline_cfg.get("model", "gpt-5.2"))
    hybrid_eq_max_new_tokens = int(borderline_cfg.get("equivalence_max_new_tokens", 220))
    hybrid_corr_max_new_tokens = int(borderline_cfg.get("correctness_max_new_tokens", 260))
    hybrid_threshold_bundle = _load_hybrid_thresholds(
        config=config,
        override_file=hybrid_calibration_file,
    )
    hybrid_thresholds = hybrid_threshold_bundle["thresholds"]
    hybrid_calibration_id = hybrid_threshold_bundle.get("calibration_id")

    logger.info(
        "Hybrid thresholds: eq_same_hi=%.3f eq_diff_lo=%.3f corr_hi=%.3f corr_lo=%.3f",
        hybrid_thresholds["eq_same_hi"],
        hybrid_thresholds["eq_diff_lo"],
        hybrid_thresholds["corr_hi"],
        hybrid_thresholds["corr_lo"],
    )
    if hybrid_calibration_id:
        logger.info("Hybrid calibration id: %s", hybrid_calibration_id)
    if not hybrid_enabled:
        logger.warning("Hybrid judging is disabled; falling back to NLI-only equivalence and no stochastic sample grading.")
    if semantic_only:
        judge_protocol = "precomputed_greedy+semantic_only"
    elif skip_greedy_correctness:
        judge_protocol = (
            f"precomputed_greedy+hybrid_borderline({hybrid_judge_provider}/{hybrid_judge_model})"
        )

    stats = {
        "total": len(records),
        "evaluated": 0,
        "skipped": 0,
        "reliably_correct": 0,
        "fragile_correct": 0,
        "self_consistent_error": 0,
        "inconsistent_error": 0,
        "not_attempted": 0,
        "correctness_unclear": 0,
        "escalated_to_human": 0,
        "missing_precomputed_greedy": 0,
        "repeated_judge_items": 0,
        "repeat_inconsistent": 0,
        "hybrid_equiv_llm_rows": 0,
        "hybrid_corr_llm_rows": 0,
    }

    all_ensemble_grades: List[List[Optional[str]]] = []

    checkpoint_every = 200  # save partial results so interrupt/crash doesn't lose progress
    try:
        for idx, rec in enumerate(tqdm(records, desc="Evaluating")):
            # Skip only rows with complete Phase-2 outputs when --only-missing is enabled.
            if (not force_recompute) and only_missing and is_phase2_record_complete(rec):
                stats["skipped"] += 1
                continue

            question = rec.get("question", "")
            greedy_answer = rec.get("greedy_answer", "")
            ground_truths = rec.get("ground_truth", []) or []
            stochastic_answers = rec.get("stochastic_answers") or []
            rec["protocol_version"] = protocol_version
            rec["config_hash"] = config_hash
            rec["prompt_version"] = rec.get("prompt_version") or prompt_version
            rec["judge_protocol"] = judge_protocol
            rec["hybrid_enabled"] = hybrid_enabled
            rec["hybrid_thresholds"] = {
                "eq_same_hi": hybrid_thresholds["eq_same_hi"],
                "eq_diff_lo": hybrid_thresholds["eq_diff_lo"],
                "corr_hi": hybrid_thresholds["corr_hi"],
                "corr_lo": hybrid_thresholds["corr_lo"],
            }
            rec["hybrid_calibration_id"] = hybrid_calibration_id
            rec["hybrid_judge_model"] = f"{hybrid_judge_provider}/{hybrid_judge_model}"
            if run_id is not None:
                rec["run_id"] = run_id
            rec["stochastic_target_n"] = int(rec.get("stochastic_target_n") or required_samples)
            rec["stochastic_actual_n"] = int(rec.get("stochastic_actual_n") or len(stochastic_answers))
            rec["is_incomplete"] = bool(
                to_bool(rec.get("is_incomplete"))
                if rec.get("is_incomplete") is not None
                else rec["stochastic_actual_n"] < rec["stochastic_target_n"]
            )
            if rec.get("dataset_name") is None:
                rec["dataset_name"] = "unknown"
            if rec.get("dataset_split") is None:
                rec["dataset_split"] = "unknown"
            if rec.get("dataset_item_hash") is None:
                stable = f"{question}\n---\n{chr(31).join(map(str, ground_truths))}"
                rec["dataset_item_hash"] = hashlib.sha256(stable.encode("utf-8")).hexdigest()
            if rec.get("model_provider") is None:
                rec["model_provider"] = infer_model_family(rec)
            if rec.get("model_id") is None:
                rec["model_id"] = rec.get("model", "")
            if rec.get("contamination_flag") is None:
                rec["contamination_flag"] = False
            if rec.get("contamination_reason") is None:
                rec["contamination_reason"] = None

            # Optional cross-family policy enforcement
            use_llm_for_record = use_llm_judge and not semantic_only and not skip_greedy_correctness
            record_judges = llm_judge_ensemble if use_llm_for_record else None
            if enforce_cross_family and use_llm_for_record:
                model_family = infer_model_family(rec)
                judge_families = set()
                if llm_judge_ensemble:
                    for judge_cfg in llm_judge_ensemble:
                        judge_families.add(str(judge_cfg.get("provider", "unknown")).lower())

                if model_family != "unknown" and model_family in judge_families:
                    message = (
                        f"Cross-family policy violation for {rec.get('question_id')} / {rec.get('model')}: "
                        f"model family '{model_family}' appears in judge set {sorted(judge_families)}"
                    )
                    if strict_comparability:
                        raise ValueError(message)
                    logger.warning(
                        f"{message}; keeping full 3-judge ensemble to avoid protocol drift."
                    )

            # Record the true per-row judge protocol.
            if semantic_only:
                rec["judge_protocol"] = "precomputed_greedy+semantic_only"
            elif skip_greedy_correctness:
                rec["judge_protocol"] = (
                    f"precomputed_greedy+hybrid_borderline({hybrid_judge_provider}/{hybrid_judge_model})"
                )
            else:
                rec["judge_protocol"] = describe_judge_protocol(
                    use_llm_judge=use_llm_for_record,
                    llm_judge_ensemble=record_judges if use_llm_for_record else None,
                    adjudicator=adjudicator_config if use_llm_for_record else None,
                )

            # ---- Step 1: Correctness evaluation ----
            repeat_consistency = None
            missing_precomputed_greedy = False
            if semantic_only or skip_greedy_correctness:
                correctness_result, missing_precomputed_greedy = _build_precomputed_correctness_result(rec)
                if missing_precomputed_greedy:
                    stats["missing_precomputed_greedy"] += 1
                repeat_consistency = rec.get("judge_repeat_consistency")
                rec["greedy_correctness_rejudged"] = False
                rec["greedy_correctness_source"] = (
                    "MISSING" if missing_precomputed_greedy else "PRECOMPUTED"
                )
            else:
                correctness_result = check_correctness(
                    prediction=greedy_answer,
                    ground_truths=ground_truths,
                    strip_articles=correctness_config.get("strip_articles", True),
                    max_length_ratio=correctness_config.get("max_length_ratio", 3.0),
                    nli_judge=nli_judge,
                    question=question,
                    use_nli_fallback=correctness_config.get("use_nli_fallback", False),
                    nli_entailment_threshold=correctness_config.get("nli_entailment_threshold", 0.5),
                    inference_client=inference_client,
                    use_llm_fallback=use_llm_for_record,
                    llm_judge_ensemble=record_judges if use_llm_for_record else None,
                    max_new_tokens=max_new_tokens,
                    failure_policy=failure_policy,
                    adjudicator=adjudicator_config if use_llm_for_record else None,
                    adjudicator_max_new_tokens=adjudicator_max_new_tokens,
                )

                # Optional repeat-eval check for judge self-consistency.
                if use_llm_for_record and should_repeat_judge(rec, repeat_eval_fraction):
                    stats["repeated_judge_items"] += 1
                    repeated = check_correctness(
                        prediction=greedy_answer,
                        ground_truths=ground_truths,
                        strip_articles=correctness_config.get("strip_articles", True),
                        max_length_ratio=correctness_config.get("max_length_ratio", 3.0),
                        nli_judge=nli_judge,
                        question=question,
                        use_nli_fallback=correctness_config.get("use_nli_fallback", False),
                        nli_entailment_threshold=correctness_config.get("nli_entailment_threshold", 0.5),
                        inference_client=inference_client,
                        use_llm_fallback=use_llm_for_record,
                        llm_judge_ensemble=record_judges if use_llm_for_record else None,
                        max_new_tokens=max_new_tokens,
                        failure_policy=failure_policy,
                        adjudicator=adjudicator_config if use_llm_for_record else None,
                        adjudicator_max_new_tokens=adjudicator_max_new_tokens,
                    )
                    same_vote = (
                        repeated.is_correct == correctness_result.is_correct
                        and repeated.grade == correctness_result.grade
                    )
                    repeat_consistency = 1.0 if same_vote else 0.0
                    if not same_vote:
                        stats["repeat_inconsistent"] += 1
                rec["greedy_correctness_rejudged"] = True
                rec["greedy_correctness_source"] = "JUDGED"

            # Store core correctness fields
            rec["greedy_correct"] = correctness_result.is_correct
            rec["correctness_match_type"] = correctness_result.match_type
            rec["correctness_unclear"] = correctness_result.is_unclear

            # Store new enriched fields (Tier 1 + 2)
            rec["correctness_grade"] = correctness_result.grade
            if correctness_result.judge_votes is not None:
                rec["correctness_judge_votes"] = correctness_result.judge_votes
            if correctness_result.judge_grades is not None:
                rec["correctness_judge_grades"] = correctness_result.judge_grades
                if not (semantic_only or skip_greedy_correctness):
                    all_ensemble_grades.append(correctness_result.judge_grades)
            if correctness_result.judge_statuses is not None:
                rec["correctness_judge_statuses"] = correctness_result.judge_statuses
            if correctness_result.judge_reasoning is not None:
                rec["correctness_judge_reasoning"] = correctness_result.judge_reasoning
            if correctness_result.decision_source is not None:
                rec["correctness_decision_source"] = correctness_result.decision_source
            if correctness_result.adjudicator_grade is not None:
                rec["correctness_adjudicator_grade"] = correctness_result.adjudicator_grade
            if correctness_result.adjudicator_status is not None:
                rec["correctness_adjudicator_status"] = correctness_result.adjudicator_status
            if correctness_result.adjudicator_reasoning is not None:
                rec["correctness_adjudicator_reasoning"] = correctness_result.adjudicator_reasoning
            if correctness_result.nli_probs is not None:
                rec["correctness_nli_probs"] = correctness_result.nli_probs
            if repeat_consistency is not None:
                rec["judge_repeat_consistency"] = repeat_consistency

            # ---- Step 2: Semantic equivalence + sample correctness (hybrid) ----
            semantic_entropy = None
            semantic_entropy_norm = None
            if stochastic_answers:
                legacy_equivalence_results_rich = nli_judge.judge_all_samples(
                    question, greedy_answer, stochastic_answers
                )
                legacy_equivalence_results = [r.judgment for r in legacy_equivalence_results_rich]
                legacy_nli_equiv_probs = [
                    {
                        "forward": round(r.prob_forward, 4),
                        "reverse": round(r.prob_reverse, 4),
                        "judgment": r.judgment,
                        "source": "NLI",
                    }
                    for r in legacy_equivalence_results_rich
                ]
                legacy_equiv_stats_bare = compute_equivalence_stats(legacy_equivalence_results)
                legacy_equiv_stats = EquivalenceStats(
                    num_same=legacy_equiv_stats_bare.num_same,
                    num_different=legacy_equiv_stats_bare.num_different,
                    num_unclear=legacy_equiv_stats_bare.num_unclear,
                    total=legacy_equiv_stats_bare.total,
                    nli_probs=legacy_nli_equiv_probs,
                )

                rec["equivalence_results_nli"] = legacy_equivalence_results
                rec["equivalence_stats_nli"] = legacy_equiv_stats.to_dict()
                rec["equivalence_ratio_nli"] = legacy_equiv_stats.equivalence_ratio
                rec["nli_equiv_probs_nli"] = legacy_nli_equiv_probs

                if hybrid_enabled:
                    equivalence_results = []
                    equivalence_decision_source = []
                    equivalence_decision_source_detail = []
                    nli_equiv_probs = []
                    for sample in stochastic_answers:
                        eq_decision = decide_equivalence_hybrid(
                            question=question,
                            answer_a=greedy_answer,
                            answer_b=sample,
                            nli_judge=nli_judge,
                            eq_same_hi=hybrid_thresholds["eq_same_hi"],
                            eq_diff_lo=hybrid_thresholds["eq_diff_lo"],
                            inference_client=inference_client,
                            judge_provider=hybrid_judge_provider,
                            judge_model=hybrid_judge_model,
                            max_new_tokens=hybrid_eq_max_new_tokens,
                        )
                        equivalence_results.append(eq_decision.label)
                        equivalence_decision_source.append(eq_decision.source)
                        equivalence_decision_source_detail.append(eq_decision.source_detail)
                        nli_equiv_probs.append(
                            {
                                "forward": _round_optional(eq_decision.prob_forward, 4),
                                "reverse": _round_optional(eq_decision.prob_reverse, 4),
                                "judgment": eq_decision.label,
                                "source": eq_decision.source,
                                "source_detail": eq_decision.source_detail,
                                "llm_label_forward": eq_decision.llm_label_forward,
                                "llm_label_reverse": eq_decision.llm_label_reverse,
                            }
                        )
                    if any(src == "LLM" for src in equivalence_decision_source):
                        stats["hybrid_equiv_llm_rows"] += 1
                else:
                    equivalence_results = legacy_equivalence_results
                    equivalence_decision_source = ["NLI"] * len(equivalence_results)
                    equivalence_decision_source_detail = ["NLI_BASELINE"] * len(equivalence_results)
                    nli_equiv_probs = legacy_nli_equiv_probs

                equiv_stats_bare = compute_equivalence_stats(equivalence_results)
                equiv_stats = EquivalenceStats(
                    num_same=equiv_stats_bare.num_same,
                    num_different=equiv_stats_bare.num_different,
                    num_unclear=equiv_stats_bare.num_unclear,
                    total=equiv_stats_bare.total,
                    nli_probs=nli_equiv_probs,
                )

                rec["equivalence_results"] = equivalence_results
                rec["equivalence_stats"] = equiv_stats.to_dict()
                rec["equivalence_ratio"] = equiv_stats.equivalence_ratio
                rec["nli_equiv_probs"] = nli_equiv_probs
                rec["equivalence_decision_source"] = equivalence_decision_source
                rec["equivalence_decision_source_detail"] = equivalence_decision_source_detail
                rec["hybrid_calibration_id"] = hybrid_calibration_id

                if hybrid_enabled and not semantic_only:
                    stochastic_sample_grades = []
                    stochastic_sample_grade_source = []
                    stochastic_sample_grade_source_detail = []
                    stochastic_sample_grade_confidence = []
                    for sample in stochastic_answers:
                        corr_decision = grade_sample_correctness_hybrid(
                            question=question,
                            sample_answer=sample,
                            ground_truths=ground_truths,
                            nli_judge=nli_judge,
                            corr_hi=hybrid_thresholds["corr_hi"],
                            corr_lo=hybrid_thresholds["corr_lo"],
                            inference_client=inference_client,
                            judge_provider=hybrid_judge_provider,
                            judge_model=hybrid_judge_model,
                            max_new_tokens=hybrid_corr_max_new_tokens,
                        )
                        stochastic_sample_grades.append(corr_decision.grade)
                        stochastic_sample_grade_source.append(corr_decision.source)
                        stochastic_sample_grade_source_detail.append(corr_decision.source_detail)
                        stochastic_sample_grade_confidence.append(
                            {
                                "p_max": _round_optional(corr_decision.p_max, 4),
                                "matched_gold_index": corr_decision.matched_gold_index,
                                "matched_gold": corr_decision.matched_gold,
                            }
                        )

                    if any(src == "LLM" for src in stochastic_sample_grade_source):
                        stats["hybrid_corr_llm_rows"] += 1

                    rec["stochastic_sample_grades"] = stochastic_sample_grades
                    rec["stochastic_sample_grade_source"] = stochastic_sample_grade_source
                    rec["stochastic_sample_grade_source_detail"] = stochastic_sample_grade_source_detail
                    rec["stochastic_sample_grade_confidence"] = stochastic_sample_grade_confidence

                    corr_metrics = compute_stochastic_correctness_metrics(
                        equivalence_results=equivalence_results,
                        sample_grades=stochastic_sample_grades,
                    )
                    rec["stochastic_correct_rate"] = _round_optional(
                        corr_metrics.get("stochastic_correct_rate"),
                        6,
                    )
                    rec["stochastic_scored_n"] = corr_metrics.get("stochastic_scored_n")
                    rec["stochastic_not_attempted_n"] = corr_metrics.get("stochastic_not_attempted_n")
                    rec["different_scored_n"] = corr_metrics.get("different_scored_n")
                    rec["different_correct_n"] = corr_metrics.get("different_correct_n")
                    rec["p_correct_given_different"] = _round_optional(
                        corr_metrics.get("p_correct_given_different"),
                        6,
                    )
                else:
                    rec["stochastic_sample_grades"] = None
                    rec["stochastic_sample_grade_source"] = None
                    rec["stochastic_sample_grade_source_detail"] = None
                    rec["stochastic_sample_grade_confidence"] = None
                    rec["stochastic_correct_rate"] = None
                    rec["stochastic_scored_n"] = None
                    rec["stochastic_not_attempted_n"] = None
                    rec["different_scored_n"] = None
                    rec["different_correct_n"] = None
                    rec["p_correct_given_different"] = None

                # ---- Step 2.5: Semantic entropy over stochastic samples ----
                if hybrid_enabled:
                    pairwise_decisions = compute_pairwise_hybrid_equivalence(
                        question=question,
                        sample_answers=stochastic_answers,
                        nli_judge=nli_judge,
                        eq_same_hi=hybrid_thresholds["eq_same_hi"],
                        eq_diff_lo=hybrid_thresholds["eq_diff_lo"],
                        inference_client=inference_client,
                        judge_provider=hybrid_judge_provider,
                        judge_model=hybrid_judge_model,
                        max_new_tokens=hybrid_eq_max_new_tokens,
                    )
                    pair_judgments = {
                        key: decision.label for key, decision in pairwise_decisions.items()
                    }
                    rec["semantic_pair_decisions"] = [
                        {
                            "i": left,
                            "j": right,
                            "label": decision.label,
                            "source": decision.source,
                            "source_detail": decision.source_detail,
                            "forward": _round_optional(decision.prob_forward, 4),
                            "reverse": _round_optional(decision.prob_reverse, 4),
                        }
                        for (left, right), decision in sorted(pairwise_decisions.items())
                    ]
                    semantic_entropy_result = compute_semantic_entropy(
                        question=question,
                        sample_answers=stochastic_answers,
                        pair_judgments=pair_judgments,
                    )
                else:
                    rec["semantic_pair_decisions"] = None
                    semantic_entropy_result = compute_semantic_entropy(
                        question=question,
                        sample_answers=stochastic_answers,
                        nli_judge=nli_judge,
                    )
                semantic_entropy = semantic_entropy_result.entropy
                semantic_entropy_norm = semantic_entropy_result.entropy_norm
                rec["semantic_entropy"] = round(semantic_entropy_result.entropy, 6)
                rec["semantic_entropy_norm"] = round(semantic_entropy_result.entropy_norm, 6)
                rec["n_semantic_clusters"] = semantic_entropy_result.n_clusters
                rec["semantic_cluster_ids"] = semantic_entropy_result.cluster_ids
                rec["semantic_cluster_sizes"] = semantic_entropy_result.cluster_sizes
            else:
                # No stochastic data -- legacy record or collection failure
                equiv_stats = EquivalenceStats(num_same=0, num_different=0, num_unclear=0, total=0)
                rec["equivalence_results"] = None
                rec["equivalence_stats"] = None
                rec["equivalence_ratio"] = None
                rec["semantic_entropy"] = None
                rec["semantic_entropy_norm"] = None
                rec["n_semantic_clusters"] = None
                rec["semantic_cluster_ids"] = None
                rec["semantic_cluster_sizes"] = None
                rec["equivalence_decision_source"] = None
                rec["equivalence_decision_source_detail"] = None
                rec["semantic_pair_decisions"] = None
                rec["stochastic_sample_grades"] = None
                rec["stochastic_sample_grade_source"] = None
                rec["stochastic_sample_grade_source_detail"] = None
                rec["stochastic_sample_grade_confidence"] = None
                rec["stochastic_correct_rate"] = None
                rec["stochastic_scored_n"] = None
                rec["stochastic_not_attempted_n"] = None
                rec["different_scored_n"] = None
                rec["different_correct_n"] = None
                rec["p_correct_given_different"] = None
                rec["equivalence_results_nli"] = None
                rec["equivalence_stats_nli"] = None
                rec["equivalence_ratio_nli"] = None
                rec["nli_equiv_probs_nli"] = None
                rec["hybrid_calibration_id"] = hybrid_calibration_id

            # ---- Step 3: Five-category labeling ----
            labels = classify_at_multiple_thresholds(
                is_correct=correctness_result.is_correct,
                equivalence_stats=equiv_stats,
                thresholds=[1.0, 0.9, 0.8, 0.7],
                unclear_treatment=unclear_treatment,
                grade=correctness_result.grade,
            )

            rec["error_label_1.0"] = labels[1.0]
            rec["error_label_0.9"] = labels[0.9]
            rec["error_label_0.8"] = labels[0.8]
            rec["error_label_0.7"] = labels[0.7]
            legacy_equiv_stats_dict = rec.get("equivalence_stats_nli")
            if legacy_equiv_stats_dict:
                legacy_equiv_stats_obj = EquivalenceStats.from_dict(legacy_equiv_stats_dict)
                legacy_labels = classify_at_multiple_thresholds(
                    is_correct=correctness_result.is_correct,
                    equivalence_stats=legacy_equiv_stats_obj,
                    thresholds=[1.0, 0.9, 0.8, 0.7],
                    unclear_treatment=unclear_treatment,
                    grade=correctness_result.grade,
                )
                rec["error_label_nli_1.0"] = legacy_labels[1.0]
                rec["error_label_nli_0.9"] = legacy_labels[0.9]
                rec["error_label_nli_0.8"] = legacy_labels[0.8]
                rec["error_label_nli_0.7"] = legacy_labels[0.7]
            else:
                rec["error_label_nli_1.0"] = None
                rec["error_label_nli_0.9"] = None
                rec["error_label_nli_0.8"] = None
                rec["error_label_nli_0.7"] = None

            semantic_entropy_label = classify_error_by_entropy(
                is_correct=correctness_result.is_correct,
                entropy=semantic_entropy,
                entropy_norm=semantic_entropy_norm,
                entropy_threshold=semantic_entropy_threshold,
                use_normalized_entropy=use_normalized_semantic_entropy,
                grade=correctness_result.grade,
            )
            rec["semantic_entropy_label"] = semantic_entropy_label

            # Trust-or-escalate routing: unclear, repeat disagreement, or incomplete samples.
            escalated = bool(rec.get("escalated_to_human", False))
            if correctness_result.is_unclear:
                escalated = True
            if repeat_consistency is not None and repeat_consistency < 1.0:
                escalated = True
            if rec.get("is_incomplete"):
                escalated = True
            rec["escalated_to_human"] = escalated
            if escalated:
                stats["escalated_to_human"] += 1
                retry_reason = "missing_precomputed_greedy_correctness" if missing_precomputed_greedy else "trust_or_escalate"
                storage_for_retry.enqueue_retry(
                    {
                        "question_id": rec.get("question_id"),
                        "model": rec.get("model"),
                        "reason": retry_reason,
                        "grade": correctness_result.grade,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    }
                )

            # Track stats at default threshold (0.9)
            label_09 = labels[0.9]
            if label_09 in stats:
                stats[label_09] += 1
            stats["evaluated"] += 1
            if rec.get("correctness_unclear"):
                stats["correctness_unclear"] += 1

            # Periodic checkpoint so interrupt/crash doesn't lose progress
            if (idx + 1) % checkpoint_every == 0:
                save_records(all_records, output_file)
                logger.debug(f"Checkpoint: saved {len(all_records)} records to {output_file}")

    except KeyboardInterrupt:
        save_records(all_records, output_file)
        logger.info(f"Interrupted. Partial results saved to {output_file} ({stats['evaluated']} evaluated). Re-run with --only-missing to resume.")
        raise

    # ---- Compute inter-rater reliability if ensemble was used ----
    reliability_info = None
    if all_ensemble_grades:
        from src.reliability import compute_ensemble_reliability
        reliability_info = compute_ensemble_reliability(all_ensemble_grades)
        stats["inter_rater_reliability"] = reliability_info
        logger.info(
            f"Inter-rater reliability (Krippendorff's alpha): "
            f"{reliability_info['krippendorff_alpha']:.4f} "
            f"(n={reliability_info['n_items']} items, "
            f"agreement={reliability_info['pairwise_agreement']:.1%})"
        )
        for rec in records:
            if rec.get("correctness_judge_grades") is not None:
                rec["inter_rater_alpha"] = reliability_info["krippendorff_alpha"]

    if strict_comparability:
        protocol_values = {
            rec.get("protocol_version")
            for rec in records
            if rec.get("protocol_version") is not None
        }
        if protocol_values != {protocol_version}:
            raise ValueError(
                f"strict comparability failed: expected protocol_version={protocol_version}, got {sorted(protocol_values)}"
            )
        judge_protocol_values = {
            rec.get("judge_protocol")
            for rec in records
            if rec.get("judge_protocol") is not None
        }
        if len(judge_protocol_values) != 1:
            raise ValueError(
                f"strict comparability failed: mixed judge protocols detected: {sorted(judge_protocol_values)}"
            )

    # ---- Save ALL records (evaluated records are modified in-place) ----
    save_records(all_records, output_file)

    # ---- Export to Parquet ----
    storage = ResultStorage(
        results_dir=results_dir,
        results_file=Path(output_file).name,
    )
    parquet_file = output_config.get("parquet_file", "results.parquet")
    parquet_path = storage.export_to_parquet(parquet_file)

    # ---- Export annotation sample if requested ----
    if export_annotation_sample > 0:
        from src.annotation import select_annotation_sample, export_annotation_sheet
        from datetime import datetime as dt
        sample = select_annotation_sample(
            [r for r in all_records if r.get("greedy_correct") is not None],
            n=export_annotation_sample,
        )
        ann_dir = Path(base_dir) / "annotations"
        ann_path = str(ann_dir / f"annotation_sample_{dt.now().strftime('%Y%m%d')}.csv")
        export_annotation_sheet(sample, ann_path, fmt="csv")
        logger.info(f"Exported {len(sample)} annotation records to {ann_path}")

    # ---- Print summary ----
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: RE-EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Run ID:                 {run_id or 'n/a'}")
    logger.info(f"Protocol version:       {protocol_version}")
    logger.info(f"Judge protocol:         {judge_protocol}")
    logger.info(f"Semantic-only mode:     {semantic_only}")
    logger.info(f"Skip greedy rejudge:    {skip_greedy_correctness}")
    logger.info(f"Hybrid enabled:         {hybrid_enabled}")
    logger.info(
        "Hybrid thresholds:      eq_same_hi=%.3f eq_diff_lo=%.3f corr_hi=%.3f corr_lo=%.3f",
        hybrid_thresholds["eq_same_hi"],
        hybrid_thresholds["eq_diff_lo"],
        hybrid_thresholds["corr_hi"],
        hybrid_thresholds["corr_lo"],
    )
    logger.info(f"Hybrid calibration id:  {hybrid_calibration_id or 'none'}")
    logger.info(f"Total records:          {stats['total']}")
    logger.info(f"Evaluated:              {stats['evaluated']}")
    logger.info(f"Skipped (already done): {stats['skipped']}")
    logger.info(f"")
    logger.info(f"Results at threshold 0.9:")
    logger.info(f"  Reliably correct:       {stats['reliably_correct']}")
    logger.info(f"  Fragile correct:        {stats['fragile_correct']}")
    logger.info(f"  Self-consistent error:  {stats['self_consistent_error']}")
    logger.info(f"  Inconsistent error:     {stats['inconsistent_error']}")
    logger.info(f"  Not attempted:          {stats['not_attempted']}")
    logger.info(f"  Judge unclear (total):  {stats['correctness_unclear']}")
    logger.info(f"  Escalated to human:     {stats['escalated_to_human']}")
    logger.info(f"  Missing precomputed:    {stats['missing_precomputed_greedy']}")
    logger.info(f"  Repeat judge items:     {stats['repeated_judge_items']}")
    logger.info(f"  Repeat disagreements:   {stats['repeat_inconsistent']}")
    logger.info(f"  Hybrid equiv LLM rows:  {stats['hybrid_equiv_llm_rows']}")
    logger.info(f"  Hybrid corr LLM rows:   {stats['hybrid_corr_llm_rows']}")
    if reliability_info:
        logger.info(f"")
        logger.info(f"Judge Ensemble Reliability:")
        logger.info(f"  Krippendorff's alpha:   {reliability_info['krippendorff_alpha']:.4f}")
        logger.info(f"  Pairwise agreement:     {reliability_info['pairwise_agreement']:.1%}")
        logger.info(f"  Grade distribution:     {reliability_info['grade_distribution']}")
    logger.info(f"")
    logger.info(f"Output: {output_file}")
    logger.info(f"Parquet: {parquet_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Re-evaluate collected results (correctness + equivalence + labeling)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input JSONL file (default: from config output.results_file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file (default: overwrite input in-place)",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only evaluate records missing complete Phase-2 outputs (safe resume mode).",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute all selected records even if already evaluated.",
    )
    parser.add_argument(
        "--strict-comparability",
        action="store_true",
        help="Enforce uniform protocol for selected records and fail on mixed settings.",
    )
    parser.add_argument(
        "--protocol-version",
        type=str,
        default=None,
        help="Protocol version tag to write into output records (default: config.protocol.version).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run id for resolving immutable raw/evaluated directories.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Filter to specific models (partial name match)",
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        default=True,
        help="Enable LLM-as-judge fallback in correctness cascade (default: enabled; costs money)",
    )
    parser.add_argument(
        "--no-llm-judge",
        dest="use_llm_judge",
        action="store_false",
        help="Disable LLM-as-judge (only valid with --semantic-only).",
    )
    parser.add_argument(
        "--export-annotation-sample",
        type=int,
        default=0,
        help="Export N records for human annotation after re-evaluation",
    )
    parser.add_argument(
        "--nli-calibration-file",
        type=str,
        default=None,
        help="Path to NLI calibration JSON (calibrated thresholds from human annotations)",
    )
    parser.add_argument(
        "--hybrid-calibration-file",
        type=str,
        default=None,
        help="Path to hybrid threshold calibration JSON (frozen eq/corr thresholds).",
    )
    parser.add_argument(
        "--skip-greedy-correctness",
        action="store_true",
        help="Reuse precomputed greedy correctness; do not re-run 3-judge greedy evaluation.",
    )
    parser.add_argument(
        "--semantic-only",
        action="store_true",
        help="Skip greedy correctness re-judging; recompute semantic equivalence + semantic entropy only.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Find config file
    config_path = args.config
    if not os.path.exists(config_path):
        parent_config = Path(__file__).parent.parent / config_path
        if parent_config.exists():
            config_path = str(parent_config)
        else:
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)

    try:
        run_reeval(
            config_path=config_path,
            input_file=args.input,
            output_file=args.output,
            only_missing=args.only_missing,
            force_recompute=args.force_recompute,
            strict_comparability=args.strict_comparability,
            protocol_version=args.protocol_version,
            run_id=args.run_id,
            models_filter=args.models,
            use_llm_judge=args.use_llm_judge,
            export_annotation_sample=args.export_annotation_sample,
            nli_calibration_file=args.nli_calibration_file,
            hybrid_calibration_file=args.hybrid_calibration_file,
            skip_greedy_correctness=args.skip_greedy_correctness,
            semantic_only=args.semantic_only,
        )
    except KeyboardInterrupt:
        logger.info("\nRe-evaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Re-evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
