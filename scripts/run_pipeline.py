#!/usr/bin/env python3
"""
Phase 1: Data Collection Pipeline for LLM Self-Consistent Error Measurement.

This script ONLY collects raw data (greedy + stochastic answers) without
evaluation. Evaluation is handled by scripts/reeval_results.py (Phase 2).

Two-phase architecture:
  Phase 1 (this script) – Expensive: makes model API calls, saves raw data.
  Phase 2 (reeval)      – Cheap: correctness cascade, equivalence, labeling.

For each question and model:
  1. Generate greedy answer (temperature ≈ 0)
  2. ALWAYS generate N stochastic samples (temperature = 0.7)
  3. Save raw record (no correctness/equivalence judgment yet)

Supports both commercial APIs (OpenAI, Anthropic, Google, xAI) and 
open-source models (via HuggingFace Inference API).
"""

import argparse
import hashlib
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Iterator, Tuple

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.providers import MultiProviderClient, build_qa_prompt
from src.truthfulqa import load_truthful_qa, load_truthful_qa_csv
from src.dataset import load_trivia_qa, load_boolq  # Backward compatibility + multi-benchmark
from src.contamination import ContaminationIndex
from src.storage import ResultStorage
from src.schemas import ResultRecord, Question

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

PROMPT_VERSION = "qa-short-v1"


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Return stable SHA-256 hash for the effective config."""
    payload = yaml.safe_dump(config, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_git_commit_hash() -> str:
    """Best-effort retrieval of current git commit hash."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def compute_dataset_item_hash(question_text: str, ground_truths: List[str]) -> str:
    """Hash a dataset item for reproducibility and drift detection."""
    joined = f"{question_text}\n---\n{chr(31).join(ground_truths)}"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_models_to_test(config: dict) -> List[Dict[str, Any]]:
    """
    Get list of models to test based on configuration.
    
    Returns list of dicts with 'provider', 'model', and 'name' keys.
    """
    models = []
    experiment_config = config.get("experiment", {})
    
    def _normalize(entry: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(entry)
        normalized["provider"] = entry["provider"]
        normalized["model"] = entry["model"]
        normalized["name"] = entry.get("name", f"{entry['provider']}/{entry['model']}")
        return normalized

    # Add commercial models if enabled
    if experiment_config.get("run_commercial", True):
        commercial = config.get("commercial_models", [])
        for m in commercial:
            models.append(_normalize(m))
    
    # Add open-source models if enabled
    if experiment_config.get("run_opensource", True):
        opensource = config.get("opensource_models", [])
        for m in opensource:
            models.append(_normalize(m))
    
    # Backward compatibility: check for old format
    if not models and "models_to_test" in config:
        for model_id in config["models_to_test"]:
            models.append({
                "provider": "huggingface",
                "model": model_id,
                "name": model_id
            })
    
    return models


def get_benchmark_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return benchmark configs, supporting both v2 and v3 config formats."""
    dataset_config = config.get("dataset", {})
    benchmarks = dataset_config.get("benchmarks")
    if isinstance(benchmarks, list) and benchmarks:
        return benchmarks
    return [dataset_config]


def load_questions_for_benchmark(benchmark_config: Dict[str, Any]) -> Iterator[Question]:
    """Load questions for one benchmark config."""
    dataset_name = benchmark_config.get("name", "truthful_qa")

    if dataset_name in ["truthful_qa", "truthfulqa"]:
        if benchmark_config.get("source") == "csv" or benchmark_config.get("path", "").endswith(".csv"):
            csv_path = benchmark_config.get("path", "TruthfulQA.csv")
            logger.info(f"Loading TruthfulQA from CSV: {csv_path}")
            return load_truthful_qa_csv(
                csv_path=csv_path,
                max_questions=benchmark_config.get("max_questions"),
                categories=benchmark_config.get("categories"),
            )
        logger.info("Loading TruthfulQA dataset (Hugging Face)")
        return load_truthful_qa(
            split=benchmark_config.get("split", "validation"),
            max_questions=benchmark_config.get("max_questions"),
            categories=benchmark_config.get("categories"),
        )
    if dataset_name in ["trivia_qa", "triviaqa"]:
        logger.info("Loading TriviaQA dataset")
        return load_trivia_qa(
            subset=benchmark_config.get("subset", "rc"),
            split=benchmark_config.get("split", "validation"),
            max_questions=benchmark_config.get("max_questions", 50),
        )
    if dataset_name in ["boolq", "bool_q"]:
        logger.info("Loading BoolQ dataset")
        return load_boolq(
            split=benchmark_config.get("split", "validation"),
            max_questions=benchmark_config.get("max_questions"),
        )
    raise ValueError(
        f"Unknown dataset: {dataset_name}. Supported: truthful_qa, trivia_qa, boolq"
    )


def check_api_keys(models: List[Dict[str, Any]], judge_config: dict) -> Dict[str, bool]:
    """
    Check which API keys are available.
    
    Returns dict mapping provider names to availability status.
    """
    api_keys = {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "google": bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
        "xai": bool(os.environ.get("XAI_API_KEY")),
        "deepseek": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "groq": bool(os.environ.get("GROQ_API_KEY")),
        "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
        "huggingface": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")),
        "huggingface_local": True,
    }
    
    # Check which providers are needed
    needed_providers = set()
    for m in models:
        needed_providers.add(m["provider"].lower())
    needed_providers.add(judge_config.get("provider", "huggingface").lower())
    
    # Report status
    missing = []
    for provider in needed_providers:
        if provider in api_keys and not api_keys[provider]:
            missing.append(provider)
    
    if missing:
        logger.warning(f"Missing API keys for providers: {missing}")
        logger.warning("Some models will be skipped. Set the required keys in .env file.")
    
    return api_keys


def run_pipeline(
    config_path: str,
    dry_run: bool = False,
    models_filter: List[str] = None,
    run_id: str = None,
    strict_samples: bool = False,
    max_provider_concurrency: int = None,
    resume_any_existing: bool = False,
):
    """
    Phase 1: Data Collection Pipeline.

    Collects raw greedy + stochastic answers for every question-model pair.
    Does NOT perform correctness checking, equivalence, or labeling —
    that is handled by reeval_results.py (Phase 2).

    Args:
        config_path: Path to config.yaml
        dry_run: If True, only load data and validate without making API calls
        models_filter: Optional list of model names to run (for selective testing)
        resume_any_existing: If True, treat any existing row as complete (legacy mode)
    """
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    protocol_config = config.get("protocol", {})
    collection_config = config.get("collection", {})
    output_config = config.get("output", {})
    dataset_config = config.get("dataset", {})

    protocol_version = str(protocol_config.get("version", "v3"))
    high_rigor = bool(protocol_config.get("high_rigor", True))
    prompt_version = str(protocol_config.get("prompt_version", PROMPT_VERSION))
    run_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    run_date = run_timestamp[:10]
    config_hash = compute_config_hash(config)
    git_commit = get_git_commit_hash()
    if run_id is None:
        run_id = protocol_config.get(
            "run_id",
            f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{config_hash[:8]}",
        )

    # Get models to test
    all_models = get_models_to_test(config)
    if models_filter:
        all_models = [m for m in all_models if any(f.lower() in m["name"].lower() for f in models_filter)]
        logger.info(f"Filtered to {len(all_models)} models matching: {models_filter}")
    if not all_models:
        logger.error("No models configured to test!")
        return
    logger.info(f"Models to test: {[m['name'] for m in all_models]}")

    # API key checks
    judge_config = config.get("judge", {})
    api_keys = check_api_keys(all_models, judge_config)
    models_to_test = []
    for m in all_models:
        provider = m["provider"].lower()
        provider_available = api_keys.get(provider, True)
        if provider_available:
            models_to_test.append(m)
        else:
            logger.warning(f"Skipping {m['name']}: No API key for {provider}")
    if not models_to_test:
        logger.error("No models available to test (all providers missing API keys)!")
        return

    # Inference settings
    inference_config = config.get("inference", {})
    greedy_config = inference_config.get("greedy", {})
    stochastic_config = inference_config.get("stochastic", {})
    rate_config = config.get("rate_limit", {})
    required_samples = int(collection_config.get("required_samples", stochastic_config.get("num_samples", 10)))
    strict_samples_effective = bool(strict_samples or collection_config.get("strict_samples", False) or high_rigor)
    retry_incomplete = bool(collection_config.get("retry_incomplete", True))
    configured_max_conc = collection_config.get("max_concurrency_per_provider")
    if max_provider_concurrency is None and configured_max_conc is not None:
        max_provider_concurrency = int(configured_max_conc)

    # Output settings
    base_dir = output_config.get("results_dir", "data/results")
    raw_dir = output_config.get("raw_dir", "raw")
    immutable_runs = bool(output_config.get("immutable_runs", high_rigor))
    results_file = output_config.get("results_file", "results.jsonl")
    if immutable_runs:
        results_dir = str(Path(base_dir) / raw_dir / run_id)
    else:
        results_dir = str(Path(base_dir) / raw_dir)

    storage = ResultStorage(results_dir=results_dir, results_file=results_file)
    resume_validation_cfg = collection_config.get("resume_validation", {}) or {}
    resume_require_non_empty_greedy = bool(
        resume_validation_cfg.get("require_non_empty_greedy", True)
    )
    resume_require_non_empty_stochastic = bool(
        resume_validation_cfg.get("require_non_empty_stochastic", True)
    )
    resume_reject_incomplete_flag = bool(
        resume_validation_cfg.get("reject_incomplete_flag", True)
    )
    resume_require_required_samples = bool(
        resume_validation_cfg.get("require_required_samples", strict_samples_effective)
    )
    resume_required_samples = required_samples if resume_require_required_samples else None
    if resume_any_existing:
        completed_pairs = storage.get_completed_pairs()
        logger.info(
            "Resume mode: any existing row counts as complete (legacy). "
            "Use with caution on interrupted runs."
        )
    else:
        completed_pairs = storage.get_completed_pairs(
            required_samples=resume_required_samples,
            require_non_empty_greedy=resume_require_non_empty_greedy,
            require_non_empty_stochastic=resume_require_non_empty_stochastic,
            reject_incomplete_flag=resume_reject_incomplete_flag,
        )
        logger.info(
            "Resume mode: valid-only "
            "(required_samples=%s, non_empty_greedy=%s, non_empty_stochastic=%s, reject_incomplete_flag=%s)",
            str(resume_required_samples),
            resume_require_non_empty_greedy,
            resume_require_non_empty_stochastic,
            resume_reject_incomplete_flag,
        )
    if completed_pairs:
        logger.info(f"Found {len(completed_pairs)} completed question-model pairs. Will resume.")

    inference_client = MultiProviderClient(
        initial_delay=rate_config.get("initial_delay", 2.0),
        max_delay=rate_config.get("max_delay", 60.0),
        backoff_factor=rate_config.get("backoff_factor", 2.0),
        max_concurrency_per_provider=max_provider_concurrency,
    )

    # Load benchmark questions
    benchmark_configs = get_benchmark_configs(config)
    contamination_cfg = config.get("contamination", {})
    contamination_index = ContaminationIndex(
        similarity_threshold=float(contamination_cfg.get("similarity_threshold", 0.93)),
        jaccard_threshold=float(contamination_cfg.get("jaccard_threshold", 0.85)),
        lookback=int(contamination_cfg.get("lookback", 5000)),
    )
    loaded_questions: List[Dict[str, Any]] = []
    benchmark_summaries: List[Dict[str, Any]] = []
    for bench in benchmark_configs:
        bench_name = str(bench.get("name", dataset_config.get("name", "truthful_qa")))
        bench_split = str(bench.get("split", dataset_config.get("split", "validation")))
        logger.info(f"Loading benchmark: {bench_name} ({bench_split})")
        questions = list(load_questions_for_benchmark(bench))
        for q in questions:
            source_key = f"{bench_name}:{bench_split}"
            contam = contamination_index.check(q.text, source_key)
            contamination_index.add(q.text, source_key)
            loaded_questions.append(
                {
                    "question": q,
                    "dataset_name": bench_name,
                    "dataset_split": bench_split,
                    "contamination_flag": contam.is_contaminated,
                    "contamination_reason": contam.reason,
                }
            )
        benchmark_summaries.append(
            {
                "name": bench_name,
                "split": bench_split,
                "max_questions": bench.get("max_questions"),
                "num_questions_loaded": len(questions),
            }
        )
    logger.info(f"Loaded {len(loaded_questions)} total benchmark questions")

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE – Phase 1 data collection validation")
        logger.info("=" * 60)
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Protocol version: {protocol_version} (high_rigor={high_rigor})")
        logger.info(f"Models to test ({len(models_to_test)}):")
        for m in models_to_test:
            track_info = f", track={m.get('track')}" if m.get("track") else ""
            release_info = f", release={m.get('release_date')}" if m.get("release_date") else ""
            logger.info(f"  - {m['name']} ({m['provider']}{track_info}{release_info})")
        logger.info(f"Benchmarks loaded: {len(benchmark_summaries)}")
        for b in benchmark_summaries:
            logger.info(f"  - {b['name']} ({b['split']}): {b['num_questions_loaded']} questions")
        logger.info(f"Total question records: {len(loaded_questions)}")
        logger.info(f"Stochastic target samples: {required_samples}")
        logger.info(f"Stochastic temperature: {stochastic_config.get('temperature', 1.0)}")
        logger.info(f"Strict sample enforcement: {strict_samples_effective}")
        logger.info(f"Max provider concurrency: {max_provider_concurrency}")
        logger.info("=" * 60)
        logger.info("Dry run complete. Remove --dry-run to execute pipeline.")
        return

    stats = {
        "total_collected": 0,
        "skipped_completed": 0,
        "errors": 0,
        "incomplete_records": 0,
        "by_model": {},
    }
    for m in models_to_test:
        stats["by_model"][m["name"]] = {"collected": 0, "errors": 0, "incomplete": 0}

    run_metadata = {
        "run_timestamp": run_timestamp,
        "run_date": run_date,
        "run_id": run_id,
        "protocol_version": protocol_version,
        "phase": "data_collection",
        "high_rigor": high_rigor,
        "config_path": config_path,
        "config_hash": config_hash,
        "git_commit": git_commit,
        "prompt_version": prompt_version,
        "models": [m["name"] for m in models_to_test],
        "model_ids": [
            {
                "name": m["name"],
                "provider": m["provider"],
                "model": m["model"],
                "snapshot_id": m.get("snapshot_id"),
                "release_date": m.get("release_date"),
                "track": m.get("track"),
                "family": m.get("family"),
                "version_index": m.get("version_index"),
                "request_overrides": m.get("request_overrides"),
            }
            for m in models_to_test
        ],
        "benchmarks": benchmark_summaries,
        "inference": {
            "greedy": dict(greedy_config),
            "stochastic": dict(stochastic_config),
            "required_samples": required_samples,
            "strict_samples": strict_samples_effective,
            "max_concurrency_per_provider": max_provider_concurrency,
        },
    }
    storage.write_run_metadata(run_metadata)

    total_iterations = len(loaded_questions) * len(models_to_test)
    pbar = tqdm(total=total_iterations, desc="Collecting")

    for qinfo in loaded_questions:
        question: Question = qinfo["question"]
        dataset_name = qinfo["dataset_name"]
        dataset_split = qinfo["dataset_split"]
        contamination_flag = qinfo["contamination_flag"]
        contamination_reason = qinfo["contamination_reason"]

        for model_config in models_to_test:
            provider = model_config["provider"]
            model = model_config["model"]
            model_name = model_config["name"]
            request_overrides = model_config.get("request_overrides")

            pbar.set_description(f"Q:{question.id[:15]}... M:{model_name[:20]}")

            pair_key = (question.id, model_name)
            if pair_key in completed_pairs:
                stats["skipped_completed"] += 1
                pbar.update(1)
                continue

            try:
                prompt = build_qa_prompt(question.text, model)

                greedy_result = inference_client.generate_greedy(
                    provider=provider,
                    model=model,
                    prompt=prompt,
                    max_new_tokens=greedy_config.get("max_new_tokens", 100),
                    request_overrides=request_overrides,
                )
                greedy_answer = greedy_result.text.strip()

                target_samples = required_samples if strict_samples_effective else int(
                    stochastic_config.get("num_samples", required_samples)
                )
                stochastic_results = inference_client.generate_stochastic(
                    provider=provider,
                    model=model,
                    prompt=prompt,
                    num_samples=target_samples,
                    temperature=stochastic_config.get("temperature", 0.7),
                    top_p=stochastic_config.get("top_p", 0.9),
                    max_new_tokens=stochastic_config.get("max_new_tokens", 100),
                    request_overrides=request_overrides,
                )
                stochastic_answers = [r.text.strip() for r in stochastic_results]
                sample_metadata = [r.request_meta or {} for r in stochastic_results]
                actual_samples = len(stochastic_answers)
                is_incomplete = actual_samples < required_samples

                if is_incomplete:
                    stats["incomplete_records"] += 1
                    stats["by_model"][model_name]["incomplete"] += 1
                    if retry_incomplete:
                        storage.enqueue_retry(
                            {
                                "run_id": run_id,
                                "question_id": question.id,
                                "model": model_name,
                                "dataset_name": dataset_name,
                                "dataset_split": dataset_split,
                                "reason": f"incomplete_samples:{actual_samples}/{required_samples}",
                                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                            }
                        )

                record = ResultRecord(
                    question_id=question.id,
                    question=question.text,
                    ground_truth=question.ground_truths,
                    model=model_name,
                    greedy_answer=greedy_answer,
                    greedy_correct=None,
                    correctness_match_type=None,
                    stochastic_answers=stochastic_answers,
                    equivalence_results=None,
                    equivalence_stats=None,
                    equivalence_ratio=None,
                    error_label_1_0=None,
                    error_label_0_9=None,
                    error_label_0_8=None,
                    error_label_0_7=None,
                    run_id=run_id,
                    protocol_version=protocol_version,
                    config_hash=config_hash,
                    prompt_version=prompt_version,
                    run_date=run_date,
                    dataset_name=dataset_name,
                    dataset_split=dataset_split,
                    dataset_item_hash=compute_dataset_item_hash(question.text, question.ground_truths),
                    model_provider=provider,
                    model_id=model,
                    model_snapshot_id=model_config.get("snapshot_id"),
                    model_release_date=model_config.get("release_date"),
                    model_track=model_config.get("track"),
                    model_family=model_config.get("family"),
                    model_version_index=model_config.get("version_index"),
                    stochastic_target_n=required_samples,
                    stochastic_actual_n=actual_samples,
                    is_incomplete=is_incomplete,
                    sample_metadata=sample_metadata,
                    contamination_flag=contamination_flag,
                    contamination_reason=contamination_reason,
                )
                storage.save_record(record)
                stats["total_collected"] += 1
                stats["by_model"][model_name]["collected"] += 1

            except Exception as e:
                logger.error(f"Error collecting {question.id} with {model_name}: {e}")
                stats["errors"] += 1
                stats["by_model"][model_name]["errors"] += 1
                storage.log_failed_pair(question.id, model_name, str(e))
                storage.enqueue_retry(
                    {
                        "run_id": run_id,
                        "question_id": question.id,
                        "model": model_name,
                        "dataset_name": dataset_name,
                        "dataset_split": dataset_split,
                        "reason": f"collection_exception:{e}",
                        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    }
                )

            pbar.update(1)

    pbar.close()
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: DATA COLLECTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Protocol version: {protocol_version}")
    logger.info(f"Total collected: {stats['total_collected']}")
    logger.info(f"Skipped (already done): {stats['skipped_completed']}")
    logger.info(f"API errors: {stats['errors']}")
    logger.info(f"Incomplete records: {stats['incomplete_records']}")
    
    for model_name, ms in stats["by_model"].items():
        logger.info(
            f"  {model_name}: {ms['collected']} collected, "
            f"{ms['errors']} errors, {ms['incomplete']} incomplete"
        )
    
    # Export raw data to Parquet
    parquet_path = storage.export_to_parquet(output_config.get("parquet_file", "results.parquet"))
    logger.info(f"\nExported raw data to {parquet_path}")
    logger.info("\nNext step: run scripts/reeval_results.py on this run directory for strict re-evaluation.")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM Self-Consistent Error Measurement Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without making API calls"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Filter to specific models (partial name match)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Deterministic run id. If omitted, generated from UTC timestamp + config hash.",
    )
    parser.add_argument(
        "--strict-samples",
        action="store_true",
        help="Enforce strict sample cardinality target (required_samples).",
    )
    parser.add_argument(
        "--max-provider-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent stochastic requests per provider.",
    )
    parser.add_argument(
        "--resume-any-existing",
        action="store_true",
        help=(
            "Legacy resume behavior: treat any existing row as complete. "
            "Default uses valid-only resume (safer for interrupted runs)."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Find config file
    config_path = args.config
    if not os.path.exists(config_path):
        # Try looking in parent directory
        parent_config = Path(__file__).parent.parent / config_path
        if parent_config.exists():
            config_path = str(parent_config)
        else:
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)
    
    try:
        run_pipeline(
            config_path,
            dry_run=args.dry_run,
            models_filter=args.models,
            run_id=args.run_id,
            strict_samples=args.strict_samples,
            max_provider_concurrency=args.max_provider_concurrency,
            resume_any_existing=args.resume_any_existing,
        )
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
