#!/usr/bin/env python3
"""
Targeted Phase-1 backfill for flagged or missing raw rows.

Re-collects greedy + stochastic generations for selected
(question_id, model) pairs and merges them into a repaired clean JSONL.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# Allow `from src...` imports when run from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.providers import MultiProviderClient, build_qa_prompt
from src.storage import ResultStorage


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def compute_config_hash(config: Dict[str, Any]) -> str:
    payload = yaml.safe_dump(config, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
                raise ValueError(f"Invalid JSONL at {path}:{line_num}: {exc}") from exc
    return rows


def pair_key(rec: Dict[str, Any]) -> Tuple[str, str]:
    return (str(rec.get("question_id", "")), str(rec.get("model", "")))


def get_models_to_test(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    experiment_config = config.get("experiment", {})

    if experiment_config.get("run_commercial", True):
        for m in config.get("commercial_models", []):
            models.append(
                {
                    "provider": m["provider"],
                    "model": m["model"],
                    "name": m.get("name", f"{m['provider']}/{m['model']}"),
                }
            )

    if experiment_config.get("run_opensource", True):
        for m in config.get("opensource_models", []):
            models.append(
                {
                    "provider": m["provider"],
                    "model": m["model"],
                    "name": m.get("name", f"{m['provider']}/{m['model']}"),
                }
            )
    return models


def build_model_lookup(
    config: Dict[str, Any],
    gemini_model: str,
) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    for m in get_models_to_test(config):
        lookup[m["name"]] = {"provider": m["provider"], "model": m["model"]}

    # Ensure historical row names resolve even if config names changed.
    defaults = {
        "GPT-5.2 (OpenAI)": {"provider": "openai", "model": "gpt-5.2"},
        "Claude Opus 4.6 (Anthropic)": {"provider": "anthropic", "model": "claude-opus-4-6"},
        "Qwen3 Next 80B (OpenRouter)": {
            "provider": "openrouter",
            "model": "qwen/qwen3-next-80b-a3b-instruct",
        },
    }
    for name, meta in defaults.items():
        lookup.setdefault(name, meta)

    # User-requested Google model for backfill.
    lookup["Gemini 3 Flash (Google)"] = {"provider": "google", "model": gemini_model}
    return lookup


def detect_flagged_rows(
    source_rows: List[Dict[str, Any]],
    required_samples: int,
) -> List[Dict[str, Any]]:
    flagged: List[Dict[str, Any]] = []
    for rec in source_rows:
        greedy = str(rec.get("greedy_answer") or "").strip()
        stochastic = rec.get("stochastic_answers") or []
        if not greedy or len(stochastic) < required_samples:
            flagged.append(rec)
    return flagged


def detect_missing_rows(
    clean_rows: List[Dict[str, Any]],
    expected_models: List[str],
) -> List[Dict[str, Any]]:
    """Build synthetic target rows for missing (question_id, model) pairs."""
    existing_pairs = {pair_key(r) for r in clean_rows}
    question_templates: Dict[str, Dict[str, Any]] = {}
    for rec in clean_rows:
        qid = str(rec.get("question_id", ""))
        if qid and qid not in question_templates:
            question_templates[qid] = rec

    missing: List[Dict[str, Any]] = []
    for qid in sorted(question_templates.keys(), key=question_sort_key):
        template = question_templates[qid]
        for model_name in expected_models:
            if (qid, model_name) in existing_pairs:
                continue
            missing.append(
                {
                    "question_id": qid,
                    "question": template.get("question", ""),
                    "ground_truth": template.get("ground_truth", []),
                    "model": model_name,
                    "greedy_answer": "",
                    "greedy_correct": None,
                    "correctness_match_type": None,
                    "stochastic_answers": [],
                    "equivalence_results": None,
                    "equivalence_stats": None,
                    "equivalence_ratio": None,
                    "error_label_1.0": None,
                    "error_label_0.9": None,
                    "error_label_0.8": None,
                    "error_label_0.7": None,
                    "dataset_name": template.get("dataset_name"),
                    "dataset_split": template.get("dataset_split"),
                    "dataset_item_hash": template.get("dataset_item_hash"),
                    "contamination_flag": template.get("contamination_flag"),
                    "contamination_reason": template.get("contamination_reason"),
                }
            )
    return missing


def detect_empty_stochastic_rows(clean_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select rows where any stochastic sample is empty/blank."""
    bad: List[Dict[str, Any]] = []
    for rec in clean_rows:
        samples = rec.get("stochastic_answers")
        if not isinstance(samples, list):
            continue
        if any(not str(x or "").strip() for x in samples):
            bad.append(rec)
    return bad


def question_sort_key(question_id: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)$", question_id)
    if m:
        return (int(m.group(1)), question_id)
    return (10**9, question_id)


def generate_greedy_non_empty(
    client: MultiProviderClient,
    provider: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    max_attempts: int,
) -> Any:
    """Generate greedy answer and retry if the model returns empty text."""
    for attempt in range(1, max_attempts + 1):
        result = client.generate_greedy(
            provider=provider,
            model=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        if str(result.text).strip():
            return result
        logger.warning(
            "Empty greedy answer for %s/%s (attempt %d/%d). Retrying.",
            provider,
            model_id,
            attempt,
            max_attempts,
        )
    raise RuntimeError(
        f"Greedy answer remained empty after {max_attempts} attempts for {provider}/{model_id}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill flagged or missing raw rows with live API calls.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode",
        choices=["auto", "flagged", "missing", "empty-stochastic"],
        default="auto",
        help=(
            "Target selection mode: "
            "flagged=repair empty/incomplete rows, "
            "missing=repair absent (question_id,model) pairs, "
            "empty-stochastic=repair rows whose stochastic samples contain blanks, "
            "auto=flagged first, then missing, then empty-stochastic."
        ),
    )
    parser.add_argument(
        "--source",
        default="data/results/raw/results_v2_joined_with_backup_and_raw.jsonl",
        help="Source raw JSONL (used for flagged-row lookup when available).",
    )
    parser.add_argument(
        "--clean",
        default="data/results/raw/results_v2_joined_with_backup_and_raw_clean.jsonl",
        help="Base clean JSONL to merge into.",
    )
    parser.add_argument(
        "--flagged",
        default="data/results/raw/results_v2_joined_with_backup_and_raw_flagged.jsonl",
        help="Flagged rows JSONL; if missing/empty, rows are auto-detected from --source.",
    )
    parser.add_argument(
        "--backfill-file",
        default="data/results/raw/results_v2_backfilled_rows.jsonl",
        help="Output JSONL with only recollected rows.",
    )
    parser.add_argument(
        "--output",
        default="data/results/raw/results_v2_joined_with_backup_and_raw_clean_backfilled.jsonl",
        help="Merged clean+backfilled output JSONL.",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-3-flash-preview",
        help="Google model id used for rows labeled 'Gemini 3 Flash (Google)'.",
    )
    parser.add_argument(
        "--required-samples",
        type=int,
        default=None,
        help="Override required stochastic sample count.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run id to stamp into backfilled records (default: auto-generated).",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow saving rows with stochastic_actual_n < required_samples.",
    )
    parser.add_argument(
        "--allow-empty-greedy",
        action="store_true",
        help="Allow empty greedy answers (default: reject and retry).",
    )
    parser.add_argument(
        "--max-greedy-attempts",
        type=int,
        default=3,
        help="Maximum greedy retries when an empty answer is returned.",
    )
    parser.add_argument(
        "--allow-empty-stochastic",
        action="store_true",
        help="Allow empty stochastic sample strings (default: reject and retry row).",
    )
    parser.add_argument(
        "--max-row-attempts",
        type=int,
        default=3,
        help="Maximum attempts to recollect one row when validation fails.",
    )
    parser.add_argument(
        "--repair-empty-samples-only",
        action="store_true",
        help=(
            "When mode=empty-stochastic, refill only blank stochastic slots "
            "instead of regenerating the full 10-sample row."
        ),
    )
    parser.add_argument(
        "--max-empty-sample-attempts",
        type=int,
        default=10,
        help="Maximum retries to refill one empty stochastic sample slot.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned backfill pairs without API calls.",
    )
    args = parser.parse_args()

    # Load env from CWD and repo root.
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

    config = load_config(args.config)
    protocol_config = config.get("protocol", {})
    inference_cfg = config.get("inference", {})
    greedy_cfg = inference_cfg.get("greedy", {})
    stochastic_cfg = inference_cfg.get("stochastic", {})
    collection_cfg = config.get("collection", {})
    rate_cfg = config.get("rate_limit", {})
    required_samples = int(
        args.required_samples
        if args.required_samples is not None
        else collection_cfg.get("required_samples", stochastic_cfg.get("num_samples", 10))
    )
    strict_samples = not args.allow_incomplete

    clean_rows = load_jsonl(args.clean)
    source_rows: List[Dict[str, Any]] = []
    if os.path.exists(args.source):
        source_rows = load_jsonl(args.source)

    targets: List[Dict[str, Any]] = []
    target_mode_used = ""

    if args.mode in ("auto", "flagged"):
        flagged_rows: List[Dict[str, Any]] = []
        if os.path.exists(args.flagged):
            flagged_rows = load_jsonl(args.flagged)
        if not flagged_rows and source_rows:
            flagged_rows = detect_flagged_rows(source_rows, required_samples)

        if flagged_rows:
            source_index = {pair_key(r): r for r in source_rows}
            targets = [source_index.get(pair_key(r), r) for r in flagged_rows]
            target_mode_used = "flagged"
        elif args.mode == "flagged":
            logger.info("No flagged rows found.")
            return

    if not targets and args.mode in ("auto", "missing"):
        expected_models = sorted({str(r.get("model", "")) for r in clean_rows if str(r.get("model", ""))})
        targets = detect_missing_rows(clean_rows, expected_models)
        target_mode_used = "missing"

    if not targets and args.mode in ("auto", "empty-stochastic"):
        targets = detect_empty_stochastic_rows(clean_rows)
        target_mode_used = "empty-stochastic"

    if not targets:
        logger.info("No target rows found. Nothing to backfill.")
        return

    model_lookup = build_model_lookup(config, gemini_model=args.gemini_model)

    missing_model_meta = sorted({r.get("model", "") for r in targets if r.get("model", "") not in model_lookup})
    if missing_model_meta:
        raise ValueError(
            "Missing provider/model mapping for model names: "
            + ", ".join(missing_model_meta)
        )

    logger.info("Target mode: %s", target_mode_used or args.mode)
    logger.info("Planned backfill rows: %d", len(targets))
    for rec in targets:
        name = rec.get("model", "")
        meta = model_lookup[name]
        logger.info(
            "  %s | %s -> %s/%s",
            rec.get("question_id", ""),
            name,
            meta["provider"],
            meta["model"],
        )

    if args.dry_run:
        logger.info("Dry run complete (no API calls made).")
        return

    run_id = args.run_id or f"run_backfill_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    protocol_version = str(protocol_config.get("version", "v3"))
    prompt_version = str(protocol_config.get("prompt_version", "qa-short-v1"))
    config_hash = compute_config_hash(config)
    dataset_defaults = config.get("dataset", {})

    client = MultiProviderClient(
        initial_delay=float(rate_cfg.get("initial_delay", 2.0)),
        max_delay=float(rate_cfg.get("max_delay", 60.0)),
        backoff_factor=float(rate_cfg.get("backoff_factor", 2.0)),
        max_concurrency_per_provider=collection_cfg.get("max_concurrency_per_provider"),
    )

    backfilled_rows: List[Dict[str, Any]] = []
    for rec in tqdm(targets, desc="Backfilling"):
        model_name = str(rec.get("model", ""))
        model_meta = model_lookup[model_name]
        provider = model_meta["provider"]
        model_id = model_meta["model"]
        question = str(rec.get("question", ""))
        prompt = build_qa_prompt(question, model_id)

        if target_mode_used == "empty-stochastic" and args.repair_empty_samples_only:
            out = dict(rec)
            stochastic_answers = [str(x or "").strip() for x in (rec.get("stochastic_answers") or [])]
            if len(stochastic_answers) != required_samples:
                raise RuntimeError(
                    f"Expected {required_samples} stochastic samples for slot repair, "
                    f"got {len(stochastic_answers)} on {rec.get('question_id')} / {model_name}"
                )
            sample_metadata = list(rec.get("sample_metadata") or [])
            while len(sample_metadata) < required_samples:
                sample_metadata.append({})

            for idx, sample in enumerate(stochastic_answers):
                if sample:
                    continue
                filled = False
                for attempt in range(1, max(1, int(args.max_empty_sample_attempts)) + 1):
                    one = client.generate_stochastic(
                        provider=provider,
                        model=model_id,
                        prompt=prompt,
                        num_samples=1,
                        temperature=float(stochastic_cfg.get("temperature", 0.7)),
                        top_p=float(stochastic_cfg.get("top_p", 0.9)),
                        max_new_tokens=int(stochastic_cfg.get("max_new_tokens", 100)),
                    )
                    candidate = str(one[0].text or "").strip() if one else ""
                    if candidate:
                        stochastic_answers[idx] = candidate
                        meta = one[0].request_meta or {}
                        meta["sample_index"] = idx
                        sample_metadata[idx] = meta
                        filled = True
                        break
                    logger.warning(
                        "Empty stochastic slot retry %d/%d for %s | %s at index %d",
                        attempt,
                        int(args.max_empty_sample_attempts),
                        rec.get("question_id"),
                        model_name,
                        idx,
                    )
                if not filled:
                    raise RuntimeError(
                        f"Failed to refill empty stochastic slot index {idx} "
                        f"for {rec.get('question_id')} / {model_name} after "
                        f"{int(args.max_empty_sample_attempts)} attempts."
                    )

            # Preserve greedy answer in slot-repair mode.
            out["stochastic_answers"] = stochastic_answers
            out["sample_metadata"] = sample_metadata
            out["timestamp"] = utc_now()
            out["run_id"] = run_id
            out["protocol_version"] = protocol_version
            out["config_hash"] = config_hash
            out["prompt_version"] = prompt_version
            out["model_provider"] = provider
            out["model_id"] = model_id
            out["stochastic_target_n"] = required_samples
            out["stochastic_actual_n"] = required_samples
            out["is_incomplete"] = False
            out.setdefault("dataset_name", dataset_defaults.get("name", "truthful_qa"))
            out.setdefault("dataset_split", dataset_defaults.get("split", "validation"))
            backfilled_rows.append(out)
            continue

        greedy_result: Any = None
        stochastic_answers: List[str] = []
        sample_metadata: List[Dict[str, Any]] = []
        actual_n = 0
        is_incomplete = False
        last_failure: str = ""
        max_row_attempts = max(1, int(args.max_row_attempts))

        for row_attempt in range(1, max_row_attempts + 1):
            if args.allow_empty_greedy:
                greedy_result = client.generate_greedy(
                    provider=provider,
                    model=model_id,
                    prompt=prompt,
                    max_new_tokens=int(greedy_cfg.get("max_new_tokens", 100)),
                )
            else:
                greedy_result = generate_greedy_non_empty(
                    client=client,
                    provider=provider,
                    model_id=model_id,
                    prompt=prompt,
                    max_new_tokens=int(greedy_cfg.get("max_new_tokens", 100)),
                    max_attempts=max(1, int(args.max_greedy_attempts)),
                )

            stochastic_results = client.generate_stochastic(
                provider=provider,
                model=model_id,
                prompt=prompt,
                num_samples=required_samples,
                temperature=float(stochastic_cfg.get("temperature", 0.7)),
                top_p=float(stochastic_cfg.get("top_p", 0.9)),
                max_new_tokens=int(stochastic_cfg.get("max_new_tokens", 100)),
            )

            stochastic_answers = [str(r.text).strip() for r in stochastic_results]
            sample_metadata = [r.request_meta or {} for r in stochastic_results]
            actual_n = len(stochastic_answers)
            is_incomplete = actual_n < required_samples
            empty_indices = [i for i, x in enumerate(stochastic_answers) if not str(x).strip()]

            if strict_samples and is_incomplete:
                last_failure = (
                    f"incomplete samples {actual_n}/{required_samples}"
                )
                logger.warning(
                    "Row retry %d/%d for %s | %s due to %s",
                    row_attempt,
                    max_row_attempts,
                    rec.get("question_id"),
                    model_name,
                    last_failure,
                )
                continue

            if (not args.allow_empty_stochastic) and empty_indices:
                last_failure = (
                    f"empty stochastic samples at indices {empty_indices}"
                )
                logger.warning(
                    "Row retry %d/%d for %s | %s due to %s",
                    row_attempt,
                    max_row_attempts,
                    rec.get("question_id"),
                    model_name,
                    last_failure,
                )
                continue

            last_failure = ""
            break

        if last_failure:
            raise RuntimeError(
                f"Failed to collect valid row for {rec.get('question_id')} / {model_name} "
                f"after {max_row_attempts} attempts: {last_failure}"
            )

        out = dict(rec)
        out["greedy_answer"] = str(greedy_result.text).strip()
        out["stochastic_answers"] = stochastic_answers
        out["timestamp"] = utc_now()
        out["run_id"] = run_id
        out["protocol_version"] = protocol_version
        out["config_hash"] = config_hash
        out["prompt_version"] = prompt_version
        out["model_provider"] = provider
        out["model_id"] = model_id
        out["stochastic_target_n"] = required_samples
        out["stochastic_actual_n"] = actual_n
        out["is_incomplete"] = is_incomplete
        out["sample_metadata"] = sample_metadata
        out.setdefault("dataset_name", dataset_defaults.get("name", "truthful_qa"))
        out.setdefault("dataset_split", dataset_defaults.get("split", "validation"))
        backfilled_rows.append(out)

    ResultStorage.write_jsonl_atomic(backfilled_rows, args.backfill_file)

    replace_keys = {pair_key(r) for r in backfilled_rows}
    merged_rows = [r for r in clean_rows if pair_key(r) not in replace_keys]
    merged_rows.extend(backfilled_rows)
    merged_rows.sort(
        key=lambda r: (
            question_sort_key(str(r.get("question_id", ""))),
            str(r.get("model", "")),
        )
    )
    ResultStorage.write_jsonl_atomic(merged_rows, args.output)

    logger.info("Backfilled rows written: %s (%d rows)", args.backfill_file, len(backfilled_rows))
    logger.info("Merged output written: %s (%d rows)", args.output, len(merged_rows))


if __name__ == "__main__":
    main()
