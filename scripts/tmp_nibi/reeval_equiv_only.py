#!/usr/bin/env python3
import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.hybrid_judging import decide_equivalence_hybrid
from src.nli_judge import NLISemanticJudge
from src.labeling import compute_equivalence_stats, classify_at_multiple_thresholds
from src.schemas import EquivalenceStats
from src.storage import ResultStorage
from src.providers import MultiProviderClient

load_dotenv()
load_dotenv(Path.home() / 'LLM_Hallucination_Measure' / '.env', override=False)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('reeval_equiv_only.log')],
)
logger = logging.getLogger(__name__)


def compute_config_hash(config: Dict[str, Any]) -> str:
    payload = yaml.safe_dump(config, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def load_hybrid_thresholds(config: Dict[str, Any], override_file: Optional[str]) -> Dict[str, Any]:
    defaults = {'eq_same_hi': 0.70, 'eq_diff_lo': 0.30, 'corr_hi': 0.70, 'corr_lo': 0.30}
    hybrid_cfg = config.get('hybrid', {}) or {}
    threshold_cfg = hybrid_cfg.get('thresholds', {}) or {}
    thresholds = {
        'eq_same_hi': float(threshold_cfg.get('eq_same_hi', defaults['eq_same_hi'])),
        'eq_diff_lo': float(threshold_cfg.get('eq_diff_lo', defaults['eq_diff_lo'])),
        'corr_hi': float(threshold_cfg.get('corr_hi', defaults['corr_hi'])),
        'corr_lo': float(threshold_cfg.get('corr_lo', defaults['corr_lo'])),
    }
    calibration_path = override_file or hybrid_cfg.get('calibration_file') or (config.get('calibration', {}) or {}).get('hybrid_calibration_file')
    calibration_id = None
    if calibration_path:
        path = Path(calibration_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
            thresholds['eq_same_hi'] = float(payload.get('eq_same_hi', thresholds['eq_same_hi']))
            thresholds['eq_diff_lo'] = float(payload.get('eq_diff_lo', thresholds['eq_diff_lo']))
            thresholds['corr_hi'] = float(payload.get('corr_hi', thresholds['corr_hi']))
            thresholds['corr_lo'] = float(payload.get('corr_lo', thresholds['corr_lo']))
            calibration_id = str(payload.get('calibration_id') or path.stem)
            logger.info('Loaded hybrid calibration from %s', path)
    return {'thresholds': thresholds, 'calibration_id': calibration_id}


def build_precomputed_correctness_result(rec: Dict[str, Any]) -> Tuple[SimpleNamespace, bool]:
    existing_grade = rec.get('correctness_grade')
    existing_bool = rec.get('greedy_correct')
    existing_unclear = rec.get('correctness_unclear')
    valid_grade = existing_grade in {'CORRECT', 'INCORRECT', 'NOT_ATTEMPTED'}
    valid_bool = existing_bool in {True, False}
    if not valid_grade and not valid_bool:
        result = SimpleNamespace(
            is_correct=False,
            match_type=rec.get('correctness_match_type'),
            is_unclear=True,
            grade='NOT_ATTEMPTED',
        )
        return result, True
    if not valid_grade:
        existing_grade = 'CORRECT' if existing_bool is True else 'INCORRECT'
    if not valid_bool:
        existing_bool = existing_grade == 'CORRECT'
    inferred_unclear = bool(existing_unclear) or existing_grade == 'NOT_ATTEMPTED'
    result = SimpleNamespace(
        is_correct=bool(existing_bool),
        match_type=rec.get('correctness_match_type'),
        is_unclear=inferred_unclear,
        grade=existing_grade,
    )
    return result, False


def load_records(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_records(records: List[Dict[str, Any]], output_path: str) -> None:
    ResultStorage.write_jsonl_atomic(records, output_path)


def main() -> int:
    ap = argparse.ArgumentParser(description='Equivalence-only hybrid re-eval: greedy vs stochastic samples.')
    ap.add_argument('--config', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--run-id', default=None)
    ap.add_argument('--hybrid-calibration-file', default=None)
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    protocol_cfg = config.get('protocol', {}) or {}
    semantic_cfg = config.get('semantic', {}) or {}
    judge_cfg = config.get('judge', {}) or {}
    nli_cfg = judge_cfg.get('nli', {}) or {}
    rate_cfg = config.get('rate_limit', {}) or {}
    hybrid_cfg = config.get('hybrid', {}) or {}
    borderline_cfg = hybrid_cfg.get('borderline_judge', {}) or {}

    protocol_version = str(protocol_cfg.get('version', 'v3'))
    prompt_version = str(protocol_cfg.get('prompt_version', 'qa-short-v1'))
    unclear_treatment = semantic_cfg.get('unclear_treatment', 'exclude')
    config_hash = compute_config_hash(config)
    threshold_bundle = load_hybrid_thresholds(config, args.hybrid_calibration_file)
    hybrid_thresholds = threshold_bundle['thresholds']
    hybrid_calibration_id = threshold_bundle.get('calibration_id')
    hybrid_enabled = True
    hybrid_judge_provider = str(borderline_cfg.get('provider', 'openai'))
    hybrid_judge_model = str(borderline_cfg.get('model', 'gpt-5.2'))
    hybrid_eq_max_new_tokens = int(borderline_cfg.get('equivalence_max_new_tokens', 220))
    run_id = args.run_id or f"equiv_only_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"

    logger.info('Loaded configuration from %s', args.config)
    logger.info('Equivalence-only mode enabled: precomputed greedy correctness + greedy-vs-stochastic hybrid equivalence only.')
    logger.info('Hybrid thresholds: eq_same_hi=%.3f eq_diff_lo=%.3f', hybrid_thresholds['eq_same_hi'], hybrid_thresholds['eq_diff_lo'])
    if hybrid_calibration_id:
        logger.info('Hybrid calibration id: %s', hybrid_calibration_id)

    records = load_records(args.input)
    logger.info('Loaded %d records from %s', len(records), args.input)

    nli_model = str(nli_cfg.get('model', 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'))
    nli_device = nli_cfg.get('device')
    nli_entailment_threshold = float(nli_cfg.get('entailment_threshold', 0.5))
    nli_different_threshold = float(nli_cfg.get('different_threshold', 0.3))
    nli_batch_size = int(nli_cfg.get('batch_size', 8))

    logger.info('Initializing NLI judge: %s', nli_model)
    nli_judge = NLISemanticJudge(
        model_name=nli_model,
        device=nli_device,
        entailment_threshold=nli_entailment_threshold,
        different_threshold=nli_different_threshold,
        batch_size=nli_batch_size,
    )

    inference_client = MultiProviderClient(
        initial_delay=rate_cfg.get('initial_delay', 2.0),
        max_delay=rate_cfg.get('max_delay', 60.0),
        backoff_factor=rate_cfg.get('backoff_factor', 2.0),
    )

    stats = {'total': len(records), 'evaluated': 0, 'hybrid_equiv_llm_rows': 0, 'missing_precomputed_greedy': 0}

    try:
        for idx, rec in enumerate(tqdm(records, desc='Equivalence-only')):
            question = rec.get('question', '')
            greedy_answer = rec.get('greedy_answer', '')
            stochastic_answers = rec.get('stochastic_answers') or []

            correctness_result, missing_precomputed = build_precomputed_correctness_result(rec)
            if missing_precomputed:
                stats['missing_precomputed_greedy'] += 1

            rec['protocol_version'] = protocol_version
            rec['config_hash'] = config_hash
            rec['prompt_version'] = rec.get('prompt_version') or prompt_version
            rec['judge_protocol'] = f'precomputed_greedy+equiv_only_hybrid_borderline({hybrid_judge_provider}/{hybrid_judge_model})'
            rec['run_id'] = run_id
            rec['greedy_correctness_rejudged'] = False
            rec['greedy_correctness_source'] = 'MISSING' if missing_precomputed else 'PRECOMPUTED'
            rec['greedy_correct'] = correctness_result.is_correct
            rec['correctness_match_type'] = correctness_result.match_type
            rec['correctness_unclear'] = correctness_result.is_unclear
            rec['correctness_grade'] = correctness_result.grade
            rec['hybrid_enabled'] = hybrid_enabled
            rec['hybrid_thresholds'] = hybrid_thresholds
            rec['hybrid_calibration_id'] = hybrid_calibration_id
            rec['hybrid_judge_model'] = f'{hybrid_judge_provider}/{hybrid_judge_model}'
            rec['equivalence_only_eval'] = True

            if stochastic_answers:
                legacy_rich = nli_judge.judge_all_samples(question, greedy_answer, stochastic_answers)
                legacy_equivalence_results = [r.judgment for r in legacy_rich]
                legacy_nli_equiv_probs = [
                    {
                        'forward': round(r.prob_forward, 4),
                        'reverse': round(r.prob_reverse, 4),
                        'judgment': r.judgment,
                        'source': 'NLI',
                    }
                    for r in legacy_rich
                ]
                legacy_stats_bare = compute_equivalence_stats(legacy_equivalence_results)
                legacy_stats = EquivalenceStats(
                    num_same=legacy_stats_bare.num_same,
                    num_different=legacy_stats_bare.num_different,
                    num_unclear=legacy_stats_bare.num_unclear,
                    total=legacy_stats_bare.total,
                    nli_probs=legacy_nli_equiv_probs,
                )

                equivalence_results = []
                equivalence_decision_source = []
                equivalence_decision_source_detail = []
                nli_equiv_probs = []
                for sample in stochastic_answers:
                    eq = decide_equivalence_hybrid(
                        question=question,
                        answer_a=greedy_answer,
                        answer_b=sample,
                        nli_judge=nli_judge,
                        eq_same_hi=hybrid_thresholds['eq_same_hi'],
                        eq_diff_lo=hybrid_thresholds['eq_diff_lo'],
                        inference_client=inference_client,
                        judge_provider=hybrid_judge_provider,
                        judge_model=hybrid_judge_model,
                        max_new_tokens=hybrid_eq_max_new_tokens,
                    )
                    equivalence_results.append(eq.label)
                    equivalence_decision_source.append(eq.source)
                    equivalence_decision_source_detail.append(eq.source_detail)
                    nli_equiv_probs.append({
                        'forward': round(eq.prob_forward, 4) if eq.prob_forward is not None else None,
                        'reverse': round(eq.prob_reverse, 4) if eq.prob_reverse is not None else None,
                        'judgment': eq.label,
                        'source': eq.source,
                        'source_detail': eq.source_detail,
                        'llm_label_forward': eq.llm_label_forward,
                        'llm_label_reverse': eq.llm_label_reverse,
                    })
                if any(src == 'LLM' for src in equivalence_decision_source):
                    stats['hybrid_equiv_llm_rows'] += 1

                equiv_stats_bare = compute_equivalence_stats(equivalence_results)
                equiv_stats = EquivalenceStats(
                    num_same=equiv_stats_bare.num_same,
                    num_different=equiv_stats_bare.num_different,
                    num_unclear=equiv_stats_bare.num_unclear,
                    total=equiv_stats_bare.total,
                    nli_probs=nli_equiv_probs,
                )

                rec['equivalence_results_nli'] = legacy_equivalence_results
                rec['equivalence_stats_nli'] = legacy_stats.to_dict()
                rec['equivalence_ratio_nli'] = legacy_stats.equivalence_ratio
                rec['nli_equiv_probs_nli'] = legacy_nli_equiv_probs
                rec['equivalence_results'] = equivalence_results
                rec['equivalence_stats'] = equiv_stats.to_dict()
                rec['equivalence_ratio'] = equiv_stats.equivalence_ratio
                rec['nli_equiv_probs'] = nli_equiv_probs
                rec['equivalence_decision_source'] = equivalence_decision_source
                rec['equivalence_decision_source_detail'] = equivalence_decision_source_detail

                labels = classify_at_multiple_thresholds(
                    is_correct=correctness_result.is_correct,
                    equivalence_stats=equiv_stats,
                    thresholds=[1.0, 0.9, 0.8, 0.7],
                    unclear_treatment=unclear_treatment,
                    grade=correctness_result.grade,
                )
                legacy_labels = classify_at_multiple_thresholds(
                    is_correct=correctness_result.is_correct,
                    equivalence_stats=legacy_stats,
                    thresholds=[1.0, 0.9, 0.8, 0.7],
                    unclear_treatment=unclear_treatment,
                    grade=correctness_result.grade,
                )
                rec['error_label_1.0'] = labels[1.0]
                rec['error_label_0.9'] = labels[0.9]
                rec['error_label_0.8'] = labels[0.8]
                rec['error_label_0.7'] = labels[0.7]
                rec['error_label_nli_1.0'] = legacy_labels[1.0]
                rec['error_label_nli_0.9'] = legacy_labels[0.9]
                rec['error_label_nli_0.8'] = legacy_labels[0.8]
                rec['error_label_nli_0.7'] = legacy_labels[0.7]
            else:
                rec['equivalence_results_nli'] = None
                rec['equivalence_stats_nli'] = None
                rec['equivalence_ratio_nli'] = None
                rec['nli_equiv_probs_nli'] = None
                rec['equivalence_results'] = None
                rec['equivalence_stats'] = None
                rec['equivalence_ratio'] = None
                rec['nli_equiv_probs'] = None
                rec['equivalence_decision_source'] = None
                rec['equivalence_decision_source_detail'] = None
                rec['error_label_1.0'] = None
                rec['error_label_0.9'] = None
                rec['error_label_0.8'] = None
                rec['error_label_0.7'] = None
                rec['error_label_nli_1.0'] = None
                rec['error_label_nli_0.9'] = None
                rec['error_label_nli_0.8'] = None
                rec['error_label_nli_0.7'] = None

            rec['semantic_pair_decisions'] = None
            rec['stochastic_sample_grades'] = None
            rec['stochastic_sample_grade_source'] = None
            rec['stochastic_sample_grade_source_detail'] = None
            rec['stochastic_sample_grade_confidence'] = None
            rec['stochastic_correct_rate'] = None
            rec['stochastic_scored_n'] = None
            rec['stochastic_not_attempted_n'] = None
            rec['different_scored_n'] = None
            rec['different_correct_n'] = None
            rec['p_correct_given_different'] = None
            rec['semantic_entropy'] = None
            rec['semantic_entropy_norm'] = None
            rec['n_semantic_clusters'] = None
            rec['semantic_cluster_ids'] = None
            rec['semantic_cluster_sizes'] = None
            rec['semantic_entropy_label'] = None

            stats['evaluated'] += 1
            if (idx + 1) % 200 == 0:
                save_records(records, args.output)
                logger.info('Checkpoint saved %d records to %s', len(records), args.output)
    except KeyboardInterrupt:
        save_records(records, args.output)
        logger.info('Interrupted. Partial results saved to %s (%d evaluated).', args.output, stats['evaluated'])
        raise

    save_records(records, args.output)
    logger.info('Saved %d evaluated records to %s', len(records), args.output)
    logger.info('Hybrid equiv LLM rows: %d', stats['hybrid_equiv_llm_rows'])
    logger.info('Missing precomputed greedy rows: %d', stats['missing_precomputed_greedy'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
