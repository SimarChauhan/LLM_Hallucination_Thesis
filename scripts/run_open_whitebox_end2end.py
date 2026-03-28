#!/usr/bin/env python3
"""
Strict open-model end-to-end white-box pipeline.

This runner performs all steps in one place:
1) Load a question set (question_id, question, ground_truth).
2) Generate greedy + stochastic answers from a local/open HF causal model.
3) Compute correctness + equivalence labels (CE/IE) using existing labeling utilities.
4) Write an analysis-ready JSONL artifact.
5) Optionally run the EMNLP-style WB probe script on the generated artifact.

This is intended for "true" open-model experiments where generation and probing
use locally accessible model weights (not API-only target internals).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybrid_judging import decide_equivalence_hybrid, grade_sample_correctness_hybrid
from src.labeling import classify_at_multiple_thresholds, compute_equivalence_stats
from src.nli_judge import NLISemanticJudge


load_dotenv()

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_INPUT = (
    _PROJECT_ROOT
    / "data"
    / "results"
    / "evaluated"
    / "results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl"
)
DEFAULT_OUT_ROOT = _PROJECT_ROOT / "data" / "results" / "analysis" / "final_analysis_ready" / "whitebox" / "open_e2e"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sanitize_for_path(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum():
            out.append(ch.lower())
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def stable_int_hash(value: str, digits: int = 8) -> int:
    h = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(h[:digits], 16)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested --device cuda, but CUDA is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested --device mps, but MPS is unavailable.")
    return requested


def resolve_dtype(torch_dtype: str) -> Optional[torch.dtype]:
    if torch_dtype == "auto":
        return None
    if torch_dtype == "float16":
        return torch.float16
    if torch_dtype == "bfloat16":
        return torch.bfloat16
    if torch_dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported --torch-dtype: {torch_dtype}")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_ground_truth(raw: Any) -> List[str]:
    if isinstance(raw, list):
        out = []
        for item in raw:
            if isinstance(item, str):
                val = item.strip()
                if val:
                    out.append(val)
        return out
    if isinstance(raw, str):
        val = raw.strip()
        return [val] if val else []
    return []


def load_question_set(input_path: Path, max_questions: int = 0) -> List[Dict[str, Any]]:
    """
    Load and dedupe questions by question_id.

    Accepts analysis-ready rows (multiple models per question) or direct question files.
    Required fields per unique question:
      - question_id
      - question
      - ground_truth (list[str] or str)
    """
    rows = load_jsonl(input_path)
    if not rows:
        raise ValueError(f"Input file is empty: {input_path}")

    seen: set[str] = set()
    questions: List[Dict[str, Any]] = []
    for row in rows:
        qid = str(row.get("question_id", "")).strip()
        question = str(row.get("question", "")).strip()
        ground_truth = normalize_ground_truth(row.get("ground_truth"))
        if not qid or not question or not ground_truth:
            continue
        if qid in seen:
            continue
        seen.add(qid)
        questions.append(
            {
                "question_id": qid,
                "question": question,
                "ground_truth": ground_truth,
                "dataset_name": row.get("dataset_name"),
                "dataset_split": row.get("dataset_split"),
            }
        )
        if max_questions > 0 and len(questions) >= max_questions:
            break

    if not questions:
        raise ValueError(
            "No usable questions found. Ensure input rows include question_id/question/ground_truth."
        )
    return questions


def build_generation_prompt(question: str) -> str:
    # Keep a stable short-answer prompt to match the existing QA style.
    return (
        "Answer the question concisely and directly.\n"
        "Do not include extra explanation unless necessary.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


class LocalCausalGenerator:
    def __init__(
        self,
        model_id_or_path: str,
        device: str,
        max_length: int,
        torch_dtype: str,
    ) -> None:
        self.model_id_or_path = model_id_or_path
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        kwargs: Dict[str, Any] = {}
        dtype = resolve_dtype(torch_dtype)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)
        self.model.eval()
        self.model.to(device)

    def generate_one(
        self,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        seed: Optional[int] = None,
    ) -> str:
        tok = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        tok = {k: v.to(self.device) for k, v in tok.items()}

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = float(temperature)
            generation_kwargs["top_p"] = float(top_p)

        if seed is not None and do_sample:
            g = torch.Generator(device=self.device)
            g.manual_seed(int(seed))
            generation_kwargs["generator"] = g

        with torch.no_grad():
            out = self.model.generate(**tok, **generation_kwargs)

        prompt_len = int(tok["input_ids"].shape[1])
        completion_ids = out[0, prompt_len:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return text


def maybe_load_nli_calibration(nli_judge: NLISemanticJudge, calibration_file: str) -> Optional[str]:
    if not calibration_file:
        return None
    path = Path(calibration_file)
    if not path.exists():
        raise FileNotFoundError(f"NLI calibration file not found: {path}")
    nli_judge.load_calibration(str(path))
    return str(path.resolve())


def evaluate_one_question(
    question_row: Dict[str, Any],
    generator: LocalCausalGenerator,
    nli_judge: NLISemanticJudge,
    model_label: str,
    stochastic_samples: int,
    greedy_max_new_tokens: int,
    stochastic_max_new_tokens: int,
    stochastic_temperature: float,
    stochastic_top_p: float,
    corr_hi: float,
    corr_lo: float,
    eq_same_hi: float,
    eq_diff_lo: float,
    seed: int,
) -> Dict[str, Any]:
    qid = str(question_row["question_id"])
    question = str(question_row["question"])
    ground_truth = list(question_row["ground_truth"])
    prompt = build_generation_prompt(question)

    greedy_answer = generator.generate_one(
        prompt=prompt,
        max_new_tokens=greedy_max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        seed=None,
    )

    stochastic_answers: List[str] = []
    for i in range(stochastic_samples):
        sample_seed = int(seed + (stable_int_hash(qid) % 1000003) + i * 9973)
        s = generator.generate_one(
            prompt=prompt,
            max_new_tokens=stochastic_max_new_tokens,
            do_sample=True,
            temperature=stochastic_temperature,
            top_p=stochastic_top_p,
            seed=sample_seed,
        )
        stochastic_answers.append(s)

    greedy_grade_decision = grade_sample_correctness_hybrid(
        question=question,
        sample_answer=greedy_answer,
        ground_truths=ground_truth,
        nli_judge=nli_judge,
        corr_hi=corr_hi,
        corr_lo=corr_lo,
        inference_client=None,
    )
    correctness_grade = greedy_grade_decision.grade
    greedy_correct = correctness_grade == "CORRECT"
    correctness_unclear = correctness_grade == "NOT_ATTEMPTED"

    eq_labels: List[str] = []
    nli_equiv_probs: List[Dict[str, Any]] = []
    eq_sources: List[str] = []
    eq_source_details: List[str] = []
    for sample in stochastic_answers:
        eq = decide_equivalence_hybrid(
            question=question,
            answer_a=greedy_answer,
            answer_b=sample,
            nli_judge=nli_judge,
            eq_same_hi=eq_same_hi,
            eq_diff_lo=eq_diff_lo,
            inference_client=None,
        )
        eq_labels.append(eq.label)
        eq_sources.append(eq.source)
        eq_source_details.append(eq.source_detail)
        nli_equiv_probs.append(
            {
                "prob_forward": float(eq.prob_forward) if eq.prob_forward is not None else None,
                "prob_reverse": float(eq.prob_reverse) if eq.prob_reverse is not None else None,
            }
        )

    eq_stats = compute_equivalence_stats(eq_labels)
    label_by_threshold = classify_at_multiple_thresholds(
        is_correct=greedy_correct,
        equivalence_stats=eq_stats,
        thresholds=[1.0, 0.9, 0.8, 0.7],
        unclear_treatment="exclude",
        grade=correctness_grade,
    )

    out: Dict[str, Any] = {
        "question_id": qid,
        "question": question,
        "ground_truth": ground_truth,
        "model": model_label,
        "model_provider": "huggingface_local",
        "model_id": generator.model_id_or_path,
        "greedy_answer": greedy_answer,
        "stochastic_answers": stochastic_answers,
        "stochastic_target_n": int(stochastic_samples),
        "stochastic_actual_n": int(len(stochastic_answers)),
        "correctness_grade": correctness_grade,
        "correctness_unclear": bool(correctness_unclear),
        "greedy_correct": bool(greedy_correct),
        "correctness_match_type": "open_model_nli_hybrid",
        "correctness_decision_source": greedy_grade_decision.source,
        "correctness_nli_probs": {
            "p_max": float(greedy_grade_decision.p_max) if greedy_grade_decision.p_max is not None else None,
            "matched_gold": greedy_grade_decision.matched_gold,
            "matched_gold_index": greedy_grade_decision.matched_gold_index,
            "source_detail": greedy_grade_decision.source_detail,
        },
        "equivalence_results": eq_labels,
        "equivalence_stats": eq_stats.to_dict(),
        "equivalence_ratio": float(eq_stats.equivalence_ratio),
        "equivalence_decision_source": eq_sources,
        "equivalence_decision_source_detail": eq_source_details,
        "nli_equiv_probs": nli_equiv_probs,
        "error_label_1.0": label_by_threshold[1.0],
        "error_label_0.9": label_by_threshold[0.9],
        "error_label_0.8": label_by_threshold[0.8],
        "error_label_0.7": label_by_threshold[0.7],
        "dataset_name": question_row.get("dataset_name"),
        "dataset_split": question_row.get("dataset_split"),
    }
    return out


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_probe_subprocess(
    input_path: Path,
    output_dir: Path,
    target_model_name: str,
    response_model_id_or_path: str,
    verifier_model_id_or_path: str,
    layer_candidates: str,
    ce_threshold: float,
    encoder_device: str,
    probe_device: str,
    torch_dtype: str,
    probe_seeds: str,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    lambda_step: float,
    max_length: int,
    batch_size: int,
    sequential_encoders: bool = False,
) -> int:
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "run_wb_cross_model_probe_emnlp2025.py"),
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--target-model-name",
        target_model_name,
        "--subset",
        "both",
        "--response-model-path-or-hf-id",
        response_model_id_or_path,
        "--verifier-model-path-or-hf-id",
        verifier_model_id_or_path,
        "--layer-candidates",
        layer_candidates,
        "--ce-threshold",
        str(ce_threshold),
        "--encoder-device",
        encoder_device,
        "--probe-device",
        probe_device,
        "--torch-dtype",
        torch_dtype,
        "--probe-seeds",
        probe_seeds,
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--lambda-step",
        str(lambda_step),
        "--max-length",
        str(max_length),
        "--batch-size",
        str(batch_size),
    ]
    if sequential_encoders:
        cmd.append("--sequential-encoders")
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run strict open-model end-to-end WB pipeline.")
    p.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Question source JSONL.")
    p.add_argument("--output-root", type=str, default=str(DEFAULT_OUT_ROOT))
    p.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run name. If omitted, derived from model + UTC timestamp.",
    )
    p.add_argument("--max-questions", type=int, default=0, help="Optional question cap for quick runs.")

    p.add_argument("--target-model-name", type=str, required=True, help="Label written to output rows (field: model).")
    p.add_argument(
        "--response-model-path-or-hf-id",
        type=str,
        required=True,
        help="Local path or HF id for target response model generation + probing.",
    )
    p.add_argument(
        "--verifier-model-path-or-hf-id",
        type=str,
        default="",
        help="Optional verifier model for probing. Defaults to response model.",
    )

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch-dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--max-length", type=int, default=1024)

    p.add_argument("--greedy-max-new-tokens", type=int, default=96)
    p.add_argument("--stochastic-max-new-tokens", type=int, default=96)
    p.add_argument("--stochastic-samples", type=int, default=10)
    p.add_argument("--stochastic-temperature", type=float, default=0.7)
    p.add_argument("--stochastic-top-p", type=float, default=0.9)

    p.add_argument("--corr-hi", type=float, default=0.70, help="NLI high threshold for CORRECT.")
    p.add_argument("--corr-lo", type=float, default=0.30, help="NLI low threshold for INCORRECT.")
    p.add_argument("--eq-same-hi", type=float, default=0.70, help="NLI high threshold for SAME.")
    p.add_argument("--eq-diff-lo", type=float, default=0.30, help="NLI low threshold for DIFFERENT.")
    p.add_argument(
        "--nli-model",
        type=str,
        default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        help="NLI model used for correctness/equivalence labeling.",
    )
    p.add_argument("--nli-device", type=str, default="", help="Optional NLI device override.")
    p.add_argument("--nli-calibration-file", type=str, default="")

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--run-probe", action="store_true", help="Run EMNLP-style WB probe after generation+labeling.")
    p.add_argument("--probe-layer-candidates", type=str, default="last8")
    p.add_argument("--probe-ce-threshold", type=float, default=1.0)
    p.add_argument("--probe-encoder-device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--probe-device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--probe-seeds", type=str, default="11,22,33")
    p.add_argument("--probe-epochs", type=int, default=300)
    p.add_argument("--probe-patience", type=int, default=25)
    p.add_argument("--probe-lr", type=float, default=2e-3)
    p.add_argument("--probe-weight-decay", type=float, default=1e-4)
    p.add_argument("--probe-lambda-step", type=float, default=0.05)
    p.add_argument("--probe-batch-size", type=int, default=4, help="Feature extraction batch size for probe script.")
    p.add_argument(
        "--probe-sequential-encoders",
        action="store_true",
        help="Load response and verifier encoders one at a time to reduce peak memory (for ~18GB machines).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.probe_ce_threshold <= 1.0):
        raise ValueError("--probe-ce-threshold must be in (0, 1].")
    if args.corr_lo >= args.corr_hi:
        raise ValueError("--corr-lo must be < --corr-hi.")
    if args.eq_diff_lo >= args.eq_same_hi:
        raise ValueError("--eq-diff-lo must be < --eq-same-hi.")

    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    nli_device = args.nli_device.strip() if args.nli_device.strip() else None

    verifier_model_id_or_path = args.verifier_model_path_or_hf_id.strip() or args.response_model_path_or_hf_id

    run_name = args.run_name.strip()
    if not run_name:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"{sanitize_for_path(args.target_model_name)}_{ts}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    records_path = run_dir / "open_whitebox_e2e.analysis_ready.jsonl"
    report_path = run_dir / "open_whitebox_e2e_run_report.json"
    probe_dir = run_dir / "probe_emnlp2025"

    t0 = time.time()
    questions = load_question_set(Path(args.input), max_questions=int(args.max_questions))

    generator = LocalCausalGenerator(
        model_id_or_path=args.response_model_path_or_hf_id,
        device=device,
        max_length=int(args.max_length),
        torch_dtype=args.torch_dtype,
    )

    nli_judge = NLISemanticJudge(
        model_name=args.nli_model,
        device=nli_device,
    )
    nli_calibration_path = maybe_load_nli_calibration(nli_judge, args.nli_calibration_file)

    out_rows: List[Dict[str, Any]] = []
    for idx, q in enumerate(questions):
        row = evaluate_one_question(
            question_row=q,
            generator=generator,
            nli_judge=nli_judge,
            model_label=args.target_model_name,
            stochastic_samples=int(args.stochastic_samples),
            greedy_max_new_tokens=int(args.greedy_max_new_tokens),
            stochastic_max_new_tokens=int(args.stochastic_max_new_tokens),
            stochastic_temperature=float(args.stochastic_temperature),
            stochastic_top_p=float(args.stochastic_top_p),
            corr_hi=float(args.corr_hi),
            corr_lo=float(args.corr_lo),
            eq_same_hi=float(args.eq_same_hi),
            eq_diff_lo=float(args.eq_diff_lo),
            seed=int(args.seed + idx * 17),
        )
        out_rows.append(row)

    write_jsonl(records_path, out_rows)

    label_counts: Dict[str, int] = {}
    grade_counts: Dict[str, int] = {}
    for row in out_rows:
        label = str(row.get("error_label_0.9", ""))
        grade = str(row.get("correctness_grade", ""))
        label_counts[label] = label_counts.get(label, 0) + 1
        grade_counts[grade] = grade_counts.get(grade, 0) + 1

    probe_return_code: Optional[int] = None
    if args.run_probe:
        probe_return_code = run_probe_subprocess(
            input_path=records_path,
            output_dir=probe_dir,
            target_model_name=args.target_model_name,
            response_model_id_or_path=args.response_model_path_or_hf_id,
            verifier_model_id_or_path=verifier_model_id_or_path,
            layer_candidates=args.probe_layer_candidates,
            ce_threshold=float(args.probe_ce_threshold),
            encoder_device=args.probe_encoder_device,
            probe_device=args.probe_device,
            torch_dtype=args.torch_dtype,
            probe_seeds=args.probe_seeds,
            epochs=int(args.probe_epochs),
            patience=int(args.probe_patience),
            lr=float(args.probe_lr),
            weight_decay=float(args.probe_weight_decay),
            lambda_step=float(args.probe_lambda_step),
            max_length=int(args.max_length),
            batch_size=int(args.probe_batch_size),
            sequential_encoders=getattr(args, "probe_sequential_encoders", False),
        )

    report = {
        "generated_at_utc": utc_now_iso(),
        "input": str(Path(args.input).resolve()),
        "rows_generated": int(len(out_rows)),
        "target_model_name": args.target_model_name,
        "response_model_path_or_hf_id": args.response_model_path_or_hf_id,
        "verifier_model_path_or_hf_id": verifier_model_id_or_path,
        "device": device,
        "torch_dtype": args.torch_dtype,
        "nli_model": args.nli_model,
        "nli_device": nli_device,
        "nli_calibration_file": nli_calibration_path,
        "generation": {
            "max_questions": int(args.max_questions),
            "greedy_max_new_tokens": int(args.greedy_max_new_tokens),
            "stochastic_max_new_tokens": int(args.stochastic_max_new_tokens),
            "stochastic_samples": int(args.stochastic_samples),
            "stochastic_temperature": float(args.stochastic_temperature),
            "stochastic_top_p": float(args.stochastic_top_p),
            "seed": int(args.seed),
        },
        "hybrid_thresholds": {
            "corr_hi": float(args.corr_hi),
            "corr_lo": float(args.corr_lo),
            "eq_same_hi": float(args.eq_same_hi),
            "eq_diff_lo": float(args.eq_diff_lo),
        },
        "distribution": {
            "correctness_grade": grade_counts,
            "error_label_0.9": label_counts,
        },
        "outputs": {
            "analysis_ready_jsonl": str(records_path),
            "run_report_json": str(report_path),
            "probe_dir": str(probe_dir) if args.run_probe else None,
        },
        "probe": {
            "enabled": bool(args.run_probe),
            "return_code": probe_return_code,
            "probe_layer_candidates": args.probe_layer_candidates,
            "probe_ce_threshold": float(args.probe_ce_threshold),
        },
        "elapsed_seconds": float(time.time() - t0),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
