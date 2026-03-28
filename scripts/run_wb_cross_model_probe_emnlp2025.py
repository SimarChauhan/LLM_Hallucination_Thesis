#!/usr/bin/env python3
"""
WB-CrossModelProbe runner implementing the method from:

    Tan et al., "Too Consistent to Detect: A Study of Self-Consistent
    Errors in LLMs", EMNLP 2025.  arXiv:2505.17656
    https://github.com/Tan-Hexiang/Too-Consistent-to-Detect

Paper-aligned choices:
1) Last-token hidden states from response model M and verifier model V.
2) Probe per layer, choose best layer by validation AUROC.
3) Probe architecture: 4-layer FFN (dim->256->128->64->1), ReLU, BCE loss.
4) Early stopping on validation loss; layer selection on validation AUROC.
5) Cross-model fusion: score = (1-lambda)*s_M + lambda*s_V,
   lambda in {0, 0.05, ..., 1.0} tuned on validation AUROC.
6) CE/IE balanced subsets with matched positive counts (correct examples).

Differences from the original paper:
- Dataset: TruthfulQA (paper uses TriviaQA/SciQ).
- Z-score feature normalization (paper does not normalize).
- Response models evaluated via API; hidden-state encoders are local proxies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
import gc
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_INPUT = (
    _PROJECT_ROOT / "data" / "results" / "evaluated"
    / "results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl"
)
DEFAULT_OUT = (
    _PROJECT_ROOT / "data" / "results" / "analysis" / "final_analysis_ready"
    / "whitebox" / "wb_cross_model_probe_emnlp2025"
)

load_dotenv()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested --encoder-device/--probe-device cuda, but CUDA is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested --encoder-device/--probe-device mps, but MPS is unavailable.")
    return requested


def release_encoder_cuda(encoder: Any) -> None:
    """
    Best-effort cleanup between sequential encoder loads.
    This is critical for very large models where allocator state can persist.
    """
    model = getattr(encoder, "model", None)
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            # Some accelerate-dispatched models may not support full .to("cpu") migration.
            pass
    for attr in ("model", "tokenizer", "processor"):
        if hasattr(encoder, attr):
            try:
                delattr(encoder, attr)
            except Exception:
                pass
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def maybe_require_hf_token(model_id_or_path: str) -> None:
    # If this is a local path, don't enforce token checks.
    if Path(model_id_or_path).exists():
        return
    low = model_id_or_path.lower()
    if "meta-llama" in low and not os.environ.get("HF_TOKEN"):
        raise ValueError(
            "Meta-Llama model requested but HF_TOKEN is not set in environment. "
            "Set HF_TOKEN (or login via huggingface-cli) before running."
        )


def from_pretrained_with_remote_code_fallback(loader: Any, model_id: str, **kwargs: Any) -> Any:
    """
    Try regular `from_pretrained(...)` first, then retry with `trust_remote_code=True`
    for newer/custom architectures (e.g., DeepSeek-V3.2) not yet mapped in base Auto*.
    """
    try:
        return loader.from_pretrained(model_id, **kwargs)
    except Exception as first_err:
        if kwargs.get("trust_remote_code", False):
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs["trust_remote_code"] = True
        try:
            return loader.from_pretrained(model_id, **retry_kwargs)
        except TypeError:
            # Some loaders might not accept trust_remote_code; preserve original error.
            raise first_err
        except Exception as retry_err:
            loader_name = getattr(loader, "__name__", str(loader))
            raise RuntimeError(
                f"from_pretrained failed for {loader_name}('{model_id}'). "
                f"Initial error: {type(first_err).__name__}: {first_err}. "
                f"Retry with trust_remote_code=True error: {type(retry_err).__name__}: {retry_err}"
            ) from retry_err


def stable_u01(key: str) -> float:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_for_question(question_id: str, seed: int, train_frac: float, val_frac: float) -> str:
    u = stable_u01(f"{seed}|{question_id}")
    if u < train_frac:
        return "train"
    if u < train_frac + val_frac:
        return "val"
    return "test"


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = y_true.astype(int)
    s = y_score.astype(float)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if y.size == 0 or pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    n = len(s_sorted)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    rank_sum_pos = float(ranks[y == 1].sum())
    return (rank_sum_pos - (pos * (pos + 1) / 2.0)) / (pos * neg)


def auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = y_true.astype(int)
    s = y_score.astype(float)
    pos = int((y == 1).sum())
    if y.size == 0 or pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = 0
    fp = 0
    recalls = [0.0]
    precisions = [1.0]
    for yi in y_sorted:
        if yi == 1:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / pos)
        precisions.append(tp / (tp + fp))
    recalls.append(1.0)
    precisions.append(0.0)
    area = 0.0
    for i in range(1, len(recalls)):
        area += (recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2.0
    return float(area)


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    y = y_true.astype(int)
    s = y_score.astype(float)
    pred = (s >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "auroc": auc_roc(y, s),
        "prauc": auc_pr(y, s),
        "accuracy_at_0_5": float((pred == y).mean()) if len(y) else float("nan"),
        "brier": float(np.mean((s - y) ** 2)) if len(y) else float("nan"),
    }


def parse_layers(spec: str, num_hidden_layers: int) -> List[int]:
    # hidden_states index: 0 is embeddings, 1..num_hidden_layers are transformer blocks
    valid = list(range(1, num_hidden_layers + 1))
    s = spec.strip().lower()
    if s == "all":
        return valid
    if s.startswith("last"):
        k = int(s[4:])
        if k <= 0:
            raise ValueError("--layer-candidates lastK must have K>0")
        return valid[-k:]
    out: List[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v < 0:
            v = (num_hidden_layers + 1) + v
        if v < 1 or v > num_hidden_layers:
            raise ValueError(f"Layer {tok} resolves to {v}, outside [1, {num_hidden_layers}]")
        out.append(v)
    out = sorted(set(out))
    if not out:
        raise ValueError("No valid layers resolved from --layer-candidates")
    return out


def infer_num_hidden_layers_from_config(config: Any) -> int:
    candidates = [
        getattr(config, "num_hidden_layers", None),
        getattr(getattr(config, "text_config", None), "num_hidden_layers", None),
        getattr(getattr(config, "language_model_config", None), "num_hidden_layers", None),
        getattr(getattr(config, "llm_config", None), "num_hidden_layers", None),
    ]
    for v in candidates:
        if isinstance(v, int) and v > 0:
            return int(v)
    raise ValueError(
        f"Could not infer num_hidden_layers from config class {type(config).__name__}. "
        "Try updating transformers or using a supported model."
    )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def ce_exact_label(row: Dict[str, Any], ce_threshold: float = 1.0) -> bool:
    # CE requires incorrectness and high semantic equivalence among samples.
    if str(row.get("correctness_grade", "")) != "INCORRECT":
        return False
    stats = row.get("equivalence_stats")
    same = 0
    diff = 0
    unc = 0
    total = 0
    if isinstance(stats, dict):
        same = int(stats.get("num_same", 0))
        diff = int(stats.get("num_different", 0))
        unc = int(stats.get("num_unclear", 0))
        total = int(stats.get("total", 0))

    ratio: Optional[float] = None
    if total > 0:
        ratio = float(same) / float(total)
    else:
        eq_ratio = row.get("equivalence_ratio")
        if isinstance(eq_ratio, (int, float)) and math.isfinite(float(eq_ratio)):
            ratio = float(eq_ratio)

    if ratio is None:
        return False

    if ce_threshold >= 1.0:
        # Paper setting: exact CE requires complete agreement and no ambiguity.
        if total > 0:
            return same == total and diff == 0 and unc == 0
        return ratio >= 1.0

    return ratio >= ce_threshold


def ie_exact_label(row: Dict[str, Any], ce_threshold: float = 1.0) -> bool:
    if str(row.get("correctness_grade", "")) != "INCORRECT":
        return False
    # Exclude rows with no equivalence data (total == 0) — these are not
    # meaningfully "inconsistent", they simply lack equivalence judgments.
    stats = row.get("equivalence_stats")
    has_equivalence = False
    if isinstance(stats, dict) and int(stats.get("total", 0)) > 0:
        has_equivalence = True
    eq_ratio = row.get("equivalence_ratio")
    if isinstance(eq_ratio, (int, float)) and math.isfinite(float(eq_ratio)):
        has_equivalence = True
    if not has_equivalence:
        return False
    return not ce_exact_label(row, ce_threshold=ce_threshold)


def build_probe_text(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def build_subset_dataframe(
    rows: Sequence[Dict[str, Any]],
    target_model_name: str,
    subset: str,
    sample_seed: int,
    ce_threshold: float,
) -> pd.DataFrame:
    target_rows = [r for r in rows if str(r.get("model", "")) == target_model_name]
    if not target_rows:
        raise ValueError(f"No rows found for target model '{target_model_name}'")

    positives = [r for r in target_rows if str(r.get("correctness_grade", "")) == "CORRECT"]
    if subset == "ce":
        negatives = [r for r in target_rows if ce_exact_label(r, ce_threshold=ce_threshold)]
    elif subset == "ie":
        negatives = [r for r in target_rows if ie_exact_label(r, ce_threshold=ce_threshold)]
    else:
        raise ValueError(f"Unsupported subset: {subset}")

    if len(negatives) == 0:
        raise ValueError(f"No negative samples in subset={subset} for model={target_model_name}")
    if len(positives) == 0:
        raise ValueError(f"No positive samples for model={target_model_name}")

    n = min(len(positives), len(negatives))
    rng = np.random.default_rng(sample_seed)
    pos_choice = rng.choice(len(positives), size=n, replace=False)
    neg_choice = rng.choice(len(negatives), size=n, replace=False)
    positives = [positives[int(i)] for i in pos_choice]
    negatives = [negatives[int(i)] for i in neg_choice]

    merged = []
    for r in positives:
        merged.append(
            {
                "question_id": str(r.get("question_id", "")),
                "question": str(r.get("question", "")),
                "greedy_answer": str(r.get("greedy_answer", "")),
                "y": 0,
                "grade": str(r.get("correctness_grade", "")),
                "error_label_0.9": str(r.get("error_label_0.9", "")),
            }
        )
    for r in negatives:
        merged.append(
            {
                "question_id": str(r.get("question_id", "")),
                "question": str(r.get("question", "")),
                "greedy_answer": str(r.get("greedy_answer", "")),
                "y": 1,
                "grade": str(r.get("correctness_grade", "")),
                "error_label_0.9": str(r.get("error_label_0.9", "")),
            }
        )
    df = pd.DataFrame(merged)
    df["probe_text"] = [build_probe_text(q, a) for q, a in zip(df["question"].tolist(), df["greedy_answer"].tolist())]
    return df


class FFNProbe(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainedProbe:
    mean: np.ndarray
    std: np.ndarray
    state_dict: Dict[str, torch.Tensor]
    input_dim: int
    best_val_auroc: float
    best_epoch: int

    def predict_proba(self, x: np.ndarray, device: str = "cpu") -> np.ndarray:
        z = (x - self.mean) / self.std
        model = FFNProbe(self.input_dim)
        model.load_state_dict(self.state_dict)
        model.eval()
        model.to(device)
        with torch.no_grad():
            xt = torch.tensor(z, dtype=torch.float32, device=device)
            logits = model(xt).squeeze(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        return probs


def train_ffn_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> TrainedProbe:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    z_train = (x_train - mean) / std
    z_val = (x_val - mean) / std

    xtr = torch.tensor(z_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train.astype(np.float32), dtype=torch.float32, device=device).view(-1, 1)
    xva = torch.tensor(z_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val.astype(np.float32), dtype=torch.float32, device=device).view(-1, 1)

    model = FFNProbe(xtr.shape[1]).to(device)
    # Tan et al. (EMNLP 2025) use plain BCELoss without class weighting;
    # subsets are balanced so pos_weight ≈ 1.0 anyway.
    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_val_auroc = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    bad = 0

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(xva)
            val_loss = loss_fn(val_logits, yva).item()
            val_prob = torch.sigmoid(val_logits.squeeze(-1)).detach().cpu().numpy()
            val_auc = auc_roc(y_val, val_prob)
        if np.isnan(val_auc):
            val_auc = -1.0

        # Early stopping on validation loss (paper-aligned); AUROC at the
        # best-loss checkpoint is recorded for layer selection.
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_val_auroc = float(val_auc)
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        raise RuntimeError("Probe training failed to find a best checkpoint.")

    return TrainedProbe(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        state_dict=best_state,
        input_dim=int(xtr.shape[1]),
        best_val_auroc=float(best_val_auroc),
        best_epoch=int(best_epoch),
    )


def pick_lambda(y_val: np.ndarray, s_m: np.ndarray, s_v: np.ndarray, step: float) -> Tuple[float, float]:
    grid = np.arange(0.0, 1.0001, step)
    best_l = 0.0
    best_auc = -1.0
    for lam in grid:
        s = (1.0 - lam) * s_m + lam * s_v
        a = auc_roc(y_val, s)
        if np.isnan(a):
            continue
        if a > best_auc:
            best_auc = float(a)
            best_l = float(lam)
    return best_l, best_auc


class CausalLastTokenEncoder:
    def __init__(
        self,
        model_id: str,
        device: str,
        max_length: int,
        batch_size: int,
        torch_dtype: str = "auto",
        device_map_mode: str = "none",
    ):
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.device_map_mode = device_map_mode
        self.processor = None

        self.tokenizer = self._load_text_tokenizer(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        dtype = None
        if torch_dtype != "auto":
            if torch_dtype == "float16":
                dtype = torch.float16
            elif torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            elif torch_dtype == "float32":
                dtype = torch.float32
            else:
                raise ValueError(f"Unsupported --torch-dtype: {torch_dtype}")

        kwargs: Dict[str, Any] = {}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if device_map_mode == "auto":
            kwargs["device_map"] = "auto"
            kwargs["low_cpu_mem_usage"] = True

        self.model = self._load_model(model_id, kwargs)
        self.model.eval()
        if device_map_mode == "auto":
            self.input_device = self._infer_input_device()
        else:
            self.model.to(device)
            self.input_device = torch.device(device)

        self.num_hidden_layers = infer_num_hidden_layers_from_config(self.model.config)

    def _load_text_tokenizer(self, model_id: str):
        try:
            return from_pretrained_with_remote_code_fallback(AutoTokenizer, model_id, use_fast=True)
        except Exception:
            try:
                return from_pretrained_with_remote_code_fallback(AutoTokenizer, model_id)
            except Exception as tok_err:
                auto_processor_cls = getattr(transformers, "AutoProcessor", None)
                if auto_processor_cls is None:
                    raise tok_err
                self.processor = from_pretrained_with_remote_code_fallback(auto_processor_cls, model_id)
                processor_tokenizer = getattr(self.processor, "tokenizer", None)
                if processor_tokenizer is None:
                    raise RuntimeError(
                        f"Model {model_id} requires a processor, but no text tokenizer was found on the processor."
                    ) from tok_err
                return processor_tokenizer

    def _load_model(self, model_id: str, kwargs: Dict[str, Any]):
        loader_attempts: List[Tuple[str, Any]] = [
            ("AutoModelForCausalLM", AutoModelForCausalLM),
            ("AutoModelForImageTextToText", getattr(transformers, "AutoModelForImageTextToText", None)),
            ("AutoModelForVision2Seq", getattr(transformers, "AutoModelForVision2Seq", None)),
            ("AutoModel", getattr(transformers, "AutoModel", None)),
        ]
        errors: List[str] = []
        for loader_name, loader_cls in loader_attempts:
            if loader_cls is None:
                continue
            try:
                return from_pretrained_with_remote_code_fallback(loader_cls, model_id, **kwargs)
            except Exception as e:
                errors.append(f"{loader_name}: {type(e).__name__}: {e}")
        raise RuntimeError(
            f"Failed to load encoder model for {model_id}. Tried: "
            + " | ".join(errors[-4:])
        )

    def _infer_input_device(self) -> torch.device:
        # For Accelerate-dispatched models (device_map=auto), route inputs to the first GPU shard.
        hf_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_map, dict):
            for loc in hf_map.values():
                if isinstance(loc, int):
                    return torch.device(f"cuda:{loc}")
                if isinstance(loc, str) and loc not in {"cpu", "disk"}:
                    return torch.device(loc)
            for loc in hf_map.values():
                if isinstance(loc, str) and loc == "cpu":
                    return torch.device("cpu")
        return next(self.model.parameters()).device

    def _extract_hidden_states(self, outputs: Any) -> Tuple[torch.Tensor, ...]:
        for attr in ("hidden_states", "decoder_hidden_states"):
            hs = getattr(outputs, attr, None)
            if hs is not None:
                return hs
        lm_out = getattr(outputs, "language_model_outputs", None)
        if lm_out is not None:
            for attr in ("hidden_states", "decoder_hidden_states"):
                hs = getattr(lm_out, attr, None)
                if hs is not None:
                    return hs
        if isinstance(outputs, dict):
            for key in ("hidden_states", "decoder_hidden_states"):
                hs = outputs.get(key)
                if hs is not None:
                    return hs
        raise RuntimeError(
            f"Model outputs for {self.model_id} do not expose hidden states (expected hidden_states/decoder_hidden_states)."
        )

    def encode_last_token_layers(self, texts: Sequence[str], layers: Sequence[int]) -> Dict[int, np.ndarray]:
        out: Dict[int, List[np.ndarray]] = {int(l): [] for l in layers}
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = list(texts[i : i + self.batch_size])
                tok = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tok = {k: v.to(self.input_device) for k, v in tok.items()}
                outputs = self.model(**tok, output_hidden_states=True, return_dict=True)
                hs_tuple = self._extract_hidden_states(outputs)
                lengths = tok["attention_mask"].sum(dim=1) - 1
                lengths = torch.clamp(lengths, min=0)
                for l in layers:
                    hs = hs_tuple[int(l)]
                    if hs.ndim < 3:
                        raise RuntimeError(
                            f"Expected hidden states with ndim >= 3 for {self.model_id} layer {int(l)}, got shape={tuple(hs.shape)}."
                        )

                    # Normalize hidden-state tensor to [batch, seq, hidden].
                    # This supports 3D ([B,S,H] or [B,H,S]) and 4D+ variants
                    # (e.g., Gemma-3n returns [B,groups,S,H]).
                    batch_seq_len = tok["input_ids"].shape[1]
                    axis_sizes = [hs.shape[d] for d in range(1, hs.ndim)]
                    seq_dim = 1 + min(
                        range(len(axis_sizes)),
                        key=lambda i: abs(axis_sizes[i] - batch_seq_len),
                    )
                    hidden_candidates = [d for d in range(1, hs.ndim) if d != seq_dim]
                    if not hidden_candidates:
                        raise RuntimeError(
                            f"Unable to infer hidden dimension for {self.model_id} layer {int(l)} shape={tuple(hs.shape)}."
                        )
                    hidden_dim = max(hidden_candidates, key=lambda d: hs.shape[d])
                    extra_dims = [d for d in range(1, hs.ndim) if d not in {seq_dim, hidden_dim}]
                    hs_seq_hidden = hs.permute([0, seq_dim, *extra_dims, hidden_dim])
                    if hs_seq_hidden.ndim > 3:
                        reduce_dims = tuple(range(2, hs_seq_hidden.ndim - 1))
                        hs_seq_hidden = hs_seq_hidden.mean(dim=reduce_dims)
                    if hs_seq_hidden.shape[0] != tok["input_ids"].shape[0]:
                        raise RuntimeError(
                            f"Batch mismatch after hidden-state normalization for {self.model_id} "
                            f"layer {int(l)}: got {tuple(hs_seq_hidden.shape)}."
                        )

                    lengths_dev = lengths.to(hs_seq_hidden.device)
                    max_idx = hs_seq_hidden.shape[1] - 1
                    if max_idx < 0:
                        raise RuntimeError(
                            f"Encoder {self.model_id} returned empty hidden states at layer {int(l)}."
                        )
                    lengths_dev = torch.clamp(lengths_dev, min=0, max=max_idx)
                    ridx = torch.arange(lengths_dev.shape[0], device=hs_seq_hidden.device)
                    vec = hs_seq_hidden[ridx, lengths_dev, :].detach().cpu().to(torch.float32).numpy()
                    out[int(l)].append(vec)
        return {l: np.concatenate(chunks, axis=0) for l, chunks in out.items()}


def choose_best_layer_and_probe(
    feats_by_layer: Dict[int, np.ndarray],
    y: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    seed: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    train_device: str,
) -> Tuple[int, TrainedProbe]:
    best_layer = -1
    best_probe: Optional[TrainedProbe] = None
    best_auc = -1.0
    for layer in sorted(feats_by_layer.keys()):
        x = feats_by_layer[layer]
        probe = train_ffn_probe(
            x_train=x[tr],
            y_train=y[tr],
            x_val=x[va],
            y_val=y[va],
            seed=seed,
            epochs=epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
            device=train_device,
        )
        if probe.best_val_auroc > best_auc:
            best_auc = probe.best_val_auroc
            best_layer = int(layer)
            best_probe = probe
    if best_probe is None:
        raise RuntimeError("No probe selected during layer search.")
    return best_layer, best_probe


def ensure_split_has_both_classes(df: pd.DataFrame) -> None:
    for sp in ["train", "val", "test"]:
        sub = df[df["split"] == sp]
        if sub.empty:
            raise ValueError(f"Split '{sp}' is empty.")
        pos = int(sub["y"].sum())
        neg = int((1 - sub["y"]).sum())
        if pos == 0 or neg == 0:
            raise ValueError(f"Split '{sp}' has single class only (pos={pos}, neg={neg}).")


def run_subset(
    df: pd.DataFrame,
    subset_name: str,
    layers_resp: List[int],
    layers_ver: List[int],
    args: argparse.Namespace,
    response_encoder: Optional[CausalLastTokenEncoder] = None,
    verifier_encoder: Optional[CausalLastTokenEncoder] = None,
    feat_m: Optional[Dict[int, np.ndarray]] = None,
    feat_v: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["split"] = [
        split_for_question(qid, args.split_seed, args.train_frac, args.val_frac)
        for qid in df["question_id"].tolist()
    ]
    # Leakage guard
    leak = int((df.groupby("question_id")["split"].nunique() > 1).sum())
    if leak > 0:
        raise RuntimeError(f"Split leakage detected: {leak}")
    ensure_split_has_both_classes(df)

    texts = df["probe_text"].tolist()
    y = df["y"].to_numpy(dtype=int)
    tr = df["split"].eq("train").to_numpy()
    va = df["split"].eq("val").to_numpy()
    te = df["split"].eq("test").to_numpy()

    # Extract all candidate layers once per model (or use pre-extracted).
    if feat_m is not None and feat_v is not None:
        pass
    elif response_encoder is not None and verifier_encoder is not None:
        feat_m = response_encoder.encode_last_token_layers(texts, layers_resp)
        feat_v = verifier_encoder.encode_last_token_layers(texts, layers_ver)
    else:
        raise ValueError("Provide either (response_encoder, verifier_encoder) or (feat_m, feat_v)")

    seeds = [int(s.strip()) for s in args.probe_seeds.split(",") if s.strip()]
    by_seed: List[Dict[str, Any]] = []
    test_scores: List[Dict[str, Any]] = []

    for seed in seeds:
        best_l_m, probe_m = choose_best_layer_and_probe(
            feat_m, y, tr, va, seed, args.epochs, args.patience, args.lr, args.weight_decay, args.probe_device
        )
        best_l_v, probe_v = choose_best_layer_and_probe(
            feat_v, y, tr, va, seed, args.epochs, args.patience, args.lr, args.weight_decay, args.probe_device
        )

        s_m_val = probe_m.predict_proba(feat_m[best_l_m][va], device=args.probe_device)
        s_v_val = probe_v.predict_proba(feat_v[best_l_v][va], device=args.probe_device)
        lam, val_auc_fused = pick_lambda(y[va], s_m_val, s_v_val, args.lambda_step)

        s_m_test = probe_m.predict_proba(feat_m[best_l_m][te], device=args.probe_device)
        s_v_test = probe_v.predict_proba(feat_v[best_l_v][te], device=args.probe_device)
        s_f_test = (1.0 - lam) * s_m_test + lam * s_v_test
        y_test = y[te]

        m_m = binary_metrics(y_test, s_m_test)
        m_v = binary_metrics(y_test, s_v_test)
        m_f = binary_metrics(y_test, s_f_test)

        by_seed.extend(
            [
                {
                    "subset": subset_name,
                    "seed": seed,
                    "variant": "target_only",
                    "best_layer_target": best_l_m,
                    "best_layer_verifier": None,
                    "lambda": 0.0,
                    "val_auroc_fused": float("nan"),
                    "probe_best_val_auroc_target": probe_m.best_val_auroc,
                    "probe_best_val_auroc_verifier": float("nan"),
                    **m_m,
                },
                {
                    "subset": subset_name,
                    "seed": seed,
                    "variant": "verifier_only",
                    "best_layer_target": None,
                    "best_layer_verifier": best_l_v,
                    "lambda": 1.0,
                    "val_auroc_fused": float("nan"),
                    "probe_best_val_auroc_target": float("nan"),
                    "probe_best_val_auroc_verifier": probe_v.best_val_auroc,
                    **m_v,
                },
                {
                    "subset": subset_name,
                    "seed": seed,
                    "variant": "cross_model_fused",
                    "best_layer_target": best_l_m,
                    "best_layer_verifier": best_l_v,
                    "lambda": lam,
                    "val_auroc_fused": val_auc_fused,
                    "probe_best_val_auroc_target": probe_m.best_val_auroc,
                    "probe_best_val_auroc_verifier": probe_v.best_val_auroc,
                    **m_f,
                },
            ]
        )

        test_indices = np.where(te)[0]
        for j, gi in enumerate(test_indices):
            test_scores.append(
                {
                    "subset": subset_name,
                    "seed": seed,
                    "question_id": df.iloc[gi]["question_id"],
                    "y_true": int(y[gi]),
                    "score_target": float(s_m_test[j]),
                    "score_verifier": float(s_v_test[j]),
                    "score_fused": float(s_f_test[j]),
                    "lambda": float(lam),
                }
            )

    return pd.DataFrame(by_seed), pd.DataFrame(test_scores)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder_device = resolve_device(args.encoder_device)
    probe_device = resolve_device(args.probe_device)
    maybe_require_hf_token(args.response_model_path_or_hf_id)
    maybe_require_hf_token(args.verifier_model_path_or_hf_id)

    rows = load_jsonl(input_path)

    if args.subset == "both":
        subsets = ["ce", "ie"]
    else:
        subsets = [args.subset]

    # Get layer counts from config (no model load)
    resp_config = from_pretrained_with_remote_code_fallback(AutoConfig, args.response_model_path_or_hf_id)
    ver_config = from_pretrained_with_remote_code_fallback(AutoConfig, args.verifier_model_path_or_hf_id)
    layers_resp = parse_layers(args.layer_candidates, infer_num_hidden_layers_from_config(resp_config))
    layers_ver = parse_layers(args.layer_candidates, infer_num_hidden_layers_from_config(ver_config))

    response_encoder: Optional[CausalLastTokenEncoder] = None
    verifier_encoder: Optional[CausalLastTokenEncoder] = None
    if not getattr(args, "sequential_encoders", False):
        response_encoder = CausalLastTokenEncoder(
            model_id=args.response_model_path_or_hf_id,
            device=encoder_device,
            max_length=args.max_length,
            batch_size=args.batch_size,
            torch_dtype=args.torch_dtype,
            device_map_mode=args.encoder_device_map,
        )
        verifier_encoder = CausalLastTokenEncoder(
            model_id=args.verifier_model_path_or_hf_id,
            device=encoder_device,
            max_length=args.max_length,
            batch_size=args.batch_size,
            torch_dtype=args.torch_dtype,
            device_map_mode=args.encoder_device_map,
        )
        layers_resp = parse_layers(args.layer_candidates, response_encoder.num_hidden_layers)
        layers_ver = parse_layers(args.layer_candidates, verifier_encoder.num_hidden_layers)

    all_seed_rows: List[pd.DataFrame] = []
    all_test_rows: List[pd.DataFrame] = []
    subset_counts: Dict[str, Dict[str, int]] = {}

    for sb in subsets:
        df = build_subset_dataframe(
            rows=rows,
            target_model_name=args.target_model_name,
            subset=sb,
            sample_seed=args.sample_seed,
            ce_threshold=args.ce_threshold,
        )
        subset_counts[sb] = {
            "rows": int(len(df)),
            "positive_correct": int((df["y"] == 0).sum()),
            "negative_error": int((df["y"] == 1).sum()),
        }

        feat_m: Optional[Dict[int, np.ndarray]] = None
        feat_v: Optional[Dict[int, np.ndarray]] = None
        if getattr(args, "sequential_encoders", False):
            resp_enc = CausalLastTokenEncoder(
                model_id=args.response_model_path_or_hf_id,
                device=encoder_device,
                max_length=args.max_length,
                batch_size=args.batch_size,
                torch_dtype=args.torch_dtype,
                device_map_mode=args.encoder_device_map,
            )
            texts = df["probe_text"].tolist()
            feat_m = resp_enc.encode_last_token_layers(texts, layers_resp)
            release_encoder_cuda(resp_enc)
            ver_enc = CausalLastTokenEncoder(
                model_id=args.verifier_model_path_or_hf_id,
                device=encoder_device,
                max_length=args.max_length,
                batch_size=args.batch_size,
                torch_dtype=args.torch_dtype,
                device_map_mode=args.encoder_device_map,
            )
            feat_v = ver_enc.encode_last_token_layers(texts, layers_ver)
            release_encoder_cuda(ver_enc)

        seed_df, test_df = run_subset(
            df=df,
            subset_name=sb,
            layers_resp=layers_resp,
            layers_ver=layers_ver,
            args=argparse.Namespace(**{**vars(args), "probe_device": probe_device}),
            response_encoder=response_encoder,
            verifier_encoder=verifier_encoder,
            feat_m=feat_m,
            feat_v=feat_v,
        )
        all_seed_rows.append(seed_df)
        all_test_rows.append(test_df)

    metrics_by_seed = pd.concat(all_seed_rows, ignore_index=True) if all_seed_rows else pd.DataFrame()
    test_scores = pd.concat(all_test_rows, ignore_index=True) if all_test_rows else pd.DataFrame()

    summary_rows: List[Dict[str, Any]] = []
    if not metrics_by_seed.empty:
        for (subset, variant), sub in metrics_by_seed.groupby(["subset", "variant"]):
            row: Dict[str, Any] = {
                "subset": subset,
                "variant": variant,
                "n_runs": int(len(sub)),
            }
            for col in [
                "auroc",
                "prauc",
                "accuracy_at_0_5",
                "brier",
                "lambda",
                "val_auroc_fused",
                "probe_best_val_auroc_target",
                "probe_best_val_auroc_verifier",
            ]:
                vals = sub[col].to_numpy(dtype=float)
                row[f"{col}_mean"] = float(np.nanmean(vals))
                row[f"{col}_std"] = float(np.nanstd(vals))
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    metrics_by_seed_path = out_dir / "wb_cross_model_probe_emnlp2025_metrics_by_seed.csv"
    summary_path = out_dir / "wb_cross_model_probe_emnlp2025_metrics_summary.csv"
    test_scores_path = out_dir / "wb_cross_model_probe_emnlp2025_test_scores.csv"
    report_path = out_dir / "wb_cross_model_probe_emnlp2025_run_report.json"

    metrics_by_seed.to_csv(metrics_by_seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    test_scores.to_csv(test_scores_path, index=False)

    report = {
        "input": str(input_path),
        "target_model_name": args.target_model_name,
        "subset_mode": args.subset,
        "subset_counts": subset_counts,
        "response_model_path_or_hf_id": args.response_model_path_or_hf_id,
        "verifier_model_path_or_hf_id": args.verifier_model_path_or_hf_id,
        "layers_response": layers_resp,
        "layers_verifier": layers_ver,
        "encoder_device": args.encoder_device,
        "probe_device": args.probe_device,
        "resolved_encoder_device": encoder_device,
        "resolved_probe_device": probe_device,
        "encoder_device_map": args.encoder_device_map,
        "probe_seeds": [int(s.strip()) for s in args.probe_seeds.split(",") if s.strip()],
        "lambda_step": args.lambda_step,
        "ce_threshold": args.ce_threshold,
        "method_notes": [
            "Last-token hidden state probing with per-layer selection (validation AUROC).",
            "4-layer FFN probes (dim->256->128->64->1) + ReLU as in paper.",
            "Cross-model fusion with lambda grid 0..1 step 0.05 (configurable).",
            f"CE/IE balanced subset construction from existing evaluated labels (CE threshold={args.ce_threshold:.2f}).",
            "Dataset/protocol differences from paper remain possible (e.g., benchmark and sampling k).",
        ],
        "outputs": {
            "metrics_by_seed_csv": str(metrics_by_seed_path),
            "metrics_summary_csv": str(summary_path),
            "test_scores_csv": str(test_scores_path),
        },
        "elapsed_seconds": time.time() - t0,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["outputs"]["run_report_json"] = str(report_path)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run near-faithful WB-CrossModelProbe (EMNLP 2025 style).")
    p.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT))

    p.add_argument(
        "--target-model-name",
        type=str,
        required=True,
        help="Model name in evaluated JSONL whose rows will be used as response-model outputs.",
    )
    p.add_argument(
        "--subset",
        type=str,
        default="both",
        choices=["ce", "ie", "both"],
        help="Run CE subset, IE subset, or both.",
    )
    p.add_argument("--sample-seed", type=int, default=42, help="Balancing seed for positive/negative matching.")

    p.add_argument(
        "--response-model-path-or-hf-id",
        type=str,
        required=True,
        help="Local path or HF id for response model M (hidden states).",
    )
    p.add_argument(
        "--verifier-model-path-or-hf-id",
        type=str,
        required=True,
        help="Local path or HF id for verifier model V (hidden states).",
    )
    p.add_argument(
        "--layer-candidates",
        type=str,
        default="all",
        help="Layer selection candidates: 'all', 'last8', or comma list (e.g. '20,24,28,32').",
    )
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--encoder-device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--probe-device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch-dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument(
        "--encoder-device-map",
        type=str,
        default="none",
        choices=["none", "auto"],
        help="Use Transformers/Accelerate sharded loading for encoders (e.g., multi-GPU via device_map=auto).",
    )

    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)

    p.add_argument("--probe-seeds", type=str, default="11,22,33")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-step", type=float, default=0.05)
    p.add_argument(
        "--ce-threshold",
        type=float,
        default=1.0,
        help="Semantic-equivalence threshold for CE labeling (1.0 reproduces strict paper CE).",
    )
    p.add_argument(
        "--sequential-encoders",
        action="store_true",
        help="Load response and verifier encoders one at a time to reduce peak memory (for ~18GB machines).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.ce_threshold <= 1.0):
        raise ValueError("--ce-threshold must be in (0, 1].")
    report = run(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
