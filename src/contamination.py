"""Lightweight contamination checks for cross-benchmark overlap.

This module flags exact and near-duplicate question prompts across datasets/splits.
The intent is not perfect de-duplication; it is a conservative risk signal for
analysis-time exclusion/sensitivity reporting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


def normalize_question_text(text: str) -> str:
    """Normalize question text for overlap detection."""
    lowered = (text or "").strip().lower()
    lowered = _PUNCT_RE.sub(" ", lowered)
    lowered = _WHITESPACE_RE.sub(" ", lowered).strip()
    return lowered


def _token_jaccard(a: str, b: str) -> float:
    toks_a = set(a.split())
    toks_b = set(b.split())
    if not toks_a or not toks_b:
        return 0.0
    inter = len(toks_a & toks_b)
    union = len(toks_a | toks_b)
    return inter / union if union else 0.0


@dataclass
class ContaminationMatch:
    is_contaminated: bool
    reason: Optional[str] = None


class ContaminationIndex:
    """In-memory index to detect overlap against previously seen questions."""

    def __init__(
        self,
        similarity_threshold: float = 0.93,
        jaccard_threshold: float = 0.85,
        lookback: int = 5000,
    ):
        self.similarity_threshold = similarity_threshold
        self.jaccard_threshold = jaccard_threshold
        self.lookback = max(100, int(lookback))
        self._exact_map: Dict[str, str] = {}  # normalized question -> source key
        self._history: List[Tuple[str, str]] = []  # (normalized question, source key)

    def check(self, question_text: str, source_key: str) -> ContaminationMatch:
        """Check if the question appears contaminated relative to prior entries."""
        norm = normalize_question_text(question_text)
        if not norm:
            return ContaminationMatch(False, None)

        exact_source = self._exact_map.get(norm)
        if exact_source and exact_source != source_key:
            return ContaminationMatch(True, f"exact_duplicate_of:{exact_source}")

        # Restrict to recent history for runtime safety.
        candidates = self._history[-self.lookback :]
        for prev_norm, prev_source in candidates:
            if prev_source == source_key:
                continue
            # Fast length prefilter.
            if abs(len(norm) - len(prev_norm)) > max(20, int(0.35 * len(prev_norm))):
                continue
            jacc = _token_jaccard(norm, prev_norm)
            if jacc < self.jaccard_threshold:
                continue
            seq = SequenceMatcher(None, norm, prev_norm).ratio()
            if seq >= self.similarity_threshold:
                return ContaminationMatch(
                    True,
                    f"near_duplicate_of:{prev_source};seq={seq:.3f};jacc={jacc:.3f}",
                )

        return ContaminationMatch(False, None)

    def add(self, question_text: str, source_key: str) -> None:
        """Add a question to the index after checking."""
        norm = normalize_question_text(question_text)
        if not norm:
            return
        self._exact_map.setdefault(norm, source_key)
        self._history.append((norm, source_key))
