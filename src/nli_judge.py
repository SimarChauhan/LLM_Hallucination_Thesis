"""
Bidirectional NLI-based Semantic Equivalence Judge using DeBERTa.

Based on the approach from:
- Kuhn et al. (2023) "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation"
- He et al. (2020) "DeBERTa: Decoding-enhanced BERT with Disentangled Attention"

Two answers are considered semantically equivalent if:
1. Answer A entails Answer B (A -> B)
2. Answer B entails Answer A (B -> A)

This bidirectional check ensures both answers share the same truth value.

Enhanced (Tier 1): all public methods now return NLIEquivalenceResult objects that
carry the raw entailment probabilities alongside the categorical judgment, enabling
downstream calibration, auditing, and threshold recalibration without re-running NLI.
"""

import json
import logging
from typing import List, Tuple, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .schemas import EquivalenceJudgment, NLIEquivalenceResult

logger = logging.getLogger(__name__)


# Default NLI model - DeBERTa trained on MNLI (Microsoft model returns 404 on HF; use community copy)
DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

# Alternative smaller models for resource-constrained environments
ALTERNATIVE_MODELS = {
    "large": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",  # Best accuracy
    "base": "microsoft/deberta-v3-base-mnli",        # Good balance, ~700MB
    "small": "microsoft/deberta-v3-small-mnli",     # Fastest, ~300MB
    "xlarge": "microsoft/deberta-v2-xlarge-mnli",    # Highest accuracy, ~3GB
}

# NLI label mapping (standard for MNLI-trained models)
NLI_LABELS = {
    0: "contradiction",
    1: "neutral",
    2: "entailment"
}


class NLISemanticJudge:
    """
    Bidirectional NLI-based semantic equivalence judge.

    Uses DeBERTa to check if two answers mutually entail each other,
    which indicates semantic equivalence.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: Optional[str] = None,
        entailment_threshold: float = 0.5,
        different_threshold: float = 0.3,
        batch_size: int = 8
    ):
        """
        Initialize the NLI judge.

        Args:
            model_name: HuggingFace model ID for NLI (must be trained on MNLI or similar)
            device: Device to run on ("cuda", "mps", "cpu", or None for auto-detect)
            entailment_threshold: Minimum probability for entailment ("same")
            different_threshold: Below this in either direction -> "different"
            batch_size: Batch size for inference (for efficiency)
        """
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.different_threshold = different_threshold
        self.batch_size = batch_size

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Loading NLI model: {model_name} on {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Get label mapping from model config
        self._setup_label_mapping()

        logger.info(f"NLI judge initialized successfully")

    def _setup_label_mapping(self):
        """Set up the mapping from model output indices to NLI labels."""
        # Check model's label2id mapping
        if hasattr(self.model.config, 'label2id'):
            label2id = self.model.config.label2id
            # Find entailment index
            for label, idx in label2id.items():
                if 'entail' in label.lower():
                    self.entailment_idx = idx
                    break
            else:
                # Default assumption: entailment is index 2 (standard for MNLI)
                self.entailment_idx = 2
        else:
            self.entailment_idx = 2

        logger.debug(f"Entailment index: {self.entailment_idx}")

    def _get_entailment_prob(self, premise: str, hypothesis: str) -> float:
        """
        Get the probability that premise entails hypothesis.

        Args:
            premise: The premise text
            hypothesis: The hypothesis text

        Returns:
            Probability of entailment (0 to 1)
        """
        # Tokenize
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            entailment_prob = probs[0, self.entailment_idx].item()

        return entailment_prob

    def _check_bidirectional_entailment(
        self,
        answer_a: str,
        answer_b: str
    ) -> Tuple[float, float, str]:
        """
        Check bidirectional entailment between two answers.

        Args:
            answer_a: First answer
            answer_b: Second answer

        Returns:
            Tuple of (prob_a_entails_b, prob_b_entails_a, judgment)
        """
        # Check A -> B
        prob_a_to_b = self._get_entailment_prob(answer_a, answer_b)

        # Check B -> A
        prob_b_to_a = self._get_entailment_prob(answer_b, answer_a)

        # Determine judgment based on bidirectional entailment
        # Both directions must show entailment for "same"
        if (prob_a_to_b >= self.entailment_threshold and
            prob_b_to_a >= self.entailment_threshold):
            judgment = "same"
        # If one direction is below different_threshold, they're clearly different
        elif (prob_a_to_b < self.different_threshold or prob_b_to_a < self.different_threshold):
            judgment = "different"
        # Ambiguous cases
        else:
            judgment = "unclear"

        return prob_a_to_b, prob_b_to_a, judgment

    def judge_equivalence(
        self,
        question: str,
        answer_a: str,
        answer_b: str
    ) -> NLIEquivalenceResult:
        """
        Judge whether two answers are semantically equivalent.

        Uses bidirectional NLI: answers are equivalent if they mutually entail each other.
        Returns an NLIEquivalenceResult with the judgment AND raw probabilities.

        Args:
            question: The original question (used for context, prepended to answers)
            answer_a: First answer (typically the greedy answer)
            answer_b: Second answer (typically a stochastic sample)

        Returns:
            NLIEquivalenceResult with judgment, prob_forward, prob_reverse
        """
        # Handle empty or very short answers
        if not answer_a or not answer_b:
            return NLIEquivalenceResult(judgment="unclear", prob_forward=0.0, prob_reverse=0.0)

        if len(answer_a.strip()) < 2 or len(answer_b.strip()) < 2:
            return NLIEquivalenceResult(judgment="unclear", prob_forward=0.0, prob_reverse=0.0)

        # Prepend question for context (helps NLI understand the domain)
        # Format: "Question: {q} Answer: {a}"
        context_a = f"Question: {question} Answer: {answer_a}"
        context_b = f"Question: {question} Answer: {answer_b}"

        try:
            prob_a_to_b, prob_b_to_a, judgment = self._check_bidirectional_entailment(
                context_a, context_b
            )

            logger.debug(
                f"NLI judgment: P(A->B)={prob_a_to_b:.3f}, P(B->A)={prob_b_to_a:.3f} -> {judgment}"
            )

            return NLIEquivalenceResult(
                judgment=judgment,
                prob_forward=prob_a_to_b,
                prob_reverse=prob_b_to_a,
            )

        except Exception as e:
            logger.error(f"Error during NLI judgment: {e}")
            return NLIEquivalenceResult(judgment="unclear", prob_forward=0.0, prob_reverse=0.0)

    def judge_all_samples(
        self,
        question: str,
        greedy_answer: str,
        sample_answers: List[str]
    ) -> List[NLIEquivalenceResult]:
        """
        Judge equivalence between greedy answer and all sample answers.

        Args:
            question: The original question
            greedy_answer: The greedy (deterministic) answer
            sample_answers: List of stochastic sample answers

        Returns:
            List of NLIEquivalenceResult, one per sample
        """
        results = []

        for i, sample in enumerate(sample_answers):
            logger.debug(f"Judging sample {i+1}/{len(sample_answers)}")
            result = self.judge_equivalence(question, greedy_answer, sample)
            results.append(result)

        return results

    def judge_batch(
        self,
        question: str,
        greedy_answer: str,
        sample_answers: List[str]
    ) -> List[NLIEquivalenceResult]:
        """
        Batch version of judge_all_samples for efficiency.

        Processes multiple comparisons in parallel using batched inference.

        Args:
            question: The original question
            greedy_answer: The greedy answer
            sample_answers: List of stochastic samples

        Returns:
            List of NLIEquivalenceResult
        """
        if not sample_answers:
            return []

        # Prepare all pairs for batched inference
        context_greedy = f"Question: {question} Answer: {greedy_answer}"
        contexts_samples = [
            f"Question: {question} Answer: {s}" for s in sample_answers
        ]

        results = []

        # Process in batches
        for i in range(0, len(contexts_samples), self.batch_size):
            batch_samples = contexts_samples[i:i + self.batch_size]

            # Forward direction: greedy -> sample
            probs_forward = self._batch_entailment(context_greedy, batch_samples)

            # Backward direction: sample -> greedy
            probs_backward = self._batch_entailment_reverse(batch_samples, context_greedy)

            # Determine judgments
            for prob_f, prob_b in zip(probs_forward, probs_backward):
                if prob_f >= self.entailment_threshold and prob_b >= self.entailment_threshold:
                    judgment = "same"
                elif prob_f < self.different_threshold or prob_b < self.different_threshold:
                    judgment = "different"
                else:
                    judgment = "unclear"
                results.append(NLIEquivalenceResult(
                    judgment=judgment,
                    prob_forward=prob_f,
                    prob_reverse=prob_b,
                ))

        return results

    def _batch_entailment(
        self,
        premise: str,
        hypotheses: List[str]
    ) -> List[float]:
        """Batch inference for premise -> multiple hypotheses."""
        premises = [premise] * len(hypotheses)

        inputs = self.tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            entailment_probs = probs[:, self.entailment_idx].tolist()

        return entailment_probs

    def _batch_entailment_reverse(
        self,
        premises: List[str],
        hypothesis: str
    ) -> List[float]:
        """Batch inference for multiple premises -> single hypothesis."""
        hypotheses = [hypothesis] * len(premises)

        inputs = self.tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            entailment_probs = probs[:, self.entailment_idx].tolist()

        return entailment_probs

    # ------------------------------------------------------------------
    # Tier 2 (C4): Threshold calibration against human annotations
    # ------------------------------------------------------------------

    def calibrate_thresholds(
        self,
        calibration_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Find optimal NLI thresholds using human-labeled calibration data.

        Performs grid search over (entailment_threshold, different_threshold)
        and selects the pair that maximises Cohen's kappa with human labels.

        Args:
            calibration_data: list of dicts, each with keys:
                - question (str)
                - answer_a (str)
                - answer_b (str)
                - human_label (str): "same", "different", or "unclear"

        Returns:
            Dict with best thresholds, kappa, and per-threshold results.
            Also updates ``self.entailment_threshold`` / ``self.different_threshold``
            in-place so subsequent calls use the calibrated values.
        """
        from sklearn.metrics import cohen_kappa_score

        # Collect NLI probs for all calibration pairs
        nli_probs = []
        human_labels = []
        for item in calibration_data:
            result = self.judge_equivalence(
                item["question"], item["answer_a"], item["answer_b"]
            )
            nli_probs.append((result.prob_forward, result.prob_reverse))
            human_labels.append(item["human_label"])

        # Grid search
        best_kappa = -1.0
        best_ent = self.entailment_threshold
        best_diff = self.different_threshold
        grid_results = []

        ent_candidates = [0.3, 0.4, 0.5, 0.6, 0.7]
        diff_candidates = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

        for ent_t in ent_candidates:
            for diff_t in diff_candidates:
                if diff_t >= ent_t:
                    continue  # invalid: different threshold must be < entailment threshold
                preds = []
                for (pf, pr) in nli_probs:
                    if pf >= ent_t and pr >= ent_t:
                        preds.append("same")
                    elif pf < diff_t or pr < diff_t:
                        preds.append("different")
                    else:
                        preds.append("unclear")
                try:
                    kappa = cohen_kappa_score(human_labels, preds)
                except Exception:
                    kappa = 0.0
                grid_results.append({
                    "entailment_threshold": ent_t,
                    "different_threshold": diff_t,
                    "kappa": round(kappa, 4),
                })
                if kappa > best_kappa:
                    best_kappa = kappa
                    best_ent = ent_t
                    best_diff = diff_t

        # Update in-place
        self.entailment_threshold = best_ent
        self.different_threshold = best_diff

        logger.info(
            f"NLI calibration complete: entailment={best_ent}, "
            f"different={best_diff}, kappa={best_kappa:.4f} "
            f"(n={len(calibration_data)})"
        )

        return {
            "entailment_threshold": best_ent,
            "different_threshold": best_diff,
            "kappa": round(best_kappa, 4),
            "n_samples": len(calibration_data),
            "grid_results": grid_results,
        }

    def save_calibration(self, path: str, calibration_result: Dict[str, Any]) -> None:
        """Persist calibration thresholds to a JSON file."""
        with open(path, "w") as f:
            json.dump(calibration_result, f, indent=2)
        logger.info(f"Saved NLI calibration to {path}")

    def load_calibration(self, path: str) -> Dict[str, Any]:
        """Load calibration thresholds from a JSON file and apply them."""
        with open(path, "r") as f:
            data = json.load(f)
        self.entailment_threshold = data["entailment_threshold"]
        self.different_threshold = data["different_threshold"]
        logger.info(
            f"Loaded NLI calibration from {path}: "
            f"entailment={self.entailment_threshold}, different={self.different_threshold}"
        )
        return data


# Factory function for easy instantiation
def create_nli_judge(
    model_size: str = "large",
    device: Optional[str] = None,
    entailment_threshold: float = 0.5,
    different_threshold: float = 0.3
) -> NLISemanticJudge:
    """
    Create an NLI judge with a specified model size.

    Args:
        model_size: One of "small", "base", "large", "xlarge"
        device: Device to use (None for auto-detect)
        entailment_threshold: Threshold for entailment ("same")
        different_threshold: Below this -> "different"

    Returns:
        NLISemanticJudge instance
    """
    model_name = ALTERNATIVE_MODELS.get(model_size, DEFAULT_NLI_MODEL)
    return NLISemanticJudge(
        model_name=model_name,
        device=device,
        entailment_threshold=entailment_threshold,
        different_threshold=different_threshold
    )


if __name__ == "__main__":
    # Test the NLI judge
    logging.basicConfig(level=logging.INFO)

    print("Testing NLI Semantic Judge...")
    print("=" * 60)

    # Create judge (will download model on first run)
    judge = create_nli_judge(model_size="base")  # Use base for faster testing

    # Test cases
    test_cases = [
        {
            "question": "What is the capital of France?",
            "answer_a": "Paris",
            "answer_b": "The capital is Paris",
            "expected": "same"
        },
        {
            "question": "What is the capital of France?",
            "answer_a": "Paris",
            "answer_b": "London",
            "expected": "different"
        },
        {
            "question": "What happens if you crack your knuckles?",
            "answer_a": "It causes arthritis",
            "answer_b": "It leads to joint problems and arthritis",
            "expected": "same"
        },
        {
            "question": "What happens if you crack your knuckles?",
            "answer_a": "It causes arthritis",
            "answer_b": "Nothing bad happens",
            "expected": "different"
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "answer_a": "Shakespeare",
            "answer_b": "William Shakespeare wrote it",
            "expected": "same"
        },
    ]

    print("\nRunning test cases:")
    print("-" * 60)

    for tc in test_cases:
        result = judge.judge_equivalence(
            tc["question"],
            tc["answer_a"],
            tc["answer_b"]
        )
        status = "+" if result.judgment == tc["expected"] else "x"
        print(f"{status} Q: {tc['question'][:40]}...")
        print(f"   A: '{tc['answer_a']}' vs '{tc['answer_b']}'")
        print(f"   Result: {result.judgment} (expected: {tc['expected']})")
        print(f"   Probs: P(A->B)={result.prob_forward:.3f}, P(B->A)={result.prob_reverse:.3f}")
        print()
