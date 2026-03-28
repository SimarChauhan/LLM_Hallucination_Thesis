"""Semantic equivalence checking using an LLM judge with 3-way classification."""

import logging
import re
from typing import List, Union, Optional

from .schemas import EquivalenceJudgment

# Support both old and new inference clients
try:
    from .providers import MultiProviderClient
except ImportError:
    MultiProviderClient = None

try:
    from .inference import HFInferenceClient
except ImportError:
    HFInferenceClient = None

logger = logging.getLogger(__name__)


# Prompt template for semantic equivalence judgment
JUDGE_PROMPT_TEMPLATE = """Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}

Do these two answers express the same factual claim?
Reply with exactly one word: "Same", "Different", or "Unclear".
- Same: Both answers make the same factual assertion
- Different: The answers make different factual claims
- Unclear: Cannot determine (e.g., one answer is gibberish, refusal, or ambiguous)

Your judgment:"""


class SemanticJudge:
    """LLM-based semantic equivalence judge with 3-way classification."""
    
    def __init__(
        self,
        inference_client: Union["MultiProviderClient", "HFInferenceClient", None] = None,
        judge_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        judge_provider: str = "huggingface"
    ):
        """
        Initialize the semantic judge.
        
        Args:
            inference_client: MultiProviderClient or HFInferenceClient for making API calls
            judge_model: Model to use for judging equivalence
            judge_provider: Provider for the judge model (used with MultiProviderClient)
        """
        self.client = inference_client
        self.judge_model = judge_model
        self.judge_provider = judge_provider
        self.raw_responses: List[str] = []  # Store for debugging
        
        # Detect client type for compatibility
        self._use_multi_provider = (
            MultiProviderClient is not None and 
            isinstance(inference_client, MultiProviderClient)
        )
    
    def _parse_judgment(self, response: str) -> EquivalenceJudgment:
        """
        Parse the judge's response into a judgment.
        
        Robustly extracts "same", "different", or "unclear" from the response,
        handling various response formats.
        
        Args:
            response: Raw response from the judge model
        
        Returns:
            One of "same", "different", "unclear"
        """
        # Store raw response for debugging
        self.raw_responses.append(response)
        
        if not response:
            return "unclear"
        
        response_lower = response.lower().strip()
        
        # Look for keywords anywhere in the response
        # Priority: same > different > unclear (most specific first)
        
        # Check for "same" - be careful not to match "not the same"
        if re.search(r'\bsame\b', response_lower):
            # Check for negations
            if re.search(r'\b(not|no|aren\'t|isn\'t|don\'t)\b.*\bsame\b', response_lower):
                return "different"
            return "same"
        
        # Check for "different"
        if re.search(r'\bdifferent\b', response_lower):
            return "different"
        
        # Check for "unclear" or related terms
        if re.search(r'\b(unclear|cannot|can\'t|unable|ambiguous|uncertain)\b', response_lower):
            return "unclear"
        
        # If none found, default to unclear
        logger.warning(f"Could not parse judgment from response: '{response[:100]}...'")
        return "unclear"
    
    def judge_equivalence(
        self,
        question: str,
        answer_a: str,
        answer_b: str
    ) -> EquivalenceJudgment:
        """
        Judge whether two answers are semantically equivalent.
        
        Args:
            question: The original question
            answer_a: First answer (typically the greedy answer)
            answer_b: Second answer (typically a stochastic sample)
        
        Returns:
            One of "same", "different", "unclear"
        """
        # Build the prompt
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            answer_a=answer_a,
            answer_b=answer_b
        )
        
        try:
            # Use greedy generation for consistent judgments
            if self._use_multi_provider:
                # New multi-provider client
                result = self.client.generate_greedy(
                    provider=self.judge_provider,
                    model=self.judge_model,
                    prompt=prompt,
                    max_new_tokens=20  # Short response expected
                )
            else:
                # Legacy HFInferenceClient
                result = self.client.generate_greedy(
                    self.judge_model,
                    prompt,
                    max_new_tokens=20  # Short response expected
                )
            
            judgment = self._parse_judgment(result.text)
            logger.debug(f"Judge response: '{result.text}' -> {judgment}")
            return judgment
            
        except Exception as e:
            logger.error(f"Error during semantic judgment: {e}")
            return "unclear"
    
    def judge_all_samples(
        self,
        question: str,
        greedy_answer: str,
        sample_answers: List[str]
    ) -> List[EquivalenceJudgment]:
        """
        Judge equivalence between greedy answer and all sample answers.
        
        Args:
            question: The original question
            greedy_answer: The greedy (deterministic) answer
            sample_answers: List of stochastic sample answers
        
        Returns:
            List of judgments, one per sample
        """
        judgments = []
        
        for i, sample in enumerate(sample_answers):
            logger.debug(f"Judging sample {i+1}/{len(sample_answers)}")
            judgment = self.judge_equivalence(question, greedy_answer, sample)
            judgments.append(judgment)
        
        return judgments
    
    def get_debug_info(self) -> dict:
        """
        Get debugging information about recent judgments.
        
        Returns:
            Dictionary with debug info
        """
        return {
            "total_judgments": len(self.raw_responses),
            "recent_responses": self.raw_responses[-10:] if self.raw_responses else []
        }
    
    def clear_debug_info(self):
        """Clear stored debug information."""
        self.raw_responses = []


if __name__ == "__main__":
    # Test the parsing logic
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing judgment parsing...\n")
    
    # Create a mock judge (won't make actual API calls)
    class MockClient:
        pass
    
    judge = SemanticJudge(MockClient(), "test-model")  # type: ignore
    
    test_responses = [
        ("Same", "same"),
        ("Different", "different"),
        ("Unclear", "unclear"),
        ("The answers are the same.", "same"),
        ("These are different claims.", "different"),
        ("I cannot determine if they match.", "unclear"),
        ("Same - both answers claim Paris is the capital", "same"),
        ("Not the same - one says Paris, the other says Lyon", "different"),
        ("", "unclear"),
        ("gibberish response xyz", "unclear"),
    ]
    
    print("Parse tests:")
    for response, expected in test_responses:
        result = judge._parse_judgment(response)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] '{response[:40]}...' -> {result} (expected: {expected})")
