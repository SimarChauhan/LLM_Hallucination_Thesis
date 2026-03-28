"""HuggingFace Inference API wrapper using the chat completions endpoint."""

import logging
import time
import os
import json
from typing import List, Optional

import requests

from .schemas import GenerationParams, GenerationResult

logger = logging.getLogger(__name__)

# HuggingFace chat completions endpoint (OpenAI-compatible)
HF_CHAT_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"


class HFInferenceClient:
    """Wrapper around HuggingFace's OpenAI-compatible chat completions API."""
    
    def __init__(
        self,
        token: Optional[str] = None,
        initial_delay: float = 2.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0
    ):
        """
        Initialize the HF Inference client.
        
        Args:
            token: HuggingFace API token (defaults to HF_TOKEN env var)
            initial_delay: Initial delay for rate limiting (seconds)
            max_delay: Maximum delay for exponential backoff (seconds)
            backoff_factor: Multiplier for exponential backoff
        """
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        if not self.token:
            raise ValueError("No HF_TOKEN or HUGGINGFACE_API_KEY found. Set one in environment variables.")
        
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
    
    def _make_request_with_retry(
        self,
        model: str,
        messages: List[dict],
        temperature: float,
        max_tokens: int,
        max_retries: int = 5
    ) -> str:
        """
        Make a chat completion request with exponential backoff on errors.
        
        Args:
            model: Model identifier (e.g., "Qwen/Qwen2.5-72B-Instruct")
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_retries: Maximum number of retry attempts
        
        Returns:
            Generated text content
        
        Raises:
            Exception: If all retries fail
        """
        delay = self.initial_delay
        last_exception = None
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    HF_CHAT_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                
                if response.ok:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                
                # Handle specific error codes
                error_text = response.text
                
                if response.status_code == 429:  # Rate limit
                    logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries}). Waiting {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * self.backoff_factor, self.max_delay)
                    continue
                elif response.status_code == 503:  # Model loading
                    logger.warning(f"Model loading (attempt {attempt + 1}/{max_retries}). Waiting {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * self.backoff_factor, self.max_delay)
                    continue
                else:
                    last_exception = Exception(f"HTTP {response.status_code}: {error_text}")
                    logger.error(f"API error (attempt {attempt + 1}/{max_retries}): {response.status_code} - {error_text[:200]}")
                    time.sleep(delay)
                    delay = min(delay * self.backoff_factor, self.max_delay)
                    
            except requests.exceptions.Timeout:
                last_exception = Exception("Request timeout")
                logger.warning(f"Timeout (attempt {attempt + 1}/{max_retries}). Waiting {delay:.1f}s...")
                time.sleep(delay)
                delay = min(delay * self.backoff_factor, self.max_delay)
            except Exception as e:
                last_exception = e
                logger.error(f"Request error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(delay)
                delay = min(delay * self.backoff_factor, self.max_delay)
        
        raise Exception(f"Failed after {max_retries} retries. Last error: {last_exception}")
    
    def generate_greedy(
        self,
        model: str,
        prompt: str,
        max_new_tokens: int = 50
    ) -> GenerationResult:
        """
        Generate a single answer using greedy decoding (low temperature).
        
        Args:
            model: Model identifier (e.g., "Qwen/Qwen2.5-72B-Instruct")
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            GenerationResult with the generated text and parameters
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Use very low temperature for near-deterministic output
        text = self._make_request_with_retry(
            model=model,
            messages=messages,
            temperature=0.01,  # Near-deterministic
            max_tokens=max_new_tokens
        )
        
        gen_params = GenerationParams(
            do_sample=False,
            temperature=0.01,
            top_p=1.0,
            top_k=None,
            max_new_tokens=max_new_tokens
        )
        
        return GenerationResult(
            text=text.strip(),
            params=gen_params,
            logprobs=None
        )
    
    def generate_stochastic(
        self,
        model: str,
        prompt: str,
        num_samples: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 50
    ) -> List[GenerationResult]:
        """
        Generate multiple stochastic samples.
        
        Args:
            model: Model identifier
            prompt: Input prompt
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter (not directly supported, using temperature only)
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            List of GenerationResult objects
        """
        messages = [{"role": "user", "content": prompt}]
        
        gen_params = GenerationParams(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=None,
            max_new_tokens=max_new_tokens
        )
        
        results = []
        for i in range(num_samples):
            try:
                text = self._make_request_with_retry(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens
                )
                results.append(GenerationResult(
                    text=text.strip(),
                    params=gen_params,
                    logprobs=None
                ))
            except Exception as e:
                logger.error(f"Failed to generate sample {i+1}/{num_samples}: {e}")
                continue
        
        if not results:
            raise Exception(f"Failed to generate any samples for prompt")
        
        return results


def build_qa_prompt(question: str, model: str = "") -> str:
    """
    Build a prompt for QA generation.
    
    Args:
        question: The question text
        model: Model identifier (for model-specific formatting)
    
    Returns:
        Formatted prompt string
    """
    prompt = f"""Answer the following question with a short, factual answer. Give only the answer, no explanation.

Question: {question}

Answer:"""
    
    return prompt


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Testing HF Inference Client (Chat Completions API)...")
    
    try:
        client = HFInferenceClient()
        
        test_prompt = build_qa_prompt("What is the capital of France?")
        print(f"\nTest prompt:\n{test_prompt}")
        
        model = "Qwen/Qwen2.5-72B-Instruct"
        print(f"\nTesting greedy generation with {model}...")
        
        result = client.generate_greedy(model, test_prompt)
        print(f"Greedy result: {result.text}")
        
    except Exception as e:
        print(f"Error: {e}")
