"""
Multi-provider LLM inference module.

Supports:
- OpenAI (GPT-4, GPT-4o, etc.)
- Anthropic (Claude 3.5, Claude 3, etc.)
- Google (Gemini 1.5, Gemini 2.0, etc.)
- xAI (Grok-2, etc.)
- DeepSeek (DeepSeek API)
- Groq (Llama 4 Maverick, etc.; OpenAI-compatible)
- HuggingFace (Open-source models via Inference API)
- HuggingFace Local (open-source models from local/Hub weights via Transformers)
"""

import logging
import os
import time
import threading
import re
import copy
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any, Tuple

import requests

from .schemas import GenerationParams, GenerationResult

logger = logging.getLogger(__name__)


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return a new dict."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_payload(base_payload: Dict[str, Any], request_overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply request overrides safely to a provider payload."""
    if not request_overrides:
        return base_payload
    if not isinstance(request_overrides, dict):
        raise ValueError("request_overrides must be a dictionary when provided.")
    return _deep_merge_dict(base_payload, request_overrides)


def _extract_chat_content(data: Dict[str, Any]) -> str:
    """Best-effort extraction of assistant text from OpenAI-style chat payloads."""
    choices = data.get("choices") or []
    if not choices:
        raise Exception(f"No choices in response payload: {data}")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        joined = "".join(parts).strip()
        if joined:
            return joined

    # Some providers return reasoning traces with null content.
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning

    raise Exception(f"Empty assistant content in response payload: {data}")


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        initial_delay: float = 2.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        max_retries: int = 5
    ):
        self.api_key = api_key
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.max_retries = max_retries
        self._thread_local = threading.local()
    
    @abstractmethod
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.01,
        top_p: Optional[float] = None,
        max_tokens: int = 50,
        response_format: Optional[Dict[str, Any]] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a response from the model."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff on failure."""
        delay = self.initial_delay
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                self._thread_local.last_retry_count = attempt
                return result
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                status_match = re.search(r"http\s+(\d{3})", str(e), flags=re.IGNORECASE)
                status_code = int(status_match.group(1)) if status_match else None

                # Fail fast on non-retriable client errors.
                # Keep retrying 429 because it is a transient quota/rate condition.
                if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                    self._thread_local.last_retry_count = attempt
                    raise e
                
                # Check for rate limit errors
                if "rate" in error_msg or "429" in error_msg or "quota" in error_msg:
                    logger.warning(f"[{self.provider_name}] Rate limited (attempt {attempt + 1}/{self.max_retries}). Waiting {delay:.1f}s...")
                elif "overloaded" in error_msg or "503" in error_msg:
                    logger.warning(f"[{self.provider_name}] Service overloaded (attempt {attempt + 1}/{self.max_retries}). Waiting {delay:.1f}s...")
                else:
                    logger.warning(f"[{self.provider_name}] Error (attempt {attempt + 1}/{self.max_retries}): {e}. Waiting {delay:.1f}s...")
                
                time.sleep(delay)
                delay = min(delay * self.backoff_factor, self.max_delay)

        self._thread_local.last_retry_count = max(self.max_retries - 1, 0)
        raise Exception(f"[{self.provider_name}] Failed after {self.max_retries} retries. Last error: {last_exception}")

    def get_last_retry_count(self) -> int:
        """Return retry count for the current thread's most recent request."""
        return int(getattr(self._thread_local, "last_retry_count", 0))


class OpenAIProvider(BaseProvider):
    """OpenAI API provider (GPT-4, GPT-4o, GPT-5.2, etc.)."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Set it in environment variables.")
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    @property
    def provider_name(self) -> str:
        return "OpenAI"
    
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.01,
        top_p: Optional[float] = None,
        max_tokens: int = 50,
        response_format: Optional[Dict[str, Any]] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        def _make_request():
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            # Newer models (e.g. GPT-5.2) require max_completion_tokens instead of max_tokens
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_tokens,
                "temperature": temperature
            }
            if top_p is not None:
                payload["top_p"] = top_p
            if response_format is not None:
                payload["response_format"] = response_format
            payload = _build_payload(payload, request_overrides)
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if not response.ok:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            data = response.json()
            return _extract_chat_content(data)
        
        return self._retry_with_backoff(_make_request)


class AnthropicProvider(BaseProvider):
    """Anthropic API provider (Claude 3.5, Claude 3, etc.)."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found. Set it in environment variables.")
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.api_version = "2023-06-01"
    
    @property
    def provider_name(self) -> str:
        return "Anthropic"
    
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.01,
        top_p: Optional[float] = None,
        max_tokens: int = 50,
        response_format: Optional[Dict[str, Any]] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        def _make_request():
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": self.api_version
            }
            
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            # Anthropic requires temperature >= 0, and uses 1.0 as default
            if temperature > 0:
                payload["temperature"] = temperature
            # Newer Claude models reject requests that include both
            # temperature and top_p. Prefer temperature when both are provided.
            if top_p is not None and "temperature" not in payload:
                payload["top_p"] = top_p
            if response_format is not None:
                logger.debug(
                    "[Anthropic] response_format requested but not natively supported by this provider wrapper; using prompt-only JSON constraints."
                )
            payload = _build_payload(payload, request_overrides)
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if not response.ok:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            data = response.json()
            return data["content"][0]["text"]
        
        return self._retry_with_backoff(_make_request)


class GoogleProvider(BaseProvider):
    """Google Gemini API provider."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found. Set it in environment variables.")
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    
    @property
    def provider_name(self) -> str:
        return "Google"
    
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.01,
        top_p: Optional[float] = None,
        max_tokens: int = 50,
        response_format: Optional[Dict[str, Any]] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        def _make_request():
            # Accept both "gemini-*" and "google/gemini-*" forms.
            model_id = model.split("/", 1)[1] if model.startswith("google/") else model
            url = f"{self.base_url}/{model_id}:generateContent?key={self.api_key}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "contents": [
                    {
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens
                }
            }
            if top_p is not None:
                payload["generationConfig"]["topP"] = top_p
            if response_format is not None:
                # Gemini supports JSON response MIME type in generation config.
                payload["generationConfig"]["responseMimeType"] = "application/json"
            payload = _build_payload(payload, request_overrides)
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if not response.ok:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            data = response.json()
            
            # Handle potential safety blocks
            if "candidates" not in data or not data["candidates"]:
                if "promptFeedback" in data:
                    raise Exception(f"Content blocked: {data['promptFeedback']}")
                raise Exception(f"No response generated: {data}")
            
            return data["candidates"][0]["content"]["parts"][0]["text"]
        
        return self._retry_with_backoff(_make_request)


class XAIProvider(BaseProvider):
    """xAI API provider (Grok)."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY not found. Set it in environment variables.")
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = "https://api.x.ai/v1/chat/completions"
    
    @property
    def provider_name(self) -> str:
        return "xAI"
    
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.01,
        top_p: Optional[float] = None,
        max_tokens: int = 50,
        response_format: Optional[Dict[str, Any]] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        def _make_request():
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            if top_p is not None:
                payload["top_p"] = top_p
            if response_format is not None:
                payload["response_format"] = response_format
            payload = _build_payload(payload, request_overrides)
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if not response.ok:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            data = response.json()
            return _extract_chat_content(data)
        
        return self._retry_with_backoff(_make_request)


class DeepSeekProvider(BaseProvider):
    """DeepSeek API provider (OpenAI-compatible)."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found. Set it in environment variables.")
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
    
    @property
    def provider_name(self) -> str:
        return "DeepSeek"
    
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.01,
        top_p: Optional[float] = None,
        max_tokens: int = 50,
        response_format: Optional[Dict[str, Any]] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        def _make_request():
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            if top_p is not None:
                payload["top_p"] = top_p
            if response_format is not None:
                payload["response_format"] = response_format
            payload = _build_payload(payload, request_overrides)
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if not response.ok:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            data = response.json()
            return _extract_chat_content(data)
        
        return self._retry_with_backoff(_make_request)


class GroqProvider(BaseProvider):
    """Groq API provider (OpenAI-compatible; e.g. Llama 4 Maverick)."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Set it in environment variables.")
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    @property
    def provider_name(self) -> str:
        return "Groq"

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.01,
        top_p: Optional[float] = None,
        max_tokens: int = 50,
        response_format: Optional[Dict[str, Any]] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        def _make_request():
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            if top_p is not None:
                payload["top_p"] = top_p
            if response_format is not None:
                payload["response_format"] = response_format
            payload = _build_payload(payload, request_overrides)
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            if not response.ok:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            data = response.json()
            return _extract_chat_content(data)
        return self._retry_with_backoff(_make_request)


class HuggingFaceProvider(BaseProvider):
    """HuggingFace Inference API provider (open-source models)."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("HF_TOKEN or HUGGINGFACE_API_KEY not found. Set it in environment variables.")
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = "https://router.huggingface.co/v1/chat/completions"
    
    @property
    def provider_name(self) -> str:
        return "HuggingFace"
    
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.01,
        top_p: Optional[float] = None,
        max_tokens: int = 50,
        response_format: Optional[Dict[str, Any]] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        def _make_request():
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            if top_p is not None:
                payload["top_p"] = top_p
            if response_format is not None:
                payload["response_format"] = response_format
            payload = _build_payload(payload, request_overrides)
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if not response.ok:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            data = response.json()
            return _extract_chat_content(data)
        
        return self._retry_with_backoff(_make_request)


class OpenRouterProvider(BaseProvider):
    """OpenRouter API provider (OpenAI-compatible; e.g. Qwen3 Next 80B)."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found. Set it in environment variables.")
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    @property
    def provider_name(self) -> str:
        return "OpenRouter"

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.01,
        top_p: Optional[float] = None,
        max_tokens: int = 50,
        response_format: Optional[Dict[str, Any]] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        def _make_request():
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            if top_p is not None:
                payload["top_p"] = top_p
            if response_format is not None:
                payload["response_format"] = response_format
            payload = _build_payload(payload, request_overrides)
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            if not response.ok:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            data = response.json()
            return _extract_chat_content(data)
        return self._retry_with_backoff(_make_request)


class HuggingFaceLocalProvider(BaseProvider):
    """Local HuggingFace Transformers provider for open-weight checkpoints."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        # Local provider does not require an API key; keep base retry/backoff behavior.
        super().__init__(api_key=None, **kwargs)
        self._models: Dict[str, Tuple[Any, Any, str, threading.Lock]] = {}
        self._load_lock = threading.Lock()
        self._requested_device = os.environ.get("HF_LOCAL_DEVICE", "auto").strip().lower()
        self._requested_dtype = os.environ.get("HF_LOCAL_TORCH_DTYPE", "auto").strip().lower()
        self._trust_remote_code = os.environ.get("HF_LOCAL_TRUST_REMOTE_CODE", "0").strip().lower() in {
            "1", "true", "yes", "y", "on"
        }

    @property
    def provider_name(self) -> str:
        return "HuggingFaceLocal"

    def _resolve_device(self) -> str:
        import torch

        requested = self._requested_device
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("HF_LOCAL_DEVICE=cuda requested, but CUDA is unavailable.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ValueError("HF_LOCAL_DEVICE=mps requested, but MPS is unavailable.")
        return requested

    def _resolve_dtype(self):
        import torch

        mapping = {
            "auto": None,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if self._requested_dtype not in mapping:
            raise ValueError(
                f"Unsupported HF_LOCAL_TORCH_DTYPE={self._requested_dtype}. "
                "Use one of: auto,float16,bfloat16,float32"
            )
        return mapping[self._requested_dtype]

    def _load_model(self, model: str) -> Tuple[Any, Any, str, threading.Lock]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with self._load_lock:
            cached = self._models.get(model)
            if cached is not None:
                return cached

            device = self._resolve_device()
            dtype = self._resolve_dtype()
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                use_fast=True,
                trust_remote_code=self._trust_remote_code,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

            kwargs: Dict[str, Any] = {
                "trust_remote_code": self._trust_remote_code,
            }
            if dtype is not None:
                kwargs["torch_dtype"] = dtype

            lm = AutoModelForCausalLM.from_pretrained(model, **kwargs)
            lm.eval()
            lm.to(device)
            model_lock = threading.Lock()
            cached = (tokenizer, lm, device, model_lock)
            self._models[model] = cached
            logger.info(
                "[HuggingFaceLocal] loaded %s on device=%s dtype=%s",
                model,
                device,
                self._requested_dtype,
            )
            return cached

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.01,
        top_p: Optional[float] = None,
        max_tokens: int = 50,
        response_format: Optional[Dict[str, Any]] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Local generation does not enforce JSON-mode server-side.
        _ = response_format
        tokenizer, lm, device, model_lock = self._load_model(model)

        # For local models we interpret near-zero temperature as greedy decoding.
        do_sample = bool(temperature is not None and float(temperature) > 0.05)
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_tokens),
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            if top_p is not None:
                gen_kwargs["top_p"] = float(top_p)

        if request_overrides:
            # Local overrides may directly set generation kwargs.
            for key, value in request_overrides.items():
                if key in {
                    "max_new_tokens",
                    "min_new_tokens",
                    "do_sample",
                    "temperature",
                    "top_p",
                    "top_k",
                    "repetition_penalty",
                    "num_beams",
                    "length_penalty",
                    "no_repeat_ngram_size",
                }:
                    gen_kwargs[key] = value

        def _make_request():
            import torch

            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with model_lock:
                with torch.no_grad():
                    output = lm.generate(**encoded, **gen_kwargs)
            prompt_len = int(encoded["input_ids"].shape[1])
            completion_ids = output[0, prompt_len:]
            return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        return self._retry_with_backoff(_make_request)


# Provider registry mapping provider names to classes
PROVIDER_REGISTRY: Dict[str, type] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "xai": XAIProvider,
    "deepseek": DeepSeekProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
    "huggingface": HuggingFaceProvider,
    "huggingface_local": HuggingFaceLocalProvider,
}


def get_provider_for_model(model_config: Dict[str, Any], **kwargs) -> BaseProvider:
    """
    Get the appropriate provider instance for a model configuration.
    
    Args:
        model_config: Dictionary with 'provider' and 'model' keys
        **kwargs: Additional arguments passed to provider constructor
    
    Returns:
        Provider instance
    """
    provider_name = model_config.get("provider", "").lower()
    
    if provider_name not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(PROVIDER_REGISTRY.keys())}")
    
    provider_class = PROVIDER_REGISTRY[provider_name]
    return provider_class(**kwargs)


class MultiProviderClient:
    """
    Unified client that manages multiple providers.
    
    This class provides a consistent interface for generating text
    across different LLM providers.
    """
    
    def __init__(
        self,
        initial_delay: float = 2.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        max_concurrency_per_provider: Optional[int] = None,
    ):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.max_concurrency_per_provider = max_concurrency_per_provider
        self._providers: Dict[str, BaseProvider] = {}
    
    def _get_provider(self, provider_name: str) -> BaseProvider:
        """Get or create a provider instance."""
        provider_name = provider_name.lower()
        
        if provider_name not in self._providers:
            if provider_name not in PROVIDER_REGISTRY:
                raise ValueError(f"Unknown provider: {provider_name}. Available: {list(PROVIDER_REGISTRY.keys())}")
            
            provider_class = PROVIDER_REGISTRY[provider_name]
            self._providers[provider_name] = provider_class(
                initial_delay=self.initial_delay,
                max_delay=self.max_delay,
                backoff_factor=self.backoff_factor
            )
        
        return self._providers[provider_name]
    
    def generate_greedy(
        self,
        provider: str,
        model: str,
        prompt: str,
        max_new_tokens: int = 50,
        response_format: Optional[Dict[str, Any]] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """
        Generate a single answer using greedy decoding (low temperature).
        
        Args:
            provider: Provider name (openai, anthropic, google, xai, huggingface)
            model: Model identifier
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            GenerationResult with the generated text and parameters
        """
        provider_instance = self._get_provider(provider)
        started = time.perf_counter()
        text = provider_instance.generate(
            model=model,
            prompt=prompt,
            temperature=0.01,  # Near-deterministic
            top_p=1.0,
            max_tokens=max_new_tokens,
            response_format=response_format,
            request_overrides=request_overrides,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        retry_count = provider_instance.get_last_retry_count()
        stripped = text.strip()
        truncation_flag = (
            len(stripped) >= int(max_new_tokens * 3.5)
            or stripped.endswith("...")
        )
        
        gen_params = GenerationParams(
            do_sample=False,
            temperature=0.01,
            top_p=1.0,
            top_k=None,
            max_new_tokens=max_new_tokens
        )
        
        return GenerationResult(
            text=stripped,
            params=gen_params,
            logprobs=None,
            request_meta={
                "latency_ms": round(latency_ms, 2),
                "retry_count": retry_count,
                "finish_reason": "completed",
                "truncation_flag": truncation_flag,
            },
        )
    
    def generate_stochastic(
        self,
        provider: str,
        model: str,
        prompt: str,
        num_samples: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 50,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[GenerationResult]:
        """
        Generate multiple stochastic samples in parallel.
        
        All samples are fired concurrently using a thread pool, which
        provides ~5-8x speedup over sequential generation.
        
        Args:
            provider: Provider name
            model: Model identifier
            prompt: Input prompt
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter (not all providers support this)
            max_new_tokens: Maximum tokens to generate
            request_overrides: Optional provider-specific payload overrides
        
        Returns:
            List of GenerationResult objects
        """
        provider_instance = self._get_provider(provider)
        
        gen_params = GenerationParams(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=None,
            max_new_tokens=max_new_tokens
        )
        
        def _generate_one(sample_idx: int) -> GenerationResult:
            started = time.perf_counter()
            text = provider_instance.generate(
                model=model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                request_overrides=request_overrides,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            retry_count = provider_instance.get_last_retry_count()
            stripped = text.strip()
            truncation_flag = (
                len(stripped) >= int(max_new_tokens * 3.5)
                or stripped.endswith("...")
            )
            return GenerationResult(
                text=stripped,
                params=gen_params,
                logprobs=None,
                request_meta={
                    "sample_index": sample_idx,
                    "latency_ms": round(latency_ms, 2),
                    "retry_count": retry_count,
                    "finish_reason": "completed",
                    "truncation_flag": truncation_flag,
                },
            )

        results = []
        max_workers = num_samples
        if self.max_concurrency_per_provider is not None:
            max_workers = max(1, min(num_samples, int(self.max_concurrency_per_provider)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_generate_one, i): i
                for i in range(num_samples)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Failed to generate sample {idx+1}/{num_samples}: {e}")
        
        if not results:
            raise Exception(f"Failed to generate any samples for prompt")

        # Preserve deterministic sample ordering even though generation is parallel.
        results.sort(key=lambda r: int((r.request_meta or {}).get("sample_index", 0)))
        return results


def build_qa_prompt(question: str, model: str = "") -> str:
    """
    Build a prompt for QA generation.
    
    Args:
        question: The question text
        model: Model identifier (for model-specific formatting if needed)
    
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
    
    print("Testing Multi-Provider Client...")
    print("Available providers:", list(PROVIDER_REGISTRY.keys()))
    
    # Test prompt
    test_prompt = build_qa_prompt("What is the capital of France?")
    print(f"\nTest prompt:\n{test_prompt}")
