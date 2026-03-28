import unittest
from concurrent.futures import Future
from unittest.mock import patch

IMPORT_ERROR = None

try:
    from src.providers import MultiProviderClient
except ImportError as exc:  # pragma: no cover - environment dependent
    IMPORT_ERROR = exc


class _DummyProvider:
    def generate(
        self,
        model,
        prompt,
        temperature=0.7,
        top_p=0.9,
        max_tokens=50,
        response_format=None,
        request_overrides=None,
    ):
        return f"sample-{temperature}-{top_p}"

    def get_last_retry_count(self):
        return 0


class _CapturingExecutor:
    last_max_workers = None

    def __init__(self, max_workers):
        self.max_workers = max_workers
        _CapturingExecutor.last_max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        fut = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover
            fut.set_exception(exc)
        return fut


@unittest.skipIf(IMPORT_ERROR is not None, f"provider deps unavailable: {IMPORT_ERROR}")
class ProviderConcurrencyTests(unittest.TestCase):
    def test_generate_stochastic_respects_max_concurrency(self):
        client = MultiProviderClient(max_concurrency_per_provider=3)

        with patch.object(client, "_get_provider", return_value=_DummyProvider()):
            with patch("src.providers.ThreadPoolExecutor", _CapturingExecutor):
                results = client.generate_stochastic(
                    provider="openai",
                    model="dummy",
                    prompt="Q",
                    num_samples=10,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=8,
                )

        self.assertEqual(_CapturingExecutor.last_max_workers, 3)
        self.assertEqual(len(results), 10)
        indices = [int(r.request_meta["sample_index"]) for r in results]
        self.assertEqual(indices, list(range(10)))


if __name__ == "__main__":
    unittest.main()
