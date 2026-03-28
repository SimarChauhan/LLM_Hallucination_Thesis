import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

IMPORT_ERROR = None

try:
    from scripts import run_pipeline
    from src.schemas import GenerationParams, GenerationResult, Question
except ImportError as exc:  # pragma: no cover - environment dependent
    IMPORT_ERROR = exc


class _FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    def generate_greedy(self, provider, model, prompt, max_new_tokens=50, response_format=None, request_overrides=None):
        return GenerationResult(
            text="Paris",
            params=GenerationParams(
                do_sample=False,
                temperature=0.01,
                top_p=1.0,
                top_k=None,
                max_new_tokens=max_new_tokens,
            ),
            request_meta={"latency_ms": 1.0, "retry_count": 0},
        )

    def generate_stochastic(
        self,
        provider,
        model,
        prompt,
        num_samples=10,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=50,
        request_overrides=None,
    ):
        # Intentionally short to verify incomplete handling.
        out = []
        for i in range(8):
            out.append(
                GenerationResult(
                    text=f"Paris_{i}",
                    params=GenerationParams(
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=None,
                        max_new_tokens=max_new_tokens,
                    ),
                    request_meta={"sample_index": i, "latency_ms": 1.0, "retry_count": 0},
                )
            )
        return out


@unittest.skipIf(IMPORT_ERROR is not None, f"pipeline deps unavailable: {IMPORT_ERROR}")
class PipelineIncompleteTests(unittest.TestCase):
    def test_pipeline_marks_incomplete_and_enqueues_retry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = {
                "protocol": {"version": "v3", "high_rigor": True, "prompt_version": "qa-short-v1"},
                "collection": {
                    "required_samples": 10,
                    "strict_samples": True,
                    "retry_incomplete": True,
                    "max_concurrency_per_provider": 2,
                },
                "inference": {
                    "greedy": {"max_new_tokens": 8},
                    "stochastic": {
                        "num_samples": 10,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_new_tokens": 8,
                    },
                },
                "output": {
                    "results_dir": tmpdir,
                    "raw_dir": "raw",
                    "results_file": "results.jsonl",
                    "parquet_file": "results.parquet",
                    "immutable_runs": False,
                },
                "dataset": {"name": "truthful_qa", "split": "validation"},
                "rate_limit": {},
                "judge": {},
                "experiment": {"run_commercial": True, "run_opensource": False},
            }

            with patch("scripts.run_pipeline.load_config", return_value=cfg):
                with patch(
                    "scripts.run_pipeline.get_models_to_test",
                    return_value=[{"provider": "openai", "model": "m", "name": "Model-A"}],
                ):
                    with patch("scripts.run_pipeline.check_api_keys", return_value={"openai": True}):
                        with patch(
                            "scripts.run_pipeline.load_questions_for_benchmark",
                            return_value=[Question(id="q1", text="Capital of France?", ground_truths=["Paris"])],
                        ):
                            with patch("scripts.run_pipeline.MultiProviderClient", _FakeClient):
                                with patch(
                                    "scripts.run_pipeline.ResultStorage.export_to_parquet",
                                    return_value=str(Path(tmpdir) / "raw" / "results.parquet"),
                                ):
                                    stats = run_pipeline.run_pipeline("unused-config-path")

            self.assertEqual(stats["incomplete_records"], 1)

            results_file = Path(tmpdir) / "raw" / "results.jsonl"
            with open(results_file, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

            self.assertEqual(len(rows), 1)
            rec = rows[0]
            self.assertTrue(rec["is_incomplete"])
            self.assertEqual(rec["stochastic_target_n"], 10)
            self.assertEqual(rec["stochastic_actual_n"], 8)

            retry_queue = Path(tmpdir) / "raw" / "retry_queue.jsonl"
            self.assertTrue(retry_queue.exists())
            with open(retry_queue, "r", encoding="utf-8") as f:
                queue_rows = [json.loads(line) for line in f if line.strip()]
            self.assertTrue(queue_rows)
            self.assertIn("incomplete_samples:8/10", queue_rows[0]["reason"])


if __name__ == "__main__":
    unittest.main()
