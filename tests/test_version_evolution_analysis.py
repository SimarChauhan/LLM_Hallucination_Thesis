import unittest

import pandas as pd

IMPORT_ERROR = None

try:
    from scripts.analyze_version_evolution import _pairwise_rows, _trend_rows, _validate
except ImportError as exc:  # pragma: no cover - environment dependent
    IMPORT_ERROR = exc


def _toy_df() -> pd.DataFrame:
    rows = []
    models = [
        ("M0", 0, {"q1": (0, 1), "q2": (0, 1), "q3": (1, 0)}),
        ("M1", 1, {"q1": (1, 0), "q2": (0, 1), "q3": (1, 0)}),
        ("M2", 2, {"q1": (1, 0), "q2": (1, 0), "q3": (1, 0)}),
    ]
    for model, idx, qmap in models:
        for qid, (correct, ce) in qmap.items():
            rows.append(
                {
                    "question_id": qid,
                    "model": model,
                    "model_track": "track-a",
                    "model_version_index": idx,
                    "greedy_correct": bool(correct),
                    "error_label_0.9": "self_consistent_error" if ce else "reliably_correct",
                    "config_hash": "hash1",
                    "judge_protocol": "proto1",
                    "prompt_version": "qa-short-v1",
                    "stochastic_actual_n": 10,
                }
            )
    return pd.DataFrame(rows)


@unittest.skipIf(IMPORT_ERROR is not None, f"analysis deps unavailable: {IMPORT_ERROR}")
class VersionEvolutionAnalysisTests(unittest.TestCase):
    def test_validate_passes_on_uniform_protocol(self):
        df = _toy_df()
        result = _validate(df, required_samples=10)
        self.assertTrue(result["passes"])
        self.assertEqual(result["rows_below_required_samples"], 0)

    def test_pairwise_rows_contains_expected_pairs(self):
        df = _toy_df()
        pairwise = _pairwise_rows(df, ce_label="0.9", bootstrap_iters=200, seed=7)
        self.assertFalse(pairwise.empty)
        self.assertEqual(len(pairwise), 6)  # 3 ordered pairs x 2 metrics
        self.assertTrue((pairwise["metric"] == "accuracy").any())
        self.assertTrue((pairwise["metric"] == "ce_rate").any())

    def test_trend_rows_reports_accuracy_and_ce(self):
        df = _toy_df()
        trends = _trend_rows(df, ce_label="0.9", bootstrap_iters=80, seed=11)
        self.assertFalse(trends.empty)
        self.assertTrue((trends["metric"] == "accuracy").any())
        self.assertTrue((trends["metric"] == "ce_rate").any())

        acc_slope = float(trends[trends["metric"] == "accuracy"]["slope_per_version"].iloc[0])
        ce_slope = float(trends[trends["metric"] == "ce_rate"]["slope_per_version"].iloc[0])
        self.assertGreater(acc_slope, 0.0)
        self.assertLess(ce_slope, 0.0)


if __name__ == "__main__":
    unittest.main()

