import unittest

IMPORT_ERROR = None

try:
    import pandas as pd
    from scripts.analyze_results import (
        _bootstrap_proportion_ci,
        table_10_paired_significance,
        validate_uniform_protocol,
    )
except ImportError as exc:  # pragma: no cover - environment dependent
    IMPORT_ERROR = exc


@unittest.skipIf(IMPORT_ERROR is not None, f"analysis deps unavailable: {IMPORT_ERROR}")
class AnalysisStrictTests(unittest.TestCase):
    def test_validate_uniform_protocol_raises_on_mixed_versions(self):
        df = pd.DataFrame(
            {
                "question_id": ["q1", "q2"],
                "model": ["m1", "m1"],
                "protocol_version": ["v2", "v3"],
            }
        )
        with self.assertRaises(ValueError):
            validate_uniform_protocol(df, require_uniform_protocol=True)

    def test_bootstrap_ci_is_deterministic_given_seed(self):
        values = pd.Series([1, 0, 1, 1, 0, 1], dtype=int).to_numpy()
        ci_1 = _bootstrap_proportion_ci(values, num_bootstrap=500, seed=123)
        ci_2 = _bootstrap_proportion_ci(values, num_bootstrap=500, seed=123)
        self.assertEqual(ci_1, ci_2)

    def test_paired_significance_generates_rows(self):
        df = pd.DataFrame(
            {
                "question_id": ["q1", "q2", "q3", "q1", "q2", "q3"],
                "model": ["A", "A", "A", "B", "B", "B"],
                "greedy_correct": [True, False, True, False, False, True],
                "error_label_0.9": [
                    "reliably_correct",
                    "inconsistent_error",
                    "reliably_correct",
                    "self_consistent_error",
                    "inconsistent_error",
                    "reliably_correct",
                ],
                "correctness_grade": [
                    "CORRECT",
                    "INCORRECT",
                    "CORRECT",
                    "INCORRECT",
                    "INCORRECT",
                    "CORRECT",
                ],
                "escalated_to_human": [False, True, False, True, True, False],
            }
        )

        out = table_10_paired_significance(df)
        self.assertFalse(out.empty)
        self.assertTrue((out["Metric"] == "accuracy").any())
        self.assertTrue((out["Metric"] == "self_consistent_error").any())


if __name__ == "__main__":
    unittest.main()
