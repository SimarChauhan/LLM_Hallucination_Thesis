import unittest

from scripts.reeval_results import (
    _build_precomputed_correctness_result,
    _build_skip_greedy_output_path,
    _ensure_skip_mode_non_overwrite,
)


class ReevalSkipGreedyModeTests(unittest.TestCase):
    def test_build_skip_greedy_output_path_for_jsonl(self):
        output = _build_skip_greedy_output_path("/tmp/results.jsonl")
        self.assertEqual(output, "/tmp/results.skip_greedy_semantic_eval.jsonl")

    def test_build_skip_greedy_output_path_without_suffix(self):
        output = _build_skip_greedy_output_path("/tmp/results")
        self.assertEqual(output, "/tmp/results.skip_greedy_semantic_eval.jsonl")

    def test_non_overwrite_guard_raises_when_same_path(self):
        with self.assertRaises(ValueError):
            _ensure_skip_mode_non_overwrite(
                input_path="/tmp/results.jsonl",
                output_path="/tmp/results.jsonl",
                skip_greedy_correctness=True,
            )

    def test_non_overwrite_guard_allows_distinct_path(self):
        _ensure_skip_mode_non_overwrite(
            input_path="/tmp/results.jsonl",
            output_path="/tmp/results.new.jsonl",
            skip_greedy_correctness=True,
        )

    def test_precomputed_correctness_uses_existing_grade_and_bool(self):
        rec = {
            "correctness_grade": "INCORRECT",
            "greedy_correct": False,
            "correctness_decision_source": "MAJORITY",
        }
        result, missing = _build_precomputed_correctness_result(rec)
        self.assertFalse(missing)
        self.assertEqual(result.grade, "INCORRECT")
        self.assertFalse(result.is_correct)
        self.assertEqual(result.decision_source, "MAJORITY")

    def test_precomputed_correctness_infers_grade_from_bool(self):
        rec = {
            "greedy_correct": True,
        }
        result, missing = _build_precomputed_correctness_result(rec)
        self.assertFalse(missing)
        self.assertEqual(result.grade, "CORRECT")
        self.assertTrue(result.is_correct)
        self.assertEqual(result.decision_source, "PRECOMPUTED")

    def test_precomputed_correctness_marks_missing_when_no_data(self):
        rec = {}
        result, missing = _build_precomputed_correctness_result(rec)
        self.assertTrue(missing)
        self.assertEqual(result.grade, "NOT_ATTEMPTED")
        self.assertFalse(result.is_correct)
        self.assertTrue(result.is_unclear)
        self.assertEqual(result.decision_source, "PRECOMPUTED_MISSING")


if __name__ == "__main__":
    unittest.main()
