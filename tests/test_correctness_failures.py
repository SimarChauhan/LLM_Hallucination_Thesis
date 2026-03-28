import unittest
from unittest.mock import patch

from src.correctness import check_correctness_llm, check_correctness_llm_ensemble
from src.schemas import CorrectnessResult


class _FailingClient:
    def generate_greedy(self, **kwargs):
        raise RuntimeError("judge backend unavailable")


class CorrectnessFailureTests(unittest.TestCase):
    def test_llm_judge_exception_returns_not_attempted(self):
        result = check_correctness_llm(
            prediction="The answer is Paris",
            ground_truths=["Paris"],
            question="What is the capital of France?",
            inference_client=_FailingClient(),
            judge_provider="openai",
            judge_model="gpt-4o",
        )

        self.assertFalse(result.is_correct)
        self.assertEqual(result.match_type, "llm_judge_failed")
        self.assertTrue(result.is_unclear)
        self.assertEqual(result.grade, "NOT_ATTEMPTED")
        self.assertEqual(result.judge_statuses, ["API_FAILED"])
        self.assertIsInstance(result.judge_reasoning, list)
        self.assertTrue(result.judge_reasoning)

    def test_ensemble_skip_policy_with_single_valid_vote_is_unresolved(self):
        failed = CorrectnessResult(
            is_correct=False,
            match_type="llm_judge_failed",
            is_unclear=True,
            grade="NOT_ATTEMPTED",
            judge_reasoning=["Judge failed"],
        )
        success = CorrectnessResult(
            is_correct=True,
            match_type="llm_judge",
            matched_answer="Paris",
            is_unclear=False,
            grade="CORRECT",
            judge_reasoning=["Correct"],
        )

        with patch("src.correctness.check_correctness_llm", side_effect=[failed, success]):
            result = check_correctness_llm_ensemble(
                prediction="Paris",
                ground_truths=["Paris"],
                question="What is the capital of France?",
                inference_client=object(),
                judges=[
                    {"provider": "openai", "model": "judge-a"},
                    {"provider": "anthropic", "model": "judge-b"},
                ],
                failure_policy="skip",
            )

        self.assertFalse(result.is_correct)
        self.assertTrue(result.is_unclear)
        self.assertEqual(result.grade, "NOT_ATTEMPTED")
        self.assertEqual(result.match_type, "llm_judge_ensemble")
        self.assertEqual(result.decision_source, "UNRESOLVED")
        self.assertIsNone(result.judge_grades[0])
        self.assertEqual(result.judge_grades[1], "CORRECT")
        self.assertEqual(result.judge_statuses[0], "API_FAILED")
        self.assertEqual(result.judge_statuses[1], "OK")
        # Only one successful vote should be counted.
        self.assertEqual(len(result.judge_votes), 1)


if __name__ == "__main__":
    unittest.main()
