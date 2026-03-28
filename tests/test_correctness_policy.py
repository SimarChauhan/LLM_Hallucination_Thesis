import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.correctness import (
    check_correctness,
    check_correctness_llm,
    check_correctness_llm_ensemble,
    check_correctness_string,
)
from src.schemas import CorrectnessResult


class _JudgeClient:
    def __init__(self, text: str):
        self._text = text
        self.last_prompt = None
        self.last_kwargs = None

    def generate_greedy(self, **kwargs):
        self.last_prompt = kwargs.get("prompt")
        self.last_kwargs = kwargs
        return SimpleNamespace(text=self._text)


class _FailingJudgeClient:
    def generate_greedy(self, **kwargs):
        raise RuntimeError("judge unavailable")


class CorrectnessPolicyTests(unittest.TestCase):
    def test_main_correctness_path_is_llm_only(self):
        # Main correctness path requires a 3-judge ensemble.
        with patch(
            "src.correctness.check_correctness_llm",
            side_effect=[
                CorrectnessResult(
                    is_correct=False,
                    grade="INCORRECT",
                    judge_reasoning=["Wrong."],
                ),
                CorrectnessResult(
                    is_correct=False,
                    grade="INCORRECT",
                    judge_reasoning=["Wrong."],
                ),
                CorrectnessResult(
                    is_correct=False,
                    grade="INCORRECT",
                    judge_reasoning=["Wrong."],
                ),
            ],
        ):
            result = check_correctness(
                prediction="Paris",
                ground_truths=["Paris"],
                question="What is the capital of France?",
                inference_client=object(),
                use_llm_fallback=True,
                llm_judge_ensemble=[
                    {"provider": "openai", "model": "judge-a"},
                    {"provider": "anthropic", "model": "judge-b"},
                    {"provider": "xai", "model": "judge-c"},
                ],
            )
        self.assertFalse(result.is_correct)
        self.assertEqual(result.grade, "INCORRECT")

    def test_main_correctness_requires_exactly_three_judges(self):
        client = _JudgeClient("Reasoning text.\nA")
        result = check_correctness(
            prediction="Paris",
            ground_truths=["Paris"],
            question="What is the capital of France?",
            inference_client=client,
            use_llm_fallback=True,
            llm_judge_ensemble=[
                {"provider": "openai", "model": "judge-a"},
                {"provider": "anthropic", "model": "judge-b"},
            ],
        )
        self.assertFalse(result.is_correct)
        self.assertTrue(result.is_unclear)
        self.assertEqual(result.grade, "NOT_ATTEMPTED")
        self.assertEqual(result.match_type, "no_correctness_judge")

    def test_main_correctness_without_judge_is_not_attempted(self):
        result = check_correctness(
            prediction="Paris",
            ground_truths=["Paris"],
            question="What is the capital of France?",
            inference_client=None,
            use_llm_fallback=False,
        )
        self.assertFalse(result.is_correct)
        self.assertTrue(result.is_unclear)
        self.assertEqual(result.grade, "NOT_ATTEMPTED")
        self.assertEqual(result.match_type, "no_correctness_judge")
        self.assertEqual(result.decision_source, "NO_JUDGE")

    def test_string_matching_is_exact_only(self):
        exact = check_correctness_string("Paris", ["Paris"])
        self.assertTrue(exact.is_correct)
        self.assertEqual(exact.match_type, "exact")

        # Previously this could be accepted via containment; now it must fail.
        non_exact = check_correctness_string("Paris or London", ["Paris"])
        self.assertFalse(non_exact.is_correct)
        self.assertIsNone(non_exact.match_type)

    def test_single_judge_incorrect_remains_incorrect(self):
        client = _JudgeClient("Maybe this is wrong.\nB")
        result = check_correctness_llm(
            prediction="Lyon",
            ground_truths=["Paris"],
            question="What is the capital of France?",
            inference_client=client,
            judge_provider="openai",
            judge_model="gpt-4o",
        )

        self.assertFalse(result.is_correct)
        self.assertFalse(result.is_unclear)
        self.assertEqual(result.grade, "INCORRECT")
        self.assertIsNone(result.match_type)

    def test_parser_accepts_grade_prefix_with_trailing_period(self):
        client = _JudgeClient("The answer matches the reference.\nGRADE: A.")
        result = check_correctness_llm(
            prediction="Paris",
            ground_truths=["Paris"],
            question="What is the capital of France?",
            inference_client=client,
            judge_provider="openai",
            judge_model="gpt-4o",
        )

        self.assertTrue(result.is_correct)
        self.assertEqual(result.grade, "CORRECT")

    def test_parser_accepts_strict_json_output(self):
        client = _JudgeClient('{"reasoning":"Matches the reference exactly.","grade":"A"}')
        result = check_correctness_llm(
            prediction="Paris",
            ground_truths=["Paris"],
            question="What is the capital of France?",
            inference_client=client,
            judge_provider="openai",
            judge_model="gpt-4o",
        )

        self.assertTrue(result.is_correct)
        self.assertEqual(result.grade, "CORRECT")

    def test_parser_accepts_embedded_json_after_free_text(self):
        client = _JudgeClient(
            'The answer aligns with one provided target.\n'
            '{"reasoning":"The predicted answer matches a valid alternative.","grade":"A"}'
        )
        result = check_correctness_llm(
            prediction="Paris",
            ground_truths=["Paris", "The capital is Paris"],
            question="What is the capital of France?",
            inference_client=client,
            judge_provider="anthropic",
            judge_model="claude-sonnet-4-5",
        )

        self.assertTrue(result.is_correct)
        self.assertEqual(result.grade, "CORRECT")

    def test_prompt_formats_multiple_gold_targets_as_alternatives(self):
        client = _JudgeClient("Reasoning text.\nA")
        _ = check_correctness_llm(
            prediction="San Francisco",
            ground_truths=["San Francisco", "SF"],
            question="Where did fortune cookies originate?",
            inference_client=client,
            judge_provider="openai",
            judge_model="gpt-4o",
        )

        self.assertIsNotNone(client.last_prompt)
        self.assertIn("Gold targets (each line is an alternative acceptable reference):", client.last_prompt)
        self.assertIn("- San Francisco", client.last_prompt)
        self.assertIn("- SF", client.last_prompt)
        self.assertIn("response_format", client.last_kwargs)
        self.assertIsInstance(client.last_kwargs["response_format"], dict)

    def test_malformed_output_does_not_use_first_character_as_grade(self):
        # No final grade line; starts with "Because..." (B) which used to be misread as INCORRECT.
        client = _JudgeClient("Because this answer seems plausible but incomplete.")
        result = check_correctness_llm(
            prediction="Lyon",
            ground_truths=["Paris"],
            question="What is the capital of France?",
            inference_client=client,
            judge_provider="openai",
            judge_model="gpt-4o",
        )

        self.assertFalse(result.is_correct)
        self.assertTrue(result.is_unclear)
        self.assertEqual(result.grade, "NOT_ATTEMPTED")
        self.assertEqual(result.match_type, "llm_judge_parse_failed")
        self.assertEqual(result.judge_statuses, ["PARSE_FAILED"])

    def test_ensemble_incorrect_remains_incorrect(self):
        incorrect = CorrectnessResult(
            is_correct=False,
            match_type=None,
            matched_answer=None,
            is_unclear=False,
            grade="INCORRECT",
            judge_reasoning=["Likely wrong."],
        )

        with patch("src.correctness.check_correctness_llm", side_effect=[incorrect, incorrect]):
            result = check_correctness_llm_ensemble(
                prediction="Lyon",
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
        self.assertFalse(result.is_unclear)
        self.assertEqual(result.grade, "INCORRECT")

    def test_ensemble_tie_returns_not_attempted(self):
        correct = CorrectnessResult(
            is_correct=True,
            match_type="llm_judge",
            matched_answer="Paris",
            is_unclear=False,
            grade="CORRECT",
            judge_reasoning=["Matches gold answer."],
        )
        incorrect = CorrectnessResult(
            is_correct=False,
            match_type=None,
            matched_answer=None,
            is_unclear=False,
            grade="INCORRECT",
            judge_reasoning=["Contradicts gold answer."],
        )
        unclear = CorrectnessResult(
            is_correct=False,
            match_type="llm_judge_not_attempted",
            matched_answer=None,
            is_unclear=True,
            grade="NOT_ATTEMPTED",
            judge_reasoning=["Insufficiently specific answer."],
        )

        with patch("src.correctness.check_correctness_llm", side_effect=[correct, incorrect, unclear]):
            result = check_correctness_llm_ensemble(
                prediction="Paris maybe",
                ground_truths=["Paris"],
                question="What is the capital of France?",
                inference_client=object(),
                judges=[
                    {"provider": "openai", "model": "judge-a"},
                    {"provider": "anthropic", "model": "judge-b"},
                    {"provider": "xai", "model": "judge-c"},
                ],
                failure_policy="skip",
            )

        self.assertFalse(result.is_correct)
        self.assertTrue(result.is_unclear)
        self.assertEqual(result.grade, "NOT_ATTEMPTED")
        self.assertEqual(result.decision_source, "UNRESOLVED")

    def test_ensemble_high_conf_incorrect_stays_incorrect(self):
        high_conf_incorrect = CorrectnessResult(
            is_correct=False,
            match_type=None,
            matched_answer=None,
            is_unclear=False,
            grade="INCORRECT",
            judge_reasoning=["Contradicts the gold answer."],
        )

        with patch("src.correctness.check_correctness_llm", side_effect=[high_conf_incorrect, high_conf_incorrect]):
            result = check_correctness_llm_ensemble(
                prediction="Lyon",
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
        self.assertFalse(result.is_unclear)
        self.assertEqual(result.grade, "INCORRECT")
        self.assertEqual(result.decision_source, "MAJORITY")

    def test_unresolved_ensemble_escalates_to_adjudicator(self):
        correct = CorrectnessResult(
            is_correct=True,
            match_type="llm_judge",
            matched_answer="Paris",
            is_unclear=False,
            grade="CORRECT",
            judge_reasoning=["Matches gold answer."],
        )
        incorrect = CorrectnessResult(
            is_correct=False,
            match_type=None,
            matched_answer=None,
            is_unclear=False,
            grade="INCORRECT",
            judge_reasoning=["Contradicts gold answer."],
        )
        unclear = CorrectnessResult(
            is_correct=False,
            match_type="llm_judge_not_attempted",
            matched_answer=None,
            is_unclear=True,
            grade="NOT_ATTEMPTED",
            judge_reasoning=["Insufficiently specific answer."],
        )

        with patch("src.correctness.check_correctness_llm", side_effect=[correct, incorrect, unclear]):
            result = check_correctness(
                prediction="Paris maybe",
                ground_truths=["Paris"],
                question="What is the capital of France?",
                inference_client=_JudgeClient("Panel tie. Adjudicator decides this is correct.\nA"),
                use_llm_fallback=True,
                llm_judge_ensemble=[
                    {"provider": "openai", "model": "judge-a"},
                    {"provider": "anthropic", "model": "judge-b"},
                    {"provider": "xai", "model": "judge-c"},
                ],
                adjudicator={"provider": "openai", "model": "judge-d"},
            )

        self.assertTrue(result.is_correct)
        self.assertEqual(result.grade, "CORRECT")
        self.assertEqual(result.decision_source, "ADJUDICATOR")
        self.assertEqual(result.adjudicator_status, "OK")
        self.assertEqual(result.adjudicator_grade, "CORRECT")

    def test_adjudicator_not_called_when_majority_exists(self):
        incorrect = CorrectnessResult(
            is_correct=False,
            match_type=None,
            matched_answer=None,
            is_unclear=False,
            grade="INCORRECT",
            judge_reasoning=["Wrong answer."],
        )
        unclear = CorrectnessResult(
            is_correct=False,
            match_type="llm_judge_not_attempted",
            matched_answer=None,
            is_unclear=True,
            grade="NOT_ATTEMPTED",
            judge_reasoning=["No attempt."],
        )

        with patch("src.correctness.check_correctness_llm", side_effect=[incorrect, incorrect, unclear]):
            with patch("src.correctness.check_correctness_llm_adjudicator") as adjudicator_mock:
                result = check_correctness(
                    prediction="Lyon",
                    ground_truths=["Paris"],
                    question="What is the capital of France?",
                    inference_client=_JudgeClient("Should not be used.\nA"),
                    use_llm_fallback=True,
                    llm_judge_ensemble=[
                        {"provider": "openai", "model": "judge-a"},
                        {"provider": "anthropic", "model": "judge-b"},
                        {"provider": "xai", "model": "judge-c"},
                    ],
                    adjudicator={"provider": "openai", "model": "judge-d"},
                )

        adjudicator_mock.assert_not_called()
        self.assertFalse(result.is_correct)
        self.assertEqual(result.grade, "INCORRECT")
        self.assertEqual(result.decision_source, "MAJORITY")

    def test_adjudicator_failure_keeps_unresolved(self):
        correct = CorrectnessResult(
            is_correct=True,
            match_type="llm_judge",
            matched_answer="Paris",
            is_unclear=False,
            grade="CORRECT",
            judge_reasoning=["Matches gold answer."],
        )
        incorrect = CorrectnessResult(
            is_correct=False,
            match_type=None,
            matched_answer=None,
            is_unclear=False,
            grade="INCORRECT",
            judge_reasoning=["Contradicts gold answer."],
        )
        unclear = CorrectnessResult(
            is_correct=False,
            match_type="llm_judge_not_attempted",
            matched_answer=None,
            is_unclear=True,
            grade="NOT_ATTEMPTED",
            judge_reasoning=["Insufficiently specific answer."],
        )

        with patch("src.correctness.check_correctness_llm", side_effect=[correct, incorrect, unclear]):
            result = check_correctness(
                prediction="Paris maybe",
                ground_truths=["Paris"],
                question="What is the capital of France?",
                inference_client=_FailingJudgeClient(),
                use_llm_fallback=True,
                llm_judge_ensemble=[
                    {"provider": "openai", "model": "judge-a"},
                    {"provider": "anthropic", "model": "judge-b"},
                    {"provider": "xai", "model": "judge-c"},
                ],
                adjudicator={"provider": "openai", "model": "judge-d"},
            )

        self.assertFalse(result.is_correct)
        self.assertTrue(result.is_unclear)
        self.assertEqual(result.grade, "NOT_ATTEMPTED")
        self.assertEqual(result.decision_source, "UNRESOLVED")
        self.assertEqual(result.adjudicator_status, "API_FAILED")


if __name__ == "__main__":
    unittest.main()
