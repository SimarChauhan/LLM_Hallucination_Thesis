import unittest

from src.hybrid_judging import (
    compute_stochastic_correctness_metrics,
    decide_equivalence_hybrid,
    grade_sample_correctness_hybrid,
)


class _FakeNLIJudge:
    def __init__(self, mapping):
        self.mapping = mapping

    def _get_entailment_prob(self, premise, hypothesis):
        key = (premise, hypothesis)
        if key not in self.mapping:
            raise KeyError(f"Missing NLI score for {key}")
        return self.mapping[key]


class _FakeGeneration:
    def __init__(self, text):
        self.text = text


class _FakeInferenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_greedy(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise RuntimeError("No fake responses left")
        return _FakeGeneration(self.responses.pop(0))


class HybridJudgingTests(unittest.TestCase):
    def test_equivalence_nli_pass_through_without_llm(self):
        question = "Q"
        a = "A"
        b = "B"
        c_a = f"Question: {question} Answer: {a}"
        c_b = f"Question: {question} Answer: {b}"
        nli = _FakeNLIJudge({(c_a, c_b): 0.91, (c_b, c_a): 0.88})
        client = _FakeInferenceClient([])

        result = decide_equivalence_hybrid(
            question=question,
            answer_a=a,
            answer_b=b,
            nli_judge=nli,
            eq_same_hi=0.70,
            eq_diff_lo=0.30,
            inference_client=client,
            judge_provider="openai",
            judge_model="gpt-5.2",
        )

        self.assertEqual(result.label, "same")
        self.assertEqual(result.source, "NLI")
        self.assertEqual(len(client.calls), 0)

    def test_equivalence_borderline_calls_llm_twice(self):
        question = "Q"
        a = "A"
        b = "B"
        c_a = f"Question: {question} Answer: {a}"
        c_b = f"Question: {question} Answer: {b}"
        nli = _FakeNLIJudge({(c_a, c_b): 0.55, (c_b, c_a): 0.56})
        client = _FakeInferenceClient([
            '{"label":"different","reasoning":"r1"}',
            '{"label":"different","reasoning":"r2"}',
        ])

        result = decide_equivalence_hybrid(
            question=question,
            answer_a=a,
            answer_b=b,
            nli_judge=nli,
            eq_same_hi=0.70,
            eq_diff_lo=0.30,
            inference_client=client,
            judge_provider="openai",
            judge_model="gpt-5.2",
        )

        self.assertEqual(result.label, "different")
        self.assertEqual(result.source, "LLM")
        self.assertEqual(len(client.calls), 2)

    def test_equivalence_borderline_disagreement_returns_unclear(self):
        question = "Q"
        a = "A"
        b = "B"
        c_a = f"Question: {question} Answer: {a}"
        c_b = f"Question: {question} Answer: {b}"
        nli = _FakeNLIJudge({(c_a, c_b): 0.55, (c_b, c_a): 0.56})
        client = _FakeInferenceClient([
            '{"label":"same","reasoning":"r1"}',
            '{"label":"different","reasoning":"r2"}',
        ])

        result = decide_equivalence_hybrid(
            question=question,
            answer_a=a,
            answer_b=b,
            nli_judge=nli,
            eq_same_hi=0.70,
            eq_diff_lo=0.30,
            inference_client=client,
            judge_provider="openai",
            judge_model="gpt-5.2",
        )

        self.assertEqual(result.label, "unclear")
        self.assertEqual(result.source, "LLM")
        self.assertEqual(len(client.calls), 2)

    def test_correctness_or_over_gold_uses_best_sample_to_gold_score(self):
        question = "Q"
        sample = "S"
        gold_1 = "G1"
        gold_2 = "G2"
        c_sample = f"Question: {question} Answer: {sample}"
        nli = _FakeNLIJudge({
            (c_sample, f"Question: {question} Answer: {gold_1}"): 0.20,
            (c_sample, f"Question: {question} Answer: {gold_2}"): 0.81,
        })
        client = _FakeInferenceClient([])

        result = grade_sample_correctness_hybrid(
            question=question,
            sample_answer=sample,
            ground_truths=[gold_1, gold_2],
            nli_judge=nli,
            corr_hi=0.70,
            corr_lo=0.30,
            inference_client=client,
            judge_provider="openai",
            judge_model="gpt-5.2",
        )

        self.assertEqual(result.grade, "CORRECT")
        self.assertEqual(result.source, "NLI")
        self.assertEqual(result.matched_gold_index, 1)
        self.assertEqual(len(client.calls), 0)

    def test_correctness_borderline_calls_llm_once(self):
        question = "Q"
        sample = "S"
        gold = "G"
        c_sample = f"Question: {question} Answer: {sample}"
        nli = _FakeNLIJudge({
            (c_sample, f"Question: {question} Answer: {gold}"): 0.52,
        })
        client = _FakeInferenceClient([
            '{"reasoning":"Looks right.","grade":"A"}',
        ])

        result = grade_sample_correctness_hybrid(
            question=question,
            sample_answer=sample,
            ground_truths=[gold],
            nli_judge=nli,
            corr_hi=0.70,
            corr_lo=0.30,
            inference_client=client,
            judge_provider="openai",
            judge_model="gpt-5.2",
        )

        self.assertEqual(result.grade, "CORRECT")
        self.assertEqual(result.source, "LLM")
        self.assertEqual(len(client.calls), 1)

    def test_stochastic_metrics_handle_not_attempted_denominators(self):
        metrics = compute_stochastic_correctness_metrics(
            equivalence_results=["same", "different", "different", "different"],
            sample_grades=["CORRECT", "NOT_ATTEMPTED", "INCORRECT", "CORRECT"],
        )

        self.assertAlmostEqual(metrics["stochastic_correct_rate"], 2 / 3)
        self.assertEqual(metrics["stochastic_scored_n"], 3)
        self.assertEqual(metrics["stochastic_not_attempted_n"], 1)
        self.assertEqual(metrics["different_scored_n"], 2)
        self.assertEqual(metrics["different_correct_n"], 1)
        self.assertAlmostEqual(metrics["p_correct_given_different"], 0.5)

    def test_stochastic_metrics_return_none_when_no_scored_samples(self):
        metrics = compute_stochastic_correctness_metrics(
            equivalence_results=["different", "different"],
            sample_grades=["NOT_ATTEMPTED", "NOT_ATTEMPTED"],
        )

        self.assertIsNone(metrics["stochastic_correct_rate"])
        self.assertEqual(metrics["stochastic_scored_n"], 0)
        self.assertEqual(metrics["stochastic_not_attempted_n"], 2)
        self.assertIsNone(metrics["p_correct_given_different"])
        self.assertEqual(metrics["different_scored_n"], 0)


if __name__ == "__main__":
    unittest.main()
