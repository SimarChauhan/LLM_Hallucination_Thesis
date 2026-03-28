import math
import unittest

from src.semantic_entropy import compute_semantic_entropy, classify_error_by_entropy


class _SimpleNLIJudge:
    def judge_equivalence(self, question, answer_a, answer_b):
        class _Result:
            def __init__(self, judgment):
                self.judgment = judgment

        if answer_a[0] == answer_b[0]:
            return _Result("same")
        return _Result("different")


class _ChainNLIJudge:
    def judge_equivalence(self, question, answer_a, answer_b):
        class _Result:
            def __init__(self, judgment):
                self.judgment = judgment

        chain_same = {
            ("a1", "a2"),
            ("a2", "a3"),
        }
        if (answer_a, answer_b) in chain_same or (answer_b, answer_a) in chain_same:
            return _Result("same")
        return _Result("different")


class SemanticEntropyTests(unittest.TestCase):
    def test_entropy_zero_when_all_samples_equivalent(self):
        judge = _SimpleNLIJudge()
        samples = ["a1", "a2", "a3"]
        result = compute_semantic_entropy("q", samples, judge)

        self.assertEqual(result.n_clusters, 1)
        self.assertEqual(result.cluster_ids, [0, 0, 0])
        self.assertEqual(result.cluster_sizes, [3])
        self.assertAlmostEqual(result.entropy, 0.0, places=8)
        self.assertAlmostEqual(result.entropy_norm, 0.0, places=8)

    def test_entropy_for_two_even_clusters(self):
        judge = _SimpleNLIJudge()
        samples = ["a1", "a2", "b1", "b2"]
        result = compute_semantic_entropy("q", samples, judge)

        self.assertEqual(result.n_clusters, 2)
        self.assertEqual(result.cluster_ids, [0, 0, 1, 1])
        self.assertEqual(result.cluster_sizes, [2, 2])
        self.assertAlmostEqual(result.entropy, math.log(2.0), places=8)
        self.assertAlmostEqual(result.entropy_norm, 0.5, places=8)

    def test_union_find_handles_transitive_same_edges(self):
        judge = _ChainNLIJudge()
        samples = ["a1", "a2", "a3"]
        result = compute_semantic_entropy("q", samples, judge)

        self.assertEqual(result.n_clusters, 1)
        self.assertEqual(result.cluster_ids, [0, 0, 0])
        self.assertEqual(result.cluster_sizes, [3])

    def test_entropy_uses_precomputed_pair_judgments(self):
        samples = ["a1", "a2", "b1"]
        pair_judgments = {
            (0, 1): "same",
            (0, 2): "different",
            (1, 2): "different",
        }
        result = compute_semantic_entropy(
            question="q",
            sample_answers=samples,
            pair_judgments=pair_judgments,
        )

        self.assertEqual(result.n_clusters, 2)
        self.assertEqual(result.cluster_ids, [0, 0, 1])
        self.assertEqual(result.cluster_sizes, [2, 1])

    def test_entropy_labeling(self):
        label_low = classify_error_by_entropy(
            is_correct=False,
            entropy=0.1,
            entropy_norm=0.1,
            entropy_threshold=0.35,
            use_normalized_entropy=True,
            grade="INCORRECT",
        )
        label_high = classify_error_by_entropy(
            is_correct=False,
            entropy=1.0,
            entropy_norm=0.8,
            entropy_threshold=0.35,
            use_normalized_entropy=True,
            grade="INCORRECT",
        )
        label_not_attempted = classify_error_by_entropy(
            is_correct=False,
            entropy=0.0,
            entropy_norm=0.0,
            entropy_threshold=0.35,
            use_normalized_entropy=True,
            grade="NOT_ATTEMPTED",
        )

        self.assertEqual(label_low, "self_consistent_error")
        self.assertEqual(label_high, "inconsistent_error")
        self.assertEqual(label_not_attempted, "not_attempted")


if __name__ == "__main__":
    unittest.main()
