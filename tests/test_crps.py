"""加权 CRPS 核：数值、导数、重复值与退化情形。"""

import math
import unittest

from snowcrps.crps import weighted_crps, weighted_crps_naive


class WeightedCrpsTests(unittest.TestCase):
    def test_matches_naive_reference(self):
        cases = [
            ([0.4, 0.4, 0.9], [0.3, 0.3, 0.4], 0.4),
            ([0.1, 0.2, 0.1, 0.4], [0.25, 0.25, 0.25, 0.25], 0.2),
            ([0.0, 0.0, 0.0, 1.0], [0.7, 0.1, 0.1, 0.1], 0.5),
            ([0.2, 0.2], [0.6, 0.4], 0.2),
        ]
        for values, weights, observation in cases:
            with self.subTest(values=values):
                fast = weighted_crps(values, weights, observation)
                slow = weighted_crps_naive(values, weights, observation)
                self.assertAlmostEqual(fast.loss, slow.loss, places=12)
                self.assertAlmostEqual(fast.observation_term, slow.observation_term, places=12)
                self.assertAlmostEqual(fast.pairwise_term, slow.pairwise_term, places=12)
                for got, want in zip(fast.gradients, slow.gradients):
                    self.assertAlmostEqual(got, want, places=12)

    def test_crps_decomposition(self):
        result = weighted_crps([0.1, 0.4, 0.9], [0.3, 0.3, 0.4], 0.4)
        self.assertAlmostEqual(result.observation_term, 0.29, places=12)
        self.assertAlmostEqual(result.pairwise_normalizer, 0.66, places=12)
        self.assertAlmostEqual(result.pairwise_term, 0.366 / 0.66, places=12)
        self.assertAlmostEqual(
            result.loss, result.observation_term - 0.5 * result.pairwise_term, places=12
        )

    def test_equal_values_keep_individual_signals(self):
        """相同预报值的成员不能去重：各自仍拿回自己的导数。"""
        result = weighted_crps([0.4, 0.4, 0.4, 0.9], [0.3, 0.3, 0.4, 0.2], 0.4)
        self.assertEqual(result.distinct_values, 2)
        self.assertEqual(len(result.gradients), 4)
        # 三个同值成员权重不同，导数应随各自权重按比例变化。
        g0, g1, g2, _ = result.gradients
        self.assertAlmostEqual(g0 / 0.3, g1 / 0.3, places=12)
        self.assertAlmostEqual(g2 / 0.4, g1 / 0.3, places=12)
        self.assertNotEqual(result.gradients[0], result.gradients[3])

    def test_observation_equal_to_forecast_has_zero_abs_derivative(self):
        """观测恰等于预报时，绝对值项导数为零，只剩成对项贡献。"""
        result = weighted_crps([0.4, 0.4, 0.9], [0.3, 0.3, 0.4], 0.4)
        # 对同值成员：obs_sign=0，sgn(x-x')=0，另一成员在上方。
        for gradient, weight in zip(result.gradients, [0.3, 0.3, 0.4]):
            self.assertGreater(gradient, 0.0)
        # y = x_i 时，若只有两个成员且都在观测同侧，导数必须严格来自成对项。
        pair = weighted_crps([0.4, 0.4], [0.5, 0.5], 0.4)
        self.assertEqual(pair.observation_term, 0.0)
        self.assertEqual(pair.loss, 0.0)
        self.assertEqual(pair.pairwise_term, 0.0)
        self.assertEqual(pair.gradients, (0.0, 0.0))

    def test_symmetric_zero_difference_gradients(self):
        """两成员同值时差为零，双方成对贡献对称为零。"""
        result = weighted_crps([0.5, 0.5], [0.5, 0.5], 0.9)
        # S = 0，CRPS = |x - y|；两成员导数都等于各自权重的符号项。
        self.assertAlmostEqual(result.loss, 0.4, places=12)
        self.assertEqual(result.gradients, (-0.5, -0.5))

    def test_gradients_match_finite_differences(self):
        values = [0.2, 0.4, 0.4, 0.9, 0.1, 0.55]
        weights = [1.5, 1.0, 2.0, 0.5, 0.25, 0.75]
        observation = 0.4
        result = weighted_crps(values, weights, observation)
        step = 1e-6
        for index in range(len(values)):
            up = list(values)
            down = list(values)
            up[index] += step
            down[index] -= step
            numeric = (
                weighted_crps(up, weights, observation).loss
                - weighted_crps(down, weights, observation).loss
            ) / (2 * step)
            self.assertAlmostEqual(numeric, result.gradients[index], places=6)

    def test_gradient_sum_matches_observation_term(self):
        """成对项对全体成员导数之和为零，故总和等于观测项导数的总和。"""
        values = [0.1, 0.3, 0.3, 0.8]
        weights = [1.0, 1.0, 1.0, 1.0]
        result = weighted_crps(values, weights, 0.4)
        # 三个值在观测下方（符号 -1）、一个在上方（+1），等权各 1/4。
        self.assertAlmostEqual(sum(result.gradients), -0.5, places=12)
        # 若观测在支持中间，两侧符号相消。
        straddle = weighted_crps([0.1, 0.8], [0.5, 0.5], 0.4)
        self.assertAlmostEqual(sum(straddle.gradients), 0.0, places=12)

    def test_many_equal_members_do_not_change_pairwise_normalizer(self):
        """N 个同值成员加一个异值成员：权重仍按 N+1 均分，不假设只有两组。"""
        values = [0.4] * 5000 + [0.9]
        weights = [1.0] * 5001
        result = weighted_crps(values, weights, 0.4)
        self.assertEqual(result.distinct_values, 2)
        self.assertEqual(len(result.gradients), 5001)
        n = 5001
        expected = 1.0 - (n - 1) * (1.0 / n) ** 2 - (1.0 / n) ** 2
        self.assertAlmostEqual(result.pairwise_normalizer, expected, places=12)

    def test_insufficient_support_raises(self):
        with self.assertRaises(ValueError):
            weighted_crps([0.4], [1.0], 0.4)
        with self.assertRaises(ValueError):
            weighted_crps([], [], 0.4)

    def test_zero_weight_members_do_not_define_pairwise(self):
        """只有一个正权成员时成对项无定义。"""
        with self.assertRaises(ValueError):
            weighted_crps([0.4, 0.9], [1.0, 0.0], 0.4)

    def test_large_duplicate_set_is_fast_and_exact(self):
        """大批重复值不构造成员对矩阵，结果与显式循环一致。"""
        values = [0.4] * 500 + [0.9]
        weights = [1.0] * 501
        result = weighted_crps(values, weights, 0.4)
        self.assertEqual(result.distinct_values, 2)
        self.assertEqual(len(result.gradients), 501)
        naive = weighted_crps_naive(values, weights, 0.4)
        self.assertAlmostEqual(result.loss, naive.loss, places=12)


if __name__ == "__main__":
    unittest.main()
