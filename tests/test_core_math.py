"""Linear algebra, GLM fitting, and the small closed-form helpers."""

import math
import random
import unittest

from src.evaluation import area_under_curve, pearson_correlation
from src.linalg import LogisticRegression, sigmoid, solve_symmetric
from src.policy import (apply_sleeping_dog_guardrail, top_k_policy,
                        true_policy_value)
from src.generator import Customer


class TestLinalg(unittest.TestCase):
    def test_solve_symmetric_known_system(self):
        a = [[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]]
        x_true = [1.0, -2.0, 0.5]
        b = [sum(a[i][j] * x_true[j] for j in range(3)) for i in range(3)]
        x = solve_symmetric(a, b)
        for got, want in zip(x, x_true):
            self.assertAlmostEqual(got, want, places=10)

    def test_singular_system_raises(self):
        with self.assertRaises(ValueError):
            solve_symmetric([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])

    def test_sigmoid_does_not_overflow(self):
        self.assertAlmostEqual(sigmoid(1000.0), 1.0)
        self.assertAlmostEqual(sigmoid(-1000.0), 0.0)
        self.assertAlmostEqual(sigmoid(0.0), 0.5)

    def test_logistic_recovers_planted_coefficients(self):
        rng = random.Random(1)
        beta = [-0.5, 1.2, -0.8]
        x, y = [], []
        for _ in range(4000):
            row = [rng.gauss(0, 1), rng.gauss(0, 1)]
            p = sigmoid(beta[0] + beta[1] * row[0] + beta[2] * row[1])
            x.append(row)
            y.append(1 if rng.random() < p else 0)
        m = LogisticRegression().fit(x, y)
        self.assertTrue(m.converged)
        # Asymptotic SE of each coefficient at n=4000 with unit-variance
        # covariates is roughly 2/sqrt(n) ~ 0.032; 0.15 is ~5 SE, which a
        # correct fit passes and a sign/intercept bug fails by a mile.
        for got, want in zip(m.coef, beta):
            self.assertLess(abs(got - want), 0.15)


class TestSmallHelpers(unittest.TestCase):
    def test_pearson_hand_values(self):
        self.assertAlmostEqual(pearson_correlation([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(pearson_correlation([1, 2, 3], [3, 2, 1]), -1.0)

    def test_area_under_curve_trapezoid(self):
        pts = [(0.0, 0.0), (0.5, 1.0), (1.0, 1.0)]
        self.assertAlmostEqual(area_under_curve(pts), 0.25 + 0.5)

    def test_true_policy_value_closed_form(self):
        pop = [Customer((), 0, 0, 0.5, "persuadable", 0.5, 0.25),
               Customer((), 0, 0, 0.5, "sleeping_dog", 0.55, 0.75)]
        # contact first only: retention (1-0.25 + 1-0.55)/2
        self.assertAlmostEqual(true_policy_value(pop, [1, 0]), (0.75 + 0.45) / 2)
        self.assertAlmostEqual(true_policy_value(pop, [0, 1]), (0.5 + 0.25) / 2)

    def test_top_k_counts_and_ties(self):
        a = top_k_policy([0.1, 0.9, 0.9, 0.3], 0.5)
        self.assertEqual(a, [0, 1, 1, 0])
        self.assertEqual(sum(top_k_policy([0.0] * 10, 0.3)), 3)

    def test_guardrail_blocks_negative_uplift_only(self):
        acts = [1, 1, 1, 0]
        up = [0.2, -0.01, 0.0, -0.5]
        self.assertEqual(apply_sleeping_dog_guardrail(acts, up), [1, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
