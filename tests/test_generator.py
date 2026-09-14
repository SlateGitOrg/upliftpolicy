"""The planted ground truth must actually be planted."""

import math
import unittest

from src.evaluation import pearson_correlation
from src.generator import (PROPENSITY_CEIL, PROPENSITY_FLOOR, SEGMENTS,
                           SEGMENT_OUTCOMES, SEGMENT_SHARES,
                           generate_population, segment_counts)
from src.policy import top_k_policy, true_policy_value

N = 8000


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pop = generate_population(N, seed=5)

    def test_deterministic(self):
        a = generate_population(50, seed=3)
        b = generate_population(50, seed=3)
        self.assertEqual([c.features for c in a], [c.features for c in b])

    def test_segment_shares_match_plan(self):
        counts = segment_counts(self.pop)
        for s in SEGMENTS:
            p = SEGMENT_SHARES[s]
            se = math.sqrt(p * (1 - p) / N)
            # 4 binomial standard errors
            self.assertLess(abs(counts[s] / N - p), 4 * se, s)

    def test_sleeping_dog_segment_has_negative_uplift_and_high_churn(self):
        p0_sd, p1_sd = SEGMENT_OUTCOMES["sleeping_dog"]
        self.assertLess(p0_sd - p1_sd, 0)
        # the property that makes a churn model rank them above persuadables
        self.assertGreater(p0_sd, SEGMENT_OUTCOMES["persuadable"][0])

    def test_logged_propensity_is_confounded_and_has_overlap(self):
        e = [c.propensity for c in self.pop]
        self.assertGreaterEqual(min(e), PROPENSITY_FLOOR)
        self.assertLessEqual(max(e), PROPENSITY_CEIL)
        risk = [c.features[0] + c.features[1] for c in self.pop]
        self.assertGreater(pearson_correlation(risk, [c.treated for c in self.pop]), 0.2)

    def test_randomised_mode(self):
        pop = generate_population(4000, seed=2, propensity_mode="randomised")
        rate = sum(c.treated for c in pop) / len(pop)
        self.assertLess(abs(rate - 0.5), 4 * math.sqrt(0.25 / 4000))

    def test_outcomes_follow_planted_probabilities(self):
        # empirical churn among untreated sleeping dogs vs planted p0
        sub = [c for c in self.pop if c.segment == "sleeping_dog" and not c.treated]
        p = SEGMENT_OUTCOMES["sleeping_dog"][0]
        rate = sum(c.churned for c in sub) / len(sub)
        self.assertLess(abs(rate - p), 4 * math.sqrt(p * (1 - p) / len(sub)))

    def test_oracle_uplift_ranking_beats_oracle_churn_ranking(self):
        """Even with PERFECT knowledge of churn risk, ranking by it loses."""
        n = len(self.pop)
        base = true_policy_value(self.pop, [0] * n)
        by_uplift = top_k_policy([c.true_uplift for c in self.pop], 0.3)
        by_churn = top_k_policy([c.p0 for c in self.pop], 0.3)
        gain_u = true_policy_value(self.pop, by_uplift) - base
        gain_c = true_policy_value(self.pop, by_churn) - base
        self.assertGreater(gain_u, 0.05)
        self.assertLess(gain_c, 0.0)


if __name__ == "__main__":
    unittest.main()
