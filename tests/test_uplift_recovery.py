"""Uplift recovery against the planted ITE, and the sleeping-dog segment.

Thresholds come from a 10-seed sweep at n=3000 (1500 train / 1500 evaluate):
  r(T-learner uplift, true ITE)  ranged 0.40 .. 0.67
  r(churn score,     true ITE)  ranged -0.29 .. -0.14
  sleeping dogs in top-30% list: churn-ranked 160..216, uplift-ranked 16..91
"""

import unittest

from src.evaluation import pearson_correlation
from src.policy import (apply_sleeping_dog_guardrail, sleeping_dog_harm,
                        targeted_segment_mix, top_k_policy)
from src.study import build_study

SEED = 104  # a fixed seed from inside the sweep, not a hand-picked best case


class TestUpliftRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st = build_study(n=3000, seed=SEED)
        cls.ev = cls.st.evaluate
        cls.true_u = [c.true_uplift for c in cls.ev]

    def seg_mean(self, scores, seg):
        idx = [i for i, c in enumerate(self.ev) if c.segment == seg]
        return sum(scores[i] for i in idx) / len(idx)

    def test_tlearner_correlates_with_true_ite(self):
        # 0.35 sits below the worst of 10 seeds (0.40): a working learner
        # clears it, a sign-flipped or constant one cannot.
        self.assertGreater(pearson_correlation(self.st.uplift_scores, self.true_u), 0.35)

    def test_transformed_outcome_is_independent_second_estimate(self):
        self.assertGreater(pearson_correlation(self.st.transformed_scores, self.true_u), 0.35)

    def test_churn_score_does_not_track_uplift(self):
        """The generic approach: churn risk is NEGATIVELY related to uplift here."""
        self.assertLess(pearson_correlation(self.st.churn_scores, self.true_u), 0.0)

    def test_sign_flipped_learner_fails_the_same_bar(self):
        flipped = [-u for u in self.st.uplift_scores]
        self.assertLess(pearson_correlation(flipped, self.true_u), 0.35)

    def test_sleeping_dog_segment_identified_as_negative(self):
        est = self.st.uplift_scores
        sd = self.seg_mean(est, "sleeping_dog")
        self.assertLess(sd, 0.0)
        for other in ("persuadable", "sure_thing", "lost_cause"):
            self.assertLess(sd, self.seg_mean(est, other), other)

    def test_churn_model_cannot_represent_negative_effect(self):
        self.assertTrue(all(0.0 <= s <= 1.0 for s in self.st.churn_scores))
        # and it scores sleeping dogs as MORE urgent than persuadables
        self.assertGreater(self.seg_mean(self.st.churn_scores, "sleeping_dog"),
                           self.seg_mean(self.st.churn_scores, "persuadable"))

    def test_uplift_list_contacts_far_fewer_sleeping_dogs(self):
        mix_c, _ = targeted_segment_mix(self.ev, top_k_policy(self.st.churn_scores, 0.3))
        mix_u, _ = targeted_segment_mix(self.ev, top_k_policy(self.st.uplift_scores, 0.3))
        # worst seed in the sweep was 184 vs 91 (2.0x)
        self.assertGreaterEqual(mix_c["sleeping_dog"], 2 * mix_u["sleeping_dog"])

    def test_guardrail_reduces_harm_of_churn_campaign(self):
        churn_list = top_k_policy(self.st.churn_scores, 0.3)
        guarded = apply_sleeping_dog_guardrail(churn_list, self.st.uplift_scores)
        self.assertLess(sleeping_dog_harm(self.ev, guarded),
                        0.5 * sleeping_dog_harm(self.ev, churn_list))
        self.assertTrue(all(self.st.uplift_scores[i] > 0
                            for i, a in enumerate(guarded) if a))


if __name__ == "__main__":
    unittest.main()
