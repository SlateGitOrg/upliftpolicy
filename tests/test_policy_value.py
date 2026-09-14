"""Doubly-robust off-policy value recovers the TRUE policy value.

Tolerances are k standard errors of the DR score mean, SE = sd(scores)/sqrt(n),
computed from the data in the test, never a hard-coded number.
"""

import math
import unittest

from src.evaluation import (bootstrap_ci, dm_value, dr_scores, dr_value,
                            ipw_value, uplift_curve, area_under_curve)
from src.policy import top_k_policy, true_policy_value
from src.study import build_study
from src.uplift import (ConstantPropensity, CrossFitNuisances, OutcomeModel,
                        TrueOutcome, TruePropensity)

SEED = 104


def se_of_mean(xs):
    n = len(xs)
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


class TestPolicyValue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st = build_study(n=3000, seed=SEED)
        cls.ev = cls.st.evaluate
        cls.acts = top_k_policy(cls.st.uplift_scores, 0.3)
        cls.truth = true_policy_value(cls.ev, cls.acts)
        cls.scores = dr_scores(cls.ev, cls.acts, cls.st.nuisances, cls.st.propensities)
        cls.se = se_of_mean(cls.scores)
        cls.bad_outcome = CrossFitNuisances(
            outcome_factory=lambda: OutcomeModel(feature_index=[]), seed=1).fit(cls.ev)
        cls.const_e = ConstantPropensity().predict(cls.ev)

    def test_dr_recovers_true_value_with_learned_nuisances(self):
        err = dr_value(self.ev, self.acts, self.st.nuisances, self.st.propensities) - self.truth
        self.assertLess(abs(err), 3 * self.se)

    def test_dr_with_oracle_nuisances(self):
        v = dr_value(self.ev, self.acts, TrueOutcome(), TruePropensity().predict(self.ev))
        self.assertLess(abs(v - self.truth), 3 * self.se)

    def test_bootstrap_ci_covers_truth(self):
        _, lo, hi = bootstrap_ci(self.scores, n_boot=200, seed=1)
        self.assertLessEqual(lo, self.truth)
        self.assertGreaterEqual(hi, self.truth)

    def test_ipw_with_naive_propensity_is_badly_biased(self):
        """Treating a confounded log as randomised: the common generic mistake."""
        err = ipw_value(self.ev, self.acts, self.const_e) - self.truth
        self.assertGreater(abs(err), 10 * self.se)

    def test_dr_survives_misspecified_propensity(self):
        err = dr_value(self.ev, self.acts, self.st.nuisances, self.const_e) - self.truth
        self.assertLess(abs(err), 3 * self.se)

    def test_dr_survives_misspecified_outcome_model(self):
        err = dr_value(self.ev, self.acts, self.bad_outcome, self.st.propensities) - self.truth
        self.assertLess(abs(err), 3 * self.se)

    def test_dr_ranks_the_two_campaigns_correctly(self):
        churn_acts = top_k_policy(self.st.churn_scores, 0.3)
        nu, e = self.st.nuisances, self.st.propensities
        self.assertGreater(dr_value(self.ev, self.acts, nu, e),
                           dr_value(self.ev, churn_acts, nu, e))
        n = len(self.ev)
        base = true_policy_value(self.ev, [0] * n)
        gain_u = self.truth - base
        gain_c = true_policy_value(self.ev, churn_acts) - base
        # measured over 10 seeds: gain_u 0.035..0.048, gain_c -0.013..0.000
        self.assertGreater(gain_u, 0.03)
        self.assertLessEqual(gain_c, 0.005)

    def test_qini_area_uplift_beats_churn(self):
        nu, e = self.st.nuisances, self.st.propensities
        au = area_under_curve(uplift_curve(self.ev, self.st.uplift_scores, nu, e))
        ac = area_under_curve(uplift_curve(self.ev, self.st.churn_scores, nu, e))
        self.assertGreater(au, ac)
        # endpoints: 0% targeted = 0 incremental; 100% identical for any ranking
        cu = uplift_curve(self.ev, self.st.uplift_scores, nu, e)
        cc = uplift_curve(self.ev, self.st.churn_scores, nu, e)
        self.assertEqual(cu[0][1], 0.0)
        self.assertAlmostEqual(cu[-1][1], cc[-1][1], places=12)


class TestBiasAcrossSeeds(unittest.TestCase):
    """Bias claims need more than one log.

    A first version asserted "DR with both nuisances wrong is off by > 3 SE" on
    a single seed.  It failed: at n=1500 one seed's SE is ~0.017, so a real
    bias of ~0.05 sits at 2.9 SE.  The claim was about BIAS, so it is now
    tested as a mean error over independent logs with the between-seed SE.
    """

    SEEDS = range(200, 206)

    @classmethod
    def setUpClass(cls):
        cls.err = {"dr": [], "dr_both_bad": [], "dm_bad": []}
        for s in cls.SEEDS:
            st = build_study(n=3000, seed=s)
            ev = st.evaluate
            acts = top_k_policy(st.uplift_scores, 0.3)
            truth = true_policy_value(ev, acts)
            bad_m = CrossFitNuisances(
                outcome_factory=lambda: OutcomeModel(feature_index=[]), seed=s).fit(ev)
            const_e = ConstantPropensity().predict(ev)
            cls.err["dr"].append(dr_value(ev, acts, st.nuisances, st.propensities) - truth)
            cls.err["dr_both_bad"].append(dr_value(ev, acts, bad_m, const_e) - truth)
            cls.err["dm_bad"].append(dm_value(ev, acts, bad_m) - truth)

    def mean_se(self, key):
        xs = self.err[key]
        return sum(xs) / len(xs), se_of_mean(xs)

    def test_dr_unbiased_with_sound_nuisances(self):
        m, se = self.mean_se("dr")
        self.assertLess(abs(m), 3 * se + 1e-9)

    def test_dr_biased_when_both_misspecified(self):
        m, se = self.mean_se("dr_both_bad")
        self.assertGreater(abs(m), 3 * se)

    # A "DM with an intercept-only outcome model is biased" test was removed:
    # over these 6 seeds its mean error was -0.0087 with SE 0.0045 (1.9 SE),
    # so that bias is not resolvable at this sample size. README says so.


if __name__ == "__main__":
    unittest.main()
