"""The standard evaluation protocol, in one place so the demo and the tests
cannot drift apart.

Two design decisions are baked in here and both were forced by measurement, not
by taste.

1. TRAIN/EVALUATE SPLIT.  The uplift model is fitted on one half of the log and
   the policy it implies is evaluated on the other half.  Fitting and evaluating
   on the same half inflated the doubly-robust estimate by a measured +0.8
   retention points over 30 seeds -- a winner's curse, not a coding error: the
   policy preferentially selects customers whose *realised* outcomes happened to
   look good, and the DR correction term is built from those same realised
   outcomes.  That is about one sixth of the entire churn-vs-uplift decision
   margin, so it is not an academic concern.

2. CROSS-FITTED NUISANCES.  mhat and ehat for each evaluation customer come from
   folds that did not contain them.  Same reasoning, applied to the nuisance
   models.

With both in place the DR estimator is unbiased to within simulation noise
(measured mean error -0.0005 over 20 seeds, simulation standard error 0.0017).
With neither it is not.
"""

from dataclasses import dataclass

from .generator import generate_population
from .uplift import (TLearner, ChurnScorer, CrossFitNuisances, PropensityModel,
                     TransformedOutcome)


@dataclass
class Study:
    train: list
    evaluate: list
    uplift_scores: list      # estimated uplift on the evaluation half
    churn_scores: list       # baseline churn probability on the evaluation half
    transformed_scores: list  # second, independent uplift estimate
    nuisances: CrossFitNuisances

    @property
    def propensities(self):
        return self.nuisances.propensities


def build_study(n=6000, seed=0, propensity_mode="confounded", crossfit_folds=5):
    """Generate a log, split it, fit every model, return the bundle."""
    pop = generate_population(n, seed=seed, propensity_mode=propensity_mode)
    half = n // 2
    train, evaluate = pop[:half], pop[half:]

    uplift_scores = TLearner().fit(train).uplift(evaluate)
    churn_scores = ChurnScorer().fit(train).score(evaluate)

    train_e = PropensityModel().fit(train).predict(train)
    transformed_scores = TransformedOutcome().fit(train, train_e).uplift(evaluate)

    nuisances = CrossFitNuisances(k=crossfit_folds, seed=seed).fit(evaluate)
    return Study(train, evaluate, uplift_scores, churn_scores,
                 transformed_scores, nuisances)
