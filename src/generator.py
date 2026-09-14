"""Synthetic retention population with a planted four-quadrant uplift structure.

The whole project rests on this file. If the ground truth is not planted
explicitly, every claim downstream ("uplift ranking beats churn ranking",
"doubly-robust recovers the true policy value") becomes unfalsifiable.

The four quadrants of a retention campaign
------------------------------------------
Write p0 for the probability a customer churns if left alone and p1 for the
probability they churn if the retention team contacts them.  Uplift is defined
throughout this repo as the RETENTION gain from contact:

    tau(x) = p0(x) - p1(x)

so positive tau means contact helps.  The four segments:

  persuadable  high p0, contact works       tau strongly positive  <- the only
                                                                     people worth
                                                                     contacting
  sure_thing   low p0, nothing to save      tau ~ 0
  lost_cause   very high p0, contact is
               irrelevant, they are gone    tau ~ 0
  sleeping_dog moderately high p0, and the
               contact itself pushes them
               out                          tau NEGATIVE

The sleeping dog is the reason a churn model is the wrong tool.  A churn model
scores p0 and has no representation of the intervention at all, so it cannot
distinguish a persuadable (p0 = 0.50, tau = +0.25) from a sleeping dog
(p0 = 0.55, tau = -0.20).  It will rank the sleeping dog HIGHER, because 0.55 >
0.50.  That is not a hypothetical: the numbers below are chosen so it happens.

Why the segment is latent and the features are noisy
----------------------------------------------------
A generator that exposes a clean segment label as a feature makes the learning
problem trivial and the result uninformative.  Here three independent binary
latent traits -- risk, responsiveness, reactance -- determine the segment, and
each trait is observed only through two noisy continuous proxies (plus two pure
noise columns).  The learner has to recover uplift through that noise, which is
what makes the recovered-vs-true correlation a real measurement.

Treatment assignment is CONFOUNDED on purpose
----------------------------------------------
Historical retention campaigns are never randomised: the team contacted people
who looked risky.  So the logged propensity e(x) depends on the risk proxies.
That is what makes the propensity model load-bearing, and therefore what makes
the double-robustness demonstration meaningful -- under a 50/50 randomised log,
IPW with a constant propensity would be correct and the whole point would
evaporate.
"""

import math
import random
from dataclasses import dataclass

SEGMENTS = ("persuadable", "sure_thing", "lost_cause", "sleeping_dog")

# Planted outcome probabilities per segment: (p0 = churn if untreated,
# p1 = churn if contacted).  These are the numbers the entire test suite
# checks against, so they are stated once, here, and never duplicated.
#
# The critical relation is p0[sleeping_dog] > p0[persuadable].  Without it the
# churn-ranked policy would accidentally avoid the sleeping dogs and the
# headline comparison would be a strawman.  Real reactance risk correlates with
# real churn risk -- a customer whose contract is about to roll over looks risky
# to a churn model AND is the customer a phone call reminds to leave -- so this
# ordering is the realistic one, not a rigged one.
SEGMENT_OUTCOMES = {
    "persuadable": (0.50, 0.25),   # tau = +0.25
    "sure_thing": (0.08, 0.06),    # tau = +0.02, inside the noise floor
    "lost_cause": (0.80, 0.79),    # tau = +0.01, inside the noise floor
    "sleeping_dog": (0.55, 0.75),  # tau = -0.20
}

# Population mix.  Chosen so that the top 30% by churn score is composed almost
# entirely of lost causes (15%) and sleeping dogs (20%) -- i.e. half of a
# churn-ranked campaign list is people the campaign actively harms, and the
# other half is people it cannot help.
SEGMENT_SHARES = {
    "persuadable": 0.25,
    "sure_thing": 0.40,
    "lost_cause": 0.15,
    "sleeping_dog": 0.20,
}

FEATURE_NAMES = (
    "risk_proxy_1",      # e.g. support tickets last quarter
    "risk_proxy_2",      # e.g. months since last upgrade
    "response_proxy_1",  # e.g. historical promo redemption
    "response_proxy_2",  # e.g. app engagement
    "reactance_proxy_1",  # e.g. contract auto-renew window open
    "reactance_proxy_2",  # e.g. quiet account, no inbound contact in 12m
    "noise_1",
    "noise_2",
)
N_FEATURES = len(FEATURE_NAMES)

# Signal-to-noise of each latent trait's proxies.  A trait shifts its two
# proxies by +/- TRAIT_SHIFT with observation noise PROXY_SD, so one proxy has
# d' = 2*TRAIT_SHIFT/PROXY_SD = 2.0 and the pair gives d' ~ 2.83.  That is
# strong enough that uplift is genuinely learnable and weak enough that an
# appreciable fraction of individuals stay statistically ambiguous -- which is
# the honest situation in real retention data and something this repo reports
# rather than hides.
TRAIT_SHIFT = 0.8
PROXY_SD = 0.8

# Confounded logging policy: contact probability rises with the observed risk
# proxies.  Bounded into [0.12, 0.88] so every customer has some chance of
# either arm (overlap / positivity).  Without overlap no off-policy estimator
# of any kind is identified, and the repo would be estimating nothing.
PROPENSITY_INTERCEPT = -0.2
PROPENSITY_RISK_WEIGHT = 0.9
PROPENSITY_REACTANCE_WEIGHT = 0.35
PROPENSITY_FLOOR = 0.12
PROPENSITY_CEIL = 0.88


@dataclass
class Customer:
    """One logged customer-campaign record.

    `features` is all the estimators are allowed to see, together with
    `treated` and `churned`.  Everything else on this object is ground truth
    used only by tests and by the demo's "true" columns.
    """

    features: tuple
    treated: int
    churned: int
    propensity: float     # true P(treated = 1 | x)
    segment: str
    p0: float             # true P(churn | untreated)
    p1: float             # true P(churn | treated)

    @property
    def true_uplift(self):
        """True retention uplift tau = p0 - p1. Positive means contact helps."""
        return self.p0 - self.p1


def _bound(value, lo, hi):
    return lo if value < lo else (hi if value > hi else value)


def _segment_from_traits(risk, responsive, reactant):
    """Map the three latent binaries onto the four quadrants.

    Reactance dominates: a customer who resents being contacted is a sleeping
    dog whatever else is true of them.  That precedence is a modelling choice
    and it is the conservative one -- it makes the sleeping-dog segment harder
    to separate from the persuadables, not easier.
    """
    if reactant:
        return "sleeping_dog"
    if risk and responsive:
        return "persuadable"
    if risk:
        return "lost_cause"
    return "sure_thing"


def _trait_probabilities():
    """Latent trait marginals that reproduce SEGMENT_SHARES exactly.

    Solved by hand from the precedence rules in `_segment_from_traits`:
        P(reactant)                         = share(sleeping_dog)
        P(risk) * (1 - P(reactant))         = share(persuadable) + share(lost_cause)
        P(responsive | risk)                = share(persuadable)
                                              / (share(persuadable) + share(lost_cause))
    """
    p_react = SEGMENT_SHARES["sleeping_dog"]
    risky_mass = SEGMENT_SHARES["persuadable"] + SEGMENT_SHARES["lost_cause"]
    p_risk = risky_mass / (1.0 - p_react)
    p_resp = SEGMENT_SHARES["persuadable"] / risky_mass
    return p_risk, p_resp, p_react


def generate_population(n, seed=0, propensity_mode="confounded"):
    """Generate `n` logged records.

    propensity_mode:
      "confounded" -- the realistic default; contact probability depends on the
                      observed risk and reactance proxies.
      "randomised" -- 50/50 coin flip, used only to show that the estimators
                      agree with each other when confounding is absent.
    """
    rng = random.Random(seed)
    p_risk, p_resp, p_react = _trait_probabilities()
    out = []
    for _ in range(n):
        risk = 1 if rng.random() < p_risk else 0
        responsive = 1 if rng.random() < p_resp else 0
        reactant = 1 if rng.random() < p_react else 0
        segment = _segment_from_traits(risk, responsive, reactant)
        p0, p1 = SEGMENT_OUTCOMES[segment]

        def proxy(trait):
            mean = TRAIT_SHIFT if trait else -TRAIT_SHIFT
            return rng.gauss(mean, PROXY_SD)

        features = (
            proxy(risk), proxy(risk),
            proxy(responsive), proxy(responsive),
            proxy(reactant), proxy(reactant),
            rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0),
        )

        if propensity_mode == "randomised":
            e = 0.5
        elif propensity_mode == "confounded":
            risk_signal = 0.5 * (features[0] + features[1])
            reactance_signal = 0.5 * (features[4] + features[5])
            z = (PROPENSITY_INTERCEPT
                 + PROPENSITY_RISK_WEIGHT * risk_signal
                 + PROPENSITY_REACTANCE_WEIGHT * reactance_signal)
            e = _bound(1.0 / (1.0 + math.exp(-z)), PROPENSITY_FLOOR, PROPENSITY_CEIL)
        else:
            raise ValueError("unknown propensity_mode: %r" % (propensity_mode,))

        treated = 1 if rng.random() < e else 0
        churn_prob = p1 if treated else p0
        churned = 1 if rng.random() < churn_prob else 0
        out.append(Customer(features, treated, churned, e, segment, p0, p1))
    return out


def segment_counts(pop):
    counts = {s: 0 for s in SEGMENTS}
    for c in pop:
        counts[c.segment] += 1
    return counts


def true_average_uplift(pop):
    return sum(c.true_uplift for c in pop) / len(pop)
