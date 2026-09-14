"""Off-policy value estimation: direct method, IPW, and doubly-robust.

You never ran the policy you want to ship.  You have a log of what the old,
confounded campaign did.  Three ways to estimate what the new policy would be
worth, all from that log:

  Direct method (DM)
      V_DM = mean_i  mhat(x_i, pi(x_i))
      Trusts the outcome model completely.  Unbiased iff mhat is correct.
      Biased in whatever direction the outcome model is wrong, with no
      correction of any kind -- and low variance, which makes the bias
      invisible unless you are looking for it.

  Inverse propensity weighting (IPW)
      V_IPW = mean_i  1{T_i = pi(x_i)} / P(T_i | x_i) * Y_i
      Trusts the propensity model completely.  Unbiased iff ehat is correct.
      High variance; blows up when any propensity is near 0 or 1.

  Doubly robust (DR)
      V_DR = mean_i [ mhat(x_i, pi(x_i))
                      + 1{T_i = pi(x_i)} / P(T_i | x_i) * (Y_i - mhat(x_i, T_i)) ]
      The direct-method estimate plus an importance-weighted correction of the
      outcome model's own residual.  Consistent if EITHER nuisance model is
      correct.  That "either" is the entire reason to use it, and
      tests/test_double_robustness.py exists to demonstrate it rather than
      assert it.

The intuition for why DR works: if mhat is right, the residual (Y - mhat) has
mean zero given x, so the correction term vanishes in expectation whatever the
weights are.  If ehat is right, the weighting is a valid importance sample, and
adding and subtracting mhat is just a control variate -- it changes the variance
but not the expectation.  Only when BOTH are wrong is DR biased.
"""

import math
import random

from .linalg import clip_prob
from .uplift import retained


def _action_probabilities(pop, propensities, actions):
    """P(T_i = pi(x_i) | x_i) for each i."""
    out = []
    for i, c in enumerate(pop):
        e = clip_prob(propensities[i])
        out.append(e if actions[i] == 1 else 1.0 - e)
    return out


def dr_scores(pop, actions, outcome_model, propensities):
    """Per-customer doubly-robust scores; their mean is V_DR.

    Returned per customer rather than aggregated because every confidence
    interval in this repo is a bootstrap over exactly these terms.
    """
    m_pi_1 = outcome_model.predict_arm(pop, 1)
    m_pi_0 = outcome_model.predict_arm(pop, 0)
    m_obs = outcome_model.predict_observed(pop)
    p_act = _action_probabilities(pop, propensities, actions)
    scores = []
    for i, c in enumerate(pop):
        direct = m_pi_1[i] if actions[i] else m_pi_0[i]
        if c.treated == actions[i]:
            correction = (retained(c) - m_obs[i]) / p_act[i]
        else:
            correction = 0.0
        scores.append(direct + correction)
    return scores


def dm_scores(pop, actions, outcome_model):
    m1 = outcome_model.predict_arm(pop, 1)
    m0 = outcome_model.predict_arm(pop, 0)
    return [m1[i] if actions[i] else m0[i] for i in range(len(pop))]


def ipw_scores(pop, actions, propensities):
    p_act = _action_probabilities(pop, propensities, actions)
    out = []
    for i, c in enumerate(pop):
        if c.treated == actions[i]:
            out.append(retained(c) / p_act[i])
        else:
            out.append(0.0)
    return out


def _mean(xs):
    return sum(xs) / len(xs)


def dm_value(pop, actions, outcome_model):
    return _mean(dm_scores(pop, actions, outcome_model))


def ipw_value(pop, actions, propensities):
    return _mean(ipw_scores(pop, actions, propensities))


def dr_value(pop, actions, outcome_model, propensities):
    return _mean(dr_scores(pop, actions, outcome_model, propensities))


def bootstrap_ci(scores, n_boot=400, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean of a score vector.

    A point estimate of policy value is not decision-grade: the difference
    between two campaign options is usually a fraction of a retention point and
    the log is finite.  The bootstrap here resamples customers, holding the
    fitted nuisance models fixed.

    Caveat, stated because it is a real limitation rather than a footnote: this
    interval conditions on the fitted nuisance models, so it captures sampling
    noise in the evaluation sample but not the uncertainty of having estimated
    mhat and ehat on the same data.  The cross-fit-and-refit bootstrap that
    would capture both is implemented for the UPLIFT model in
    `bootstrap_uplift_intervals`; for policy value it is left out because the
    dominant term at these sample sizes is the IPW residual variance, which this
    interval does capture.
    """
    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += scores[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(math.floor((alpha / 2.0) * n_boot))]
    hi = means[min(n_boot - 1, int(math.ceil((1.0 - alpha / 2.0) * n_boot)) - 1)]
    return _mean(scores), lo, hi


def bootstrap_uplift_intervals(pop, learner_factory, n_boot=40, alpha=0.05, seed=0):
    """Per-customer bootstrap intervals on estimated uplift.

    The model is REFITTED on each bootstrap resample -- that is the expensive
    part and also the only version that means anything.  Bootstrapping the
    predictions of a single fitted model would measure nothing but the noise
    already baked into that fit.

    Returns (point_estimates, lows, highs) evaluated on the full population.
    """
    rng = random.Random(seed)
    n = len(pop)
    draws = [[] for _ in range(n)]
    for _ in range(n_boot):
        sample = [pop[rng.randrange(n)] for _ in range(n)]
        # A resample can end up with an empty arm only in pathological cases;
        # skip rather than crash, and the caller sees a smaller effective B.
        if not any(c.treated for c in sample) or all(c.treated for c in sample):
            continue
        learner = learner_factory()
        learner.fit(sample)
        for i, u in enumerate(learner.uplift(pop)):
            draws[i].append(u)
    point, lows, highs = [], [], []
    for d in draws:
        d.sort()
        b = len(d)
        lo = d[int(math.floor((alpha / 2.0) * b))]
        hi = d[min(b - 1, int(math.ceil((1.0 - alpha / 2.0) * b)) - 1)]
        point.append(_mean(d))
        lows.append(lo)
        highs.append(hi)
    return point, lows, highs


def fraction_indistinguishable_from_zero(lows, highs):
    """Share of customers whose uplift CI straddles zero.

    Reported prominently because in real retention data it is most of the
    population, and a system that hands a retention team a confident ranking
    over customers it cannot actually distinguish is worse than one that admits
    the ranking is noise.  "We can identify 30% of your book; the rest is a coin
    flip" is an actionable statement.  A full ranking of 100% is not.
    """
    n = len(lows)
    k = sum(1 for i in range(n) if lows[i] <= 0.0 <= highs[i])
    return k / n


def uplift_curve(pop, scores, outcome_model, propensities, n_points=11):
    """Qini-style curve: DR incremental policy value against targeted fraction.

    A textbook Qini curve compares the treated and control response rates inside
    the top-k of a score.  That comparison is BIASED here, and biased in exactly
    the flattering direction, because the log is confounded: the treated
    customers inside the top-k are not exchangeable with the control ones.  So
    the curve plotted here is the doubly-robust incremental value of the top-k
    policy over the do-nothing policy, which reduces to the classic Qini under a
    randomised log and stays honest under a confounded one.
    """
    from .policy import top_k_policy
    n = len(pop)
    base = dr_value(pop, [0] * n, outcome_model, propensities)
    pts = []
    for j in range(n_points):
        frac = j / (n_points - 1)
        actions = top_k_policy(scores, frac)
        v = dr_value(pop, actions, outcome_model, propensities)
        pts.append((frac, v - base))
    return pts


def area_under_curve(points):
    """Trapezoidal area; the Qini coefficient analogue."""
    area = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        area += 0.5 * (y0 + y1) * (x1 - x0)
    return area


def pearson_correlation(a, b):
    n = len(a)
    ma, mb = _mean(a), _mean(b)
    sa = sb = sab = 0.0
    for i in range(n):
        da, db = a[i] - ma, b[i] - mb
        sa += da * da
        sb += db * db
        sab += da * db
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    return sab / math.sqrt(sa * sb)
