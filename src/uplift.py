"""Uplift (individual treatment effect) estimators and the churn-model baseline.

Outcome convention for the whole estimation layer: the modelled outcome is
RETENTION, y = 1 - churned.  Uplift is then

    tau(x) = m1(x) - m0(x),    mt(x) = E[retained | T = t, X = x]

which is identical to p0 - p1 in the generator's churn parameterisation.  Doing
the sign flip once, here, is deliberate: policy value is an expected retention
rate, and an estimator whose sign convention flips halfway through the pipeline
is the single most common way an uplift system ends up targeting exactly the
wrong people.

Three estimators live here:

  ChurnScorer       the baseline that must be beaten.  A plain propensity-to-
                    churn model.  It has NO notion of the intervention, so it
                    cannot express negative uplift even in principle.
  TLearner          two outcome models, one per arm; tau = m1 - m0.
  TransformedOutcome a single regression on the propensity-transformed outcome,
                    which targets tau directly.  Included because it fails and
                    succeeds in different places from the T-learner, and having
                    two independent uplift estimates is how you notice when one
                    of them has quietly broken.
"""

from .linalg import LogisticRegression, ConstantModel, clip_prob


def retained(customer):
    return 1 - customer.churned


class PropensityModel:
    """Estimates e(x) = P(treated = 1 | x) from the logged data."""

    def __init__(self, feature_index=None, ridge=1e-3):
        # feature_index lets a caller deliberately fit on the WRONG columns,
        # which is how the misspecification tests are constructed.
        self.feature_index = feature_index
        self.model = LogisticRegression(ridge=ridge)

    def _rows(self, pop):
        if self.feature_index is None:
            return [list(c.features) for c in pop]
        return [[c.features[j] for j in self.feature_index] for c in pop]

    def fit(self, pop):
        self.model.fit(self._rows(pop), [c.treated for c in pop])
        return self

    def predict(self, pop):
        return [clip_prob(p) for p in self.model.predict(self._rows(pop))]


class ConstantPropensity:
    """Assumes the log was a fair coin flip. Wrong whenever it was not.

    This is the misspecified propensity used in the double-robustness test, and
    it is not a strawman: assuming a historical campaign was effectively random
    is the most common propensity error in practice.
    """

    def __init__(self, value=0.5):
        self.value = value

    def fit(self, pop):
        return self

    def predict(self, pop):
        return [self.value] * len(pop)


class TruePropensity:
    """Oracle propensity, available only because the data is synthetic."""

    def fit(self, pop):
        return self

    def predict(self, pop):
        return [clip_prob(c.propensity) for c in pop]


class OutcomeModel:
    """Arm-specific retention models m0(x), m1(x).

    `feature_index=[]` produces intercept-only models, i.e. a deliberately
    misspecified outcome model that knows the base retention rate in each arm
    and nothing else.
    """

    def __init__(self, feature_index=None, ridge=1e-3):
        self.feature_index = feature_index
        self.models = {}

    def _rows(self, pop):
        if self.feature_index is None:
            return [list(c.features) for c in pop]
        return [[c.features[j] for j in self.feature_index] for c in pop]

    def fit(self, pop):
        for arm in (0, 1):
            sub = [c for c in pop if c.treated == arm]
            if not sub:
                raise ValueError("no logged records in arm %d" % arm)
            y = [retained(c) for c in sub]
            if self.feature_index is not None and len(self.feature_index) == 0:
                self.models[arm] = ConstantModel().fit(None, y)
            else:
                self.models[arm] = LogisticRegression().fit(self._rows(sub), y)
        return self

    def predict_arm(self, pop, arm):
        return self.models[arm].predict(self._rows(pop))

    def predict_observed(self, pop):
        """m_{T_i}(x_i) -- the model's prediction for the arm actually logged."""
        rows = self._rows(pop)
        return [self.models[c.treated].predict_one(rows[i])
                for i, c in enumerate(pop)]


class TrueOutcome:
    """Oracle outcome model. Used only to isolate propensity misspecification."""

    def fit(self, pop):
        return self

    def predict_arm(self, pop, arm):
        return [1.0 - (c.p1 if arm == 1 else c.p0) for c in pop]

    def predict_observed(self, pop):
        return [1.0 - (c.p1 if c.treated else c.p0) for c in pop]


class TLearner:
    """Uplift by differencing two arm-specific outcome models."""

    def __init__(self, feature_index=None):
        self.outcome = OutcomeModel(feature_index=feature_index)

    def fit(self, pop):
        self.outcome.fit(pop)
        return self

    def uplift(self, pop):
        m1 = self.outcome.predict_arm(pop, 1)
        m0 = self.outcome.predict_arm(pop, 0)
        return [m1[i] - m0[i] for i in range(len(pop))]


class TransformedOutcome:
    """Uplift via the Horvitz-Thompson transformed outcome, fitted by weighted OLS.

    Z_i = Y_i * (T_i - e_i) / (e_i * (1 - e_i))  has E[Z | X = x] = tau(x), so a
    plain regression of Z on x estimates uplift directly rather than as a
    difference of two nuisance models.

    The trap, and the reason this is not the default estimator: Z has enormous
    variance wherever e(x) is near 0 or 1.  With the generator's propensity
    bounded into [0.12, 0.88] the weight peaks around 9.5, which is tolerable.
    On a log with real positivity violations this estimator falls apart while
    the T-learner merely becomes biased, and that difference is why both are
    here.
    """

    def __init__(self):
        self.coef = None

    def fit(self, pop, propensities):
        n = len(pop)
        p = len(pop[0].features) + 1
        z = []
        for i, c in enumerate(pop):
            e = clip_prob(propensities[i])
            z.append(retained(c) * (c.treated - e) / (e * (1.0 - e)))
        design = [[1.0] + list(c.features) for c in pop]
        # Normal equations with a ridge; OLS by hand because there is no numpy.
        xtx = [[0.0] * p for _ in range(p)]
        xty = [0.0] * p
        for i in range(n):
            row = design[i]
            zi = z[i]
            for j in range(p):
                xty[j] += row[j] * zi
                rj = row[j]
                for k in range(j, p):
                    xtx[j][k] += rj * row[k]
        for j in range(p):
            for k in range(j):
                xtx[j][k] = xtx[k][j]
            if j > 0:
                xtx[j][j] += 1e-3 * n
        from .linalg import solve_symmetric
        self.coef = solve_symmetric(xtx, xty)
        return self

    def uplift(self, pop):
        out = []
        for c in pop:
            v = self.coef[0]
            for j, f in enumerate(c.features):
                v += self.coef[j + 1] * f
            out.append(v)
        return out


class ChurnScorer:
    """The baseline: a plain propensity-to-churn model.

    Trained on the pooled log with no treatment indicator, which is exactly how
    a retention churn model is built in practice -- the historical campaign flag
    is usually not even in the feature store.  Its score is P(churn), and the
    generic campaign targets the top decile of it.
    """

    def fit(self, pop):
        self.model = LogisticRegression().fit(
            [list(c.features) for c in pop], [c.churned for c in pop])
        return self

    def score(self, pop):
        """Higher = more likely to churn."""
        return self.model.predict([list(c.features) for c in pop])


class CrossFitNuisances:
    """K-fold cross-fitted outcome and propensity estimates.

    Why this exists, and why it is not optional
    -------------------------------------------
    The first working version of this repo fitted mhat and ehat on the SAME
    sample the doubly-robust estimator was then evaluated on.  Over 30 seeds
    that produced a persistent +0.005 bias in V_DR -- about a tenth of the whole
    churn-vs-uplift decision margin -- with no accompanying error message.  The
    cause is the standard one: the outcome model has partly memorised its own
    training residuals, so the (Y - mhat) correction term is no longer mean-zero
    given x, and the cross-term between nuisance error and the estimator does
    not vanish.

    Cross-fitting removes it: each customer's mhat and ehat come from a model
    that never saw that customer.  This is the Chernozhukov et al. double/
    debiased ML construction, and it turns DR from "approximately right" into
    "right up to sampling noise".

    Exposes the same interface as OutcomeModel plus a `propensities` list, so it
    drops straight into `evaluation.dr_value`.
    """

    def __init__(self, outcome_factory=None, propensity_factory=None, k=5, seed=0):
        self.outcome_factory = outcome_factory or (lambda: OutcomeModel())
        self.propensity_factory = propensity_factory or (lambda: PropensityModel())
        self.k = k
        self.seed = seed
        self.m0 = None
        self.m1 = None
        self.m_obs = None
        self.propensities = None

    def fit(self, pop):
        import random as _random
        n = len(pop)
        rng = _random.Random(self.seed)
        order = list(range(n))
        rng.shuffle(order)
        folds = [order[i::self.k] for i in range(self.k)]
        self.m0 = [0.0] * n
        self.m1 = [0.0] * n
        self.m_obs = [0.0] * n
        self.propensities = [0.5] * n
        for f in folds:
            held = set(f)
            train = [pop[i] for i in range(n) if i not in held]
            test = [pop[i] for i in f]
            om = self.outcome_factory().fit(train)
            pm = self.propensity_factory().fit(train)
            a1 = om.predict_arm(test, 1)
            a0 = om.predict_arm(test, 0)
            obs = om.predict_observed(test)
            es = pm.predict(test)
            for j, i in enumerate(f):
                self.m1[i] = a1[j]
                self.m0[i] = a0[j]
                self.m_obs[i] = obs[j]
                self.propensities[i] = es[j]
        self._n = n
        return self

    def _check(self, pop):
        if len(pop) != self._n:
            raise ValueError(
                "cross-fitted estimates are tied to the population they were fitted on")

    def predict_arm(self, pop, arm):
        self._check(pop)
        return list(self.m1 if arm == 1 else self.m0)

    def predict_observed(self, pop):
        self._check(pop)
        return list(self.m_obs)
