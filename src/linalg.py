"""Small dense linear algebra and GLM fitting, pure standard library.

numpy/scikit-learn are not available in this environment (see the README's
implementation note), so the two things every estimator here needs -- solving a
small symmetric system and fitting a logistic regression -- are written out.

The design choice worth recording: logistic regression is fitted by IRLS
(Newton-Raphson on the log-likelihood) rather than by gradient descent.  Gradient
descent needs a learning rate, and a learning rate is exactly the kind of
invented constant that quietly biases every downstream causal estimate when the
optimiser has not converged.  IRLS converges quadratically and its stopping rule
is a real convergence criterion on the coefficient vector, not a step budget.
"""

import math

# Ridge penalty applied to every logistic fit except the intercept.
# Rationale, not taste: the T-learner fits a separate model on the treated and
# the control arm.  Under confounded assignment one arm can be thin in a corner
# of covariate space and the MLE there is separable (coefficients run to
# infinity).  A small ridge bounds the fit without materially biasing it at the
# sample sizes used here; 1e-3 is roughly 1/n for n ~ 1000, i.e. it contributes
# about one pseudo-observation's worth of information per coefficient.
DEFAULT_RIDGE = 1e-3

# Probabilities are clipped away from 0 and 1 before any division.  1e-6 keeps
# the largest possible inverse-propensity weight at 1e6, which is already far
# past the point where an IPW estimate is meaningful, so the clip never silently
# rescues a bad design -- it only prevents a ZeroDivisionError from masking one.
PROB_EPS = 1e-6


def clip_prob(p, eps=PROB_EPS):
    if p < eps:
        return eps
    if p > 1.0 - eps:
        return 1.0 - eps
    return p


def sigmoid(z):
    # Branching avoids overflow in exp for large |z|; math.exp(710) raises.
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def solve_symmetric(a, b):
    """Solve a x = b for small dense symmetric positive definite `a`.

    Gaussian elimination with partial pivoting.  Cholesky would be marginally
    faster but fails outright on the mildly indefinite Hessians that appear
    mid-IRLS before the ridge has taken effect; pivoted elimination degrades
    gracefully and we check the pivot magnitude explicitly.
    """
    n = len(b)
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot_row][col]) < 1e-12:
            raise ValueError("singular system in solve_symmetric")
        if pivot_row != col:
            m[col], m[pivot_row] = m[pivot_row], m[col]
        piv = m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] / piv
            if f == 0.0:
                continue
            row_r = m[r]
            row_c = m[col]
            for c in range(col, n + 1):
                row_r[c] -= f * row_c[c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = m[r][n]
        for c in range(r + 1, n):
            s -= m[r][c] * x[c]
        x[r] = s / m[r][r]
    return x


class LogisticRegression:
    """Binary logistic regression fitted by ridge-penalised IRLS.

    `fit` takes a design matrix WITHOUT an intercept column; the intercept is
    added internally so callers cannot forget it. Supports observation weights,
    which the transformed-outcome learner needs.
    """

    def __init__(self, ridge=DEFAULT_RIDGE, max_iter=50, tol=1e-8):
        self.ridge = ridge
        self.max_iter = max_iter
        self.tol = tol
        self.coef = None
        self.n_iter = 0
        self.converged = False

    def fit(self, x, y, weights=None):
        n = len(x)
        if n == 0:
            raise ValueError("cannot fit on an empty sample")
        p = len(x[0]) + 1
        design = [[1.0] + list(row) for row in x]
        w_obs = [1.0] * n if weights is None else list(weights)
        beta = [0.0] * p
        for it in range(self.max_iter):
            hess = [[0.0] * p for _ in range(p)]
            grad = [0.0] * p
            for i in range(n):
                row = design[i]
                z = 0.0
                for j in range(p):
                    z += beta[j] * row[j]
                mu = sigmoid(z)
                wi = w_obs[i]
                r = wi * (y[i] - mu)
                s = wi * max(mu * (1.0 - mu), 1e-10)
                for j in range(p):
                    grad[j] += r * row[j]
                    rj = s * row[j]
                    hj = hess[j]
                    for k in range(j, p):
                        hj[k] += rj * row[k]
            # Symmetrise and add the ridge (index 0 is the intercept: unpenalised,
            # because penalising it shrinks the base rate towards 50% and a
            # churn base rate is nowhere near 50%).
            for j in range(p):
                for k in range(j):
                    hess[j][k] = hess[k][j]
                if j > 0:
                    hess[j][j] += self.ridge * n
                    grad[j] -= self.ridge * n * beta[j]
            try:
                step = solve_symmetric(hess, grad)
            except ValueError:
                break
            max_step = 0.0
            for j in range(p):
                beta[j] += step[j]
                max_step = max(max_step, abs(step[j]))
            self.n_iter = it + 1
            if max_step < self.tol:
                self.converged = True
                break
        self.coef = beta
        return self

    def predict_one(self, row):
        b = self.coef
        z = b[0]
        for j, v in enumerate(row):
            z += b[j + 1] * v
        return sigmoid(z)

    def predict(self, x):
        return [self.predict_one(row) for row in x]


class ConstantModel:
    """Degenerate model that predicts the (weighted) sample mean.

    Used deliberately in the double-robustness tests as the *misspecified*
    component: it ignores covariates entirely, which is the worst realistic
    outcome-model or propensity-model failure short of an adversarial one.
    """

    def __init__(self, value=None):
        self.value = value

    def fit(self, x, y, weights=None):
        if self.value is None:
            self.value = sum(y) / len(y) if y else 0.5
        return self

    def predict_one(self, row):
        return self.value

    def predict(self, x):
        return [self.value] * len(x)
