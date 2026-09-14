"""Targeting policies and their exact (oracle) value.

A policy here is just a list of 0/1 actions aligned with the population list:
1 = contact this customer, 0 = leave them alone.

Everything in this module is deterministic given a score vector, so a policy can
be compared against its own oracle value without any Monte Carlo error -- see
`true_policy_value`.
"""

from .generator import SEGMENTS


def top_k_policy(scores, k_fraction):
    """Target the top `k_fraction` of the population by `scores` (descending).

    Ties are broken by index so the policy is a pure function of the scores.
    """
    n = len(scores)
    k = int(round(k_fraction * n))
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    actions = [0] * n
    for i in order[:k]:
        actions[i] = 1
    return actions


def positive_uplift_policy(uplift, threshold=0.0):
    """Contact everyone whose estimated uplift exceeds `threshold`.

    Unlike top-k this has no budget, so it is the right policy when contact is
    cheap, and it is also the shape the guardrail takes.
    """
    return [1 if u > threshold else 0 for u in uplift]


def apply_sleeping_dog_guardrail(actions, uplift):
    """Hard block: never contact a customer with negative estimated uplift.

    Stated in the README's quality layer as a guardrail rather than a scoring
    tweak on purpose.  A budgeted top-k rule over ANY score -- including a
    correct uplift score -- will happily fill the last slots of the budget with
    negative-uplift customers once it runs out of positive ones.  The guardrail
    is what makes "we never pay to lose a customer" a property of the system
    rather than a property of a particular k.
    """
    return [a if (a == 1 and uplift[i] > 0.0) else 0
            for i, a in enumerate(actions)]


def true_policy_value(pop, actions):
    """Exact expected retention rate under `actions`, in [0, 1].

    No sampling: the generator plants p0 and p1 per customer, so the value of
    any policy on this finite population is known in closed form.  This is the
    number every estimator in `evaluation.py` is graded against.
    """
    total = 0.0
    for i, c in enumerate(pop):
        churn = c.p1 if actions[i] else c.p0
        total += 1.0 - churn
    return total / len(pop)


def true_policy_uplift(pop, actions):
    """Incremental retention versus contacting nobody."""
    return true_policy_value(pop, actions) - true_policy_value(pop, [0] * len(pop))


def targeted_segment_mix(pop, actions):
    """How the campaign list breaks down by true segment."""
    counts = {s: 0 for s in SEGMENTS}
    targeted = 0
    for i, c in enumerate(pop):
        if actions[i]:
            counts[c.segment] += 1
            targeted += 1
    return counts, targeted


def sleeping_dog_harm(pop, actions):
    """Retention lost, per head of population, by contacting sleeping dogs.

    Reported separately from net policy value because it is the number a
    retention director can act on: it is the damage the campaign does, not the
    damage net of whatever good it also did.
    """
    harm = 0.0
    for i, c in enumerate(pop):
        if actions[i] and c.true_uplift < 0.0:
            harm += -c.true_uplift
    return harm / len(pop)
