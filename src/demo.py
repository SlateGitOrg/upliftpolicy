"""The 60-second artefact: churn-ranked targeting vs uplift-ranked targeting,
both evaluated off-policy with a doubly-robust estimator.

ASCII only -- the target console is Windows cp1252.
"""

from .evaluation import (area_under_curve, bootstrap_ci, bootstrap_uplift_intervals,
                         dm_value, dr_scores, dr_value,
                         fraction_indistinguishable_from_zero, ipw_value,
                         pearson_correlation, uplift_curve)
from .generator import SEGMENTS, SEGMENT_OUTCOMES, segment_counts
from .policy import (apply_sleeping_dog_guardrail, sleeping_dog_harm,
                     targeted_segment_mix, top_k_policy, true_policy_value)
from .study import build_study
from .uplift import ConstantPropensity, CrossFitNuisances, OutcomeModel, TLearner

N = 6000
SEED = 7
BUDGET = 0.30
BOOTSTRAPS = 400
# 25 refits keeps the demo inside its 60-second budget; the 2.5%/97.5%
# percentiles from 25 draws are coarse, which the README states.
UPLIFT_BOOTSTRAPS = 25

# A retention campaign is only worth running if the incremental retention it
# buys exceeds what the contact costs.  These are illustrative unit economics,
# stated explicitly rather than buried, so the expected-value column below can
# be checked by hand: GBP 12 to make a retention contact, GBP 340 lifetime value
# of a retained subscriber.  They are the only place money enters the project.
CONTACT_COST_GBP = 12.0
CUSTOMER_VALUE_GBP = 340.0


# Number of independent logs used for the misspecification RMSE table. 5 keeps
# the demo inside its 60-second budget. The gaps it is meant to show (IPW with a
# naive propensity, DR with both models wrong) are 7-30x the per-seed error of
# sound DR. The smaller DM-vs-DR gap is NOT resolved at this size.
MISSPEC_SEEDS = 5


def misspecification_study(n_seeds):
    """Re-run the whole protocol on fresh logs and report RMSE per estimator."""
    import math
    acc = {k: {"dm": [], "ipw": [], "dr": []} for k in
           ("both models sound", "outcome model MISSPECIFIED",
            "propensity model MISSPECIFIED", "BOTH misspecified")}
    for s in range(n_seeds):
        st = build_study(n=N, seed=900 + s)
        pop = st.evaluate
        act = top_k_policy(st.uplift_scores, BUDGET)
        truth = true_policy_value(pop, act)
        good = st.nuisances
        bad_m = CrossFitNuisances(
            outcome_factory=lambda: OutcomeModel(feature_index=[]), seed=s).fit(pop)
        good_e = st.propensities
        bad_e = ConstantPropensity().predict(pop)
        for key, om, es in (
                ("both models sound", good, good_e),
                ("outcome model MISSPECIFIED", bad_m, good_e),
                ("propensity model MISSPECIFIED", good, bad_e),
                ("BOTH misspecified", bad_m, bad_e)):
            acc[key]["dm"].append(dm_value(pop, act, om) - truth)
            acc[key]["ipw"].append(ipw_value(pop, act, es) - truth)
            acc[key]["dr"].append(dr_value(pop, act, om, es) - truth)
    return {k: {e: math.sqrt(sum(x * x for x in v) / len(v)) for e, v in d.items()}
            for k, d in acc.items()}


def rule(char="-", width=78):
    return char * width


def header(title):
    print()
    print(rule("="))
    print(title)
    print(rule("="))


def pct(x, places=2):
    return "%+*.*f%%" % (places + 5, places, 100.0 * x)


def main():
    print(rule("="))
    print("UPLIFTPOLICY -- retention targeting by uplift, evaluated doubly-robustly")
    print(rule("="))
    print("Outcome modelled throughout is RETENTION. Uplift tau = P(stay|contact)")
    print("- P(stay|no contact). Positive tau means the contact helps.")

    study = build_study(n=N, seed=SEED)
    ev = study.evaluate
    n = len(ev)
    nu = study.nuisances
    e = study.propensities
    do_nothing = [0] * n
    base_value = true_policy_value(ev, do_nothing)

    header("1. THE POPULATION (planted ground truth)")
    counts = segment_counts(ev)
    print("%-14s %7s %9s %9s %9s" % ("segment", "n", "P(churn|0)", "P(churn|1)", "uplift"))
    print(rule("-", 53))
    for s in SEGMENTS:
        p0, p1 = SEGMENT_OUTCOMES[s]
        print("%-14s %7d %9.2f %9.2f %9s"
              % (s, counts[s], p0, p1, pct(p0 - p1, 1)))
    print(rule("-", 53))
    print("Note P(churn|0): sleeping_dog %.2f > persuadable %.2f."
          % (SEGMENT_OUTCOMES["sleeping_dog"][0], SEGMENT_OUTCOMES["persuadable"][0]))
    print("A churn model therefore ranks the customers it harms ABOVE the ones")
    print("it can help. That is the whole problem, in one inequality.")
    print()
    print("Logged campaign is CONFOUNDED: contact rate rose with risk.")
    treated_rate = {s: 0.0 for s in SEGMENTS}
    for c in ev:
        treated_rate[c.segment] += c.treated
    print("  observed contact rate by segment: " + ", ".join(
        "%s %.2f" % (s[:11], treated_rate[s] / max(counts[s], 1)) for s in SEGMENTS))

    header("2. UPLIFT RECOVERY (T-learner, fitted on the training half)")
    true_u = [c.true_uplift for c in ev]
    print("Pearson r(estimated uplift, true uplift)      : %+.3f"
          % pearson_correlation(study.uplift_scores, true_u))
    print("Pearson r(transformed-outcome uplift, true)   : %+.3f"
          % pearson_correlation(study.transformed_scores, true_u))
    print("Pearson r(churn score, true uplift)           : %+.3f"
          % pearson_correlation(study.churn_scores, true_u))
    print()
    print("%-14s %14s %14s" % ("segment", "mean true tau", "mean est. tau"))
    print(rule("-", 44))
    for s in SEGMENTS:
        idx = [i for i in range(n) if ev[i].segment == s]
        mt = sum(true_u[i] for i in idx) / len(idx)
        me = sum(study.uplift_scores[i] for i in idx) / len(idx)
        print("%-14s %14s %14s" % (s, pct(mt, 1), pct(me, 1)))
    print(rule("-", 44))
    print("The churn score is a probability: it is positive for everyone and")
    print("cannot express 'contacting this customer costs you money' at all.")

    header("3. THE HEADLINE -- top %d%% by churn vs top %d%% by uplift"
           % (int(BUDGET * 100), int(BUDGET * 100)))
    policies = [
        ("do nothing", do_nothing),
        ("top-30% by churn score", top_k_policy(study.churn_scores, BUDGET)),
        ("top-30% by uplift", top_k_policy(study.uplift_scores, BUDGET)),
    ]
    print("%-24s %8s %8s %22s" % ("policy", "true V", "DR V", "DR 95% CI"))
    print(rule("-", 66))
    results = {}
    for name, actions in policies:
        tv = true_policy_value(ev, actions)
        point, lo, hi = bootstrap_ci(dr_scores(ev, actions, nu, e),
                                     n_boot=BOOTSTRAPS, seed=11)
        results[name] = (tv, point, lo, hi, actions)
        print("%-24s %8.4f %8.4f   [%.4f, %.4f]" % (name, tv, point, lo, hi))
    print(rule("-", 66))
    churn_tv = results["top-30% by churn score"][0]
    uplift_tv = results["top-30% by uplift"][0]
    print("Incremental retention vs doing nothing (true):")
    print("  churn-ranked  : %s  <- the campaign DESTROYS retention" % pct(churn_tv - base_value))
    print("  uplift-ranked : %s" % pct(uplift_tv - base_value))
    print()
    print("The churn-ranked DR interval [%.4f, %.4f] contains the do-nothing"
          % (results["top-30% by churn score"][2], results["top-30% by churn score"][3]))
    print("value %.4f. On this much data you cannot even say the standard" % base_value)
    print("campaign beats sending nothing. A point estimate would have hidden that.")

    print()
    print("Who ends up on each campaign list:")
    print("%-24s %7s %7s %7s %7s %10s" % ("policy", "persu.", "sure", "lost", "SLEEP", "harm done"))
    print(rule("-", 68))
    for name in ("top-30% by churn score", "top-30% by uplift"):
        actions = results[name][4]
        mix, total = targeted_segment_mix(ev, actions)
        print("%-24s %7d %7d %7d %7d %10s"
              % (name, mix["persuadable"], mix["sure_thing"], mix["lost_cause"],
                 mix["sleeping_dog"], pct(-sleeping_dog_harm(ev, actions))))
    print(rule("-", 68))
    mix_c, tot_c = targeted_segment_mix(ev, results["top-30% by churn score"][4])
    print("%.0f%% of the churn-ranked campaign list is sleeping dogs."
          % (100.0 * mix_c["sleeping_dog"] / tot_c))
    print("Harm column = retention lost per head of population by contacting")
    print("customers with negative true uplift. At GBP %.0f per retained"
          % CUSTOMER_VALUE_GBP)
    print("subscriber that is GBP %.0f of lifetime value destroyed per 1,000"
          % (1000 * sleeping_dog_harm(ev, results["top-30% by churn score"][4])
             * CUSTOMER_VALUE_GBP))
    print("customers in the book, before the GBP %.0f/contact cost."
          % CONTACT_COST_GBP)

    header("4. EXPECTED VALUE OF THE CAMPAIGN")
    print("%-24s %8s %14s %14s" % ("policy", "contacts", "contact cost", "net GBP/1000"))
    print(rule("-", 64))
    for name in ("top-30% by churn score", "top-30% by uplift"):
        tv, _, _, _, actions = results[name]
        k = sum(actions)
        cost = k * CONTACT_COST_GBP
        gain = (tv - base_value) * n * CUSTOMER_VALUE_GBP
        net_per_1000 = 1000.0 * (gain - cost) / n
        print("%-24s %8d %14s %14s"
              % (name, k, "GBP %8.0f" % cost, "GBP %+8.0f" % net_per_1000))
    print(rule("-", 64))

    header("5. QINI-STYLE CURVE (DR incremental value vs targeted fraction)")
    curve_u = uplift_curve(ev, study.uplift_scores, nu, e)
    curve_c = uplift_curve(ev, study.churn_scores, nu, e)
    print("%10s %14s %14s" % ("targeted", "uplift-ranked", "churn-ranked"))
    print(rule("-", 40))
    for (f, vu), (_, vc) in zip(curve_u, curve_c):
        bar_u = "#" * int(round(max(vu, 0.0) * 400))
        print("%9.0f%% %14s %14s  %s" % (100 * f, pct(vu), pct(vc), bar_u))
    print(rule("-", 40))
    print("Area under curve: uplift %.4f, churn %.4f (ratio %.2fx)"
          % (area_under_curve(curve_u), area_under_curve(curve_c),
             area_under_curve(curve_u) / max(area_under_curve(curve_c), 1e-9)))

    header("6. WHY DOUBLY ROBUST -- one nuisance model broken at a time")
    actions = results["top-30% by uplift"][4]
    truth = uplift_tv
    good_nu = nu
    bad_outcome = CrossFitNuisances(
        outcome_factory=lambda: OutcomeModel(feature_index=[]), seed=SEED).fit(ev)
    bad_e = ConstantPropensity().predict(ev)
    print("True value of the policy being evaluated: %.4f" % truth)
    print()
    print("%-34s %10s %10s %10s" % ("scenario", "DM err", "IPW err", "DR err"))
    print(rule("-", 68))
    rows = [
        ("both models sound",
         dm_value(ev, actions, good_nu) - truth,
         ipw_value(ev, actions, e) - truth,
         dr_value(ev, actions, good_nu, e) - truth),
        ("outcome model MISSPECIFIED",
         dm_value(ev, actions, bad_outcome) - truth,
         ipw_value(ev, actions, e) - truth,
         dr_value(ev, actions, bad_outcome, e) - truth),
        ("propensity model MISSPECIFIED",
         dm_value(ev, actions, good_nu) - truth,
         ipw_value(ev, actions, bad_e) - truth,
         dr_value(ev, actions, good_nu, bad_e) - truth),
        ("BOTH misspecified",
         dm_value(ev, actions, bad_outcome) - truth,
         ipw_value(ev, actions, bad_e) - truth,
         dr_value(ev, actions, bad_outcome, bad_e) - truth),
    ]
    for name, a, b, c in rows:
        print("%-34s %10s %10s %10s" % (name, pct(a), pct(b), pct(c)))
    print(rule("-", 68))
    print()
    print("One seed is an anecdote. Repeating the whole protocol over %d"
          % MISSPEC_SEEDS)
    print("independent logs, root-mean-square error of each estimator:")
    print()
    print("%-34s %10s %10s %10s" % ("scenario", "DM rmse", "IPW rmse", "DR rmse"))
    print(rule("-", 68))
    for name, errs in misspecification_study(MISSPEC_SEEDS).items():
        print("%-34s %9.2f%% %9.2f%% %9.2f%%"
              % (name, 100 * errs["dm"], 100 * errs["ipw"], 100 * errs["dr"]))
    print(rule("-", 68))
    print("DR survives either single failure and does NOT survive both. That")
    print("last row matters: doubly robust is two chances, not a guarantee.")

    header("7. HOW MUCH OF THE BOOK CAN WE ACTUALLY RANK?")
    _, lows, highs = bootstrap_uplift_intervals(
        study.train, lambda: TLearner(), n_boot=UPLIFT_BOOTSTRAPS, seed=3)
    frac_zero = fraction_indistinguishable_from_zero(lows, highs)
    print("Bootstrap (B=%d, refitting the T-learner each time) 95%% intervals"
          % UPLIFT_BOOTSTRAPS)
    print("on individual uplift:")
    print("  uplift indistinguishable from zero : %5.1f%% of customers" % (100 * frac_zero))
    print("  confidently positive uplift        : %5.1f%%"
          % (100 * sum(1 for i in range(len(lows)) if lows[i] > 0) / len(lows)))
    print("  confidently NEGATIVE (sleeping dog): %5.1f%%"
          % (100 * sum(1 for i in range(len(highs)) if highs[i] < 0) / len(highs)))
    print()
    print("This is the number a retention director should be given first. A")
    print("ranked list of the whole book implies a precision that is not there.")

    header("8. GUARDRAIL -- never pay to lose a customer")
    # 90% rather than 30%: at a 30% budget every selected customer already has
    # positive estimated uplift, so the guardrail is a no-op and demonstrates
    # nothing. It only bites once the budget is large enough that a top-k rule
    # starts scraping into negative-uplift territory -- which is exactly the
    # situation it exists for, and exactly the situation a "spend the whole
    # retention budget" mandate creates.
    wide = top_k_policy(study.uplift_scores, 0.90)
    guarded = apply_sleeping_dog_guardrail(wide, study.uplift_scores)
    print("%-34s %9s %9s %10s" % ("policy", "contacts", "true V", "harm"))
    print(rule("-", 64))
    for name, a in (("top-90% by uplift", wide),
                    ("  + negative-uplift guardrail", guarded)):
        print("%-34s %9d %9.4f %10s"
              % (name, sum(a), true_policy_value(ev, a), pct(-sleeping_dog_harm(ev, a))))
    guarded_churn = apply_sleeping_dog_guardrail(
        results["top-30% by churn score"][4], study.uplift_scores)
    print("%-34s %9d %9.4f %10s"
          % ("top-30% by churn + guardrail", sum(guarded_churn),
             true_policy_value(ev, guarded_churn), pct(-sleeping_dog_harm(ev, guarded_churn))))
    print(rule("-", 64))
    print("The guardrail rescues most of the churn-ranked campaign's damage")
    print("without changing the churn model at all -- but it needs an uplift")
    print("model to do it, which is the point.")

    print()
    print(rule("="))
    print("BOTTOM LINE: ranking by churn bought %s incremental retention."
          % pct(churn_tv - base_value))
    print("Ranking by uplift bought %s. The difference is %s of the"
          % (pct(uplift_tv - base_value), pct(uplift_tv - churn_tv)))
    print("subscriber base, and it is entirely about WHO is on the list.")
    print(rule("="))


if __name__ == "__main__":
    main()
