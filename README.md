# upliftpolicy

> Retention targeting by uplift with doubly-robust policy evaluation, including the customers your outreach drives away.

> **Implementation note.** The runnable reference is pure Python standard library: logistic regression by hand-written IRLS instead of scikit-learn, a T-learner and a transformed-outcome learner instead of EconML's X-/R-learners and causal forest, and a synthetic generator only. Criteo-UPLIFT / Hillstrom validation, DuckDB, the FastAPI scoring service and Docker remain the target, not the reference. The expected-value calculation is printed by the demo instead of served over HTTP.

`FLAGSHIP` · **AI / ML Engineering** · Expert · ~5-6 weeks · Telecom / subscription fintech

**Primary language:** Python
**Tags:** `causal-ml`, `uplift`, `off-policy-evaluation`, `qini`, `econml`, `fastapi`

---

## The problem

A retention team builds a churn model and sends a discount to everyone with a high churn score. A large share of those customers were going to churn regardless, so the discount is wasted. Another share were never going to churn but happily accept the discount, actively destroying revenue. The model is accurate and the campaign loses money - and because the campaign 'retained' the customers who were never leaving, it reports a success.

## ⭐ The differentiator

Models **uplift - individual treatment effect - rather than churn propensity**, evaluated by **Qini curve and doubly-robust off-policy value estimation**, so targeting optimises incremental retention per pound. It explicitly identifies the **sleeping-dog segment**: customers made *more* likely to leave by being contacted, which a propensity model cannot represent at all because it has no notion of the intervention.

This is the sentence to lead with when someone asks you to walk through the
project. Everything else in this repo exists to make it true and to prove it.

## Data

A documented synthetic generator with a **known individual treatment effect function including a sleeping-dog segment**, plus validation against the openly available **Criteo-UPLIFT** and **Hillstrom MineThatData** datasets.

> No paid API key is required to run or demo this project. Where a paid
> service would add value it is wired as an optional enhancement behind an
> interface with an offline mock as the default implementation.

## Stack

- Python: scikit-learn, EconML
- DuckDB
- FastAPI for the scoring service
- Docker, CI, pytest

## Core capabilities

- Uplift estimators (T-, X-, R-learner and causal forest) behind a common evaluation harness
- Qini and uplift-at-k evaluation with bootstrap confidence bands
- Doubly-robust policy-value estimation for any proposed targeting rule
- Segment discovery reporting persuadables, sure things, lost causes and sleeping dogs
- Scoring service returning uplift plus a recommended action and its expected-value calculation

## Repository layout

```
src/learners/
src/evaluation/
src/policy/
service/
generator/
test/recovery/
```

## Build plan

1. Generator with a known ITE function including a sleeping-dog segment. Without it you cannot demonstrate the central point.
2. Fit a plain churn model first and evaluate it by Qini. Watch it underperform. Keep that comparison.
3. Uplift learners, then doubly-robust policy value.
4. Service and the expected-value calculation last - that calculation is what makes it a business tool.

## Testing strategy

Assert recovered uplift correlates with the generator's true ITE above a stated threshold. Assert the **policy-value estimator recovers the true policy value** on synthetic data. Assert the sleeping-dog segment is identified - a model that cannot find negative uplift will happily recommend contacting those customers.

Tests assert **correctness**, not merely that the code runs. A green suite on
this repo is a claim about behaviour under adversarial conditions; treat any
test that would pass against a deliberately broken implementation as a bug in
the test.

## Quality & safety layer

A fairness slice check on uplift across protected-proxy segments, plus a hard guardrail preventing any customer with negative estimated uplift from being targeted regardless of their churn score.

## Measurable outcome

> On a seeded synthetic book of 3,000 evaluated customers with a confounded historical campaign, contacting the top 30% by uplift adds +4.47 points of retention, while contacting the top 30% by churn score *loses* 0.23 points. Sleeping dogs, the customers the contact drives away, make up 37% of the churn-ranked list (336 of 900) and 4% of the uplift-ranked list (40 of 900). At an illustrative GBP 12 per contact and GBP 340 per retained subscriber, that is GBP +11,599 against GBP -4,376 net per 1,000 customers.

(The spec originally said "2.2x more customers per pound" and "the 8% of customers the contact drives away". Neither matches what is measured. The churn-ranked campaign's incremental retention is negative, so a per-pound ratio is not meaningful, and the planted sleeping-dog share is 20%. The claim above is the measured one.)

State it in these terms — business units, not technical ones — in your CV
bullet and in the first thirty seconds of describing the project.

## Measured results

From `python -m src.demo` (n=6000 log, 3000 train / 3000 evaluate, seed 7, deterministic; 16 s of CPU, 37 s wall-clock on an unloaded laptop):

| Metric | Value |
|---|---|
| Pearson r(T-learner uplift, true ITE) | +0.638 |
| Pearson r(transformed-outcome uplift, true ITE) | +0.630 |
| Pearson r(churn score, true ITE) | -0.143 |
| True value: do nothing / top-30% churn / top-30% uplift | 0.6136 / 0.6113 / 0.6583 |
| DR estimate (95% bootstrap CI), top-30% uplift | 0.6492 [0.6265, 0.6706] |
| Incremental retention, churn-ranked vs uplift-ranked | -0.23% vs +4.47% |
| Sleeping dogs on the 900-contact list, churn vs uplift | 336 vs 40 |
| Net value per 1,000 customers (GBP 12 contact, GBP 340 LTV) | -4,376 vs +11,599 |
| Qini-style area (DR incremental curve), uplift vs churn | 0.0479 vs 0.0164 (2.92x) |
| Mean estimated uplift, sleeping-dog segment (true -20%) | -10.0% |
| Individuals whose uplift CI covers zero (B=25 refits) | 43.4% |

Off-policy estimator RMSE over 5 independent logs, top-30% uplift policy:

| Scenario | DM | IPW | DR |
|---|---|---|---|
| both nuisance models sound | 0.93% | 0.56% | 0.52% |
| outcome model misspecified (intercept only) | 1.50% | 0.56% | 0.49% |
| propensity misspecified (assume 50/50) | 0.93% | 20.83% | 0.49% |
| both misspecified | 1.50% | 20.83% | 4.36% |

The suite (`tests/`) checks: uplift correlation with the planted ITE above 0.35 (the worst of a 10-seed sweep was 0.40); the sleeping-dog segment has the lowest mean estimated uplift, and it is negative; DR recovers the oracle policy value within 3 SE; IPW with a naive propensity is off by more than 10 SE; DR survives either single misspecification; and, averaged over 6 seeds, DR is biased when both are wrong.

### Limitations

- Synthetic data only. Criteo-UPLIFT and Hillstrom validation is not implemented.
- Learners are logistic-regression based (T-learner, transformed outcome). There is no causal forest or R-learner. The T-learner overstates lost-cause uplift (+13.6% estimated vs +1% true) because the arm models are linear in the features and the segment structure is not.
- Individual precision is limited: 43.4% of customers have a bootstrap uplift interval that covers zero. That interval comes from only 25 refits, which kept the demo within its time budget, so its percentile endpoints are coarse.
- Policy-value bootstrap CIs hold the fitted nuisance models fixed, so they understate uncertainty from nuisance estimation.
- A single seed cannot resolve DR bias below about 3 SE (0.017 at n=1500). Bias claims are tested across seeds for that reason. Even over 6 seeds, the direct method with an intercept-only outcome model showed a mean error of -0.87 points at 1.9 SE, so the suite makes no DM-bias claim.
- The expected-value calculation uses stated illustrative unit economics, not real ones. There is no FastAPI service; the demo prints the calculation.

## Interview questions this project answers

- **Why is a churn model the wrong tool for a retention campaign?**
- **What is a Qini curve?**
- **How do you evaluate a targeting policy you never ran?**

## What this deliberately is *not*

- Not another churn classifier - the churn model exists here only as the baseline it beats.
- Not an A/B testing platform, though it consumes experimental data where available.


## Run it now

```bash
python -m unittest discover -s tests -v   # the suite
python -m src.demo                        # the 60-second artefact
```

Requires Python 3.11+. The runnable core uses **only the standard
library** (including `sqlite3`), so there is nothing to install.

## Getting started

```bash
git clone https://github.com/SlateGitOrg/upliftpolicy.git
cd upliftpolicy
python -m unittest discover -s tests -v   # the suite
python -m src.demo                        # the 60-second artefact
```

Nothing to install - the runnable reference has no dependencies.

<details>
<summary>Target workflow for the full build (not yet implemented)</summary>

These commands describe the production stack this project grows into.
None of them work in this repository today.

```bash
# git clone <your-fork-url> upliftpolicy
# cd upliftpolicy
# pip install -e .
# python -m generator --customers 500000
# python -m src.learners fit --all
# python -m src.evaluation qini --compare-churn-baseline
# python -m src.policy value --rule top_k:0.3
# pytest test/recovery
```

</details>

Docker is supported but optional — every path above works on a plain
Windows/macOS/Linux laptop without a cloud account.

## Definition of done

- [ ] The differentiator above is implemented, and a test proves it
- [ ] The measurable outcome is produced by a command anyone can run
- [ ] `README` explains the one decision a generic version gets wrong
- [ ] CI runs the full suite on every push and is green on `main`
- [ ] A recruiter can see the headline artefact in under 60 seconds

## Licence

MIT — see [LICENSE](LICENSE).
