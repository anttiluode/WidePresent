# Temporal semantic uncertainty attack

Date: 2026-08-11

The earlier WidePresent attacks separated three problems:

```text
REPRESENTATION
    preserve candidate temporal coordinates

IDENTIFICATION
    determine which coordinate actually governs source validity

DECISION
    reuse or refresh under asymmetric cost
```

`temporal_validity.py` handles the first and third when source semantics are already known.

This attack asks what the runtime should do in the middle period when semantics are **not yet identified**.

The implementation is:

- `temporal_validity_learning.py` — reusable semantic posterior;
- `experiments/semantic_uncertainty_attack.py` — matched attack.

The main result is:

> **When temporal semantics are underidentified, carrying semantic uncertainty into the reuse/refresh decision beats brittle hard semantic selection in expected utility. A conservative worst-case policy can nearly eliminate dangerous stale reuse, but pays for that safety with extra refreshes. Once rate-diverse experience identifies the semantics, all three policies collapse to the same ordinary source-specific validity model.**

This is Bayesian model averaging / robust decision-making applied to the temporal-validity problem. It is not presented as a new statistical method.

## 1. Candidate semantic models

For each source, the learner considers:

```text
world_hazard
    P(valid) = exp(-lambda * world_age_seconds)

event_hazard
    P(valid) = exp(-lambda * event_age)

until_change
    P(valid) = 1 unless explicit invalidation occurred
```

The two hazard models integrate over a log-spaced decay-rate grid.

Each source therefore has posterior uncertainty over:

```text
which temporal coordinate matters
and
what its hazard rate is
```

The source name itself is not used to infer the semantic model.

## 2. Three decision strategies

### MAP / hard selector

Choose the highest-posterior semantic class and behave as if the alternatives do not exist.

This matches the earlier `hazard_selector` logic.

### Bayesian model average

Predict validity by averaging over semantic hypotheses and their nuisance hazard rates:

\[
P(valid\mid D)
=
\sum_m P(m\mid D)
\int P(valid\mid m,\lambda)P(\lambda\mid m,D)d\lambda.
\]

Then apply the ordinary reuse utility threshold.

This is the expected-utility attacker.

### Robust worst-case

Take the minimum predicted validity among semantic hypotheses with at least `0.10` posterior mass.

This deliberately asks:

> if every still-plausible temporal interpretation must agree that reuse is safe, how much stale reuse can we eliminate?

It is intentionally conservative and is not claimed to be Bayes-optimal.

## 3. Environment

The attack uses the same operational semantics as `EVENT_AGENT_AGE_PLANE_ATTACK.md`:

```text
weather
    stochastic validity hazard in world time

discourse
    stochastic validity hazard in intervening event count

state
    deterministic validity until explicit invalidation
```

Training labels are noisy realized validity outcomes.

The decision utility remains:

```text
valid reuse   +1.00
stale reuse   -1.50
refresh       +0.55
```

so the reuse threshold is:

\[
P(valid) \ge 0.82.
\]

The learner is not given the hidden true semantic class.

## 4. Narrow experience leaves the semantic posterior unresolved

Training tempo:

```text
0.95 .. 1.05 seconds / event
```

Five-seed posterior weights for `weather` were approximately:

```text
seed 0  world .316  event .684
seed 1  world .553  event .447
seed 2  world .807  event .193
seed 3  world .535  event .465
seed 4  world .492  event .508
```

For `discourse`:

```text
seed 0  world .242  event .758
seed 1  world .767  event .233
seed 2  world .370  event .630
seed 3  world .709  event .291
seed 4  world .494  event .506
```

The `state` source assigns essentially all posterior mass to `until_change` in every seed.

This is exactly the instability predicted by the earlier identifiability result.

A hard selector can therefore choose the wrong coordinate even with thousands of audited outcomes, because the experienced tempo does not separate the candidate explanations strongly enough.

## 5. Narrow-training OOD result

Five seeds, `6000` training cases, `3500` test cases per regime.

The main OOD aggregate averages:

```text
dense
sparse
dense + long delivery delay
```

| strategy | action agreement | utility | dangerous stale reuse | refresh rate |
|---|---:|---:|---:|---:|
| MAP / hard selector | **0.877** | 0.703 | 0.059 | 0.429 |
| Bayesian average | 0.857 | **0.710** | 0.055 | 0.456 |
| robust worst-case | 0.846 | 0.706 | **0.0005** | 0.576 |
| oracle semantic axis | 0.991 | 0.729 | 0.002 | 0.428 |

This is an important distinction between **classification accuracy** and **decision quality**.

The MAP selector has the highest action agreement among the uncertainty-aware learned policies.

But the Bayesian average has higher expected utility.

Why?

Because under semantic ambiguity, a wrong hard commitment can produce stale reuse, and stale reuse costs much more than an unnecessary refresh.

The Bayesian average becomes slightly more conservative exactly where its semantic hypotheses disagree.

## 6. Dense rate shift

Under dense interaction:

| strategy | agreement | utility | bad reuse | refresh |
|---|---:|---:|---:|---:|
| MAP | 0.840 | 0.715 | 0.077 | 0.346 |
| Bayesian average | 0.820 | **0.724** | 0.072 | 0.375 |
| robust | 0.805 | 0.719 | **0.000** | 0.534 |
| oracle axis | 0.991 | 0.750 | 0.002 | 0.344 |

The model average sacrifices nominal action agreement for better utility.

The robust policy eliminates stale reuse almost completely but refreshes more than half the time.

## 7. Sparse rate shift

Under sparse interaction:

| strategy | agreement | utility | bad reuse | refresh |
|---|---:|---:|---:|---:|
| MAP | **0.942** | 0.678 | 0.027 | 0.597 |
| Bayesian average | 0.925 | **0.680** | 0.024 | 0.620 |
| robust | 0.926 | 0.679 | **0.002** | 0.665 |
| oracle axis | 0.992 | 0.686 | 0.002 | 0.599 |

The gain is smaller because sparse interaction already pushes many cases toward refresh.

## 8. Seed brittleness is reduced

Aggregate OOD utility by seed gives another useful view.

### MAP selector

```text
mean  0.7033
std   0.0132
min   0.6908
max   0.7274
```

### Bayesian average

```text
mean  0.7103
std   0.0084
min   0.6972
max   0.7235
```

### Robust worst-case

```text
mean  0.7060
std   0.0007
min   0.7050
max   0.7072
```

So the robust policy is not best on mean utility, but it is dramatically less sensitive to which ambiguous semantic explanation happened to win a training seed.

That is a useful safety/robustness tradeoff rather than a generic accuracy win.

## 9. Wide experience makes the disagreement disappear

Training tempo:

```text
0.40 .. 1.60 seconds / event
```

Across all five seeds the semantic posterior becomes essentially:

```text
weather
    world_hazard  ~1.0

discourse
    event_hazard  ~1.0

state
    until_change  ~1.0
```

Then:

| strategy | OOD agreement | OOD utility | bad reuse | refresh |
|---|---:|---:|---:|---:|
| MAP | 0.990 | 0.729 | 0.003 | 0.427 |
| Bayesian average | 0.990 | 0.729 | 0.003 | 0.427 |
| robust | 0.990 | 0.729 | 0.003 | 0.427 |
| oracle axis | 0.990 | 0.729 | 0.003 | 0.427 |

The uncertainty machinery becomes irrelevant once experience identifies the semantics.

This is exactly what one wants from it.

It is not a permanent second architecture.

It is a temporary response to underidentification.

## 10. What this changes in the runtime

The practical pipeline is now:

```text
EVIDENCE
    world age
    knowledge age
    event age
    explicit invalidation
        |
        v
SEMANTIC POSTERIOR
    P(world-time semantics)
    P(event-distance semantics)
    P(until-change semantics)
        |
        v
POSTERIOR P(valid now)
        |
        v
UTILITY LAYER
        |
        +---- REUSE
        |
        +---- REFRESH
```

When the posterior is concentrated, this reduces to the ordinary source-specific model already favored by the earlier attack.

When the posterior is diffuse, it avoids pretending that an arbitrary MAP choice is known truth.

## 11. What survives and what dies

### Survives

- preserve multiple temporal coordinates while semantics are unresolved;
- represent uncertainty over *which coordinate matters*;
- propagate that uncertainty into the reuse/refresh decision;
- use rate-diverse experience to collapse the ambiguity when possible;
- tune conservatism according to stale-reuse versus refresh cost.

### Dies

- hard source-semantic selection as the only runtime choice;
- interpreting one fitted temporal axis as certainty when competing axes have comparable likelihood;
- using action accuracy as the only metric when stale reuse and refresh have asymmetric costs;
- keeping ambiguity machinery after the semantic posterior has already collapsed.

## 12. Relation to WidePresent

This result further demotes the original architectural story.

The strongest current object is not:

```text
wide temporal neural layer
```

but:

```text
temporal coordinates
+ semantic uncertainty
+ source validity model
+ utility-aware active refresh
```

The important moving-present question has become a systems question:

> **What temporal interpretation of this evidence is still plausible, and is reuse safe under that uncertainty?**

That is a cleaner consequence of the identifiability work than forcing the runtime to choose one meaning of age too early.

## 13. Next attacker

The obvious next attack is calibration under semantic drift.

So far each source has one stable hidden temporal rule.

A real source can change behavior:

```text
cache TTL changes
API freshness policy changes
conversation workflow changes
source starts emitting explicit versions
```

A posterior that becomes certain and stays certain forever would then be dangerous.

The next useful question is therefore:

> **Can the runtime detect that its learned temporal semantics have stopped predicting validity, reopen semantic uncertainty, and trigger refresh/relearning without a hand-written reset?**

That is more realistic than adding another coordinate or neural layer.
