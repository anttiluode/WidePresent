# Temporal semantic drift attack

Date: 2026-08-11

The previous attack showed that uncertainty over temporal semantics should be carried into the reuse/refresh decision while the semantics are underidentified.

That creates an obvious new failure mode:

> **What happens after the runtime becomes highly confident in the correct temporal rule and the source later changes that rule?**

A posterior that is allowed to become certain but is never allowed to forget can turn yesterday's correct temporal semantics into today's dangerous stale policy.

The implementation is:

- `temporal_validity_online.py` — cumulative / rolling online semantic posterior;
- `experiments/semantic_drift_attack.py` — prequential drift attack.

The result is again deliberately boring:

> **A fixed rolling evidence window adapts to temporal-semantic drift far better than a frozen or forever-cumulative posterior, and approaches an oracle reset without knowing the switch time. An extra confidence-triggered reset detector does not cleanly beat the simpler rolling baseline.**

This is ordinary nonstationary estimation, not a new temporal learning algorithm.

## 1. Drift world

One anonymous source is observed online.

Interaction tempo is rate-diverse throughout:

```text
0.40 .. 1.60 seconds / event
```

so world age and event distance are identifiable from recent evidence.

The hidden source rule is:

```text
before audit 1000
    world_hazard
    P(valid) = exp(-lambda * world_age)

after audit 1000
    event_hazard
    P(valid) = exp(-lambda * event_age)
```

The hazard scale itself stays the same.

Only the **coordinate in which validity changes** switches.

Each online case is evaluated prequentially:

1. runtime predicts `P(valid)`;
2. runtime chooses REUSE / REFRESH using the usual utility threshold;
3. the noisy validity outcome is revealed;
4. the learner updates.

So the post-drift scores measure real adaptation lag rather than fitting after seeing the entire changed segment.

## 2. Strategies

### Frozen

Fit on the first `300` audits and never learn again.

This is an intentionally weak stationary baseline.

### Cumulative

Keep updating but never forget any audit.

This is the important attacker because ordinary Bayesian learning is often described as solving uncertainty automatically.

It does—if the hidden model is stationary.

After drift, the large body of high-confidence pre-drift evidence creates semantic inertia.

### Rolling

Keep only the most recent `240` audits in the semantic posterior.

No drift time or reset event is supplied.

### Oracle reset

Reset cumulative evidence exactly at audit `1000`.

This knows the hidden switch time and is therefore an upper adaptation reference, not a deployable policy.

### Sentinel reset

Exploratory extra machinery:

```text
primary cumulative posterior
+
recent rolling posterior
```

If recent MAP semantics disagree with primary MAP semantics and recent confidence exceeds `0.95`, reset the primary learner to the recent audit window.

This was included to test whether an explicit semantic-change detector earns its complexity.

## 3. Twenty-seed result

Default evaluation:

```text
20 seeds
2000 audits per stream
300 warm-up audits
semantic switch at 1000
rolling window 240
```

### Before drift

| learner | agreement | utility | bad reuse | refresh |
|---|---:|---:|---:|---:|
| frozen | 0.949 | 0.653 | 0.026 | 0.535 |
| cumulative | **0.963** | **0.654** | 0.026 | 0.523 |
| rolling | 0.935 | 0.652 | 0.038 | 0.526 |
| oracle-reset track | **0.963** | **0.654** | 0.026 | 0.523 |

The rolling learner pays a small stable-regime price because it deliberately throws old information away.

That is the cost of remaining adaptable.

## 4. After the semantic switch

Across the entire post-drift half:

| learner | agreement | utility | bad reuse | refresh |
|---|---:|---:|---:|---:|
| frozen | 0.869 | 0.6280 | 0.084 | 0.530 |
| cumulative forever | 0.875 | 0.6281 | 0.080 | 0.531 |
| **rolling 240** | **0.937** | **0.6400** | 0.030 | 0.569 |
| oracle reset | 0.948 | 0.6408 | **0.023** | 0.573 |

The cumulative learner barely improves over freezing.

That is the main result.

The problem is not that Bayesian uncertainty failed initially.

The problem is that after hundreds of stable observations the posterior was appropriately confident in a model that later stopped being true.

Without forgetting, old evidence becomes a form of **semantic inertia**.

## 5. Early versus late adaptation

First `200` audits after drift:

| learner | agreement | utility | bad reuse |
|---|---:|---:|---:|
| frozen | 0.872 | 0.631 | 0.083 |
| cumulative | 0.856 | 0.628 | 0.094 |
| rolling | **0.907** | **0.637** | 0.051 |
| oracle reset | 0.892 | 0.634 | 0.066 |

The rolling learner is actually better than the oracle reset very early because it enters the switch with a usable recent posterior instead of being reset to complete ignorance.

Late period, beginning `300` audits after drift:

| learner | agreement | utility | bad reuse |
|---|---:|---:|---:|
| frozen | 0.868 | 0.627 | 0.085 |
| cumulative | 0.882 | 0.629 | 0.076 |
| rolling | 0.946 | 0.641 | 0.025 |
| oracle reset | **0.966** | **0.643** | **0.011** |

Once enough post-switch evidence accumulates, the oracle reset retains an advantage because it has no contamination from pre-switch data.

The rolling learner gets most of that benefit without being told when the change happened.

## 6. Posterior adaptation delay

Define adaptation as:

```text
P(event_hazard) > 0.90
```

after the hidden switch.

### Frozen

```text
success 0 / 20
```

### Cumulative

```text
success 7 / 20
mean delay among successes ~743 audits
```

Even after roughly another thousand observations, most cumulative learners never become 90% confident in the new semantic axis.

### Rolling 240

```text
success 20 / 20
mean delay ~181 audits
```

### Oracle reset

```text
success 20 / 20
mean delay ~140 audits
```

That difference is the operational cost of retaining old pre-switch evidence.

## 7. Extra reset detector did not earn itself

The `sentinel` variant uses a recent rolling posterior as a detector but makes predictions from a cumulative primary posterior until recent evidence strongly disagrees.

With recent confidence threshold `0.95`, the exploratory twenty-seed run was approximately:

```text
post-drift utility    0.639
late utility          0.642
```

compared with rolling:

```text
post-drift utility    0.640
late utility          0.641
```

The sentinel gains somewhat in the late stable segment after reset, but loses during the detection lag.

It also produced false pre-drift reset behavior in `2/20` seeds at that threshold.

Changing the threshold trades false resets against detection delay.

There was no clean enough win to justify adding this machinery to the practical runtime.

So the current default remains:

> **simple rolling evidence beats a hand-built semantic-change sentinel unless a later task proves the detector is worth its extra thresholds.**

This negative result should prevent the project from turning every failure into another module.

## 8. What this adds to the three-stage decomposition

The runtime pipeline is now:

```text
1. REPRESENTATION
   preserve temporal coordinates

2. IDENTIFICATION
   infer which coordinate governs validity

3. DECISION
   reuse / refresh under utility
```

The drift result adds a fourth requirement:

```text
4. NONSTATIONARITY
   do not let old semantic certainty become permanent
```

More explicitly:

```text
candidate temporal coordinates
        |
        v
recent audited validity outcomes
        |
        v
semantic posterior
        |
        v
P(valid now)
        |
        v
utility-aware action
        |
        v
new outcome
        |
        +------> update recent semantic evidence
```

The loop matters because the source itself may change.

## 9. Practical implication

`temporal_validity_online.py` therefore supports:

```python
OnlineSemanticAccumulator(window=None)
```

for a stationary cumulative posterior, and:

```python
OnlineSemanticAccumulator(window=240)
```

for a rolling posterior.

The window is not metaphysically a "wide present."

It is an ordinary adaptation timescale.

Its size expresses a bias-variance / stability-plasticity tradeoff:

```text
long window
    stable estimates
    slow semantic adaptation

short window
    rapid adaptation
    noisier semantic inference
```

That is established nonstationary-learning territory.

## 10. What survived

The research line now supports a fairly coherent engineering rule:

> **Temporal validity should be represented as an uncertain, source-specific and potentially nonstationary relation between evidence and current context—not as one permanent scalar freshness rule.**

Operationally:

- retain world age separately from arrival age;
- retain event/structural age when relevant;
- include explicit version/change evidence;
- infer source semantics only when experience identifies them;
- average over plausible semantics while ambiguity remains;
- forget old semantic evidence when the source is allowed to drift;
- refresh actively when posterior validity falls below the utility threshold.

None of those steps requires oscillatory geometry or a special neural temporal substrate.

## 11. Next attack

The next obvious attacker is **active identification**.

So far the runtime passively waits for audited outcomes to distinguish candidate semantics.

But an agent can choose *when* to refresh or probe.

If the posterior is split between:

```text
world-time semantics
and
event-distance semantics
```

some refresh timings are much more informative than others.

For example, a probe after:

```text
many events but little wall time
```

or:

```text
much wall time but few events
```

can distinguish the hypotheses quickly.

The next useful question is therefore:

> **Can the same active refresh budget be scheduled to maximize semantic information as well as immediate freshness, and does that reduce future tool calls or stale reuse?**

That connects the earlier active-sensing result back to the now much cleaner semantic-identification problem.
