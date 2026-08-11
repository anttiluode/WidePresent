# Event-agent age-plane attack

Date: 2026-08-11

This is the first WidePresent age-plane test that leaves the hand-designed fading-sum toy and asks an operational agent question:

> **Given a cached piece of evidence, should the agent reuse it or refresh the tool?**

The experiment is:

`experiments/event_agent_age_plane_attack.py`

The result is mixed in exactly the useful way:

- simple timestamp/position policies can look excellent IID while relying on the wrong temporal proxy;
- a content-conditioned `(Δt, Δn)` representation improves robustness under rate shift;
- once training contains enough episode-level rate diversity, an ordinary per-source survival/hazard model identifies the correct temporal semantics and becomes the strongest learned policy;
- arrival recency alone fails badly when tool results are delayed.

So this is **not an architecture win for WidePresent**.

It is the first evidence that the age-coordinate diagnosis survives in an agent-like reuse/refresh problem, and also that boring survival analysis remains the primary attacker.

## 1. The event-agent world

Every case ends in the same decision:

```text
REUSE cached evidence
or
REFRESH the tool
```

The three evidence types have deliberately different temporal semantics.

### Weather-like evidence

Validity decays with elapsed wall time:

\[
P(\text{valid})=e^{-\lambda \Delta t}.
\]

### Discourse/reference evidence

Validity decays with intervening event count:

\[
P(\text{valid})=e^{-\lambda \Delta n}.
\]

### State/reservation evidence

It does **not** become stale merely because time or events pass.

It remains valid until an explicit invalidation/state-change event is observed.

This is important because it prevents the benchmark from reducing to "everything gets older, just with different constants."

## 2. Asynchronous evidence

Every cached tool result has two times:

```text
valid/world time       when the result was true / generated
arrival/knowledge time when the result reached the agent
```

The delivery delay is random.

Therefore a result can be:

```text
arrived very recently
but
already old in world time
```

The `long_delay` and `dense_long` regimes make this mismatch especially large.

## 3. Operational utility

Correct reuse is best:

```text
valid reuse   +1.00
refresh       +0.55
stale reuse   -1.50
```

So refresh is safe but costs latency/tool budget, while stale reuse is dangerous.

The Bayes-optimal reuse threshold is

\[
P(\text{valid}) \ge 0.82.
\]

Learned policies are **not** given this oracle probability.

Training supplies noisy outcomes only:

```text
was this cached fact actually still correct?
```

Each model chooses its operational reuse threshold on a held-out part of the training data by observed utility.

Evaluation reports:

- action agreement with the Bayes-optimal reuse/refresh decision;
- expected operational utility;
- refresh rate;
- `bad_reuse`: fraction of cases where the model reuses evidence that the oracle says should be refreshed.

## 4. Representations / attackers

### `arrival_logit`

Source type + arrival age + invalidation flag.

This is the intentionally weak "it arrived recently, therefore it is fresh" baseline.

### `timestamp_logit`

Source type + valid/world age + arrival age + invalidation.

No structural/event-distance coordinate.

### `position_logit`

Source type + event distance + invalidation.

No wall-time coordinate.

### `raw_both_logit`

Both time coordinates are present, but one linear model must share their slopes across source types.

### `age_plane_logit`

Explicit source-conditioned interactions:

```text
source x valid/world age
source x event distance
source x arrival age
source x invalidation
```

This is the minimal age-plane representation under test.

It is not claimed as novel.

### `gbdt_both`

A generic nonlinear boosted-tree attacker receiving the same raw information.

### `hazard_selector`

The strongest boring semantic attacker.

For weather and discourse separately it fits two exponential survival candidates:

```text
hazard in seconds
hazard in event count
```

and chooses the axis with the larger training likelihood.

State evidence uses the explicit invalidation signal.

This is ordinary survival-style modeling, deliberately designed to kill any claim that the age-plane needs a special neural architecture.

## 5. Rate manipulation

The key manipulation is **episode-level interaction tempo**.

Each episode draws one base inter-event gap; individual events jitter slightly around it.

This matters. If every gap were independently redrawn from a wide interval, a long history would average back toward one common rate and erase the rate variation needed to identify temporal yoking.

Two training conditions are compared:

```text
narrow: 0.95..1.05 seconds/event
wide:   0.40..1.60 seconds/event
```

Test regimes:

```text
IID         0.95..1.05 s/event
dense       0.25..0.45 s/event
sparse      1.80..2.40 s/event
long_delay  nominal rate, much longer tool latency
dense_long  dense rate + long tool latency
```

The evidence semantics never change.

Only interaction rate and delivery delay change.

## 6. Five-seed result: narrow-rate training

`6000` training cases and `3500` cases per test regime per seed.

### Dense rate shift

| model | action agreement | utility | bad reuse |
|---|---:|---:|---:|
| timestamp only | 0.809 | 0.702 | 0.165 |
| position only | 0.796 | **0.717** | **0.004** |
| raw both | 0.791 | 0.718 | 0.012 |
| age plane | 0.801 | 0.720 | 0.016 |
| boosted tree | 0.811 | 0.713 | 0.109 |
| hazard selector | **0.866** | **0.725** | 0.059 |
| oracle | 1.000 | 0.750 | 0.000 |

### Sparse rate shift

| model | action agreement | utility | bad reuse |
|---|---:|---:|---:|
| timestamp only | 0.905 | 0.672 | **0.003** |
| position only | 0.917 | 0.667 | 0.075 |
| raw both | 0.855 | 0.640 | 0.111 |
| age plane | 0.903 | 0.668 | 0.068 |
| boosted tree | 0.900 | 0.672 | 0.012 |
| hazard selector | **0.939** | **0.678** | 0.010 |
| oracle | 1.000 | 0.684 | 0.000 |

The important fact is not that one representation dominates everything.

It is that all of these models are near the top IID, yet their OOD error structure differs sharply.

At narrow training rates the source-specific hazard selector still sometimes picks the wrong axis. Across the five seeds it selected combinations such as:

```text
weather -> seconds, discourse -> events     (correct)
weather -> seconds, discourse -> seconds    (wrong discourse axis)
weather -> events,  discourse -> events     (wrong weather axis)
```

That is the identifiability problem appearing in an agent task rather than in the exponential proof toy.

## 7. Five-seed result: wide-rate training

The same models are trained with episode tempos ranging from `0.40..1.60 s/event`.

Now the hazard selector identifies the correct semantics in **all five seeds**:

```text
weather   -> seconds
discourse -> events
state     -> explicit invalidation
```

### Dense rate shift

| model | action agreement | utility | bad reuse |
|---|---:|---:|---:|
| timestamp only | 0.824 | 0.708 | 0.151 |
| position only | 0.796 | 0.718 | **0.004** |
| raw both | 0.808 | 0.722 | 0.018 |
| age plane | **0.899** | 0.739 | 0.009 |
| boosted tree | 0.826 | 0.720 | 0.106 |
| **hazard selector** | **0.953** | **0.747** | 0.015 |
| oracle | 1.000 | 0.750 | 0.000 |

### Sparse rate shift

| model | action agreement | utility | bad reuse |
|---|---:|---:|---:|
| timestamp only | 0.907 | 0.674 | **0.003** |
| position only | 0.913 | 0.666 | 0.077 |
| raw both | 0.923 | 0.676 | 0.038 |
| age plane | 0.923 | 0.674 | 0.064 |
| boosted tree | 0.910 | 0.670 | 0.037 |
| **hazard selector** | **0.957** | **0.682** | 0.021 |
| oracle | 1.000 | 0.684 | 0.000 |

### Dense + long tool delay

| model | action agreement | utility | bad reuse |
|---|---:|---:|---:|
| arrival only | 0.808 | 0.701 | 0.174 |
| timestamp only | 0.823 | 0.717 | 0.126 |
| position only | 0.795 | 0.719 | **0.003** |
| raw both | 0.804 | 0.722 | 0.014 |
| age plane | **0.890** | 0.740 | 0.008 |
| boosted tree | 0.820 | 0.718 | 0.109 |
| **hazard selector** | **0.949** | **0.748** | 0.018 |
| oracle | 1.000 | 0.751 | 0.000 |

## 8. Arrival time gets explicitly killed

Long-delay evaluation isolates the old provenance problem.

With wide-rate training:

```text
arrival-only action agreement  0.856
arrival-only utility           0.671

valid timestamp action         0.927
valid timestamp utility        0.695
```

The result is unsurprising but operationally important:

> **recently received does not mean recently valid.**

The age-plane branch did not create this fact; it simply continues to respect it.

## 9. What survived

### A. Temporal semantics really can be content/source dependent

The same generic action, `reuse or refresh`, requires three different temporal rules in one environment:

```text
weather   -> wall-time hazard
discourse -> event-distance hazard
state     -> explicit change, no age decay
```

A universal scalar recency is therefore the wrong abstraction for this environment.

### B. IID success can hide the wrong temporal cause

A model can be excellent at the nominal rate while using seconds as a proxy for event count or vice versa.

The failure appears only when interaction tempo changes.

This is the earlier yoking-identifiability result in agent form.

### C. Explicit age-plane interactions help a cheap learner

With wide training-rate diversity, the source-conditioned linear age-plane representation is substantially more robust on dense OOD than:

- timestamp-only logistic regression;
- event-position-only logistic regression;
- the same raw features without source/age interactions;
- the generic boosted tree used here.

That is a useful representation result.

It is not an architecture novelty claim.

### D. Standard hazard modeling is stronger once semantics are identifiable

Once the training distribution contains enough rate diversity, the per-source survival model selects the correct axis in all five seeds and beats the age-plane linear policy on the important OOD regimes.

This is the main kill.

A competent agent runtime may not need a mysterious temporal substrate. It may need explicit temporal coordinates plus ordinary source-specific validity models.

## 10. What did not survive

The benchmark does **not** support:

- a special dual-clock neural module;
- oscillatory or geometric temporal dynamics;
- one universal "present width";
- arrival time as sufficient freshness;
- the claim that simply presenting both ages to a generic nonlinear model guarantees the correct invariant.

The boosted tree sees the same raw coordinates and remains brittle under rate shift because the nominal correlations still permit proxy solutions.

## 11. Revised WidePresent object

The smallest useful object now looks less like a neural layer and more like an agent-runtime contract:

```text
EVIDENCE
    content/source identity
    valid/world age Δt
    structural/event age Δn
    arrival/knowledge age when needed
    explicit invalidation/change markers when available
        |
        v
SOURCE/CONTENT-SPECIFIC VALIDITY MODEL
        |
        v
P(still valid now)
        |
        +---- high enough ----> REUSE
        |
        +---- too uncertain --> REFRESH
```

The age plane is useful because it refuses to collapse temporal coordinates before the validity model has earned the right to do so.

## 12. Strongest interpretation

The best sentence from this attack is now:

> **Temporal age is not a property of evidence until you specify the coordinate in which relevance changes.**

For some evidence that coordinate is seconds.

For some it is intervening structure.

For some age is irrelevant and only a state-change event matters.

The empirical question is therefore not "how wide is the present?"

It is:

> **What transformation makes this kind of evidence cease to be usable, and has the agent seen enough variation to identify that transformation rather than a correlated proxy?**

## 13. Next attack

Do not make another synthetic hazard world immediately.

The next useful step should use a **language/tool agent** that receives textual histories and must choose tool reuse versus refresh under manipulated interaction tempo.

The benchmark should preserve the same hidden semantic distinction:

- some tool outputs expire with wall time;
- some references expire with intervening conversational structure;
- some facts remain valid until an explicit change event.

Then compare:

1. raw timestamped history;
2. timestamp + message position;
3. deterministic age-plane side channel;
4. source-specific hazard/validity resolver outside the LLM;
5. the LLM alone with no temporal preprocessing.

If the external deterministic resolver wins, that may be the useful product: **do not ask the language model to rediscover temporal semantics that the runtime can represent and calibrate explicitly.**
