# Active temporal-semantic identification

Date: 2026-08-11

The earlier active-refresh attack asked:

> **Which stale/uncertain cached fact should the agent refresh now?**

The semantic-identifiability work creates a different active question:

> **Which probe would most quickly tell the agent what temporal rule the source obeys?**

Those are not the same uncertainty.

The implementation is:

- `temporal_validity_active.py` — probe acquisition scores;
- `experiments/active_semantic_identification.py` — equal-budget attack.

The main result is:

> **Probing where competing temporal semantics disagree identifies the source rule far faster than random refreshes or probing only near the current freshness threshold. Exact one-step semantic information gain helps most at very small budgets, but simple model disagreement nearly catches it by moderate budgets.**

This is ordinary active model discrimination / information-gain scheduling applied to temporal validity. It is not claimed as a new active-learning method.

## 1. Initial experience is exactly confounded

Each run begins with `200` audited observations at:

```text
1 second / event
```

so:

```text
world_age == event_age
```

The hidden source is equally likely in the benchmark to obey:

```text
world_hazard
or
event_hazard
```

Because the initial audits lie on the confounded line, both semantic models begin with essentially equal posterior mass:

```text
P(true semantic) ~= 0.50
```

No amount of repeating that same operating rhythm would identify the axis efficiently.

## 2. Probe opportunities

At each active step the agent receives a pool of candidate audit opportunities.

Candidates vary:

```text
event age: 1 .. 20 events
interaction tempo: 0.20 .. 2.50 seconds/event
```

so some opportunities deliberately separate the coordinates:

```text
many events / little wall time
or
few events / much wall time
```

Every strategy gets the same number of probes.

The probe outcome is one binary audit:

```text
cache still valid
or
cache stale
```

and the semantic posterior is updated.

## 3. Four schedules

### Random

Uniformly pick one candidate opportunity.

### Freshness uncertainty

Prefer model-averaged validity near the operational reuse threshold:

```text
P(valid) ~= 0.82
```

This is the natural continuation of the earlier uncertainty-triggered refresh idea.

### Semantic disagreement

Prefer opportunities with large weighted variance in predicted validity across semantic hypotheses:

\[
\sum_m w_m(p_m-\bar p)^2.
\]

This asks for a case on which the competing temporal theories make different predictions.

### Semantic information gain

Choose the opportunity with maximum expected one-step reduction in entropy over semantic class:

```text
world_hazard
event_hazard
until_change
```

Hazard-rate uncertainty is integrated inside each class.

This is the stronger acquisition attacker.

## 4. Why freshness uncertainty is not enough

A fact can be near the reuse boundary under **both** temporal models.

That makes it action-relevant but semantically uninformative.

Conversely, an opportunity can be far from the immediate action boundary yet strongly separate:

```text
world-time model says VALID
event-distance model says STALE
```

So:

> **state uncertainty and model uncertainty are different targets for active sensing.**

That distinction is the main conceptual result of this attack.

## 5. Equal-budget result

Twenty seeds for each hidden semantic (`world_hazard` and `event_hazard`), so `40` source runs per method.

Future evaluation uses a 50/50 mix of dense and sparse rate-shift cases.

### Zero probes

All methods are identical:

```text
true semantic posterior weight  .500
identified >.90                  .000
future OOD agreement             .808
future OOD utility               .649
bad stale reuse                  .070
```

### Two probes

| schedule | true semantic weight | >.90 identified | OOD agreement | OOD utility |
|---|---:|---:|---:|---:|
| random | .512 | .000 | .811 | .649 |
| freshness uncertainty | .520 | .000 | .816 | .651 |
| disagreement | .582 | .000 | .841 | .651 |
| **information gain** | **.600** | .000 | **.847** | **.653** |

Even two deliberately diagnostic observations start separating the semantic models.

## 6. Five probes

| schedule | true semantic weight | >.90 identified | OOD agreement | OOD utility | bad reuse |
|---|---:|---:|---:|---:|---:|
| random | .525 | .050 | .818 | .649 | .069 |
| freshness uncertainty | .537 | .000 | .823 | .651 | .063 |
| disagreement | .658 | .075 | .869 | .657 | .049 |
| **information gain** | **.708** | **.225** | **.884** | **.659** | **.040** |

At this budget exact information gain has a clear advantage.

## 7. Ten probes

| schedule | true semantic weight | >.90 identified | OOD agreement | OOD utility | bad reuse |
|---|---:|---:|---:|---:|---:|
| random | .578 | .050 | .839 | .652 | .060 |
| freshness uncertainty | .592 | .075 | .843 | .654 | .051 |
| disagreement | .762 | .450 | .904 | .660 | .044 |
| **information gain** | **.805** | **.575** | **.920** | **.662** | **.037** |

This is the clearest separation.

The old active-refresh heuristic:

```text
probe where current validity is uncertain
```

is useful but not targeted enough for identifying the temporal rule.

## 8. Twenty probes

| schedule | true semantic weight | >.90 identified | OOD agreement | OOD utility |
|---|---:|---:|---:|---:|
| random | .630 | .250 | .853 | .652 |
| freshness uncertainty | .648 | .150 | .866 | .656 |
| disagreement | .911 | **.775** | .948 | .667 |
| information gain | **.916** | .750 | **.948** | **.667** |

Now simple disagreement has essentially caught the exact information-gain scheduler.

That is important for the engineering conclusion.

The expensive/fancy acquisition rule does not keep a decisive advantage.

## 9. Forty probes

| schedule | true semantic weight | >.90 identified | OOD agreement | OOD utility | bad reuse |
|---|---:|---:|---:|---:|---:|
| random | .735 | .375 | .887 | .657 | .042 |
| freshness uncertainty | .772 | .300 | .913 | .663 | .029 |
| disagreement | **.975** | **.950** | .968 | .669 | .015 |
| information gain | .974 | .925 | **.972** | **.670** | **.014** |

Again, disagreement and exact information gain are functionally close.

## 10. What kind of probes do the semantic schedulers choose?

A ten-probe diagnostic looked at the ratio:

\[
\frac{world\ age}{event\ age}
\]

Random opportunities had a median ratio around:

```text
1.19 seconds/event
```

and relatively modest coordinate contrast.

The semantic schedulers deliberately moved toward cases where the axes disagree strongly.

`semantic_disagreement` selected mostly high-tempo contrasts, with median ratio around:

```text
2.42 seconds/event
```

Information gain used both sides of the contrast more aggressively; its mean absolute log-ratio was larger than random/freshness scheduling.

The qualitative rule is simpler than the exact acquisition formula:

> **If you are unsure whether a source ages in seconds or events, observe it where seconds and events disagree.**

That is almost embarrassingly obvious once stated, but the earlier generic active-refresh logic did not distinguish this from state uncertainty.

## 11. Practical API

`temporal_validity_active.py` provides:

```python
freshness_uncertainty_score(...)
semantic_disagreement_score(...)
semantic_information_gain_score(...)
choose_probe(...)
```

A candidate probe is represented by:

```python
ProbeOpportunity(
    coordinates=TemporalCoordinates(...),
    payload=...
)
```

The runtime can therefore choose between:

```text
immediate action uncertainty
or
longer-term semantic information
```

without changing the validity model itself.

## 12. What survives and what dies

### Survives

- active refresh can be used to learn *semantics*, not just current state;
- the probe should separate plausible temporal coordinates;
- a small number of diagnostic probes can dramatically accelerate identification;
- improved semantic identification reduces future OOD stale reuse.

### Dies / is demoted

- "uncertainty-triggered refresh" as one universal concept;
- freshness uncertainty as the best acquisition signal when model semantics are unresolved;
- exact information gain as obviously necessary at moderate probe budgets.

A simple disagreement score is a strong boring attacker.

## 13. New runtime picture

The practical loop now becomes:

```text
EVIDENCE
    world age
    event age
    invalidation
        |
        v
SEMANTIC POSTERIOR
        |
        +-----------------------------+
        |                             |
        v                             v
P(valid current item)         semantic disagreement
        |                             |
        v                             v
reuse / refresh              choose diagnostic probe
        |                             |
        +-------------+---------------+
                      |
                      v
              audited outcome
                      |
                      v
              update posterior
```

This finally reconnects active sensing to the temporal-identifiability result in a nontrivial way.

## 14. Next attack

The missing piece is **budget arbitration**.

A real agent does not have separate free budgets for:

```text
refreshing because the current answer is risky
and
probing because the semantic model is uncertain.
```

They compete for the same tool calls.

So the next useful benchmark should make one scheduler allocate a fixed refresh budget between:

1. immediate expected utility;
2. long-term semantic information gain.

If semantic exploration only helps on long horizons and burns too many current tool calls, its practical value may disappear.

That is the next place to attack before treating active semantic identification as a product feature.
