# Temporal Validity Runtime

Date: 2026-08-11

The WidePresent attacks have repeatedly reduced the problem.

The current practical object is not a wave substrate, a fixed temporal window, or a special recurrent layer.

It is a small runtime that separates:

```text
REPRESENTATION
    preserve temporal coordinates

VALIDITY SEMANTICS
    map those coordinates to P(still usable now)

DECISION
    trade stale-reuse risk against refresh cost
```

The implementation is:

`temporal_validity.py`

with a smoke/demo in:

`experiments/temporal_validity_runtime_demo.py`

## 1. Evidence carries more than one age

A cached evidence item stores:

```text
source
key
value

valid_time
    when the value was true / observed in the world

known_time
    when the agent learned or received it

known_structural_index
    conversation/event index when the agent learned it

source_version
    optional change epoch/version
```

At decision time the runtime derives:

```text
world_age_seconds
knowledge_age_seconds
event_age
invalidated
```

This keeps the coordinates separate until a source-specific validity model decides which ones matter.

## 2. Why valid time and known time are separate

A result can be:

```text
received 4 seconds ago
but
observed 8.5 seconds ago
```

For a source whose freshness depends on world observation time, arrival recency is the wrong coordinate.

The demo contains exactly this pair:

```text
same knowledge age = 4.0 s

world age 7.5 s -> REUSE
world age 8.5 s -> REFRESH
```

This is the operational form of the earlier provenance result:

> recently received does not imply recently valid.

## 3. Built-in validity models

### `WorldTimeTTL`

Hard validity in world/source time.

```python
WorldTimeTTL(8.0)
```

Meaning:

```text
valid while world_age <= 8 seconds
```

### `EventDistanceTTL`

Hard validity in structural/event distance.

```python
EventDistanceTTL(8)
```

Meaning:

```text
valid while event_age <= 8
```

Wall-clock time is irrelevant.

### `UntilChange`

Evidence remains valid indefinitely until the source version/change epoch differs.

This supports facts such as:

```text
reservation state
configuration state
inventory snapshot with explicit invalidation
```

that do not become stale merely because time passes.

### `ExponentialAgePlane`

Generic probabilistic fading:

\[
P(valid)
=
\exp(-a\Delta t-b\Delta n).
\]

This is included as a compact fallback/model family from the age-plane experiments.

It is not claimed as novel.

## 4. Utility-aware action policy

The runtime does not equate:

```text
P(valid) > 0.5
```

with:

```text
safe to reuse.
```

The action threshold depends on utility.

For the benchmark utilities:

```text
valid reuse   +1.00
stale reuse   -1.50
refresh       +0.55
```

reuse is preferred when:

\[
p(1.0)+(1-p)(-1.5)\ge0.55.
\]

So:

\[
p \ge 0.82.
\]

The implementation derives this threshold from the supplied utilities rather than hard-coding it.

That separates:

```text
belief
from
decision cost.
```

## 5. Runtime API

Minimal example:

```python
from temporal_validity import (
    Evidence,
    CurrentContext,
    TemporalValidityRuntime,
    WorldTimeTTL,
    EventDistanceTTL,
    UntilChange,
)

runtime = TemporalValidityRuntime(
    {
        "weather": WorldTimeTTL(8.0),
        "discourse": EventDistanceTTL(8),
        "reservation": UntilChange(),
    }
)

decision = runtime.evaluate(evidence, now)
```

The result contains:

```text
action
p_valid
reuse threshold
expected reuse utility
refresh utility
temporal coordinates
validity model description
```

and can be converted to a simple runtime state dictionary.

## 6. Demo result

`experiments/temporal_validity_runtime_demo.py` currently asserts:

```text
weather_7.5s
    REUSE

weather_8.5s
    REFRESH

discourse_8turns
    REUSE

discourse_9turns
    REFRESH

reservation_old_same_version
    REUSE

reservation_changed
    REFRESH
```

The reservation evidence in the demo is:

```text
3 hours old
20 structural events old
```

and remains valid because its source version is unchanged.

This directly kills the universal assumption:

```text
older evidence is always less valid.
```

## 7. Relation to the language-agent benchmarks

The runtime is the reusable implementation behind the strongest `resolver` condition.

The benchmark ladder is now:

```text
raw timestamps/history
    model performs temporal arithmetic + semantics + decision

age-plane side channel
    runtime performs arithmetic
    model performs semantics + decision

validity runtime
    runtime performs arithmetic + known source semantics + utility decision
    model consumes resolved state / action requirement
```

If the runtime-resolved condition beats the language model's direct temporal reasoning, that is not evidence for a new neural architecture.

It is evidence for moving a deterministic/estimable systems problem out of the LLM.

## 8. Relation to semantics discovery

The runtime does not magically know source semantics.

That is a separate problem.

The current project therefore distinguishes:

```text
KNOWN SEMANTICS
    configure WorldTimeTTL / EventDistanceTTL / UntilChange

UNKNOWN SEMANTICS
    learn or estimate the validity model from audited experience
```

`docs/LANGUAGE_SEMANTICS_DISCOVERY.md` attacks the second case.

The single-rate identifiability result says that even a perfect learner cannot infer the correct yoking if the experienced temporal coordinates remain confounded.

So the complete pipeline is:

```text
preserve coordinates
    ↓
collect rate-diverse / causally informative experience
    ↓
identify source validity semantics
    ↓
estimate P(valid now)
    ↓
apply utility threshold
    ↓
reuse or refresh
```

## 9. What this module deliberately does not do

It does not:

- infer source semantics from source names;
- claim timestamps are novel;
- implement a neural memory architecture;
- add oscillations or graph geometry;
- assume every source decays with age;
- collapse valid time into arrival time;
- force one universal validity rule.

It is intentionally small enough that established survival, caching, database, or control methods can replace pieces of it when appropriate.

## 10. Practical interpretation

The strongest practical sentence from the current research line is:

> **Do not ask a language model to rediscover temporal validity if the runtime can represent, estimate, and enforce it explicitly.**

The LLM remains useful for:

```text
understanding the user's request
mapping language to source/key
interpreting content
deciding what the user means
communicating the result
```

The runtime handles:

```text
what time the evidence refers to
when it arrived
how much structure has intervened
whether an explicit change occurred
how likely it is still valid
whether refreshing is worth the cost
```

That is currently a more defensible product direction than a special WidePresent neural layer.
