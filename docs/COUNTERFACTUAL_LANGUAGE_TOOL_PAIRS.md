# Counterfactual language/tool temporal pairs

Date: 2026-08-11

This is the harder companion to `experiments/language_tool_validity_benchmark.py`.

Aggregate accuracy can hide the exact failure WidePresent is trying to diagnose. A model may be correct on many cases while using the wrong correlated temporal proxy.

This suite therefore asks a counterfactual question:

> **When only the causal temporal coordinate changes, does the agent change its tool decision? When only a non-causal temporal coordinate changes, does it remain invariant?**

The experiment is:

`experiments/counterfactual_language_tool_pairs.py`

The generated cases are directly compatible with the existing language runner:

```bash
python experiments/language_tool_validity_benchmark.py run-openai ...
```

No new model interface or scoring convention is introduced.

## 1. Two kinds of pair

Every pair has two variants, `A` and `B`.

### Causal flip pair

A source's true validity coordinate crosses its contract boundary.

The correct action must change:

```text
A  REUSE
B  REFRESH
```

### Non-causal invariance pair

A different temporal coordinate changes substantially while the causal validity coordinate stays on the same side of the boundary.

The correct action must stay the same.

This prevents a model from scoring well merely because it is "temporally sensitive" in a generic sense.

The desired behavior is **selective sensitivity**.

## 2. Weather pairs

Weather validity is defined by world/source observation age:

```text
valid through 8.0 seconds after observed_at
```

### `weather_flip`

Held fixed:

- cached value and location;
- `received_at`;
- current decision time;
- arrival/knowledge age;
- six intervening messages;
- neutral message contents.

Only `observed_at` changes:

```text
A world age = 7.5 s -> REUSE
B world age = 8.5 s -> REFRESH
```

This is an especially strong provenance pair.

A policy based on "how recently did I receive this?" cannot solve it because arrival age is identical.

### `weather_invariant`

World age is held fixed on one side of the validity boundary while structural distance changes:

```text
A 2 intervening messages
B 14 intervening messages
```

The action must remain unchanged.

This attacks event-count heuristics.

## 3. Discourse pairs

Discourse validity is defined by structural/message distance:

```text
valid through 8 intervening conversation messages
```

Wall-clock seconds do not expire it.

### `discourse_flip`

Wall-clock span is held essentially fixed.

Only intervening-message count changes:

```text
A 8 messages -> REUSE
B 9 messages -> REFRESH
```

This attacks wall-time recency heuristics.

### `discourse_invariant`

The event distance is held fixed on one side of the boundary.

Elapsed wall time changes sharply:

```text
A about 3 seconds since cache receipt
B about 24 seconds since cache receipt
```

The action must remain unchanged.

## 4. Reservation pairs

Reservation state has no age-decay rule.

It is valid until an explicit change event occurs.

### `reservation_flip`

Timing and event count are identical.

One neutral message is replaced by:

```text
Reservation-change notification:
the cached reservation status is no longer current.
```

So:

```text
A no invalidation -> REUSE
B invalidation    -> REFRESH
```

### `reservation_invariant`

Invalidation state is held fixed.

Wall time and event count both change sharply.

The action must remain unchanged.

This explicitly tests whether the model invents a generic "old means stale" rule for a state whose contract does not contain one.

## 5. Conditions remain paired

Each individual case is still rendered in the three existing language-agent conditions:

### `raw`

Timestamped history + explicit validity contract.

The model must do arithmetic/counting itself.

### `age_plane`

Same history plus deterministic:

```text
world age
arrival age
intervening-message count
invalidation flag
```

No action recommendation.

### `resolver`

Same history plus the deterministic contract result.

The counterfactual suite therefore tests both:

```text
within-case representation effect
and
within-pair causal consistency
```

## 6. Pair metrics

The pair scorer reports ordinary case metrics plus stronger counterfactual metrics.

### `pair_relation`

Did the model exhibit the correct relation between the two actions?

```text
causal pair     -> actions differ
invariance pair -> actions match
```

This is deliberately weaker than being correct on both cases.

A model can flip in the right relation while still choosing the two labels backwards.

### `both_oracle`

Both actions must individually match the oracle.

This is the main causal-consistency metric.

### `both_task`

Both actions must also be executed correctly:

```text
REUSE
    direct answer contains cached value

REFRESH
    correct refresh_source call
    correct source
    correct key
```

### `causal_flip_success`

Fraction of causal pairs in which the model actually changes action.

### `spurious_flip_rate`

Fraction of non-causal pairs in which the model changes action anyway.

The ideal signature is:

```text
causal_flip_success = 1
spurious_flip_rate  = 0
both_oracle         = 1
both_task           = 1
```

## 7. Deterministic diagnostic

A local dry run used `10` pairs per family, `60` pairs / `120` cases total.

The generator assertions and an oracle-response end-to-end scorer test passed:

```text
pair relation      1.000
both oracle        1.000
both task          1.000
causal flip        1.000
spurious flip      0.000
```

These are harness checks, not LLM results.

The useful deterministic characterization is how deliberately wrong one-coordinate policies behave.

### Mean pair-relation accuracy

| metadata theory | mean pair relation | both actions oracle-correct |
|---|---:|---:|
| arrival-age TTL | 0.400 | 0.350 |
| world-time TTL | 0.567 | 0.567 |
| event-position TTL | 0.567 | 0.567 |
| invalidation only | 0.667 | 0.583 |
| source-specific resolver | **1.000** | **1.000** |

More importantly, the failures are selective.

### Causal flip families

| policy | weather flip | discourse flip | reservation flip |
|---|---:|---:|---:|
| arrival age | 0 | 0 | 1 |
| world time | **1** | 0 | 1 |
| event position | 0 | **1** | 1 |
| invalidation only | 0 | 0 | **1** |
| resolver | **1** | **1** | **1** |

So the suite identifies the temporal theory embodied by the policy:

```text
world-seconds policy
    understands weather
    misunderstands discourse

event-position policy
    understands discourse
    misunderstands weather

invalidation-only
    understands reservation
    misunderstands both fading sources
```

The weather flip is also an explicit valid-time versus arrival-time test:

```text
arrival age identical
world age crosses the boundary
```

An arrival-recency policy has no way to flip correctly.

## 8. Running the counterfactual LLM gate

Generate a pilot:

```bash
python experiments/counterfactual_language_tool_pairs.py generate \
  --output counterfactual_cases.jsonl \
  --pairs-per-family 20
```

Check the pair construction:

```bash
python experiments/counterfactual_language_tool_pairs.py sanity \
  --input counterfactual_cases.jsonl
```

Use the existing real language-agent runner:

```bash
python experiments/language_tool_validity_benchmark.py run-openai \
  --input counterfactual_cases.jsonl \
  --output counterfactual_responses.jsonl \
  --model gpt-5 \
  --conditions raw age_plane resolver
```

Then score causal consistency:

```bash
python experiments/counterfactual_language_tool_pairs.py score \
  --cases counterfactual_cases.jsonl \
  --responses counterfactual_responses.jsonl
```

The current ChatGPT execution environment still has no `OPENAI_API_KEY` and no cached local language model, so genuine LLM scores are intentionally not recorded here.

## 9. Pre-registered interpretation

### `raw` already has high `both_oracle`

Then a capable LLM can operationalize the explicit validity contract directly from timestamps and history structure.

The temporal side channel is convenience, not evidence for a necessary runtime layer.

### `age_plane > raw`

Then deterministic arithmetic/counting improves causal temporal behavior without supplying the action label.

The interesting question becomes *which pair families improve*.

For example:

```text
weather flip improves
    -> model was confusing observed age with arrival recency

discourse flip improves
    -> explicit structural distance helped

invariance pairs improve
    -> side channel reduced spurious use of the wrong coordinate
```

This is much more diagnostic than an aggregate accuracy increase.

### `resolver > age_plane`

Then the boring runtime design wins again:

> apply known temporal validity semantics outside the language model and give the model the resolved state.

### `resolver` fails counterfactual pairs

Then the temporal problem has already been solved upstream.

Remaining failures are instruction following, function calling, cached-value reproduction, or other general agent behavior.

## 10. Why this is stronger than the first language benchmark

The first benchmark asks:

> Did the agent make the right tool decision across a distribution?

This one asks:

> **What variable caused the agent to change that decision?**

That is closer to the actual WidePresent question.

An IID model can be accurate while using seconds as a proxy for turns, arrival time as a proxy for valid time, or generic age as a proxy for explicit state change.

Counterfactual pairs expose those substitutions directly.

## 11. Current frontier

No special neural architecture is implied.

The surviving research object is still a runtime/diagnostic question:

> **Can the agent preserve and selectively use the temporal coordinate that is causally relevant for each kind of evidence, rather than a correlated proxy?**

The external language-model run is now set up to answer that at the level of tool behavior rather than synthetic classifier accuracy.
