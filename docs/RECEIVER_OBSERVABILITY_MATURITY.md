# Receiver observability maturity — cross-project note

**Date:** 2026-08-15  
**Status:** measurement bridge from `anttiluode/Dig`. **Not a WidePresent win and not a new time coordinate.**

## Why this note exists

WidePresent already has a deliberately ordinary receiver-side temporal primitive in `receiver_present.py`:

```text
one objective now
+ source -> receiver path delays
+ path frontier
+ arrived / in-flight distinction
```

For a source `s` and receiver `r`, the path frontier answers a deterministic causal question:

> **Could an event from world time `t` physically have arrived at this receiver yet?**

That remains useful and requires no new theory.

Dig now adds a second receiver-side question that should not be confused with transport eligibility:

> **Once responses have begun to arrive, how much evidence has accumulated for distinguishing the alternative causes that matter to this receiver?**

Finite-horizon observability and discrimination are established control-theory / signal-detection ideas. Nothing here claims otherwise.

---

# The Dig measurement

Dig used one fixed reconstructed Hay `cell1.asc` morphology from the public FCI model.

Protocol:

```text
16 fixed source locations
6 fixed receiver locations
same 0.02 nA / 0.5 ms source impulse
matched no-stimulus subtraction
response prefixes at 0.5, 1, 2, 5, 10, 20, 40, 80, 120 ms
```

The first exploratory analysis used normalized trajectory geometry. It was useful but non-monotone, so Dig explicitly replaced it with a cumulative pairwise discrimination quantity:

```text
D_C,T^2(i,j)
    = integral_0^T || h_i(t) - h_j(t) ||^2 dt
```

for source responses `h_i,h_j` under receiver/readout `C`.

This quantity passed the required monotonicity guard over every tested source pair and random readout projection.

Pair-specific maturity is defined only relative to the same receiver's 120 ms value:

```text
M_C,T(i,j)
    = D_C,T^2(i,j) / D_C,120^2(i,j).
```

This is a matrix over candidate causes, not one scalar attached to an event.

---

# The important empirical separation

The earlier run measured aggregate six-port response-energy arrival:

```text
10 ms    98.23% of eventual aggregate response energy
20 ms    99.47%
```

The monotone pairwise discrimination gate found at the same horizons:

```text
10 ms
    median source-pair maturity       94.45%
    10th-percentile pair maturity     50.43%
    fraction pairs >=90% mature       53.3%

20 ms
    median source-pair maturity       99.70%
    10th-percentile pair maturity     80.39%
    fraction pairs >=90% mature       80.0%
```

Thus:

```text
almost all aggregate response energy present
```

does **not** imply:

```text
all candidate causes are almost fully distinguishable.
```

Different source pairs mature at different rates.

This is a stronger and cleaner reason to keep transport/arrival separate from discrimination maturity than the earlier normalized-shape analysis.

---

# Four quantities that must remain separate

## 1. World / event age

```text
now - event_time
```

When did the event happen?

## 2. Transport eligibility / path frontier

Given the source->receiver delay model:

```text
could the event have reached this receiver yet?
```

This is what `receiver_present.py` already captures.

## 3. Aggregate response arrival

```text
how much total response energy has accumulated?
```

This can be useful for instrumentation but is not enough to characterize decision readiness.

## 4. Receiver-relative discrimination matrix

```text
D_C,T[i,j]
```

How separated are candidate causes `i,j` by the response observed through readout `C` over horizon `T`?

This depends on:

```text
receiver/readout
observation horizon
candidate alternatives
noise / precision
measurement metric
```

It is therefore **not** an intrinsic scalar age attached to the event.

---

# Waiting versus routing

The monotone Dig gate also measured the same response tensor through fixed random readout dimensions `k=1..6`.

At 120 ms, median pairwise discrimination energy as a fraction of the physical six-port reference saturated around:

```text
k=1     12.8%
k=2     29.7%
k=3     45.9%
k=4     64.4%
k=5     88.7%
k=6    100.0%
```

Waiting helps every fixed readout because `D_C,T` accumulates monotonically.

But waiting cannot recover distinctions discarded by the output bottleneck.

This gives a clean systems interpretation:

```text
WAIT
    increase T under the current readout C
    -> accumulate more evidence
    -> cannot exceed that readout's asymptotic discrimination ceiling

ROUTE
    change C
    -> change what evidence is observed
    -> can raise or alter the attainable ceiling
```

In the toy, a one-dimensional readout reached 10% of the six-port reference by 20 ms but never reached 25% even at 120 ms. A two-dimensional readout reached 25% only by 40 ms and never reached 50%.

That does not prove a WidePresent or PivotPoint architecture. It only makes the `wait` versus `route` distinction mathematically concrete.

---

# Important correction from the first Dig analysis

The first Dig analysis used L2-normalized source trajectories, cosine distance and entropy effective rank. That shape rank peaked early and then declined.

Do not interpret that as literal observability decreasing with time. An observer retaining the full prefix can always ignore later samples.

The monotone `D_C,T^2` gate was added precisely to remove that ambiguity and passed its monotonicity checks.

So if this note is used later, prefer:

```text
finite-horizon cumulative pairwise discrimination
```

over:

```text
normalized trajectory rank
```

for claims about evidence accumulation.

---

# Why this is relevant to WidePresent but not evidence for H1-H7

WidePresent's active hypotheses concern online-agent temporal bookkeeping and whether explicit temporal state improves decisions.

The neuron experiment does **not** test any of those hypotheses.

It suggests one future asynchronous-agent distinction:

```text
AVAILABLE
    a result is allowed to have arrived

ARRIVING / ACCUMULATING
    evidence is physically becoming available

DISCRIMINATIVE ENOUGH FOR THIS DECISION
    the relevant alternatives are separated enough under the current readout
```

The last line requires a task/noise criterion supplied independently by the downstream problem. Dig does not invent that threshold.

---

# Connection to PivotPoint

PivotPoint already asks:

```text
what can I do now that changes what I will be able to read next?
```

A clean systems decomposition is:

```text
WAIT
    increase observation horizon T

ROUTE
    change effective readout / observation map C

PROBE
    inject a new discriminating input

ACT
    change the controlled state/world through the action channel

MODULATE / GATE
    potentially change the internal dynamics themselves
```

Only the last category necessarily changes the internal dynamics/operator.

This is better than calling every useful action "geometry deformation."

---

# Stop condition

Do **not** modify `receiver_present.py` merely because this note exists.

The current sequence should be:

1. Keep `path_frontier` as the deterministic transport baseline.
2. Treat `D_C,T[i,j]` as an external measurement object, not a required runtime field.
3. Build a benchmark only if there is a concrete online decision where `wait` versus `route/probe/act` depends on a known task threshold and current readout ceiling.
4. Compare against a boring deterministic resolver first.
5. If the resolver solves it, use the resolver and stop.

## One-line state

> **Path frontier says whether evidence could have arrived. Aggregate energy says how much response arrived. A receiver-relative finite-horizon discrimination matrix says which alternatives have actually separated. Dig shows these are not numerically identical, but WidePresent has not yet earned any extra architecture.**
