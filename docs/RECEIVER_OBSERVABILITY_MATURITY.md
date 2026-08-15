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

A 2026-08-15 experiment in `anttiluode/Dig` exposed a second question that should not be confused with the first:

> **Once a response has begun to arrive, how much of the receiver's eventual ability to distinguish alternative causes is already present in the finite response prefix?**

This note calls that **receiver observability maturity** only as project bookkeeping. Finite-horizon observability and discrimination are established control-theory ideas; the name is not a novelty claim.

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

For each temporal prefix, the exploratory analysis compared the normalized source-response geometry with its 120 ms version.

Important result:

```text
six-receiver view at 0.5 ms:

response energy accumulated           24.6% of 120 ms total
final nearest-neighbour identities    14 / 16 already correct
```

For the soma-only view:

```text
at 2 ms:

response energy accumulated           30.4%
pairwise source-distance correlation
with 120 ms geometry                  > 0.90
```

Thus in this model/protocol:

```text
response arrival fraction
!=
source-relationship maturity
```

The two evolve on different curves.

That is the only reason this note belongs here.

---

# Three quantities that must remain separate

## 1. World / event age

```text
now - event_time
```

When did the source event occur?

## 2. Transport eligibility / path frontier

Given the source->receiver delay model:

```text
could the event have reached this receiver yet?
```

This is what `receiver_present.py` already captures.

## 3. Receiver-relative discrimination maturity

Given the response that has arrived so far:

```text
how much evidence is present for distinguishing the candidate causes
that matter to this receiver/task?
```

This depends on:

```text
receiver/readout
observation horizon
noise / precision
candidate alternatives
decision metric
```

It is therefore **not** an intrinsic scalar age attached to the event.

---

# Important correction from Dig

The first Dig analysis used L2-normalized source trajectories, cosine distance and entropy effective rank.

That produced a useful warning:

```text
six-port normalized entropy rank

0.5 ms   8.38
1 ms     9.38
2 ms     9.77
5 ms     9.37
...
120 ms   8.52
```

The normalized shape rank peaks early and then declines.

Do not interpret that as literal observability decreasing when the receiver waits longer. An observer retaining the full prefix can always ignore later samples.

The non-monotonicity comes from the chosen normalized geometry: adding a shared late response can align row-normalized source trajectories and reduce entropy effective rank while preserving the earlier discriminative samples.

Therefore the next Dig gate is explicitly moving to a monotone finite-horizon discrimination quantity such as

```text
D_T^2(i,j)
    = integral_0^T || h_i(t) - h_j(t) ||^2 dt
```

for source impulse responses `h_i,h_j`.

Only after that gate should this note acquire a stronger numerical definition.

---

# Why this is relevant to WidePresent but not evidence for H1-H7

WidePresent's active hypotheses concern online-agent temporal bookkeeping and whether explicit temporal state improves decisions.

The neuron experiment does **not** test any of those hypotheses.

It only suggests a useful conceptual distinction for future asynchronous-agent work:

```text
AVAILABLE
    a result is allowed to have arrived

ARRIVING / IN FLIGHT
    a process has begun to influence the receiver

DISCRIMINATIVE ENOUGH
    the currently available prefix can support the next decision
```

An agent may therefore face a decision like:

```text
wait for more evidence
vs
route to another receiver/tool
vs
probe again
vs
act now
```

But whether exposing such a maturity estimate helps an LLM/agent is an open engineering question and needs its own registered benchmark.

---

# Connection to PivotPoint

PivotPoint already asks:

```text
what can I do now that changes what I will be able to read next?
```

A clean control-theory-inspired decomposition is:

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

This is better than calling every useful action "geometry deformation."

Again, the mapping is conceptual bookkeeping, not a novelty claim.

---

# Stop condition

Do **not** modify `receiver_present.py` merely because this note exists.

The sequence should be:

1. Dig runs the monotone finite-horizon discrimination gate.
2. If the distinction between transport arrival and discrimination maturity survives, define a minimal receiver-side measurement API.
3. Only then create an agent benchmark where `wait` versus `route/probe/act` decisions depend on that measurement.
4. If an ordinary deterministic resolver solves the benchmark, use the resolver and stop.

## One-line state

> **Path frontier answers whether evidence could have arrived. A separate receiver-relative finite-horizon quantity may answer whether enough distinguishing evidence has unfolded. The neuron toy shows those are not numerically identical, but WidePresent has not yet earned that extra runtime state.**
