# WidePresent formalism v0.1 — a moving chart over temporal beliefs

Date: 2026-08-11

This note removes the bicycle metaphor and states the research object minimally.

Nothing here claims a new theory of time. Bitemporal records, event-time processing, delayed-observation filtering and continuous-time models are prior art. The purpose is to define exactly what WidePresent means when we test it.

## 1. Sparse temporal ledger

Let the agent/runtime hold a ledger of items

\[
\mathcal{L} = \{e_i\}_{i=1}^N
\]

with

\[
e_i = (v_i,\; t_i^w,\; t_i^k,\; q_i,\; s_i,\; \sigma_i).
\]

Where:

- `v_i` — represented value/content/state;
- `t_i^w` — **world / event / valid time**: when the item belongs in the modeled world;
- `t_i^k` — **knowledge / arrival / transaction time**: when the agent acquired/created the item;
- `q_i` — temporal kind, e.g. observation, prediction, action, memory;
- `s_i` — source;
- `sigma_i` — uncertainty / confidence metadata.

The ledger is not the present. It is history plus beliefs with temporal provenance.

## 2. Moving now

Let objective runtime time be

\[
t = t_{now}.
\]

For every item define two relative coordinates:

\[
r_i(t) = t_i^w - t
\]

and

\[
a_i(t) = t - t_i^k.
\]

`r_i` says where the item's world-valid time lies relative to now:

```text
r < 0    world-past
r = 0    world-now
r > 0    world-future
```

`a_i >= 0` is how long the agent has known/carried the item.

A late observation can therefore satisfy

```text
r << 0
but
a ~= 0
```

which means **old in the world, newly known to the agent**.

That distinction is lost if all temporal information is collapsed into one generic recency number.

## 3. The wide present is a projection, not the ledger

Choose past and future horizons `H_p`, `H_f` and a temporal resolution / basis.

Define a projection centered on current time:

\[
S_t = \Pi_t\left(\mathcal{L}, W, H_p, H_f\right)
\]

where `W` contains source-specific event-time progress / completeness information such as watermarks.

The projection retains items whose world-relative coordinate lies inside

\[
-H_p \le r_i(t) \le H_f.
\]

A simple fixed-grid implementation can expose channels such as

```text
relative world time
observation value
observation present/missing mask
observation knowledge age
prediction value
prediction mask
prediction knowledge age
uncertainty
source completeness / watermark state
```

A learned basis, LMU/HiPPO basis or sparse attention implementation could represent the same semantics without literal bins. The grid is an implementation, not the claim.

## 4. The key property: time changes state without content

Suppose no new message, sensor observation or memory arrives between `t` and `t + Delta`.

The ledger may be unchanged:

\[
\mathcal{L}_{t+\Delta} = \mathcal{L}_t.
\]

But the working temporal state is not:

\[
S_{t+\Delta} = \Pi_{t+\Delta}(\mathcal{L}) \ne \Pi_t(\mathcal{L}) = S_t.
\]

because

\[
r_i(t+\Delta)=r_i(t)-\Delta
\]

and

\[
a_i(t+\Delta)=a_i(t)+\Delta.
\]

So predictions move toward/past due time, observations become older, deadlines approach, and evidence intervals can become complete even when **zero new semantic content** is appended.

This is the cleanest statement of the original intuition.

### Event-driven state

A conventional event recurrence is schematically

\[
h_{k+1}=F(h_k,x_k).
\]

No new `x_k`, no transition.

### Clock-derived state

WidePresent adds a state component

\[
S(t)=\Pi(\mathcal{L},t)
\]

whose coordinates move under the external clock even between events.

This is standard behavior in many control, streaming and real-time systems. The question is whether making it explicit in the **working state consumed by an AI agent** improves its decisions.

## 5. Predictions as future-valid beliefs

A prediction created at knowledge time `t_c` for target time `t_f` is simply

\[
e_p=(\hat v, t_f, t_c, \text{prediction}, s, \sigma).
\]

Initially

\[
r_p(t_c)=t_f-t_c>0.
\]

As objective time advances, `r_p` decreases toward zero without needing a new token.

At

\[
t=t_f
\]

the prediction is due.

A validating observation may arrive later, at knowledge time `t_k > t_f`, while still carrying world time `t_f`. This preserves the difference between

```text
target time
and
validation arrival time
```

rather than overwriting both with "the latest turn."

Prediction rendezvous is therefore a special case of bitemporal belief bookkeeping, not claimed as a new forecasting primitive.

## 6. Completeness is different from value

For evidence source `s`, let `w_s(t)` be an event-time watermark: the greatest world time up to which the source claims no additional older events are expected.

For a world-time coordinate `tau`, one simple completeness indicator is

\[
c_s(\tau,t)=\mathbf{1}[w_s(t)\ge\tau].
\]

This distinguishes

```text
value absent, interval incomplete
```

from

```text
value absent, interval complete
```

without hallucinating an observation.

Real systems may require probabilistic lateness models rather than hard watermarks. The semantic distinction remains.

## 7. Three increasingly strong hypotheses

### Kernel hypothesis

The model does not need a wide matrix. It merely benefits when exact temporal arithmetic is performed outside the LLM and supplied as structured current state.

This is H1.

### Bitemporal hypothesis

Separating `t_world` and `t_known` reduces source/provenance errors beyond scalar age features.

This is H2.

### Wide-working-state hypothesis

A bounded relative-time projection around `now` provides useful inductive bias beyond an ordinary temporal kernel, timestamped model or continuous-time memory.

This is H3 and the first genuinely WidePresent-shaped architectural test.

## 8. What would falsify the architectural idea?

If a deterministic scalar/vector temporal kernel, timestamp Transformer, LMU/HiPPO state, or established delay-aware filter matches the wide projection across the registered tasks, then the matrix geometry is unnecessary.

The surviving result would simply be:

> explicit temporal bookkeeping is useful for agents.

That is acceptable.

## 9. What this formalism does not assert

It does not assert that:

- brains implement this representation;
- human consciousness is a bitemporal buffer;
- a fixed temporal grid is optimal;
- absolute time should replace semantic/event segmentation;
- every cognitive process should be time-yoked;
- a system with `S_t` feels time passing.

It defines an engineering object whose consequences can be measured.
