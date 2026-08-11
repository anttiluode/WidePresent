# Content-conditioned yoking

Date: 2026-08-11

The age-plane formulation becomes more useful when different kinds of evidence obey different temporal semantics.

A real agent should not assume that every memory becomes irrelevant according to one universal combination of:

\[
(\Delta t,\Delta n).
\]

This note tests the smallest possible content-dependent case.

The experiment is:

`experiments/content_conditioned_yoking.py`

## 1. Two event types, two true temporal metrics

Type 0 is genuinely wall-time-yoked:

\[
k_0=\exp(-0.4\Delta t).
\]

Type 1 is genuinely structure/event-yoked:

\[
k_1=\exp(-0.2\Delta n).
\]

At the nominal rate

\[
\Delta t=0.5\Delta n,
\]

so

\[
k_0
=\exp[-0.4(0.5\Delta n)]
=\exp(-0.2\Delta n)
=k_1.
\]

The event types therefore have different **true yoking** but identical nominal behavior.

This is deliberate.

## 2. Two models

### Universal temporal metric

One kernel is shared by both event types:

\[
k=\exp(-a\Delta t-b\Delta n).
\]

### Content-conditioned metric

Each event type gets its own orientation:

\[
k_c
=\exp(-a_c\Delta t-b_c\Delta n).
\]

This is not presented as novel. It is a stripped-down diagnostic model.

## 3. Fixed-rate training

With only the nominal `0.5 s/event` rate, both models fit essentially perfectly.

Representative fit:

```text
universal:
    a ~= 0.100
    b ~= 0.150
    loss ~= 0

conditioned:
    type 0 a ~= 0.100, b ~= 0.150
    type 1 a ~= 0.100, b ~= 0.150
    loss ~= 0
```

Nothing in the data tells the conditioned model that the event types should differ.

This is another expression of the identifiability ridge.

## 4. Narrow rate diversity

Training over `0.45..0.55 s/event` changes the picture.

Representative local fit:

### Universal

```text
a ~= 0.201
b ~= 0.100
loss ~= 8.8e-5
```

The one shared metric compromises between the two incompatible targets.

### Conditioned

```text
time type:
    a ~= 0.375
    b ~= 0.013

structure type:
    a ~= 0.011
    b ~= 0.194

loss ~= 8e-7
```

The type-conditioned model begins to recover the correct axes.

## 5. Wide rate diversity

Training over `0.25..0.90 s/event` sharpens the separation.

### Universal

```text
a ~= 0.203
b ~= 0.092
loss ~= 3.5e-3
```

The compromise becomes visibly wrong because the rate-diverse data expose the conflict between the event types.

### Conditioned

```text
time type:
    a ~= 0.393
    b ~= 0.003

structure type:
    a ~= 0.004
    b ~= 0.198

loss ~= 3e-6
```

True values are:

```text
time type:      a=0.4, b=0
structure type: a=0,   b=0.2
```

So once the data identify the distinction, a content-conditioned metric can recover it.

## 6. Why this matters more than one universal "present width"

The original WidePresent intuition often spoke as if there might be one useful temporal width.

This experiment argues against that simplification.

Different evidence types can have different yoking orientations even when they look identical at one operating rhythm.

A more realistic agent might have patterns such as:

```text
weather observation
    mostly wall-time-yoked

conversation reference
    partly structure/turn-yoked

calendar deadline
    future wall-time-yoked

inventory / reservation state
    valid-until-change rather than simple age decay

cached tool result
    source-specific wall-time freshness
```

The examples above are engineering intuitions, not results of this toy.

The lesson is only:

> **do not assume one temporal metric is correct for every content/source type.**

## 7. Prior art pressure

This direction is heavily occupied.

Mozer, Kazakov & Lindsey's CT-GRU already makes storage and retrieval **timescale selection depend on the current event and recurrent context**, with timestamps driving intrinsic decay across fixed physical time scales.

Self-Attentive Hawkes Process work explicitly notes that sequence-position encoding alone ignores real inter-event intervals and introduces continuous-time temporal encoding for event attention.

Modern irregular-event models likewise use continuous-time positional representations and multiscale temporal mechanisms.

So "content chooses a timescale" is not a WidePresent novelty claim.

The narrower distinction here is between **which coordinate the scale belongs to**:

```text
physical seconds
versus
structural/event distance
```

and whether that distinction is statistically identifiable from the experienced rate distribution.

## 8. A possible practical representation

Instead of a single age scalar, evidence could carry a minimal coordinate:

\[
z_i=(\Delta t_i,\Delta n_i,c_i),
\]

where `c_i` is a learned/source/content representation.

A gating network could choose or mix kernels over the age plane:

\[
k_i
=\sum_r g_r(c_i)\,
\exp(-a_r\Delta t_i-b_r\Delta n_i).
\]

This is deliberately generic.

Pure time, pure structure, and mixed yoking are all special cases.

The important discipline would be:

- do not make `g_r` semantic magic;
- compare against ordinary continuous-time/event models;
- test rate-shift OOD;
- measure yoking directly rather than inferring it from accuracy;
- preserve uncertainty when the training rate distribution underidentifies the orientation.

## 9. What to do next

The synthetic exponentials have now yielded three fairly stable principles:

1. **yoking is unidentifiable at one fixed rate;**
2. **rate variance controls how well it can be identified;**
3. **different content types may require different yoking orientations.**

The next useful move should leave the hand-designed exponential world.

A good next benchmark would use a small event-agent environment where event types have different temporal semantics and where the model must choose whether to trust, refresh, or ignore prior evidence under rate shift.

That would connect the age-plane formalism back to the external TicToc-style tool-decision problem without reducing the task to timestamp arithmetic.
