# Time versus structure is an identifiability problem

Date: 2026-08-11

This note extracts the strongest result from the clock/structure-yoking branch so far.

The result is not that a fixed clock is universally useful.

It is:

> **At one fixed event rate, the data cannot in general tell whether a fading memory is yoked to elapsed seconds, event distance, or a mixture of both.**

That is an identifiability statement.

It explains why a network can fit nominal-rate data perfectly and still choose the wrong temporal invariant for rate shift.

The committed demonstration is:

`experiments/time_structure_identifiability.py`

## 1. Generic two-coordinate fading kernel

Give every prior event two ages relative to the current decision:

- physical age `Δt` in seconds;
- structural age `Δn` in event positions.

Consider the simple kernel

\[
k(\Delta t,\Delta n)=\exp(-a\Delta t-b\Delta n).
\]

`a` measures physical-time decay and `b` measures event-distance decay.

The pure axes are:

```text
physical-time-yoked:  a > 0, b = 0
structure-yoked:      a = 0, b > 0
mixed:                 a > 0, b > 0
```

## 2. Exact single-rate non-identifiability

Suppose training always has one fixed inter-event gap `c`.

Then for every event distance,

\[
\Delta t=c\Delta n.
\]

Substitute this into the kernel:

\[
k
=\exp[-a(c\Delta n)-b\Delta n]
=\exp[-(ac+b)\Delta n].
\]

The observations therefore identify only

\[
q=ac+b.
\]

They cannot identify `a` and `b` separately.

There is an entire line of equivalent solutions:

\[
ac+b=q.
\]

For the current assay,

```text
nominal gap c = 0.5 s/event
physical tau = 2.5 s      -> a*=0.4 /s
structural tau = 5 events -> b*=0.2 /event
```

and both targets give

\[
q=0.4\times0.5=0.2.
\]

So all of these are exactly indistinguishable at the training rate:

```text
a=0.00, b=0.20
 a=0.08, b=0.16
 a=0.20, b=0.10
 a=0.34, b=0.03
 a=0.40, b=0.00
```

The experiment asserts that their nominal kernels are equal to floating-point tolerance.

## 3. Rank statement

The exponent observed at rate `c_j` is

\[
q_j=a c_j+b.
\]

Across rates this is a linear system with design rows

\[
[c_j,\;1].
\]

With one unique rate, the design matrix has rank 1.

With at least two distinct rates, it has rank 2.

Therefore, in this simple model:

> **rate diversity is what makes physical-time versus event-distance yoking identifiable from data.**

The fixed clock does not magically reveal a fact already present in one-rate examples. A hard clock path supplies an inductive constraint that picks one point in an otherwise underdetermined solution set.

## 4. Five equivalent fits, five different futures

A local exploratory run initialized the same two-parameter model from five different points and trained only at the fixed `0.5 s/event` rate.

All five fits converged to essentially zero kernel error while landing at very different decompositions:

| fit | `a` | `b` | `0.5a+b` |
|---:|---:|---:|---:|
| 0 | 0.030 | 0.185 | 0.200 |
| 1 | 0.080 | 0.160 | 0.200 |
| 2 | 0.200 | 0.100 | 0.200 |
| 3 | 0.340 | 0.030 | 0.200 |
| 4 | 0.390 | 0.005 | 0.200 |

At the nominal rate they are the same model for this task.

After rate shift they are not.

Normalized kernel MSE for compressed/stretched conditions:

| fit | time target, compressed | time target, stretched | structure target, compressed | structure target, stretched |
|---:|---:|---:|---:|---:|
| 0 | 0.091 | 0.113 | 0.0004 | 0.0011 |
| 1 | 0.074 | 0.075 | 0.0034 | 0.0072 |
| 2 | 0.037 | 0.023 | 0.026 | 0.034 |
| 3 | 0.0045 | 0.0016 | 0.101 | 0.075 |
| 4 | 0.0001 | 0.00004 | 0.150 | 0.090 |

So models that are experimentally indistinguishable on the complete nominal training condition make radically different OOD predictions.

This is not ordinary overfitting in the sense of insufficient training examples. Infinite examples at exactly one rate do not remove the ambiguity.

## 5. Rate diversity breaks the valley

The same optimizer was then trained on several rates instead of one.

### Time-yoked target

With narrow rate diversity (`0.45..0.55 s/event`), fits moved toward

```text
a ~= 0.375..0.396
b ~= 0.013..0.002
```

With wide diversity (`0.25..0.90 s/event`), they moved further toward the true axis:

```text
a ~= 0.392..0.399
b ~= 0.003..0.0004
```

True value:

```text
a = 0.4
b = 0
```

### Structure-yoked target

With narrow diversity, fits moved toward

```text
a ~= 0.008..0.026
b ~= 0.196..0.187
```

With wide diversity:

```text
a ~= 0.001..0.007
b ~= 0.199..0.196
```

True value:

```text
a = 0
b = 0.2
```

So the ambiguity is not permanent. The training distribution has to contain evidence that separates the coordinates.

## 6. Why the GRU results now make more sense

The earlier learned-GRU assay found:

```text
event GRU        SYI ~ 1.00
explicit-dt GRU  SYI ~ 0.91
blank-clock GRU  SYI ~ 0.51, high variance
```

At the nominal training rate, `dt` is almost constant. There is little or no statistical evidence forcing the network to use it.

Likewise, the blank-clock network always sees the same rhythm of roughly five clock ticks per content event. It can learn that rhythm as another structural regularity rather than learning an invariant physical-time law.

The identifiability result therefore suggests a useful distinction:

```text
clock exposure
    a time signal is present
    but the data may not identify how it should be used

clock constraint
    some state dynamics are forced to advance in physical time
    even when the training distribution cannot identify that choice

rate-diverse learning
    the data itself separates time age from event age
    and can in principle learn the yoking
```

## 7. Dual-yoked memory is now a hedge, not a claim

This changes the motivation for keeping both coordinates.

A dual representation

\[
(\Delta t,\Delta n)
\]

or a small bank of kernels over both coordinates is useful when the system does not yet know what a task's memory horizon should be yoked to.

It can preserve both possibilities until enough evidence exists to choose or combine them.

A generic mixed kernel family is

\[
k_r(\Delta t,\Delta n)
=\exp(-a_r\Delta t-b_r\Delta n).
\]

Axis-aligned filters have a simple interpretation:

```text
b_r = 0 -> pure physical-time memory
 a_r = 0 -> pure structural/event memory
```

Mixed filters interpolate between them.

This is a much less mystical interpretation of "dual-yoked memory" than two separate clocks in a brain-like substrate.

## 8. Strong prior art / attacker

This is not a claim that continuous-time recurrent memory is new.

Mozer, Kazakov & Lindsey's **Discrete Event, Continuous Time RNNs** (2017, arXiv:1710.04110) already introduced a CT-GRU in which event timestamps drive intrinsic continuous-time decay across multiple memory scales, while a standard GRU can receive timestamps merely as extra inputs. Their experiments found the two approaches broadly similar across their tested datasets.

That work is a direct architectural adversary for WidePresent.

Norman-Haignere et al. (Nature Neuroscience, 2025, DOI `10.1038/s41593-025-02060-8`) provide the time-versus-structure yoking assay that motivated this branch. Their result concerns human auditory cortex, not WidePresent.

Skrill & Norman-Haignere (NeurIPS 2023) measured position-yoked versus structure-yoked integration in language models and reported a learned transition across layers.

The present contribution is therefore best treated as an internal formal clarification:

> **single-rate data confounds the two coordinates exactly in the exponential-kernel case.**

## 9. What this kills

It kills several sloppy interpretations:

- "the model saw timestamps, so it had enough information to learn absolute time";
- "blank ticks necessarily force time-yoked cognition";
- "failure under rate shift proves the network cannot represent time";
- "a one-rate benchmark can determine what the integration window is yoked to".

All four can fail because the training problem itself is underidentified.

## 10. What to test next

The next serious experiment should be a **rate-diversity curve** rather than another architecture zoo.

For each model family, gradually increase the amount of timing diversity available during training and measure:

1. recovered structure-yoking index;
2. OOD compressed/stretched accuracy;
3. how much rate diversity is needed before explicit `dt` becomes useful;
4. whether a hard physical-time channel helps specifically in the low-diversity regime;
5. whether CT-GRU / continuous-time SSM baselines learn the correct decomposition as soon as it is identifiable;
6. whether retaining both `(Δt, Δn)` coordinates is more sample-efficient than forcing either one.

The interesting quantity may therefore be not "does a clock help?" but:

> **How much evidence is required before a learner can discover what its own memory horizon should be yoked to?**

That question is both cleaner and harder to explain away with timestamp bookkeeping.
