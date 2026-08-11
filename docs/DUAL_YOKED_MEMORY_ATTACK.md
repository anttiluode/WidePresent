# Dual-yoked memory attack

Date: 2026-08-11

This experiment tests the surviving architectural idea after the clock/structure-yoking branch was sharpened by the identifiability result.

The question is no longer:

> should a model have a clock?

It is:

> **Can one bounded state preserve both an absolute-time integration channel and an event/structure integration channel, and does that help when a task needs both simultaneously?**

The experiment is:

`experiments/dual_yoked_memory_attack.py`

## 1. Task

Every episode contains two independent content channels.

Channel 0 matters through a physical-time fading rule:

\[
s_t=\sum_i x^{(0)}_i e^{-(T-t_i)/2.5\mathrm{s}}.
\]

Channel 1 matters through an event-distance fading rule:

\[
s_n=\sum_i x^{(1)}_i e^{-(N-i)/5\mathrm{events}}.
\]

The network predicts both signs.

So this is not a benchmark where the evaluator alternates between a "time task" and a "structure task" after training. Both invariants are needed in the same episode.

Training is centered around `0.5 s/event`.

Evaluation uses compressed and stretched rates separated by a factor of three in total duration, following the useful structure-yoking assay logic.

## 2. Matched state budget

The principal comparison keeps `16` state numbers at readout.

### Event GRU

```text
16 learned recurrent coordinates
updates only when content arrives
```

This gives the network a strong structural/event-time bias.

### dt GRU

```text
16 learned recurrent coordinates
input = [content, dt]
updates only when content arrives
```

This is the cheap attacker: tell an ordinary GRU elapsed time and let it decide what to do.

### Decay GRU

```text
16 learned recurrent coordinates
before each event:
    h_k <- exp(-r_k * dt) h_k
then:
    GRUCell(content, h)
```

The physical-time decay rates `r_k` are trainable.

This is a simple continuous-time-decay attacker, **not** an exact reproduction of CT-GRU.

### Hard dual state

```text
8 free event-GRU coordinates
+
8 deterministic physical-time fading coordinates
```

The physical bank uses four broad log-spaced time constants from `0.5` to `8 s`, for both content channels.

Both paths see both input channels. The output head decides what to use.

The split therefore supplies two invariants but does not hard-code "channel 0 must use clock, channel 1 must use structure" into the readout.

## 3. Three-seed endpoint result

The reported metric averages compressed and stretched test performance.

### Training at one nominal rate

| model | physical-time head | structure head | joint |
|---|---:|---:|---:|
| event GRU | 0.867 ± 0.003 | 0.896 ± 0.002 | 0.777 ± 0.005 |
| dt GRU | 0.863 ± 0.005 | 0.896 ± 0.003 | 0.773 ± 0.002 |
| decay GRU | 0.881 ± 0.002 | 0.895 ± 0.004 | 0.788 ± 0.005 |
| **hard dual** | **0.906 ± 0.002** | 0.896 ± 0.007 | **0.811 ± 0.009** |

The dual state improves the physical-time component while preserving the structure component.

The plain `dt` feature does not help at this one-rate training condition, consistent with the identifiability argument: the network can simply ignore a nearly constant input.

### Moderate training rate diversity: ±0.15 s/event

| model | physical-time head | structure head | joint |
|---|---:|---:|---:|
| event GRU | 0.864 ± 0.006 | 0.901 ± 0.005 | 0.778 ± 0.007 |
| dt GRU | 0.861 ± 0.002 | 0.897 ± 0.003 | 0.772 ± 0.003 |
| decay GRU | 0.876 ± 0.004 | 0.895 ± 0.004 | 0.784 ± 0.002 |
| **hard dual** | **0.899 ± 0.006** | 0.899 ± 0.009 | **0.808 ± 0.003** |

The ordering barely changes.

That is **not** what the cleanest version of the identifiability story predicted. Once rate diversity is available, an expressive learned continuous-time model should in principle be able to identify the useful physical-time dependence.

The fact that these small GRUs do not catch up means one of several things may be true:

1. the training set/optimization is still too weak;
2. concatenated `dt` is a poor parameterization for discovering the invariant;
3. the simple decay-GRU is not expressive enough in the right way;
4. the hard dual basis is genuinely easier to learn from;
5. the synthetic task itself favors the hand-provided exponential bank.

This result does not distinguish those explanations.

## 4. The decay GRU did not spontaneously split into two clocks

Across the three seeds, its learned physical decay rates remained tightly clustered.

Approximate endpoint summaries:

```text
zero rate-diversity training:
    min ~0.086 /s
    max ~0.138 /s
    mean ~0.109 /s

±0.15 s/event training:
    min ~0.086 /s
    max ~0.141 /s
    mean ~0.110 /s
```

There was no obvious emergence of:

```text
some r_k ~= 0     -> structural/event memory
some r_k >> 0     -> physical-time memory
```

So the attractive story "a continuous-time GRU will automatically discover separate yoking coordinates" is **not supported by this toy**.

## 5. What is actually positive

A modest, concrete result survives:

> **Providing simultaneous basis states with different invariants can make a mixed-invariant task easier under rate shift.**

The improvement is not huge, but it is consistent and occurs in the intended output rather than through a generic score increase.

At the zero-diversity endpoint, hard dual versus event GRU is roughly:

```text
physical-time head: +3.9 percentage points
structure head:      ~0
joint:               +3.4 points
```

That is a much narrower claim than "WidePresent needs two clocks."

## 6. Why this still does not earn a WidePresent architecture claim

The hard dual path is privileged.

It supplies a bank of physical-time exponential features in advance. Even though the bank does not contain the exact `2.5 s` target time constant, it is deliberately designed to span useful physical horizons.

That is an inductive prior.

Strong attackers remain:

- CT-GRU as described by Mozer, Kazakov & Lindsey (2017);
- ODE-RNN / ODE-LSTM style continuous-time recurrent states;
- modern continuous-time state-space models;
- learned multi-timescale exponential filters;
- attention with both physical-time and position/event-distance bias;
- a generic kernel bank over `(Δt, Δn)` rather than a named split architecture.

If any of those match the result, the useful engineering lesson is simply to preserve both temporal coordinates.

## 7. Connection to the identifiability result

The dual state can be reinterpreted as a hedge against the ridge

\[
ac+b=q.
\]

Instead of forcing one learned hidden state to choose a point on that ridge, it keeps explicit basis functions near both axes:

```text
physical-time basis
structure/event basis
```

A downstream model can combine them.

That may be particularly useful when training rate diversity is weak.

But this experiment also shows that "rate diversity exists" is not enough by itself to guarantee an ordinary GRU will exploit it.

So there are now two separate questions:

1. **statistical identifiability** — is enough evidence present in the training distribution?
2. **architectural accessibility** — is the useful invariant easy for the model/optimizer to represent and discover?

Those should not be conflated.

## 8. The strongest next attack

Do not make the hard dual network larger.

The next serious attacker should implement a genuine multi-timescale continuous-time recurrent baseline rather than our crude decay-GRU approximation.

The most relevant older reference is:

> Mozer, Kazakov & Lindsey, **Discrete Event, Continuous Time RNNs**, 2017, arXiv:1710.04110.

Their CT-GRU explicitly represents multiple memory timescales and uses timestamps to drive decay, while context-dependent gates select scales for storage/retrieval.

That is almost tailor-made to attack the current result.

A fair next gate is therefore:

```text
hard dual 16-state
vs
CT-GRU-like 16-state / matched parameter budget
vs
ordinary event GRU
vs
ordinary dt GRU
```

on the same simultaneous time+structure task and rate-diversity sweep.

### Kill condition

If CT-GRU-like dynamics match the hard dual state, drop the named dual architecture and keep only the general lesson:

> preserve physical-time and structural integration modes when both may matter.

### Interesting survival condition

If the hard axis-aligned state remains more robust/sample-efficient specifically under low rate diversity, then the project has a more precise architectural hypothesis:

> **explicitly preserving unresolved temporal invariants can outperform asking a learned recurrent state to discover the decomposition from underidentified data.**

That is where the branch currently stands.
