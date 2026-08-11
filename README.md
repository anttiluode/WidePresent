# WidePresent

**Researching explicit temporal state for online AI — with a moving `now`, hard prior-art controls, and permission for the idea to die.**

The project started from a bicycle-chain intuition: a fixed-pitch loop keeps moving whether or not anything interesting happens; one point is `now`; state behind it is recent past; predictions can sit ahead of it and later arrive at the present.

The first day of research already changed that picture substantially.

## Current question

Not:

> How do we make an AI conscious of time?

But:

> **Does an online agent make better decisions when objective time is maintained as explicit, continuously updated state rather than being left implicit in token/event order or passive timestamp text?**

And, only if the answer is yes:

> **Does a bounded, temporally typed working state around a moving `now` add anything beyond a boring deterministic temporal kernel?**

No consciousness claim is made here.

---

## The first thing we killed

`experiments/gate0_clock_vs_event.py` makes elapsed duration the target while event density is spuriously correlated with duration during training and reversed OOD.

Seed 0:

```text
event_index_iid        0.9440
event_index_ood        0.0570
timestamp_iid          1.0000
timestamp_ood          1.0000
fixed_tick_iid         1.0000
fixed_tick_ood         1.0000
```

So event count can impersonate time and catastrophically fail under a rate shift. But an exact timestamp solves the toy just as well as a fixed tick clock.

**Gate 0 therefore provides zero evidence for a special WidePresent architecture.**

---

## Direct prior art found immediately

The broad versions of the idea are already occupied:

- Clockwork RNNs and Phased LSTMs explicitly organize neural computation by time;
- Time2Vec and continuous-time models expose timestamps / elapsed time;
- LMU and HiPPO provide principled continuous sliding-history state;
- time cells and theta sequences provide biological temporal organization;
- phenomenology and active-inference work already discuss an extended present containing retention and anticipation;
- Time-Aware World Models condition dynamics explicitly on `dt`;
- delayed-observation filtering and out-of-sequence-measurement theory already handle late evidence;
- stream processors distinguish event time from processing time and use watermarks for late data;
- bitemporal databases distinguish when a fact is true from when the system learns/stores it;
- recent conversational-memory work already imports bitemporality into AI memory.

Most directly, Cheng et al.'s **TicToc** benchmark (ACL 2026) calls out *temporal blindness* in multi-turn LLM agents: the same conversational context can demand a different tool-use decision after more real-world time has elapsed. Their timestamp condition still leaves substantial misalignment.

See [`docs/PRIOR_ART_MAP.md`](docs/PRIOR_ART_MAP.md).

The project rule is:

> **Borrow mechanisms, not conclusions. Re-test everything that matters.**

---

## The important correction: there is more than one time

The bicycle chain originally gave us one temporal axis. A real online agent needs at least two:

```text
world / event / valid time       when the represented event belongs in the world
knowledge / arrival time         when the agent learned about it
```

Example:

```text
Sunday:    event happens
Tuesday:   agent is told what happened Sunday
```

At Tuesday's `now`, that information is simultaneously:

```text
old in world time
new in knowledge time
```

A one-dimensional "memory age" cannot express that cleanly.

`bitemporal_present.py` implements a sparse ledger and a fixed-width projection around `now` with separate observation/prediction channels, knowledge ages and source watermarks.

See [`docs/TWO_CLOCKS_AND_WATERMARKS.md`](docs/TWO_CLOCKS_AND_WATERMARKS.md).

---

## The second correction: absence is not evidence of absence

If a sensor, tool or user can report events late, then:

```text
nothing happened
```

and

```text
nothing has arrived yet
```

are different states.

Stream processors already solve the bookkeeping problem with event-time progress / watermarks. WidePresent imports that distinction rather than pretending to invent it.

`experiments/gate1b_late_evidence_sanity.py` is an executable sanity check: the naive "no arrival = empty" rule produces false-empty declarations; a valid bounded-delay watermark abstains while the interval is incomplete and can make zero false-empty claims once the evidence horizon has closed.

Again: **this is prior-art logic used as a prerequisite, not a WidePresent win.**

---

## Architecture pivot: temporal kernel before temporal neuron

Objective time is usually known to the runtime exactly. Asking a language model to infer it from token statistics is unnecessary work.

So `temporal_kernel.py` implements the deliberately boring baseline:

```text
input timestamped events/facts
        |
        v
exact deterministic bookkeeping
        |
        +--> current decision time
        +--> world age
        +--> knowledge age
        +--> elapsed since tool observation
        +--> time-to-deadline / lateness
        +--> evidence completeness
        |
        v
model receives temporal state
```

The kernel does **not** decide whether something is stale unless an external task actually specifies a validity threshold. On human-preference benchmarks, deriving `stale=true` from the target labels would be cheating.

See [`docs/ARCHITECTURE_PIVOT_TEMPORAL_KERNEL.md`](docs/ARCHITECTURE_PIVOT_TEMPORAL_KERNEL.md).

---

## The first real external gate: TicToc

TicToc's raw scenarios are extremely useful for H1. A conversation can have three alternate final timestamps while its semantic content remains fixed; human preferences for direct answer vs tool refresh are collected separately for the time variants.

The official timestamp condition exposes wall-clock timestamps as text. WidePresent now asks a narrower question:

> **Is the same temporal information more usable when exact elapsed durations are derived by the runtime instead of requiring the LLM to subtract timestamp strings?**

`experiments/tictoc_temporal_kernel_adapter.py` prepares this condition without copying the external dataset and without using human preference labels.

Compare:

```text
A. no temporal information
B. passive timestamp text                 [official TicToc condition]
C. deterministic derived temporal state  [WidePresent H1]
D. learned/wide temporal state            [only later]
```

The crucial first comparison is **B vs C**.

If C does not beat B, stop. There is no reason to build a fancy temporal representation for this failure mode.

See [`docs/TICTOC_GATE_PLAN.md`](docs/TICTOC_GATE_PLAN.md) and [`docs/PREREG_GATE_1C_TOKEN_TIME.md`](docs/PREREG_GATE_1C_TOKEN_TIME.md).

---

## Gate 1A pilot: rate-warp forecasting

Before finding TicToc, we ran a small equal-information synthetic forecast pilot. Models observe an irregularly sampled continuous signal and predict a fixed future horizon under held-out sampling rates.

Three-seed exploratory mean RMSE:

| model | IID | slow OOD | fast OOD |
|---|---:|---:|---:|
| Event GRU | 0.6862 | 0.7037 | 0.6949 |
| dt-GRU | 0.6868 | 0.7009 | 0.7051 |
| Timestamp Transformer | 0.6756 | 0.6992 | 0.6473 |
| Same-grid GRU | 0.6829 | 0.6995 | 0.6592 |
| Wide matrix MLP | **0.6692** | **0.6956** | **0.6352** |

The Wide matrix is numerically best in this small pilot, especially at fast-rate OOD, but the margin over the timestamp Transformer is small and seed-sensitive. Fixed binning also gives free denoising, and no LMU/HiPPO control has run.

**Recorded verdict: not a positive result.**

See [`docs/GATE1A_PILOT.md`](docs/GATE1A_PILOT.md).

---

## What `widepresent.py` now means

The original v0.1 code still implements the simple one-axis past/now/future register:

```text
[-past ... -2 -1] [0] [+1 +2 ... +future]
        retention  now      prediction
```

A future prediction advances toward zero and becomes due at `now`.

Keep it as the historical minimal substrate. It is **not** currently the preferred final architecture.

The more serious state is in `bitemporal_present.py`.

---

## Hypothesis ladder

The project now has explicit places to die:

```text
H0  event count != elapsed time                         known / sanity only
H1  derived temporal state > passive timestamp text     current external gate
H2  world-time + knowledge-time > scalar age kernel     open
H3  wide relative-time projection > ordinary kernel     open
H4  future-coordinate rendezvous > normal forecasting   open
H5  cyclic/oscillatory geometry adds value               quarantined
H6  operational temporal self-location                  descriptive only
H7  consciousness                                        out of scope
```

See [`docs/HYPOTHESIS_LADDER.md`](docs/HYPOTHESIS_LADDER.md).

---

## Relationship to earlier Antti Luode repos

They are useful as an idea generator and negative-results archive, not as axioms.

- **KYY** is a reminder to use strong algebraic / ordinary controls before calling geometry special.
- **Visertäjä** is a useful negative precedent: oscillator trajectories trained, but a parameter-matched GRU beat them on the temporal discriminator.
- **Clockfield / Liquid-NN-With-Adaptive-Local-Time** explored content-dependent clocks. WidePresent currently makes the opposite foundational choice: the objective base clock is not negotiable by content.
- **GeometricNeuronPlusField** motivates distributed dynamic representations, but none of its biological/geometric conclusions are imported.

If an old mechanism eventually helps, it must enter because a registered WidePresent gate exposes a specific failure that mechanism addresses.

---

## Run local sanity experiments

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python experiments/gate0_clock_vs_event.py
python experiments/demo_prediction_rendezvous.py
python experiments/gate1b_late_evidence_sanity.py
```

A GitHub Actions workflow is included, but connector visibility of Actions is currently unavailable, so the existence of the workflow is not being reported as a successful CI run.

---

## Current status

The project became **less grand and more interesting** during its first research pass.

The original claim "give AI a present" collided with substantial prior art. What survived is sharper:

> **Many online-agent failures may come not from an absence of temporal information, but from representing time as passive context rather than maintained state.**

That hypothesis is externally testable now.

And if a deterministic temporal kernel solves it, that is the result. We do not need a temporal neuron merely because the bicycle was a good way to notice the problem.
