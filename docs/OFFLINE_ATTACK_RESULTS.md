# Offline-mode attack: provenance and active sensing

Date: 2026-08-11

This note records an explicit attempt to kill the sleep-inspired branch in
`docs/OFFLINE_MODE_SLEEP_ANALOGY.md`.

The result is mostly negative for a special offline/sleep-like mode.

## Question

The earlier boundary-wave toy found:

- continuous uncontrolled input made current-topology decoding difficult;
- quiet helped;
- quiet plus a controlled self-probe helped further.

That could tempt an interpretation like:

> a temporal system benefits from entering a separate offline mode.

Before accepting that, this attack asks whether simpler explanations are sufficient:

1. explicit temporal provenance;
2. ordinary active sensing;
3. a fresh diagnostic query while remaining online.

## Benchmark

`experiments/provenance_vs_offline_attack.py`

This benchmark is deliberately event-based rather than wave-based so that the two
times are exact.

There are three hidden states. The environment switches from an old state to a new
state late in the episode (`t = 82..94`). Sensors generate noisy observations of the
state that was true at their **world/valid time**. Delivery has a long-tailed delay, so
old-state observations can arrive close to the final decision at `t = 100`.

The task is to identify the current state.

The benchmark therefore contains the exact ambiguity WidePresent is interested in:

```text
world / valid time       when the evidence was generated / true
arrival / knowledge time when the evidence became available to the agent
```

## Conditions

### Passive baselines

`normal:order`
: latest 24 arrived labels; no timestamps.

`normal:arrival`
: label counts binned by arrival age only.

`normal:valid`
: label counts binned by world/valid age only.

`normal:bitemp`
: joint bins of world age and delivery delay.

### Mode controls

`quiet:arrival`
: stop generating external observations at `t=92`; no diagnostic probe.

`quiet_probe:arrival`
: same quiet interval, then three noisy diagnostic queries at `t=95,96,97`.

`online_probe:arrival`
: keep external observations running and add the exact same three diagnostic queries.

`online_probe:order`
: same online probe, but remove explicit timestamp features from the readout.

The active probe is intentionally strong and boring. It is not a proposed mechanism.
It is an attacker: if asking the current world directly solves the problem while the
system remains online, a separate sleep-like state is unnecessary on this task.

## Five-seed exploratory attack

`3000` episodes per seed, `35%` held-out test split, logistic regression readout.
Chance is `1/3`.

| condition | mean accuracy | std |
|---|---:|---:|
| normal:order | 0.707 | 0.010 |
| normal:arrival | 0.694 | 0.010 |
| **normal:valid** | **0.747** | 0.009 |
| normal:bitemp | 0.739 | 0.015 |
| **quiet:arrival** | **0.579** | 0.011 |
| quiet_probe:arrival | 0.966 | 0.005 |
| **online_probe:arrival** | **0.975** | 0.005 |
| online_probe:order | 0.965 | 0.006 |

## Verdicts

### V1 — world-time provenance helps

`normal:valid` beats both arrival-time and timestamp-free order baselines by roughly
four to five percentage points.

This supports a narrow WidePresent claim:

> evidence is easier to use when the representation preserves when it was true, not
> merely when it arrived.

This is not a novelty claim; bitemporal systems already make exactly this distinction.

### V2 — the second clock is not needed here

The full bitemporal representation did not beat the valid-time-only representation.

That is a useful negative result.

Arrival/knowledge time can matter in other tasks -- source reliability, latency
prediction, watermark completeness, late corrections -- but this benchmark does not
require it once world time is supplied.

So H2 (`world + knowledge time > scalar/valid time`) is **not supported by this toy**.

### V3 — quiet by itself is harmful

Stopping external observation generation without adding a probe reduces accuracy from
about `0.69--0.75` to `0.58`.

So there is no generic benefit from silence here.

This kills the naive statement:

> fewer incoming events make temporal inference better.

### V4 — active sensing explains the big gain

Quiet plus a fresh diagnostic probe scores `0.966`.

But the **same probe while remaining online** scores `0.975`.

Therefore the gain does not require a separate offline mode on this benchmark.

The strongest current explanation is ordinary active sensing:

> when old delayed evidence makes the current state ambiguous, ask a fresh question
> about the current state.

The result remains high (`0.965`) even when the online-probe readout is stripped of
explicit timestamps, because the fresh probe arrives near the decision and carries
strong current evidence.

## A wave-side attacker that did *not* work

Before building the exact event-time benchmark, we tried a stronger attacker inside
the spring-graph family:

> remain continuously driven, reveal the exact driving input, and recover the hidden
> topology using a transfer-function / cross-correlation estimate.

The naive finite-window estimators used in scratch runs stayed near chance while the
clean self-probe remained strong. We therefore do **not** claim that known input
provenance already replaces quiet in the wave system.

That failure may be numerical / estimator-limited, or it may reflect the fact that the
remote topology is encoded in long transient returns that the simple online estimator
failed to isolate. It is recorded rather than tuned away.

## What survives of the sleep analogy?

Very little is needed.

The previous statement:

> an offline interval may provide a cleaner channel for internal operations

is still mechanically true in the boundary-wave toy, but it is no longer an
architectural hypothesis by itself.

A separate `OFFLINE` state earns its keep only if it does something that cannot be
matched by:

- explicit temporal provenance;
- online active sensing;
- ordinary filtering/change-point detection;
- source-aware system identification;
- a current-state query.

At the moment, the evidence does **not** show that.

## Revised architecture

Do not add a sleep module.

Keep instead two optional operations:

```text
1. provenance-preserving state
   preserve when evidence was true/generated

2. active temporal refresh
   when uncertainty about NOW is high, request / inject fresh current-state evidence
```

These operations can run online.

An offline mode can be reconsidered later only if a task requires internal
reorganization that cannot coexist with online sensing.

## Stronger WidePresent interpretation

The attack leaves a simpler picture:

```text
OLD EVIDENCE arriving now
        +
CURRENT EVIDENCE arriving now
        |
        v
preserve temporal provenance
        |
        +---- if NOW still uncertain ----> actively refresh
```

This is closer to a temporal operating system than to sleep.

## New kill ladder

1. **Passive temporal state:** does valid/world time beat arrival order/timestamps?
2. **Second clock:** does arrival/knowledge time add beyond valid time?
3. **Active refresh:** does asking for fresh evidence beat passive temporal memory?
4. **Wave/delay substrate:** does a physical/dynamical implementation add anything
   beyond an event ledger plus active refresh?
5. **Offline mode:** only test if online refresh cannot perform the required internal
   reorganization.

The sleep analogy is therefore demoted from mechanism to inspiration.
