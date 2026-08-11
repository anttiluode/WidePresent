# Gate 1C preregistration — decorrelate token time from wall-clock time

Date registered: 2026-08-11

## Motivation

The motivating failure is not ordinary date arithmetic. It is the possibility that a conversational agent treats sequence progression as a proxy for elapsed world time.

Recent work already makes this testable:

- Wang et al. (EMNLP Findings 2025) propose a Token-Time Hypothesis and show that LLMs can extract some elapsed-time information from token counts.
- Cheng et al. (ACL Findings 2026) show poor time-sensitive tool-use alignment in multi-turn agents; timestamp text helps only modestly.
- Sehgal et al. (2026 preprint) report much better strategic behavior when remaining wall-clock time is explicitly refreshed than when agents are merely told an initial deadline.

WidePresent therefore should not test "can a model do temporal arithmetic?" It should **break the correlation between tokens and time**.

## Core paired design

Construct paired interaction histories with:

- identical message text;
- identical token counts;
- identical turn count;
- identical tool results;
- different silent wall-clock gaps between turns.

Only the objective elapsed time differs.

The correct action depends on that elapsed time.

Example abstractly:

```text
previous observation validity: 10 minutes
conversation content: identical

arm FRESH: user returns after 30 seconds  -> reuse observation
arm STALE: user returns after 3 hours     -> refresh observation
```

No semantic cue in the text is allowed to reveal which arm is active.

## Conditions

### A. Stationary context
No timestamps or temporal side channel.

This condition is informationally incapable of distinguishing a perfectly paired fresh/stale item. It is a negative control, not a serious model.

### B. Timestamp text
Attach ISO timestamps to messages/tool results and `now`.

The model must convert timestamp relations into the action policy.

### C. Derived temporal kernel
Provide structured fields such as:

```text
world_age_seconds
knowledge_age_seconds
freshness_threshold_seconds
stale
remaining_deadline_seconds
```

No special learned time architecture.

### D. WidePresent state
Provide a moving temporally typed working projection or learned temporal state, with the same source timing information as C.

## Task families

1. **Staleness / tool refresh** — whether a previous observation should be reused.
2. **Deadline policy** — same interaction state, different remaining wall time.
3. **Late evidence** — whether absence can be treated as evidence of absence.
4. **Prediction due state** — forecast target time passed vs not yet reached.
5. **Multi-source freshness** — two facts have identical content but different world/knowledge ages.

## Metrics

- paired consistency: fraction of fresh/stale pairs receiving the two different required actions;
- balanced action accuracy;
- unnecessary refresh rate;
- stale-reuse rate;
- temporal arithmetic error when explicit ages are requested;
- calibration where the policy is probabilistic rather than thresholded.

## Primary comparisons

The central comparison is **B vs C**.

If timestamp text performs as well as derived temporal state, the temporal kernel adds little.

The architectural comparison is **C vs D**.

If the deterministic kernel matches WidePresent, the learned moving state has not earned its keep.

## Kill conditions

1. Timestamp augmentation reaches the same paired accuracy as structured temporal state across task families.
2. Deterministic temporal-kernel features match or beat the learned WidePresent state under matched source information.
3. Any apparent WidePresent advantage disappears when token count, turn count, content and time information are exactly matched.

## What a positive result would mean

A result where C > B would support a narrow engineering claim:

> temporal information is more reliable for agents when converted into explicit dynamic state than when left as passive timestamp text.

A further D > C result would be required before claiming a learned WidePresent representation adds value.

Neither result would establish consciousness, subjective duration, or a human neural mechanism.
