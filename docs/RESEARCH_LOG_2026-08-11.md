# Research log — 2026-08-11

## Origin

The immediate intuition was a bicycle drivetrain:

- fixed chain pitch suggested a content-independent temporal grain;
- a link at the engagement point suggested a privileged `now`;
- the loop suggested phase / recurrence rather than unbounded history;
- the road suggested a separate monotonic coordinate needed to distinguish successive revolutions;
- links behind and ahead suggested retention and forecast around a temporally thick present.

Useful metaphor, but no mechanical property is accepted as a neural principle merely because bicycles have it.

## First correction

The broad statement "AI lacks time" is false.

Many systems have sample clocks, timestamps, continuous-time dynamics or external schedulers. The narrower concern is **token/event time**: if the internal state updates only when events arrive and no objective time coordinate is compulsory, adjacency in computation can be mistaken for adjacency in the world.

This matches an everyday failure mode of conversational models: without reliable timestamps or an external clock, they can infer wildly wrong elapsed durations from discourse alone.

## First literature collision

The project immediately collides with strong prior art:

- Clockwork RNN — prescribed module clocks;
- Phased LSTM — oscillatory time gate;
- Time2Vec — explicit time embedding;
- LMU/HiPPO — continuous sliding history;
- hippocampal time cells — elapsed-time coding including empty gaps;
- theta sequences — cyclic ordered representations that can extend behind/ahead of current position;
- active-inference computational phenomenology — retention/present/protention already formalized;
- time-consciousness literature — extended/field-like experienced present.

Therefore the bicycle idea is **not** a standalone conceptual discovery.

## The empirical hook that survives

Norman-Haignere et al. supply a measurable distinction between **time-yoked** and **structure-yoked** integration.

Their human auditory-cortex result is strongly time-yoked even in non-primary cortex. Their trained DeepSpeech2 model develops more structure yoking across layers.

This suggests an experiment rather than a story:

> Measure how AI integration windows move when event structure is stretched while objective time remains explicit.

We should not assume time-yoked is universally better. Higher cognition may need both.

## Hypothesis version 0.1

A useful online agent may benefit from **two simultaneous organizations**:

1. **absolute-time scaffold** — content-blind, continuously advancing, supplies age/now/time-to-arrival;
2. **structure/content computation** — adaptive attention, recurrence, gating, semantic segmentation, memory retrieval.

The absolute scaffold is not allowed to disappear just because the content layer finds a convenient event boundary.

This is deliberately weaker than "the brain works this way" and much weaker than a consciousness claim.

## Gate 0 result

Synthetic duration classification with a reversed event-density confound:

```text
event_index_iid        0.9440
event_index_ood        0.0570
timestamp_iid          1.0000
timestamp_ood          1.0000
fixed_tick_iid         1.0000
fixed_tick_ood         1.0000
```

Interpretation:

- token/event count can look like an excellent clock until event rate shifts;
- elapsed time fixes the problem;
- fixed ticks offer no advantage over a direct timestamp on this trivial task.

That last result is the important control.

## Next move

Do not add geometry, oscillators, Clockfield, AIS, theta/gamma or consciousness machinery yet.

Build Gate 1 with equal timing information and try to kill WidePresent using boring baselines.
