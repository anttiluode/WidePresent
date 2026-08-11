# Hypothesis ladder — what is actually left to discover?

Date: 2026-08-11

This ladder exists to stop the project from sliding from a useful engineering result into a larger claim by rhetoric.

## H0 — event count is not elapsed time

**Status: known / sanity-checked.**

When event density changes, counting updates is an unreliable proxy for objective duration. Gate 0 demonstrates the trivial case. Existing timestamp-aware and continuous-time models already solve this class of problem.

No novelty.

## H1 — derived temporal state is more usable than passive timestamp text

Instead of making the language model subtract timestamps and repeatedly rediscover age/freshness-related quantities, the runtime supplies label-blind derived quantities such as:

```text
current decision time
elapsed since each prior observation
elapsed since latest tool result
world age vs knowledge age
incomplete vs complete evidence interval
future deadline / time-to-due
```

**Primary adversary:** TicToc timestamp condition.

**Positive criterion:** on the same benchmark samples and with the same underlying clock information, derived temporal state materially improves normalized human alignment / paired consistency over timestamp text.

**Kill:** timestamp text matches it.

This would be an engineering result, not a new theory of time.

## H2 — bitemporal state matters beyond scalar age

World/event time and knowledge/arrival time are different. Late observations and retrieved old memories make that distinction unavoidable.

**Primary adversaries:** timestamped sequence models, delayed-observation filters, ordinary stream-processing/database logic.

**Positive criterion:** explicitly exposing both coordinates to the learned agent reduces source confusion or improves delayed-observation decisions beyond a scalar-age kernel.

**Kill:** ordinary delay-aware filtering / bookkeeping is sufficient.

No novelty in bitemporality itself.

## H3 — a wide relative-time working projection adds value

Project the bitemporal ledger into a bounded state centered on now, with rows/slots representing relative world time and channels representing observation, prediction, knowledge age, uncertainty and completeness.

**Primary adversaries:**

- temporal kernel scalar/vector side-channel;
- timestamp Transformer;
- LMU / HiPPO continuous history;
- Time-Aware World Model style `dt` conditioning;
- delay-aware belief filters.

**Positive criterion:** the explicit relative-time geometry gives better OOD rate/delay robustness, calibration or source/provenance discrimination under matched information/resources.

**Kill:** kernel or established continuous-time memory matches it.

This is the first point at which the particular WidePresent representation might earn an architectural claim.

## H4 — future coordinates / prediction rendezvous add value

Predictions are stored at their future-valid world times and become due as `now` reaches those coordinates. Evidence may arrive after the target time, preserving both target-time and validation-time semantics.

**Primary adversaries:** ordinary multi-horizon forecasting heads, model-predictive control, event queues, probabilistic world models.

**Positive criterion:** better deadline calibration, late-validation handling or temporal provenance than standard forecasting machinery.

**Kill:** standard scheduled forecasts are equivalent.

## H5 — cyclic/oscillatory geometry is useful

Only now would the original bicycle loop, theta-like phase, KYY ring, or Visertäjä oscillators return.

They need a named failure in H1–H4 that a cyclic coordinate plausibly fixes.

**Kill by default:** do not add them merely because they look brain-like or resonate with earlier repos.

## H6 — temporal self-location

If an agent reliably separates:

```text
observed now
recent world state
newly learned old evidence
retrieved memory
prediction not yet due
prediction due but not yet validated
```

then it has an operational form of temporal self-location: its state explicitly says where beliefs sit relative to a moving present.

This is a functional description. It does not imply subjective experience.

## H7 — consciousness

**Out of scope.**

No result in this repository, including success on every gate above, would establish phenomenal consciousness or a felt present.

If the project ever reaches this question, it requires independent theory and evidence rather than renaming temporal state organization as consciousness.
