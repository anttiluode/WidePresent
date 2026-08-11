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

## H1.5 — what is the integration window yoked to?

A model can integrate over a fixed number of events/structural units or over a fixed amount of objective time. Those are indistinguishable at one fixed presentation rate and diverge only under rate shift.

**Status: positive proof-of-mechanism, architectural claim not earned.**

`experiments/structure_yoking_clock_assay.py` constructs a deliberately symmetric task in which time-yoked and structure-yoked targets are identical at the nominal training rate. Under a factor-of-three structure-duration shift:

- per-event fading state has structure-yoking index `1`;
- explicit-`dt` fading state has index `0`;
- content-blind fixed-clock fading state has index `0`.

For an absolute-time-yoked target, explicit `dt` / clock state generalizes better under compression/stretching. For a structure-yoked target, the event-yoked state wins by a comparable margin.

So the useful question is not "is a clock better?" It is:

> **Which invariant should the task preserve when content rate changes?**

**Primary adversaries:** ordinary continuous-time RNN/SSM state, elapsed-`dt` conditioning, timestamp-aware decay, learned adaptive receptive fields.

**Positive criterion for anything specifically WidePresent-shaped:** on a network free to organize its own receptive field, an absolute/content-blind clock constraint changes the measured yoking index in a way that predicts rate-shift performance and beats equally informed `dt`/continuous-time baselines.

**Kill:** explicit `dt` or a standard continuous-time state learns the same invariant with equal robustness/resources.

The scalar assay currently favors that kill: explicit `dt` and fixed ticking are nearly identical.

## H2 — bitemporal state matters beyond scalar age

World/event time and knowledge/arrival time are different. Late observations and retrieved old memories make that distinction unavoidable.

**Primary adversaries:** timestamped sequence models, delayed-observation filters, ordinary stream-processing/database logic.

**Positive criterion:** explicitly exposing both coordinates to the learned agent reduces source confusion or improves delayed-observation decisions beyond a scalar-age kernel.

**Kill:** ordinary delay-aware filtering / bookkeeping is sufficient.

No novelty in bitemporality itself.

## H3 — a wide relative-time working projection adds value

The original H3 imagined a bounded matrix centered on `now`. The yoking work has narrowed this into a smaller candidate representation: preserve multiple temporal coordinates such as

```text
Δt = valid/world age in seconds
Δn = structural/event-position age
```

until a source/content-specific validity model has enough evidence to decide which transformation actually governs relevance.

**Current status: agent-level representation result; special architecture not earned.**

`experiments/event_agent_age_plane_attack.py` tests an operational `REUSE cached result` versus `REFRESH tool` decision with three simultaneous semantics:

```text
weather   -> validity hazard in wall time
discourse -> validity hazard in intervening event count
state     -> valid until explicit invalidation; no age decay
```

Tool results also have separate valid/world and arrival/knowledge times.

Under episode-level rate shift, a source-conditioned linear age-plane representation is markedly more robust than timestamp-only, position-only, the same raw coordinates without source/age interactions, and the generic boosted tree used in the attack. Under long tool delays, arrival-only recency degrades sharply, again showing that `arrived recently` is not equivalent to `valid recently`.

However, when training contains enough rate diversity to identify the temporal semantics, a boring per-source survival/hazard resolver selects:

```text
weather   -> seconds
discourse -> events
state     -> explicit invalidation
```

in all five reported seeds and beats the age-plane linear policy on the important dense/sparse OOD regimes.

So the current surviving claim is representational/diagnostic:

> **Do not collapse evidence into one scalar notion of age before the task/source semantics identify which temporal coordinate governs validity.**

This is not evidence for a special neural substrate.

**Primary adversaries:**

- source-specific hazard/survival models;
- deterministic validity/freshness resolvers;
- timestamp Transformer / relative-time attention;
- continuous-time RNN/SSM state;
- LMU / HiPPO continuous history;
- delay-aware belief filters.

**Positive criterion for a specifically WidePresent architecture:** a bounded relative-time representation must beat source-aware hazard/validity resolvers and established continuous-time memory under matched information/resources on a language/tool-agent task, not merely on synthetic summaries.

**Kill:** ordinary source-specific validity modeling matches or beats it.

At present, that kill is favored once rate diversity makes the semantics identifiable.

## H4 — future coordinates / prediction rendezvous add value

Predictions are stored at their future-valid world times and become due as `now` reaches those coordinates. Evidence may arrive after the target time, preserving both target-time and validation-time semantics.

**Primary adversaries:** ordinary multi-horizon forecasting heads, model-predictive control, event queues, probabilistic world models.

**Positive criterion:** better deadline calibration, late-validation handling or temporal provenance than standard forecasting machinery.

**Kill:** standard scheduled forecasts are equivalent.

## H5 — cyclic/oscillatory geometry is useful

**Status after the 2026-08-11 matched lake attack: simple version negative.**

`docs/LAKE_VS_LEDGER_ATTACK.md` compared a fixed local spring/wave state against timestamped age bins and matched exponential filters under the same delivered evidence, the same valid-time information, the same linear readout family, and a 60-number state budget.

The pure wave state lost in IID, sparse-rate OOD, dense-rate OOD, and long-delay OOD. Adding a slow diffusive "oil" field recovered the boring filter performance, but the wave coordinates added essentially nothing beyond the slow field.

More strongly, the linear graph-diffusion state is exactly an orthogonal basis transform of independent exponential modes. Its useful fading memory therefore does not require local graph geometry.

So the old permission to reintroduce the bicycle loop, theta-like phase, KYY ring, Visertäjä oscillators, or generic wave interference is withdrawn **by default**.

They may return only if a named later failure exposes something a non-oscillatory temporal filter cannot represent or compute efficiently, for example a genuinely phase-dependent interaction, nonlinear local routing, or a physical implementation constraint.

**Current kill rule:** do not add oscillations merely because they are temporally rich, brain-like, or physically intuitive.

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
