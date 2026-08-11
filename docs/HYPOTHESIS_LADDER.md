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

The repo now has two external-facing language/tool gates:

- `experiments/run_tictoc_b_vs_c.py` for the published TicToc timestamp-vs-derived-kernel comparison;
- `experiments/language_tool_validity_benchmark.py` for paired `raw` / `age_plane` / `resolver` tool-use decisions.

Both are prepared and scored reproducibly. Genuine external LLM sampling is still pending in the current ChatGPT execution environment because no API key/local language model is available. No proxy classifier is substituted for that missing result.

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

The later identifiability work sharpens the interpretation further:

> at one fixed rate, wall-time age and event distance can be statistically inseparable even when both are represented perfectly.

So "the model saw timestamps" is not equivalent to "the data identified which temporal coordinate should govern memory."

## H2 — bitemporal state matters beyond scalar age

World/event time and knowledge/arrival time are different. Late observations and retrieved old memories make that distinction unavoidable.

**Primary adversaries:** timestamped sequence models, delayed-observation filters, ordinary stream-processing/database logic.

**Positive criterion:** explicitly exposing both coordinates to the learned agent reduces source confusion or improves delayed-observation decisions beyond a scalar-age kernel.

**Kill:** ordinary delay-aware filtering / bookkeeping is sufficient.

No novelty in bitemporality itself.

The language counterfactual weather pair now gives a direct behavioral version of this point:

```text
arrival age held fixed
world/valid age crosses the freshness boundary
```

A policy based on arrival recency cannot solve that pair.

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

The language branch now attacks that claim at three increasingly strict levels:

1. `experiments/language_tool_validity_benchmark.py`
   - known validity contract;
   - compare raw timestamps, deterministic age-plane arithmetic, and external resolver.

2. `experiments/counterfactual_language_tool_pairs.py`
   - causal flip pairs and non-causal invariance pairs;
   - weather/world-time, discourse/event-distance, reservation/until-change;
   - behavioral fingerprint against simple temporal theories;
   - calendar-shift metamorphic invariance.

3. `experiments/language_semantics_discovery_benchmark.py`
   - arbitrary source labels;
   - narrow audit experience where seconds/events are underidentified;
   - wide rate-diverse experience where the correct semantic coordinate becomes identifiable;
   - final hidden cases matched exactly across narrow/wide experience.

The practical implementation is now `temporal_validity.py`, which turns the surviving idea into an explicit runtime contract:

```text
REPRESENTATION
    valid time
    known time
    structural index
    source version/change state

VALIDITY SEMANTICS
    WorldTimeTTL
    EventDistanceTTL
    UntilChange
    ExponentialAgePlane / other source models

DECISION
    P(valid now)
    + stale-reuse cost
    + refresh cost
    -> REUSE / REFRESH
```

This is currently a more defensible engineering endpoint than a special WidePresent neural layer.

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

The current decomposition is:

```text
1. REPRESENTATION
   preserve the candidate coordinates

2. IDENTIFICATION
   experience must vary enough to determine which coordinate matters

3. DECISION
   map validity uncertainty and refresh cost into action
```

Many earlier WidePresent branches mixed these three problems together.

### H3a — semantic uncertainty while identification is incomplete

**Status: useful runtime result; standard Bayesian/robust machinery.**

`experiments/semantic_uncertainty_attack.py` compares hard semantic selection, Bayesian model averaging and a conservative worst-case policy when narrow-rate training leaves wall-time versus event-distance semantics unresolved.

Under narrow-rate OOD, hard MAP semantic selection has higher action agreement but lower expected utility than Bayesian model averaging because a wrong hard commitment causes expensive stale reuse. A robust worst-case policy nearly eliminates stale reuse but refreshes substantially more often.

With wide rate-diverse experience the posterior collapses onto the correct source semantics and MAP, model averaging, robust prediction and the oracle-axis reference become essentially identical.

So uncertainty machinery is justified only while the semantics are genuinely unresolved.

Implementation: `temporal_validity_learning.py`.

### H3b — learned temporal semantics can drift

**Status: rolling forgetting positive; explicit drift detector not earned.**

`experiments/semantic_drift_attack.py` switches one source from a world-time hazard to an event-distance hazard halfway through a rate-diverse online stream.

A forever-cumulative posterior develops semantic inertia: after the switch, most seeds do not reach `P(new semantic)>0.90` within the remaining stream. A rolling `240`-audit posterior adapts in all reported seeds and approaches the utility of an oracle reset that knows the true switch time.

An extra confidence-triggered sentinel/reset mechanism did not cleanly beat the rolling posterior and introduced threshold/false-reset tradeoffs, so it is not part of the default runtime.

Implementation: `temporal_validity_online.py`.

This adds a fourth stage:

```text
4. NONSTATIONARITY
   old semantic certainty must be allowed to expire
```

### H3c — active semantic probing

**Status: free-probe positive; shared-budget default negative.**

`experiments/active_semantic_identification.py` shows that, given an equal *separate* diagnostic-probe budget, querying cases where wall-time and event-distance models disagree identifies the true semantic rule far faster than random probes or probes chosen only because current `P(valid)` is near the reuse threshold. Exact semantic information gain helps most at very small budgets, while simple disagreement nearly catches it at moderate budgets.

However, `experiments/active_probe_budget_arbitration.py` removes the free diagnostic budget. Each round has six cached requests and only one refresh/tool call. When semantic probing competes directly with protecting the riskiest current request, dedicated explore-then-exploit scheduling learns the semantic rule faster but loses total operational utility over the tested `20..320` round horizons.

The myopic risk policy learns the source semantics incidentally from the refreshes it already needed to make.

Therefore the current default is:

```text
refresh by immediate expected operational risk
learn source semantics opportunistically from refresh outcomes
add dedicated semantic probes only when a separate value-of-information case is demonstrated
```

Implementation: `temporal_validity_active.py`.

This kills the temptation to turn every semantic uncertainty into a new active-exploration controller.

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