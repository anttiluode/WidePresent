# WidePresent

**A research program on giving online AI an explicit, temporally extended present.**

This repository starts from a narrow engineering question, not a consciousness claim:

> **Does an online agent benefit from carrying state in a content-independent temporal coordinate frame with a moving `now`, rather than letting event/token order stand in for elapsed time?**

The motivating image was a bicycle drivetrain. The chain has fixed pitch and keeps advancing; one link is at the engagement point now, links behind it are recent history, links approaching it can carry predictions. The metaphor is disposable. The testable architecture is not.

## Claim boundary first

None of these ingredients is new by itself:

- recurrent nets already carry state;
- Clockwork RNNs assign modules prescribed clock rates;
- Phased LSTMs use oscillatory time gates and handle asynchronous events;
- Time2Vec and many irregular-time models explicitly encode timestamps;
- LMU/HiPPO maintain mathematically principled representations of continuous history over sliding windows;
- predictive-processing and active-inference models integrate beliefs about past and future;
- hippocampal time cells and theta sequences provide biological examples of explicit temporal organization;
- phenomenology has discussed an extended or "specious" present for more than a century.

So **WidePresent currently claims no new primitive**.

The potentially useful remainder is the combination:

```text
                 absolute / wall-clock time
                         |
                         v
past retention  <----  NOW  ---->  scheduled prediction
(age-indexed)          (zero)         (time-to-arrival indexed)
                         |
                         v
                prediction matures
                and meets observation
```

with one hard constraint:

> **Content may change what is represented at a time coordinate, but content does not get to decide whether time advanced.**

That makes `now` an architectural coordinate, not a word the model can merely emit.

## Why investigate this now?

Norman-Haignere et al. (Nature Neuroscience, 2025) directly separated **time-yoked** integration (fixed absolute duration) from **structure-yoked** integration (windows stretching with phonemes/words). Human auditory cortex was predominantly time-yoked, with only a very small change in integration window under a threefold change in speech-structure duration. In their DeepSpeech2 analysis, however, training produced a transition toward structure-yoked integration across network layers.

That result does **not** imply that human-like cognition requires a WidePresent. It gives us a clean experimental axis: *time vs. structure yoking* can be measured rather than argued about.

A second motivation is practical. Token-centric systems can receive two adjacent symbols separated by 20 ms or 20 hours unless timestamps or an external process tell them otherwise. This is not true of all AI: sampled audio systems, robotics stacks, continuous-time models and timestamp-aware agents can represent objective time. WidePresent targets the narrower failure mode where **event order is allowed to masquerade as elapsed time**.

## Gate 0 — passed, but deliberately weak

`experiments/gate0_clock_vs_event.py` creates streams where the class is defined only by elapsed duration. During training, event density is spuriously correlated with duration; at OOD test the correlation is reversed.

Seed 0:

```text
event_index_iid        0.9440
event_index_ood        0.0570
timestamp_iid          1.0000
timestamp_ood          1.0000
fixed_tick_iid         1.0000
fixed_tick_ood         1.0000
```

This proves only the obvious prerequisite: **event count is not a clock**. A model given exact timestamps solves the task just as well as a fixed-tick representation. Therefore Gate 0 is *not* evidence that WidePresent is better than existing time-aware models.

That equality is important. The real research begins at Gate 1.

## The first falsifiable WidePresent

The minimal substrate in `widepresent.py` has three regions:

```text
[-past ... -2 -1] [0] [+1 +2 ... +future]
        retention  now      prediction
```

Every coordinate is an integer multiple of a fixed `dt`.

A prediction placed at `+k` is not simply read immediately. The external clock advances. The prediction approaches zero. When it reaches `now`, it is returned as a **matured prediction** and can be compared with the actual observation.

That gives a very simple primitive:

```text
forecast(t + tau)  --time-->  now  <-->  observation(t + tau)
```

`experiments/demo_prediction_rendezvous.py` demonstrates this mechanically.

## Research ladder

### Gate 1 — equal timing information

Compare WidePresent against models that receive the **same timing information**:

- GRU/LSTM + exact `dt`;
- Time2Vec-style timestamp encoding;
- LMU/HiPPO continuous-history memory;
- a continuous-time ODE/CDE baseline where practical;
- a small Transformer with explicit elapsed-time features.

Tasks must include rate warps, long silent intervals, asynchronous modalities and deadline-sensitive prediction. Parameter count, state size and compute are reported.

**Kill condition:** if explicit timestamps or a continuous-time baseline erase the WidePresent advantage, we say so. Then the useful contribution may be an interface/diagnostic representation rather than a better sequence model.

### Gate 2 — prediction rendezvous

Test the one feature that is less common as an architectural primitive: forecasts live at **future time coordinates** and later mature into present observations.

Candidate tasks:

- irregularly sampled moving-object interception;
- audio/video streams with independent clocks and controlled offsets;
- delayed actuator outcomes;
- branching forecasts where several future slots carry distributions, not point predictions.

Metrics include timing error, value error, calibration, and robustness to changes in event density.

### Gate 3 — temporal provenance

Ask whether a fixed temporal coordinate frame reduces source confusion among:

```text
observed-now
remembered-past
predicted-future
retrieved-old-memory
```

The content can be identical. The source/time coordinate differs.

### Gate 4 — TCI-style model microscope

Adapt the temporal-context-invariance logic of Norman-Haignere et al. to hidden representations. Measure each model's integration window under stretched and compressed structure and compute a structure-yoking index.

This is a **measurement**, not a target. We should not force a model to look brain-like and then celebrate that it does.

### Gate 5 — only then ask about a "present"

If the architecture earns practical behavior, test phenomena usually discussed under temporal cognition:

- duration judgements;
- empty intervals / waiting;
- temporal-order errors;
- prediction-vs-observation source errors;
- interruption and staleness;
- prospective vs retrospective timing.

Even a positive result would establish temporal organization or temporal self-location, **not phenomenal consciousness**.

## Relationship to earlier Antti Luode repositories

Earlier repositories are treated as an idea generator and negative-results archive, not as axioms.

- **KYY** suggests useful controls for persistent structured state and taught us to compare geometry against strong algebraic baselines.
- **Visertäjä** showed that oscillator/phase trajectory state can train, but its temporal discriminator lost to a parameter-matched GRU. That negative result is a reason *not* to assume that dynamic-looking state is useful.
- **Liquid-NN-With-Adaptive-Local-Time / Clockfield** explored content-dependent local update rates. WidePresent currently makes the opposite foundational choice: the base clock is content-blind. Content-dependent gating, if used at all, sits *inside* the temporal scaffold.
- **GeometricNeuronPlusField** and related work are inspiration for thinking in distributed state and modes, but no biological/geometric claim is imported here.

The project rule is simple:

> **Borrow mechanisms, not conclusions. Re-test everything that matters.**

## Run

```bash
pip install -r requirements.txt
python experiments/gate0_clock_vs_event.py
python experiments/demo_prediction_rendezvous.py
```

## Key prior art / starting references

See [`docs/PRIOR_ART_MAP.md`](docs/PRIOR_ART_MAP.md) and [`docs/PREREG_GATE_1.md`](docs/PREREG_GATE_1.md).

The current status is intentionally modest:

> **We have a crisp question, a minimal clock-first substrate, one sanity gate, strong prior-art controls, and several easy ways for the idea to die.**
