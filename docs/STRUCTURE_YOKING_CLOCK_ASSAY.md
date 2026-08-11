# Clock versus structure yoking — the skipped assay

Date: 2026-08-11

This experiment returns to the question that got skipped while WidePresent drifted into provenance/database territory:

> **Does a content-blind clock change a network's own time-yoked versus structure-yoked integration behavior, and does that matter under rate shift?**

The answer is now split in two:

1. a hard clock-constrained fading state can enforce time-yoking and gains exactly on time-yoked rate shifts;
2. merely exposing a free recurrent network to blank clock ticks does **not** guarantee a useful time-yoked representation.

The experiments are:

- `experiments/structure_yoking_clock_assay.py` — transparent scalar proof-of-mechanism;
- `experiments/clock_gru_yoking_attack.py` — learned GRU attack.

## 1. Why this assay is different from Gate 0

Gate 0 merely showed that event count is not elapsed time. A timestamp or fixed tick trivially solved a duration task.

This assay instead asks about **integration width**.

Thirty-two signed observations must be integrated with a fading memory. At the nominal inter-event gap of `0.5 s`, two possible worlds are deliberately identical:

### Time-yoked world

\[
y=\operatorname{sign}\left(\sum_i x_i e^{-(T-t_i)/\tau}+\epsilon\right),
\qquad \tau=2.5\;\text{s}.
\]

### Structure-yoked world

\[
y=\operatorname{sign}\left(\sum_i x_i e^{-d_i/\kappa}+\epsilon\right),
\qquad \kappa=\tau/0.5=5\;\text{events}.
\]

At exactly `0.5 s/event`, these weighting functions are the same.

So **the complete nominal training dataset cannot reveal whether the true integration rule belongs to seconds or event positions**.

Only a rate shift can reveal the inductive bias.

## 2. Scalar constrained models

All three systems have one fading scalar state plus the same linear binary readout.

### Event-yoked

\[
h_j=\alpha h_{j-1}+x_j.
\]

Decay happens only when content arrives.

If the inter-event gap changes, its physical time constant changes with it.

### Explicit-dt

\[
h_j=e^{-r\Delta t_j}h_{j-1}+x_j.
\]

This is the boring continuous-time attacker. It is event-driven but knows elapsed time.

### Content-blind clock

The state decays on a fixed `0.1 s` clock even when no content arrives. Between two content events separated by `dt`, the accumulated decay is

\[
\alpha_{tick}^{dt/0.1}.
\]

Content cannot vote to stop or stretch the clock.

## 3. Structure-yoking index

Following the useful convention in Norman-Haignere et al. (2025), compressed and stretched conditions differ symmetrically by `sqrt(3)`, giving a factor-of-three difference in structure duration.

For physical integration width `W`, define

\[
SYI=\frac{\log(W_{stretched}/W_{compressed})}{\log 3}.
\]

Then:

- `SYI = 0`: absolute-time-yoked;
- `SYI = 1`: structure/event-yoked.

In the deliberately transparent scalar system:

| model | SYI |
|---|---:|
| event decay | **1.000** |
| explicit dt | **0.000** |
| content-blind clock | **0.000** |

This part is architectural, not an empirical discovery. A per-event leak necessarily stretches in physical time when events stretch; a fixed clock or explicit continuous-time decay does not.

## 4. Five-seed scalar result

Training uses only the nominal `0.5 s/event` condition. The same trained model is then evaluated in compressed (`0.289 s/event`), nominal (`0.5 s/event`), and stretched (`0.866 s/event`) conditions.

### If the real target is time-yoked

| model | compressed | nominal | stretched |
|---|---:|---:|---:|
| event-yoked | 0.904 | 0.940 | 0.885 |
| explicit dt | **0.952** | 0.940 | **0.928** |
| content-blind clock | **0.952** | 0.940 | **0.928** |

### If the real target is structure-yoked

| model | compressed | nominal | stretched |
|---|---:|---:|---:|
| event-yoked | **0.938** | 0.939 | **0.942** |
| explicit dt | 0.895 | 0.939 | 0.898 |
| content-blind clock | 0.897 | 0.939 | 0.896 |

Standard deviations across five seeds were roughly `0.002--0.005`.

## 5. The scalar result is symmetric

A fixed clock is **not universally better**.

It is better when the task's relevant integration horizon is tied to objective seconds.

It is worse when the task's relevant integration horizon is tied to structural/event distance.

At the nominal rate, all systems look equally good because time and structure are confounded.

Rate shift reveals which invariant the architecture has chosen.

This is the empirical hook that had been skipped.

## 6. The boring attacker wins against clock uniqueness

The explicit-`dt` system and the fixed clock are almost numerically identical.

That means the scalar assay does **not** establish that an independently ticking substrate is necessary.

A much cheaper implementation can preserve the same absolute-time invariant:

```text
when an event arrives:
    decay state by exp(-r * elapsed_seconds)
    then ingest the event
```

No blank clock updates are required.

So the scalar claim is not:

> WidePresent needs a continuously ticking neural layer.

It is:

> **A model must choose, or learn, what its integration width is yoked to. If objective-time yoking matters under rate shift, event count alone is the wrong invariant.**

## 7. Harder learned-GRU attack

The scalar system is constrained to implement the intended leak, so the next test gave the network freedom to organize its own recurrent state.

All GRUs were trained only at the nominal rate, where time-yoked and structure-yoked targets are indistinguishable.

### Models

`event_gru`
: input is one content value per recurrent update.

`dt_gru`
: input is `[content_value, dt]` once per event. During training, `dt=0.5` is constant, so the network is free to ignore it.

`clock_gru`
: input is `[content_value, content_mask]` on a fixed `0.1 s` clock. Blank ticks occur between events. Under rate shift, the number of blank ticks changes.

### Measured receptive-field yoking

The final-logit gradient with respect to each previous content value was used as a perturbational influence measure. Two integration widths were computed:

- the physical age containing 80% of total gradient mass;
- gradient-weighted mean age.

Three exploratory seeds gave:

| model | SYI, 80% width | SYI, mean age |
|---|---:|---:|
| event GRU | **1.000 ± 0.000** | **1.002 ± 0.007** |
| dt GRU | 0.906 ± 0.066 | 0.917 ± 0.020 |
| blank-tick clock GRU | **0.513 ± 0.288** | **0.510 ± 0.246** |

This result is important for two reasons.

First, merely handing a GRU `dt` does not force it to become time-yoked. At a single training rate, the scalar is constant and the network mostly learns an event-yoked receptive field anyway.

Second, blank clock ticks **do change the learned yoking**: the clock GRU moves substantially toward absolute-time yoking.

But that change is unstable across seeds and, crucially, does not buy rate-shift robustness.

## 8. Learned-GRU accuracy

### Time-yoked target

| model | compressed | nominal | stretched |
|---|---:|---:|---:|
| event GRU | 0.901 | 0.937 | 0.891 |
| dt GRU | 0.909 | 0.937 | 0.888 |
| blank-tick clock GRU | **0.861** | 0.931 | **0.803** |

### Structure-yoked target

| model | compressed | nominal | stretched |
|---|---:|---:|---:|
| event GRU | **0.936** | 0.939 | **0.938** |
| dt GRU | 0.928 | 0.938 | 0.925 |
| blank-tick clock GRU | **0.867** | 0.930 | **0.814** |

So the naive clock-GRU is not rescued by having a more time-like receptive field.

It overfits the training rhythm: five blank ticks per content event. Changing that rhythm produces a new recurrent regime that the network never learned to use.

The variance in clock-GRU SYI across seeds is itself evidence against treating blank ticks as a hard invariant.

## 9. Revised distinction: clock exposure versus clock constraint

This is now the key conceptual result of the yoking branch.

```text
CLOCK EXPOSURE
    provide dt or blank ticks
    learned network may ignore, reinterpret, or overfit them

CLOCK CONSTRAINT
    force part of state evolution / forgetting / receptive width
    to advance according to objective elapsed time
```

The scalar clock/continuous-time leak is a **constraint**.

The clock-GRU is mostly **exposure**.

Only the former reliably produces `SYI=0` in this assay.

That gives a precise meaning to the morning phrase:

> **a clock attention cannot override**

If this idea is worth pursuing, the clock must define an invariant part of the memory dynamics rather than merely becoming another feature available to attention or recurrence.

## 10. Relation to the Norman-Haignere work

Norman-Haignere et al. (Nature Neuroscience, 2025, DOI `10.1038/s41593-025-02060-8`) tested absolute-time versus structure-yoked integration by stretching and compressing speech. They define a structure-yoking index near `0` for time-yoked integration and near `1` for integration windows that scale with structure duration.

Their biological result is not evidence for WidePresent. It supplies a clean assay concept.

Skrill & Norman-Haignere (NeurIPS 2023) reported a transition in language models from more position-yoked integration in earlier layers toward more structure-yoked integration in later layers.

The WidePresent question is different:

> can an external/content-blind temporal invariant stop a learned system from silently converting objective seconds into event/structure distance?

The current answer is:

- hard continuous-time decay: yes mechanically;
- passive `dt` feature: not at a single rate;
- blank clock ticks: they alter yoking, but not reliably or beneficially enough.

## 11. What should be attacked next

The strongest next model is not another blank-tick GRU.

It is a **hybrid network with two deliberately separated paths**:

```text
content path
    free learned recurrence / attention

absolute-time path
    deterministic or constrained continuous-time decay
    whose width cannot be changed by content attention

readout
    may combine both
```

Then ask whether the network learns to use the fixed-time path only on time-yoked tasks while retaining its free structural path on structure-yoked tasks.

This is much stricter than saying "add a clock token."

Primary attackers must include:

1. standard continuous-time RNN/SSM with explicit `dt` dynamics;
2. learned multi-timescale exponential memory;
3. attention with relative physical-time bias;
4. ordinary event-position attention;
5. matched hybrid model where both time and structure paths are non-geometric.

If those match the proposed WidePresent path, there is no special architectural claim.

## 12. Status

**Positive proof-of-mechanism for the yoking distinction. Negative result for naive clock exposure. Architectural question still open.**

The surviving question is now sharper than "does AI know time passes?":

> **Can a model preserve a content-independent absolute-time integration channel while simultaneously retaining structure-yoked processing, and does that mixed invariant improve rate-shift generalization beyond ordinary continuous-time baselines?**
