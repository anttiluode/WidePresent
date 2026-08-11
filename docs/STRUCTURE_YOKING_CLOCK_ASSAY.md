# Clock versus structure yoking — the skipped assay

Date: 2026-08-11

This experiment returns to the question that got skipped while WidePresent drifted into provenance/database territory:

> **Does a content-blind clock change a network's own time-yoked versus structure-yoked integration behavior, and does that matter under rate shift?**

The answer in the minimal assay is **yes to the first part, conditionally yes to the second, and no to any claim that fixed ticking is uniquely necessary.**

The experiment is `experiments/structure_yoking_clock_assay.py`.

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

## 2. Models

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

In this deliberately transparent system:

| model | SYI |
|---|---:|
| event decay | **1.000** |
| explicit dt | **0.000** |
| content-blind clock | **0.000** |

This part is architectural, not an empirical discovery. A per-event leak necessarily stretches in physical time when events stretch; a fixed clock or explicit continuous-time decay does not.

## 4. Five-seed exploratory result

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

## 5. The result is symmetric

This is the useful part.

A fixed clock is **not universally better**.

It is better when the task's relevant integration horizon is tied to objective seconds.

It is worse when the task's relevant integration horizon is tied to structural/event distance.

At the nominal rate, all systems look equally good because time and structure are confounded.

Rate shift reveals which invariant the architecture has chosen.

That is exactly the experiment we had not run.

## 6. The boring attacker wins against clock uniqueness

The explicit-`dt` system and the fixed clock are almost numerically identical.

That means this assay does **not** establish that an independently ticking substrate is necessary.

A much cheaper implementation can preserve the same absolute-time invariant:

```text
when an event arrives:
    decay state by exp(-r * elapsed_seconds)
    then ingest the event
```

No blank clock updates are required.

So the current claim is not:

> WidePresent needs a continuously ticking neural layer.

It is:

> **A model must choose, or learn, what its integration width is yoked to. If objective-time yoking matters under rate shift, event count alone is the wrong invariant.**

That is narrower and stronger.

## 7. Relation to the Norman-Haignere work

Norman-Haignere et al. (Nature Neuroscience, 2025, DOI `10.1038/s41593-025-02060-8`) tested absolute-time versus structure-yoked integration by stretching and compressing speech. They define a structure-yoking index near `0` for time-yoked integration and near `1` for integration windows that scale with structure duration.

Their biological result is not evidence for WidePresent. It supplies a clean assay concept.

Skrill & Norman-Haignere (NeurIPS 2023) also reported a transition in language models from more position-yoked integration in earlier layers toward more structure-yoked integration in later layers.

The WidePresent question is different:

> can an external/content-blind clock preserve an absolute-time invariant in a learned system that would otherwise organize memory by event/structure distance?

This toy says yes mechanically, and says that the benefit appears exactly when the task itself is time-yoked.

## 8. What should be attacked next

The scalar leaky-state toy is intentionally easy. It proves the distinction but not architectural value.

The next serious version should use a network that is actually free to organize its own receptive field:

1. train an RNN/attention model on nominal-rate data only;
2. compare event-position baseline, explicit-`dt` baseline, and an absolute-clock-constrained model;
3. estimate integration windows by perturbation/cross-context correlation rather than reading a known scalar leak;
4. compute the yoking index from those measured windows;
5. test whether yoking predicts OOD rate-shift accuracy;
6. include both time-yoked and structure-yoked tasks so a clock cannot win by construction;
7. keep model size and information matched.

A particularly important adversary is a normal continuous-time RNN/SSM with `dt` conditioning. If it learns the correct invariant as well as the clock-constrained model, then WidePresent again has no special architectural claim.

## 9. Status

**Positive proof-of-mechanism, not a positive architecture result.**

What survived the attack is the original morning question in a cleaner form:

> **What is the integration window yoked to?**

That is a better next axis for WidePresent than more timestamp bookkeeping.
