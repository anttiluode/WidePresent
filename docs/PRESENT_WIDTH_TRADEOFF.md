# Present width as an observability–staleness tradeoff

Date: 2026-08-11

This note records the main idea that emerged from revisiting the Rajapinta repositories
through the current WidePresent lake/boundary picture.

## Claim under test

A temporally extended present may have an engineering role that is more precise than
"more memory":

> **The useful width of the present is the temporal aperture that makes the current
> hidden state observable without admitting so much old history that obsolete state
> contaminates the estimate.**

This is not a claim about phenomenal consciousness or a universal human timescale.
It is a task/system-specific hypothesis.

## Stationary world

In `boundary_spectrum_observability.py`, three hidden graph interiors share the same
local four-edge stem. Only one boundary node is observed.

A single instantaneous state is near chance for identifying the hidden topology.
The spectrum of an increasingly wide boundary trace becomes progressively more
informative. In exploratory scratch runs across three seeds, 18-second windows
reached about 0.91 accuracy for three classes while instantaneous readout remained
about 0.32–0.37.

Interpretation: delayed returns and resonances require physical time to reveal remote
structure.

This is expected inverse-wave behavior, not a WidePresent discovery.

## Changing world — first attempt failed

The first nonstationary version continuously drove a wave system while the hidden
operator switched topology.

Result: classification of the new/current topology stayed near chance at essentially
all tested window widths.

This experiment was not tuned into success.

The failure exposed a missing variable: a boundary trace contained mixed waves
created under different hidden operators, but the representation did not identify
**which epoch generated which component of the signal**.

That is temporal provenance / bitemporality in physical form.

## Controlled probe after a switch

`boundary_width_staleness.py` makes the temporal source ambiguity more controlled:

```text
old topology
    |
old probe
    |
operator switches at t=18 s
    |
new probe at ~18.5 s
    |
read current topology at t=30 s
```

The classifier sees only a fixed-dimensional spectrum of the boundary history.

Exploratory scratch accuracy for current topology:

| width | seed 0 | seed 1 | seed 2 |
|---:|---:|---:|---:|
| 2 s | 0.766 | 0.778 | 0.766 |
| 4 s | 0.791 | 0.728 | 0.759 |
| 6 s | 0.804 | 0.842 | 0.842 |
| 8 s | 0.690 | 0.715 | 0.709 |
| 10 s | 0.620 | 0.652 | 0.665 |
| 12 s | 0.608 | 0.627 | 0.646 |
| 16 s | 0.456 | 0.424 | 0.449 |
| 20 s | 0.519 | 0.506 | 0.468 |

The broad pattern is the important part:

```text
too narrow-ish          less accumulated current response
intermediate (~4–6 s)   best current-state readout in this toy
too wide                obsolete old-topology history enters
```

Do not interpret the numeric 4–6 s optimum biologically. It is an arbitrary property
of this toy's graph size, coupling, probe timing and damping.

## Why this matters for WidePresent

The original WidePresent picture could have become a fixed-size sliding matrix.
This experiment suggests a more principled target.

Let `H` be temporal aperture. Define current-state utility schematically as

\[
U(H) = O(H) - S(H),
\]

where:

- `O(H)` is information gained about the current hidden state as delayed evidence
  arrives;
- `S(H)` is contamination / staleness from state that was valid earlier but is no
  longer current.

The useful present width is then

\[
H^* = \arg\max_H U(H).
\]

This is only schematic; the actual experiments should use explicit task loss,
calibration, information measures, or estimator error rather than inventing a
hand-designed utility.

## Connection to bitemporal state

The failed mixed-echo experiment points toward a stronger representation.

Every boundary component ideally needs something like:

```text
world-valid epoch        when the hidden state that generated it was valid
arrival / knowledge time when its consequence reached the boundary
```

For a delayed physical system those can differ substantially.

Thus the existing WidePresent distinction

```text
world time != knowledge/arrival time
```

is not merely database bookkeeping. A propagating medium realizes it naturally:
an event occurs remotely at one world time and only becomes observable at the
boundary later.

## Rajapinta connection, stripped down

The older repositories used `rajapinta` for thaw/freeze surfaces, horizons and
spectral sampling.

The part retained here is much weaker and more useful:

> **a boundary is a readout map from hidden dynamics into an observable temporal
> signal.**

The Connes/spectral language is not required. Ordinary graph Laplacians, system
identification and inverse boundary problems are enough.

If spectral geometry re-enters, it should enter as a diagnostic tool for the hidden
operator, not as a metaphysical premise.

## Next registered questions

1. **Same-dimensional encoders.** Compare FFT, raw resampling, LMU/HiPPO, SSM and a
   tiny causal Transformer with equal output state dimension.
2. **Known switch time vs unknown switch time.** Measure how temporal provenance
   changes the optimal aperture.
3. **Multiple asynchronous boundaries.** Test whether world-time / arrival-time
   typing improves fusion when modalities have different propagation delays.
4. **Prediction rendezvous.** Send forecast and observation along independent delay
   routes to a common now-boundary; compare local mismatch with explicit timestamped
   queues.
5. **Adaptive aperture.** Learn when to widen/narrow the readout while keeping the
   objective clock fixed and content-independent.

## Kill condition

If a conventional state estimator or timestamp-aware sequence model achieves the
same current-state accuracy/calibration across delay and switch distributions with
less state/compute, WidePresent does not need a special boundary architecture.

The surviving conceptual result would still be useful:

> **context length and present width are not the same quantity; the latter can be
> tied to physical observability and staleness.**
