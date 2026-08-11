# Present width versus temporal provenance

Date: 2026-08-11

The first changing-world boundary experiment suggested an appealing story:

> a present can be too narrow to observe the current world and too wide because old-world echoes contaminate it.

That story has an important confound. The original boundary summary used a single FFT-like spectrum over the whole aperture. A global spectrum preserves frequency but largely destroys **when** each component arrived.

If old and new evidence are mixed and temporal address is discarded, of course a long window can become ambiguous.

This note records the control.

## Experiment

`experiments/boundary_provenance_control.py`

A hidden spring graph changes topology at `t=9 s`. One standardized-but-randomized probe is launched under the old operator at `t=5 s`; another is launched under the new/current operator at `t=11 s`. At `t=18 s`, the task is to classify the current topology from the boundary history.

Every representation gets exactly **64 features**:

```text
global_psd    1 temporal bin x 64 frequency bins

tf_4x16       4 temporal bins x 16 frequency bins

tf_8x8        8 temporal bins x 8 frequency bins
```

So time-frequency representations do not get a larger feature budget. They only spend some spectral resolution on preserving coarse arrival-time provenance.

## Exploratory scratch result

With 180 examples and seeds 0--2:

### 6 s aperture

| representation | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| global PSD | 0.968 | 0.921 | 0.937 |
| 4x16 TF | 0.857 | 0.857 | 0.841 |
| 8x8 TF | 0.667 | 0.714 | 0.730 |

For a short aperture, spending dimensions on temporal bins hurts frequency resolution. The global spectrum wins.

### 14 s aperture

| representation | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| global PSD | 0.619 | 0.619 | 0.683 |
| 4x16 TF | 0.810 | 0.810 | 0.810 |
| 8x8 TF | 0.857 | 0.857 | 0.857 |

For a long aperture containing old and new echoes, preserving coarse temporal provenance recovers much of the lost current-state information.

## Revised interpretation

The simple claim

> "a too-wide present is inherently bad"

is therefore **weakened**.

A better statement is:

> **A wide history becomes dangerous when evidence from different world-times is combined in a representation that loses its temporal provenance.**

There is still a finite-resolution tradeoff. With only 64 numbers, allocating more temporal bins means fewer frequency bins. A representation must decide how much precision to spend on:

```text
WHAT spectral structure arrived
versus
WHEN it arrived
```

That is much closer to WidePresent's core question than a universal fixed present duration.

## Connection to bitemporality

The control echoes the ledger distinction:

```text
world/valid time       when the evidence was true/generated
knowledge/arrival time when the evidence became available here
```

A global FFT is a miniature example of what goes wrong when multiple world-times are collapsed into an undifferentiated current state.

The next architecture should therefore preserve temporal coordinates or source epochs *inside* the wide state rather than merely choosing a window size.

## Connection to the offline/sleep-inspired branch

An offline phase may help by reorganizing a mixed trace into temporally/provenance-typed components. But this control creates a kill condition for that story:

> if explicit provenance tagging solves the interference problem, a special sleep-like mode may be unnecessary.

Conversely, an offline mode becomes interesting if it can **infer or repair provenance when the tags are not directly known**.

## New research question

The sharper problem is now:

> **Given a fixed state budget, how should an online agent allocate representation between content resolution and temporal provenance so that old evidence remains available without masquerading as the present?**

This can be tested against ordinary time-frequency features, timestamped memory, change-point detection, state-space filters, and bitemporal event stores before invoking any wave-specific mechanism.
