# Rajapinta without the black hole — boundary observability for WidePresent

Date: 2026-08-11

This note imports one useful idea from the older Rajapinta repositories while
explicitly refusing to import their ontology.

The useful idea is:

> **A thin boundary can be where a distributed dynamical state becomes observable,
> even when the state itself lives in a larger hidden substrate.**

No black-hole, Clockfield, consciousness, or noncommutative-geometry claim is
required for this statement.

## 1. Why this reappeared

The current WidePresent branch has already separated:

```text
persistent substrate      K
fast moving field          x(t)
transient routing/gating   g(t)
slow modulation            m(t)
slow plasticity             theta(t)
```

The lake picture says that the substrate can remain fixed while waves move through
it. The question then becomes: **where is `now` read?**

The older Rajapinta work suggests treating `now` not only as a coordinate but as a
**readout / interaction surface**.

That is useful even if every physical interpretation in those repositories is
wrong.

## 2. The non-speculative mathematical core

Consider a linear wave system

\[
M \ddot{x}(t) + C \dot{x}(t) + K x(t) = B u(t),
\]

but suppose an agent does not observe the full state `x(t)`. It receives only a
boundary measurement

\[
y(t)=H x(t).
\]

An instantaneous value `y(t_now)` can be highly ambiguous. A temporal trace

\[
Y_{now,H} = \{y(t): t_{now}-H \le t \le t_{now}\}
\]

can contain delayed reflections, resonances, interference, and modal frequencies
that depend on the hidden operator `K`.

This is ordinary observability / inverse-problem territory. The point for
WidePresent is that **temporal width can be necessary because information about
hidden state is spread over arrival time and frequency rather than contained in an
instantaneous sample.**

## 3. Prior art that makes this a respectable control, not a novelty claim

There is strong mathematical prior art connecting boundary spectral measurements,
graph wave dynamics, and reconstruction of hidden graph structure.

- Blåsten, Isozaki, Lassas & Lu (2021), *Gel'fand's inverse problem for the graph
  Laplacian*: under stated graph conditions, boundary eigenvalue/eigenfunction data
  can determine the unknown weighted interior graph.
  https://arxiv.org/abs/2101.10026

- Takayama (2021), *Graph recovery from graph wave equation*: graph Laplacian modes
  are extracted from sampled graph-wave signals and used to reconstruct the graph.
  https://arxiv.org/abs/2111.12874

- Li, Gao, Geng & Yang (2024), *Vertex Weight Reconstruction in the Gel'fand's
  Inverse Problem on Connected Weighted Graphs*: reconstructs interior weights from
  Neumann boundary spectral data via a graph-wave boundary-control method.
  https://arxiv.org/abs/2407.17222

Therefore WidePresent must not claim:

```text
boundary signals reveal hidden geometry
spectra encode graph structure
wave echoes permit inverse reconstruction
```

Those are established ideas.

The new question, if any, is whether **an online learned agent benefits from
maintaining such boundary history as its working present**.

## 4. The link to Connes is diagnostic, not foundational

The older `ConnesClockfieldRajapinta` repository used the intuition that geometry can
be described spectrally and then attached a thaw/freeze boundary to a time-varying
operator.

For WidePresent we keep only a much weaker statement:

> the spectrum of a persistent operator is a useful description of the dynamics
> that a temporal boundary trace can reveal.

There is no need to call the graph Laplacian a Dirac operator. There is no need for
a dynamic spectral triple. There is no need for a black-hole horizon.

If a graph-wave implementation eventually uses spectral coordinates, they are
ordinary eigenmodes of a chosen operator unless a stronger mathematical reason is
earned.

## 5. A possible definition of `now`

Instead of representing now only as the center index of a buffer,

```text
past <------ NOW ------> future
```

consider a dynamical implementation:

```text
hidden persistent substrate
       waves / delayed state
              |
              v
       [ NOW BOUNDARY ]
              |
              v
       temporal trace seen
         by online policy
```

`NOW` is then the place where presently arriving state is sampled, compared,
interfered, or handed to the policy.

The **wide present** is not necessarily a spatial slab around the boundary. It can
be the recent temporal history of that thin surface.

This distinction is important:

```text
thin spatial surface
wide temporal observation
```

A one-node sensor can have a temporally high-dimensional present.

## 6. First toy: hidden topology behind one boundary node

`experiments/boundary_spectrum_observability.py` creates three 24-node spring
graphs:

```text
path
fork
loop
```

All three have exactly the same four-edge stem adjacent to boundary node 0. Only
remote topology differs.

For every sample the experiment randomizes:

- edge stiffnesses;
- damping;
- impulse amplitude;
- measurement time;
- small observation noise.

The only observed variable is displacement at node 0.

Two representations are compared:

```text
instantaneous:
    y(now), dy/dt(now)

wide:
    normalized spectrum of y(t) over a window ending at now
```

The classifier is deliberately boring logistic regression.

### Exploratory scratch result

Before committing the experiment, the same design was run for three seeds.
Chance is 1/3.

Approximate accuracy:

| boundary representation | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| instantaneous | 0.339 | 0.365 | 0.323 |
| spectrum, 2 s | 0.513 | 0.561 | 0.529 |
| spectrum, 4 s | 0.545 | 0.608 | 0.614 |
| spectrum, 6 s | 0.751 | 0.730 | 0.677 |
| spectrum, 10 s | 0.746 | 0.799 | 0.767 |
| spectrum, 14 s | 0.825 | 0.836 | 0.788 |
| spectrum, 18 s | 0.915 | 0.915 | 0.910 |

This is **not a WidePresent positive result**. It demonstrates an expected physical
fact in a deliberately favorable toy: delayed boundary response accumulates
information about remote topology.

Its value is conceptual and methodological:

> **the width of a temporal state can be interpreted as an observability horizon.**

## 7. Why this is more interesting than "memory length"

A conventional framing says:

> longer context contains more old samples.

The boundary-observability framing says:

> some current facts are not observable instantaneously at all; they become
> observable only after the system has had enough physical time to answer through
> delays, echoes, and resonances.

That makes temporal width tied to the dynamics of the substrate.

For a path length `d` and finite propagation speed `c`, remote structure cannot
influence the boundary before a causal return time on the rough order of

\[
T_{obs} \sim 2d/c.
\]

The exact observability time depends on the system, topology, boundary conditions,
and measurement setup. The important point is qualitative: **there can be a minimum
useful present width set by propagation geometry rather than by an arbitrary model
context size.**

## 8. WidePresent consequence: adaptive width may be earned

This suggests a new research question that does not require a fixed one-second or
five-second present.

Let

\[
H^*(t)
\]

be the shortest history horizon at which the current boundary trace becomes
sufficient for a registered task.

Then a model could be evaluated on:

```text
accuracy versus temporal width
calibration versus temporal width
mutual information versus temporal width
observability rank / conditioning versus temporal width
```

The "present" might widen when the relevant dynamics are slow or distant and narrow
when local fast signals suffice.

Important: that is **adaptive readout width**, not adaptive objective time. It does
not violate WidePresent's rule that elapsed time itself is content-independent.

## 9. Relation to the uploaded Baker & Cariani paper

The time-domain-brain paper is relevant as an idea source because it emphasizes:

- circulating temporal signals;
- delay networks;
- temporal correlation;
- wave interference;
- content-addressable regeneration;
- recent context carried by reverberating traces.

The boundary-observability experiment does not test their brain theory. It simply
uses a conventional graph-wave system where temporal signals genuinely contain
spectral information about a persistent substrate.

This is a safer bridge from the paper to AI than asserting holographic brain
storage.

## 10. Next experiments

### B1 — width curve under equal compute

The current toy gives the wide representation more dimensions as width increases.
Compress every window to the same dimensional budget and test whether useful
information still rises with physical duration.

### B2 — raw time versus spectrum versus learned encoder

Compare equal-size features from:

- raw samples;
- FFT magnitude/phase;
- wavelet/scattering features;
- LMU/HiPPO;
- tiny causal Transformer;
- learned SSM.

If all equivalent encoders show the same width curve, the phenomenon belongs to the
system, not the representation.

### B3 — changing hidden state

Let the interior topology/couplings change slowly. Ask how long a boundary window
should remain before old echoes become misleading. This creates an actual
**present-width tradeoff** between observability and staleness.

### B4 — prediction rendezvous at the boundary

Launch a forecast through a delay path so it reaches the boundary at its target
world time. Deliver the observation through an independent path. Test whether a
local boundary comparator provides useful prediction-error signals.

### B5 — bitemporal late evidence

Let an event occur in the hidden system at world time `t_w` but become visible at the
boundary only at arrival time `t_k`. This physically realizes the world-time versus
knowledge-time distinction already present in WidePresent.

## 11. Kill condition

This Rajapinta branch adds nothing architectural if a standard timestamped history
encoder or ordinary state estimator obtains the same robustness with less state and
compute.

Even then, the boundary picture may remain useful as an **experimental design
principle**:

> don't ask whether the model remembers enough tokens; ask how much temporal
> aperture the environment requires before the relevant state is observable.

That is the part worth keeping.
