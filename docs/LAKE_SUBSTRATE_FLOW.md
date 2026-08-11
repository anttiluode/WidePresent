# Lake substrate, moving flow — candidate WidePresent implementation

Date: 2026-08-11

This note records a new branch of the WidePresent search. It is **not** promoted into the main architecture and makes no novelty claim.

The motivating observations were:

1. a bicycle chain whose material contacts transport effects around a fixed mechanism;
2. a small amount of oil placed at one contact can spread through repeated local contacts;
3. lake waves propagate and interfere while the lake / elastic support remains where it is;
4. Baker & Cariani's 2025 time-domain-brain hypothesis separates circulating temporal signals from longer-lived delay-path / resonance structure.

The useful engineering abstraction is not "the brain is a lake." It is:

> **A persistent substrate can remain fixed while state, modulation, and information flow through it on different timescales.**

## 1. Lagrangian chain versus Eulerian lake

The bicycle-chain picture naturally tracks material elements as they move. That is Lagrangian in spirit: follow the links.

The lake picture is closer to an Eulerian field description: keep the spatial support fixed and watch displacement, velocity, phase, energy, or another field pass through it.

For a graph / mesh with incidence matrix `B`, edge stiffnesses `k`, and graph stiffness/Laplacian-like operator

\[
K = B^T\,\mathrm{diag}(k)\,B,
\]

a minimal damped driven wave system is

\[
M\ddot x(t) + C\dot x(t) + Kx(t)=Fu(t).
\]

On the **fast inference timescale**, `M`, `C`, and `K` may be fixed while `x(t)` and `dx/dt` move continuously.

This is the clean meaning of:

> **the matrix stays; the flow does not.**

Nothing about this is novel wave physics. The research question is whether this representation is useful for the temporal-working-state problem WidePresent is studying.

## 2. Do not call the teeth attention

A useful correction to the bicycle metaphor:

```text
which tooth CAN touch which link        topology / adjacency / local operator
what gets transmitted at that contact  local scattering / coupling rule
which available transfer is favored now attention / gating candidate
what repeated use changes slowly        plasticity / modulation candidate
```

So a tooth-to-link contact is not attention by itself.

Let `A_ij` encode the physically available local edges. A transient gate `g_ij(t)` can make an effective coupling

\[
K_{ij}^{eff}(t) = g_{ij}(t) K_{ij}.
\]

If `g_ij(t)` depends on current task, query, phase, goal, or another control signal, it is reasonable to compare it with **attention or dynamic routing**.

The important difference from ordinary Transformer attention is locality: the gate chooses among edges that the substrate already makes available rather than creating an unconstrained dense all-to-all interaction.

This distinction should be tested rather than sold as an advantage.

## 3. The oil slick is a second field, not necessarily memory

The bicycle-oil observation suggests another state variable living on the same topology.

Let `m(t)` be a slowly transported / diffusing modulator:

\[
\dot m(t) = -D L m(t) - \lambda m(t) + s(t),
\]

where `L` is the graph Laplacian, `D` is a transport/diffusion rate, and `s(t)` injects modulator locally.

A single local deposit can then spread through repeated local interactions.

The modulator might alter damping, gain, excitability, or coupling without changing the graph itself, for example

\[
C_i(m) = \frac{C_0}{1+\beta m_i}.
\]

That gives a useful separation:

```text
fast field          signal / wave / activation
slow field          modulatory "oil" / eligibility / local gain history
very slow substrate learned couplings / geometry / long-term memory
```

The oil field should **not** automatically be called memory. It becomes memory only if its retained state carries information useful for future behavior.

## 4. A third timescale: actual substrate plasticity

If learning modifies the medium itself, write a separate slow parameter field `theta`:

\[
\dot\theta = \epsilon\,\Phi(x, \hat x, e, \theta), \qquad \epsilon \ll 1.
\]

`Phi` could be Hebbian, anti-Hebbian, delta/error-correcting, homeostatic, or learned.

This matters because "the matrix stays" is only meant **relative to fast signal propagation**. Over learning time the effective medium may change.

A disciplined architecture therefore separates:

```text
tau_flow        << tau_modulation << tau_plasticity
```

rather than calling every dynamic quantity "weights."

## 5. Connection to Baker & Cariani (2025)

The uploaded paper is relevant as an idea source because its proposed system explicitly contains several analogous separations:

- circulating and propagating temporally patterned signals;
- reverberatory short-term traces in delay paths;
- quasi-permanent assemblies / synaptic changes that select delay paths;
- correlation and mismatch operations between circulating and incoming signals;
- amplification / attenuation via loop gain;
- gating of propagating waves as an analogy for attentional focus;
- holographic-like reconstruction through interference / correlation.

These are hypotheses about brain computation, not established engineering requirements for WidePresent.

The especially useful import is **signal-centricity**: the active computational object can be a trajectory / propagating pattern on a persistent operator rather than a static activation vector attached to a node.

## 6. WidePresent bridge: a stationary implementation of a moving present

Current WidePresent often describes a relative-time chart centered on `now`:

```text
past <---- NOW ----> future
```

A literal implementation can shift values across bins as time advances.

The lake view suggests a different implementation:

> keep the substrate fixed and let temporal phase / activation propagate through it.

Then `now` need not mean "physically shift the matrix." It can be a fixed readout/interference surface that moving temporal state reaches.

A future prediction could be launched into a delay/wave path chosen so that its state reaches the readout surface near its target time. An observation arriving there can interfere, correlate, or be compared with the matured prediction.

This gives an alternative formulation of prediction rendezvous:

```text
prediction launched
      |
      v
stationary delay / wave substrate
      |
      v
arrival at NOW readout  <-->  observation
```

This is conceptually close to delay-line and wave-memory prior art. It is only interesting if it improves a registered WidePresent task over simpler clocks, queues, ring buffers, LMU/HiPPO, or timestamp-aware models.

## 7. Where attention would actually live

A branching junction is the clean test.

Suppose a wave reaches node `j` with two legal outgoing branches.

Without gating, the static coupling operator determines the split.

With a control field `q(t)`, define local gains

\[
g_{j\to a}(t),\;g_{j\to b}(t).
\]

The same substrate remains present, but the current signal is preferentially routed through one branch.

That is much closer to **attention** than the substrate itself.

A useful phrase is:

> **attention is not the road map; attention is the temporary traffic control on the road map.**

In a local physical fabric, this becomes **topology-constrained attention** or **local dynamic routing**.

## 8. Small executable sanity check

`experiments/lake_flow_gate_demo.py` implements a symmetric Y-shaped spring graph.

Three runs are compared:

1. no gate — energy should split symmetrically;
2. left gate — the same fixed graph preferentially sends energy left;
3. right gate — mirror of the second case.

A separate slow graph-diffusion field starts from one node as a toy "oil" deposit and spreads over the fixed topology.

The demo does **not** establish useful attention, memory, learning, or brain relevance. It exists to make the metaphors mechanically unambiguous.

## 9. Research questions exposed by this picture

### Q1 — Is a flowing state better than a shifted buffer?

Compare a stationary wave/delay substrate with the existing WidePresent matrix, using exactly matched timing information and state capacity.

### Q2 — Does locality help or hurt dynamic routing?

Compare topology-constrained gates against dense attention and ordinary graph message passing.

### Q3 — Can a slow modulator provide useful temporal context?

Test whether a diffusing/advecting gain field helps delayed-observation, interruption, or asynchronous-modality tasks beyond explicit scalar ages.

### Q4 — Does prediction rendezvous emerge naturally?

Launch forecasts into delay paths and score whether they meet observations near their due times more robustly than explicit priority queues / timestamp features.

### Q5 — When should the substrate itself learn?

Only add plasticity after the fixed-substrate system exposes a specific failure. Compare additive Hebbian writes, delta/error-correcting writes, decay, and ordinary trained parameters.

## 10. Kill conditions

This branch should die if:

- a ring buffer or deterministic temporal kernel matches it;
- dynamic routing gains disappear when compared to a parameter-matched local attention baseline;
- the oil field adds no information beyond a scalar elapsed-time feature;
- wave propagation merely adds numerical complexity without robustness or useful inductive bias;
- substrate plasticity recreates generic fast weights with no benefit from locality.

The point of the lake is therefore not to justify waves.

It is to separate **what stays**, **what flows**, **what gates**, and **what learns** clearly enough that each can be tested independently.
